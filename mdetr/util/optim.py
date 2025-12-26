# multimodal_framework/mdetr/util/optim.py
from bisect import bisect_right
import math
from typing import Optional

import torch


def update_ema(model, model_ema, decay: float) -> None:
    """
    Exponential Moving Average (EMA) update:

        w_ema = w_ema * decay + (1 - decay) * w

    Notes:
      - Works with DDP-wrapped models (model.module)
      - For non-floating tensors/buffers, we copy directly (no averaging)
      - Safely handles minor key mismatches (skips missing keys)
    """
    if model_ema is None:
        return
    if not (0.0 <= float(decay) < 1.0):
        raise ValueError(f"EMA decay must be in [0,1). Got {decay}.")

    with torch.no_grad():
        # Unwrap DDP
        if hasattr(model, "module"):
            model = model.module

        msd = model.state_dict()
        emasd = model_ema.state_dict()

        for k, ema_v in emasd.items():
            if k not in msd:
                continue

            model_v = msd[k].detach()

            # Move/cast model tensor to EMA tensor's device/dtype
            if model_v.device != ema_v.device or model_v.dtype != ema_v.dtype:
                model_v = model_v.to(device=ema_v.device, dtype=ema_v.dtype)

            if torch.is_floating_point(ema_v):
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)
            else:
                # ints/bools/etc: copy exact value
                ema_v.copy_(model_v)


def adjust_learning_rate(
    optimizer,
    epoch: int,
    curr_step: int,
    num_training_steps: int,
    args,
) -> None:
    """
    Update per-group learning rates according to a schedule.

    Supports schedules:
      - "step": step decay every `args.lr_drop` epochs (gamma=0.1 ** (epoch // lr_drop))
      - "multistep": 0.5 ** (#milestones passed), milestones every 50 epochs after lr_drop
      - "linear_with_warmup": base/backbone uses step; text uses linear warmup+decay
      - "all_linear_with_warmup": all groups use linear warmup+decay
      - "all_cosine_with_warmup": cosine after warmup for all groups

    Param group selection by group "name":
      - "base", "backbone"        -> base schedule multiplier
      - "text", "text_encoder"    -> text schedule multiplier
      - "logit_scales"            -> configurable (defaults to base schedule)

    Extra (optional) group-level keys supported:
      - "base_lr": initial LR to schedule from (recommended)
      - "lr_scale": extra multiplicative factor (default 1.0)
      - "schedule": override one of {"base","text","none"} for this group
      - "freeze_lr": if True, forces lr=0 for that group

    Optional args:
      - args.logit_scales_schedule in {"base","text","none","constant","fixed"}
        If not present, logit_scales follow "base".
    """
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, got {num_training_steps}")

    # Warmup steps (used by the warmup schedules)
    num_warmup_steps = round(float(getattr(args, "fraction_warmup_steps", 0.0)) * float(num_training_steps))
    num_warmup_steps = int(max(0, num_warmup_steps))

    schedule = str(getattr(args, "schedule", "step")).lower()
    lr_drop = int(getattr(args, "lr_drop", 1))
    epochs = int(getattr(args, "epochs", 1))

    # --- compute gamma (and text_encoder_gamma) ---
    if schedule == "step":
        gamma = 0.1 ** (epoch // max(1, lr_drop))
        text_encoder_gamma = gamma

    elif schedule == "multistep":
        milestones = list(range(lr_drop, epochs, 50))
        gamma = 0.5 ** bisect_right(milestones, epoch)
        text_encoder_gamma = gamma

    elif schedule == "linear_with_warmup":
        # base/backbone uses step
        gamma = 0.1 ** (epoch // max(1, lr_drop))

        # text encoder uses linear warmup then linear decay
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

    elif schedule == "all_linear_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        text_encoder_gamma = gamma

    elif schedule == "all_cosine_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))  # warmup
        else:
            progress = float(curr_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            gamma = 0.5 * (1.0 + math.cos(math.pi * progress))
        text_encoder_gamma = gamma

    else:
        raise NotImplementedError(f"Unknown schedule: {schedule!r}")

    # Helper: choose multiplier for a param group
    def _group_mult(group_name: str, group_schedule_override: Optional[str] = None) -> float:
        name = (group_name or "base").lower()
        override = (group_schedule_override or "").lower().strip() if group_schedule_override is not None else ""

        # group-level override wins
        if override in ("none", "constant", "fixed"):
            return 1.0
        if override in ("text", "text_encoder"):
            return text_encoder_gamma
        if override in ("base", "backbone"):
            return gamma

        # default behavior by name
        if name in ("base", "backbone"):
            return gamma
        if name in ("text", "text_encoder"):
            return text_encoder_gamma

        # logit scales: configurable via args (default base)
        if name in ("logit_scales", "logit_scale", "logitscales"):
            mode = str(getattr(args, "logit_scales_schedule", "base")).lower()
            if mode in ("none", "constant", "fixed"):
                return 1.0
            if mode in ("text", "text_encoder"):
                return text_encoder_gamma
            return gamma

        # all other groups follow base schedule by default
        return gamma

    # --- apply schedule by group ---
    for g in optimizer.param_groups:
        name = g.get("name", "base")

        # If your training script set base_lr, schedule from it; else use current LR as baseline
        base_lr = float(g.get("base_lr", g.get("lr", 0.0)))
        lr_scale = float(g.get("lr_scale", 1.0))

        if bool(g.get("freeze_lr", False)):
            g["lr"] = 0.0
            continue

        mult = _group_mult(name, g.get("schedule", None))
        g["lr"] = base_lr * mult * lr_scale
