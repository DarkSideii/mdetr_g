from bisect import bisect_right
import math
from typing import Optional

import torch


def update_ema(model, model_ema, decay: float) -> None:
    """
    Update model_ema parameters/buffers via exponential moving average:

        ema = ema * decay + (1 - decay) * model

    - Unwraps DDP models (model.module).
    - Non-floating tensors are copied directly.
    - Missing keys in either state_dict are skipped.
    """
    if model_ema is None:
        return
    if not (0.0 <= float(decay) < 1.0):
        raise ValueError(f"EMA decay must be in [0,1). Got {decay}.")

    with torch.no_grad():
        if hasattr(model, "module"):
            model = model.module

        msd = model.state_dict()
        emasd = model_ema.state_dict()

        for k, ema_v in emasd.items():
            if k not in msd:
                continue

            model_v = msd[k].detach()

            if model_v.device != ema_v.device or model_v.dtype != ema_v.dtype:
                model_v = model_v.to(device=ema_v.device, dtype=ema_v.dtype)

            if torch.is_floating_point(ema_v):
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)
            else:
                ema_v.copy_(model_v)


def adjust_learning_rate(
    optimizer,
    epoch: int,
    curr_step: int,
    num_training_steps: int,
    args,
) -> None:
    """
    Update optimizer group LRs based on `args.schedule`.

    Schedules:
      - step:                0.1 ** (epoch // lr_drop)
      - multistep:           0.5 ** (#milestones passed), milestones every 50 epochs after lr_drop
      - linear_with_warmup:  base/backbone use step; text uses warmup + linear decay
      - all_linear_with_warmup: warmup + linear decay for all groups
      - all_cosine_with_warmup: warmup + cosine decay for all groups

    Group routing is by param_group["name"]:
      - base/backbone -> base multiplier
      - text/text_encoder -> text multiplier
      - logit_scales -> configurable via args.logit_scales_schedule (default: base)

    Optional per-group keys:
      - base_lr:     schedule baseline (recommended)
      - lr_scale:    extra multiplicative factor (default 1.0)
      - schedule:    override {"base","text","none"} for this group
      - freeze_lr:   force lr=0 for this group
    """
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, got {num_training_steps}")

    num_warmup_steps = round(float(getattr(args, "fraction_warmup_steps", 0.0)) * float(num_training_steps))
    num_warmup_steps = int(max(0, num_warmup_steps))

    schedule = str(getattr(args, "schedule", "step")).lower()
    lr_drop = int(getattr(args, "lr_drop", 1))
    epochs = int(getattr(args, "epochs", 1))

    if schedule == "step":
        gamma = 0.1 ** (epoch // max(1, lr_drop))
        text_encoder_gamma = gamma

    elif schedule == "multistep":
        milestones = list(range(lr_drop, epochs, 50))
        gamma = 0.5 ** bisect_right(milestones, epoch)
        text_encoder_gamma = gamma

    elif schedule == "linear_with_warmup":
        gamma = 0.1 ** (epoch // max(1, lr_drop))

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
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(curr_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            gamma = 0.5 * (1.0 + math.cos(math.pi * progress))
        text_encoder_gamma = gamma

    else:
        raise NotImplementedError(f"Unknown schedule: {schedule!r}")

    def _group_mult(group_name: str, group_schedule_override: Optional[str] = None) -> float:
        name = (group_name or "base").lower()
        override = (group_schedule_override or "").lower().strip() if group_schedule_override is not None else ""

        if override in ("none", "constant", "fixed"):
            return 1.0
        if override in ("text", "text_encoder"):
            return text_encoder_gamma
        if override in ("base", "backbone"):
            return gamma

        if name in ("base", "backbone"):
            return gamma
        if name in ("text", "text_encoder"):
            return text_encoder_gamma

        if name in ("logit_scales", "logit_scale", "logitscales"):
            mode = str(getattr(args, "logit_scales_schedule", "base")).lower()
            if mode in ("none", "constant", "fixed"):
                return 1.0
            if mode in ("text", "text_encoder"):
                return text_encoder_gamma
            return gamma

        return gamma

    for g in optimizer.param_groups:
        name = g.get("name", "base")

        base_lr = float(g.get("base_lr", g.get("lr", 0.0)))
        lr_scale = float(g.get("lr_scale", 1.0))

        if bool(g.get("freeze_lr", False)):
            g["lr"] = 0.0
            continue

        mult = _group_mult(name, g.get("schedule", None))
        g["lr"] = base_lr * mult * lr_scale
