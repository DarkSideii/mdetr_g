from bisect import bisect_right
import math
import torch


def update_ema(model, model_ema, decay: float) -> None:
    """
    Apply exponential moving average (EMA) update.

    The weights are updated in-place as:
        w_ema = w_ema * decay + (1 - decay) * w

    Args:
        model: Active model that is being optimized.
        model_ema: Running-average model (same architecture as `model`).
        decay: Exponential decay parameter in [0, 1).
    """
    with torch.no_grad():
        # Unwrap DDP
        if hasattr(model, "module"):
            model = model.module

        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


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

    Groups are selected by the param group "name":
        - "base", "backbone"        -> use `gamma`
        - "text", "text_encoder"    -> use `text_encoder_gamma`
        - others (e.g., "logit_scales") follow `gamma` by default
    """
    num_warmup_steps = round(args.fraction_warmup_steps * num_training_steps)

    # --- compute gamma (and text_encoder_gamma) ---
    if args.schedule == "step":
        gamma = 0.1 ** (epoch // args.lr_drop)
        text_encoder_gamma = gamma

    elif args.schedule == "multistep":
        milestones = list(range(args.lr_drop, args.epochs, 50))
        gamma = 0.5 ** bisect_right(milestones, epoch)
        text_encoder_gamma = gamma

    elif args.schedule == "linear_with_warmup":
        # base/backbone uses step
        gamma = 0.1 ** (epoch // args.lr_drop)
        # text encoder uses linear warmup then linear decay
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

    elif args.schedule == "all_linear_with_warmup":
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        gamma = text_encoder_gamma

    elif args.schedule == "all_cosine_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))  # warmup
        else:
            progress = float(curr_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            gamma = 0.5 * (1.0 + math.cos(math.pi * progress))
        text_encoder_gamma = gamma

    else:
        raise NotImplementedError

    # --- apply schedule by group name ---
    for g in optimizer.param_groups:
        name = g.get("name", "base")
        base_lr = g.get("base_lr", g["lr"])  # fall back to current lr if not provided

        if name in ("base", "backbone"):
            mult = gamma
        elif name in ("text", "text_encoder"):
            mult = text_encoder_gamma
        else:
            # extras (e.g., logit scales) follow the base schedule by default
            mult = gamma

        g["lr"] = base_lr * mult
