"""
Training/evaluation loops with gradient accumulation and optional EMA.

Logs a small set of primary loss components plus total loss, and can report
caption↔box alignment misses when a positive map is available.
"""

import math
import time
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

import mdetr.util.dist as dist
from mdetr.util.metrics import MetricLogger, SmoothedValue
from mdetr.util.misc import targets_to
from mdetr.util.optim import adjust_learning_rate, update_ema


# Loss components to log (total "loss" is logged separately).
LOG_KEYS = {
    "loss",                    # total weighted loss (includes aux if present)
    "loss_ce",                 # main CE
    "loss_bbox",               # main L1
    "loss_giou",               # main GIoU
    "loss_contrastive_align",  # object–token alignment
    "contrastive_loss",        # global image–text InfoNCE
}


def _update_lr_meters(logger: MetricLogger, optimizer: torch.optim.Optimizer) -> None:
    """Update LR meters for named optimizer groups (base/backbone/text/logit_scales)."""
    if "lr" not in logger.meters:
        logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # Param groups keyed by explicit 'name' (as set in main.py).
    group_map = {g.get("name", f"group{i}"): g for i, g in enumerate(optimizer.param_groups)}

    def has_trainable_params(g):
        return any(getattr(p, "requires_grad", False) for p in g.get("params", []))

    # Base LR (fallback to first group if "base" isn't named).
    base = group_map.get("base", None)
    if base is None and len(optimizer.param_groups) > 0:
        logger.update(lr=float(optimizer.param_groups[0]["lr"]))
    elif base is not None:
        logger.update(lr=float(base["lr"]))

    # Backbone LR.
    bb = group_map.get("backbone")
    if bb and float(bb["lr"]) > 0.0 and has_trainable_params(bb):
        if "lr_backbone" not in logger.meters:
            logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        logger.update(lr_backbone=float(bb["lr"]))

    # Text encoder LR.
    txt = group_map.get("text")
    if txt and float(txt["lr"]) > 0.0 and has_trainable_params(txt):
        if "lr_text" not in logger.meters:
            logger.add_meter("lr_text", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        logger.update(lr_text=float(txt["lr"]))

    # Optional: logit scales (added via model.extra_optim_groups with name="logit_scales").
    lg = group_map.get("logit_scales")
    if lg and float(lg["lr"]) > 0.0 and has_trainable_params(lg):
        if "lr_logit" not in logger.meters:
            logger.add_meter("lr_logit", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        logger.update(lr_logit=float(lg["lr"]))


def _move_targets_to_device(raw_targets, device):
    """Move tensor-valued target fields to `device` (leaves non-tensors unchanged)."""
    targets = []
    for t in raw_targets:
        td = {}
        for k, v in t.items():
            td[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        targets.append(td)
    return targets


def _report_alignment_issues(
    phase: str,
    step_tag: str,
    raw_targets,
    positive_map: Optional[torch.Tensor],
) -> None:
    """
    Print only items that have ≥1 GT box with no token alignment.

    A box is considered unaligned if its corresponding row in `positive_map` sums to
    zero. If `positive_map` isn't provided, falls back to `tokens_positive`.
    """
    if raw_targets is None or len(raw_targets) == 0:
        return

    cur = 0
    any_printed = False

    for i, t in enumerate(raw_targets):
        k = int(t.get("boxes", torch.zeros(0)).shape[0])
        if k == 0:
            continue

        failed_rows = []
        if isinstance(positive_map, torch.Tensor):
            pm_slice = positive_map[cur : cur + k] if k > 0 else None
            cur += k
            if pm_slice is not None and pm_slice.numel() > 0:
                zero_mask = pm_slice.sum(dim=1) == 0
                failed_rows = torch.nonzero(zero_mask, as_tuple=False).view(-1).cpu().tolist()
        else:
            cur += k
            toks = t.get("tokens_positive", [])
            if isinstance(toks, list):
                for row_idx, spans in enumerate(toks):
                    bad = (not spans) or (spans and spans[0] is None)
                    if bad:
                        failed_rows.append(row_idx)

        if not failed_rows:
            continue

        if not any_printed:
            print(f"[ALIGN-ISSUE][{phase}] step={step_tag}")
            any_printed = True

        name = t.get("filename_stem", f"item_{i}")
        cap = t.get("caption", "")
        print(f" - {name}: '{cap}'")

        phrases = t.get("phrases_searched", None)
        if phrases is None:
            phrases = ["<none>"] * k

        for j in failed_rows:
            ph = phrases[j] if j < len(phrases) else "<none>"
            print(f"   box {j}: phrase={repr(ph)} -> NOT FOUND")


def train_one_epoch(
    model: nn.Module,
    criterion: Optional[nn.Module],
    contrastive_criterion: Optional[nn.Module],
    data_loader: Iterable,
    weight_dict: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0.0,
    model_ema: Optional[nn.Module] = None,
):
    """Train for one epoch with gradient accumulation; step LR schedule per optimizer update."""
    model.train()
    if isinstance(criterion, nn.Module):
        criterion.train()
    if isinstance(contrastive_criterion, nn.Module):
        contrastive_criterion.train()

    logger = MetricLogger(delimiter=" ")
    logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header, print_freq = f"Epoch [{epoch}]", 50

    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))

    # Number of optimizer updates this epoch.
    num_updates_per_epoch = (len(data_loader) + accum_steps - 1) // accum_steps
    num_steps_total = num_updates_per_epoch * args.epochs

    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(data_loader)

    # Iterate by optimizer updates (not micro-batches).
    for update_idx in logger.log_every(range(num_updates_per_epoch), print_freq, header):
        update_start = time.perf_counter()
        micro_batches_done = 0

        # Sum raw losses across micro-batches for this update.
        loss_sums: Dict[str, torch.Tensor] = {}

        # Process up to accum_steps micro-batches.
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            micro_batches_done += 1

            samples = batch["samples"].to(device)
            raw_targets = batch["targets"]
            captions = batch.get("captions", None)
            if captions is None:
                captions = [t["caption"] for t in raw_targets]

            targets = _move_targets_to_device(raw_targets, device)

            positive_map = batch.get("positive_map")
            if isinstance(positive_map, (list, tuple)):
                positive_map = [pm.to(device) for pm in positive_map]
            elif positive_map is not None:
                positive_map = positive_map.to(device)

            if isinstance(positive_map, torch.Tensor):
                _report_alignment_issues("TRAIN", f"e{epoch}/u{update_idx}", raw_targets, positive_map)
            else:
                _report_alignment_issues("TRAIN", f"e{epoch}/u{update_idx}", raw_targets, None)

            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

            loss_dict: Dict[str, torch.Tensor] = {}

            # Detection losses.
            if (criterion is not None) and (not getattr(args, "no_detection", False)):
                det_losses = criterion(outputs, targets, positive_map)
                loss_dict.update(det_losses)

            # Contrastive loss (global image–text).
            if (contrastive_criterion is not None) and getattr(args, "contrastive_loss", False):
                if memory_cache is not None:
                    t_pool = memory_cache.get("text_pooled_op")
                    i_pool = memory_cache.get("img_pooled_op")
                    if (t_pool is not None) and (i_pool is not None):
                        loss_dict["contrastive_loss"] = contrastive_criterion(t_pool, i_pool)

            # Total weighted loss used for backprop.
            loss_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

            if not math.isfinite(float(loss_total)):
                reduced = {k: (v.item() if hasattr(v, "item") else v) for k, v in loss_dict.items()}
                print("Non-finite loss, aborting.", reduced)
                raise RuntimeError("Non-finite loss encountered.")

            (loss_total / accum_steps).backward()

            # Accumulate detached losses for logging.
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.detach()

        if micro_batches_done == 0:
            break

        # If the final update used fewer micro-batches, rescale grads to match accum_steps.
        if 0 < micro_batches_done < accum_steps:
            corr = accum_steps / float(micro_batches_done)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(corr)

        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Update LR schedule before optimizer.step().
        step_global = epoch * num_updates_per_epoch + update_idx
        adjust_learning_rate(
            optimizer=optimizer,
            epoch=epoch,
            curr_step=step_global,
            num_training_steps=num_steps_total,
            args=args,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if model_ema is not None:
            step = epoch * num_updates_per_epoch + update_idx + 1
            eff_decay = min(args.ema_decay, 1.0 - 1.0 / step)
            update_ema(model, model_ema, eff_decay)

        _update_lr_meters(logger, optimizer)

        loss_avg = {k: v / micro_batches_done for k, v in loss_sums.items()}
        loss_red = dist.reduce_dict(loss_avg)
        loss_scaled = {k: loss_red[k] * weight_dict[k] for k in loss_red if k in weight_dict}
        total_weighted = sum(loss_scaled.values())

        compact = {k: loss_scaled[k] for k in LOG_KEYS if k in loss_scaled}
        logger.update(**compact)
        logger.update(loss=total_weighted.item())

        iter_time = time.perf_counter() - update_start
        logger.update(time=iter_time)

    logger.synchronize_between_processes()
    print("Averaged stats:", logger)
    return {k: m.global_avg for k, m in logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: Optional[nn.Module],
    contrastive_criterion: Optional[nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    postprocessors: Dict[str, nn.Module],
    evaluator_list,
    device: torch.device,
    args,
):
    """Run evaluation and update evaluators (e.g., COCO AP); returns reduced scalar stats."""
    model.eval()
    for crit in (criterion, contrastive_criterion):
        if isinstance(crit, nn.Module):
            crit.eval()

    logger = MetricLogger(delimiter=" ")
    header, print_freq = "Test:", 50
    eval_step = 0

    for batch_dict in logger.log_every(data_loader, print_freq, header):
        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        captions = [t["caption"] for t in targets]

        positive_map = batch_dict.get("positive_map")
        if isinstance(positive_map, (list, tuple)):
            positive_map = [pm.to(device) for pm in positive_map]
        elif positive_map is not None:
            positive_map = positive_map.to(device)

        # Debug: alignment issue logging (disabled by default).
        # if isinstance(positive_map, torch.Tensor):
        #     _report_alignment_issues("EVAL", f"step{eval_step}", targets, positive_map)
        # else:
        #     _report_alignment_issues("EVAL", f"step{eval_step}", targets, None)
        eval_step += 1

        targets = targets_to(targets, device)

        memory_cache = model(samples, captions, encode_and_save=True)
        outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        loss_dict: Dict[str, torch.Tensor] = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if (contrastive_criterion is not None) and (memory_cache is not None):
            loss_dict["contrastive_loss"] = contrastive_criterion(
                memory_cache["text_pooled_op"],
                memory_cache["img_pooled_op"],
            )

        loss_red = dist.reduce_dict(loss_dict)
        loss_scl = {k: loss_red[k] * weight_dict[k] for k in loss_red if k in weight_dict}
        compact = {k: loss_scl[k] for k in LOG_KEYS if k in loss_scl}
        logger.update(**compact)
        if loss_scl:
            logger.update(loss=sum(loss_scl.values()).item())

        # Detection eval (COCO-style AP).
        if not getattr(args, "no_detection", False):
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors["bbox"](outputs, orig_target_sizes)

            for r in results:
                if "labels" in r:
                    r["labels"] = torch.ones_like(r["labels"])

            res = {t["image_id"].item(): output for t, output in zip(targets, results)}
            for evaluator in evaluator_list:
                evaluator.update(res)

    logger.synchronize_between_processes()
    print("Averaged stats:", logger)

    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

    stats = {k: m.global_avg for k, m in logger.meters.items()}

    for evaluator in evaluator_list:
        coco_eval = getattr(evaluator, "coco_eval", None)
        if isinstance(coco_eval, dict) and coco_eval.get("bbox") is not None:
            ce = coco_eval["bbox"].stats
            if hasattr(ce, "tolist"):
                ce = ce.tolist()
            stats["coco_eval_bbox"] = ce
            stats["ap"] = ce[0]
            stats["ap50"] = ce[1]
            stats["ap75"] = ce[2]

    return stats
