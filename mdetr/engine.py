
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.

"""
Training/evaluation loops with gradient accumulation and optional EMA.

Logs a small set of primary loss components plus total loss, and can report
caption↔box alignment misses when a positive map is available.

Optional eval-only additions (gated by args.eval_phrase_box_metrics):
- phrase_box_recall@1 / @5 / @10 at IoU >= 0.50
- phrase_box_ece@0.50 from the top-1 phrase-conditioned box score
"""

import math
import time
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import mdetr.util.dist as dist
from mdetr.util import box_ops
from mdetr.util.metrics import MetricLogger, SmoothedValue
from mdetr.util.misc import targets_to
from mdetr.util.optim import adjust_learning_rate, update_ema

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PHRASE_BOX_IOU_THRESHOLD = 0.50


# Loss components to log (total "loss" is logged separately).
LOG_KEYS = {
    "loss",                    # total weighted loss (includes aux if present)
    "loss_ce",                 # main CE
    "loss_bbox",               # main L1
    "loss_giou",               # main GIoU
    "loss_contrastive_align",  # object–token alignment
    "contrastive_loss",        # global image–text InfoNCE
}


def _is_main_process() -> bool:
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _update_lr_meters(logger: MetricLogger, optimizer: torch.optim.Optimizer) -> None:
    """Update LR meters for named optimizer groups (base/backbone/text/logit_scales)."""
    if "lr" not in logger.meters:
        logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    group_map = {g.get("name", f"group{i}"): g for i, g in enumerate(optimizer.param_groups)}

    def has_trainable_params(g):
        return any(getattr(p, "requires_grad", False) for p in g.get("params", []))

    base = group_map.get("base", None)
    if base is None and len(optimizer.param_groups) > 0:
        logger.update(lr=float(optimizer.param_groups[0]["lr"]))
    elif base is not None:
        logger.update(lr=float(base["lr"]))

    bb = group_map.get("backbone")
    if bb and float(bb["lr"]) > 0.0 and has_trainable_params(bb):
        if "lr_backbone" not in logger.meters:
            logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        logger.update(lr_backbone=float(bb["lr"]))

    txt = group_map.get("text")
    if txt and float(txt["lr"]) > 0.0 and has_trainable_params(txt):
        if "lr_text" not in logger.meters:
            logger.add_meter("lr_text", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        logger.update(lr_text=float(txt["lr"]))

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


def _num_target_boxes(target) -> int:
    boxes = target.get("boxes", [])
    if hasattr(boxes, "shape"):
        return int(boxes.shape[0])
    return len(boxes)


def _resize_bool_mask(mask_like, target_len: int, device: torch.device, drop_last_if_one_extra: bool = False):
    if mask_like is None:
        return None

    if torch.is_tensor(mask_like):
        mask = mask_like.to(device)
    else:
        mask = torch.as_tensor(mask_like, device=device)

    mask = mask.flatten()

    if drop_last_if_one_extra and mask.numel() == target_len + 1:
        mask = mask[:-1]

    if mask.numel() > target_len:
        mask = mask[:target_len]
    elif mask.numel() < target_len:
        pad = torch.zeros(target_len - mask.numel(), dtype=mask.dtype, device=device)
        mask = torch.cat([mask, pad], dim=0)

    mask = mask > 0
    return mask if mask.any() else None


def _resize_token_mask(token_mask, target_len: int, device: torch.device):
    if token_mask is None:
        return None

    token_mask = token_mask.to(device).bool()
    if token_mask.shape[1] > target_len:
        token_mask = token_mask[:, :target_len]
    elif token_mask.shape[1] < target_len:
        pad = torch.zeros(
            token_mask.shape[0],
            target_len - token_mask.shape[1],
            dtype=torch.bool,
            device=device,
        )
        token_mask = torch.cat([token_mask, pad], dim=1)

    return token_mask


def _compute_ece(confidences, correctness, n_bins: int = 15) -> float:
    if len(confidences) == 0:
        return 0.0

    ece = 0.0
    total = float(len(confidences))

    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins

        if bin_idx == 0:
            idxs = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            idxs = [i for i, c in enumerate(confidences) if lo < c <= hi]

        if not idxs:
            continue

        bin_conf = sum(float(confidences[i]) for i in idxs) / len(idxs)
        bin_acc = sum(float(correctness[i]) for i in idxs) / len(idxs)
        ece += (len(idxs) / total) * abs(bin_acc - bin_conf)

    return float(ece)


def _gather_list_across_processes(values):
    gathered = dist.all_gather(list(values))
    merged = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def _extract_valid_token_mask(outputs, device: torch.device, num_token_classes: Optional[int] = None):
    tokenized = outputs.get("tokenized", None)

    if tokenized is not None:
        attn = None
        try:
            attn = tokenized["attention_mask"]
        except Exception:
            attn = getattr(tokenized, "attention_mask", None)

        if attn is not None:
            valid = attn.to(device).bool()

            special = None
            try:
                special = tokenized["special_tokens_mask"]
            except Exception:
                special = getattr(tokenized, "special_tokens_mask", None)

            if special is not None:
                special = special.to(device).bool()
                min_w = min(int(special.shape[-1]), int(valid.shape[-1]))
                special = special[:, :min_w]
                valid = valid[:, :min_w]
                valid = valid & (~special)

            if num_token_classes is not None:
                valid = _resize_token_mask(valid, num_token_classes, device)
            return valid

    if num_token_classes is None:
        pred_logits = outputs.get("pred_logits", None)
        if pred_logits is not None and pred_logits.shape[-1] > 0:
            num_token_classes = int(pred_logits.shape[-1] - 1)

    if num_token_classes is None:
        return None

    bsz = int(outputs["pred_logits"].shape[0])
    return torch.ones((bsz, num_token_classes), dtype=torch.bool, device=device)


def _char_to_token_with_fallback(tokenized, batch_index: int, char_pos: int, is_end: bool = False):
    if tokenized is None or not hasattr(tokenized, "char_to_token"):
        return None

    candidates = [char_pos]
    if is_end:
        candidates.extend([char_pos - 1, char_pos - 2, char_pos - 3])
    else:
        candidates.extend([char_pos + 1, char_pos + 2, char_pos + 3])

    for pos in candidates:
        if pos is None or pos < 0:
            continue

        try:
            tok = tokenized.char_to_token(batch_index, pos)
        except TypeError:
            try:
                tok = tokenized.char_to_token(pos)
            except Exception:
                tok = None
        except Exception:
            tok = None

        if tok is not None:
            return int(tok)

    return None


def _get_target_char_spans(targets, batch_index: int, tgt_index: int):
    target = targets[batch_index]
    if not isinstance(target, dict):
        return None

    spans = None
    if target.get("tokens_positive", None) is not None:
        spans = target["tokens_positive"][int(tgt_index)]
    elif target.get("tokens", None) is not None:
        spans = target["tokens"][int(tgt_index)]

    if spans is None or spans == []:
        return None
    if isinstance(spans, (list, tuple)) and len(spans) > 0 and spans[0] is None:
        return None
    return spans


def _get_phrase_positive_mask(
    batch_positive_map,
    targets,
    batch_index: int,
    tgt_index: int,
    num_token_classes: int,
    device: torch.device,
    tokenized=None,
):
    if batch_positive_map is not None:
        if isinstance(batch_positive_map, (list, tuple)):
            if batch_index < len(batch_positive_map):
                per_item_pm = batch_positive_map[batch_index]
                if per_item_pm is not None and int(tgt_index) < len(per_item_pm):
                    mask = _resize_bool_mask(
                        per_item_pm[int(tgt_index)],
                        target_len=num_token_classes,
                        device=device,
                        drop_last_if_one_extra=True,
                    )
                    if mask is not None:
                        return mask
        elif torch.is_tensor(batch_positive_map):
            if batch_positive_map.ndim == 3:
                mask = _resize_bool_mask(
                    batch_positive_map[batch_index, int(tgt_index)],
                    target_len=num_token_classes,
                    device=device,
                    drop_last_if_one_extra=True,
                )
                if mask is not None:
                    return mask
            else:
                offset = 0
                for i in range(batch_index):
                    offset += _num_target_boxes(targets[i])
                flat_index = offset + int(tgt_index)
                if 0 <= flat_index < int(batch_positive_map.shape[0]):
                    mask = _resize_bool_mask(
                        batch_positive_map[flat_index],
                        target_len=num_token_classes,
                        device=device,
                        drop_last_if_one_extra=True,
                    )
                    if mask is not None:
                        return mask

    target = targets[batch_index]
    if isinstance(target, dict) and target.get("positive_map", None) is not None:
        mask = _resize_bool_mask(
            target["positive_map"][int(tgt_index)],
            target_len=num_token_classes,
            device=device,
            drop_last_if_one_extra=True,
        )
        if mask is not None:
            return mask

    spans = _get_target_char_spans(targets, batch_index, tgt_index)
    if spans is None or tokenized is None:
        return None

    mask = torch.zeros(num_token_classes, dtype=torch.bool, device=device)
    for beg, end in spans:
        beg_pos = _char_to_token_with_fallback(tokenized, batch_index, int(beg), is_end=False)
        end_pos = _char_to_token_with_fallback(tokenized, batch_index, int(end) - 1, is_end=True)
        if beg_pos is None or end_pos is None:
            continue
        beg_pos = max(0, min(num_token_classes - 1, beg_pos))
        end_pos = max(0, min(num_token_classes - 1, end_pos))
        if end_pos >= beg_pos:
            mask[beg_pos : end_pos + 1] = True

    return mask if mask.any() else None


def _phrase_key_from_mask(mask: torch.Tensor):
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    return tuple(int(i) for i in idx.tolist())


def _collect_phrase_groups_for_item(
    batch_positive_map,
    targets,
    batch_index: int,
    num_token_classes: int,
    device: torch.device,
    tokenized=None,
):
    target = targets[batch_index]
    n_targets = _num_target_boxes(target)
    groups = {}

    for tgt_index in range(n_targets):
        pos_mask = _get_phrase_positive_mask(
            batch_positive_map=batch_positive_map,
            targets=targets,
            batch_index=batch_index,
            tgt_index=tgt_index,
            num_token_classes=num_token_classes,
            device=device,
            tokenized=tokenized,
        )
        if pos_mask is None or not pos_mask.any():
            continue

        key = _phrase_key_from_mask(pos_mask)
        if len(key) == 0:
            continue

        if key not in groups:
            groups[key] = {"mask": pos_mask, "tgt_indices": []}
        groups[key]["tgt_indices"].append(int(tgt_index))

    return list(groups.values())


def _compute_phrase_box_metrics_batch(
    outputs,
    targets,
    batch_positive_map,
    ks=(1, 5, 10),
    iou_threshold: float = PHRASE_BOX_IOU_THRESHOLD,
):
    pred_logits = outputs.get("pred_logits", None)
    pred_boxes = outputs.get("pred_boxes", None)
    if pred_logits is None or pred_boxes is None:
        return None

    device = pred_logits.device
    num_classes = int(pred_logits.shape[-1])
    if num_classes <= 1:
        return None

    num_token_classes = num_classes - 1
    tokenized = outputs.get("tokenized", None)
    valid_token_mask = _extract_valid_token_mask(outputs, device=device, num_token_classes=num_token_classes)
    if valid_token_mask is None:
        return None

    pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)

    hits = {int(k): 0.0 for k in ks}
    total = 0.0
    confidences = []
    correctness = []

    bsz, num_queries = pred_logits.shape[:2]

    for batch_index in range(bsz):
        if batch_index >= len(targets):
            break

        target = targets[batch_index]
        gt_boxes = target.get("boxes", None)
        if gt_boxes is None or len(gt_boxes) == 0:
            continue

        gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        phrase_groups = _collect_phrase_groups_for_item(
            batch_positive_map=batch_positive_map,
            targets=targets,
            batch_index=batch_index,
            num_token_classes=num_token_classes,
            device=device,
            tokenized=tokenized,
        )
        if not phrase_groups:
            continue

        token_logits = pred_logits[batch_index, :, :num_token_classes].clone()
        token_logits = token_logits.masked_fill(~valid_token_mask[batch_index].unsqueeze(0), float("-inf"))
        bg_logits = pred_logits[batch_index, :, -1:].clone()
        class_probs = torch.softmax(torch.cat([token_logits, bg_logits], dim=-1), dim=-1)
        token_probs = class_probs[:, :num_token_classes]

        for group in phrase_groups:
            pos_mask = group["mask"]
            tgt_indices = group["tgt_indices"]
            if pos_mask is None or not pos_mask.any() or len(tgt_indices) == 0:
                continue

            phrase_scores = token_probs[:, pos_mask].sum(dim=-1)
            if not torch.isfinite(phrase_scores).any():
                continue

            gt_phrase_boxes = gt_boxes_xyxy[tgt_indices]
            ious = box_ops.box_iou(pred_boxes_xyxy[batch_index], gt_phrase_boxes)[0]
            max_iou_per_query = ious.max(dim=1).values

            total += 1.0
            for k in hits:
                k_eff = min(int(k), int(num_queries))
                if k_eff <= 0:
                    continue
                topk_idx = torch.topk(phrase_scores, k_eff).indices
                hits[k] += float((max_iou_per_query[topk_idx] >= iou_threshold).any().item())

            top1_idx = int(torch.argmax(phrase_scores).item())
            confidences.append(float(phrase_scores[top1_idx].item()))
            correctness.append(float(max_iou_per_query[top1_idx].item() >= iou_threshold))

    return {
        "hits": hits,
        "total": total,
        "conf": confidences,
        "correct": correctness,
    }


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

    num_updates_per_epoch = (len(data_loader) + accum_steps - 1) // accum_steps
    num_steps_total = num_updates_per_epoch * args.epochs

    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(data_loader)

    for update_idx in logger.log_every(range(num_updates_per_epoch), print_freq, header):
        update_start = time.perf_counter()
        micro_batches_done = 0
        loss_sums: Dict[str, torch.Tensor] = {}

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

            if (criterion is not None) and (not getattr(args, "no_detection", False)):
                det_losses = criterion(outputs, targets, positive_map)
                loss_dict.update(det_losses)

            if (contrastive_criterion is not None) and getattr(args, "contrastive_loss", False):
                if memory_cache is not None:
                    t_pool = memory_cache.get("text_pooled_op")
                    i_pool = memory_cache.get("img_pooled_op")
                    if (t_pool is not None) and (i_pool is not None):
                        loss_dict["contrastive_loss"] = contrastive_criterion(t_pool, i_pool)

            loss_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

            if not math.isfinite(float(loss_total)):
                reduced = {k: (v.item() if hasattr(v, "item") else v) for k, v in loss_dict.items()}
                print("Non-finite loss, aborting.", reduced)
                raise RuntimeError("Non-finite loss encountered.")

            (loss_total / accum_steps).backward()

            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.detach()

        if micro_batches_done == 0:
            break

        if 0 < micro_batches_done < accum_steps:
            corr = accum_steps / float(micro_batches_done)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(corr)

        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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


def _named_coco_bbox_metrics(coco_eval_bbox_stats):
    if hasattr(coco_eval_bbox_stats, "tolist"):
        coco_eval_bbox_stats = coco_eval_bbox_stats.tolist()
    ce = list(coco_eval_bbox_stats)

    return {
        "ap": ce[0],
        "ap50": ce[1],
        "ap75": ce[2],
        "ap_small": ce[3],
        "ap_medium": ce[4],
        "ap_large": ce[5],
        "ar@1": ce[6],
        "ar@10": ce[7],
        "ar@100": ce[8],
        "ar_small": ce[9],
        "ar_medium": ce[10],
        "ar_large": ce[11],
    }


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
    """
    Run evaluation and update evaluators.

    Eval/test behavior:
    - show tqdm progress bar on main process
    - print average losses to console
    - print normal COCO summary block to console
    - return only phrase-box metrics in the final stats dict

    Validation-during-training behavior:
    - preserve the original return payload with losses + named COCO metrics
    """
    model.eval()
    for crit in (criterion, contrastive_criterion):
        if isinstance(crit, nn.Module):
            crit.eval()

    logger = MetricLogger(delimiter=" ")
    eval_mode = bool(getattr(args, "eval", False)) or bool(getattr(args, "test", False))

    run_phrase_box_metrics = (
        bool(getattr(args, "eval_phrase_box_metrics", False))
        and eval_mode
    )
    phrase_box_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    phrase_box_total = 0.0
    phrase_box_confidences = []
    phrase_box_correctness = []

    progress = None
    if eval_mode and _is_main_process() and tqdm is not None:
        total_samples = None
        try:
            total_samples = len(data_loader.dataset)
        except Exception:
            pass
        progress = tqdm(total=total_samples, desc="Test", unit="img", dynamic_ncols=True)

    try:
        iterator = data_loader if eval_mode else logger.log_every(data_loader, 50, "Test:")
        for batch_dict in iterator:
            samples = batch_dict["samples"].to(device)
            targets = batch_dict["targets"]
            captions = [t["caption"] for t in targets]

            positive_map = batch_dict.get("positive_map")
            if isinstance(positive_map, (list, tuple)):
                positive_map = [pm.to(device) for pm in positive_map]
            elif positive_map is not None:
                positive_map = positive_map.to(device)

            targets = targets_to(targets, device)

            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

            if run_phrase_box_metrics:
                outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
                metric_batch = _compute_phrase_box_metrics_batch(
                    outputs=outputs_without_aux,
                    targets=targets,
                    batch_positive_map=positive_map,
                    ks=(1, 5, 10),
                    iou_threshold=PHRASE_BOX_IOU_THRESHOLD,
                )
                if metric_batch is not None:
                    phrase_box_total += metric_batch["total"]
                    for k in phrase_box_hits:
                        phrase_box_hits[k] += metric_batch["hits"][k]
                    phrase_box_confidences.extend(metric_batch["conf"])
                    phrase_box_correctness.extend(metric_batch["correct"])

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
            if compact:
                logger.update(**compact)
            if loss_scl:
                logger.update(loss=sum(loss_scl.values()).item())

            if not getattr(args, "no_detection", False):
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors["bbox"](outputs, orig_target_sizes)

                for r in results:
                    if "labels" in r:
                        r["labels"] = torch.ones_like(r["labels"])

                res = {t["image_id"].item(): output for t, output in zip(targets, results)}
                for evaluator in evaluator_list:
                    evaluator.update(res)

            if progress is not None:
                batch_n = len(targets)
                progress.update(batch_n)
                if progress.total is not None:
                    remaining = max(int(progress.total - progress.n), 0)
                    progress.set_postfix_str(f"remaining={remaining}")
    finally:
        if progress is not None:
            progress.close()

    logger.synchronize_between_processes()

    loss_order = [
        "loss",
        "loss_ce",
        "loss_bbox",
        "loss_giou",
        "loss_contrastive_align",
        "contrastive_loss",
    ]
    avg_losses = {k: logger.meters[k].global_avg for k in loss_order if k in logger.meters}

    if eval_mode and _is_main_process():
        if avg_losses:
            pretty_losses = " ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items())
            print(f"Average losses: {pretty_losses}")
        else:
            print("Average losses: none")
    elif not eval_mode:
        print("Averaged stats:", logger)

    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

    if eval_mode:
        returned_stats = {}
        if run_phrase_box_metrics:
            counts = {
                "hit1": torch.tensor(phrase_box_hits[1], dtype=torch.float64, device=device),
                "hit5": torch.tensor(phrase_box_hits[5], dtype=torch.float64, device=device),
                "hit10": torch.tensor(phrase_box_hits[10], dtype=torch.float64, device=device),
                "total": torch.tensor(phrase_box_total, dtype=torch.float64, device=device),
            }
            counts = dist.reduce_dict(counts, average=False)

            hit1 = float(counts["hit1"].item())
            hit5 = float(counts["hit5"].item())
            hit10 = float(counts["hit10"].item())
            total = float(counts["total"].item())
            denom = total if total > 0 else 1.0

            returned_stats["phrase_box_recall@1"] = hit1 / denom
            returned_stats["phrase_box_recall@5"] = hit5 / denom
            returned_stats["phrase_box_recall@10"] = hit10 / denom

            phrase_box_confidences = _gather_list_across_processes(phrase_box_confidences)
            phrase_box_correctness = _gather_list_across_processes(phrase_box_correctness)
            returned_stats["phrase_box_ece@0.50"] = _compute_ece(
                phrase_box_confidences,
                phrase_box_correctness,
                n_bins=15,
            )
        return returned_stats

    stats = dict(avg_losses)
    for evaluator in evaluator_list:
        coco_eval = getattr(evaluator, "coco_eval", None)
        if isinstance(coco_eval, dict) and coco_eval.get("bbox") is not None:
            stats.update(_named_coco_bbox_metrics(coco_eval["bbox"].stats))
    return stats
