"""
Training/evaluation loops with gradient accumulation and optional EMA.

Logs a small set of primary loss components plus total loss, and can report
caption↔box alignment misses when a positive map is available.

Optional eval-only additions (gated by args.eval_token_span_metrics):
- contrastive_recall@1 / @5 / @10 on matched object-query -> token retrieval
- token_span_ece from the soft token prediction head (pred_logits)
"""

import math
import time
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _extract_valid_token_mask(outputs, device: torch.device):
    tokenized = outputs.get("tokenized", None)

    if tokenized is not None:
        attn = None
        try:
            attn = tokenized["attention_mask"]
        except Exception:
            attn = getattr(tokenized, "attention_mask", None)

        if attn is not None:
            return attn.to(device).bool()

    if outputs.get("text_attention_mask", None) is not None:
        return outputs["text_attention_mask"].to(device).bool()

    if outputs.get("proj_tokens", None) is not None:
        bsz, num_tokens = outputs["proj_tokens"].shape[:2]
        return torch.ones((bsz, num_tokens), dtype=torch.bool, device=device)

    return None


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


def _get_alignment_positive_mask(
    batch_positive_map,
    targets,
    batch_index: int,
    tgt_index: int,
    tokenized,
    num_tokens: int,
    device: torch.device,
):
    # Preferred path when the collate function already provides positive maps.
    if batch_positive_map is not None:
        if isinstance(batch_positive_map, (list, tuple)):
            if batch_index < len(batch_positive_map):
                per_item_pm = batch_positive_map[batch_index]
                if per_item_pm is not None and int(tgt_index) < len(per_item_pm):
                    mask = _resize_bool_mask(
                        per_item_pm[int(tgt_index)],
                        target_len=num_tokens,
                        device=device,
                        drop_last_if_one_extra=True,
                    )
                    if mask is not None:
                        return mask
        elif torch.is_tensor(batch_positive_map):
            if batch_positive_map.ndim == 3:
                mask = _resize_bool_mask(
                    batch_positive_map[batch_index, int(tgt_index)],
                    target_len=num_tokens,
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
                        target_len=num_tokens,
                        device=device,
                        drop_last_if_one_extra=True,
                    )
                    if mask is not None:
                        return mask

    target = targets[batch_index]

    # Fallback if the target already carries its own positive map.
    if isinstance(target, dict) and target.get("positive_map", None) is not None:
        mask = _resize_bool_mask(
            target["positive_map"][int(tgt_index)],
            target_len=num_tokens,
            device=device,
            drop_last_if_one_extra=True,
        )
        if mask is not None:
            return mask

    # Final fallback: reconstruct token supervision from char spans.
    spans = None
    if isinstance(target, dict) and target.get("tokens_positive", None) is not None:
        spans = target["tokens_positive"][int(tgt_index)]
    elif isinstance(target, dict) and target.get("tokens", None) is not None:
        spans = target["tokens"][int(tgt_index)]

    if spans is None:
        return None
    if not spans:
        return None
    if isinstance(spans, (list, tuple)) and len(spans) > 0 and spans[0] is None:
        return None

    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    for beg, end in spans:
        beg_pos = _char_to_token_with_fallback(tokenized, batch_index, int(beg), is_end=False)
        end_pos = _char_to_token_with_fallback(tokenized, batch_index, int(end) - 1, is_end=True)
        if beg_pos is None or end_pos is None:
            continue

        beg_pos = max(0, min(num_tokens - 1, beg_pos))
        end_pos = max(0, min(num_tokens - 1, end_pos))
        if end_pos >= beg_pos:
            mask[beg_pos : end_pos + 1] = True

    return mask if mask.any() else None


def _get_align_scale(model: nn.Module, criterion: Optional[nn.Module], args, device, dtype):
    model_for_scale = model.module if hasattr(model, "module") else model

    if getattr(args, "align_scale_mode", "learnable") == "learnable" and hasattr(model_for_scale, "logit_scale_align"):
        try:
            return model_for_scale.logit_scale_align.exp().detach().to(device=device, dtype=dtype)
        except Exception:
            pass

    if criterion is not None and hasattr(criterion, "temperature"):
        try:
            temp = float(getattr(criterion, "temperature"))
            if temp > 0:
                return torch.as_tensor(1.0 / temp, device=device, dtype=dtype)
        except Exception:
            pass

    temp = float(getattr(args, "temperature_NCE", 0.07))
    temp = temp if temp > 0 else 0.07
    return torch.as_tensor(1.0 / temp, device=device, dtype=dtype)


def _mask_class_logits_for_valid_tokens(cls_logits: torch.Tensor, valid_token_mask_1d: torch.Tensor, num_tokens: int):
    """
    Masks padded / invalid token positions while preserving the optional no-object logit.

    The returned tensor is still suitable for a full softmax over token classes + no-object.
    """
    cls_logits = cls_logits.clone()
    num_classes = int(cls_logits.numel())
    if num_classes <= 0:
        return cls_logits

    has_no_object = num_classes > num_tokens
    num_token_classes = num_classes - 1 if has_no_object else num_classes

    valid_cls_mask = torch.zeros(num_token_classes, dtype=torch.bool, device=cls_logits.device)
    use_n = min(num_token_classes, num_tokens, int(valid_token_mask_1d.numel()))
    if use_n > 0:
        valid_cls_mask[:use_n] = valid_token_mask_1d[:use_n]

    if num_token_classes > 0:
        cls_logits[:num_token_classes] = cls_logits[:num_token_classes].masked_fill(~valid_cls_mask, float("-inf"))

    return cls_logits


def _compute_eval_token_metrics_batch(
    outputs,
    targets,
    indices,
    batch_positive_map,
    model: nn.Module,
    criterion: Optional[nn.Module],
    args,
    ks=(1, 5, 10),
):
    if outputs.get("proj_queries", None) is None or outputs.get("proj_tokens", None) is None:
        return None

    proj_queries = outputs["proj_queries"]
    proj_tokens = outputs["proj_tokens"]
    device = proj_queries.device
    num_tokens = int(proj_tokens.shape[1])

    valid_token_mask = _extract_valid_token_mask(outputs, device)
    valid_token_mask = _resize_token_mask(valid_token_mask, num_tokens, device)
    if valid_token_mask is None:
        return None

    q = F.normalize(proj_queries, p=2, dim=-1)
    t = F.normalize(proj_tokens, p=2, dim=-1)
    align_scale = _get_align_scale(model, criterion, args, device=device, dtype=q.dtype)
    align_logits = align_scale * torch.einsum("bqd,btd->bqt", q, t)

    pred_logits = outputs.get("pred_logits", None)
    tokenized = outputs.get("tokenized", None)

    hits = {int(k): 0.0 for k in ks}
    total = 0.0
    token_span_confidences = []
    token_span_correctness = []

    for batch_index, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue

        valid_mask_b = valid_token_mask[batch_index]
        valid_count = int(valid_mask_b.sum().item())
        if valid_count <= 0:
            continue

        for query_idx, gt_idx in zip(src_idx.tolist(), tgt_idx.tolist()):
            pos_mask = _get_alignment_positive_mask(
                batch_positive_map=batch_positive_map,
                targets=targets,
                batch_index=batch_index,
                tgt_index=gt_idx,
                tokenized=tokenized,
                num_tokens=num_tokens,
                device=device,
            )
            if pos_mask is None:
                continue

            pos_mask = pos_mask & valid_mask_b
            if not pos_mask.any():
                continue

            # Contrastive retrieval Recall@K from proj_queries -> proj_tokens.
            logits = align_logits[batch_index, query_idx].masked_fill(~valid_mask_b, float("-inf"))
            ranking = torch.argsort(logits, descending=True)
            total += 1.0
            for k in hits:
                k_eff = min(int(k), valid_count)
                if k_eff <= 0:
                    continue
                topk = ranking[:k_eff]
                hits[k] += float(pos_mask[topk].any().item())

            # Token-span calibration from the soft token prediction head (pred_logits).
            if pred_logits is not None:
                cls_logits = _mask_class_logits_for_valid_tokens(
                    pred_logits[batch_index, query_idx],
                    valid_token_mask_1d=valid_mask_b,
                    num_tokens=num_tokens,
                )
                cls_probs = torch.softmax(cls_logits, dim=-1)
                cls_conf_val, cls_pred_idx = torch.max(cls_probs, dim=-1)

                cls_target_mask = pos_mask
                if cls_probs.numel() > cls_target_mask.numel():
                    pad = torch.zeros(
                        cls_probs.numel() - cls_target_mask.numel(),
                        dtype=torch.bool,
                        device=device,
                    )
                    cls_target_mask = torch.cat([cls_target_mask, pad], dim=0)
                elif cls_probs.numel() < cls_target_mask.numel():
                    cls_target_mask = cls_target_mask[: cls_probs.numel()]

                token_span_confidences.append(float(cls_conf_val.item()))
                token_span_correctness.append(
                    1.0 if bool(cls_target_mask[int(cls_pred_idx.item())].item()) else 0.0
                )

    return {
        "hits": hits,
        "total": total,
        "token_span_conf": token_span_confidences,
        "token_span_correct": token_span_correctness,
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

    run_token_span_metrics = (
        bool(getattr(args, "eval_token_span_metrics", False))
        and (bool(getattr(args, "eval", False)) or bool(getattr(args, "test", False)))
    )

    contrastive_hits = {1: 0.0, 5: 0.0, 10: 0.0}
    contrastive_total = 0.0
    token_span_confidences = []
    token_span_correctness = []

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

        if run_token_span_metrics and criterion is not None and hasattr(criterion, "matcher"):
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

            try:
                indices = criterion.matcher(outputs_without_aux, targets, positive_map)
            except TypeError:
                try:
                    indices = criterion.matcher(outputs_without_aux, targets)
                except TypeError:
                    indices = None

            if indices is not None:
                metric_batch = _compute_eval_token_metrics_batch(
                    outputs=outputs_without_aux,
                    targets=targets,
                    indices=indices,
                    batch_positive_map=positive_map,
                    model=model,
                    criterion=criterion,
                    args=args,
                    ks=(1, 5, 10),
                )
                if metric_batch is not None:
                    contrastive_total += metric_batch["total"]
                    for k in contrastive_hits:
                        contrastive_hits[k] += metric_batch["hits"][k]
                    token_span_confidences.extend(metric_batch["token_span_conf"])
                    token_span_correctness.extend(metric_batch["token_span_correct"])

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

    if run_token_span_metrics:
        counts = {
            "hit1": torch.tensor(contrastive_hits[1], dtype=torch.float64, device=device),
            "hit5": torch.tensor(contrastive_hits[5], dtype=torch.float64, device=device),
            "hit10": torch.tensor(contrastive_hits[10], dtype=torch.float64, device=device),
            "total": torch.tensor(contrastive_total, dtype=torch.float64, device=device),
        }
        counts = dist.reduce_dict(counts, average=False)

        hit1 = float(counts["hit1"].item())
        hit5 = float(counts["hit5"].item())
        hit10 = float(counts["hit10"].item())
        total = float(counts["total"].item())
        denom = total if total > 0 else 1.0

        stats["contrastive_recall@1"] = hit1 / denom
        stats["contrastive_recall@5"] = hit5 / denom
        stats["contrastive_recall@10"] = hit10 / denom

        token_span_confidences = _gather_list_across_processes(token_span_confidences)
        token_span_correctness = _gather_list_across_processes(token_span_correctness)
        stats["token_span_ece"] = _compute_ece(token_span_confidences, token_span_correctness, n_bins=15)

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
