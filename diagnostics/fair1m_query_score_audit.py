#!/usr/bin/env python3
"""Read-only FAIR1M query-score diagnostic for the published MDETR-G architecture.

Measures high-confidence duplicate predictions, score-vs-IoU misranking,
top-100 vs top-300 GT coverage, and a GT-informed *diagnostic* score oracle.
The oracle uses validation labels and is NOT a deployable AP result.

No model, data, checkpoint, attention, or postprocessing code is modified.
"""
import argparse
import contextlib
import copy
import csv
import io
import json
from argparse import Namespace
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

import mdetr.util.misc as utils
from geo_datasets import build_dataset
from main import _load_dataset_config
from mdetr.models import build_model
from mdetr.models.postprocessors import build_postprocessors

SCORE_CUTS = (0.1, 0.3, 0.5, 0.7)
BINS = ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
        (0.3, 0.5), (0.5, 0.7), (0.7, 1.000001))
METRICS = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")


def iou_matrix(boxes, gt):
    if boxes.shape[0] == 0 or gt.shape[0] == 0:
        return torch.zeros((boxes.shape[0], gt.shape[0]), dtype=torch.float32)
    lt = torch.maximum(boxes[:, None, :2], gt[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    inter_wh = (rb - lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(-1)
    area_g = (gt[:, 2:] - gt[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_b[:, None] + area_g[None, :] - inter).clamp(min=1e-9)


def size_bin(area):
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def group_for(cls):
    if cls in ("dry cargo ship", "liquid cargo ship"):
        return "cargo ships"
    if cls == "motorboat":
        return "motorboats"
    return "land vehicles"


def scopes(cls, size):
    return (
        ("group", group_for(cls)),
        ("class", cls),
        ("size", size),
        ("class_size", cls + " / " + size),
    )


def match_top100(ious, threshold):
    """Score-ordered greedy, class-agnostic matching; rows are sorted by score.

    Duplicate means an IoU>=threshold overlap exists, but all qualifying GT
    are already claimed by higher-scoring detections.
    """
    matched = set()
    result = []
    for row in ious[:100]:
        if row.numel() == 0:
            result.append(("background", -1, 0.0))
            continue
        best_iou, nearest = row.max(dim=0)
        eligible = [j for j in range(len(row))
                    if j not in matched and float(row[j]) >= threshold]
        if eligible:
            j = max(eligible, key=lambda k: float(row[k]))
            matched.add(j)
            result.append(("tp", j, float(best_iou)))
        elif float(best_iou) >= threshold:
            result.append(("duplicate", int(nearest), float(best_iou)))
        elif float(best_iou) >= 0.1:
            result.append(("localization", int(nearest), float(best_iou)))
        else:
            result.append(("background", -1, float(best_iou)))
    return result


def coco_records(image_id, boxes, scores, category_id):
    xywh = boxes.clone()
    xywh[:, 2:] -= xywh[:, :2]
    return [
        {"image_id": image_id, "category_id": category_id,
         "bbox": box, "score": float(score)}
        for box, score in zip(xywh.tolist(), scores.tolist())
    ]


def coco_metrics(gt, detections, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.useCats = 0
        ev.params.imgIds = sorted(image_ids)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(zip(METRICS, [float(x) for x in ev.stats[:6]]))


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def scope_rows(scope_stats, which):
    rows = []
    for (kind, name), c in sorted(scope_stats.items()):
        if kind != which or c["gt"] == 0:
            continue
        n = c["gt"]
        rows.append({
            "name": name,
            "gt": n,
            "matched50_top100": c["tp50"],
            "recall50_top100": round(c["tp50"] / n, 4),
            "recall75_top100": round(c["tp75"] / n, 4),
            "coverage50_top100": round(c["cover50_100"] / n, 4),
            "coverage50_top300": round(c["cover50_300"] / n, 4),
            "coverage75_top300": round(c["cover75_300"] / n, 4),
            "mean_best_iou300": round(c["sum_best_iou300"] / n, 4),
            "gt_with_good75_but_bad_score_order": c["misrank75"],
            "gt_with_iou50_only_after_rank100": c["after100_50"],
            "duplicate50": c["duplicate50"],
            "duplicate50_score_ge_03": c["duplicate50_hi"],
            "localization50": c["localization50"],
            "localization50_score_ge_03": c["localization50_hi"],
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--output", default="runs/fair1m_query_score_audit")
    p.add_argument("--batch-size", type=int, default=2)
    args_cli = p.parse_args()

    output = Path(args_cli.output)
    output.mkdir(parents=True, exist_ok=True)

    # The checkpoint must be one you trust; weights_only=False unpickles args.
    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu",
                            weights_only=False)
    args = copy.deepcopy(checkpoint["args"])
    if isinstance(args, dict):
        args = Namespace(**args)
    args.dataset_config = args_cli.dataset_config
    args.dataset_file = "fair1m"
    _load_dataset_config(args)
    if args.dataset_file != "fair1m":
        raise ValueError("Dataset config unexpectedly overrides dataset_file")

    device = torch.device("cuda")
    model, _, _, _ = build_model(args)
    use_ema = bool(getattr(args, "ema", False)
                   and checkpoint.get("model_ema") is not None)
    weights = checkpoint["model_ema"] if use_ema else checkpoint["model"]
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()

    args.tokenizer = model.tokenizer
    dataset = build_dataset("fair1m", "val", args)
    coco = dataset.coco
    category_ids = sorted(coco.getCatIds())
    if not category_ids:
        raise RuntimeError("No COCO category IDs in validation set")
    category_id = category_ids[0]

    loader = DataLoader(
        dataset, batch_size=args_cli.batch_size, shuffle=False, num_workers=2,
        collate_fn=partial(utils.collate_fn, False),
    )
    postprocess = build_postprocessors(args, "fair1m")["bbox"]

    scope_stats = defaultdict(Counter)
    global_stats = defaultdict(Counter)
    score_bins = defaultdict(Counter)
    score_thresholds = defaultdict(Counter)
    scores_by_status = defaultdict(list)
    gt_rows, image_rows, high_fp = [], [], []
    baseline_records, oracle_records = [], []
    seen = set()

    with torch.inference_mode():
        for batch_no, batch in enumerate(loader, start=1):
            samples = batch["samples"].to(device)
            targets = batch["targets"]
            captions = [t["caption"] for t in targets]
            memory = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False,
                            memory_cache=memory)
            target_sizes = torch.stack(
                [t["orig_size"] for t in targets]
            ).to(device)
            predictions = postprocess(outputs, target_sizes)

            for target, pred in zip(targets, predictions):
                image_id = int(target["image_id"].reshape(-1)[0])
                if image_id in seen:
                    raise RuntimeError("Duplicate validation image ID: %d" % image_id)
                seen.add(image_id)

                anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
                gt_boxes = torch.tensor([
                    [a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2],
                     a["bbox"][1] + a["bbox"][3]]
                    for a in anns
                ], dtype=torch.float32).reshape(-1, 4)
                keys = [
                    (coco.cats[int(a["category_id"])]["name"].lower(),
                     size_bin(float(a["area"])))
                    for a in anns
                ]

                boxes = pred["boxes"].detach().float().cpu()
                scores = pred["scores"].detach().float().cpu()
                order = torch.argsort(scores, descending=True, stable=True)
                boxes, scores = boxes[order], scores[order]
                if len(scores) != 300:
                    raise RuntimeError(
                        "Expected 300 queries; got %d for image %d" %
                        (len(scores), image_id)
                    )

                overlaps = iou_matrix(boxes, gt_boxes)
                max_overlaps = (overlaps.max(dim=1).values if len(anns)
                                else torch.zeros(len(scores)))

                baseline_records.extend(coco_records(
                    image_id, boxes[:100], scores[:100], category_id
                ))

                # Label-informed score oracle: diagnostic only, NEVER test AP.
                oracle_order = torch.argsort(
                    max_overlaps, descending=True, stable=True
                )[:100]
                oracle_records.extend(coco_records(
                    image_id, boxes[oracle_order],
                    max_overlaps[oracle_order], category_id
                ))

                matches = {
                    50: match_top100(overlaps, 0.5),
                    75: match_top100(overlaps, 0.75),
                }

                per_image = Counter()
                for rank in range(100):
                    score = float(scores[rank])
                    for t in (50, 75):
                        status, gt_j, best_iou = matches[t][rank]
                        global_stats[t][status] += 1
                        per_image["%s_%d" % (status, t)] += 1
                        scores_by_status[(t, status)].append(score)
                        for cut in SCORE_CUTS:
                            if score >= cut:
                                score_thresholds[(t, cut)][status] += 1
                        for lo, hi in BINS:
                            if lo <= score < hi:
                                score_bins[(t, lo, hi)][status] += 1
                                break
                        if gt_j >= 0:
                            cls, size = keys[gt_j]
                            for scope in scopes(cls, size):
                                scope_stats[scope][status + str(t)] += 1
                                if t == 50 and score >= 0.3:
                                    if status == "duplicate":
                                        scope_stats[scope]["duplicate50_hi"] += 1
                                    elif status == "localization":
                                        scope_stats[scope]["localization50_hi"] += 1

                    status50, nearest, best_iou = matches[50][rank]
                    if score >= 0.3 and status50 != "tp":
                        cls, size = keys[nearest] if nearest >= 0 else ("", "")
                        high_fp.append({
                            "image_id": image_id, "rank": rank + 1,
                            "score": round(score, 6), "status50": status50,
                            "status75": matches[75][rank][0],
                            "nearest_gt_iou": round(best_iou, 5),
                            "nearest_gt_class": cls, "nearest_gt_size": size,
                            "x0": round(float(boxes[rank, 0]), 2),
                            "y0": round(float(boxes[rank, 1]), 2),
                            "x1": round(float(boxes[rank, 2]), 2),
                            "y1": round(float(boxes[rank, 3]), 2),
                        })

                for j, ann in enumerate(anns):
                    cls, size = keys[j]
                    col = overlaps[:, j]
                    best_iou300, best_rank = col.max(dim=0)
                    best_iou100 = float(col[:100].max())
                    b300 = float(best_iou300)
                    good50 = col >= 0.5
                    good75 = col >= 0.75
                    weak = (col >= 0.5) & (col < 0.75)

                    max_score50 = (
                        float(scores[good50].max()) if good50.any() else None
                    )
                    max_score75 = (
                        float(scores[good75].max()) if good75.any() else None
                    )
                    misrank75 = bool(
                        good75.any() and weak.any() and
                        float(scores[weak].max()) > float(scores[good75].max())
                    )
                    after100 = b300 >= 0.5 and best_iou100 < 0.5

                    gt_row = {
                        "image_id": image_id, "class": cls, "size": size,
                        "area": round(float(ann["area"]), 2),
                        "best_iou300": round(b300, 5),
                        "best_iou100": round(best_iou100, 5),
                        "rank_of_best_iou": int(best_rank) + 1,
                        "score_of_best_iou": round(float(scores[best_rank]), 6),
                        "max_score_iou50": max_score50,
                        "max_score_iou75": max_score75,
                        "num_queries_iou50": int(good50.sum()),
                        "num_queries_iou75": int(good75.sum()),
                        "misrank75": int(misrank75),
                        "iou50_only_after_rank100": int(after100),
                    }
                    gt_rows.append(gt_row)

                    for scope in scopes(cls, size):
                        c = scope_stats[scope]
                        c["gt"] += 1
                        c["sum_best_iou300"] += b300
                        c["cover50_100"] += int(best_iou100 >= 0.5)
                        c["cover50_300"] += int(b300 >= 0.5)
                        c["cover75_300"] += int(b300 >= 0.75)
                        c["misrank75"] += int(misrank75)
                        c["after100_50"] += int(after100)

                image_rows.append({
                    "image_id": image_id, "gt": len(anns),
                    "tp50": per_image["tp_50"],
                    "duplicate50": per_image["duplicate_50"],
                    "localization50": per_image["localization_50"],
                    "background50": per_image["background_50"],
                    "tp75": per_image["tp_75"],
                    "duplicate75": per_image["duplicate_75"],
                })

            if batch_no % 25 == 0:
                print("Processed %d images" % len(seen), flush=True)

    expected_gt = sum(c["gt"] for (kind, _), c in scope_stats.items()
                      if kind == "group")
    print("Validation: %d images, %d GT" % (len(seen), expected_gt))
    if len(seen) != 461 or expected_gt != 2334:
        raise RuntimeError("Expected 461 images / 2334 GT; check validation split")

    print("Computing COCO baseline and GT-informed score-oracle metrics...")
    baseline_ap = coco_metrics(coco, baseline_records, seen)
    oracle_ap = coco_metrics(coco, oracle_records, seen)

    for which, filename in (
        ("group", "per_group.csv"),
        ("class", "per_class.csv"),
        ("size", "per_size.csv"),
        ("class_size", "per_class_size.csv"),
    ):
        rows = scope_rows(scope_stats, which)
        if rows:
            save_csv(output / filename, rows, list(rows[0]))

    score_rows = []
    for (t, lo, hi), c in sorted(score_bins.items()):
        total = sum(c.values())
        score_rows.append({
            "iou_threshold": t / 100, "score_min": lo,
            "score_max_exclusive": hi, "detections": total,
            "tp": c["tp"], "duplicate": c["duplicate"],
            "localization": c["localization"], "background": c["background"],
            "precision_at_bin": round(c["tp"] / total, 5) if total else None,
        })
    save_csv(output / "score_bins.csv", score_rows, list(score_rows[0]))

    threshold_rows = []
    for (t, cut), c in sorted(score_thresholds.items()):
        total = sum(c.values())
        threshold_rows.append({
            "iou_threshold": t / 100, "score_ge": cut,
            "detections": total, "tp": c["tp"],
            "duplicate": c["duplicate"], "localization": c["localization"],
            "background": c["background"],
            "precision": round(c["tp"] / total, 5) if total else None,
        })
    save_csv(output / "score_thresholds.csv", threshold_rows,
             list(threshold_rows[0]))

    save_csv(output / "per_gt.csv", gt_rows, list(gt_rows[0]))
    save_csv(output / "per_image.csv", image_rows, list(image_rows[0]))
    high_fp.sort(key=lambda row: row["score"], reverse=True)
    fp_columns = ("image_id", "rank", "score", "status50", "status75",
                  "nearest_gt_iou", "nearest_gt_class", "nearest_gt_size",
                  "x0", "y0", "x1", "y1")
    save_csv(output / "high_score_false_positives.csv",
             high_fp[:500], fp_columns)

    quantiles = {}
    for (t, status), values in sorted(scores_by_status.items()):
        a = np.asarray(values, dtype=np.float64)
        quantiles["iou%d_%s" % (t, status)] = {
            "count": len(values), "median_score": float(np.median(a)),
            "p90_score": float(np.quantile(a, 0.9)),
            "p99_score": float(np.quantile(a, 0.99)),
        }

    report = {
        "validation_images": len(seen),
        "validation_gt": expected_gt,
        "score_definition": "1 - P(no-object), original MDETR-G",
        "query_count": 300,
        "evaluation_max_detections": 100,
        "checkpoint_ema": use_ema,
        "baseline_coco": baseline_ap,
        "gt_informed_iou_score_oracle_NOT_DEPLOYABLE": oracle_ap,
        "global_status_top100": {
            str(t): dict(c) for t, c in global_stats.items()
        },
        "score_quantiles_by_status": quantiles,
        "notes": [
            "GT-informed oracle uses validation ground truth; never report as test AP.",
            "Duplicate means overlap >= IoU threshold with already-matched GT.",
            "Nearest-GT class attribution is approximate for crowded scenes.",
            "GT coverage ignores score; greedy TP counts are score-ranked.",
            "No architecture, data, model weights, or inference settings changed.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("\nCOCO AP baseline: %.4f | AP50: %.4f | AP75: %.4f" %
          (baseline_ap["ap"], baseline_ap["ap50"], baseline_ap["ap75"]))
    print("GT-informed score oracle (DIAGNOSTIC ONLY): AP %.4f | AP75 %.4f" %
          (oracle_ap["ap"], oracle_ap["ap75"]))
    for t in (50, 75):
        print("IoU %.2f, top100 status: %s" %
              (t / 100, dict(global_stats[t])))
    print("Saved diagnostic files to", output)


if __name__ == "__main__":
    main()
