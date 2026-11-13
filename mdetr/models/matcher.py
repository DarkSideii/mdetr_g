# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.
"""Per-image Hungarian assignment for box and token supervision."""

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from mdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """Match foreground targets one-to-one; unmatched queries are background."""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Softmax(-1)
        weights = (cost_class, cost_bbox, cost_giou)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
            raise ValueError("Matching costs must be finite, nonnegative, and not all zero")

    @torch.no_grad()
    def forward(self, outputs, targets, positive_map):
        """Combine token-weighted probability, normalized-box L1, and negative GIoU."""
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4):
            raise ValueError("Matcher expects logits [B,Q,T+1] and normalized boxes [B,Q,4]")
        batch_size, _, token_dim = logits.shape
        if len(targets) != batch_size:
            raise ValueError("Matcher target count must equal the model batch size")
        if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
            raise ValueError("Matcher received non-finite prediction logits or boxes")
        if (boxes[..., 2:] < 0).any():
            raise ValueError("Matcher predicted cxcywh boxes have negative width/height")
        sizes = []
        for image_index, target in enumerate(targets):
            target_boxes = target["boxes"]
            if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
                raise ValueError(f"Matcher image {image_index}: target boxes must have shape [N,4]")
            if not torch.isfinite(target_boxes).all() or (target_boxes[:, 2:] <= 0).any():
                raise ValueError(
                    f"Matcher image {image_index}: target boxes must be finite with positive width/height"
                )
            sizes.append(len(target_boxes))
        if not torch.is_tensor(positive_map) or positive_map.shape != (sum(sizes), token_dim):
            raise ValueError(
                "Matcher needs one positive_map row per object and one column per logit"
            )
        if not torch.isfinite(positive_map).all() or (positive_map < 0).any():
            raise ValueError("Matcher positive_map weights must be finite and nonnegative")
        row_sums = positive_map.float().sum(-1)
        # Allow rounding of fractional token weights in lower-precision maps.
        normalization_tol = 1e-4
        if positive_map.is_floating_point():
            normalization_tol = max(normalization_tol, 3 * torch.finfo(positive_map.dtype).eps)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=normalization_tol, rtol=0):
            raise ValueError("Matcher positive_map rows must retain normalized fractional weights")
        if (positive_map[:, -1] != 0).any():
            raise ValueError(
                "Matcher foreground positive_map must not include the no-object column"
            )

        indices = []
        offset = 0
        # Explicit FP32 inputs alone do not prevent autocast of matrix products.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            for image_index, size in enumerate(sizes):
                if size == 0:
                    empty = torch.empty(0, dtype=torch.int64)
                    indices.append((empty, empty.clone()))
                    continue
                out_prob = self.norm(logits[image_index].float())
                out_bbox = boxes[image_index].float()
                tgt_bbox = targets[image_index]["boxes"].to(
                    device=boxes.device, dtype=torch.float32
                )
                token_rows = positive_map[offset : offset + size].to(
                    device=logits.device, dtype=torch.float32
                )
                offset += size

                # [Q,T] @ [T,N] avoids cross-image costs and a [Q,N,T] temporary.
                cost_class = -(out_prob @ token_rows.transpose(0, 1))
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
                )
                cost = (
                    self.cost_class * cost_class
                    + self.cost_bbox * cost_bbox
                    + self.cost_giou * cost_giou
                )
                if not torch.isfinite(cost).all():
                    raise ValueError(f"Matcher image {image_index}: non-finite Hungarian cost")
                source, target = linear_sum_assignment(cost.cpu())
                indices.append(
                    (
                        torch.as_tensor(source, dtype=torch.int64),
                        torch.as_tensor(target, dtype=torch.int64),
                    )
                )
        return indices


def build_matcher(args):
    """Factory for the configured matcher."""
    if args.set_loss == "hungarian":
        matcher = HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
        )
    else:
        raise ValueError(f"Only hungarian accepted, got {args.set_loss}")
    return matcher
