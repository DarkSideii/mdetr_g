# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Independent matching/loss formulas and supervision-gradient invariants."""

import unittest

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mdetr.models.matcher import HungarianMatcher
from mdetr.models.mdetr import SetCriterion
from .common import device


def _reference_giou(boxes, targets):
    """Pairwise GIoU from normalized cxcywh, independent of production box_ops."""
    box_min, box_max = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
    target_min = targets[:, :2] - targets[:, 2:] / 2
    target_max = targets[:, :2] + targets[:, 2:] / 2
    intersection = (
        (torch.minimum(box_max[:, None], target_max) - torch.maximum(box_min[:, None], target_min))
        .clamp_min(0)
        .prod(-1)
    )
    union = (
        (box_max - box_min).prod(-1)[:, None] + (target_max - target_min).prod(-1) - intersection
    )
    enclosing = (
        torch.maximum(box_max[:, None], target_max) - torch.minimum(box_min[:, None], target_min)
    ).prod(-1)
    return intersection / union - (enclosing - union) / enclosing


class TestMatcherAndLosses(unittest.TestCase):
    def test_hungarian_cost_matches_independent_formula_per_image(self):
        torch.manual_seed(12)
        output = {
            "pred_logits": torch.randn(2, 5, 9),
            "pred_boxes": torch.tensor(
                [
                    [
                        [0.20, 0.30, 0.12, 0.11],
                        [0.70, 0.30, 0.10, 0.12],
                        [0.40, 0.60, 0.16, 0.10],
                        [0.75, 0.80, 0.10, 0.13],
                        [0.12, 0.76, 0.09, 0.10],
                    ],
                    [
                        [0.80, 0.20, 0.10, 0.10],
                        [0.25, 0.75, 0.16, 0.12],
                        [0.25, 0.22, 0.15, 0.11],
                        [0.45, 0.64, 0.13, 0.14],
                        [0.73, 0.71, 0.11, 0.09],
                    ],
                ]
            ),
        }
        targets = [
            {"boxes": torch.tensor([[0.21, 0.32, 0.12, 0.10], [0.73, 0.81, 0.10, 0.12]])},
            {"boxes": torch.tensor([[0.24, 0.22, 0.13, 0.11]])},
        ]
        positive = torch.zeros((3, 9))
        positive[0, 1], positive[0, 3] = 0.25, 0.75
        positive[1, 2], positive[2, 4] = 1.0, 1.0
        matcher = HungarianMatcher(cost_class=3, cost_bbox=5, cost_giou=2)
        actual = matcher(output, targets, positive)

        offset = 0
        for i, target in enumerate(targets):
            n = len(target["boxes"])
            pred = output["pred_boxes"][i]
            cost = (
                -3 * output["pred_logits"][i].softmax(-1) @ positive[offset : offset + n].T
                + 5 * torch.cdist(pred, target["boxes"], p=1)
                - 2 * _reference_giou(pred, target["boxes"])
            )
            expected_src, expected_tgt = linear_sum_assignment(cost.detach().numpy())
            self.assertEqual(actual[i][0].tolist(), expected_src.tolist())
            self.assertEqual(actual[i][1].tolist(), expected_tgt.tolist())
            self.assertEqual(
                len(set(actual[i][0].tolist())), n, "Each GT must have a distinct query"
            )
            offset += n

    def test_token_class_cost_changes_assignment_when_geometry_ties(self):
        logits = torch.tensor(
            [[[10.0, 0.0, 0.0, -5.0], [0.0, 10.0, 0.0, -5.0], [0.0, 0.0, 10.0, -5.0]]]
        )
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * 3])
        gt = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])}]
        positive = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        matcher = HungarianMatcher(cost_class=100.0, cost_bbox=0.0, cost_giou=0.0)
        src, tgt = matcher({"pred_logits": logits, "pred_boxes": boxes}, gt, positive)[0]
        mapping = dict(zip(tgt.tolist(), src.tolist()))
        self.assertEqual(mapping, {0: 0, 1: 1})
        flipped = matcher({"pred_logits": logits, "pred_boxes": boxes}, gt, positive.flip(0))[0]
        flipped_mapping = dict(zip(flipped[1].tolist(), flipped[0].tolist()))
        self.assertEqual(flipped_mapping, {0: 1, 1: 0})

    def test_empty_image_uses_no_other_images_targets(self):
        outputs = {"pred_logits": torch.randn(2, 3, 5), "pred_boxes": torch.full((2, 3, 4), 0.2)}
        targets = [{"boxes": torch.empty((0, 4))}, {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        positive = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        result = HungarianMatcher(1, 5, 2)(outputs, targets, positive)
        self.assertEqual([len(s) for s, _ in result], [0, 1])
        self.assertEqual(result[1][1].tolist(), [0])

    def test_independent_ce_and_box_losses_with_fractional_tokens(self):
        criterion = SetCriterion(4, HungarianMatcher(1, 5, 2), 0.1, ["labels", "boxes"], 0.07)
        logits = torch.tensor(
            [
                [
                    [2.0, 1.0, -1.0, 0.0, -2.0],
                    [0.0, 1.0, 2.0, -1.0, -3.0],
                    [-1.0, -2.0, 0.0, 1.0, 3.0],
                ]
            ],
            requires_grad=True,
        )
        boxes = torch.tensor(
            [[[0.20, 0.28, 0.14, 0.12], [0.74, 0.73, 0.16, 0.10], [0.30, 0.52, 0.14, 0.13]]],
            requires_grad=True,
        )
        targets = [
            {
                "boxes": torch.tensor([[0.70, 0.70, 0.14, 0.12], [0.24, 0.26, 0.12, 0.12]]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        positive = torch.tensor([[0.0, 0.4, 0.6, 0.0, 0.0], [0.8, 0.0, 0.0, 0.2, 0.0]])
        indices = [(torch.tensor([0, 1]), torch.tensor([1, 0]))]
        ce = criterion.loss_labels({"pred_logits": logits}, targets, positive, indices, 2)[
            "loss_ce"
        ]

        target_distribution = torch.zeros_like(logits)
        target_distribution[0, 0] = positive[1]
        target_distribution[0, 1] = positive[0]
        target_distribution[0, 2, -1] = 1
        per_query = -(target_distribution * F.log_softmax(logits.float(), -1)).sum(-1)
        expected_ce = (per_query[0, 0] + per_query[0, 1] + 0.1 * per_query[0, 2]) / 2
        self.assertTrue(torch.allclose(ce, expected_ce, atol=1e-6))

        actual = criterion.loss_boxes({"pred_boxes": boxes}, targets, positive, indices, 2)
        matched = boxes[0, :2]
        ground = targets[0]["boxes"][[1, 0]]
        expected_l1 = (matched - ground).abs().sum() / 2
        expected_giou = (1 - torch.diag(_reference_giou(matched, ground))).sum() / 2
        self.assertTrue(torch.allclose(actual["loss_bbox"], expected_l1, atol=1e-6))
        self.assertTrue(torch.allclose(actual["loss_giou"], expected_giou, atol=1e-6))

    def test_matched_box_grads_are_distinct_unmatched_box_has_none(self):
        criterion = SetCriterion(6, HungarianMatcher(1, 5, 2), 0.1, ["boxes"], 0.07)
        boxes = torch.tensor(
            [[[0.40, 0.30, 0.15, 0.12], [0.60, 0.70, 0.12, 0.11], [0.50, 0.50, 0.10, 0.10]]],
            requires_grad=True,
        )
        targets = [{"boxes": torch.tensor([[0.20, 0.30, 0.15, 0.12], [0.80, 0.70, 0.12, 0.11]])}]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        loss = criterion.loss_boxes({"pred_boxes": boxes}, targets, None, indices, 2)["loss_bbox"]
        loss.backward()
        self.assertGreater(float(boxes.grad[0, 0, 0]), 0.0)
        self.assertLess(float(boxes.grad[0, 1, 0]), 0.0)
        self.assertTrue(torch.equal(boxes.grad[0, 2], torch.zeros(4)))

    def test_empty_targets_keep_background_loss_without_box_gradients(self):
        logits = torch.zeros(2, 3, 5, requires_grad=True)
        boxes = torch.full((2, 3, 4), 0.2, requires_grad=True)
        targets = [
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
            for _ in range(2)
        ]
        criterion = SetCriterion(4, HungarianMatcher(1, 5, 2), 0.1, ["labels", "boxes"], 0.07)
        losses = criterion({"pred_logits": logits, "pred_boxes": boxes}, targets, torch.empty(0, 5))
        expected = -0.1 * logits.log_softmax(-1)[..., -1].sum()
        torch.testing.assert_close(losses["loss_ce"], expected)
        self.assertEqual(float(losses["loss_bbox"].detach()), 0.0)
        self.assertEqual(float(losses["loss_giou"].detach()), 0.0)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad[..., -1] < 0).all().item())
        torch.testing.assert_close(boxes.grad, torch.zeros_like(boxes))

    def test_extreme_logits_are_finite_and_nans_are_rejected(self):
        matcher = HungarianMatcher(1, 5, 2)
        outputs = {
            "pred_logits": torch.tensor(
                [[[10000.0, -10000.0, 0.0, 0.0], [-10000.0, 10000.0, 0.0, 0.0]]]
            ),
            "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]]),
        }
        targets = [{"boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]])}]
        positive = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual(len(matcher(outputs, targets, positive)[0][0]), 1)
        criterion = SetCriterion(3, matcher, 0.1, ["labels", "boxes"], 0.07)
        losses = criterion(outputs, [dict(targets[0], labels=torch.tensor([0]))], positive)
        for name in ("loss_ce", "loss_bbox", "loss_giou"):
            self.assertTrue(torch.isfinite(losses[name]).item(), name)
        invalid = {**outputs, "pred_logits": outputs["pred_logits"].clone()}
        invalid["pred_logits"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            matcher(invalid, targets, positive)

    def test_mixed_precision_autocast_retains_finite_costs_losses_and_grads(self):
        d = device()
        matcher = HungarianMatcher(1, 5, 2)
        criterion = SetCriterion(3, matcher, 0.1, ["labels", "boxes"], 0.07)
        targets = [
            {
                "boxes": torch.tensor(
                    [[0.20001, 0.20, 0.0002, 0.0001], [0.79, 0.70, 0.12, 0.09]], device=d
                ),
                "labels": torch.tensor([0, 1], device=d),
            }
        ]
        positive = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=d)

        # Saturated softmax probabilities can legitimately produce zero FP16 CE gradients.
        logit_cases = (
            (False, [[[2.0, -2.0, 1.0, 0.0], [-2.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]]]),
            (True, [[[100.0, -100.0, 10.0, 0.0], [-50.0, 70.0, 5.0, 0.0], [0.0, 0.0, 0.0, 50.0]]]),
        )
        dtypes = (torch.float16,) if d.type == "cuda" else (torch.bfloat16, torch.float16)
        for dtype in dtypes:
            for saturated, values in logit_cases:
                with self.subTest(dtype=str(dtype), saturated=saturated):
                    logits = torch.tensor(values, device=d, requires_grad=True)
                    boxes = torch.tensor(
                        [
                            [
                                [0.20, 0.20, 0.0002, 0.0001],
                                [0.80, 0.70, 0.12, 0.09],
                                [0.50, 0.50, 0.15, 0.15],
                            ]
                        ],
                        device=d,
                        requires_grad=True,
                    )
                    predictions = {"pred_logits": logits.to(dtype), "pred_boxes": boxes.to(dtype)}
                    baseline = matcher(predictions, targets, positive)[0]
                    with torch.autocast(device_type=d.type, dtype=dtype):
                        match = matcher(predictions, targets, positive)[0]
                        losses = criterion(predictions, targets, positive)
                        objective = (
                            losses["loss_ce"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
                        )
                    self.assertEqual(match[0].tolist(), baseline[0].tolist())
                    self.assertEqual(match[1].tolist(), baseline[1].tolist())
                    self.assertTrue(torch.isfinite(objective).item())
                    objective.backward()
                    for tensor in (logits, boxes):
                        self.assertIsNotNone(tensor.grad)
                        self.assertTrue(torch.isfinite(tensor.grad).all().item())
                    self.assertGreater(float(boxes.grad.abs().sum()), 0.0)
                    if not saturated:
                        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
