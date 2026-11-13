# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Actual training-loop regression for unequal object counts and partial windows."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import torch
from torch import nn


class _ScalarDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.decoded = 0

    def forward(self, samples, captions, encode_and_save=True, memory_cache=None):
        if encode_and_save:
            return {"synthetic": True}
        self.decoded += 1
        return {"toy_output": self.weight}


class _CountNormalizedCriterion(nn.Module):
    """Same per-microbatch /max(num_boxes,1) contract as SetCriterion."""

    def forward(self, outputs, targets, positive_map):
        n = sum(len(t["labels"]) for t in targets)
        numerator = sum(
            len(t["labels"]) * t["gradient_per_object"] + t.get("background_gradient", 0.0)
            for t in targets
        )
        return {"loss_bbox": outputs["toy_output"] * numerator / max(n, 1)}


def _batches(specs):
    return [
        {
            "samples": torch.zeros((1, 3, 4, 4)),
            "targets": [
                {
                    "caption": "arbitrary",
                    "labels": torch.zeros(n, dtype=torch.long),
                    "gradient_per_object": float(slope),
                }
            ],
        }
        for n, slope in specs
    ]


class TestGradientAccumulationContract(unittest.TestCase):
    def _train(self, specs, steps, batches=None):
        from mdetr.engine import train_one_epoch

        model = _ScalarDetector()
        optimizer = torch.optim.SGD(
            [{"name": "base", "params": [model.weight], "lr": 1.0, "base_lr": 1.0}], lr=1.0
        )
        args = SimpleNamespace(
            epochs=1, grad_accum_steps=steps, no_detection=False, schedule="step", lr_drop=35
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(
                model,
                _CountNormalizedCriterion(),
                None,
                _batches(specs) if batches is None else batches,
                {"loss_bbox": 1.0},
                optimizer,
                torch.device("cpu"),
                epoch=0,
                args=args,
                max_norm=0.0,
            )
        return float(model.weight.detach()), model.decoded

    def test_unequal_object_counts_equal_one_large_batch_gradient(self):
        specs = [(1, 1.0), (9, 3.0)]
        observed, decoded = self._train(specs, steps=2)
        # Correct: (1*1 + 9*3)/10 = 2.8. Old equal-microbatch averaging=2.0.
        self.assertAlmostEqual(observed, -2.8, places=5)
        self.assertEqual(decoded, 2)

    def test_partial_final_accumulation_window_preserves_denominator(self):
        specs = [(1, 1.0), (9, 3.0), (4, -2.0)]
        observed, decoded = self._train(specs, steps=2)
        self.assertAlmostEqual(observed, -2.8 + 2.0, places=5)
        self.assertEqual(decoded, 3)

    def test_accumulation_matches_physical_batches_with_empty_images(self):
        specs = [(2, 1.0), (0, 0.0), (7, 3.0), (1, -2.0), (0, 0.0)]
        microbatches = _batches(specs)
        for batch in microbatches:
            batch["targets"][0]["background_gradient"] = 0.3
        physical = [
            {
                "samples": torch.cat(
                    [batch["samples"] for batch in microbatches[start : start + 3]]
                ),
                "targets": [
                    target
                    for batch in microbatches[start : start + 3]
                    for target in batch["targets"]
                ],
            }
            for start in range(0, len(microbatches), 3)
        ]
        accumulated, decoded = self._train(specs, 3, batches=microbatches)
        expected, physical_decoded = self._train(specs, 1, batches=physical)
        self.assertAlmostEqual(accumulated, expected, places=6)
        self.assertEqual(decoded, 5)
        self.assertEqual(physical_decoded, 2)

    def test_all_empty_window_keeps_background_gradients(self):
        batches = _batches([(0, 0.0)] * 3)
        for batch in batches:
            batch["targets"][0]["background_gradient"] = 0.3
        observed, decoded = self._train([], 3, batches=batches)
        self.assertAlmostEqual(observed, -0.9, places=6)
        self.assertEqual(decoded, 3)
