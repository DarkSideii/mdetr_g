# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Class-agnostic COCO evaluation against perfect predictions."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import torch


class TestClassAgnosticCoco(unittest.TestCase):
    def test_perfect_predictions_score_one_with_arbitrary_category_ids(self):
        # Neither declared category is 1: hardcoded placeholders must fail.
        from pycocotools.coco import COCO
        from geo_datasets.dota_eval import DotaEvaluator

        coco = COCO()
        coco.dataset = {
            "info": {},
            "licenses": [],
            "images": [{"id": 19, "width": 100, "height": 100}],
            "categories": [{"id": 9, "name": "arbitrary-a"}, {"id": 47, "name": "arbitrary-b"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 19,
                    "category_id": 9,
                    "bbox": [5.0, 5.0, 20.0, 20.0],
                    "area": 400.0,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 19,
                    "category_id": 47,
                    "bbox": [60.0, 60.0, 25.0, 20.0],
                    "area": 500.0,
                    "iscrowd": 0,
                },
            ],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
            evaluator = DotaEvaluator(SimpleNamespace(coco=coco), use_cats=False)
            evaluator.update(
                {
                    19: {
                        "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0], [60.0, 60.0, 85.0, 80.0]]),
                        "scores": torch.tensor([0.99, 0.98]),
                        "labels": torch.tensor([0, 1]),
                    }
                }
            )
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
        ap, ap50 = evaluator.coco_eval["bbox"].stats[:2]
        self.assertAlmostEqual(float(ap), 1.0, places=5)
        self.assertAlmostEqual(float(ap50), 1.0, places=5)


# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Class-agnostic COCO export with arbitrary declared category IDs."""
