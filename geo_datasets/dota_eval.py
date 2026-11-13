# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""COCO bounding-box evaluation for class-aware or class-agnostic DOTA outputs."""

import numpy as np
import torch
from mdetr.datasets.coco_eval import CocoEvaluator, convert_to_xywh


def _to_py_int(x):
    """Convert an image id (tensor/np/int/array) to a Python int."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return int(x.view(-1)[0].item())
    try:
        return int(x)
    except Exception:
        return int(np.asarray(x).item())


def _num_instances(pred):
    """Return number of detections in a prediction dict."""
    boxes = pred.get("boxes")
    return 0 if boxes is None else len(boxes)


class DotaEvaluator(CocoEvaluator):
    """Evaluate boxes with optional class-agnostic matching."""

    def __init__(self, dataset, use_cats: bool = False, verbose: bool = False):
        # Unwrap Subset to reach the dataset with `.coco`.
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        super().__init__(dataset.coco, iou_types=["bbox"], useCats=use_cats)
        self.verbose = verbose
        self.use_cats = use_cats
        # COCO pools catIds even with useCats=False; export under a declared ID.
        category_ids = sorted(int(category_id) for category_id in self.coco_gt.getCatIds())
        self._agnostic_category_id = category_ids[0] if category_ids else None
        if not self.use_cats and self._agnostic_category_id is None and self.coco_gt.anns:
            raise ValueError("DOTA COCO ground truth has annotations but no declared categories")

    def update(self, predictions):
        # pycocotools expects plain int keys.
        fixed = {_to_py_int(img_id): pred for img_id, pred in predictions.items()}
        return super().update(fixed)

    def prepare_for_coco_detection(self, predictions):
        """Export pixel xyxy predictions to COCO xywh records."""
        coco_results = []
        for image_id, pred in predictions.items():
            if _num_instances(pred) == 0:
                continue

            image_id = _to_py_int(image_id)

            boxes = torch.as_tensor(pred["boxes"]).detach()
            boxes_xywh = convert_to_xywh(boxes).cpu().tolist()

            if torch.is_tensor(pred["scores"]):
                scores = pred["scores"].detach().cpu().tolist()
            else:
                scores = list(pred["scores"])

            if self.use_cats:
                if torch.is_tensor(pred.get("labels", None)):
                    labels = pred["labels"].detach().cpu().tolist()
                else:
                    labels = list(pred.get("labels", []))
            else:
                if self._agnostic_category_id is None:
                    raise ValueError(
                        "Cannot export detections: DOTA COCO ground truth declares no categories"
                    )

            coco_results.extend(
                {
                    "image_id": image_id,
                    "category_id": (
                        int(labels[k]) + 1 if self.use_cats else self._agnostic_category_id
                    ),
                    "bbox": box,
                    "score": float(scores[k]),
                }
                for k, box in enumerate(boxes_xywh)
            )

        return coco_results
