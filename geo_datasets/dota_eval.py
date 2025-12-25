# geo_datasets/dota_eval.py
import numpy as np
import torch
from multimodal_framework.mdetr.datasets.coco_eval import CocoEvaluator, convert_to_xywh


def _to_py_int(x):
    """Safely convert image ids (tensor/np/int) to plain Python int."""
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
    b = pred.get("boxes", None)
    if b is None:
        return 0
    if torch.is_tensor(b):
        return int(b.shape[0])
    return len(b)


class DotaEvaluator(CocoEvaluator):
    """
    COCO evaluator with a class-agnostic export option.

    If use_cats=False (default here), every detection is exported with category_id=1.
    This avoids pycocotools silently dropping predictions when labels are token indices
    (as in MDETR’s token-based classification).
    """

    def __init__(self, dataset, use_cats: bool = False, verbose: bool = False):
        # unwrap Subset → reach the underlying dataset with .coco
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        super().__init__(dataset.coco, iou_types=["bbox"], useCats=use_cats)
        self.verbose = verbose
        self.use_cats = use_cats  # keep a local flag for export logic

    def update(self, predictions):
        # Ensure pure Python int keys for pycocotools
        fixed = {_to_py_int(img_id): pred for img_id, pred in predictions.items()}
        return super().update(fixed)

    def prepare_for_coco_detection(self, predictions):
        """
        Convert model outputs to COCO json format.

        When self.use_cats is False, we ignore per-detection 'labels' and force
        category_id=1 to ensure all detections are evaluated (class-agnostic AP).
        """
        coco_results = []
        for image_id, pred in predictions.items():
            if _num_instances(pred) == 0:
                continue

            image_id = _to_py_int(image_id)

            # boxes come from postprocessor as xyxy tensors → convert to xywh
            if torch.is_tensor(pred["boxes"]):
                boxes_xywh = convert_to_xywh(pred["boxes"]).cpu().tolist()
            else:
                boxes_xywh = convert_to_xywh(torch.as_tensor(pred["boxes"])).cpu().tolist()

            # ensure CPU lists
            if torch.is_tensor(pred["scores"]):
                scores = pred["scores"].detach().cpu().tolist()
            else:
                scores = list(pred["scores"])

            if self.use_cats:
                # Use provided labels (assumed 0-based in pipeline) and map to COCO 1-based ids
                if torch.is_tensor(pred.get("labels", None)):
                    labels = pred["labels"].detach().cpu().tolist()
                else:
                    labels = list(pred.get("labels", []))
                get_cat = lambda k: int(labels[k]) + 1
            else:
                # Class-agnostic: force a single category id (1) for all detections
                get_cat = lambda k: 1

            coco_results.extend(
                {
                    "image_id": image_id,
                    "category_id": get_cat(k),
                    "bbox": box,
                    "score": float(scores[k]),
                }
                for k, box in enumerate(boxes_xywh)
            )

        return coco_results
