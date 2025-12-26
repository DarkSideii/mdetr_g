# geo_datasets/dior_rsvg_eval.py
import numpy as np
import torch
from mdetr.datasets.coco_eval import CocoEvaluator, convert_to_xywh


def _to_py_int(x):
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
    b = pred.get("boxes", None)
    if b is None:
        return 0
    if torch.is_tensor(b):
        return int(b.shape[0])
    return len(b)


class DiorRSVGEvaluator(CocoEvaluator):
    """
    COCO evaluator with optional class-agnostic export.
    """

    def __init__(self, dataset, use_cats: bool = False, verbose: bool = False):
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        super().__init__(dataset.coco, iou_types=["bbox"], useCats=use_cats)
        self.verbose = verbose
        self.use_cats = use_cats

    def update(self, predictions):
        fixed = {_to_py_int(img_id): pred for img_id, pred in predictions.items()}
        return super().update(fixed)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for image_id, pred in predictions.items():
            if _num_instances(pred) == 0:
                continue

            image_id = _to_py_int(image_id)

            if torch.is_tensor(pred["boxes"]):
                boxes_xywh = convert_to_xywh(pred["boxes"]).cpu().tolist()
            else:
                boxes_xywh = convert_to_xywh(torch.as_tensor(pred["boxes"])).cpu().tolist()

            if torch.is_tensor(pred["scores"]):
                scores = pred["scores"].detach().cpu().tolist()
            else:
                scores = list(pred["scores"])

            if self.use_cats:
                if torch.is_tensor(pred.get("labels", None)):
                    labels = pred["labels"].detach().cpu().tolist()
                else:
                    labels = list(pred.get("labels", []))
                get_cat = lambda k: int(labels[k]) + 1
            else:
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
