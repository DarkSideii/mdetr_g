from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mdetr.util import box_ops


class PostProcessFlickr(nn.Module):
    """Processor for Flickr30k entities phrase-level recall@k evaluation."""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, positive_map, items_per_batch_element):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        batch_size = target_sizes.shape[0]
        prob = F.softmax(out_logits, -1)

        # boxes: cxcywh -> xyxy -> absolute pixel coords
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        cum_sum = np.cumsum(items_per_batch_element)
        curr_batch_index = 0

        # binarize the map if not already binary
        pos = positive_map > 1e-6

        predicted_boxes = [[] for _ in range(batch_size)]

        assert len(pos) == cum_sum[-1]
        if len(pos) == 0:
            return predicted_boxes

        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1

        for i in range(len(pos)):
            # score per query = max over positive tokens
            scores, _ = torch.max(pos[i].unsqueeze(0) * prob[curr_batch_index, :, :], dim=-1)
            _, indices = torch.sort(scores, descending=True)

            assert items_per_batch_element[curr_batch_index] > 0
            predicted_boxes[curr_batch_index].append(boxes[curr_batch_index][indices].to("cpu").tolist())
            if i == len(pos) - 1:
                break

            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(cum_sum)

        return predicted_boxes


class PostProcessPhrasecut(nn.Module):
    """Processor for Phrasecut evaluation."""

    def __init__(self, score_thresh=0.7):
        super().__init__()
        self.score_thresh = score_thresh

    @torch.no_grad()
    def forward(self, results):
        final_results = []
        for elem in results:
            keep = elem["scores"] > self.score_thresh
            boxes = elem["boxes"][keep].view(-1, 4)
            boxes[..., 2:] -= boxes[..., :2]  # xyxy -> xywh
            res = {"boxes": boxes.tolist()}
            if "masks" in elem:
                res["masks"] = elem["masks"][keep].any(0).squeeze(0).cpu().numpy()
            final_results.append(res)
        return final_results


class PostProcess(nn.Module):
    """Convert model outputs into COCO-style detection results."""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)

        # (class-agnostic) use objectness = 1 - P(background)
        scores = 1 - prob[:, :, -1]
        labels = torch.ones_like(scores, dtype=torch.long)  # force class 1

        # boxes: cxcywh -> xyxy -> absolute pixel coords
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        if "pred_isfinal" in outputs:
            is_final = outputs["pred_isfinal"].sigmoid()
            scores_refexp = scores * is_final.view_as(scores)
            for i in range(len(results)):
                results[i]["scores_refexp"] = scores_refexp[i]

        return results


def build_postprocessors(args, dataset_name) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}

    if dataset_name == "flickr":
        postprocessors["flickr_bbox"] = PostProcessFlickr()

    if dataset_name == "phrasecut":
        postprocessors["phrasecut"] = PostProcessPhrasecut()

    return postprocessors
