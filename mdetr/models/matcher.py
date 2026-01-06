"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from mdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Compute a 1-to-1 assignment between predictions and targets using the Hungarian algorithm.

    Targets do not include the "no_object" class; unmatched queries are treated as background.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Initialize matching cost weights."""
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Softmax(-1)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, positive_map):
        """Compute the optimal bipartite matching for the batch."""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = self.norm(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        assert len(tgt_bbox) == len(positive_map)

        cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        assert cost_class.shape == cost_bbox.shape

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


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
