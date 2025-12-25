import os
import subprocess
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from multimodal_framework.mdetr.util.dist import is_main_process as _is_main_process


def get_sha() -> str:
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff_idx = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff_idx else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass

    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(do_round, batch):
    batch = list(zip(*batch))
    final_batch: Dict[str, Any] = {}

    final_batch["samples"] = NestedTensor.from_tensor_list(batch[0], do_round)
    final_batch["targets"] = batch[1]

    # Positive map (train)
    if "positive_map" in batch[1][0]:
        # Since elements have different #boxes, collapse the batch-dim (no padding).
        max_len = max(v["positive_map"].shape[1] for v in batch[1])
        nb_boxes = sum(v["positive_map"].shape[0] for v in batch[1])

        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)
        assert cur_count == len(batched_pos_map)

        final_batch["positive_map"] = batched_pos_map.float()

    # Positive map (eval)
    if "positive_map_eval" in batch[1][0]:
        max_len = max(v["positive_map_eval"].shape[1] for v in batch[1])
        nb_boxes = sum(v["positive_map_eval"].shape[0] for v in batch[1])

        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map_eval"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)
        assert cur_count == len(batched_pos_map)

        final_batch["positive_map_eval"] = batched_pos_map.float()

    # Answers (for QA-style datasets)
    if "answer" in batch[1][0] or "answer_type" in batch[1][0]:
        answers: Dict[str, Tensor] = {}
        for f in batch[1][0].keys():
            if "answer" not in f:
                continue
            answers[f] = torch.stack([b[f] for b in batch[1]])
        final_batch["answers"] = answers

    return final_batch


class NestedTensor:
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list: List[Tensor], do_round: bool = False):
        # Only 3D images supported: [C,H,W]
        if tensor_list[0].ndim != 3:
            raise ValueError("not supported")

        # Build batch tensor & mask
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = (len(tensor_list),) + max_size
        b, c, h, w = batch_shape

        if do_round:
            # Round to multiple-of-128 to avoid FPN rounding issues
            p = 128
            h = h if h % p == 0 else (h // p + 1) * p
            w = w if w % p == 0 else (w // p + 1) * p
            batch_shape = (b, c, h, w)

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False

        return cls(tensor, mask)

    def __repr__(self) -> str:
        return repr(self.tensors)


def interpolate(
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    assert input.shape[0] != 0 or input.shape[1] != 0, "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension,
        # so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)


# def targets_to(targets: List[Dict[str, Any]], device):
#     """Moves the target dicts to the given device."""
#     excluded_keys = [
#         "questionId", "tokens_positive", "tokens", "dataset_name", "sentence_id",
#         "original_img_id", "nb_eval", "task_id", "original_id",
#     ]
#     return [
#         {k: v.to(device) if k not in excluded_keys else v for k, v in t.items() if k != "caption"}
#         for t in targets
#     ]


def targets_to(targets: List[Dict[str, Any]], device) -> List[Dict[str, Any]]:
    """Moves all tensors in a list of target dicts to the given device."""
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]


def is_main_process() -> bool:
    """
    True if this is the main process.
    In single-GPU (no torch.distributed) this is always True.
    """
    return _is_main_process()
