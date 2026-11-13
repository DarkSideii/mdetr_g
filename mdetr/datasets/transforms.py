# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.

"""Image/target transforms; pixel xyxy boxes become normalized cxcywh in Normalize."""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from mdetr.util.box_ops import box_xyxy_to_cxcywh
from mdetr.util.misc import interpolate


# Box-aligned fields only; evaluation phrases may each refer to multiple boxes.
_OBJECT_FIELDS = (
    "boxes",
    "labels",
    "area",
    "iscrowd",
    "masks",
    "keypoints",
    "isfinal",
    "positive_map",
    "tokens_positive",
    "tokens",
    "phrases_searched",
)


def _filter_object_fields(target, keep, fields=_OBJECT_FIELDS):
    """Filter box-aligned tensors and span lists together."""
    keep_list = None
    for field in fields:
        if field not in target:
            continue
        values = target[field]
        if len(values) != len(keep):
            raise ValueError(f"{field}: {len(values)} rows for {len(keep)} objects")
        if torch.is_tensor(values):
            target[field] = values[keep]
        else:
            if keep_list is None:
                keep_list = keep.tolist()
            selected = [value for value, retain in zip(values, keep_list) if retain]
            target[field] = tuple(selected) if isinstance(values, tuple) else selected


def hflip(image, target):
    """Flip image geometry only; preserve captions and all token-span metadata."""
    flipped_image = F.hflip(image)

    w = image.size[0]

    if target is None:
        return flipped_image, None
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor(
            [w, 0, w, 0]
        )
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    """Resize image/target; `size` may be a scalar min_size or a (w,h) tuple."""

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size, interpolation=InterpolationMode.BICUBIC)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    """Pad on bottom/right by (pad_x, pad_y)."""
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(target["masks"], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    """Randomly choose a resize scale from a list."""

    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad:
    """Random pad up to max_pad on each axis."""

    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect:
    """Randomly choose between two transforms with probability p for transforms1."""

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    """Convert PIL image to a torch Tensor."""

    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing:
    """Apply torchvision RandomErasing to the image only."""

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize:
    """Normalize image tensor and convert boxes to normalized cxcywh if present."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class RemoveDifficult:
    """Optionally drop crowd/difficult instances based on target["iscrowd"]."""

    def __init__(self, enabled=False):
        self.remove_difficult = enabled

    def __call__(self, image, target=None):
        if target is None:
            return image, None
        target = target.copy()
        keep = ~target["iscrowd"].to(torch.bool) | (not self.remove_difficult)
        _filter_object_fields(target, keep)
        return image, target


class Compose:
    """Compose a sequence of (image, target) transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        transforms = "".join(f"\n    {transform}" for transform in self.transforms)
        return f"{self.__class__.__name__}({transforms}\n)"
