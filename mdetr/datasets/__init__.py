# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.
"""
Dataset factory for the mdetr.datasets package.
"""

import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    """Unwrap Subset layers to reach the underlying dataset (placeholder for COCO API access)."""
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset


def build_dataset(dataset_file: str, image_set: str, args):
    """Build the requested dataset split."""
    if dataset_file == "coco":
        return build_coco(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")
