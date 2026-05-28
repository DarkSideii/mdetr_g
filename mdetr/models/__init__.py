# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.

"""Model factory wrapper."""

from .mdetr import build as _build_mdetr

def build_model(args):
    """Build the MDETR model/criteria tuple using the canonical builder."""
    return _build_mdetr(args)
