#!/usr/bin/env python3
# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Offline, class-agnostic regression tests for the MDETR-G code in --repo."""

import argparse
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


DEFAULT_CASES = {
    "aliases": {
        "inspection-rover": ["rover"],
        "harbor-crane": ["cargo-crane"],
    },
    "images": [
        {
            "caption": "Two harbor-cranes stand near the water while inspection-rovers move nearby.",
            "objects": [
                {"class": "harbor-crane", "box": [0.24, 0.30, 0.12, 0.16]},
                {"class": "harbor-crane", "box": [0.73, 0.64, 0.10, 0.13]},
                {"class": "inspection-rover", "box": [0.51, 0.48, 0.08, 0.07]},
            ],
        },
        {
            "caption": "A cargo-crane is beside the dock, and two rovers are crossing the road.",
            "objects": [
                {"class": "harbor-crane", "box": [0.70, 0.22, 0.15, 0.12]},
                {"class": "inspection-rover", "box": [0.33, 0.72, 0.06, 0.08]},
                {"class": "inspection-rover", "box": [0.55, 0.75, 0.07, 0.06]},
            ],
        },
    ],
}


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=root.parent,
        help="MDETR-G checkout containing mdetr/ and geo_datasets/",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cases", type=Path, help="Optional custom JSON; otherwise use built-in generic test cases"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional full checkpoint for a STRICTLY OFFLINE real-model check",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="Optional dataset JSON override for --checkpoint checks only",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of actual cached image-caption samples to check with --checkpoint",
    )
    parser.add_argument(
        "--force-reference-attn",
        action="store_true",
        help="Use a differentiable PyTorch attention reference in synthetic tests",
    )
    parser.add_argument(
        "--require-native-msda",
        action="store_true",
        help="Fail rather than skip if the native MMCV CUDA kernel cannot be tested",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / "mdetr" / "models" / "transformer.py").is_file():
        parser.error(f"Not an MDETR-G repository: {repo}")
    cases = None
    if args.cases is not None:
        cases = args.cases.expanduser().resolve()
        if not cases.is_file():
            parser.error(f"Caption cases not found: {cases}")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser()
        checkpoint = (
            (repo / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        )
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
    dataset_config = None
    if args.dataset_config is not None:
        if checkpoint is None:
            parser.error("--dataset-config requires --checkpoint")
        dataset_config = args.dataset_config.expanduser()
        if not dataset_config.is_absolute():
            dataset_config = repo / dataset_config
        dataset_config = dataset_config.resolve()
        if not dataset_config.is_file():
            parser.error(f"Dataset config not found: {dataset_config}")

    environment = {
        "DOTA_GV2_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "AWS_EC2_METADATA_DISABLED": "true",
        "MDETR_INTEGRITY_REPO": str(repo),
        "MDETR_INTEGRITY_CASES": str(cases) if cases is not None else "",
        "MDETR_INTEGRITY_CASE_DATA": json.dumps(DEFAULT_CASES) if cases is None else "",
        "MDETR_INTEGRITY_CHECKPOINT": str(checkpoint) if checkpoint is not None else "",
        "MDETR_INTEGRITY_DATASET_CONFIG": str(dataset_config) if dataset_config is not None else "",
        "MDETR_INTEGRITY_DEVICE": args.device,
        "MDETR_INTEGRITY_SPLIT": args.split,
        "MDETR_INTEGRITY_SAMPLES": str(args.samples),
        "MDETR_INTEGRITY_FORCE_REFERENCE": str(int(args.force_reference_attn)),
        "MDETR_INTEGRITY_REQUIRE_NATIVE": str(int(args.require_native_msda)),
    }
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    with patch.dict(os.environ, environment):
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(root), str(repo)]
        try:
            try:
                import torch
            except ImportError as exc:
                parser.error(f"The repository's Python environment must have PyTorch: {exc}")
            if args.device == "cuda" and not torch.cuda.is_available():
                parser.error("CUDA requested but unavailable")
            effective_device = (
                "cuda"
                if args.device == "auto" and torch.cuda.is_available()
                else ("cpu" if args.device == "auto" else args.device)
            )
            os.environ["MDETR_INTEGRITY_DEVICE"] = effective_device

            from suite.common import offline_network

            print(
                f"MDETR-G integrity suite | repo={repo} | device={effective_device} | offline=1",
                flush=True,
            )
            if args.cases is None:
                print("Using built-in generic caption cases (no JSON file required).", flush=True)
            print(
                "Synthetic tests use production matcher/loss/encoder/decoder with tiny fixtures. "
                "Native MMCV/CUDA checks are separate from the labeled CPU reference.",
                flush=True,
            )
            if checkpoint is None:
                print(
                    "Real checkpoint + local dataset checks: SKIPPED (pass --checkpoint to enable).",
                    flush=True,
                )

            with offline_network():
                suite = unittest.defaultTestLoader.discover(
                    str(root / "suite"), "test_*.py", top_level_dir=str(root)
                )
                outcome = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if outcome.wasSuccessful() else 1
        finally:
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode


if __name__ == "__main__":
    raise SystemExit(main())
