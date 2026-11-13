# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Opt-in checks of an actual checkpoint and its local dataset/tokenizer."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .common import class_spans, device, offline_network


class TestActualCheckpointAndCachedCaptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("MDETR_INTEGRITY_CHECKPOINT")
        if not path:
            raise unittest.SkipTest("No --checkpoint: real-model/data check is opt-in")
        cls.enterClassContext(
            patch.dict(
                os.environ,
                {
                    "DOTA_GV2_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "AWS_EC2_METADATA_DISABLED": "true",
                },
            )
        )
        cls.enterClassContext(offline_network())

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not {"args", "model"} <= checkpoint.keys():
            raise ValueError("Expected a full MDETR-G checkpoint with 'args' and 'model'")
        args = deepcopy(checkpoint["args"])
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
        config_path = os.environ.get("MDETR_INTEGRITY_DATASET_CONFIG")
        if config_path:
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Dataset config must be a JSON object")
            for key, value in config.items():
                setattr(args, key, value)
            args.dataset_config = config_path
        args.device = str(device())
        cls.args = args

        from mdetr.models import build_model
        from geo_datasets import build_dataset

        # Build production code; pretrained/tokenizer files must already be local.
        cls.model, cls.criterion, _, _ = build_model(args)
        cls.model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        cls.model.to(device()).eval()
        cls.criterion.eval()
        args.tokenizer = cls.model.tokenizer
        cls.dataset = build_dataset("dota", os.environ.get("MDETR_INTEGRITY_SPLIT", "val"), args)

        count = int(os.environ.get("MDETR_INTEGRITY_SAMPLES", "2"))
        cls.sample_indices = random.Random(42).sample(
            range(len(cls.dataset)), min(count, len(cls.dataset))
        )
        if not cls.sample_indices:
            raise AssertionError("Dataset has no samples")

    def test_real_caption_target_token_map_and_prediction_roundtrip(self):
        from mdetr.util.misc import collate_fn
        from mdetr.util import box_ops

        checked = objects = oracle_hits = 0
        aliases = getattr(self.args, "class_caption_aliases", None) or {}
        deformable = getattr(self.model, "is_deformable", False)
        for sample_index in self.sample_indices:
            image, target = self.dataset[sample_index]
            n = len(target["boxes"])
            self.assertEqual(target["boxes"].shape, (n, 4))
            self.assertEqual(len(target["labels"]), n)
            self.assertEqual(len(target["tokens_positive"]), n)
            self.assertEqual(target["positive_map"].shape[0], n)
            self.assertTrue(torch.isfinite(target["boxes"]).all().item())
            self.assertTrue((target["boxes"][:, 2:] > 0).all().item())
            self.assertTrue(((target["boxes"] >= 0) & (target["boxes"] <= 1)).all().item())
            torch.testing.assert_close(
                target["positive_map"].sum(-1), torch.ones(n), atol=1e-4, rtol=0
            )

            names = target.get("phrases_searched")
            if names is not None:
                self.assertEqual(len(names), n)
                for name, spans in zip(names, target["tokens_positive"]):
                    self.assertTrue(name, "Target class name is missing")
                    recognized = class_spans(target["caption"], name, aliases.get(name, ()))
                    for begin, end in spans:
                        self.assertIn(
                            (begin, end),
                            recognized,
                            f"Class {name!r} has a different class's phrase in {target['caption']!r}",
                        )

            batch = collate_fn(False, [(image, target)])
            sample = batch["samples"].to(device())
            captions = [target["caption"]]
            references = {}
            handles = []
            if deformable:
                for index, layer in enumerate(self.model.transformer.decoder.layers):

                    def capture(_module, inputs, layer_index=index):
                        references[layer_index] = inputs[1].detach().clone()

                    handles.append(layer.cross_attn_img.register_forward_pre_hook(capture))
            try:
                with torch.no_grad():
                    cache = self.model(sample, captions, encode_and_save=True)
                    tokenized = cache["tokenized"]
                    expected = torch.zeros_like(target["positive_map"], dtype=torch.bool)
                    for row_index, spans in enumerate(target["tokens_positive"]):
                        for start, stop in spans:
                            self.assertTrue(0 <= start < stop <= len(target["caption"]))
                            first = tokenized.char_to_token(0, start)
                            last = tokenized.char_to_token(0, stop - 1)
                            self.assertIsNotNone(first, "Caption span start is not tokenizable")
                            self.assertIsNotNone(last, "Caption span end is not tokenizable")
                            self.assertLess(last, expected.shape[1] - 1)
                            expected[row_index, first : last + 1] = True
                    self.assertTrue(
                        torch.equal(expected, target["positive_map"] > 0),
                        "Dataset token rows disagree with the model's actual tokenizer",
                    )
                    padding = ~tokenized.attention_mask[0].detach().cpu().bool()
                    self.assertFalse((expected[:, : len(padding)] & padding).any().item())

                    output = self.model(sample, captions, encode_and_save=False, memory_cache=cache)
                    gt_device = {
                        key: value.to(device()) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    losses = self.criterion(
                        output, [gt_device], target["positive_map"].to(device())
                    )
            finally:
                for handle in handles:
                    handle.remove()

            for name, loss in losses.items():
                self.assertTrue(torch.isfinite(loss).all().item(), name)
            for key in ("pred_boxes", "pred_logits"):
                self.assertTrue(torch.isfinite(output[key]).all().item(), key)
            boxes = output["pred_boxes"]
            self.assertTrue(((boxes >= 0) & (boxes <= 1)).all().item())
            if deformable:
                self.assertEqual(len(references), len(self.model.transformer.decoder.layers))
                for index, prior in enumerate(output.get("aux_outputs", [])):
                    centers = prior["pred_boxes"][..., :2].clamp(1e-5, 1 - 1e-5)
                    expected_refs = centers[:, :, None] * cache["valid_ratios"][:, None]
                    torch.testing.assert_close(
                        references[index + 1], expected_refs, atol=1e-4, rtol=1e-4
                    )

            if n:
                gt = box_ops.box_cxcywh_to_xyxy(target["boxes"].to(device()))
                pred = box_ops.box_cxcywh_to_xyxy(boxes[0])
                iou = box_ops.box_iou(pred, gt)[0]
                oracle_hits += int((iou.max(0).values >= 0.50).sum().item())
            objects += n
            checked += 1

        self.assertGreater(checked, 0)
        # Geometry is informational; integrity tests do not assert model accuracy.
        print(
            f"\nREAL CHECKPOINT: {checked} local image-caption samples, {objects} objects, "
            f"best-of-all-queries IoU@.50 {oracle_hits}/{objects}; outbound network blocked"
        )
