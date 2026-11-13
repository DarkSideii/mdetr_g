# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Production model forward, loss, backward, and optimizer integration checks."""

import unittest

import torch

from .common import (
    device,
    load_cases,
    make_tiny_model,
    production_maps_and_targets,
    synthetic_samples,
    weighted_loss,
)


class TestEndToEndModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(219)
        self.payload = load_cases()
        self.model, self.tokenizer = make_tiny_model(self.payload, device())
        _, self.targets_cpu, self.positive_map_cpu = production_maps_and_targets(
            self.payload, tokenizer=self.tokenizer
        )
        self.inputs = synthetic_samples(device(), count=len(self.targets_cpu))
        from mdetr.models.matcher import HungarianMatcher
        from mdetr.models.mdetr import SetCriterion

        self.matcher = HungarianMatcher(1, 5, 2)
        self.criterion = SetCriterion(
            255,
            self.matcher,
            eos_coef=0.1,
            losses=["labels", "boxes", "cardinality", "contrastive_align"],
            temperature=0.07,
            align_scale_mode="learnable",
        ).to(device())

    def _forward(self):
        captions = [target["caption"] for target in self.targets_cpu]
        cache = self.model(self.inputs, captions, encode_and_save=True)
        result = self.model(self.inputs, captions, encode_and_save=False, memory_cache=cache)
        return cache, result

    def test_full_forward_decode_alignment_postprocess(self):
        from mdetr.models.postprocessors import PostProcess
        from mdetr.util.box_ops import box_cxcywh_to_xyxy

        _, result = self._forward()
        batch = len(self.targets_cpu)
        self.assertEqual(tuple(result["pred_boxes"].shape), (batch, self.model.num_queries, 4))
        self.assertEqual(tuple(result["pred_logits"].shape), (batch, self.model.num_queries, 256))
        self.assertEqual(len(result["aux_outputs"]), 1)
        self.assertEqual(tuple(result["proj_tokens"].shape[:2]), (batch, 256))
        self.assertTrue(torch.isfinite(result["pred_logits"]).all().item())
        self.assertTrue(torch.isfinite(result["pred_boxes"]).all().item())

        size = torch.tensor([[32, 48]] * batch, device=device())
        processed = PostProcess()(result, size)
        self.assertEqual(len(processed), batch)
        expected = (
            box_cxcywh_to_xyxy(result["pred_boxes"])
            * torch.tensor([48.0, 32.0, 48.0, 32.0], device=device())[None, None]
        )
        for index, entry in enumerate(processed):
            torch.testing.assert_close(entry["boxes"], expected[index], atol=1e-5, rtol=0)
            expected_scores = 1 - result["pred_logits"][index].softmax(-1)[:, -1]
            torch.testing.assert_close(entry["scores"], expected_scores, atol=1e-5, rtol=0)
            self.assertTrue(
                torch.all(entry["labels"] == 1),
                "The existing evaluator is intentionally class-agnostic",
            )

    def test_full_loss_backward_reaches_encoder_decoder_and_query_features(self):
        self.model.zero_grad(set_to_none=True)
        _cache, result = self._forward()
        targets = [
            {key: val.to(device()) if torch.is_tensor(val) else val for key, val in target.items()}
            for target in self.targets_cpu
        ]
        positive = self.positive_map_cpu.to(device())
        losses = self.criterion(result, targets, positive)
        expected = {
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "loss_contrastive_align",
            "loss_ce_0",
            "loss_bbox_0",
            "loss_giou_0",
            "loss_contrastive_align_0",
        }
        self.assertTrue(expected.issubset(losses), f"Missing losses: {expected - set(losses)}")
        for name in expected:
            self.assertTrue(torch.isfinite(losses[name]).item(), f"Nonfinite {name}")
        weights = {
            key: (
                5.0 if key.startswith("loss_bbox") else 2.0 if key.startswith("loss_giou") else 1.0
            )
            for key in expected
        }
        loss = weighted_loss(losses, weights)
        self.assertTrue(loss.requires_grad)
        loss.backward()

        required_gradients = {
            "learned query identities": self.model.query_embed.weight.grad,
            "bbox regression head": self.model.bbox_embed.layers[-1].weight.grad,
            "decoder reference head": self.model.transformer.decoder.ref_point_head.weight.grad,
            "encoder self-attention": self.model.transformer.encoder.layers[
                0
            ].self_attn.mod.value_proj.weight.grad,
            "decoder image attention": self.model.transformer.decoder.layers[
                0
            ].cross_attn_img.mod.value_proj.weight.grad,
            "decoder text attention": self.model.transformer.decoder.layers[
                0
            ].cross_attn_txt.in_proj_weight.grad,
            "text feature adapter": self.model.transformer.resizer.fc.weight.grad,
            "classification head": self.model.class_embed.weight.grad,
            "visual backbone": self.model.backbone.stages[0].weight.grad,
        }
        for name, gradient in required_gradients.items():
            self.assertIsNotNone(gradient, f"No gradient reached {name}")
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, f"Zero gradient in {name}")
        for parameter in self.model.transformer.text_encoder.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

        indices = self.matcher(result, targets, positive)
        first_image_matches = indices[0][0]
        self.assertEqual(len(first_image_matches), len(self.targets_cpu[0]["boxes"]))
        gradients = self.model.query_embed.weight.grad[first_image_matches]
        if len(first_image_matches) >= 2:
            self.assertFalse(
                torch.allclose(gradients[0], gradients[1], atol=1e-7, rtol=1e-5),
                "Distinct matched queries received identical gradients",
            )

    def test_one_caption_class_can_ground_multiple_separate_objects(self):
        # One phrase can supervise multiple distinct object boxes.
        from mdetr.models.matcher import HungarianMatcher
        from mdetr.models.mdetr import SetCriterion

        payload = {
            "images": [
                {
                    "caption": "Several beacons are placed on the platform.",
                    "objects": [
                        {"class": "beacon", "box": [0.2, 0.3, 0.1, 0.12]},
                        {"class": "beacon", "box": [0.8, 0.7, 0.12, 0.1]},
                    ],
                }
            ]
        }
        model, tokenizer = make_tiny_model(payload, device())
        _, targets, maps = production_maps_and_targets(payload, tokenizer=tokenizer)
        self.assertTrue(torch.equal((maps[0] > 0), (maps[1] > 0)))
        samples = synthetic_samples(device(), count=1)
        memory = model(samples, [targets[0]["caption"]], encode_and_save=True)
        output = model(samples, [targets[0]["caption"]], encode_and_save=False, memory_cache=memory)
        matcher = HungarianMatcher(1, 5, 2)
        criterion = SetCriterion(255, matcher, 0.1, ["labels", "boxes", "contrastive_align"], 0.07)
        gt_device = [
            {
                key: value.to(device()) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        losses = criterion(output, gt_device, maps.to(device()))
        matched_queries, matched_targets = matcher(output, gt_device, maps.to(device()))[0]
        self.assertEqual(len(set(matched_queries.tolist())), 2)
        self.assertEqual(sorted(matched_targets.tolist()), [0, 1])
        (losses["loss_ce"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]).backward()
        self.assertIsNotNone(model.transformer.decoder.ref_point_head.weight.grad)
        self.assertGreater(
            float(model.transformer.decoder.ref_point_head.weight.grad.abs().sum()), 0.0
        )

    def test_two_step_training_loss_stays_finite(self):
        # Require a real parameter update, not an arbitrary two-step loss drop.
        self.model.train()
        old_queries = self.model.query_embed.weight.detach().clone()
        old_boxes = self.model.bbox_embed.layers[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-4
        )
        targets = [
            {key: val.to(device()) if torch.is_tensor(val) else val for key, val in target.items()}
            for target in self.targets_cpu
        ]
        positive = self.positive_map_cpu.to(device())
        weights = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_contrastive_align": 1.0,
            "loss_ce_0": 1.0,
            "loss_bbox_0": 5.0,
            "loss_giou_0": 2.0,
            "loss_contrastive_align_0": 1.0,
        }
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, output = self._forward()
            losses = self.criterion(output, targets, positive)
            objective = weighted_loss(losses, weights)
            self.assertTrue(torch.isfinite(objective).item())
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            optimizer.step()
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all().item(), f"Non-finite parameter {name}")
        self.assertFalse(torch.equal(self.model.query_embed.weight, old_queries))
        self.assertFalse(torch.equal(self.model.bbox_embed.layers[-1].weight, old_boxes))

    def test_extreme_learned_scales_keep_model_gradients_finite(self):
        targets = [
            {
                key: value.to(device()) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in self.targets_cpu
        ]
        for log_scale in (1000.0, -1000.0):
            with self.subTest(log_scale=log_scale):
                self.model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    self.model.logit_scale_cls.fill_(log_scale)
                    self.model.logit_scale_align.fill_(log_scale)
                _, output = self._forward()
                losses = self.criterion(output, targets, self.positive_map_cpu.to(device()))
                objective = losses["loss_ce"] + losses["loss_contrastive_align"]
                self.assertTrue(torch.isfinite(objective).item())
                objective.backward()
                for scale in (self.model.logit_scale_cls, self.model.logit_scale_align):
                    self.assertIsNotNone(scale.grad)
                    self.assertTrue(torch.isfinite(scale.grad).all().item())
