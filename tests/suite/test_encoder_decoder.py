# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Encoder geometry and supervised decoder-reference regression checks."""

import unittest

import torch
from torch import nn

from .common import device, load_cases, make_tiny_model, synthetic_samples


class _InspectableDecoderLayer(nn.Module):
    """Deterministic decoder layer: exposes reference-update contract in isolation."""

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        memory,
        spatial_shapes,
        level_start_index,
        memory_key_padding_mask,
        text_memory,
        text_key_padding_mask,
        pos,
    ):
        return tgt + 0.25 * torch.tanh(query_pos) + 0.025 * memory.mean(dim=0, keepdim=True)


class TestDecoderReferenceRegression(unittest.TestCase):
    def test_supervised_box_centers_seed_following_layer_references(self):
        from mdetr.models.transformer import DeformableTransformerDecoder

        torch.manual_seed(43)
        decoder = DeformableTransformerDecoder(
            _InspectableDecoderLayer(), num_layers=3, d_model=32, n_levels=3
        )
        box_head = nn.Linear(32, 4)
        q, b = 5, 2
        query = torch.randn(q, b, 32)
        memory = torch.randn(17, b, 32)
        shapes = torch.tensor([[3, 3], [2, 3], [1, 2]], dtype=torch.long)
        starts = torch.tensor([0, 9, 15], dtype=torch.long)
        ratios = torch.tensor(
            [[[1.0, 1.0], [0.75, 0.8], [0.5, 0.7]], [[0.9, 0.8], [0.8, 0.8], [0.7, 0.6]]]
        )

        references, reference_grad_flags, handles = [], [], []

        def capture_reference(_module, args):
            references.append(args[2].detach().clone())
            reference_grad_flags.append(args[2].requires_grad)

        for layer in decoder.layers:
            handles.append(layer.register_forward_pre_hook(capture_reference))
        try:
            _hidden, layer_boxes = decoder(
                tgt=torch.zeros_like(query),
                query_pos=query,
                memory=memory,
                spatial_shapes=shapes,
                level_start_index=starts,
                valid_ratios=ratios,
                memory_key_padding_mask=None,
                text_memory=None,
                text_key_padding_mask=None,
                pos=None,
                bbox_embed=box_head,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(references), 3)
        self.assertEqual(reference_grad_flags, [True, False, False])
        self.assertEqual(tuple(layer_boxes.shape), (3, q, b, 4))
        initial_ref = decoder.ref_point_head(query).sigmoid().permute(1, 0, 2)
        torch.testing.assert_close(
            references[0], initial_ref[:, :, None] * ratios[:, None], rtol=1e-5, atol=1e-5
        )
        for level in (1, 2):
            supervised_center = layer_boxes[level - 1, :, :, :2].permute(1, 0, 2)
            torch.testing.assert_close(
                references[level],
                supervised_center[:, :, None] * ratios[:, None],
                rtol=1e-5,
                atol=1e-5,
                msg="Decoder must sample around previous SUPERVISED box centers",
            )

        (layer_boxes.square().sum()).backward()
        self.assertGreater(float(decoder.ref_point_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(box_head.weight.grad.abs().sum()), 0.0)
        # Only supervised box predictions may refine sampling references.
        self.assertFalse(any("refine_head" in name for name, _ in decoder.named_parameters()))

    def test_decoder_rejects_missing_shared_bbox_head(self):
        from mdetr.models.transformer import DeformableTransformerDecoder

        decoder = DeformableTransformerDecoder(_InspectableDecoderLayer(), 2, 32, 1)
        with self.assertRaisesRegex(ValueError, "bounding-box head|bbox_embed|requires"):
            decoder(
                torch.zeros(2, 1, 32),
                torch.zeros(2, 1, 32),
                torch.zeros(4, 1, 32),
                torch.tensor([[2, 2]]),
                torch.tensor([0]),
                torch.ones(1, 1, 2),
                None,
                None,
                None,
                None,
                bbox_embed=None,
            )


class TestActualEncoderDecoderWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(49)
        cls.payload = load_cases()
        cls.model, cls.tokenizer = make_tiny_model(cls.payload, device())
        cls.images = synthetic_samples(device(), count=len(cls.payload["images"]))

    def test_encoder_spatial_shapes_masks_valid_ratios(self):
        images = self.images
        memory = self.model(
            images, [x["caption"] for x in self.payload["images"]], encode_and_save=True
        )
        shapes = memory["spatial_shapes"]
        lengths = shapes.prod(dim=1)
        self.assertEqual(len(shapes), 3)
        self.assertEqual(int(lengths.sum()), memory["img_memory"].shape[0])
        self.assertEqual(int(lengths.sum()), memory["mask"].shape[1])
        expected_starts = torch.cat(
            [torch.tensor([0], device=shapes.device), lengths.cumsum(0)[:-1]]
        )
        torch.testing.assert_close(memory["level_start_index"], expected_starts)
        torch.testing.assert_close(memory["valid_ratios"], torch.ones_like(memory["valid_ratios"]))

        # Padding must propagate through levels and influence valid ratios.
        padded = synthetic_samples(device(), count=1)
        padded.mask[:, -8:, :] = True
        memory_pad = self.model(
            padded, [self.payload["images"][0]["caption"]], encode_and_save=True
        )
        self.assertTrue((memory_pad["valid_ratios"][..., 1] < 1).all().item())
        self.assertTrue((memory_pad["valid_ratios"] > 0).all().item())

    def test_encoder_reference_grid_rescales_source_and_target_padding(self):
        from mdetr.models.transformer import _get_reference_points

        shapes = torch.tensor([[3, 4], [2, 2]])
        ratios = torch.tensor([[[0.5, 2 / 3], [0.5, 0.5]], [[1.0, 1.0], [1.0, 1.0]]])
        actual = _get_reference_points(shapes, ratios, torch.device("cpu"))
        expected = []
        for batch_ratios in ratios.tolist():
            points = []
            for level, (height, width) in enumerate(shapes.tolist()):
                source_width, source_height = batch_ratios[level]
                for y in range(height):
                    for x in range(width):
                        points.append(
                            [
                                [
                                    (x + 0.5) / (width * source_width) * target_width,
                                    (y + 0.5) / (height * source_height) * target_height,
                                ]
                                for target_width, target_height in batch_ratios
                            ]
                        )
            expected.append(points)
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-6)

    def test_model_decoder_refs_equal_previous_predicted_centers(self):
        refs, handles = {}, []
        for idx, layer in enumerate(self.model.transformer.decoder.layers):
            handles.append(
                layer.cross_attn_img.register_forward_pre_hook(
                    lambda _module, inputs, i=idx: refs.__setitem__(i, inputs[1].detach().clone())
                )
            )
        try:
            cache = self.model(
                self.images, [x["caption"] for x in self.payload["images"]], encode_and_save=True
            )
            out = self.model(
                self.images,
                [x["caption"] for x in self.payload["images"]],
                encode_and_save=False,
                memory_cache=cache,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(refs), self.model.transformer.decoder.num_layers)
        self.assertIn("aux_outputs", out)
        for idx, aux in enumerate(out["aux_outputs"]):
            actual_next_reference = refs[idx + 1]
            expected = aux["pred_boxes"][..., :2][:, :, None, :] * cache["valid_ratios"][:, None]
            torch.testing.assert_close(
                actual_next_reference,
                expected,
                atol=5e-5,
                rtol=5e-5,
                msg="Regression: references must be tied to box predictions",
            )
        self.assertTrue(torch.isfinite(out["pred_boxes"]).all().item())
        self.assertTrue(((out["pred_boxes"] >= 0) & (out["pred_boxes"] <= 1)).all().item())

    def test_same_caption_different_images_reaches_encoded_memory_and_boxes(self):
        caption = self.payload["images"][0]["caption"]
        images = synthetic_samples(device(), count=2)
        control = synthetic_samples(device(), count=2)
        control.tensors[1].copy_(control.tensors[0])
        control_cache = self.model(control, [caption, caption], encode_and_save=True)
        control_output = self.model(
            control, [caption, caption], encode_and_save=False, memory_cache=control_cache
        )
        torch.testing.assert_close(
            control_cache["img_memory"][:, 0], control_cache["img_memory"][:, 1]
        )
        torch.testing.assert_close(control_output["pred_boxes"][0], control_output["pred_boxes"][1])
        cache = self.model(images, [caption, caption], encode_and_save=True)
        out = self.model(images, [caption, caption], encode_and_save=False, memory_cache=cache)
        self.assertGreater(
            (cache["img_memory"][:, 0] - cache["img_memory"][:, 1]).abs().mean().item(), 1e-6
        )
        self.assertGreater((out["pred_boxes"][0] - out["pred_boxes"][1]).abs().mean().item(), 1e-7)
        # Nonzero conditioning alone does not claim the model learned accurate localization.
