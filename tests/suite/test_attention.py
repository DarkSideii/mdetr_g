# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Independent sampling checks and production MMCV wrapper contracts."""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .common import reference_sample


class TestDeformableAttention(unittest.TestCase):
    def test_independent_bilinear_sampler_location_and_gradients(self):
        # Exact pixel centers catch swapped axes and incorrect normalization.
        value = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1).requires_grad_()
        locations = torch.tensor(
            [[[[[[0.25, 0.25]]]], [[[[0.75, 0.75]]]]]], dtype=torch.float32, requires_grad=True
        )
        weights = torch.ones(1, 2, 1, 1, 1)
        output = reference_sample(value, torch.tensor([[2, 2]]), locations, weights)
        torch.testing.assert_close(output[0, :, 0], torch.tensor([1.0, 4.0]), atol=1e-6, rtol=0)
        output.square().sum().backward()
        self.assertGreater(float(value.grad.abs().sum()), 0.0)
        self.assertGreater(float(locations.grad.abs().sum()), 0.0)

    def test_independent_multilevel_sampler_checks_level_boundaries(self):
        val = torch.arange(1, 8, dtype=torch.float32).reshape(1, 7, 1, 1)
        loc = torch.tensor([[[[[[0.25, 0.25]], [[5 / 6, 0.5]]]]]])  # exact pixel centers
        weights = torch.tensor([[[[[0.25], [0.75]]]]])
        actual = reference_sample(val, torch.tensor([[2, 2], [1, 3]]), loc, weights)
        expected = 0.25 * 1.0 + 0.75 * 7.0
        self.assertAlmostEqual(float(actual[0, 0, 0]), expected, places=5)
        with self.assertRaisesRegex(ValueError, "Feature length"):
            reference_sample(val[:, :-1], torch.tensor([[2, 2], [1, 3]]), loc, weights)

    def test_production_wrapper_preserves_batch_query_memory_layouts(self):
        from mdetr.models import transformer as module

        class BatchFirstBackend(nn.Module):
            def __init__(self, embed_dims, num_heads, num_levels, num_points, batch_first=True):
                super().__init__()
                self.batch_first = batch_first

            def forward(
                self,
                query,
                reference_points,
                value,
                spatial_shapes,
                level_start_index,
                key_padding_mask,
            ):
                if not self.batch_first:
                    query, value = query.transpose(0, 1), value.transpose(0, 1)
                assert query.shape == (2, 3, 8)
                assert value.shape == (2, 7, 8)
                assert spatial_shapes.tolist() == [[2, 2], [1, 3]]
                assert level_start_index.tolist() == [0, 4]
                memory = value.masked_fill(key_padding_mask[..., None], 0).mean(1, keepdim=True)
                output = query + memory + reference_points.mean((2, 3))[..., None]
                return output if self.batch_first else output.transpose(0, 1)

        class SequenceFirstBackend(BatchFirstBackend):
            def __init__(self, embed_dims, num_heads, num_levels, num_points):
                super().__init__(embed_dims, num_heads, num_levels, num_points, batch_first=False)

        for backend in (BatchFirstBackend, SequenceFirstBackend):
            with (
                self.subTest(layout=backend.__name__),
                patch.object(module, "_HAS_MMCV", True),
                patch.object(module, "_MMCV_MSDA", backend, create=True),
            ):
                wrapper = module._MSDeformAttnWrapper(8, 2, 2, 2)
                query = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8).requires_grad_()
                value = torch.arange(112, dtype=torch.float32).reshape(7, 2, 8).requires_grad_()
                references = torch.full((2, 3, 2, 2), 0.4, requires_grad=True)
                mask = torch.zeros(2, 7, dtype=torch.bool)
                mask[1, 4:] = True
                output = wrapper(
                    query,
                    references,
                    value,
                    torch.tensor([[2, 2], [1, 3]]),
                    torch.tensor([0, 4]),
                    mask,
                )
                expected = query + value.masked_fill(mask.T[..., None], 0).mean(0)[None] + 0.4
                torch.testing.assert_close(output, expected)
                output.sum().backward()
                for tensor in (query, value, references):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item())
                    self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def _native_or_skip(self):
        from mdetr.models import transformer as module

        required = os.environ.get("MDETR_INTEGRITY_REQUIRE_NATIVE", "0") == "1"
        if not torch.cuda.is_available() or not module._HAS_MMCV:
            if required:
                self.fail(
                    "Native MMCV/CUDA attention unavailable but --require-native-msda was specified"
                )
            self.skipTest(
                "Native MMCV/CUDA not available; synthetic integration uses a labeled reference sampler"
            )
        try:
            from mmcv.ops.multi_scale_deform_attn import (
                MultiScaleDeformableAttnFunction,
                multi_scale_deformable_attn_pytorch,
            )
        except ImportError as exc:
            if required:
                self.fail(f"MMCV function unavailable: {exc}")
            self.skipTest(f"MMCV low-level function not available: {exc}")
        return module, MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch

    def test_native_cuda_forward_and_backward_match_torch_reference(self):
        _module, op, ref = self._native_or_skip()
        torch.manual_seed(23)
        shapes = torch.tensor([[4, 5], [2, 3], [1, 2]], device="cuda", dtype=torch.long)
        starts = torch.tensor([0, 20, 26], device="cuda", dtype=torch.long)
        batch, queries, heads, dim, levels, points = 2, 7, 4, 8, 3, 4
        value = torch.randn(batch, 28, heads, dim, device="cuda", requires_grad=True)
        location = (
            (torch.rand(batch, queries, heads, levels, points, 2, device="cuda") * 0.8 + 0.1)
            .detach()
            .requires_grad_()
        )
        weights = torch.rand(
            batch, queries, heads, levels, points, device="cuda", requires_grad=True
        )
        actual = op.apply(value, shapes, starts, location, weights, 64)
        expected = ref(value, shapes, location, weights)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)

        perturbation = torch.randn_like(actual)
        grads_cuda = torch.autograd.grad(
            actual, (value, location, weights), grad_outputs=perturbation, retain_graph=True
        )
        grads_ref = torch.autograd.grad(
            expected, (value, location, weights), grad_outputs=perturbation
        )
        for label, (a, b) in zip(("value", "locations", "weights"), zip(grads_cuda, grads_ref)):
            torch.testing.assert_close(
                a, b, atol=1e-4, rtol=2e-3, msg=f"Native MMCV backward differs for {label}"
            )

    def test_real_msda_wrapper_batch_query_memory_contract(self):
        module, _fn, _ref = self._native_or_skip()
        torch.manual_seed(24)
        wrapper = (
            module._MSDeformAttnWrapper(d_model=32, n_levels=3, n_heads=4, n_points=4).cuda().eval()
        )
        shapes = torch.tensor([[4, 5], [2, 3], [1, 2]], device="cuda")
        starts = torch.tensor([0, 20, 26], device="cuda")
        query = torch.randn(7, 2, 32, device="cuda", requires_grad=True)
        value = torch.randn(28, 2, 32, device="cuda", requires_grad=True)
        refs = (torch.rand(2, 7, 3, 2, device="cuda") * 0.8 + 0.1).requires_grad_()
        mask = torch.zeros((2, 28), dtype=torch.bool, device="cuda")
        mask[1, 26:] = True
        output = wrapper(query, refs, value, shapes, starts, mask)
        self.assertEqual(tuple(output.shape), (7, 2, 32))
        backend_query = query.transpose(0, 1).contiguous() if wrapper.batch_first else query
        backend_value = value.transpose(0, 1).contiguous() if wrapper.batch_first else value
        expected = wrapper.mod(
            query=backend_query,
            value=backend_value,
            reference_points=refs,
            spatial_shapes=shapes,
            level_start_index=starts,
            key_padding_mask=mask,
        )
        if wrapper.batch_first:
            expected = expected.transpose(0, 1)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
        output.sum().backward()
        for name, tensor in (("query", query), ("value", value), ("references", refs)):
            self.assertIsNotNone(tensor.grad, f"No gradient into {name}")
            self.assertTrue(torch.isfinite(tensor.grad).all().item(), name)
            if name != "query":
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0, name)
        initial_query_gradient = query.grad.detach().clone()
        torch.testing.assert_close(value.grad[26:, 1], torch.zeros_like(value.grad[26:, 1]))

        # Zero-initialized attention weights can hide query dependence behind
        # the backend residual. Exercise learned offsets and weights explicitly.
        with torch.no_grad():
            wrapper.mod.sampling_offsets.weight.normal_(mean=0.0, std=0.01)
            wrapper.mod.attention_weights.weight.normal_(mean=0.0, std=0.01)
        wrapper.zero_grad(set_to_none=True)
        query.grad = None
        value.grad = None
        refs.grad = None

        output = wrapper(query, refs, value, shapes, starts, mask)
        output.sum().backward()
        self.assertIsNotNone(query.grad, "Query gradient path is disconnected")
        self.assertTrue(torch.isfinite(query.grad).all().item(), "query")
        self.assertGreater(
            float(query.grad.abs().sum()),
            0.0,
            "Query gradient remains zero after query-dependent weights are enabled",
        )
        self.assertFalse(
            torch.allclose(query.grad, initial_query_gradient, atol=1e-6, rtol=1e-5),
            "Query sensitivity must not come only from the backend residual",
        )
