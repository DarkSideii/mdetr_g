#!/usr/bin/env python3
"""CUDA smoke test: the patched MMCV wrapper returns a pure attention update."""
import torch

from mdetr.models.transformer import _MSDeformAttnWrapper


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MMCV deformable-attention smoke test")
    torch.manual_seed(42)
    device = torch.device("cuda")
    wrapper = _MSDeformAttnWrapper(
        d_model=256, n_levels=1, n_heads=8, n_points=4
    ).to(device).eval()

    assert wrapper.mod.dropout.p == 0.0, (
        "MMCV internal dropout must be disabled; outer layer owns dropout"
    )
    query = torch.randn(4, 2, 256, device=device)
    value = torch.randn(4, 2, 256, device=device)
    refs = torch.rand(2, 4, 1, 2, device=device)
    shapes = torch.tensor([[2, 2]], dtype=torch.long, device=device)
    starts = torch.tensor([0], dtype=torch.long, device=device)

    with torch.inference_mode():
        wrapper_output = wrapper(
            query, refs, value, shapes, starts, key_padding_mask=None
        ).transpose(0, 1)
        q = query.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()
        kwargs = dict(
            query=q, value=v, reference_points=refs,
            spatial_shapes=shapes, level_start_index=starts,
        )
        pure_attention = wrapper.mod(
            **kwargs, identity=torch.zeros_like(q)
        )
        default_mmcv = wrapper.mod(**kwargs)

    torch.testing.assert_close(
        wrapper_output, pure_attention, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        default_mmcv - pure_attention, q, atol=1e-5, rtol=1e-4
    )
    print("PASS: wrapper returns attention only (no internal residual).")
    print("PASS: default MMCV adds the query, but patched wrapper does not.")
    print("PASS: MMCV internal dropout=0; encoder/decoder outer dropout remains.")


if __name__ == "__main__":
    main()
