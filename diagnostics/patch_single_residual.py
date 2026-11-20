#!/usr/bin/env python3
"""Convert the MDETR-G MMCV attention wrapper to canonical single residual.

Patches ONLY mdetr/models/transformer.py in a local checkout.
- MMCV returns attention-only output (identity=0).
- MMCV's internal dropout is disabled (dropout=0.0).
- Existing encoder/decoder outer residual + dropout + LayerNorm remain.
- Existing parameter names/shapes are unchanged.

Usage: python diagnostics/patch_single_residual.py [--repo .] [--check]
"""
import argparse
import ast
import difflib
from pathlib import Path

OLD_DOC = '"""Adapt MMCV tensor layouts while retaining its native residual behavior."""'
NEW_DOC = '"""Adapt MMCV tensor layouts; outer transformer layers own the residual."""'
OLD_CONSTRUCTOR = '''                num_points=n_points,
                batch_first=True,
'''
NEW_CONSTRUCTOR = '''                num_points=n_points,
                batch_first=True,
                dropout=0.0,  # outer encoder/decoder layers already apply dropout
'''
OLD_CALL = '''                query=q,
                reference_points=reference_points,
                value=v,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
'''
NEW_CALL = '''                query=q,
                reference_points=reference_points,
                value=v,
                identity=torch.zeros_like(q),  # MMCV must return attention update only
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
'''
OLD_FALLBACK = '''        except TypeError:
            out = self.mod(
                q, reference_points, v, spatial_shapes, level_start_index, key_padding_mask
            )
'''
NEW_FALLBACK = '''        except TypeError as exc:
            raise RuntimeError(
                "Canonical single-residual wrapper requires MMCV's keyword API "
                "(verified with MMCV 2.2.0). Positional fallback is unsafe."
            ) from exc
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--check", action="store_true",
                        help="Print intended diff without modifying any file")
    args = parser.parse_args()
    path = Path(args.repo).resolve() / "mdetr/models/transformer.py"
    if not path.is_file():
        raise SystemExit(f"Cannot find {path}")
    original = path.read_text(encoding="utf-8")

    # Only patch the wrapper, never another transformer class.
    begin = original.find("class _MSDeformAttnWrapper(nn.Module):")
    end = original.find("\nclass FeatureResizer(nn.Module):", begin)
    if begin < 0 or end < 0:
        raise SystemExit("Expected wrapper boundaries not found; refusing to patch")
    section = original[begin:end]

    if NEW_DOC in section and NEW_CALL in section and NEW_CONSTRUCTOR in section:
        print("Already patched: canonical single-residual wrapper detected.")
        return

    pairs = (
        (OLD_DOC, NEW_DOC),
        (OLD_CONSTRUCTOR, NEW_CONSTRUCTOR),
        (OLD_CALL, NEW_CALL),
        (OLD_FALLBACK, NEW_FALLBACK),
    )
    modified_section = section
    for old, new in pairs:
        if modified_section.count(old) != 1:
            raise SystemExit(
                f"Expected exactly one match for patch anchor: {old[:75]!r}; "
                "repository version differs, refusing to change anything"
            )
        modified_section = modified_section.replace(old, new, 1)

    updated = original[:begin] + modified_section + original[end:]
    ast.parse(updated, filename=str(path))

    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(path) + " (before)",
        tofile=str(path) + " (canonical)",
    ))
    print(diff)
    if args.check:
        print("CHECK ONLY: no files changed.")
        return

    backup = path.with_name("transformer.py.pre_single_residual.bak")
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}; refusing to overwrite")
    backup.write_text(original, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    print(f"Patched {path}")
    print(f"Original backed up to {backup}")
    print("NOTE: the new architecture requires a fresh training run.")
    print("Restore with: cp mdetr/models/transformer.py.pre_single_residual.bak "
          "mdetr/models/transformer.py")


if __name__ == "__main__":
    main()
