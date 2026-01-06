"""
Move a random fraction of image/XML *pairs* into a test directory.

Only images with a matching `<stem>.xml` are eligible. When an image is selected, its
matching XML is moved as well.

Creates these subfolders under `--test-dir`:
  - "test images"
  - "test xmls"

Files are **moved** (removed from the source dirs).

Examples:
  python splits.py --images-dir ./images --xml-dir ./xmls --test-dir ./test
  python splits.py --images-dir ./data --xml-dir ./data --test-dir ./test --pct 0.10 --seed 123
  python splits.py --images-dir ./images --xml-dir ./xmls --test-dir ./test --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sample_count(n: int, pct: float) -> int:
    """Return how many items to sample from `n` at fraction `pct` (min 1 when pct>0)."""
    if n <= 0 or pct <= 0:
        return 0
    k = int(n * pct)
    if k == 0:
        k = 1  # ensure at least one pair when pct>0
    return min(k, n)


def move_file(src: Path, dst_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    """Move `src` into `dst_dir`, optionally overwriting. Honors `dry_run`."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")

    if dry_run:
        print(f"[DRY-RUN] MOVE {src} -> {dst}")
        return

    if dst.exists() and overwrite:
        dst.unlink()

    shutil.move(str(src), str(dst))


def main() -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Directory containing images")
    ap.add_argument("--xml-dir", required=True, help="Directory containing XML files")
    ap.add_argument("--test-dir", required=True, help="Destination root directory (will create subfolders inside)")
    ap.add_argument("--pct", type=float, default=0.10, help="Fraction of pairs to move (default: 0.10)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if destination file exists")
    ap.add_argument("--dry-run", action="store_true", help="Print what would move, but don't move anything")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    xml_dir = Path(args.xml_dir).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()

    if not images_dir.is_dir():
        raise NotADirectoryError(f"--images-dir is not a directory: {images_dir}")
    if not xml_dir.is_dir():
        raise NotADirectoryError(f"--xml-dir is not a directory: {xml_dir}")

    # Destination subdirs (names are fixed, including spaces).
    dst_images = test_dir / "test images"
    dst_xmls = test_dir / "test xmls"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_xmls.mkdir(parents=True, exist_ok=True)

    # Eligible pairs: image files with a matching `<stem>.xml`.
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    pairs: list[tuple[Path, Path]] = []
    missing_xml: list[Path] = []

    for img in sorted(images):
        xml = xml_dir / f"{img.stem}.xml"
        if xml.is_file():
            pairs.append((img, xml))
        else:
            missing_xml.append(img)

    if missing_xml:
        print(f"Note: {len(missing_xml)} image(s) skipped because matching XML was not found.")

    rng = random.Random(args.seed)

    k = sample_count(len(pairs), args.pct)
    chosen_pairs = rng.sample(pairs, k=k) if k else []

    print(f"Found {len(pairs)} valid image+xml pair(s). Moving {len(chosen_pairs)} (~{args.pct*100:.1f}%).")
    print(f"Destination folders:\n  {dst_images}\n  {dst_xmls}")

    for img_path, xml_path in chosen_pairs:
        move_file(img_path, dst_images, overwrite=args.overwrite, dry_run=args.dry_run)
        move_file(xml_path, dst_xmls, overwrite=args.overwrite, dry_run=args.dry_run)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
