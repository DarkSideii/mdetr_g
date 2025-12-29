# geo_datasets/dior_rsvg_preprocess.py
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from botocore.config import Config
from botocore.exceptions import ClientError

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm

import mdetr.datasets.transforms as T
from util.augs import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomRotate90,
    RandomVerticalFlip,
    RandomGaussianNoise
)

# ─────────────────────────────── S3 config ────────────────────────────────
_S3_CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
)

# ─────────────────────────────── cache base ────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_CACHE_BASE = _THIS_DIR.parent / "dior_rsvg_cache"
_CACHE_BASE.mkdir(parents=True, exist_ok=True)

# Bump this if you change dataset materialization format (prevents stale parquet reuse)
_CACHE_VERSION = "dior_rsvg_per_object_caption_objspan_v3"


# ────────────────────────────── helpers ──────────────────────────────
def _is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _split_s3_prefix_uri(uri: str) -> Tuple[str, str]:
    """
    Parse an S3 *prefix* URI: s3://bucket/optional/prefix
    Returns (bucket, prefix_with_trailing_slash_or_empty).
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[5:]
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _split_s3_object_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key and return (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 object uri: {uri}")
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key


def _join_s3_prefix(prefix_uri: str, filename: str) -> str:
    b, p = _split_s3_prefix_uri(prefix_uri)
    return f"s3://{b}/{p}{filename}"


def _cache_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _norm_alnum(s: str) -> str:
    """Lowercase and keep only [a-z0-9]."""
    if not s:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _find_object_mention_span(
    caption: str,
    cls_name: str,
    max_ngram_words: int = 6,
    min_ratio: float = 0.55,
) -> Optional[Tuple[int, int]]:
    """
    Find a character span in `caption` that best corresponds to `cls_name`.

    We try to match the *object phrase* inside the full caption, NOT the whole caption.
    Works for:
      - groundtrackfield  vs  "ground track field"
      - baseballfield     vs  "baseball field"
      - partial mentions: groundtrackfield vs "track field"

    Returns (tok_beg, tok_end) as character indices into caption, end-exclusive.
    Returns None if no good match is found.
    """
    caption = caption or ""
    cls_norm = _norm_alnum(cls_name or "")
    if not caption or not cls_norm:
        return None

    words = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(caption)]
    if not words:
        return None

    best_span: Optional[Tuple[int, int]] = None
    best_key: Tuple[float, int, int] = (-1.0, 0, 0)  # (score, -start, -span_len)

    n = len(words)
    for i in range(n):
        for j in range(i, min(n, i + max_ngram_words)):
            start = words[i][1]
            end = words[j][2]
            span_len = end - start

            # Normalized phrase = concatenation of the selected words (spaces removed)
            phrase_norm = "".join(words[k][0].lower() for k in range(i, j + 1))
            if not phrase_norm:
                continue

            # Score: prefer exact, then substring, then fuzzy ratio
            if phrase_norm == cls_norm:
                score = 3.0
            elif phrase_norm in cls_norm:
                # caption uses subset of class (e.g. "trackfield" in "groundtrackfield")
                score = 2.0 + (len(phrase_norm) / max(1, len(cls_norm)))
            elif cls_norm in phrase_norm:
                # caption phrase contains class but with extra words (penalize extras)
                score = 2.0 + (len(cls_norm) / max(1, len(phrase_norm)))
            else:
                score = SequenceMatcher(None, phrase_norm, cls_norm).ratio()

            # Tie-breaks:
            # - earlier start preferred
            # - shorter span preferred (keeps just the object phrase, not adjectives)
            key = (float(score), -start, -span_len)
            if key > best_key:
                best_key = key
                best_span = (start, end)

    if best_span is None:
        return None

    # If it’s only a fuzzy ratio match, enforce a minimum; substring/exact are >2 so they pass.
    if best_key[0] < float(min_ratio):
        return None

    return best_span


def _dict_to_table(d: Dict[int, Dict]) -> pa.Table:
    rows: List[dict] = []
    for i, rec in d.items():
        if not rec["annotations"]:
            rows.append(
                dict(
                    img_id=i,
                    stem=rec["filename_stem"],
                    path=rec["image_path"],
                    caption=rec.get("caption", ""),
                    bbox=[0.0, 0.0, 0.0, 0.0],
                    area=0.0,
                    cat_id=0,
                    iscrowd=0,
                    tok_beg=0,
                    tok_end=0,
                    orig_h=rec["orig_size"][0],
                    orig_w=rec["orig_size"][1],
                )
            )
            continue

        for a in rec["annotations"]:
            rows.append(
                dict(
                    img_id=i,
                    stem=rec["filename_stem"],
                    path=rec["image_path"],
                    caption=rec.get("caption", ""),
                    bbox=a["bbox"],
                    area=a["area"],
                    cat_id=a["category_id"],
                    iscrowd=a.get("iscrowd", 0),
                    tok_beg=a.get("tok_beg", 0),
                    tok_end=a.get("tok_end", 0),
                    orig_h=rec["orig_size"][0],
                    orig_w=rec["orig_size"][1],
                )
            )
    return pa.Table.from_pylist(rows)


def _table_to_dict(t: pa.Table) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for r in t.to_pylist():
        i = r["img_id"]
        out.setdefault(
            i,
            dict(
                image_id=i,
                image_path=r["path"],
                caption=r.get("caption", ""),
                filename_stem=r["stem"],
                orig_size=(r["orig_h"], r["orig_w"]),
                annotations=[],
            ),
        )
        if r["cat_id"] and r["area"] > 0:
            out[i]["annotations"].append(
                dict(
                    bbox=r["bbox"],
                    area=r["area"],
                    category_id=r["cat_id"],
                    iscrowd=r.get("iscrowd", 0),
                    tok_beg=r.get("tok_beg", 0),
                    tok_end=r.get("tok_end", 0),
                )
            )
    return out


def _build_coco_like(ann_dict: Dict[int, Dict], idx_to_class: Dict[int, str]) -> Dict:
    imgs, anns, aid = [], [], 0
    for i, rec in ann_dict.items():
        h, w = rec["orig_size"]
        imgs.append(dict(id=i, file_name=rec["image_path"], height=h, width=w))
        for a in rec["annotations"]:
            anns.append(
                dict(
                    id=aid,
                    image_id=i,
                    category_id=a["category_id"],
                    bbox=_xyxy_to_xywh(a["bbox"]),
                    area=a["area"],
                    iscrowd=a.get("iscrowd", 0),
                )
            )
            aid += 1

    cats = [{"id": cid, "name": idx_to_class[cid]} for cid in sorted(idx_to_class)]
    return {"images": imgs, "annotations": anns, "categories": cats, "info": {}, "licenses": []}


# ─────────────────────────────── Dataset ────────────────────────────────
@dataclass
class _Sources:
    # NO DEFAULTS: must come from JSON
    images_dir: str         # local dir OR s3://bucket/prefix/to/images
    annotations_dir: str    # local dir OR s3://bucket/prefix/to/annotations


class DiorRSVGModulatedDetection(Dataset):
    """
    DIOR-RSVG Pascal VOC XML (RefExp-style materialization).

    IMPORTANT:
      - Each <object><description> is treated as a SEPARATE caption.
      - Therefore, each physical image can appear N times in the dataset:
            (image, caption_i, box_i)

    For each dataset item:
      - caption = one object's <description> (or fallback to class name)
      - annotations = ONLY that object's GT box
      - tok_beg/tok_end spans cover ONLY the object mention inside the caption
        (best match vs <name>), fallback to full caption if no match is found
      - optional positive_map is built if tokenizer is provided
    """

    def __init__(
        self,
        sources: _Sources,
        tokenizer=None,
        max_text_len: int = 255,   # ties to your fixed 255-token head
    ):
        super().__init__()

        if not sources.images_dir or not sources.annotations_dir:
            raise ValueError("DIOR-RSVG: images_dir and annotations_dir must be provided (from JSON).")

        self.sources = sources
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)

        # class mappings are per-dataset (no global leakage)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        # per-source cache dir (prevents mixing train vs eval caches)
        ck = _cache_key(_CACHE_VERSION, self.sources.images_dir, self.sources.annotations_dir)
        self.cache_dir = _CACHE_BASE / ck
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.arrow_cache = self.cache_dir / "annotations.parquet"
        self.classes_json = self.cache_dir / "classes.json"

        cache_found = self.arrow_cache.exists() and self.classes_json.exists()
        if cache_found:
            table = pq.read_table(self.arrow_cache)
            ann_raw = _table_to_dict(table)
            loaded = json.loads(self.classes_json.read_text())
            self.class_to_idx = {k: int(v) for k, v in loaded.items()}
            self.idx_to_class = {int(v): k for k, v in loaded.items()}
        else:
            ann_raw = self._build_annotations()
            pq.write_table(_dict_to_table(ann_raw), self.arrow_cache, compression="zstd")
            self.classes_json.write_text(json.dumps(self.class_to_idx))

        self.annotations = ann_raw
        self.ids_all = list(self.annotations.keys())
        self.ids = self.ids_all

        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations, self.idx_to_class)
        self.coco.createIndex()

        self._s3 = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        path = rec["image_path"]
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        path = str(path)

        if _is_s3(path):
            bucket, key = _split_s3_object_uri(path)
            s3 = self._get_s3()
            img_bin = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            img = Image.open(io.BytesIO(img_bin)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")

        # deep copy via json round-trip to avoid in-place transforms mutating cache dict
        rec_copy = json.loads(json.dumps(rec))
        img, tgt = _ConvertDiorRSVGToTarget(
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
        )(img, rec_copy)

        return img, tgt

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3"] = None
        return state

    def _get_s3(self):
        if self._s3 is None:
            self._s3 = boto3.client("s3", config=_S3_CFG)
        return self._s3

    def _build_annotations(self) -> Dict[int, Dict]:
        # enforce local-vs-s3 consistency per source (but allow different buckets/prefixes)
        ann_is_s3 = _is_s3(self.sources.annotations_dir)
        img_is_s3 = _is_s3(self.sources.images_dir)
        if ann_is_s3 != img_is_s3:
            raise ValueError(
                "DIOR-RSVG: images_dir and annotations_dir must both be local or both be s3://..."
            )

        if ann_is_s3:
            return self._build_annotations_s3()
        return self._build_annotations_local()

    def _build_annotations_local(self) -> Dict[int, Dict]:
        labels_dir = Path(self.sources.annotations_dir)
        images_dir = Path(self.sources.images_dir)

        if not labels_dir.exists():
            raise FileNotFoundError(f"Annotations dir not found: {labels_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")

        xml_paths = sorted(labels_dir.rglob("*.xml"))
        ann: Dict[int, Dict] = {}
        img_id = 0

        for xml_path in tqdm(xml_paths, desc="Parsing DIOR-RSVG XML", unit="file"):
            try:
                xml_bytes = xml_path.read_bytes()
            except Exception:
                continue

            samples = self._parse_single_xml(xml_bytes, xml_stem=xml_path.stem)
            if not samples:
                continue

            for s in samples:
                img_path = images_dir / s["filename"]

                cap_idx = int(s.get("cap_idx", 0))
                stem_base = Path(s["filename"]).stem
                stem = f"{stem_base}_cap{cap_idx:03d}"

                rec = dict(
                    image_id=img_id,
                    image_path=str(img_path),
                    caption=s["caption"],
                    filename_stem=stem,
                    orig_size=s["orig_size"],
                    annotations=s["annotations"],
                )
                ann[img_id] = rec
                img_id += 1

        return ann

    def _build_annotations_s3(self) -> Dict[int, Dict]:
        ann_bucket, ann_prefix = _split_s3_prefix_uri(self.sources.annotations_dir)
        img_bucket, img_prefix = _split_s3_prefix_uri(self.sources.images_dir)

        s3 = boto3.client("s3", config=_S3_CFG)
        paginator = s3.get_paginator("list_objects_v2")

        xml_objs: List[dict] = [
            obj
            for page in paginator.paginate(Bucket=ann_bucket, Prefix=ann_prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].lower().endswith(".xml")
        ]

        ann: Dict[int, Dict] = {}
        img_id = 0

        for obj in tqdm(xml_objs, desc="Parsing DIOR-RSVG XML (S3)", unit="file"):
            xml_key = obj["Key"]
            try:
                xml_bytes = s3.get_object(Bucket=ann_bucket, Key=xml_key)["Body"].read()
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code == "NoSuchKey":
                    continue
                raise

            samples = self._parse_single_xml(xml_bytes, xml_stem=Path(xml_key).stem)
            if not samples:
                continue

            for s in samples:
                img_s3_path = f"s3://{img_bucket}/{img_prefix}{s['filename']}"

                cap_idx = int(s.get("cap_idx", 0))
                stem_base = Path(s["filename"]).stem
                stem = f"{stem_base}_cap{cap_idx:03d}"

                rec = dict(
                    image_id=img_id,
                    image_path=img_s3_path,
                    caption=s["caption"],
                    filename_stem=stem,
                    orig_size=s["orig_size"],
                    annotations=s["annotations"],
                )
                ann[img_id] = rec
                img_id += 1

        return ann

    def _parse_single_xml(self, xml_bytes: bytes, xml_stem: str) -> Optional[List[Dict]]:
        """
        Return a list of samples, one per <object>:
          sample = {filename, orig_size=(h,w), caption, annotations=[{bbox,..., tok_beg, tok_end}], cap_idx}
        """
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return None

        filename = root.findtext("filename") or f"{xml_stem}.jpg"

        # size from XML if present
        size_node = root.find("size")
        if size_node is not None:
            try:
                w = int(size_node.findtext("width") or 0)
                h = int(size_node.findtext("height") or 0)
            except Exception:
                w = h = 0
        else:
            w = h = 0

        # First pass: collect valid per-object samples (caption + single box)
        tmp_samples: List[Dict] = []
        max_x = 0.0
        max_y = 0.0

        cap_idx = 0
        for obj_node in root.findall("object"):
            cls_name = (obj_node.findtext("name") or "unknown").strip() or "unknown"
            if cls_name not in self.class_to_idx:
                cid = len(self.class_to_idx) + 1
                self.class_to_idx[cls_name] = cid
                self.idx_to_class[cid] = cls_name

            desc = (obj_node.findtext("description") or "").strip()
            if not desc:
                desc = cls_name

            # ONE caption per object (no concatenation)
            caption = desc

            # IMPORTANT: token span should cover ONLY the object phrase inside the caption
            span = _find_object_mention_span(caption, cls_name)
            if span is None:
                # Fallback (keeps training from silently losing supervision if captions don’t contain the class phrase)
                tok_beg, tok_end = 0, len(caption)
            else:
                tok_beg, tok_end = span

            bnd = obj_node.find("bndbox")
            if bnd is None:
                cap_idx += 1
                continue

            try:
                x1 = float(bnd.findtext("xmin"))
                y1 = float(bnd.findtext("ymin"))
                x2 = float(bnd.findtext("xmax"))
                y2 = float(bnd.findtext("ymax"))
            except (TypeError, ValueError):
                cap_idx += 1
                continue

            x0 = min(x1, x2)
            y0 = min(y1, y2)
            x1c = max(x1, x2)
            y1c = max(y1, y2)

            # Clamp to non-negative always
            x0 = max(0.0, x0)
            y0 = max(0.0, y0)
            x1c = max(0.0, x1c)
            y1c = max(0.0, y1c)

            # If XML size exists, clamp into it
            if w > 0 and h > 0:
                x0 = min(float(w), x0)
                y0 = min(float(h), y0)
                x1c = min(float(w), x1c)
                y1c = min(float(h), y1c)

            if (x1c - x0) < 2 or (y1c - y0) < 2:
                cap_idx += 1
                continue

            max_x = max(max_x, x1c)
            max_y = max(max_y, y1c)

            area = float((x1c - x0) * (y1c - y0))
            ann = dict(
                bbox=[x0, y0, x1c, y1c],
                area=area,
                category_id=self.class_to_idx[cls_name],
                iscrowd=0,
                tok_beg=int(tok_beg),
                tok_end=int(tok_end),
            )

            tmp_samples.append(
                dict(
                    filename=filename,
                    caption=caption,
                    annotations=[ann],
                    cap_idx=int(cap_idx),
                )
            )
            cap_idx += 1

        # Determine orig_size (h,w). If missing in XML, infer from boxes if possible.
        if h <= 0 or w <= 0:
            if max_x > 0 and max_y > 0:
                w = int(max_x + 1)
                h = int(max_y + 1)
            else:
                h = w = 0

        orig_size = (int(h), int(w))

        # Keep one empty sample if no valid objects (optional safety)
        if not tmp_samples:
            return [
                dict(
                    filename=filename,
                    orig_size=orig_size,
                    caption="",
                    annotations=[],
                    cap_idx=0,
                )
            ]

        # Attach orig_size to each sample and return
        out: List[Dict] = []
        for s in tmp_samples:
            out.append(
                dict(
                    filename=s["filename"],
                    orig_size=orig_size,
                    caption=s["caption"],
                    annotations=s["annotations"],
                    cap_idx=s["cap_idx"],
                )
            )
        return out


# ───────────────────────── transforms ─────────────────────────
def _dior_rsvg_transforms(split: str):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if split == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.50),
                RandomVerticalFlip(prob=0.50),
                RandomRotate90(prob=0.25),
                RandomColorJitter(prob=0.25, brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                RandomGaussianBlur(prob=0.10, radius=(0.1, 0.6)),
                RandomGaussianNoise(prob=0.08, std=5.0),
                normalize,
            ])
    if split in ("val", "test"):
        return T.Compose([normalize])
    raise ValueError(split)


class _WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base: DiorRSVGModulatedDetection, tfm):
        self.base, self.tfm, self.coco = base, tfm, base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        img, tgt = self.tfm(img, tgt)
        return img, tgt


# ───────────────────── dataset builder ─────────────────────
def _require(args, key: str) -> str:
    v = getattr(args, key, None)
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ValueError(f"DIOR-RSVG: missing required config key: {key}")
    return str(v)


def build_dior_rsvg(set_name: str, args):
    """
    set_name ∈ {"train", "val", "test"}.

    Behavior:
      - Train mode (args.eval==False): uses train_* dirs and splits into train/val
      - Eval mode  (args.eval==True): uses eval_* dirs and returns 100% of eval set

    Required JSON keys:
      Train mode:
        - train_images_dir
        - train_annotations_dir
        - val_split_ratio
      Eval mode:
        - eval_images_dir
        - eval_annotations_dir
    """
    eval_mode = bool(getattr(args, "eval", False)) or bool(getattr(args, "test", False))

    if eval_mode:
        images_dir = _require(args, "eval_images_dir")
        annotations_dir = _require(args, "eval_annotations_dir")
    else:
        images_dir = _require(args, "train_images_dir")
        annotations_dir = _require(args, "train_annotations_dir")

    sources = _Sources(images_dir=images_dir, annotations_dir=annotations_dir)

    full = DiorRSVGModulatedDetection(
        sources=sources,
        tokenizer=getattr(args, "tokenizer", None),
        max_text_len=int(getattr(args, "max_text_len", 255)),
    )

    ids_all = list(full.ids_all)

    if eval_mode:
        # 100% of held-out eval set
        full.ids = ids_all
        return _WrappedDataset(full, _dior_rsvg_transforms("val"))

    # Train mode: random split train/val
    if set_name not in ("train", "val"):
        raise ValueError(f"Train mode only supports set_name in ('train','val'), got {set_name!r}")

    seed = int(getattr(args, "seed", 42))
    val_ratio = float(_require(args, "val_split_ratio"))  # must come from JSON

    rng = random.Random(seed)
    ids_shuf = ids_all[:]
    rng.shuffle(ids_shuf)

    n = len(ids_shuf)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else n  # keep both sides non-empty if possible

    val_ids = ids_shuf[:n_val]
    train_ids = ids_shuf[n_val:]

    full.ids = train_ids if set_name == "train" else val_ids
    return _WrappedDataset(full, _dior_rsvg_transforms("train" if set_name == "train" else "val"))


# ───────────────────── convert-to-target ─────────────────────
class _ConvertDiorRSVGToTarget:
    """
    Converter:
      - boxes, labels (0-based)
      - tokens_positive spans
      - (optional) positive_map if tokenizer is available
    """

    def __init__(self, tokenizer=None, max_text_len: int = 255):
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)

    def _build_positive_map(self, caption: str, tokens_positive: List[List[Tuple[int, int]]]) -> Optional[torch.Tensor]:
        if self.tokenizer is None:
            return None
        if caption is None:
            caption = ""

        enc = self.tokenizer(
            caption,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        offsets = enc.get("offset_mapping", None)
        if offsets is None:
            return None

        # offsets is either [(s,e), ...] or [[(s,e), ...]]
        if (
            len(offsets) > 0
            and isinstance(offsets[0], (list, tuple))
            and len(offsets[0]) == 2
            and isinstance(offsets[0][0], int)
        ):
            offs = offsets
        else:
            offs = offsets[0]

        pm = torch.zeros((len(tokens_positive), self.max_text_len), dtype=torch.float32)
        for i, spans in enumerate(tokens_positive):
            for (beg, end) in spans:
                beg = int(beg)
                end = int(end)
                if end <= beg:
                    continue
                for j, (s, e) in enumerate(offs):
                    s = int(s)
                    e = int(e)
                    # special tokens often have (0,0)
                    if s == 0 and e == 0:
                        continue
                    # overlap test with [beg,end)
                    if e <= beg or s >= end:
                        continue
                    pm[i, j] = 1.0
        return pm

    def __call__(self, image, record: Dict):
        W, H = image.size  # PIL: (width, height)

        anns = record.get("annotations", [])
        caption = record.get("caption", "")

        if len(anns) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            tokens_positive: List[List[Tuple[int, int]]] = []
        else:
            boxes = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32)
            boxes[:, 0::2].clamp_(0, W)
            boxes[:, 1::2].clamp_(0, H)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]

            labels_1b = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)[keep]
            labels = labels_1b - 1  # 0-based

            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.tensor([a.get("iscrowd", 0) for a in anns], dtype=torch.int64)[keep]

            spans_all = [[(int(a.get("tok_beg", 0)), int(a.get("tok_end", 0)))] for a in anns]
            tokens_positive = [s for s, k in zip(spans_all, keep.tolist()) if k]

        positive_map = self._build_positive_map(caption, tokens_positive)

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=torch.tensor([record["image_id"]]),
            area=area,
            iscrowd=iscrowd,
            orig_size=torch.tensor([record["orig_size"][0], record["orig_size"][1]]),
            size=torch.tensor([H, W]),
            caption=caption,
            tokens_positive=tokens_positive,
            filename_stem=record.get("filename_stem", "N/A"),
        )
        if positive_map is not None:
            target["positive_map"] = positive_map

        return image, target
