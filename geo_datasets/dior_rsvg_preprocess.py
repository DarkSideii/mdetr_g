# geo_datasets/dior_rsvg_preprocess.py
from __future__ import annotations

import hashlib
import io
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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

# ─────────────────────────────── S3 config ────────────────────────────────
_S3_CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
)

# ─────────────────────────────── cache base ────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_CACHE_BASE = _THIS_DIR / "dior_rsvg_cache"
_CACHE_BASE.mkdir(parents=True, exist_ok=True)


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
    DIOR-RSVG Pascal VOC XML.

    - caption built by concatenating per-object <description> fields.
    - each GT box stores (tok_beg, tok_end) char spans into caption.
    - optionally builds per-sample positive_map if tokenizer is provided.
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
        ck = _cache_key(self.sources.images_dir, self.sources.annotations_dir)
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

            parsed = self._parse_single_xml(xml_bytes, xml_stem=xml_path.stem)
            if parsed is None:
                continue

            img_path = images_dir / parsed["filename"]
            rec = dict(
                image_id=img_id,
                image_path=str(img_path),
                caption=parsed["caption"],
                filename_stem=Path(parsed["filename"]).stem,
                orig_size=parsed["orig_size"],
                annotations=parsed["annotations"],
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

            parsed = self._parse_single_xml(xml_bytes, xml_stem=Path(xml_key).stem)
            if parsed is None:
                continue

            img_s3_path = f"s3://{img_bucket}/{img_prefix}{parsed['filename']}"
            rec = dict(
                image_id=img_id,
                image_path=img_s3_path,
                caption=parsed["caption"],
                filename_stem=Path(parsed["filename"]).stem,
                orig_size=parsed["orig_size"],
                annotations=parsed["annotations"],
            )
            ann[img_id] = rec
            img_id += 1

        return ann

    def _parse_single_xml(self, xml_bytes: bytes, xml_stem: str) -> Optional[Dict]:
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

        caption = ""
        ann_list: List[Dict] = []

        for obj_node in root.findall("object"):
            cls_name = (obj_node.findtext("name") or "unknown").strip() or "unknown"
            if cls_name not in self.class_to_idx:
                cid = len(self.class_to_idx) + 1
                self.class_to_idx[cls_name] = cid
                self.idx_to_class[cid] = cls_name

            desc = (obj_node.findtext("description") or "").strip()
            if not desc:
                desc = cls_name

            if caption:
                caption += ". "
            tok_beg = len(caption)
            caption += desc
            tok_end = len(caption)

            bnd = obj_node.find("bndbox")
            if bnd is None:
                continue

            try:
                x1 = float(bnd.findtext("xmin"))
                y1 = float(bnd.findtext("ymin"))
                x2 = float(bnd.findtext("xmax"))
                y2 = float(bnd.findtext("ymax"))
            except (TypeError, ValueError):
                continue

            x0 = min(x1, x2)
            y0 = min(y1, y2)
            x1c = max(x1, x2)
            y1c = max(y1, y2)

            if w > 0 and h > 0:
                x0 = max(0.0, min(float(w), x0))
                y0 = max(0.0, min(float(h), y0))
                x1c = max(0.0, min(float(w), x1c))
                y1c = max(0.0, min(float(h), y1c))

            if (x1c - x0) < 2 or (y1c - y0) < 2:
                continue

            area = float((x1c - x0) * (y1c - y0))
            ann_list.append(
                dict(
                    bbox=[x0, y0, x1c, y1c],
                    area=area,
                    category_id=self.class_to_idx[cls_name],
                    iscrowd=0,
                    tok_beg=int(tok_beg),
                    tok_end=int(tok_end),
                )
            )

        # Keep images even if empty (safe for detection pipelines)
        if h <= 0 or w <= 0:
            if ann_list:
                ys = [a["bbox"][1] for a in ann_list] + [a["bbox"][3] for a in ann_list]
                xs = [a["bbox"][0] for a in ann_list] + [a["bbox"][2] for a in ann_list]
                h = int(max(ys) + 1)
                w = int(max(xs) + 1)
            else:
                h = w = 0

        return dict(filename=filename, orig_size=(h, w), caption=caption, annotations=ann_list)


# ───────────────────────── transforms ─────────────────────────
def _dior_rsvg_transforms(split: str):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if split == "train":
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    if split in ("val", "test"):
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
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
        if len(offsets) > 0 and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) == 2 and isinstance(offsets[0][0], int):
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
