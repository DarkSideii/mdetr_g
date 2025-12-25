from __future__ import annotations

import io
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

import multimodal_framework.mdetr.datasets.transforms as T

# ─────────────────────────────── setup ────────────────────────────────
random.seed(42)

_S3_CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
)

# ─────────────────────────────── cache paths ────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_CACHE_DIR = _PARENT_DIR / "dior_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARROW_CACHE = _CACHE_DIR / "dior_annotations.parquet"
CLASSES_JSON = _CACHE_DIR / "dior_classes.json"

# ─────────────────────────────── DIOR settings ────────────────────────────────
# Keep original DIOR object names exactly as they appear in XML.
CLASS_TO_IDX: Dict[str, int] = {}
IDX_TO_CLASS: Dict[int, str] = {}


# ────────────────────────────── pyarrow helpers ─────────────────────────────
def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]


def _dict_to_table(d: Dict[int, Dict]) -> pa.Table:
    rows: List[dict] = []
    for i, rec in d.items():
        if not rec["annotations"]:
            # sentinel row so image exists in cache
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
                )
            )
    return out


def _build_coco_like(ann_dict: Dict[int, Dict]) -> Dict:
    """Return a COCO-style JSON dictionary (no saving to disk)."""
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

    cats = [{"id": cid, "name": IDX_TO_CLASS[cid]} for cid in sorted(IDX_TO_CLASS)]
    return {"images": imgs, "annotations": anns, "categories": cats, "info": {}, "licenses": []}


# ─────────────────────────────── Dataset ────────────────────────────────
@dataclass
class _S3Sources:
    bucket: str = "research-geodatasets"
    images_prefix: str = "DIOR/images/"       # e.g., 11746_1.png
    labels_prefix: str = "DIOR/annotations/"  # Pascal VOC-style XML files


class DiorModulatedDetection(Dataset):
    """
    DIOR with Pascal VOC-style XML annotations stored on S3.

    Eval-only for zero-shot:
    - No CSV
    - No human captions (caption=""); VLM provides captions at runtime
    - Class names are exactly from <object><name> text.
    """

    def __init__(
        self,
        tokenizer=None,          # kept for API parity, unused here
        return_tokens: bool = False,  # ignored; we DO NOT build positive maps
        sources: _S3Sources = _S3Sources(),
        seed: int = 42,
    ):
        super().__init__()
        self._s3 = None
        self.sources = sources
        self.seed = seed
        self.prepare = _ConvertDiorToTarget()

        cache_found = ARROW_CACHE.exists() and CLASSES_JSON.exists()
        if cache_found:
            table = pq.read_table(ARROW_CACHE)
            ann_raw = _table_to_dict(table)
            # load class mapping
            try:
                loaded = json.loads(CLASSES_JSON.read_text())
                CLASS_TO_IDX.update({k: int(v) for k, v in loaded.items()})
                for name, cid in CLASS_TO_IDX.items():
                    IDX_TO_CLASS[cid] = name
            except Exception:
                pass
        else:
            ann_raw = self._build_annotations()
            pq.write_table(_dict_to_table(ann_raw), ARROW_CACHE, compression="zstd")
            CLASSES_JSON.write_text(json.dumps(CLASS_TO_IDX))

        self.annotations = ann_raw
        self.ids_all = list(self.annotations.keys())
        self.ids = self.ids_all  # eval-only, no split

        # COCO API object for evaluator/loader compatibility
        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
        self.coco.createIndex()

    # ────────────── Dataset plumbing ───────────────
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        path = rec["image_path"]
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        path = str(path)

        if path.startswith("s3://"):
            key = path.removeprefix(f"s3://{self.sources.bucket}/")
            s3 = self._get_s3()
            img_bin = s3.get_object(Bucket=self.sources.bucket, Key=key)["Body"].read()
            img = Image.open(io.BytesIO(img_bin)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")

        if rec["orig_size"] == (0, 0):
            W, H = img.size
            rec["orig_size"] = (H, W)

        # json round-trip = cheap deep copy to avoid in-place modifications
        return self.prepare(img, json.loads(json.dumps(rec)))

    def __getstate__(self):
        # prevent a live client/socket pool from being pickled into workers
        state = self.__dict__.copy()
        state["_s3"] = None
        state.pop("s3", None)
        return state

    def _get_s3(self):
        existing = getattr(self, "_s3", None) or getattr(self, "s3", None)
        if existing is not None:
            self._s3 = existing
            return self._s3
        self._s3 = boto3.client("s3", config=_S3_CFG)
        return self._s3

    # ────────────── Annotation builder ─────────────
    def _build_annotations(self) -> Dict[int, Dict]:
        """
        Scan S3 once, parse Pascal VOC XML → xyxy bboxes.

        Object class names are taken exactly from <object><name> in the XML
        (after .strip()) and assigned contiguous ids [1..K] based on first
        appearance. Captions are left empty ("") and filled by GeoChat at eval.
        """
        s3 = boto3.client("s3", config=_S3_CFG)
        paginator = s3.get_paginator("list_objects_v2")
        xml_objs: List[dict] = [
            obj
            for page in paginator.paginate(
                Bucket=self.sources.bucket, Prefix=self.sources.labels_prefix
            )
            for obj in page.get("Contents", [])
            if obj["Key"].lower().endswith(".xml")
        ]

        ann: Dict[int, Dict] = {}
        img_id = 0

        # Single tqdm over all XML files
        for obj in tqdm(xml_objs, desc="Parsing DIOR XML", unit="file"):
            xml_key = obj["Key"]
            try:
                xml_bytes = (
                    s3.get_object(Bucket=self.sources.bucket, Key=xml_key)["Body"]
                    .read()
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code == "NoSuchKey":
                    continue
                raise

            try:
                root = ET.fromstring(xml_bytes)
            except ET.ParseError:
                continue

            # filename
            filename = root.findtext("filename")
            if not filename:
                filename = f"{Path(xml_key).stem}.png"

            # image size from XML (optional; we still clamp with real W/H at __getitem__)
            size_node = root.find("size")
            if size_node is not None:
                try:
                    w = int(size_node.findtext("width") or 0)
                    h = int(size_node.findtext("height") or 0)
                except Exception:
                    w = h = 0
            else:
                w = h = 0

            img_s3_path = f"s3://{self.sources.bucket}/{self.sources.images_prefix}{filename}"
            rec = dict(
                image_id=img_id,
                image_path=img_s3_path,
                caption="",  # VLM fills at test-time
                filename_stem=Path(filename).stem,
                orig_size=(h, w),
                annotations=[],
            )

            # parse <object> nodes
            for obj_node in root.findall("object"):
                raw_name = obj_node.findtext("name")
                if not raw_name:
                    continue

                cls_name = raw_name.strip()
                if cls_name not in CLASS_TO_IDX:
                    cid = len(CLASS_TO_IDX) + 1
                    CLASS_TO_IDX[cls_name] = cid
                    IDX_TO_CLASS[cid] = cls_name

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

                if w > 0 and h > 0:
                    x0 = max(0.0, min(x1, x2))
                    y0 = max(0.0, min(y1, y2))
                    x1c = min(w, max(x1, x2))
                    y1c = min(h, max(y1, y2))
                else:
                    x0 = min(x1, x2)
                    y0 = min(y1, y2)
                    x1c = max(x1, x2)
                    y1c = max(y1, y2)

                if (x1c - x0) < 2 or (y1c - y0) < 2:
                    # tiny / degenerate box or clearly broken XML
                    continue

                area = float(max(0.0, x1c - x0) * max(0.0, y1c - y0))
                rec["annotations"].append(
                    dict(
                        bbox=[x0, y0, x1c, y1c],
                        area=area,
                        category_id=CLASS_TO_IDX[cls_name],
                        iscrowd=0,
                    )
                )

            if rec["annotations"]:
                if rec["orig_size"] == (0, 0):
                    ys = [a["bbox"][1] for a in rec["annotations"]] + [
                        a["bbox"][3] for a in rec["annotations"]
                    ]
                    xs = [a["bbox"][0] for a in rec["annotations"]] + [
                        a["bbox"][2] for a in rec["annotations"]
                    ]
                    rec["orig_size"] = (int(max(ys) + 1), int(max(xs) + 1))

                ann[img_id] = rec
                img_id += 1

        return ann


# ───────────────────────── transforms ─────────────────────────
def _dior_transforms(split: str):
    """
    Simple transforms for DIOR.

    For this zero-shot evaluation setting, we only really need the 'val' branch:
    - Resize shorter side to 704 (max_size=704)
    - ToTensor + Normalize
    """
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if split in ("train", "val", "test"):
        # deterministic resize + normalize; no random augs
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    else:
        raise ValueError(split)


class _WrappedDataset(torch.utils.data.Dataset):
    """Attach transforms while preserving .coco for evaluator."""

    def __init__(self, base: DiorModulatedDetection, tfm):
        self.base, self.tfm, self.coco = base, tfm, base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        img, tgt = self.tfm(img, tgt)
        return img, tgt


# ───────────────────── dataset builders ─────────────────────
def build_dior(set_name: str, args):
    """
    set_name ∈ {"train", "val", "test"}.

    For zero-shot experiments we typically call this with set_name="val" or "test".
    Captions are expected to be provided by an external VLM at inference time.
    """
    full = DiorModulatedDetection(
        tokenizer=getattr(args, "tokenizer", None),
        return_tokens=False,
        sources=_S3Sources(
            bucket=getattr(args, "s3_bucket", "research-geodatasets"),
            images_prefix=getattr(args, "images_prefix", "DIOR/images/"),
            labels_prefix=getattr(args, "labels_prefix", "DIOR/annotations/"),
        ),
        seed=getattr(args, "seed", 42),
    )

    full.ids = full.ids_all

    if set_name == "train":
        return _WrappedDataset(full, _dior_transforms("train"))
    if set_name in ("val", "test"):
        return _WrappedDataset(full, _dior_transforms("val"))

    raise ValueError(set_name)


# ───────────────────── convert-to-target ─────────────────────
class _ConvertDiorToTarget:
    """
    Minimal converter: no positive maps, no GPT augmentation.
    Just boxes, labels, basic metadata.
    """

    def __init__(self):
        pass

    def __call__(self, image, record: Dict):
        W, H = image.size  # PIL: (width, height)

        original_annotations = record["annotations"]
        if len(original_annotations) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(
                [a["bbox"] for a in original_annotations], dtype=torch.float32
            )
            # clamp to actual image size
            boxes[:, 0::2].clamp_(0, W)
            boxes[:, 1::2].clamp_(0, H)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]

            labels_1b = torch.tensor(
                [a["category_id"] for a in original_annotations], dtype=torch.int64
            )[keep]
            labels = labels_1b - 1  # 0-based for model

            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.tensor(
                [a.get("iscrowd", 0) for a in original_annotations], dtype=torch.int64
            )[keep]

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=torch.tensor([record["image_id"]]),
            area=area,
            iscrowd=iscrowd,
            orig_size=torch.tensor([record["orig_size"][0], record["orig_size"][1]]),
            size=torch.tensor([H, W]),
            caption=record.get("caption", ""),
            filename_stem=record.get("filename_stem", "N/A"),
        )

        return image, target
