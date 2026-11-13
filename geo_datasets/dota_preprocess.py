# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Ship/plane DOTA loading, caption supervision, and transforms for local or S3 data."""

from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from PIL import Image
from pycocotools.coco import COCO
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from tqdm import tqdm

import mdetr.datasets.transforms as T
from util.augs import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomRotate90,
    RandomVerticalFlip,
)

if TYPE_CHECKING:
    import pyarrow as pa

load_dotenv()

# Initialize the optional client only when caption generation is requested.
_openai = None
_OPENAI_INITIALIZED = False
_OPENAI_WARNED = False

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_CACHE_DIR = _PARENT_DIR / "dota_cache"

ARROW_CACHE = _CACHE_DIR / "dota_annotations.parquet"
CLASSES_JSON = _CACHE_DIR / "dota_classes.json"

CLASS_NAMES = ["ship", "plane"]
CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

CAPTION_KEYS = {
    "ship": ["ship", "ships"],
    "plane": ["plane", "planes"],
}

# Split caption lists of the form "1. ... 2. ...".
_SPLIT_RE = re.compile(r"\d+\.\s*")


def _split(txt: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(txt or "") if p.strip()]


def _evaluation_caption(txt: str) -> str:
    """Choose the first numbered caption (or the original unnumbered caption)."""
    captions = _split(txt)
    return captions[0] if captions else (txt or "").strip()


_CAP_PROMPT = (
    "Generate NEW sentences that lightly paraphrase the Existing text.\n"
    "\n"
    "OBLIGATORY RULES (follow ALL):\n"
    "1) Let ALLOWED be the set of words among {'plane','planes','ship','ships'} that appear in the Existing text (case-insensitive).\n"
    "2) You MUST use only nouns from ALLOWED. Do NOT use any other class noun. If ALLOWED={'plane'}, both outputs must use 'plane' and must NOT mention 'planes','ship','ships'. If ALLOWED={'planes','ships'}, you may use only those two, preserving plurality.\n"
    "3) Do NOT use synonyms for these nouns. Banned words include: aircraft, airplane, aeroplane, jet, craft, vessel, watercraft, boat (and their plurals/case variants).\n"
    "4) Preserve plurality EXACTLY as in the Existing text: if it uses 'plane', you must use 'plane'; if it uses 'planes', you must use 'planes'; similarly for ship/ships.\n"
    "5) Do NOT add or remove mentions of these nouns beyond what is present in the Existing text. Do NOT introduce new objects, numbers, locations, or facts.\n"
    "6) Keep the meaning and structure close to the Existing text; only lightly rephrase other words (use short synonyms) without changing meaning.\n"
    "7) Each output line MUST be a complete grammatical sentence that starts with a capital letter and ends with a period.\n"
    "8) Output EXACTLY the lines, with NO numbering, bullets, dashes, or extra headers—just the sentences on separate lines.\n"
)


def _call_gpt(existing: List[str], need: int) -> List[str]:
    """Generate up to `need` new caption sentences (best-effort)."""
    if need <= 0:
        return []

    global _openai, _OPENAI_INITIALIZED, _OPENAI_WARNED
    if not _OPENAI_INITIALIZED:
        _OPENAI_INITIALIZED = True
        try:
            from openai import OpenAI

            _openai = OpenAI()
        except Exception as exc:
            _OPENAI_WARNED = True
            print(f"[DOTA] Caption augmentation unavailable: {exc}")
    if _openai is None:
        return []

    user_msg = (
        _CAP_PROMPT
        + "\n\nHere are the existing sentences:\n"
        + "\n".join(f"- {s}" for s in existing)
    )

    try:
        resp = _openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.1,
        )
    except Exception as e:
        if not _OPENAI_WARNED:
            _OPENAI_WARNED = True
            print(f"[DOTA] GPT caption augmentation failed; continuing without augmentation: {e}")
        return []

    raw = resp.choices[0].message.content or ""
    line_prefix = re.compile(r"^\s*(?:\d+\s*[.)-]?\s*|[-\u2013\u2014*\u2022]+\s+)")
    lines = [line_prefix.sub("", line).strip() for line in raw.splitlines() if line.strip()]
    return lines[:need]


def create_positive_map(tokenized, tokens_positive):
    """Build a normalized (objects, 256) token map from character spans."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for beg, end in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None

            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None

            if beg_pos is None or end_pos is None:
                continue

            positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def _find_one_of(caption: str, candidates: List[str]) -> Optional[Tuple[int, int, str]]:
    """Find the first case-insensitive match of any candidate; returns (start, end, matched_text)."""
    if not caption:
        return None
    lower = caption.lower()
    for w in candidates:
        idx = lower.find(w)
        if idx != -1:
            return (idx, idx + len(w), caption[idx : idx + len(w)])
    return None


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]


def _parse_dota_hbb_text(raw_txt: str, width: int, height: int) -> List[Dict]:
    """Parse original DOTA HBB rows: class_name x1 y1 x2 y2."""
    annotations = []
    for ln in raw_txt.splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls_raw = parts[0].lower()
        if cls_raw not in CLASS_TO_IDX:
            continue
        try:
            x1, y1, x2, y2 = map(float, parts[1:5])
        except ValueError:
            continue

        x0 = max(0.0, min(x1, x2))
        y0 = max(0.0, min(y1, y2))
        x1c = min(width, max(x1, x2))
        y1c = min(height, max(y1, y2))
        if (x1c - x0) < 2 or (y1c - y0) < 2:
            continue

        annotations.append(
            dict(
                bbox=[x0, y0, x1c, y1c],
                area=float((x1c - x0) * (y1c - y0)),
                category_id=CLASS_TO_IDX[cls_raw],
                phrase=cls_raw,
                iscrowd=0,
            )
        )
    return annotations


def _dict_to_table(d: Dict[int, Dict]) -> pa.Table:
    import pyarrow as pa

    rows = []
    for i, rec in d.items():
        for a in rec["annotations"]:
            rows.append(
                dict(
                    img_id=i,
                    stem=rec["filename_stem"],
                    path=rec["image_path"],
                    caption=rec["caption"],
                    bbox=a["bbox"],
                    area=a["area"],
                    cat_id=a["category_id"],
                    phrase=a.get("phrase", ""),
                    iscrowd=a["iscrowd"],
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
                caption=r["caption"],
                filename_stem=r["stem"],
                orig_size=(r["orig_h"], r["orig_w"]),
                annotations=[],
            ),
        )
        out[i]["annotations"].append(
            dict(
                bbox=r["bbox"],
                area=r["area"],
                category_id=r["cat_id"],
                phrase=r.get("phrase", None),
                iscrowd=r["iscrowd"],
            )
        )
    return out


def _build_coco_like(ann_dict: Dict[int, Dict]) -> Dict:
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
                    iscrowd=a["iscrowd"],
                )
            )
            aid += 1

    cats = [{"id": cid, "name": IDX_TO_CLASS[cid]} for cid in sorted(IDX_TO_CLASS)]
    return {
        "images": imgs,
        "annotations": anns,
        "categories": cats,
        "info": {},
        "licenses": [],
    }


@dataclass(frozen=True)
class _S3Sources:
    bucket: str
    images_prefix: str
    labels_prefix: str
    csv_key: str
    dataset_format: str = "s3"


def _get_cfg_str(
    args,
    arg_key: str,
    env_keys: List[str],
) -> str:
    """Resolve a required string from arguments, then environment variables."""
    v = None
    if args is not None:
        v = getattr(args, arg_key, None)

    if isinstance(v, str):
        v = v.strip()
    if v is not None and str(v).strip():
        return str(v).strip()

    for ek in env_keys:
        ev = os.getenv(ek)
        if ev is not None and str(ev).strip():
            return str(ev).strip()

    raise ValueError(
        f"[DOTA] Missing required config for '{arg_key}'. "
        f"Provide args.{arg_key} or set one of: {', '.join(env_keys)} (e.g. via .env)."
    )


def _get_arg_float(args, key: str, default: float) -> float:
    v = getattr(args, key, None)
    if v is None:
        return float(default)
    return float(v)


def _get_arg_int(args, key: str, default: int) -> int:
    v = getattr(args, key, None)
    if v is None:
        return int(default)
    return int(v)


def _local_path(value: str, *, is_dir: bool, key: str) -> str:
    """Resolve local config paths relative to the repository, not the shell's CWD."""
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = _PARENT_DIR / path
    path = path.resolve()

    if is_dir and not path.is_dir():
        raise FileNotFoundError(f"[DOTA] {key} directory does not exist: {path}")
    if not is_dir and not path.is_file():
        raise FileNotFoundError(f"[DOTA] {key} file does not exist: {path}")
    return str(path)


def _sources_from_args(args, set_name: str = "train") -> _S3Sources:
    """Resolve local/S3 sources; test requires its own paths, not training fallbacks."""
    if set_name not in {"train", "val", "test"}:
        raise ValueError(f"[DOTA] Invalid split: {set_name}")

    mode = (
        str(getattr(args, "dataset_format", None) or os.getenv("DOTA_DATASET_FORMAT", "s3"))
        .strip()
        .lower()
    )
    if mode not in {"s3", "local"}:
        raise ValueError("[DOTA] dataset_format must be 's3' or 'local'")

    test = set_name == "test"
    key_prefix = "test_" if test else ""
    env_prefix = ("DOTA_LOCAL_" if mode == "local" else "DOTA_S3_") + ("TEST_" if test else "")

    images_key = key_prefix + "images_prefix"
    labels_key = key_prefix + "labels_prefix"
    csv_key = key_prefix + "csv_key"
    images = _get_cfg_str(args, images_key, [env_prefix + "IMAGES_PREFIX"])
    labels = _get_cfg_str(args, labels_key, [env_prefix + "LABELS_PREFIX"])
    csv_path = _get_cfg_str(args, csv_key, [env_prefix + "CSV_KEY"])

    if mode == "local":
        images = _local_path(images, is_dir=True, key=images_key)
        labels = _local_path(labels, is_dir=True, key=labels_key)
        csv_path = _local_path(csv_path, is_dir=False, key=csv_key)
        bucket = ""
    else:
        bucket = _get_cfg_str(args, "bucket", ["DOTA_S3_BUCKET"])

    return _S3Sources(
        bucket=bucket,
        images_prefix=images,
        labels_prefix=labels,
        csv_key=csv_path,
        dataset_format=mode,
    )


class DotaModulatedDetection(Dataset):
    """DOTA annotations, caption sampling, and local/S3 image loading."""

    def __init__(
        self,
        tokenizer=None,
        return_tokens: bool = False,
        augment_train: bool = False,
        sources: Optional[_S3Sources] = None,
        val_split_ratio: float = 0.2,
        seed: int = 42,
        set_name: str = "train",
    ):
        super().__init__()
        self._s3 = None
        if set_name not in {"train", "val", "test"}:
            raise ValueError(f"[DOTA] Invalid split: {set_name}")

        self.sources = sources if sources is not None else _sources_from_args(args=None)

        self.seed = seed
        self.prepare = _ConvertDotaToTarget(tokenizer, return_tokens)

        # Local data must not reuse annotations from the legacy S3 cache.
        if self.sources.dataset_format == "local":
            cache_table = cache_classes = None
        elif set_name == "test":
            source_id = json.dumps(self.sources.__dict__, sort_keys=True)
            fingerprint = hashlib.sha256(source_id.encode()).hexdigest()[:16]
            cache_table = _CACHE_DIR / f"dota_test_{fingerprint}.parquet"
            cache_classes = _CACHE_DIR / f"dota_test_{fingerprint}_classes.json"
        else:
            # Preserve the old S3 training cache for existing installations.
            cache_table, cache_classes = ARROW_CACHE, CLASSES_JSON

        cache_found = cache_table is not None and cache_table.exists() and cache_classes.exists()
        if cache_found:
            import pyarrow.parquet as pq

            ann_raw = _table_to_dict(pq.read_table(cache_table))
        else:
            ann_raw = self._build_annotations()

        self.annotations = ann_raw
        self.ids_all = list(ann_raw)
        self.ids = self.ids_all

        if set_name == "test":
            # Test data is independent and never augmented or split.
            self.ids_train, self.ids_val = [], []
            self.ids_all = self._one_caption_per_chip(self.ids_all)
            self.ids = self.ids_all
        else:
            self.ids_train, self.ids_val = self._split_groups(ann_raw, val_split_ratio)
            if not self.ids_val:
                n = max(1, int(len(self.ids_all) * val_split_ratio))
                self.ids_val, self.ids_train = self.ids_all[:n], self.ids_all[n:]

            if set_name == "val":
                self.ids_val = self._one_caption_per_chip(self.ids_val)
            elif not cache_found:
                before = set(ann_raw)
                if augment_train:
                    self._augment_captions_once(ann_raw, only_ids=set(self.ids_train))
                    new_train_ids = sorted(set(ann_raw) - before)
                else:
                    # Training still uses every numbered caption as its own sample.
                    new_train_ids = self._explode_captions_no_gpt(
                        ann_raw, only_ids=set(self.ids_train)
                    )
                self.ids_train = list(self.ids_train) + new_train_ids

        if not cache_found and cache_table is not None:
            import pyarrow.parquet as pq

            cache_table.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(_dict_to_table(ann_raw), cache_table, compression="zstd")
            cache_classes.write_text(json.dumps(CLASS_TO_IDX))

        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
        self.coco.createIndex()

    def _one_caption_per_chip(self, ids: List[int]) -> List[int]:
        """Keep the lowest-ID copy of each chip and its first numbered caption."""
        chosen = {}
        for image_id in sorted(ids):
            stem = self.annotations[image_id]["filename_stem"]
            chosen.setdefault(stem, image_id)

        selected = list(chosen.values())
        for image_id in selected:
            rec = self.annotations[image_id]
            rec["caption"] = _evaluation_caption(rec.get("caption", ""))
        return selected

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        if self.sources.dataset_format == "local":
            with Image.open(rec["image_path"]) as im:
                img = im.convert("RGB")
        else:
            key = rec["image_path"].removeprefix(f"s3://{self.sources.bucket}/")
            s3 = self._get_s3()
            img_bin = s3.get_object(Bucket=self.sources.bucket, Key=key)["Body"].read()
            with Image.open(io.BytesIO(img_bin)) as im:
                img = im.convert("RGB")

        if rec["orig_size"] == (0, 0):
            rec["orig_size"] = img.size[::-1]

        return self.prepare(img, copy.deepcopy(rec))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3"] = None
        return state

    def _get_s3(self):
        if self.sources.dataset_format == "local":
            raise RuntimeError("[DOTA] Local datasets must not open an S3 connection")
        if getattr(self, "_s3", None) is None:
            import boto3
            from botocore.config import Config

            self._s3 = boto3.client(
                "s3",
                config=Config(
                    retries={"max_attempts": 10, "mode": "adaptive"},
                    connect_timeout=10,
                    read_timeout=60,
                ),
            )
        return self._s3

    def _build_annotations(self) -> Dict[int, Dict]:
        """Build an annotation dict from the selected backend (TXT HBB -> xyxy)."""
        if self.sources.dataset_format == "local":
            return self._build_annotations_local()

        from botocore.exceptions import ClientError

        s3 = self._get_s3()

        csv_rows = list(
            csv.DictReader(
                s3.get_object(Bucket=self.sources.bucket, Key=self.sources.csv_key)["Body"]
                .read()
                .decode()
                .splitlines()
            )
        )
        info_by_stem = {Path(r["Image"]).stem: r for r in csv_rows}

        paginator = s3.get_paginator("list_objects_v2")
        txt_objs = [
            obj
            for page in paginator.paginate(
                Bucket=self.sources.bucket, Prefix=self.sources.labels_prefix
            )
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".txt")
        ]

        ann, img_id = {}, 0
        for obj in tqdm(txt_objs, desc="Parsing DOTA TXT", unit="file"):
            stem = Path(obj["Key"]).stem
            if stem not in info_by_stem:
                continue

            img_s3_path = info_by_stem[stem]["Image"]
            caption_text = info_by_stem[stem].get("Description", "") or ""

            key_img = img_s3_path.removeprefix(f"s3://{self.sources.bucket}/")
            try:
                img_bin = s3.get_object(Bucket=self.sources.bucket, Key=key_img)["Body"].read()
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") != "NoSuchKey":
                    raise
                fallback_key = f"{self.sources.images_prefix}{stem}.png"
                img_bin = s3.get_object(Bucket=self.sources.bucket, Key=fallback_key)["Body"].read()
                img_s3_path = f"s3://{self.sources.bucket}/{fallback_key}"

            with Image.open(io.BytesIO(img_bin)) as im:
                W, H = im.size

            rec = dict(
                image_id=img_id,
                image_path=img_s3_path,
                caption=caption_text,
                filename_stem=stem,
                orig_size=(H, W),
                annotations=[],
            )

            raw_txt = (
                s3.get_object(Bucket=self.sources.bucket, Key=obj["Key"])["Body"].read().decode()
            )

            rec["annotations"] = _parse_dota_hbb_text(raw_txt, W, H)

            if rec["annotations"]:
                ann[img_id] = rec
                img_id += 1

        return ann

    def _build_annotations_local(self) -> Dict[int, Dict]:
        """Read the local CSV, image files, and DOTA HBB .txt labels; never call S3."""
        csv_path = Path(self.sources.csv_key)
        images_dir = Path(self.sources.images_prefix)
        labels_dir = Path(self.sources.labels_prefix)

        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"Image", "Description"}.issubset(reader.fieldnames):
                raise ValueError(f"[DOTA] CSV needs 'Image' and 'Description' columns: {csv_path}")
            info_by_stem = {}
            for row_number, row in enumerate(reader, start=2):
                image_reference = (row.get("Image") or "").strip()
                if not image_reference:
                    raise ValueError(f"[DOTA] Empty Image field at {csv_path}:{row_number}")
                if not (row.get("Description") or "").strip():
                    raise ValueError(f"[DOTA] Empty Description at {csv_path}:{row_number}")
                stem = Path(image_reference.replace("\\", "/")).stem
                if stem in info_by_stem:
                    raise ValueError(
                        f"[DOTA] Duplicate CSV image stem '{stem}' at line {row_number}; "
                        "expected one row per chip with captions in Description"
                    )
                info_by_stem[stem] = row

        image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        images_by_stem = {}
        for image_path in images_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in image_suffixes:
                continue
            if image_path.stem in images_by_stem:
                raise ValueError(
                    f"[DOTA] Duplicate local image stem '{image_path.stem}': "
                    f"{images_by_stem[image_path.stem]} and {image_path}"
                )
            images_by_stem[image_path.stem] = image_path

        label_paths = sorted(
            path
            for path in labels_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        if not label_paths:
            raise ValueError(f"[DOTA] No .txt annotations found beneath {labels_dir}")

        ann = {}
        seen_stems = set()
        for label_path in tqdm(label_paths, desc="Parsing local DOTA TXT", unit="file"):
            stem = label_path.stem
            if stem not in info_by_stem:
                continue
            if stem in seen_stems:
                raise ValueError(f"[DOTA] Multiple annotation files for {stem}")
            seen_stems.add(stem)

            image_path = images_by_stem.get(stem)
            if image_path is None:
                raise FileNotFoundError(
                    f"[DOTA] CSV and annotation refer to {stem}, but no image "
                    f"with this stem exists under {images_dir}"
                )

            with Image.open(image_path) as im:
                width, height = im.size

            annotations = _parse_dota_hbb_text(
                label_path.read_text(encoding="utf-8-sig"), width, height
            )
            if not annotations:
                continue

            img_id = len(ann)
            ann[img_id] = dict(
                image_id=img_id,
                image_path=str(image_path),
                caption=info_by_stem[stem].get("Description", "") or "",
                filename_stem=stem,
                orig_size=(height, width),
                annotations=annotations,
            )

        if not ann:
            raise ValueError(
                f"[DOTA] No usable labeled chips matched between {csv_path}, "
                f"{images_dir}, and {labels_dir}. Check the config paths and filenames."
            )

        print(
            f"[DOTA] Local files: matched {len(ann)} labeled chips from "
            f"{len(info_by_stem)} CSV rows, no S3 requests"
        )
        return ann

    def _explode_captions_no_gpt(self, ann: Dict[int, Dict], only_ids=None) -> list[int]:
        """Split enumerated captions into one-caption-per-sample (no augmentation)."""
        if not ann:
            return []
        ids = list(only_ids) if only_ids is not None else list(ann.keys())
        nxt = (max(ann) + 1) if ann else 0
        new_ids: list[int] = []

        for img_id in ids:
            base = ann[img_id]
            sents = _split(base.get("caption", "") or "")
            if len(sents) <= 1:
                continue
            base["caption"] = sents[0]
            for s in sents[1:]:
                dup = base.copy()
                dup["annotations"] = [a.copy() for a in base["annotations"]]
                dup["caption"] = s
                dup["image_id"] = nxt
                ann[nxt] = dup
                new_ids.append(nxt)
                nxt += 1
        return new_ids

    def _augment_captions_once(self, ann, only_ids=None):
        """Top up per-image caption lists via GPT to a target count (best-effort)."""
        if not ann:
            return
        image_ids = [image_id for image_id in ann if only_ids is None or image_id in only_ids]
        nxt = max(ann) + 1
        for img_id in image_ids:
            base = ann[img_id]
            caps = list(dict.fromkeys(_split(base["caption"])))
            if not caps:
                continue
            seen = set(caps)
            for _ in range(3):
                if len(caps) >= 5:
                    break
                generated = _call_gpt(caps, 5 - len(caps))
                added_any = False
                for caption in generated:
                    if caption and caption not in seen:
                        caps.append(caption)
                        seen.add(caption)
                        added_any = True
                if not added_any:
                    break

            base["caption"] = caps[0]
            for extra in caps[1:]:
                dup = base.copy()
                dup["annotations"] = [a.copy() for a in base["annotations"]]
                dup["caption"] = extra
                dup["image_id"] = nxt
                ann[nxt] = dup
                nxt += 1

    def _split_groups(self, ann_dict, val_ratio):
        """Group-aware train/val split (keeps stems together)."""
        ids = list(ann_dict)
        stems = [ann_dict[i]["filename_stem"] for i in ids]
        tr, vl = next(
            GroupShuffleSplit(train_size=1 - val_ratio, random_state=self.seed).split(
                ids, groups=stems
            )
        )
        return [ids[i] for i in tr], [ids[i] for i in vl]


def _dota_transforms(split: str):
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
                RandomColorJitter(
                    prob=0.25, brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
                ),
                RandomGaussianBlur(prob=0.03, radius=(0.00, 0.15)),
                T.RandomResize([704], max_size=704),
                normalize,
            ]
        )
    if split == "val":
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    raise ValueError(split)


class _WrappedDataset(torch.utils.data.Dataset):
    """Apply transforms on top of a DotaModulatedDetection dataset."""

    def __init__(self, base: DotaModulatedDetection, tfm):
        self.base, self.tfm, self.coco = base, tfm, base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        img, tgt = self.tfm(img, tgt)
        return img, tgt


def build_dota(set_name: str, args):
    """Expand training captions; use one caption per chip for validation/test."""
    sources = _sources_from_args(args, set_name)
    use_augmentation = (
        bool(getattr(args, "augment_train", sources.dataset_format == "s3")) and set_name == "train"
    )
    print(
        f"[DOTA] {set_name}: source={sources.dataset_format} "
        f"CSV={sources.csv_key} images={sources.images_prefix} "
        f"labels={sources.labels_prefix}"
    )

    full = DotaModulatedDetection(
        tokenizer=args.tokenizer,
        return_tokens=True,
        val_split_ratio=_get_arg_float(args, "val_split_ratio", 0.2),
        seed=_get_arg_int(args, "seed", 42),
        augment_train=use_augmentation,
        sources=sources,
        set_name=set_name,
    )

    if set_name == "train":
        full.ids = full.ids_train
        return _WrappedDataset(full, _dota_transforms("train"))

    if set_name == "val":
        full.ids = full.ids_val
        return _WrappedDataset(full, _dota_transforms("val"))

    if set_name == "test":
        full.ids = full.ids_all
        return _WrappedDataset(full, _dota_transforms("val"))

    raise ValueError(set_name)


class _ConvertDotaToTarget:
    """Convert a DOTA record into the MDETR-style target dict (with optional token alignment)."""

    def __init__(self, tokenizer=None, return_tokens: bool = False):
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer

    def __call__(self, image, record: Dict):
        W, H = image.size

        original_annotations = record["annotations"]
        boxes = torch.as_tensor(
            [a["bbox"] for a in original_annotations], dtype=torch.float32
        ).reshape(-1, 4)
        boxes[:, 0::2].clamp_(0, W)
        boxes[:, 1::2].clamp_(0, H)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        labels_1b = torch.tensor(
            [a["category_id"] for a in original_annotations], dtype=torch.int64
        )[keep]
        labels = labels_1b - 1

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.tensor([a["iscrowd"] for a in original_annotations], dtype=torch.int64)[
            keep
        ]

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=torch.tensor([record["image_id"]]),
            area=area,
            iscrowd=iscrowd,
            orig_size=torch.tensor([H, W]),
            size=torch.tensor([H, W]),
            caption=record["caption"],
            filename_stem=record.get("filename_stem", "N/A"),
        )

        if self.return_tokens and self.tokenizer is not None:
            caption = record.get("caption") or ""
            tokens_positive = []
            phrases_searched = []

            kept_annotations = [ann for i, ann in enumerate(original_annotations) if keep[i]]
            for annotation in kept_annotations:
                cat_id = int(annotation["category_id"])
                cls_name = IDX_TO_CLASS.get(cat_id, None)
                phrases_searched.append(cls_name)

                if cls_name in CAPTION_KEYS:
                    span = _find_one_of(caption, CAPTION_KEYS[cls_name])
                    if span is not None:
                        (beg, end, matched_text) = span
                        annotation["phrase"] = matched_text
                        tokens_positive.append([(beg, end)])
                    else:
                        tokens_positive.append([])
                else:
                    tokens_positive.append([])

            tokenized = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256,
                return_offsets_mapping=True,
            )
            positive_map = create_positive_map(tokenized, tokens_positive)

            target["tokens_positive"] = tokens_positive
            target["phrases_searched"] = phrases_searched
            target["tokens"] = tokens_positive
            target["positive_map"] = positive_map

        return image, target
