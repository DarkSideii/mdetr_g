"""
DOTA dataset loader/preprocessing for MDETR-style training.

Key behaviors:
- Reads labels/images from S3 and builds a COCO-like view for evaluation.
- Caches parsed annotations to a local Parquet file for faster startup.
- Optionally augments caption lists via GPT (best-effort; training should not crash).
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from dataclasses import dataclass
from botocore.config import Config
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from openai import OpenAI
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

# Global setup.
load_dotenv()
random.seed(42)

# OpenAI is optional; if unavailable, caption augmentation is disabled.
_OPENAI_WARNED = False
try:
    _openai = OpenAI()
except Exception as e:
    _openai = None
    _OPENAI_WARNED = True
    print(f"[DOTA] OpenAI client disabled (no API key or init error): {e}")

_S3_CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
)

# Cache paths.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_CACHE_DIR = _PARENT_DIR / "dota_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARROW_CACHE = _CACHE_DIR / "dota_annotations.parquet"
CLASSES_JSON = _CACHE_DIR / "dota_classes.json"

# Dataset settings.
CLASS_NAMES = ["ship", "plane"]
CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}  # contiguous [1..K]
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Caption keyword variants used for token alignment.
CAPTION_KEYS = {
    "ship": ["ship", "ships"],
    "plane": ["plane", "planes"],
}

# Split caption lists of the form "1. ... 2. ...".
_SPLIT_RE = re.compile(r"\d+\.\s*")
def _split(txt: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(txt or "") if p.strip()]

# Caption augmentation prompt (kept verbatim).
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

def _split_sents(txt: str) -> List[str]:
    parts = re.split(r"\d+\.\s*", txt or "")
    return [p.strip() for p in parts if p.strip()]

def _call_gpt(existing: List[str], need: int) -> List[str]:
    """Generate up to `need` new caption sentences (best-effort)."""
    if need <= 0:
        return []

    global _OPENAI_WARNED
    if _openai is None:
        if not _OPENAI_WARNED:
            _OPENAI_WARNED = True
            print("[DOTA] OpenAI client is None -> skipping GPT caption augmentation.")
        return []

    prefix = (_CAP_PROMPT + "\n\n") if _CAP_PROMPT else ""
    user_msg = prefix + "Here are the existing sentences:\n" + "\n".join(f"- {s}" for s in existing)

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

    raw = resp.choices[0].message.content
    lines = [
        re.compile(r"^\s*(?:\d+\s*[.)-]?\s*|[-\u2013\u2014*\u2022]+\s+)").sub("", l).strip()
        for l in raw.split("\n")
        if l.strip()
    ]
    return lines[:need]

# Token alignment helpers.
def create_positive_map(tokenized, tokens_positive):
    """Build a (boxes x tokens) alignment map in the tokenizer's max_length space."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
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
            return (idx, idx + len(w), caption[idx:idx + len(w)])
    return None

# PyArrow helpers for cached annotations.
def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]

def _dict_to_table(d: Dict[int, Dict]) -> pa.Table:
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
    return {"images": imgs, "annotations": anns, "categories": cats, "info": {}, "licenses": []}

@dataclass
class _S3Sources:
    bucket: str = "research-geodatasets"
    images_prefix: str = "DOTAv2/images/"
    labels_prefix: str = "DOTAv2/labels/"
    csv_key: str = "DOTAv2/dota_sentencesv2.csv"

def _get_arg_str(args, key: str, default: str) -> str:
    v = getattr(args, key, None)
    if v is None:
        return str(default)
    if isinstance(v, str) and not v.strip():
        return str(default)
    return str(v)

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

def _sources_from_args(args) -> _S3Sources:
    """Build S3 source config from args (bucket/prefix/key only)."""
    defaults = _S3Sources()
    return _S3Sources(
        bucket=_get_arg_str(args, "bucket", defaults.bucket),
        images_prefix=_get_arg_str(args, "images_prefix", defaults.images_prefix),
        labels_prefix=_get_arg_str(args, "labels_prefix", defaults.labels_prefix),
        csv_key=_get_arg_str(args, "csv_key", defaults.csv_key),
    )

class DotaModulatedDetection(Dataset):
    """
    DOTA v2 dataset wrapper (S3-backed) with optional caption augmentation and caching.
    """
    def __init__(
        self,
        tokenizer=None,
        return_tokens: bool = False,
        augment_train: bool = False,
        sources: _S3Sources = _S3Sources(),
        val_split_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self._s3 = None
        self.sources = sources
        self.seed = seed
        self.prepare = _ConvertDotaToTarget(tokenizer, return_tokens)

        cache_found = ARROW_CACHE.exists() and CLASSES_JSON.exists()
        if cache_found:
            ann_raw = _table_to_dict(pq.read_table(ARROW_CACHE))
        else:
            ann_raw = self._build_annotations()

        self.annotations = ann_raw
        self.ids_all = list(ann_raw)
        self.ids = self.ids_all

        self.ids_train, self.ids_val = self._split_groups(ann_raw, val_split_ratio)
        if not self.ids_val:
            n = max(1, int(len(self.ids_all) * val_split_ratio))
            self.ids_val, self.ids_train = self.ids_all[:n], self.ids_all[n:]

        if not cache_found:
            new_val_ids = self._explode_captions_no_gpt(ann_raw, only_ids=set(self.ids_val))
            self.ids_val = list(self.ids_val) + new_val_ids

            if augment_train:
                before = set(ann_raw.keys())
                self._augment_captions_once(ann_raw, only_ids=set(self.ids_train))
                after = set(ann_raw.keys())
                new_train_ids = sorted(after - before)
                self.ids_train = list(self.ids_train) + new_train_ids

        if not cache_found:
            pq.write_table(_dict_to_table(ann_raw), ARROW_CACHE, compression="zstd")
            CLASSES_JSON.write_text(json.dumps(CLASS_TO_IDX))

        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
        self.coco.createIndex()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        key = rec["image_path"].removeprefix(f"s3://{self.sources.bucket}/")
        s3 = self._get_s3()
        img_bin = s3.get_object(Bucket=self.sources.bucket, Key=key)["Body"].read()
        img = Image.open(io.BytesIO(img_bin)).convert("RGB")

        if rec["orig_size"] == (0, 0):
            rec["orig_size"] = img.size[::-1]

        return self.prepare(img, json.loads(json.dumps(rec)))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3"] = None
        return state

    def _get_s3(self):
        if getattr(self, "_s3", None) is None:
            self._s3 = boto3.client("s3", config=_S3_CFG)
        return self._s3

    def _build_annotations(self) -> Dict[int, Dict]:
        """Scan S3 once and build an annotation dict (TXT HBB -> xyxy) keyed by image_id."""
        s3 = boto3.client("s3", config=_S3_CFG)

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
            for page in paginator.paginate(Bucket=self.sources.bucket, Prefix=self.sources.labels_prefix)
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
                s3.get_object(Bucket=self.sources.bucket, Key=obj["Key"])["Body"]
                .read()
                .decode()
            )

            for ln in raw_txt.splitlines():
                line = ln.strip()
                if not line:
                    continue
                parts = line.split()
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
                x1c = min(W, max(x1, x2))
                y1c = min(H, max(y1, y2))
                if (x1c - x0) < 2 or (y1c - y0) < 2:
                    continue

                rec["annotations"].append(
                    dict(
                        bbox=[x0, y0, x1c, y1c],
                        area=float((x1c - x0) * (y1c - y0)),
                        category_id=CLASS_TO_IDX[cls_raw],
                        phrase=cls_raw,
                        iscrowd=0,
                    )
                )

            if rec["annotations"]:
                ann[img_id] = rec
                img_id += 1

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
        rows = [
            {"img_id": k, "stem": v["filename_stem"], "caption": v["caption"]}
            for k, v in ann.items()
            if (only_ids is None or k in only_ids)
        ]
        ds = HFDataset.from_list(rows).remove_columns(["stem"])

        def _exp(batch):
            out_id, out_cap = [], []
            for img_id, cap in zip(batch["img_id"], batch["caption"]):
                sents_raw = _split(cap)
                seen, uniq_sents = set(), []
                for s in sents_raw:
                    if s and s not in seen:
                        uniq_sents.append(s)
                        seen.add(s)

                if 1 <= len(uniq_sents) < 5:
                    attempts = 0
                    while len(uniq_sents) < 5 and attempts < 3:
                        need = 5 - len(uniq_sents)
                        gens = _call_gpt(uniq_sents, need)
                        added_any = False
                        for g in gens:
                            if g and g not in seen:
                                uniq_sents.append(g)
                                seen.add(g)
                                added_any = True
                        if not added_any:
                            break
                        attempts += 1

                if len(uniq_sents) > 0:
                    out_id.extend([img_id] * len(uniq_sents))
                    out_cap.extend(uniq_sents)
            return {"img_id": out_id, "caption": out_cap}

        ds = ds.map(_exp, batched=True, batch_size=1000, remove_columns=[], load_from_cache_file=False)

        nxt = max(ann) + 1
        bucket: Dict[int, List[str]] = {}
        for img_id, cap in zip(ds["img_id"], ds["caption"]):
            bucket.setdefault(img_id, []).append(cap)

        for img_id, caps in bucket.items():
            base = ann[img_id]
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

# Transforms.
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
                RandomColorJitter(prob=0.25, brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                RandomGaussianBlur(prob=0.03, radius=(0.00, 0.15)),
                T.RandomResize([704], max_size=704),
                normalize,
            ]
        )
    elif split == "val":
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    else:
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
    """
    Build the requested split dataset.

    - train: uses ids_train (+ optional augmentation)
    - val:   uses ids_val
    - test:  uses ids_all (no split)
    """
    sources = _sources_from_args(args)

    full = DotaModulatedDetection(
        tokenizer=args.tokenizer,
        return_tokens=True,
        val_split_ratio=_get_arg_float(args, "val_split_ratio", 0.2),
        seed=_get_arg_int(args, "seed", 42),
        augment_train=(set_name == "train"),
        sources=sources,
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
        boxes = torch.as_tensor([a["bbox"] for a in original_annotations], dtype=torch.float32)
        boxes[:, 0::2].clamp_(0, W)
        boxes[:, 1::2].clamp_(0, H)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        labels_1b = torch.tensor([a["category_id"] for a in original_annotations], dtype=torch.int64)[keep]
        labels = labels_1b - 1

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.tensor([a["iscrowd"] for a in original_annotations], dtype=torch.int64)[keep]

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
