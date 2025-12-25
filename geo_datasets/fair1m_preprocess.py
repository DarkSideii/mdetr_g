from __future__ import annotations

import csv
import io
import json
import random
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import os
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

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

import multimodal_framework.mdetr.datasets.transforms as T
from util.augs import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomRotate90,
    RandomVerticalFlip,
    RandomGaussianNoise
)

# ───────────────────────── setup ─────────────────────────
load_dotenv()
random.seed(42)
_openai = OpenAI()

# S3 client config (retries + sane timeouts)
_S3_CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
    proxies={"http": None, "https": None},   # ← ignore HTTP(S)_PROXY in workers
    s3={"addressing_style": "virtual"},
)

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_CACHE_DIR = _PARENT_DIR / "fair1m_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARROW_CACHE = _CACHE_DIR / "fm_annotations.parquet"
CLASSES_JSON = _CACHE_DIR / "fm_classes.json"

# ─────────────────────── FAIR1M classes ───────────────────────
FAIR1M_CLASSES: List[str] = [
    "Other",
    "A220",
    "A321",
    "A330",
    "A350",
    "Boeing737",
    "Boeing747",
    "Boeing777",
    "Boeing787",
    "ARJ21",
    "C919",
]
# contiguous 1-based ids
CLASS_TO_IDX: Dict[str, int] = {name: i + 1 for i, name in enumerate(FAIR1M_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

# XML name → canonical class name mapping (case-insensitive for other-airplane)
XML_NAME_TO_CANONICAL: Dict[str, str] = {
    "other-airplane": "Other",
    # add any irregulars here if you see them
}

# ─────────────────────── misc helpers ───────────────────────
_SPLIT_RE = re.compile(r"\d+\.\s*")

def _split(txt: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(txt or "") if p.strip()]

def _xyxy_to_xywh(b: List[float]) -> List[float]:
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]

def _find_exact(caption: str, phrase: str) -> Optional[Tuple[int, int]]:
    if not caption or not phrase:
        return None
    i = caption.find(phrase)
    return (i, i + len(phrase)) if i != -1 else None

def create_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            if beg_pos is None:
                for off in (1, 2):
                    try:
                        beg_pos = tokenized.char_to_token(beg + off)
                        if beg_pos is not None:
                            break
                    except Exception:
                        pass

            if end_pos is None:
                for off in (2, 3):
                    try:
                        end_pos = tokenized.char_to_token(end - off)
                        if end_pos is not None:
                            break
                    except Exception:
                        pass

            if beg_pos is None or end_pos is None:
                continue
            positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def _split_s3_uri(uri: str) -> Tuple[str, str]:
    """
    's3://bucket/key/..' -> (bucket, 'key/..')
    Accepts already-key strings (no scheme) and returns (self.bucket, key) if needed.
    """
    u = uri.strip()
    if u.startswith("s3://"):
        u = u[5:]
        bucket, _, key = u.partition("/")
        return bucket, key
    # not a full URI; treat as key under default bucket (caller provides fallback)
    return "", u

def _disable_proxies_globally():
    # neutralize any proxies that break TLS (common on WSL/corp networks)
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","NO_PROXY",
              "http_proxy","https_proxy","all_proxy","no_proxy"):
        os.environ.pop(k, None)

def _is_ssl_error(e: Exception) -> bool:
    # botocore wraps urllib3.SSLError; match defensively
    s = repr(e)
    return ("SSLError" in s) or ("WRONG_VERSION_NUMBER" in s) or ("_ssl.c:" in s)

# ─────────────────────── caption augmentation ───────────────────────
_CAP_PROMPT = """
Given one or more original sentences, generate new sentences.
Preserve capitalized class names exactly (e.g., 'A220', 'Boeing787', 'ARJ21', 'C919', 'Other') and the word 'aircraft' without
modification. Keep the same overall structure and word order around those preserved
terms so the sentence remains coherent. Replace other words with synonyms or
rephrase minor phrases to introduce variation without changing the core meaning or
adding new concepts. Maintain grammatical correctness and fluency. Do not add commentary.

Use the following as an example format as example:
Sentence1
Sentence2
Sentence3
""".strip()

def _call_gpt(existing: List[str], need: int) -> List[str]:
    if need <= 0:
        return []
    user_msg = _CAP_PROMPT + "\n\nHere are the existing sentences:\n" + "\n".join(
        f"- {s}" for s in existing
    )
    resp = _openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content
    lines = [
        re.compile(r"^\s*(?:\d+\s*[.)-]?\s*|[-\u2013\u2014*\u2022]+\s+)").sub("", l).strip()
        for l in raw.split("\n")
        if l.strip()
    ]
    return lines[:need]

# ─────────────────────── pyarrow helpers ───────────────────────
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

# ─────────────────────── FAIR1M dataset ───────────────────────
class FAIR1MModulatedDetection(Dataset):
    def __init__(
        self,
        tokenizer=None,
        return_tokens: bool = False,
        augment_train: bool = False,
        bucket: str = "research-geodatasets",
        xml_prefix: str = "FAIR1M/labelXmls/",
        img_prefix: str = "FAIR1M/images/",
        csv_key: str = "FAIR1M/fair1m_sentences.csv",
        val_split_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        # S3 client will be created lazily per-process (DataLoader worker safe)
        self._s3 = None
        self.bucket = bucket
        self.seed = seed
        self.img_prefix = img_prefix
        self.prepare = _ConvertFAIR1MToTarget(tokenizer, return_tokens)

        cache_found = ARROW_CACHE.exists() and CLASSES_JSON.exists()
        if cache_found:
            ann_raw = _table_to_dict(pq.read_table(ARROW_CACHE))
        else:
            captions_by_stem, images_by_stem = self._load_captions_csv(csv_key)
            ann_raw = self._build_annotations(xml_prefix, captions_by_stem, images_by_stem)

        self.ids_all = list(ann_raw)
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

            pq.write_table(_dict_to_table(ann_raw), ARROW_CACHE, compression="zstd")
            CLASSES_JSON.write_text(json.dumps(CLASS_TO_IDX))

        self.annotations = ann_raw

        # COCO API
        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
        self.coco.createIndex()

    # Ensure the boto client isn’t pickled into worker processes
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3"] = None
        return state

    # Build a fresh client with optional SSL/verify overrides
    def _new_s3_client(self, use_ssl: Optional[bool] = None, verify: Optional[bool] = None):
        _disable_proxies_globally()
        kw = {}
        if use_ssl is not None:
            kw["use_ssl"] = use_ssl
        if verify is not None:
            kw["verify"] = verify
        return boto3.client("s3", config=_S3_CFG, **kw)

    def _get_s3(self):
        if getattr(self, "_s3", None) is None:
            # Neutralize proxy vars inside workers (fixes SSL WRONG_VERSION_NUMBER with proxies)
            _disable_proxies_globally()
            self._s3 = self._new_s3_client(
                use_ssl=(os.getenv("S3_USE_SSL", "1") == "1"),
                verify=None if os.getenv("S3_VERIFY", "1") == "1" else False,
            )
        return self._s3

    # Centralized S3 read with one-shot SSL fallback
    def _s3_read_bytes(self, bucket: str, key: str) -> bytes:
        s3 = self._get_s3()
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:
            if _is_ssl_error(e):
                # retry once with no verify / no TLS
                self._s3 = self._new_s3_client(use_ssl=False, verify=False)
                return self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            raise

    # List XML candidates with the same SSL fallback
    def _s3_list_xml_candidates(self, bucket: str, prefix: str) -> List[str]:
        s3 = self._get_s3()
        try:
            paginator = s3.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].lower().endswith(".xml")]
            return keys
        except Exception as e:
            if _is_ssl_error(e):
                self._s3 = self._new_s3_client(use_ssl=False, verify=False)
                paginator = self._s3.get_paginator("list_objects_v2")
                keys = []
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].lower().endswith(".xml")]
                return keys
            raise

    # ─────────────── Dataset plumbing ───────────────
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        img_bucket, img_key = _split_s3_uri(rec["image_path"])
        use_bucket = img_bucket or self.bucket
        key = img_key if img_bucket else rec["image_path"].removeprefix(f"s3://{self.bucket}/")

        img_bin = self._s3_read_bytes(use_bucket, key)
        img = Image.open(io.BytesIO(img_bin)).convert("RGB")

        if rec["orig_size"] == (0, 0):  # backfill dims if XML missed them
            rec["orig_size"] = img.size[::-1]

        return self.prepare(img, json.loads(json.dumps(rec)))

    # ─────────────── CSV: images + captions ───────────────
    def _load_captions_csv(self, csv_key: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Reads CSV and returns:
          captions_by_stem: {stem -> caption_string}
          images_by_stem:   {stem -> 's3://bucket/key.jpg'}
        CSV has a column "Image" with full s3 path like:
          s3://research-geodatasets/FAIR1M/images/t_10013_2.jpg
        """
        raw = (
            self._s3_read_bytes(self.bucket, csv_key)
            .decode("utf-8", errors="ignore")
            .splitlines()
        )
        rows = list(csv.DictReader(raw))

        # Robust header detection
        img_cols = ["Image", "image", "filename", "file_name", "file", "File"]
        cap_cols = ["Description", "description", "Caption", "caption", "text", "Text"]
        img_col = next((c for c in img_cols if rows and c in rows[0]), None)
        cap_col = next((c for c in cap_cols if rows and c in rows[0]), None)

        captions_by_stem: Dict[str, str] = {}
        images_by_stem: Dict[str, str] = {}

        for r in rows:
            if not img_col:
                continue
            img_uri = str(r[img_col]).strip()
            if not img_uri:
                continue
            stem = Path(img_uri).stem  # e.g., t_10013_2
            images_by_stem[stem] = img_uri
            caption = str(r.get(cap_col, "")).strip() if cap_col else ""
            captions_by_stem[stem] = caption

        return captions_by_stem, images_by_stem

    # ─────────────── XML → annotations ───────────────
    def _build_annotations(
        self,
        xml_prefix: str,
        captions_by_stem: Dict[str, str],
        images_by_stem: Dict[str, str],
    ) -> Dict[int, Dict]:
        """
        Build annotations by fetching XMLs that correspond to images listed in the CSV.
        For each image stem from CSV, try:
          1) EXACT: {xml_prefix}/{stem}.xml
          2) FALLBACK: list_objects_v2 with Prefix={xml_prefix}/{stem} and take the first *.xml
        """
        ann: Dict[int, Dict] = {}
        img_id = 0

        bucket = self.bucket

        for stem, img_uri in tqdm(
            images_by_stem.items(),
            total=len(images_by_stem),
            desc="Fetching XMLs for CSV images",
            unit="img",
        ):
            # Construct expected XML key
            exact_xml_key = f"{xml_prefix}{stem}.xml"

            # fetch XML bytes with fallback to prefix listing
            xml_bytes: Optional[bytes] = None

            try:
                xml_bytes = self._s3_read_bytes(bucket, exact_xml_key)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") != "NoSuchKey":
                    raise
                # fallback: list by prefix
                candidates = self._s3_list_xml_candidates(bucket, f"{xml_prefix}{stem}")
                if candidates:
                    xml_bytes = self._s3_read_bytes(bucket, candidates[0])

            if xml_bytes is None:
                # No XML found for this image — skip
                continue

            root = ET.fromstring(xml_bytes)

            # Prefer XML's filename if present, otherwise derive from CSV image
            fname_xml = (root.findtext("./source/filename") or root.findtext("./filename") or "").strip()
            if fname_xml:
                fname = fname_xml
            else:
                # fall back to CSV filename
                fname = Path(img_uri).name

            # image dims (from XML or fallback later)
            try:
                W = int(float(root.findtext("./size/width")))
                H = int(float(root.findtext("./size/height")))
            except Exception:
                W = H = 0

            # record uses *CSV image path* verbatim
            rec = dict(
                image_id=img_id,
                image_path=img_uri,  # keep full s3://... from CSV
                caption=captions_by_stem.get(stem, ""),
                filename_stem=stem,
                orig_size=(H, W),
                annotations=[],
            )

            # parse objects
            for objnode in root.findall("./objects/object"):
                raw_name = (objnode.findtext("./possibleresult/name") or "").strip()
                mapped = XML_NAME_TO_CANONICAL.get(raw_name.lower(), raw_name)
                if mapped not in FAIR1M_CLASSES:
                    continue

                pts = []
                for p in objnode.findall("./points/point"):
                    try:
                        x_str, y_str = p.text.split(",")
                        x, y = float(x_str), float(y_str)
                        pts.append((x, y))
                    except Exception:
                        continue
                if len(pts) < 2:
                    continue

                xs, ys = zip(*pts)
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)

                if W > 0 and H > 0:
                    x0 = max(0.0, min(x0, W)); x1 = max(0.0, min(x1, W))
                    y0 = max(0.0, min(y0, H)); y1 = max(0.0, min(y1, H))

                if (x1 - x0) < 1 or (y1 - y0) < 1:
                    continue

                cat_id = CLASS_TO_IDX[mapped]
                rec["annotations"].append(
                    dict(
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        area=float((x1 - x0) * (y1 - y0)),
                        category_id=cat_id,
                        phrase=mapped,
                        iscrowd=0,
                    )
                )

            if rec["annotations"] and rec["orig_size"] is not None:
                ann[img_id] = rec
                img_id += 1

        return ann

    # ─────────────── caption explode & augment ───────────────
    def _explode_captions_no_gpt(self, ann: Dict[int, Dict], only_ids=None) -> list[int]:
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

    def _augment_captions_once(self, ann: Dict[int, Dict], only_ids=None):
        rows = [
            {"img_id": k, "stem": v["filename_stem"], "caption": v["caption"]}
            for k, v in ann.items()
            if (only_ids is None or k in only_ids)
        ]
        if not rows:
            return
        ds = HFDataset.from_list(rows).remove_columns(["stem"])

        def _exp(batch):
            out_id, out_cap = [], []
            for img_id, cap in zip(batch["img_id"], batch["caption"]):
                sents_raw = _split(cap)
                seen, uniq_sents = set(), []
                for s in sents_raw:
                    if s and s not in seen:
                        uniq_sents.append(s); seen.add(s)

                if 1 <= len(uniq_sents) < 5:
                    attempts = 0
                    while len(uniq_sents) < 5 and attempts < 3:
                        need = 5 - len(uniq_sents)
                        gens = _call_gpt(uniq_sents, need)
                        added_any = False
                        for g in gens:
                            if g and g not in seen:
                                uniq_sents.append(g); seen.add(g); added_any = True
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

    # ─────────────── util ───────────────
    def _split_groups(self, ann_dict, val_ratio):
        ids = list(ann_dict)
        stems = [ann_dict[i]["filename_stem"] for i in ids]
        tr, vl = next(
            GroupShuffleSplit(train_size=1 - val_ratio, random_state=self.seed).split(
                ids, groups=stems
            )
        )
        return [ids[i] for i in tr], [ids[i] for i in vl]

# ─────────────────────── transforms & wrapper ───────────────────────
def _fm_transforms(split: str):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if split == "train":
        return T.Compose([
            # orientation diversity for aerial targets
            T.RandomHorizontalFlip(p=0.50),
            RandomVerticalFlip(prob=0.50),
            RandomRotate90(prob=0.50),
            RandomColorJitter(prob=0.25, brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
            RandomGaussianBlur(prob=0.10, radius=(0.05, 0.40)),
            RandomGaussianNoise(prob=0.05, std=5.0),
            T.RandomResize([704], max_size=704),
            normalize,
        ])

    elif split == "val":
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    else:
        raise ValueError(split)

class _WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base: FAIR1MModulatedDetection, tfm):
        self.base, self.tfm, self.coco = base, tfm, base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        img, tgt = self.tfm(img, tgt)
        return img, tgt

# ─────────────────────── oversampling ───────────────────────
# NEW: COCO 'small' area threshold in pixels at original image resolution
_COCO_SMALL_MAX_AREA = 32 * 32  # 1024

def _multiplier_for_img(rec: Dict, factors: Dict[int, int]) -> int:
    present = {a["category_id"] for a in rec["annotations"]}
    mult = 1
    for cid in present:
        if cid in factors:
            mult = max(mult, int(factors[cid]))
    return mult

def _oversample_ids_by_classes(
    ids: list[int], annotations: Dict[int, Dict], factors: Dict[int, int]
) -> list[int]:
    out = []
    for img_id in ids:
        mult = _multiplier_for_img(annotations[img_id], factors)
        out.extend([img_id] * mult)
    return out

# NEW: also oversample any image containing a COCO-'small' object
def _oversample_ids_by_classes_and_small(
    ids: list[int],
    annotations: Dict[int, Dict],
    class_factors: Dict[int, int],
    small_factor: int = 3,
    small_max_area: int = _COCO_SMALL_MAX_AREA,
) -> list[int]:
    out = []
    for img_id in ids:
        rec = annotations[img_id]
        # existing class-based upweighting
        mult = _multiplier_for_img(rec, class_factors)
        # small-object upweighting (any annotation with area ≤ 32^2)
        has_small = any(float(a.get("area", 0.0)) <= small_max_area for a in rec["annotations"])
        if has_small:
            mult = max(mult, int(small_factor))
        out.extend([img_id] * mult)
    return out

def build_fair1m(set_name: str, args):
    full = FAIR1MModulatedDetection(
        tokenizer=getattr(args, "tokenizer", None),
        return_tokens=True,
        val_split_ratio=getattr(args, "val_split_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        augment_train=(set_name == "train"),
        bucket="research-geodatasets",
        xml_prefix="FAIR1M/labelXmls/",
        img_prefix="FAIR1M/images/",
        csv_key="FAIR1M/fair1m_sentences.csv",
    )

    if set_name == "train":
        full.ids = full.ids_train

        # Oversample only ARJ21 ×15 and C919 ×35
        OVERSAMPLE_FACTORS = {
            CLASS_TO_IDX["ARJ21"]: 15,
            CLASS_TO_IDX["C919"]: 35,
        }
        base_n = len(full.ids)
        full.ids = _oversample_ids_by_classes_and_small(
            full.ids,
            full.annotations,
            OVERSAMPLE_FACTORS,
            small_factor=15,
            small_max_area=_COCO_SMALL_MAX_AREA,
        )
        print(f"[oversample] train ids expanded: {base_n} → {len(full.ids)} (class + small)")
        return _WrappedDataset(full, _fm_transforms("train"))

    if set_name == "val":
        full.ids = full.ids_val
        return _WrappedDataset(full, _fm_transforms("val"))

    raise ValueError(set_name)

# ─────────────────────── token alignment ───────────────────────
class _ConvertFAIR1MToTarget:
    def __init__(self, tokenizer=None, return_tokens: bool = False):
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.id2phrase = [IDX_TO_CLASS[cid] for cid in sorted(IDX_TO_CLASS)]

    def __call__(self, image, record: Dict):
        W, H = image.size

        original_annotations = record["annotations"]
        boxes = torch.as_tensor([a["bbox"] for a in original_annotations], dtype=torch.float32)
        boxes[:, 0::2].clamp_(0, W)
        boxes[:, 1::2].clamp_(0, H)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        labels_1b = torch.tensor(
            [a["category_id"] for a in original_annotations], dtype=torch.int64
        )[keep]
        labels = labels_1b - 1  # 0-based for model

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.tensor([a["iscrowd"] for a in original_annotations], dtype=torch.int64)[keep]

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

        if self.return_tokens and self.tokenizer is not None:
            caption = record.get("caption") or ""
            tokens_positive = []
            phrases_searched = []

            kept_annotations = [ann for i, ann in enumerate(original_annotations) if keep[i]]
            for annotation in kept_annotations:
                phrase = annotation.get("phrase", None)
                phrases_searched.append(phrase)
                sp = _find_exact(caption, phrase) if phrase else None
                tokens_positive.append([sp] if sp is not None else [])

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
