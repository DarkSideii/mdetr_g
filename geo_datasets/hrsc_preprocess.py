from __future__ import annotations

import csv
import io
import json
import random
import re
import time
import urllib3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config as _S3Config
from botocore.exceptions import (
    ClientError,
    ReadTimeoutError,
    ConnectTimeoutError,
    EndpointConnectionError,
)
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
from boto3.s3.transfer import TransferConfig

import multimodal_framework.mdetr.datasets.transforms as T
from util.augs import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomRotate90,
    RandomVerticalFlip,
    RandomGaussianNoise
)

# ─────────────────────────────── setup ────────────────────────────────
load_dotenv()
random.seed(42)
_openai = OpenAI()
_S3_CFG = _S3Config(
    retries={"max_attempts": 15, "mode": "adaptive"},
    connect_timeout=20,
    read_timeout=120,
    max_pool_connections=64,
)
_TRANSFER_CFG = TransferConfig(
    max_concurrency=1,
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=False,
)

# ─────────────────────────────── cache paths ────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_CACHE_DIR = _PARENT_DIR / "hrsc_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARROW_CACHE = _CACHE_DIR / "hrsc_annotations.parquet"
ROLES_JSON = _CACHE_DIR / "hrsc_roles.json"

# ───────────────────────────── category mapping ─────────────────────────────
ROLE_ID_TO_NAME = {1: "ship"}  # single class
ROLE_ID_ORDER = [1]
ROLE_TO_IDX = {rid: i + 1 for i, rid in enumerate(ROLE_ID_ORDER)}  # contiguous 1-based

# ───────── COCO small threshold (in pixels^2, original image space) ─────────
_COCO_SMALL_AREA_MAX = 32 * 32  # < 1024 px^2 is "small" per COCO

_SPLIT_RE = re.compile(r"\d+\.\s*")


def _split(txt: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(txt or "") if p.strip()]


def _cat_idx(rid: int) -> int:
    """Map role-id → contiguous [1..K] category id used by MDETR."""
    assert rid in ROLE_TO_IDX, f"Unknown role_id {rid}. Expected one of {ROLE_ID_ORDER}"
    return ROLE_TO_IDX[rid]


# ───────────────────────── caption augmentation prompt ──────────────────────
_CAP_PROMPT = (
    "Given one or more original sentences, generate two new sentences. "
    "You MUST keep the class name \"ship\" or its plural \"ships\" exactly as it appears in the original sentence "
    "(case and plurality preserved). Do not introduce additional mentions of \"ship\" or \"ships\" beyond what is "
    "already in the original sentence. Maintain the same overall structure and word order around the preserved term "
    "so the sentence remains coherent. Replace other words with synonyms or rephrase short phrases to introduce "
    "variation, without changing the core meaning or adding new concepts. Ensure grammatical correctness and fluency, "
    "keeping the style as close as possible to the original while making each new sentence feel fresh. Do not add commentary. "
    "Use the following as an example format: Sentence1\nSentence2\n"
)

def _split_sents(txt: str) -> List[str]:
    parts = re.split(r"\d+\.\s*", txt or "")
    return [p.strip() for p in parts if p.strip()]


def _call_gpt(existing: List[str], need: int) -> List[str]:
    if need <= 0:
        return []
    prefix = (_CAP_PROMPT + "\n\n") if _CAP_PROMPT else ""
    user_msg = prefix + "Here are the existing sentences:\n" + "\n".join(f"- {s}" for s in existing)
    resp = _openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    lines = [
        re.compile(r"^\s*(?:\d+\s*[.)-]?\s*|[-\u2013\u2014*\u2022]+\s+)").sub("", l).strip()
        for l in raw.split("\n")
        if l.strip()
    ]
    return lines[:need]


def create_positive_map(tokenized, tokens_positive):
    """positive_map[i,j] = 1 iff box i ↔ token j."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1) or tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2) or tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None

            if beg_pos is None or end_pos is None:
                continue
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


# ────────────────────────────── pyarrow helpers ─────────────────────────────
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
    cats = [{"id": cid, "name": ROLE_ID_TO_NAME.get(rid, f"role_{rid}")} for rid, cid in ROLE_TO_IDX.items()]
    return {"images": imgs, "annotations": anns, "categories": cats, "info": {}, "licenses": []}


# ───────────────────── HRSC2016-MS dataset (COCO-compatible) ─────────────────
class HRSC2016MSModulatedDetection(Dataset):
    """
    HRSC2016-MS (single class: 'ship'), DOTA-style modulated detection pipeline.
    *ONLY* processes rows present in the CSV. No XML or image fallback beyond those rows.
    """

    def __init__(
        self,
        tokenizer=None,
        return_tokens: bool = False,
        augment_train: bool = False,
        bucket: str = "research-geodatasets",
        annotations_prefix: str = "HRSC2016-MS/Annotations/",
        csv_key: str = "HRSC2016-MS/hrsc_sentencesv2.csv",
        val_split_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self._s3 = None
        self.bucket = bucket
        self.seed = seed
        self.annotations_prefix = annotations_prefix
        self.prepare = _ConvertHRSCToTarget(tokenizer, return_tokens)

        cache_found = ARROW_CACHE.exists() and ROLES_JSON.exists()
        if cache_found:
            ann_raw = _table_to_dict(pq.read_table(ARROW_CACHE))
        else:
            ann_raw = self._build_annotations(csv_key)

        self.ids_all = list(ann_raw)
        # Guard: if CSV was empty or nothing matched, keep splits robust
        if len(self.ids_all) == 0:
            self.ids_train, self.ids_val = [], []
        else:
            self.ids_train, self.ids_val = self._split_groups(ann_raw, val_split_ratio)
            if not self.ids_val:
                n = max(1, int(len(self.ids_all) * val_split_ratio))
                self.ids_val, self.ids_train = self.ids_all[:n], self.ids_all[n:]

        if not cache_found:
            # 1) VAL: explode to 1 sample per caption (NO GPT)
            new_val_ids = self._explode_captions_no_gpt(ann_raw, only_ids=set(self.ids_val))
            self.ids_val = list(self.ids_val) + new_val_ids

            # 2) TRAIN: split + optional GPT top-up
            if augment_train and self.ids_train:
                before = set(ann_raw.keys())
                self._augment_captions_once(ann_raw, only_ids=set(self.ids_train))
                after = set(ann_raw.keys())
                new_train_ids = sorted(after - before)
                self.ids_train = list(self.ids_train) + new_train_ids

        # Persist cache after augmentation
        if not cache_found:
            pq.write_table(_dict_to_table(ann_raw), ARROW_CACHE, compression="zstd")
            ROLES_JSON.write_text(json.dumps(ROLE_TO_IDX))

        self.annotations = ann_raw

        # COCO API object
        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
        self.coco.createIndex()

    # ────────────── Dataset plumbing ───────────────
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        key = rec["image_path"].removeprefix(f"s3://{self.bucket}/")
        img_bin = self._read_s3_bytes(key)  # ← robust reader
        img = Image.open(io.BytesIO(img_bin)).convert("RGB")

        if rec["orig_size"] == (0, 0):  # populate lazily if missing
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

    def _read_s3_bytes(self, key: str, max_retries: int = 5) -> bytes:
        """Robustly read S3 object bytes with retries; fallback to download_fileobj."""
        last_e = None
        for attempt in range(max_retries):
            try:
                return self._get_s3().get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except (urllib3.exceptions.SSLError, ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError) as e:
                last_e = e
                self._s3 = None  # reset client for fresh sockets
                time.sleep(min(1.0 * (2 ** attempt), 8.0))
            except ClientError:
                raise
        # Fallback using transfer manager
        try:
            bio = io.BytesIO()
            self._get_s3().download_fileobj(self.bucket, key, bio, Config=_TRANSFER_CFG)
            bio.seek(0)
            return bio.read()
        except Exception as e2:
            raise last_e or e2

    # ────────────── annotation builder (CSV → XML → Image) ─────────────
    def _build_annotations(self, csv_key: str) -> Dict[int, Dict]:
        """
        STRICT: Process *only* rows listed in the CSV.
        For each CSV row:
          - derive <stem> = Path(Image).stem
          - read XML at f"{self.annotations_prefix}{stem}.xml"
          - read image at CSV's Image path
        Skip rows with missing XML or with zero <object><name>ship</name> boxes.
        """
        # 1) Load CSV
        key_csv = csv_key.removeprefix(f"s3://{self.bucket}/") if csv_key.startswith("s3://") else csv_key
        csv_rows = list(csv.DictReader(self._read_s3_bytes(key_csv).decode().splitlines()))

        ann: Dict[int, Dict] = {}
        next_id = 0

        for row in tqdm(csv_rows, desc="Building HRSC from CSV rows", unit="row"):
            img_s3_path = row.get("Image") or row.get("image") or ""
            if not img_s3_path:
                continue
            caption_text = row.get("Description", "") or row.get("caption", "") or ""
            stem = Path(img_s3_path).stem

            # 2) Load XML strictly by CSV stem
            xml_key = f"{self.annotations_prefix}{stem}.xml"
            try:
                xml_bytes = self._read_s3_bytes(xml_key)
            except ClientError:
                # XML missing → skip this CSV row
                continue
            except Exception:
                # Network/parse/other issues → skip row
                continue

            try:
                root = ET.fromstring(xml_bytes)
            except ET.ParseError:
                continue

            # 3) Read image size from image file (robust)
            try:
                key_img = img_s3_path.removeprefix(f"s3://{self.bucket}/")
                img_bin = self._read_s3_bytes(key_img)  # ← robust reader
                with Image.open(io.BytesIO(img_bin)) as im:
                    W, H = im.size
            except Exception:
                # Try XML size fields as a fallback
                w_txt = root.findtext("./size/width") or "0"
                h_txt = root.findtext("./size/height") or "0"
                try:
                    W, H = int(float(w_txt)), int(float(h_txt))
                except Exception:
                    # Unreliable dims → skip row to avoid degenerate boxes
                    continue

            rec = dict(
                image_id=next_id,
                image_path=img_s3_path,
                caption=caption_text,
                filename_stem=stem,
                orig_size=(H, W),
                annotations=[],
            )

            # 4) Collect boxes: <name>ship</name> only
            for obj_node in root.findall("object"):
                name_txt = (obj_node.findtext("name") or "").strip().lower()
                if name_txt != "ship":
                    continue

                bb = obj_node.find("bndbox")
                if bb is None:
                    continue

                def _num(tag: str) -> float:
                    v = bb.findtext(tag)
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                xmin, ymin, xmax, ymax = _num("xmin"), _num("ymin"), _num("xmax"), _num("ymax")
                x0, x1 = min(xmin, xmax), max(xmin, xmax)
                y0, y1 = min(ymin, ymax), max(ymin, ymax)
                if W > 0 and H > 0:
                    x0, x1 = max(0.0, x0), min(float(W), x1)
                    y0, y1 = max(0.0, y0), min(float(H), y1)
                if (x1 - x0) < 2 or (y1 - y0) < 2:
                    continue

                rec["annotations"].append(
                    dict(
                        bbox=[x0, y0, y1 and x1][0:3] if False else [x0, y0, x1, y1],  # keep xyxy
                        area=float((x1 - x0) * (y1 - y0)),
                        category_id=_cat_idx(1),
                        phrase="ship",
                        iscrowd=0,
                    )
                )

            # 5) Keep only rows with at least one valid box
            if rec["annotations"]:
                ann[next_id] = rec
                next_id += 1

        return ann

    # ───────────── caption splitting / augmentation ─────────────
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

    def _augment_captions_once(self, ann, only_ids=None):
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

        nxt = max(ann) + 1 if ann else 0
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

    # ───────────── util helpers ─────────────
    def _split_groups(self, ann_dict, val_ratio):
        ids = list(ann_dict)
        if not ids:
            return [], []
        stems = [ann_dict[i]["filename_stem"] for i in ids]
        tr, vl = next(
            GroupShuffleSplit(train_size=1 - val_ratio, random_state=self.seed).split(
                ids, groups=stems
            )
        )
        return [ids[i] for i in tr], [ids[i] for i in vl]


# ───────────────────────── transforms (as in your original) ─────────────────────────
def _hrsc_transforms(split: str):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if split == "train":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.50),
            RandomVerticalFlip(prob=0.50),
            RandomRotate90(prob=0.50),
            RandomColorJitter(prob=0.40, brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
            RandomGaussianBlur(prob=0.10, radius=(0.1, 0.6)),
            RandomGaussianNoise(prob=0.08, std=5.0),
            T.RandomResize([704], max_size=704),
            normalize,
        ])

    elif split == "val":
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    else:
        raise ValueError(split)

class _WrappedDataset(torch.utils.data.Dataset):
    """Attach transforms while preserving .coco for evaluator."""
    def __init__(self, base: HRSC2016MSModulatedDetection, tfm):
        self.base, self.tfm, self.coco = base, tfm, base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        img, tgt = self.tfm(img, tgt)  # transform AFTER convert
        return img, tgt


# ───────────────────── dataset builder (kept entry point) ─────────────────────
def build_hrsc2016ms(set_name: str, args):
    """
    Entry point matching your __init__ mapper; instantiates HRSC2016MSModulatedDetection.
    """
    full = HRSC2016MSModulatedDetection(
        tokenizer=args.tokenizer,
        return_tokens=True,
        val_split_ratio=getattr(args, "val_split_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        augment_train=(set_name == "train"),
    )

    if set_name == "train":
        full.ids = full.ids_train

        base_n = len(full.ids)
        full.ids = _oversample_ids_by_coco_small(full.ids, full.annotations, area_thr=_COCO_SMALL_AREA_MAX, mult=15)
        print(f"[oversample] train ids expanded: {base_n} → {len(full.ids)}")

        return _WrappedDataset(full, _hrsc_transforms("train"))

    if set_name == "val":
        full.ids = full.ids_val
        return _WrappedDataset(full, _hrsc_transforms("val"))

    raise ValueError(set_name)


# ───────────────────── minority oversampling (not used; single class) ─────────────────────
def _multiplier_for_img(rec: Dict, factors: Dict[int, int]) -> int:
    present = {a["category_id"] for a in rec["annotations"]}
    mult = 1
    for cid in present:
        if cid in factors:
            mult = max(mult, int(factors[cid]))
    return mult


def _oversample_ids_by_roles(ids: list[int], annotations: Dict[int, Dict], factors: Dict[int, int]) -> list[int]:
    out = []
    for img_id in ids:
        mult = _multiplier_for_img(annotations[img_id], factors)
        out.extend([img_id] * mult)
    return out


# ───────────────────── NEW: COCO-small oversampling helpers ─────────────────────
def _record_has_coco_small(rec: Dict, area_thr: float = _COCO_SMALL_AREA_MAX) -> bool:
    """
    True iff any annotation in this record is 'small' by COCO (area < 32^2 px, original image space).
    Falls back to bbox area if 'area' is missing.
    """
    for a in rec.get("annotations", []):
        area_val = a.get("area", None)
        if area_val is None:
            try:
                x0, y0, x1, y1 = a["bbox"]
                area_val = float((x1 - x0) * (y1 - y0))
            except Exception:
                continue
        try:
            if float(area_val) < float(area_thr):
                return True
        except Exception:
            continue
    return False


def _oversample_ids_by_coco_small(
    ids: List[int],
    annotations: Dict[int, Dict],
    area_thr: float = _COCO_SMALL_AREA_MAX,
    mult: int = 15,
) -> List[int]:
    """
    For each image id in 'ids', if it contains *any* COCO-small object, repeat that id 'mult' times; else once.
    """
    out: List[int] = []
    for img_id in ids:
        rec = annotations[img_id]
        if _record_has_coco_small(rec, area_thr=area_thr):
            out.extend([img_id] * int(mult))
        else:
            out.append(img_id)
    return out


# ───────────────────── token alignment for HRSC (DOTA-style) ─────────────────────
CAPTION_KEYS = {
    "ship": ["ship", "ships"],
}

def _find_one_of(caption: str, candidates: List[str]) -> Optional[Tuple[int, int, str]]:
    """
    DOTA-style: case-insensitive substring match for any candidate.
    Returns (start, end, matched_text) or None.
    NOTE: No word-boundary check; 'ship' will match inside 'ships'.
    """
    if not caption:
        return None
    lower = caption.lower()
    for w in candidates:
        idx = lower.find(w)
        if idx != -1:
            return (idx, idx + len(w), caption[idx:idx + len(w)])
    return None


class _ConvertHRSCToTarget:
    def __init__(self, tokenizer=None, return_tokens: bool = False):
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.id2phrase = [ROLE_ID_TO_NAME.get(rid, f"role_{rid}") for rid in ROLE_ID_ORDER]

    def __call__(self, image, record: Dict):
        W, H = image.size

        original_annotations = record["annotations"]
        boxes = torch.as_tensor([a["bbox"] for a in original_annotations], dtype=torch.float32)
        boxes[:, 0::2].clamp_(0, W)
        boxes[:, 1::2].clamp_(0, H)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        labels_1b = torch.tensor([a["category_id"] for a in original_annotations], dtype=torch.int64)[keep]
        labels = labels_1b - 1  # 0-based for the model

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

        # DOTA-style positive map: per-kept-box, find ship/ships span in caption
        if self.return_tokens and self.tokenizer is not None:
            caption = record.get("caption") or ""
            tokens_positive = []
            phrases_searched = []

            kept_annotations = [ann for i, ann in enumerate(original_annotations) if keep[i]]
            for annotation in kept_annotations:
                cls_name = "ship"
                phrases_searched.append(cls_name)

                span = _find_one_of(caption, CAPTION_KEYS[cls_name])
                if span is not None:
                    (beg, end, matched_text) = span
                    annotation["phrase"] = matched_text
                    tokens_positive.append([(beg, end)])
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
