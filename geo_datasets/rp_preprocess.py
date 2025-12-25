from __future__ import annotations

import csv
import io
import json
import random
import re
import tempfile
from pathlib import Path
from typing import Dict, List

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
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
)

load_dotenv()
random.seed(42)
_openai = OpenAI()

# ─────────────────────────────── cache paths ────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_CACHE_DIR = _PARENT_DIR / "rareplanes_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARROW_CACHE = _CACHE_DIR / "rp_annotations.parquet"
ROLES_JSON = _CACHE_DIR / "rp_roles.json"

# ───────────────────────────── category mapping ─────────────────────────────
ROLE_ID_TO_NAME = {
    1: "Small Civil Transport/Utility",
    2: "Medium Civil Transport/Utility",
    3: "Large Civil Transport/Utility",
    4: "Military Transport/Utility/AWAC",
    5: "Military Bomber",
    6: "Military Fighter/Interceptor/Attack",
    7: "Military Trainer",
}
ROLE_ID_ORDER = [1, 2, 3, 4, 5, 6, 7]
ROLE_TO_IDX = {rid: i + 1 for i, rid in enumerate(ROLE_ID_ORDER)}  # 1-based

_SPLIT_RE = re.compile(r"\d+\.\s*")


def _split(txt: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(txt or "") if p.strip()]


def _cat_idx(rid: int) -> int:
    """Map role-id → contiguous [1..K] category id used by MDETR."""
    assert rid in ROLE_TO_IDX, f"Unknown role_id {rid}. Expected one of {ROLE_ID_ORDER}"
    return ROLE_TO_IDX[rid]


# ───────────────────────── caption augmentation prompt ──────────────────────
_CAP_PROMPT = """
Given one or more original sentences, generate new sentences.
Preserve any capitalized class names exactly (e.g., 'Small Civil Transport/Utility', 'Large Civil Transport/Utility', 
'Medium Civil Transport/Utility', Military Fighter/Interceptor/Attack', 'Military Bomber', etc) and the word 'aircraft' without
modification. Keep the same overall structure and word order around those preserved
terms so the sentence remains coherent. Replace other words with synonyms or
rephrase minor phrases to introduce variation without changing the core meaning or
adding new concepts. Maintain grammatical correctness and fluency. Do not add commentary.

Use the following as an example format as example:
Sentence1
Sentence2
Sentence3
""".strip()


def _split_sents(txt: str) -> List[str]:
    """Coco captions are single sentences; RP captions may be enumerated."""
    parts = re.split(r"\d+\.\s*", txt or "")
    return [p.strip() for p in parts if p.strip()]


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
    # if random.random() < 0.05:
    #     print("\n[_call_gpt sample]", lines[:need])
    return lines[:need]


def create_positive_map(tokenized, tokens_positive):
    """Construct a map such that positive_map[i,j] = True iff box i is associated to token j."""
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

            assert beg_pos is not None and end_pos is not None
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
                    phrase=a.get("phrase", ""),  # ← add this
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
                phrase=r.get("phrase", None),  # ← add this
                iscrowd=r["iscrowd"],
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
                    iscrowd=a["iscrowd"],
                )
            )
            aid += 1

    cats = [
        {"id": cid, "name": ROLE_ID_TO_NAME.get(rid, f"role_{rid}")}
        for rid, cid in ROLE_TO_IDX.items()
    ]
    return {"images": imgs, "annotations": anns, "categories": cats, "info": {}, "licenses": []}


# ───────────────────── RarePlanes dataset (COCO-compatible) ──────────────────
class RarePlanesModulatedDetection(Dataset):
    def __init__(
        self,
        tokenizer=None,
        return_tokens: bool = False,
        augment_train: bool = False,
        bucket: str = "research-geodatasets",
        geojson_prefix: str = "RarePlanes/geojsons/",
        csv_key: str = "RarePlanes/rp_sentences.csv",
        val_split_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.seed = seed
        self.prepare = _ConvertRPToTarget(tokenizer, return_tokens)

        cache_found = ARROW_CACHE.exists() and ROLES_JSON.exists()
        if cache_found:
            ann_raw = _table_to_dict(pq.read_table(ARROW_CACHE))
        else:
            ann_raw = self._build_annotations(geojson_prefix, csv_key)

        self.ids_all = list(ann_raw)
        self.ids_train, self.ids_val = self._split_groups(ann_raw, val_split_ratio)
        if not self.ids_val:
            n = max(1, int(len(self.ids_all) * val_split_ratio))
            self.ids_val, self.ids_train = self.ids_all[:n], self.ids_all[n:]

        if not cache_found:
            # 1) VAL: explode to 1 sample per caption (NO GPT)
            new_val_ids = self._explode_captions_no_gpt(ann_raw, only_ids=set(self.ids_val))
            self.ids_val = list(self.ids_val) + new_val_ids

            # 2) TRAIN: original behavior (split + GPT top-up)
            if augment_train:
                before = set(ann_raw.keys())
                self._augment_captions_once(ann_raw, only_ids=set(self.ids_train))
                after = set(ann_raw.keys())
                new_train_ids = sorted(after - before)
                self.ids_train = list(self.ids_train) + new_train_ids

        # write cache only after optional augmentation, so it’s persisted
        if not cache_found:
            pq.write_table(_dict_to_table(ann_raw), ARROW_CACHE, compression="zstd")
            ROLES_JSON.write_text(json.dumps(ROLE_TO_IDX))

        self.annotations = ann_raw

        # 3. COCO API object so dataloader/evaluator can consume directly
        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
        self.coco.createIndex()

    # ────────────── Dataset plumbing ───────────────
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rec = self.annotations[self.ids[idx]]
        key = rec["image_path"].removeprefix(f"s3://{self.bucket}/")
        img_bin = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        img = Image.open(io.BytesIO(img_bin)).convert("RGB")

        if rec["orig_size"] == (0, 0):  # populate lazily if missing
            rec["orig_size"] = img.size[::-1]

        return self.prepare(img, json.loads(json.dumps(rec)))

    # ────────────── Annotation builder ─────────────
    def _build_annotations(self, prefix: str, csv_key: str) -> Dict[int, Dict]:
        """Scan S3 once, convert GeoJSON → pixel bboxes. Slow – cached later."""
        csv_rows = list(
            csv.DictReader(
                self.s3.get_object(Bucket=self.bucket, Key=csv_key)["Body"]
                .read()
                .decode()
                .splitlines()
            )
        )
        info_by_stem = {Path(r["Image"]).stem: r for r in csv_rows}

        paginator = self.s3.get_paginator("list_objects_v2")
        geojson_objs = [
            obj
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".geojson")
        ]

        ann, img_id, tiles_seen = {}, 0, 0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for obj in tqdm(geojson_objs, desc="Parsing RP GeoJSON", unit="file"):
                # if tiles_seen >= 100:
                #     break
                stem = Path(obj["Key"]).stem
                if stem not in info_by_stem:
                    continue

                png_key = info_by_stem[stem]["Image"].removeprefix(f"s3://{self.bucket}/")
                aux_key = f"{png_key}.aux.xml"

                # download once per tile
                png_path = tmpdir / f"{stem}.png"
                if not png_path.exists():
                    png_path.write_bytes(
                        self.s3.get_object(Bucket=self.bucket, Key=png_key)["Body"].read()
                    )
                try:
                    (tmpdir / f"{stem}.png.aux.xml").write_bytes(
                        self.s3.get_object(Bucket=self.bucket, Key=aux_key)["Body"].read()
                    )
                except self.s3.exceptions.NoSuchKey:
                    pass

                with rasterio.open(png_path) as ds:
                    W, H = ds.width, ds.height
                    to_pixel = lambda lon, lat: ds.index(lon, lat)[::-1]

                    # create record on first encounter
                    if stem not in {v["filename_stem"] for v in ann.values()}:
                        ann[img_id] = dict(
                            image_id=img_id,
                            image_path=info_by_stem[stem]["Image"],
                            caption=info_by_stem[stem]["Description"],
                            filename_stem=stem,
                            orig_size=(H, W),
                            annotations=[],
                        )
                        img_id += 1
                        tiles_seen += 1

                    rec = ann[
                        next(i for i, v in ann.items() if v["filename_stem"] == stem)
                    ]

                    gj = json.loads(
                        self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                    )
                    for feat in gj.get("features", []):
                        xs, ys = zip(
                            *[to_pixel(lon, lat) for lon, lat in feat["geometry"]["coordinates"][0]]
                        )
                        x0, x1 = sorted((max(0, min(xs)), min(W, max(xs))))
                        y0, y1 = sorted((max(0, min(ys)), min(H, max(ys))))
                        if (x1 - x0) < 2 or (y1 - y0) < 2:
                            continue

                        # This block correctly extracts and stores the phrase
                        role_id_raw = feat["properties"].get("role_id", None)
                        role_text = feat["properties"].get("role", None)
                        assert (
                            role_id_raw is not None and role_text is not None
                        ), f"Missing role info for tile {stem}"
                        role_id = int(role_id_raw)

                        rec["annotations"].append(
                            dict(
                                bbox=[x0, y0, x1, y1],
                                area=float((x1 - x0) * (y1 - y0)),
                                category_id=_cat_idx(role_id),
                                phrase=role_text,
                                iscrowd=0,
                            )
                        )

        # self._augment_captions_once(ann)
        return ann

    def _explode_captions_no_gpt(self, ann: Dict[int, Dict], only_ids=None) -> list[int]:
        """
        For each image in only_ids:
          - Split existing caption into sentences (enumerations supported).
          - Duplicate so each sentence is its own example.
        Returns list of NEW image_ids created.
        """
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
            # seed base with the first sentence
            base["caption"] = sents[0]
            # duplicate for remaining sentences
            for s in sents[1:]:
                dup = base.copy()
                dup["annotations"] = [a.copy() for a in base["annotations"]]
                dup["caption"] = s
                dup["image_id"] = nxt
                ann[nxt] = dup
                new_ids.append(nxt)
                nxt += 1
        return new_ids

    # ───────────── caption augmentation ─────────────
    def _augment_captions_once(self, ann, only_ids=None):
        """
        For each image:
        - Split caption into sentences (enumerations supported) and de-dup (order-preserving).
        - If 0 sentences: skip augmentation for this image.
        - If 1..4 sentences: generate exactly (5 - current_count) new sentences with GPT
          (retry up to 3 times to reach 5 unique), then explode to 1 sample per sentence.
        - If >=5 sentences: skip GPT and just explode.
        """
        rows = [
            {"img_id": k, "stem": v["filename_stem"], "caption": v["caption"]}
            for k, v in ann.items()
            if (only_ids is None or k in only_ids)
        ]
        ds = HFDataset.from_list(rows).remove_columns(["stem"])

        def _exp(batch):
            out_id, out_cap = [], []
            for img_id, cap in zip(batch["img_id"], batch["caption"]):
                # 1) split + order-preserving dedup
                sents_raw = _split(cap)
                seen, uniq_sents = set(), []
                for s in sents_raw:
                    if s and s not in seen:
                        uniq_sents.append(s);
                        seen.add(s)

                # 2) top-up to 5 unique sentences (if we have at least one)
                if 1 <= len(uniq_sents) < 5:
                    attempts = 0
                    while len(uniq_sents) < 5 and attempts < 3:
                        need = 5 - len(uniq_sents)
                        gens = _call_gpt(uniq_sents, need)
                        # append only truly new lines
                        added_any = False
                        for g in gens:
                            if g and g not in seen:
                                uniq_sents.append(g);
                                seen.add(g);
                                added_any = True
                        if not added_any:
                            # model repeated itself—stop early to avoid a tight loop
                            break
                        attempts += 1

                # 3) if 0 sentences, skip; otherwise emit (one row per sentence)
                if len(uniq_sents) > 0:
                    out_id.extend([img_id] * len(uniq_sents))
                    out_cap.extend(uniq_sents)

            return {"img_id": out_id, "caption": out_cap}

        ds = ds.map(
            _exp,
            batched=True,
            batch_size=1000,
            remove_columns=[],
            load_from_cache_file=False,
        )

        # explode: first sentence stays on base; others become duplicates
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

    # ───────────── util helpers ─────────────
    def _split_groups(self, ann_dict, val_ratio):
        ids = list(ann_dict)
        stems = [ann_dict[i]["filename_stem"] for i in ids]
        tr, vl = next(
            GroupShuffleSplit(train_size=1 - val_ratio, random_state=self.seed).split(
                ids, groups=stems
            )
        )
        return [ids[i] for i in tr], [ids[i] for i in vl]


def _rp_transforms(split: str):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_scales = [512, 544, 576, 608, 640, 672, 704]

    if split == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.50),
                RandomVerticalFlip(prob=0.50),
                RandomRotate90(prob=0.50),
                RandomColorJitter(prob=0.30, brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                RandomGaussianBlur(prob=0.10, radius=(0.05, 0.40)),
                T.RandomResize(train_scales, max_size=704),
                normalize,
            ]
        )
    elif split == "val":
        return T.Compose([T.RandomResize([704], max_size=704), normalize])
    else:
        raise ValueError(split)


class _WrappedDataset(torch.utils.data.Dataset):
    """Attach transforms while preserving .coco for evaluator."""

    def __init__(self, base: RarePlanesModulatedDetection, tfm):
        self.base, self.tfm, self.coco = base, tfm, base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        img, tgt = self.tfm(img, tgt)  # transform AFTER convert
        return img, tgt


# ────────────────────── minority oversampling helpers ───────────────────────
def _multiplier_for_img(rec: Dict, factors: Dict[int, int]) -> int:
    """
    Decide the oversample multiplier for one image:
    - rec["annotations"][*]["category_id"] is contiguous [1..K]
    - 'factors' maps contiguous ids -> multiplier
    - take MAX multiplier across present classes (most-minority supersedes)
    """
    present = {a["category_id"] for a in rec["annotations"]}
    mult = 1
    for cid in present:
        if cid in factors:
            mult = max(mult, int(factors[cid]))
    return mult


def _oversample_ids_by_roles(
    ids: list[int], annotations: Dict[int, Dict], factors: Dict[int, int]
) -> list[int]:
    """Expand id list by repeating each id 'mult' times according to _multiplier_for_img."""
    out = []
    for img_id in ids:
        mult = _multiplier_for_img(annotations[img_id], factors)
        out.extend([img_id] * mult)
    return out


def build_rareplanes(set_name: str, args):
    full = RarePlanesModulatedDetection(
        tokenizer=args.tokenizer,
        return_tokens=True,
        val_split_ratio=getattr(args, "val_split_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        augment_train=(set_name == "train"),
    )

    if set_name == "train":
        full.ids = full.ids_train

        OVERSAMPLE_FACTORS = {
            ROLE_TO_IDX[4]: 8,   # AWAC
            ROLE_TO_IDX[6]: 8,   # Fighter/Attack
            ROLE_TO_IDX[5]: 16,  # Bomber
            ROLE_TO_IDX[7]: 16,  # Trainer
        }
        base_n = len(full.ids)
        full.ids = _oversample_ids_by_roles(full.ids, full.annotations, OVERSAMPLE_FACTORS)
        print(f"[oversample] train ids expanded: {base_n} → {len(full.ids)}")
        return _WrappedDataset(full, _rp_transforms("train"))

    if set_name == "val":
        full.ids = full.ids_val
        return _WrappedDataset(full, _rp_transforms("val"))

    raise ValueError(set_name)


# ───────────────────── token alignment helpers (exact-match only) ─────────────────────
def _find_exact(caption: str, phrase: str):
    """Return (start, end) for an exact-case phrase occurrence, or None."""
    if not caption or not phrase:
        return None
    i = caption.find(phrase)  # exact case, respects "/"
    return (i, i + len(phrase)) if i != -1 else None


class _ConvertRPToTarget:
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

        labels_1b = torch.tensor(
            [a["category_id"] for a in original_annotations], dtype=torch.int64
        )[keep]
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
                phrase = annotation.get("phrase", None)
                phrases_searched.append(phrase)
                # --- REVERTED TO _find_exact ---
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
