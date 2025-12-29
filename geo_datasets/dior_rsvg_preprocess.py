from __future__ import annotations

import hashlib
import io
import json
import random
import re
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
from util.augs import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomRotate90,
    RandomVerticalFlip,
    RandomGaussianNoise,
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
_CACHE_VERSION = "dior_rsvg_strict_phrase_v1"

# ───────────────────────────── category/phrase mapping ─────────────────────────────
# Canonical DIOR class keys (stable order)
DIOR_CLASS_KEYS: Tuple[str, ...] = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expresswayservicearea",
    "expresswaytollstation",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)
DIOR_KEY_TO_CID: Dict[str, int] = {k: i + 1 for i, k in enumerate(DIOR_CLASS_KEYS)}
DIOR_CID_TO_KEY: Dict[int, str] = {v: k for k, v in DIOR_KEY_TO_CID.items()}

# Strict surface forms expected in DIOR-RSVG captions.
# Matching is strict token order (case-insensitive), allowing separators [-_ ] between tokens.
DIOR_KEY_TO_PHRASES: Dict[str, Tuple[str, ...]] = {
    "airplane": ("airplane",),
    "airport": ("airport",),
    "baseballfield": ("baseball field", "baseballfield"),
    "basketballcourt": ("basketball court", "basketballcourt"),
    "bridge": ("bridge",),
    "chimney": ("chimney",),
    "dam": ("dam",),
    "expresswayservicearea": ("expressway service area", "expressway-service-area"),
    "expresswaytollstation": ("expressway toll station", "expressway-toll-station"),
    "golffield": ("golf field", "golffield"),
    "groundtrackfield": ("ground track field", "groundtrackfield"),
    "harbor": ("harbor",),
    "overpass": ("overpass",),
    "ship": ("ship",),
    "stadium": ("stadium",),
    "storagetank": ("storage tank", "storagetank"),
    "tenniscourt": ("tennis court", "tenniscourt"),
    "trainstation": ("train station", "trainstation"),
    "vehicle": ("vehicle",),
    "windmill": ("windmill", "wind mill"),
}

# ────────────────────────────── helpers ──────────────────────────────
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


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


def _cache_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]


def _norm_key(s: str) -> str:
    """Lowercase and keep only [a-z0-9] for class-key matching."""
    if not s:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _compile_phrase_pattern(phrase: str) -> re.Pattern:
    """
    Compile a strict, case-insensitive pattern that matches `phrase` by token order.
    Between tokens, allow separators: whitespace, hyphen, underscore.
    """
    toks = [m.group(0).lower() for m in _WORD_RE.finditer(phrase)]
    if not toks:
        return re.compile(r"a^")  # never match
    if len(toks) == 1:
        pat = rf"\b{re.escape(toks[0])}\b"
    else:
        sep = r"(?:[\s_-]+)"
        pat = r"\b" + sep.join(re.escape(t) for t in toks) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)


# Precompile patterns once (fast, deterministic)
_DIOR_PATTERNS: Dict[str, List[Tuple[str, re.Pattern]]] = {}
for _k, _phr_list in DIOR_KEY_TO_PHRASES.items():
    _DIOR_PATTERNS[_k] = [(p, _compile_phrase_pattern(p)) for p in _phr_list]


def _find_strict_span(caption: str, class_key: str) -> Optional[Tuple[int, int, str]]:
    """
    Return (start,end,phrase_used) for the earliest strict phrase match in caption.
    Matching is case-insensitive, and only allows separators [-_ ] between tokens.
    """
    caption = caption or ""
    if not caption:
        return None
    if class_key not in _DIOR_PATTERNS:
        return None

    best = None  # (start, end, phrase)
    for phrase, pat in _DIOR_PATTERNS[class_key]:
        for m in pat.finditer(caption):
            s, e = int(m.start()), int(m.end())
            if best is None or s < best[0] or (s == best[0] and (e - s) > (best[1] - best[0])):
                best = (s, e, phrase)
            break  # only need earliest occurrence for this phrase
    return best


def _char_span_to_token_span(tokenized, beg: int, end: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert [beg,end) char span into [beg_pos,end_pos] token span using the exact
    same fallback logic as create_positive_map().
    Returns (beg_pos, end_pos) or (None, None) if not found.
    """
    beg = int(beg)
    end = int(end)
    if end <= beg:
        return None, None

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

    return beg_pos, end_pos


def create_positive_map(tokenized, tokens_positive: List[List[Tuple[int, int]]], max_text_len: int) -> torch.Tensor:
    """Construct positive_map[i,j] = 1 iff box i is associated to token j (row-normalized)."""
    positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg = int(beg)
            end = int(end)
            if end <= beg:
                continue

            beg_pos, end_pos = _char_span_to_token_span(tokenized, beg, end)
            if beg_pos is None or end_pos is None:
                continue

            positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def _debug_print_random_alignments(
    annotations: Dict[int, Dict],
    tokenizer,
    max_text_len: int,
    ids: List[int],
    n: int = 10,
    seed: int = 42,
) -> None:
    """
    Print N random examples:
      - caption (with the matched phrase bracketed)
      - object class key + phrase
      - char span [tok_beg,tok_end)
      - token span [beg_pos,end_pos] used to fill positive_map
      - active token indices in positive_map row
      - token strings in that span
    Does NOT load images (fast).
    """
    if tokenizer is None:
        print("[DIOR-RSVG debug] tokenizer is None -> cannot compute token positions / positive_map.")
        return

    rng = random.Random(int(seed))
    cand = []
    for i in ids:
        rec = annotations.get(i, None)
        if not rec:
            continue
        anns = rec.get("annotations", []) or []
        if not anns:
            continue
        a0 = anns[0]
        beg = int(a0.get("tok_beg", 0))
        end = int(a0.get("tok_end", 0))
        cap = (rec.get("caption", "") or "")
        if end > beg and cap:
            cand.append(i)

    if not cand:
        print("[DIOR-RSVG debug] no candidate samples with non-empty tok span found.")
        return

    k = min(int(n), len(cand))
    picks = rng.sample(cand, k)

    print(f"\n[DIOR-RSVG debug] printing {k} random alignment samples (seed={seed})\n")

    for j, img_id in enumerate(picks, start=1):
        rec = annotations[img_id]
        a0 = rec["annotations"][0]
        caption = rec.get("caption", "") or ""
        stem = rec.get("filename_stem", "N/A")

        cls_key = a0.get("cls_key", "")
        phrase = a0.get("phrase", "")
        tok_beg = int(a0.get("tok_beg", 0))
        tok_end = int(a0.get("tok_end", 0))

        marked = caption
        if tok_end > tok_beg and 0 <= tok_beg <= len(caption) and 0 <= tok_end <= len(caption):
            marked = caption[:tok_beg] + "[[" + caption[tok_beg:tok_end] + "]]" + caption[tok_end:]

        tokenized = tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
            return_offsets_mapping=True,
        )

        tokens_positive = [[(tok_beg, tok_end)]]
        positive_map = create_positive_map(tokenized, tokens_positive, max_text_len=max_text_len)

        beg_pos, end_pos = _char_span_to_token_span(tokenized, tok_beg, tok_end)

        active = torch.nonzero(positive_map[0] > 0, as_tuple=False).flatten().tolist()
        active_span = (min(active), max(active)) if active else None

        # Token strings (best-effort)
        tokens_str = None
        try:
            ids0 = tokenized["input_ids"][0].tolist()
            toks = tokenizer.convert_ids_to_tokens(ids0)
            if beg_pos is not None and end_pos is not None:
                lo = max(0, beg_pos)
                hi = min(len(toks) - 1, end_pos)
                tokens_str = toks[lo : hi + 1]
        except Exception:
            tokens_str = None

        print(f"--- [{j}/{k}] img_id={img_id} stem={stem}")
        print(f"class_key: {cls_key!r} | phrase: {phrase!r}")
        print(f"char_span: [{tok_beg}, {tok_end}) -> substring: {caption[tok_beg:tok_end]!r}")
        print(f"token_span (char_to_token): {beg_pos} .. {end_pos}")
        print(f"positive_map active idxs: {active_span} (count={len(active)})")

        if tokens_str is not None:
            print(f"tokens in span: {tokens_str}")

        # Keep the caption readable (don’t dump massive strings)
        if len(marked) > 240:
            print(f"caption: {marked[:240]} ...")
        else:
            print(f"caption: {marked}")
        print()

    print("[DIOR-RSVG debug] done.\n")


# ────────────────────────────── pyarrow helpers ─────────────────────────────
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
                    cls_key="",
                    phrase="",
                    tok_beg=0,
                    tok_end=0,
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
                    cls_key=a.get("cls_key", ""),
                    phrase=a.get("phrase", ""),
                    tok_beg=a.get("tok_beg", 0),
                    tok_end=a.get("tok_end", 0),
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
                    cls_key=r.get("cls_key", ""),
                    phrase=r.get("phrase", ""),
                    tok_beg=r.get("tok_beg", 0),
                    tok_end=r.get("tok_end", 0),
                    iscrowd=r.get("iscrowd", 0),
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
                    iscrowd=a.get("iscrowd", 0),
                )
            )
            aid += 1

    cats = [{"id": DIOR_KEY_TO_CID[k], "name": k} for k in DIOR_CLASS_KEYS]
    return {"images": imgs, "annotations": anns, "categories": cats, "info": {}, "licenses": []}


# ─────────────────────────────── Dataset ────────────────────────────────
@dataclass
class _Sources:
    # NO DEFAULTS: must come from JSON
    images_dir: str  # local dir OR s3://bucket/prefix/to/images
    annotations_dir: str  # local dir OR s3://bucket/prefix/to/annotations


class DiorRSVGModulatedDetection(Dataset):
    """
    DIOR-RSVG Pascal VOC XML (RefExp-style materialization).

    Key changes vs previous version:
      - strict class→phrase mapping (no fuzzy matching)
      - positive_map built via tokenizer.char_to_token (MDETR-style), not offset-overlap
    """

    def __init__(
        self,
        sources: _Sources,
        tokenizer=None,
        max_text_len: int = 256,
        strict_phrase_match: bool = True,
    ):
        super().__init__()

        if not sources.images_dir or not sources.annotations_dir:
            raise ValueError("DIOR-RSVG: images_dir and annotations_dir must be provided (from JSON).")

        self.sources = sources
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)
        self.strict_phrase_match = bool(strict_phrase_match)

        # per-source cache dir (prevents mixing train vs eval caches)
        ck = _cache_key(_CACHE_VERSION, self.sources.images_dir, self.sources.annotations_dir)
        self.cache_dir = _CACHE_BASE / ck
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.arrow_cache = self.cache_dir / "annotations.parquet"

        if self.arrow_cache.exists():
            table = pq.read_table(self.arrow_cache)
            ann_raw = _table_to_dict(table)
        else:
            ann_raw = self._build_annotations()
            pq.write_table(_dict_to_table(ann_raw), self.arrow_cache, compression="zstd")

        self.annotations = ann_raw
        self.ids_all = list(self.annotations.keys())
        self.ids = self.ids_all

        self.coco = COCO()
        self.coco.dataset = _build_coco_like(self.annotations)
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
        ann_is_s3 = _is_s3(self.sources.annotations_dir)
        img_is_s3 = _is_s3(self.sources.images_dir)
        if ann_is_s3 != img_is_s3:
            raise ValueError("DIOR-RSVG: images_dir and annotations_dir must both be local or both be s3://...")

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
          sample = {filename, orig_size=(h,w), caption, annotations=[{bbox,..., tok_beg, tok_end, phrase, cls_key}], cap_idx}
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

        tmp_samples: List[Dict] = []
        max_x = 0.0
        max_y = 0.0

        cap_idx = 0
        for obj_node in root.findall("object"):
            cls_raw = (obj_node.findtext("name") or "unknown").strip() or "unknown"
            cls_key = _norm_key(cls_raw)

            if cls_key not in DIOR_KEY_TO_CID:
                raise ValueError(
                    f"DIOR-RSVG: unknown class name {cls_raw!r} (normalized={cls_key!r}) in {xml_stem}.xml"
                )

            # caption = per-object <description> (fallback to canonical phrase)
            desc = (obj_node.findtext("description") or "").strip()
            if not desc:
                desc = DIOR_KEY_TO_PHRASES[cls_key][0]
            caption = desc

            # bbox
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

            # STRICT phrase span (no fuzzy)
            span = _find_strict_span(caption, cls_key)
            if span is None:
                if self.strict_phrase_match:
                    raise ValueError(
                        f"DIOR-RSVG: could not find a strict phrase match for class={cls_raw!r} "
                        f"(key={cls_key!r}) in caption={caption!r} (file={xml_stem}.xml)"
                    )
                tok_beg, tok_end, phrase_used = 0, 0, DIOR_KEY_TO_PHRASES[cls_key][0]
            else:
                tok_beg, tok_end, phrase_used = span

            area = float((x1c - x0) * (y1c - y0))
            ann = dict(
                bbox=[x0, y0, x1c, y1c],
                area=area,
                category_id=DIOR_KEY_TO_CID[cls_key],
                cls_key=cls_key,
                phrase=phrase_used,
                tok_beg=int(tok_beg),
                tok_end=int(tok_end),
                iscrowd=0,
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
            ]
        )
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

    Optional debug key:
      - debug_print_n  (int): if >0, prints N random caption↔token alignments
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
        max_text_len=int(getattr(args, "max_text_len", 256)),
        strict_phrase_match=bool(getattr(args, "strict_phrase_match", True)),
    )

    ids_all = list(full.ids_all)

    debug_n = getattr(args, "debug_print_n", 10)
    debug_seed = int(getattr(args, "seed", 42))

    if eval_mode:
        full.ids = ids_all
        if debug_n > 0:
            _debug_print_random_alignments(
                annotations=full.annotations,
                tokenizer=full.tokenizer,
                max_text_len=full.max_text_len,
                ids=full.ids,
                n=debug_n,
                seed=debug_seed,
            )
        return _WrappedDataset(full, _dior_rsvg_transforms("val"))

    # Train mode: random split train/val
    if set_name not in ("train", "val"):
        raise ValueError(f"Train mode only supports set_name in ('train','val'), got {set_name!r}")

    val_ratio = float(_require(args, "val_split_ratio"))

    rng = random.Random(debug_seed)
    ids_shuf = ids_all[:]
    rng.shuffle(ids_shuf)

    n = len(ids_shuf)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else n

    val_ids = ids_shuf[:n_val]
    train_ids = ids_shuf[n_val:]

    full.ids = train_ids if set_name == "train" else val_ids

    if debug_n > 0:
        _debug_print_random_alignments(
            annotations=full.annotations,
            tokenizer=full.tokenizer,
            max_text_len=full.max_text_len,
            ids=full.ids,
            n=debug_n,
            seed=debug_seed,
        )

    return _WrappedDataset(full, _dior_rsvg_transforms("train" if set_name == "train" else "val"))


# ───────────────────── convert-to-target ─────────────────────
class _ConvertDiorRSVGToTarget:
    """
    Converter:
      - boxes, labels (0-based)
      - tokens_positive spans (strict)
      - positive_map (char_to_token-based) if tokenizer provided
    """

    def __init__(self, tokenizer=None, max_text_len: int = 256):
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)

    def __call__(self, image, record: Dict):
        W, H = image.size  # PIL: (width, height)

        caption = record.get("caption", "") or ""
        anns = record.get("annotations", []) or []

        if len(anns) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            tokens_positive: List[List[Tuple[int, int]]] = []
            phrases_searched: List[str] = []
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

            kept_anns = [a for a, k in zip(anns, keep.tolist()) if k]
            tokens_positive = []
            phrases_searched = []
            for a in kept_anns:
                beg = int(a.get("tok_beg", 0))
                end = int(a.get("tok_end", 0))
                phrase = a.get("phrase", "")
                phrases_searched.append(str(phrase) if phrase is not None else "")
                if end > beg:
                    tokens_positive.append([(beg, end)])
                else:
                    tokens_positive.append([])

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=torch.tensor([record["image_id"]]),
            area=area,
            iscrowd=iscrowd,
            orig_size=torch.tensor([record["orig_size"][0], record["orig_size"][1]]),
            size=torch.tensor([H, W]),
            caption=caption,
            filename_stem=record.get("filename_stem", "N/A"),
        )

        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_offsets_mapping=True,
            )
            positive_map = create_positive_map(tokenized, tokens_positive, max_text_len=self.max_text_len)

            target["tokens_positive"] = tokens_positive
            target["phrases_searched"] = phrases_searched
            target["positive_map"] = positive_map

        return image, target
