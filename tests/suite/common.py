# Copyright (c) 2026 Nicholas Harvey.
#
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.

"""Generic synthetic data and explicitly labeled offline test components."""

from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import re
import socket
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F


def device():
    return torch.device(os.environ.get("MDETR_INTEGRITY_DEVICE", "cpu"))


def load_cases():
    path = os.environ.get("MDETR_INTEGRITY_CASES")
    raw = (
        Path(path).read_text(encoding="utf-8")
        if path
        else os.environ.get("MDETR_INTEGRITY_CASE_DATA", "")
    )
    if not raw:
        raise ValueError("No caption cases configured; run tests/run.py")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Caption cases must be a JSON object")
    if not isinstance(payload.get("images"), list) or not payload["images"]:
        raise ValueError("Cases JSON requires a nonempty 'images' list")
    aliases = payload.get("aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(key, str)
        or not (
            isinstance(value, str)
            or isinstance(value, list)
            and all(isinstance(alias, str) for alias in value)
        )
        for key, value in aliases.items()
    ):
        raise ValueError("Aliases must map class names to a string or list of strings")
    for i, image in enumerate(payload["images"]):
        if (
            not isinstance(image, dict)
            or not isinstance(image.get("caption"), str)
            or not image["caption"].strip()
        ):
            raise ValueError(f"Case {i} has no caption")
        if not isinstance(image.get("objects"), list) or not image["objects"]:
            raise ValueError(f"Case {i} has no objects")
        for obj in image["objects"]:
            if (
                not isinstance(obj, dict)
                or not isinstance(obj.get("class"), str)
                or not obj["class"].strip()
            ):
                raise ValueError("Every object needs an arbitrary nonempty class string")
            if not isinstance(obj.get("box"), list) or len(obj["box"]) != 4:
                raise ValueError("Every object's box must be [cx,cy,w,h] normalized")
    return payload


@contextmanager
def offline_network():
    """Block outbound Python DNS/IP sockets while allowing local Unix IPC."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Network access blocked: integrity tests require local files/caches")

    def guard(method):
        def guarded(sock, *args, **kwargs):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                forbidden()
            return method(sock, *args, **kwargs)

        return guarded

    with ExitStack() as stack:
        stack.enter_context(patch("socket.getaddrinfo", side_effect=forbidden))
        stack.enter_context(patch("socket.create_connection", side_effect=forbidden))
        for name in ("connect", "connect_ex", "send", "sendall", "sendto"):
            stack.enter_context(
                patch.object(socket.socket, name, guard(getattr(socket.socket, name)))
            )
        yield


def _plural(word):
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def class_spans(caption, class_name, aliases=()):
    """Fixture-only phrase matcher; aliases handle irregular names/inflections."""
    phrases = [class_name] + ([aliases] if isinstance(aliases, str) else list(aliases))
    patterns = []
    for phrase in phrases:
        words = re.split(r"[\s_-]+", phrase.casefold().strip())
        if not all(words):
            continue
        stem = r"[\s_-]+".join(re.escape(w) for w in words[:-1])
        last = "(?:" + "|".join((re.escape(_plural(words[-1])), re.escape(words[-1]))) + ")"
        patterns.append(stem + (r"[\s_-]+" if stem else "") + last)
    if not patterns:
        return []
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(sorted(patterns, key=len, reverse=True)) + r")(?!\w)", re.I
    )
    return [match.span() for match in pattern.finditer(caption)]


def fast_tokenizer(payload=None):
    """An actual Hugging Face *fast* tokenizer built locally, no downloads."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    if payload is None:
        payload = load_cases()
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    sentences = [item["caption"] for item in payload["images"]]
    for sentence in sentences:
        for token in re.findall(r"\w+|[^\w\s]", sentence):
            if token not in vocab:
                vocab[token] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=256,
        clean_up_tokenization_spaces=False,
    )


def production_maps_and_targets(payload=None, tokenizer=None, image_size=32):
    """Exercise the repository's real create_positive_map with arbitrary classes."""
    from geo_datasets.dota_preprocess import create_positive_map

    payload = payload or load_cases()
    tokenizer = tokenizer or fast_tokenizer(payload)
    names = sorted({obj["class"] for image in payload["images"] for obj in image["objects"]})
    class_to_id = {name: i for i, name in enumerate(names)}
    targets, maps = [], []
    aliases_by_class = payload.get("aliases", {})
    for image_index, item in enumerate(payload["images"]):
        spans, boxes, labels = [], [], []
        for obj in item["objects"]:
            if obj.get("spans") is not None:
                object_spans = [tuple(pair) for pair in obj["spans"]]
            else:
                object_spans = class_spans(
                    item["caption"], obj["class"], aliases_by_class.get(obj["class"], ())
                )
            if not object_spans:
                raise ValueError(
                    f"Missing class phrase for {obj['class']!r} in {item['caption']!r}; "
                    "add an alias or correct the caption"
                )
            if any(
                len(span) != 2
                or not all(isinstance(index, int) for index in span)
                or not 0 <= span[0] < span[1] <= len(item["caption"])
                for span in object_spans
            ):
                raise ValueError(f"Invalid character spans for class {obj['class']!r}")
            spans.append(object_spans)
            box = [float(v) for v in obj["box"]]
            cx, cy, w, h = box
            if not (
                w > 0
                and h > 0
                and 0 <= cx - w / 2 < cx + w / 2 <= 1
                and 0 <= cy - h / 2 < cy + h / 2 <= 1
            ):
                raise ValueError(f"Invalid normalized box for class {obj['class']!r}: {box}")
            boxes.append(box)
            labels.append(class_to_id[obj["class"]])
        tokenized = tokenizer(
            item["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )
        positive_map = create_positive_map(tokenized, spans)
        if positive_map.shape != (len(boxes), 256):
            raise AssertionError(f"Production positive_map has wrong shape {positive_map.shape}")
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "caption": item["caption"],
            "tokens_positive": spans,
            "tokens": spans,
            "phrases_searched": [obj["class"] for obj in item["objects"]],
            "positive_map": positive_map,
            "size": torch.tensor([image_size, image_size]),
            "orig_size": torch.tensor([image_size, image_size]),
            "image_id": torch.tensor([image_index]),
        }
        targets.append(target)
        maps.append(positive_map)
    return tokenizer, targets, torch.cat(maps, dim=0)


def reference_sample(value, spatial_shapes, locations, weights):
    """Pure PyTorch bilinear multi-scale sampler, independent of the MMCV extension.

    Shapes: value [B,S,H,D], locations [B,Q,H,L,P,2], weights [B,Q,H,L,P].
    """
    batch, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = locations.shape
    if levels != spatial_shapes.shape[0] or weights.shape != (
        batch,
        queries,
        heads,
        levels,
        points,
    ):
        raise ValueError("Invalid multi-scale sampling shapes")
    if int(spatial_shapes.prod(1).sum()) != value.shape[1]:
        raise ValueError("Feature length does not equal sum of per-level areas")
    offset, accum = 0, None
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        count = height * width
        feature = (
            value[:, offset : offset + count]
            .permute(0, 2, 3, 1)
            .reshape(batch * heads, dim, height, width)
        )
        grid = (
            (locations[:, :, :, level] * 2 - 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * heads, queries, points, 2)
        )
        sample = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        weight = (
            weights[:, :, :, level].permute(0, 2, 1, 3).reshape(batch * heads, 1, queries, points)
        )
        term = (sample * weight).sum(dim=-1)
        accum = term if accum is None else accum + term
        offset += count
    if offset != value.shape[1]:
        raise ValueError("Feature length does not equal sum of per-level areas")
    return (
        accum.view(batch, heads, dim, queries)
        .permute(0, 3, 1, 2)
        .reshape(batch, queries, heads * dim)
    )


class ReferenceMSDA(nn.Module):
    """MMCV-style fixture using the independent sampler; native parity is tested separately."""

    def __init__(
        self, embed_dims, num_heads, num_levels, num_points, batch_first=True, dropout=0.0
    ):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.batch_first = batch_first
        self.heads, self.levels, self.points = num_heads, num_levels, num_points
        self.offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        angles = torch.arange(num_heads, dtype=torch.float32) * (2 * math.pi / num_heads)
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        directions = directions / directions.abs().amax(-1, keepdim=True)
        grid = directions[:, None, None].repeat(1, num_levels, num_points, 1)
        grid *= torch.arange(1, num_points + 1)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(grid.flatten())

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
        identity=None,
        query_pos=None,
    ):
        identity = query if identity is None else identity
        query = query if query_pos is None else query + query_pos
        if not self.batch_first:
            query, value = query.transpose(0, 1), value.transpose(0, 1)
        expected_starts = torch.cat(
            (spatial_shapes.new_zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        if not torch.equal(level_start_index.to(expected_starts), expected_starts):
            raise ValueError("level_start_index disagrees with spatial_shapes")
        batch, queries, channels = query.shape
        val = self.value_proj(value)
        if key_padding_mask is not None:
            val = val.masked_fill(key_padding_mask[..., None], 0)
        val = val.reshape(batch, -1, self.heads, channels // self.heads)
        offsets = self.offsets(query).reshape(
            batch, queries, self.heads, self.levels, self.points, 2
        )
        scale = spatial_shapes[:, [1, 0]].to(query).reshape(1, 1, 1, self.levels, 1, 2)
        locations = reference_points[:, :, None, :, None, :] + offsets / scale
        weights = self.attention(query).reshape(
            batch, queries, self.heads, self.levels * self.points
        )
        weights = weights.softmax(dim=-1).reshape(
            batch, queries, self.heads, self.levels, self.points
        )
        out = self.output_proj(reference_sample(val, spatial_shapes, locations, weights))
        out = out if self.batch_first else out.transpose(0, 1)
        return identity + self.dropout(out)


class TinyBackbone(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = [channels] * 3
        self.num_channels = channels
        self.stages = nn.ModuleList(
            [
                nn.Conv2d(3, channels, 3, 2, 1),
                nn.Conv2d(channels, channels, 3, 2, 1),
                nn.Conv2d(channels, channels, 3, 2, 1),
            ]
        )
        from mdetr.models.position_encoding import PositionEmbeddingSine

        self.position_embedding = PositionEmbeddingSine(channels // 2, normalize=True)

    def forward(self, samples):
        from mdetr.util.misc import NestedTensor

        src, mask = samples.decompose()
        features, positions = [], []
        for conv in self.stages:
            src = F.relu(conv(src))
            mask = F.interpolate(mask[:, None].float(), size=src.shape[-2:], mode="nearest")[
                :, 0
            ].bool()
            feature = NestedTensor(src, mask)
            features.append(feature)
            positions.append(self.position_embedding(feature))
        return features, positions


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=24):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.norm(self.embedding(input_ids)))


def make_tiny_model(payload=None, target_device=None, use_text_cross=True):
    """Real MDETR-G + real encoder/decoder; only pretrained dependencies are tiny/offline."""
    import mdetr.models.transformer as module
    from mdetr.models.mdetr import MDETR

    payload = payload or load_cases()
    target_device = target_device or device()
    tokenizer = fast_tokenizer(payload)
    text_encoder = TinyTextEncoder(tokenizer.vocab_size)
    native = (
        target_device.type == "cuda"
        and module._HAS_MMCV
        and os.getenv("MDETR_INTEGRITY_FORCE_REFERENCE", "0") != "1"
    )

    with (
        patch.object(module.AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch.object(module.AutoModel, "from_pretrained", return_value=text_encoder),
    ):
        if native:
            transformer = module.Transformer(
                d_model=32,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=2,
                dim_feedforward=64,
                dropout=0.0,
                num_feature_levels=3,
                deform_num_points=2,
                use_text_cross_attn=use_text_cross,
                text_encoder_type="test-only",
                freeze_text_encoder=True,
            )
        else:
            with (
                patch.object(module, "_HAS_MMCV", True),
                patch.object(module, "_MMCV_MSDA", ReferenceMSDA, create=True),
            ):
                transformer = module.Transformer(
                    d_model=32,
                    nhead=4,
                    num_encoder_layers=1,
                    num_decoder_layers=2,
                    dim_feedforward=64,
                    dropout=0.0,
                    num_feature_levels=3,
                    deform_num_points=2,
                    use_text_cross_attn=use_text_cross,
                    text_encoder_type="test-only",
                    freeze_text_encoder=True,
                )
    model = MDETR(
        TinyBackbone(),
        transformer,
        num_classes=255,
        num_queries=max(12, max(len(image["objects"]) for image in payload["images"])),
        aux_loss=True,
        contrastive_align_loss=True,
        align_scale_mode="learnable",
    ).to(target_device)
    model.eval()
    model._integrity_attention_backend = (
        "MMCV CUDA" if native else "PyTorch reference (synthetic wiring only)"
    )
    return model, tokenizer


def synthetic_samples(target_device=None, count=2):
    from mdetr.util.misc import NestedTensor

    target_device = target_device or device()
    images = []
    for i in range(count):
        gen = torch.Generator().manual_seed(1023 + i)
        image = torch.rand((3, 32, 32), generator=gen)
        # Spatially non-constant signal tests actual image conditioning.
        image[0] += torch.linspace(0, 0.5 + i * 0.25, 32)[None, :]
        images.append(image)
    return NestedTensor(
        torch.stack(images).to(target_device),
        torch.zeros((count, 32, 32), dtype=torch.bool, device=target_device),
    )


def weighted_loss(losses, weights):
    return sum(value * weights[name] for name, value in losses.items() if name in weights)
