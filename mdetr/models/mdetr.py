# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.
"""MDETR model, losses, and build utilities."""

import math

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import mdetr.util.dist as dist
from mdetr.util import box_ops
from mdetr.util.misc import NestedTensor
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import Transformer, build_transformer


def _bounded_logit_scale(log_scale):
    """Cap before exponentiation to keep saturated-scale gradients finite."""
    value = log_scale.float() if log_scale.dtype in (torch.float16, torch.bfloat16) else log_scale
    return value.clamp(max=math.log(100.0)).exp().clamp(max=100.0).to(log_scale.dtype)


class MDETR(nn.Module):
    """Modulated DETR with optional image–text contrastive losses."""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss: bool = False,
        contrastive_hdim: int = 64,
        contrastive_loss: bool = False,
        contrastive_align_loss: bool = False,
        cls_scale_mode: str = "learnable",
        align_scale_mode: str = "learnable",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        if hasattr(transformer, "tokenizer"):
            self.tokenizer = transformer.tokenizer

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.is_deformable = isinstance(transformer, Transformer)
        if self.is_deformable:
            print("[Decoder] Reference-relative boxes; predicted centers seed the next layer")

        # Project each selected backbone level to the transformer width.
        self.multi_scale = getattr(transformer, "multi_scale", False)
        self.num_feature_levels = getattr(transformer, "num_feature_levels", 1)
        if self.multi_scale and self.num_feature_levels > 1:
            if hasattr(backbone, "channels"):
                in_ch = backbone.channels[-self.num_feature_levels :]
            else:
                in_ch = [backbone.num_channels] * self.num_feature_levels
            self.input_proj = nn.ModuleList(
                [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_ch]
            )
        else:
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.cls_scale_mode = cls_scale_mode
        self.logit_scale_cls = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        if self.cls_scale_mode == "learnable":
            self.logit_scale_cls.requires_grad_(True)
        elif self.cls_scale_mode == "off":
            self.logit_scale_cls.requires_grad_(False)
        else:
            raise ValueError(f"Unknown cls_scale_mode={self.cls_scale_mode!r}")

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
            txt_h = getattr(self.transformer, "text_hidden_dim", None)
            if txt_h is None:
                txt_h = self.transformer.text_encoder.config.hidden_size
            self.contrastive_projection_text = nn.Linear(txt_h, contrastive_hdim, bias=False)

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

            # Keep in state_dict across modes for checkpoint compatibility.
            self.logit_scale_align = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            self.align_scale_mode = align_scale_mode

            if self.align_scale_mode == "learnable":
                self.logit_scale_align.requires_grad_(True)
            elif self.align_scale_mode == "hardcoded":
                self.logit_scale_align.requires_grad_(False)
            else:
                raise ValueError(f"Unknown align_scale_mode={self.align_scale_mode!r}")
        else:
            self.logit_scale_align = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Reject incompatible reference heads even when strict=False.
        if self.is_deformable and any(
            "transformer.decoder.refine_head." in key for key in state_dict
        ):
            raise RuntimeError(
                "This checkpoint contains the removed independent-reference decoder heads. "
                "It cannot be loaded into this model. Start a fresh training run."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self, samples: NestedTensor, captions, encode_and_save: bool = True, memory_cache=None
    ):
        """Encode into a cache (encode_and_save=True) or decode from an existing cache."""
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None

            features, pos = self.backbone(samples)

            srcs, masks, poses2 = [], [], []
            for lvl, (f, p) in enumerate(zip(features, pos)):
                s, m = f.decompose()
                proj = (
                    self.input_proj[lvl](s)
                    if isinstance(self.input_proj, nn.ModuleList)
                    else self.input_proj(s)
                )
                srcs.append(proj)
                masks.append(m)
                poses2.append(p)

            memory_cache = self.transformer(
                srcs,
                masks,
                self.query_embed.weight,
                poses2,
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            if self.contrastive_loss:
                if memory_cache.get("text_pooled_op") is not None:
                    memory_cache["text_pooled_op"] = self.contrastive_projection_text(
                        memory_cache["text_pooled_op"]
                    )
                if memory_cache.get("img_pooled_op") is not None:
                    memory_cache["img_pooled_op"] = self.contrastive_projection_image(
                        memory_cache["img_pooled_op"]
                    )

            return memory_cache

        assert memory_cache is not None

        decoder_kwargs = {"bbox_embed": self.bbox_embed} if self.is_deformable else {}
        decoded = self.transformer(
            mask=memory_cache["mask"],
            query_embed=memory_cache["query_embed"],
            pos_embed=memory_cache["pos_embed"],
            encode_and_save=False,
            text_memory=memory_cache["text_memory_resized"],
            img_memory=memory_cache["img_memory"],
            text_attention_mask=memory_cache["text_attention_mask"],
            spatial_shapes=memory_cache.get("spatial_shapes"),
            level_start_index=memory_cache.get("level_start_index"),
            valid_ratios=memory_cache.get("valid_ratios"),
            **decoder_kwargs,
        )

        if self.is_deformable:
            hs, outputs_coord = decoded
        else:
            # Vanilla attention is a separate architecture with no sampling references.
            hs = decoded
            outputs_coord = self.bbox_embed(hs).sigmoid()
        if torch.isnan(outputs_coord).any():
            raise RuntimeError("NaNs in pred_boxes – check token masking and attention masks.")

        outputs_class = self.class_embed(hs)
        if self.cls_scale_mode == "learnable":
            outputs_class = outputs_class * _bounded_logit_scale(self.logit_scale_cls)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }

        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1),
                p=2,
                dim=-1,
            )
            align_scale = (
                _bounded_logit_scale(self.logit_scale_align)
                if self.logit_scale_align is not None
                else None
            )
            out.update(
                {
                    "proj_queries": proj_queries[-1],
                    "proj_tokens": proj_tokens,
                    "tokenized": memory_cache["tokenized"],
                    "logit_scale_align": align_scale,
                }
            )

            if self.aux_loss:
                shared_alignment = {
                    "proj_tokens": proj_tokens,
                    "tokenized": memory_cache["tokenized"],
                }
                if self.align_scale_mode == "learnable" and align_scale is not None:
                    shared_alignment["logit_scale_align"] = align_scale
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                        "proj_queries": c,
                        **shared_alignment,
                    }
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                ]
        elif self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return out


class ContrastiveCriterion(nn.Module):
    """Symmetric image↔text InfoNCE loss on pooled embeddings."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):
        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)
        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature

        labels = torch.arange(logits.size(0), device=pooled_image.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2.0


class SetCriterion(nn.Module):
    """Loss computation for DETR-like models."""

    def __init__(
        self,
        num_classes,
        matcher,
        eos_coef,
        losses,
        temperature,
        align_scale_mode: str = "learnable",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        self.align_scale_mode = align_scale_mode

        # Preserve the criterion state-dict schema used by existing callers.
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
        """NLL loss against per-box token supervision (last index is background)."""
        logits = outputs["pred_logits"].float().log_softmax(-1)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1
        loss_ce = (loss_ce * eos_coef).sum() / num_boxes

        return {"loss_ce": loss_ce}

    def _validate_token_supervision(self, outputs, targets, positive_map):
        """Require token maps and character spans to supervise the same objects."""
        tokenized = outputs.get("tokenized")
        require_spans = "contrastive_align" in self.losses
        if tokenized is None:
            if require_spans:
                raise ValueError("Contrastive alignment requires the model's tokenized captions")
            return
        input_ids = tokenized.input_ids
        if (
            not torch.is_tensor(input_ids)
            or input_ids.ndim != 2
            or input_ids.shape[0] != len(targets)
        ):
            raise ValueError("Model token IDs must have shape [batch, tokens]")
        token_count = input_ids.shape[1]
        if "proj_tokens" in outputs and outputs["proj_tokens"].shape[:2] != input_ids.shape:
            raise ValueError("Projected text and the model's tokenized captions disagree in shape")
        attention_mask = getattr(tokenized, "attention_mask", None)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("Model text attention mask and token IDs disagree in shape")
            attention_mask = attention_mask.detach().cpu().bool()

        # Tokenization and span validation run on CPU; retain fractional targets.
        supplied = positive_map.detach().cpu() > 0
        offset = 0
        for image_index, target in enumerate(targets):
            count = len(target["boxes"])
            spans_per_object = target.get("tokens_positive", target.get("tokens"))
            if spans_per_object is None and count and require_spans:
                raise ValueError(
                    f"Contrastive alignment: batch {image_index}: missing per-object character spans"
                )
            if spans_per_object is not None and len(spans_per_object) != count:
                raise ValueError(
                    f"Contrastive alignment: batch {image_index}: expected one span list per target box"
                )
            for object_index in range(count):
                context = f"Contrastive alignment: batch {image_index}, target {object_index}"
                row = supplied[offset + object_index]
                if row[token_count:].any():
                    raise ValueError(
                        f"{context}: positive_map includes positions beyond the model's text tokens"
                    )
                if attention_mask is not None:
                    width = min(len(row), token_count)
                    if (row[:width] & ~attention_mask[image_index, :width]).any():
                        raise ValueError(f"{context}: positive_map includes padded text tokens")
                if spans_per_object is None:
                    continue
                object_spans = spans_per_object[object_index]
                if not object_spans:
                    raise ValueError(
                        f"{context}: foreground target has an empty character-span list"
                    )
                expected = torch.zeros_like(row)
                for beg, end in object_spans:
                    span_context = f"{context}, character span ({beg}, {end})"
                    if "caption" in target:
                        span_context += f", caption={target['caption']!r}"
                    if beg < 0 or end <= beg:
                        raise ValueError(f"{span_context}: expected a nonempty, nonnegative span")
                    try:
                        begin_token = tokenized.char_to_token(image_index, beg)
                        end_token = tokenized.char_to_token(image_index, end - 1)
                    except (IndexError, ValueError, TypeError, OverflowError) as exc:
                        raise ValueError(f"{span_context}: exact token lookup failed") from exc
                    if begin_token is None or end_token is None:
                        raise ValueError(
                            f"{span_context}: exact endpoints do not map to tokens "
                            "(check the caption, character offsets, and truncation)"
                        )
                    if not (0 <= begin_token <= end_token < min(token_count, len(row) - 1)):
                        raise ValueError(
                            f"{span_context}: span reaches the no-object column "
                            "or is outside the model's foreground text tokens"
                        )
                    expected[begin_token : end_token + 1] = True
                if not torch.equal(row, expected):
                    actual_indices = row.nonzero().flatten().tolist()
                    expected_indices = expected.nonzero().flatten().tolist()
                    raise ValueError(
                        f"{context}: positive_map tokens {actual_indices} disagree with character-span "
                        f"tokens {expected_indices}; check target order, caption tokenization, and cached maps"
                    )
            offset += count

    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        tokenized = outputs["tokenized"]
        normalized_text_emb = outputs["proj_tokens"].float()  # [B, T, h]
        normalized_img_emb = outputs["proj_queries"].float()  # [B, Q, h]

        # Keep similarity reductions in FP32 under autocast.
        with torch.autocast(device_type=normalized_img_emb.device.type, enabled=False):
            logits = torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2))
            if self.align_scale_mode == "learnable":
                scale = outputs.get("logit_scale_align", None)
                logits = logits / self.temperature if scale is None else logits * scale
            elif self.align_scale_mode == "hardcoded":
                logits = logits / self.temperature
            else:
                raise ValueError(f"Unknown align_scale_mode={self.align_scale_mode!r}")

        # Mask PAD tokens (attention_mask: 1=valid, 0=PAD).
        if hasattr(tokenized, "attention_mask"):
            pad_mask = (tokenized.attention_mask == 0).to(logits.device)
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)

        # Matching, soft-token CE, and alignment share the validated token rows.
        token_count = logits.shape[-1]
        support = positive_map[:, :token_count] > 0
        if support.shape[1] < token_count:
            support = F.pad(support, (0, token_count - support.shape[1]))
        pos_map = torch.zeros_like(logits, dtype=torch.bool)
        offset = 0
        for image_index, ((source, target), item) in enumerate(zip(indices, targets)):
            pos_map[image_index, source] = support[target + offset].to(logits.device)
            offset += len(item["boxes"])

        positive_logits = -logits.masked_fill(~pos_map, 0)
        negative_logits = logits

        boxes_with_pos = pos_map.any(2)
        nb_pos_b2t = pos_map.sum(2) + 1e-6
        loss_b2t = (
            (positive_logits.sum(2) / nb_pos_b2t + negative_logits.logsumexp(2))
            .masked_fill(~boxes_with_pos, 0)
            .sum()
        )

        tokens_with_pos = pos_map.any(1)
        nb_pos_t2b = pos_map.sum(1) + 1e-6
        loss_t2b = (
            (positive_logits.sum(1) / nb_pos_t2b + negative_logits.logsumexp(1))
            .masked_fill(~tokens_with_pos, 0)
            .sum()
        )

        total = (loss_b2t + loss_t2b) / 2
        return {"loss_contrastive_align": total / num_boxes}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device

        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        # FP32 area products avoid underflow for small boxes without detaching.
        src_boxes = outputs["pred_boxes"][idx].float()
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).float()

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "contrastive_align": self.loss_contrastive_align,
        }
        if loss not in loss_map:
            raise ValueError(f"Unsupported loss: {loss!r}")
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, positive_map):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # One normalized row per object; multiple positive spans are supported.
        logit_dim = outputs_without_aux["pred_logits"].shape[-1]
        if not torch.is_tensor(positive_map) or positive_map.ndim != 2:
            raise ValueError("positive_map must be a 2-D tensor with one row per target box")
        if len(targets) != outputs_without_aux["pred_logits"].shape[0]:
            raise ValueError("Target count must match the model batch size")
        target_count = sum(len(t["boxes"]) for t in targets)
        if positive_map.shape[0] != target_count:
            raise ValueError(
                f"positive_map has {positive_map.shape[0]} rows for {target_count} target boxes"
            )
        if any(len(t["labels"]) != len(t["boxes"]) for t in targets):
            raise ValueError("Each target box must have one label")
        if not torch.isfinite(positive_map).all() or (positive_map < 0).any():
            raise ValueError("positive_map weights must be finite and nonnegative")
        row_sums = positive_map.sum(-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=0):
            raise ValueError("Each target box's positive_map weights must sum to one")
        # The last logit is background, not a caption token.
        if (positive_map[:, logit_dim - 1 :] != 0).any():
            raise ValueError(
                "positive_map assigns foreground tokens to the no-object column "
                "or beyond the classifier's token capacity"
            )

        # Zero padding/trimming is safe; nonzero removed columns were rejected.
        if positive_map.shape[1] > logit_dim:
            pos_map = positive_map[:, :logit_dim]
        elif positive_map.shape[1] < logit_dim:
            pos_map = F.pad(positive_map, (0, logit_dim - positive_map.shape[1]))
        else:
            pos_map = positive_map

        self._validate_token_supervision(outputs_without_aux, targets, pos_map)
        indices = self.matcher(outputs_without_aux, targets, pos_map)

        num_boxes = torch.as_tensor(
            [target_count], dtype=torch.float, device=outputs_without_aux["pred_logits"].device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, pos_map, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, pos_map)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, pos_map, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """Simple feed-forward MLP."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    """Build model, criteria, and loss weights from args."""
    if getattr(args, "masks", False):
        raise ValueError("This model supports bounding-box detection, not mask prediction")
    num_classes = getattr(args, "num_classes", 255)
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        cls_scale_mode=getattr(args, "cls_scale_mode", "learnable"),
        align_scale_mode=getattr(args, "align_scale_mode", "learnable"),
    )

    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce": args.ce_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }

    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
            align_scale_mode=getattr(args, "align_scale_mode", "learnable"),
        ).to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE).to(device)
    else:
        contrastive_criterion = None

    # Extra optimizer group for learnable temperature parameters.
    extra = []
    if model.logit_scale_cls.requires_grad:
        extra.append(model.logit_scale_cls)
    if model.logit_scale_align is not None and model.logit_scale_align.requires_grad:
        extra.append(model.logit_scale_align)

    model.extra_optim_groups = []
    if extra:
        model.extra_optim_groups.append(
            {
                "name": "logit_scales",
                "params": extra,
                "lr": getattr(args, "logit_scale_lr", args.lr),
                "base_lr": getattr(args, "logit_scale_lr", args.lr),
                "weight_decay": 0.0,
            }
        )

    return model, criterion, contrastive_criterion, weight_dict
