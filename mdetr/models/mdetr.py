from typing import Optional
import math

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import multimodal_framework.mdetr.util.dist as dist
from multimodal_framework.mdetr.util import box_ops
from multimodal_framework.mdetr.util.misc import NestedTensor, interpolate
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import dice_loss, sigmoid_focal_loss
from .transformer import build_transformer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MDETR(nn.Module):
    """
    Modulated DETR:
    - Multi-scale deformable encoder/decoder over image features
    - Text encoder fused via cross-attention inside decoder (optionally)
    - **Fixed** 255-token linear classification head (+1 background)
    - Optional global contrastive and token–box contrastive align losses
    """

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
        qa_dataset: Optional[str] = None,
        split_qa_heads: bool = True,
        predict_final: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # expose tokenizer if present (used by datasets & alignment)
        if hasattr(transformer, "tokenizer"):
            self.tokenizer = transformer.tokenizer

        # heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # fixed 255 (+background)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ---- Multi-level projections (one per backbone stage) --------------
        self.multi_scale = getattr(transformer, "multi_scale", False)
        self.num_feature_levels = getattr(transformer, "num_feature_levels", 1)
        if self.multi_scale and self.num_feature_levels > 1:
            if hasattr(backbone, "channels"):
                in_ch = backbone.channels[-self.num_feature_levels:]  # e.g., last 3–4 levels
            else:
                in_ch = [backbone.num_channels] * self.num_feature_levels
            self.input_proj = nn.ModuleList(
                [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_ch]
            )
        else:
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # Learnable temperatures
        self.logit_scale_cls = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # for cls logits

        self.backbone = backbone
        self.aux_loss = aux_loss

        # ---- global image–text contrastive (optional) ----------------------
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
            # robust to any HF text encoder
            txt_h = getattr(self.transformer, "text_hidden_dim", None)
            if txt_h is None:
                txt_h = self.transformer.text_encoder.config.hidden_size
            self.contrastive_projection_text = nn.Linear(txt_h, contrastive_hdim, bias=False)

        # ---- token–box contrastive alignment (optional) --------------------
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)
            self.logit_scale_align = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        else:
            self.logit_scale_align = None

        # (Optional) QA heads (kept for completeness; not used in your setup)
        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if qa_dataset is not None:
            if split_qa_heads:
                self.answer_type_head = nn.Linear(hidden_dim, 5)
                if qa_dataset == "gqa":
                    self.answer_rel_head = nn.Linear(hidden_dim, 1594)
                    self.answer_obj_head = nn.Linear(hidden_dim, 3)
                    self.answer_global_head = nn.Linear(hidden_dim, 111)
                    self.answer_attr_head = nn.Linear(hidden_dim, 403)
                    self.answer_cat_head = nn.Linear(hidden_dim, 678)
                elif qa_dataset == "clevr":
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_binary_head = nn.Linear(hidden_dim, 1)
                    self.answer_attr_head = nn.Linear(hidden_dim, 15)
                    self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)
                else:
                    assert False, f"Invalid qa dataset {qa_dataset}"
            else:
                assert qa_dataset == "gqa", "Clevr QA is not supported with unified head"
                self.answer_head = nn.Linear(hidden_dim, 1853)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, samples: NestedTensor, captions, encode_and_save: bool = True, memory_cache=None):
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        # ------------------------ 1) ENCODE & CACHE -------------------------
        if encode_and_save:
            assert memory_cache is None

            features, pos = self.backbone(samples)  # these lists are already length K
            feats = features
            poses = pos

            srcs, masks, poses2 = [], [], []
            for lvl, (f, p) in enumerate(zip(feats, poses)):
                s, m = f.decompose()
                proj = self.input_proj[lvl](s) if isinstance(self.input_proj, nn.ModuleList) else self.input_proj(s)
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

        # --------------------------- 2) DECODE -------------------------------
        else:
            assert memory_cache is not None

            hs = self.transformer(
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
            )

            out = {}

            # (Optional) QA heads
            if self.qa_dataset is not None:
                if self.split_qa_heads:
                    if self.qa_dataset == "gqa":
                        answer_embeds = hs[0, :, -6:]
                        hs = hs[:, :, :-6]
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
                        out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                        out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
                        out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
                    elif self.qa_dataset == "clevr":
                        answer_embeds = hs[0, :, -4:]
                        hs = hs[:, :, :-4]
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
                        out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                    else:
                        assert False, f"Invalid qa dataset {self.qa_dataset}"
                else:
                    answer_embeds = hs[0, :, -1]
                    hs = hs[:, :, :-1]
                    out["pred_answer"] = self.answer_head(answer_embeds)

            # Boxes
            outputs_coord = self.bbox_embed(hs).sigmoid()
            if torch.isnan(outputs_coord).any():
                raise RuntimeError("NaNs in pred_boxes – check token masking and attention masks.")

            # Fixed linear classification head (+ temperature)
            outputs_class = self.class_embed(hs)  # [L,B,Q, num_classes+1]
            outputs_class = outputs_class * self.logit_scale_cls.exp().clamp(max=100.0)

            out.update(
                {
                    "pred_logits": outputs_class[-1],  # [B,Q,C]
                    "pred_boxes": outputs_coord[-1],   # [B,Q,4]
                }
            )

            # Optional "is-final" head (CLEVR-Ref+)
            outputs_isfinal = None
            if self.isfinal_embed is not None:
                outputs_isfinal = self.isfinal_embed(hs)
                out["pred_isfinal"] = outputs_isfinal[-1]

            # Token–box contrastive alignment outputs
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1),
                    p=2,
                    dim=-1,
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                        "logit_scale_align": (
                            self.logit_scale_align.exp().clamp(max=100.0)
                            if self.logit_scale_align is not None
                            else None
                        ),
                    }
                )

            # Aux losses from intermediate decoder layers
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                    ]
                else:
                    out["aux_outputs"] = [
                        {"pred_logits": a, "pred_boxes": b}
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                    ]
                if outputs_isfinal is not None:
                    assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                    for i in range(len(outputs_isfinal[:-1])):
                        out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]

            return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class ContrastiveCriterion(nn.Module):
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


class QACriterionGQA(nn.Module):
    def __init__(self, split_qa_heads: bool):
        super().__init__()
        self.split_qa_heads = split_qa_heads

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.cross_entropy(
                output["pred_answer"], answers["answer"], reduction="mean"
            )
            attr_total = (output["pred_answer"].argmax(-1)) == answers["answer"]
            loss["accuracy_answer_total"] = attr_total.float().mean()
            return loss

        device = output["pred_answer_type"].device

        loss["loss_answer_type"] = F.cross_entropy(
            output["pred_answer_type"], answers["answer_type"]
        )
        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_obj = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_rel = answers["answer_type"] == 2
        is_global = answers["answer_type"] == 3
        is_cat = answers["answer_type"] == 4

        obj_norm = is_obj.sum() if is_obj.any() else 1.0
        loss["loss_answer_obj"] = (
            F.cross_entropy(
                output["pred_answer_obj"], answers["answer_obj"], reduction="none"
            )
            .masked_fill(~is_obj, 0)
            .sum()
            / obj_norm
        )
        obj_acc = (output["pred_answer_obj"].argmax(-1)) == answers["answer_obj"]
        loss["accuracy_answer_obj"] = (
            obj_acc[is_obj].sum() / is_obj.sum()
            if is_obj.any()
            else torch.as_tensor(1.0, device=device)
        )

        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(
                output["pred_answer_attr"], answers["answer_attr"], reduction="none"
            )
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum()
            if is_attr.any()
            else torch.as_tensor(1.0, device=device)
        )

        rel_norm = is_rel.sum() if is_rel.any() else 1.0
        loss["loss_answer_rel"] = (
            F.cross_entropy(
                output["pred_answer_rel"], answers["answer_rel"], reduction="none"
            )
            .masked_fill(~is_rel, 0)
            .sum()
            / rel_norm
        )
        rel_acc = (output["pred_answer_rel"].argmax(-1)) == answers["answer_rel"]
        loss["accuracy_answer_rel"] = (
            rel_acc[is_rel].sum() / is_rel.sum()
            if is_rel.any()
            else torch.as_tensor(1.0, device=device)
        )

        global_norm = is_global.sum() if is_global.any() else 1.0
        loss["loss_answer_global"] = (
            F.cross_entropy(
                output["pred_answer_global"], answers["answer_global"], reduction="none"
            )
            .masked_fill(~is_global, 0)
            .sum()
            / global_norm
        )
        global_acc = (output["pred_answer_global"].argmax(-1)) == answers["answer_global"]
        loss["accuracy_answer_global"] = (
            global_acc[is_global].sum() / is_global.sum()
            if is_global.any()
            else torch.as_tensor(1.0, device=device)
        )

        cat_norm = is_cat.sum() if is_cat.any() else 1.0
        loss["loss_answer_cat"] = (
            F.cross_entropy(
                output["pred_answer_cat"], answers["answer_cat"], reduction="none"
            )
            .masked_fill(~is_cat, 0)
            .sum()
            / cat_norm
        )
        cat_acc = (output["pred_answer_cat"].argmax(-1)) == answers["answer_cat"]
        loss["accuracy_answer_cat"] = (
            cat_acc[is_cat].sum() / is_cat.sum()
            if is_cat.any()
            else torch.as_tensor(1.0, device=device)
        )

        loss["accuracy_answer_total"] = (
            type_acc
            * (
                is_obj * obj_acc
                + is_rel * rel_acc
                + is_attr * attr_acc
                + is_global * global_acc
                + is_cat * cat_acc
            )
        ).sum() / type_acc.numel()

        return loss


class QACriterionClevr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, answers):
        loss = {}

        loss["loss_answer_type"] = F.cross_entropy(
            output["pred_answer_type"], answers["answer_type"]
        )
        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_binary = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_reg = answers["answer_type"] == 2

        binary_norm = is_binary.sum() if is_binary.any() else 1.0
        loss["loss_answer_binary"] = (
            F.binary_cross_entropy_with_logits(
                output["pred_answer_binary"], answers["answer_binary"], reduction="none"
            )
            .masked_fill(~is_binary, 0)
            .sum()
            / binary_norm
        )
        bin_acc = (output["pred_answer_binary"].sigmoid() > 0.5) == answers["answer_binary"]
        loss["accuracy_answer_binary"] = (
            bin_acc[is_binary].sum() / is_binary.sum() if is_binary.any() else torch.as_tensor(1.0)
        )

        reg_norm = is_reg.sum() if is_reg.any() else 1.0
        loss["loss_answer_reg"] = (
            F.cross_entropy(
                output["pred_answer_reg"], answers["answer_reg"], reduction="none"
            )
            .masked_fill(~is_reg, 0)
            .sum()
            / reg_norm
        )
        reg_acc = (output["pred_answer_reg"].argmax(-1)) == answers["answer_reg"]
        loss["accuracy_answer_reg"] = (
            reg_acc[is_reg].sum() / is_reg.sum() if is_reg.any() else torch.as_tensor(1.0)
        )

        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(
                output["pred_answer_attr"], answers["answer_attr"], reduction="none"
            )
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0)
        )

        loss["accuracy_answer_total"] = (
            type_acc * (is_binary * bin_acc + is_reg * reg_acc + is_attr * attr_acc)
        ).sum() / type_acc.numel()

        return loss


class SetCriterion(nn.Module):
    """Loss computation for DETR-like models."""

    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_isfinal(self, outputs, targets, positive_map, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_isfinal = outputs["pred_isfinal"][idx].squeeze(-1)
        target_isfinal = torch.cat(
            [t["isfinal"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss_isfinal = F.binary_cross_entropy_with_logits(
            src_isfinal, target_isfinal, reduction="none"
        )
        losses = {"loss_isfinal": loss_isfinal.sum() / num_boxes}

        acc = (src_isfinal.sigmoid() > 0.5) == (target_isfinal > 0.5)
        acc = acc.sum() if acc.numel() == 0 else acc.float().mean()
        losses["accuracy_isfinal"] = acc
        return losses

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
        """Classification loss (NLL) against per-box → token-index supervision (last index = background)."""
        logits = outputs["pred_logits"].log_softmax(-1)  # [B,Q, num_tokens(+bg)]

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]  # (sum_K, num_tokens), no background

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1  # default to background
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1
        loss_ce = (loss_ce * eos_coef).sum() / num_boxes

        return {"loss_ce": loss_ce}

    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        tokenized = outputs["tokenized"]
        normalized_text_emb = outputs["proj_tokens"]   # [B, T, h]
        normalized_img_emb = outputs["proj_queries"]   # [B, Q, h]

        # Use model-provided scale if present; otherwise fall back to 1/temperature
        scale = outputs.get("logit_scale_align", None)
        if scale is None:
            logits = torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        else:
            logits = torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) * scale

        # Mask PAD tokens (attention_mask: 1=valid, 0=PAD)
        if hasattr(tokenized, "attention_mask"):
            pad_mask = (tokenized.attention_mask == 0).to(logits.device)  # [B, T], True=PAD
            NEG_LARGE = -1e9
            logits = logits.masked_fill(pad_mask[:, None, :], NEG_LARGE)

        # build pos_map via char_to_token
        pos_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)

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

                    pos_map[i, idx_src[j], beg_pos : end_pos + 1] = True

        pos_map = pos_map.to(logits.device)

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
        # Count predictions that are NOT background (last logit)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

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

    def loss_masks(self, outputs, targets, positive_map, indices, num_boxes):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks[tgt_idx].flatten(1)

        return {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

    # ---- utils ----
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "isfinal": self.loss_isfinal,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, positive_map):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # ---- align positive_map width to the model's token dimension ----
        logit_dim = outputs_without_aux["pred_logits"].shape[-1]  # includes background
        if positive_map is None:
            pos_map = None
        else:
            if positive_map.shape[1] != logit_dim:
                if positive_map.shape[1] > logit_dim:
                    # Dataset positive_map is wider; truncate overflow
                    pos_map = positive_map[:, :logit_dim]
                else:
                    # Add a zero column to account for background
                    pad_cols = logit_dim - positive_map.shape[1]
                    pos_map = torch.nn.functional.pad(positive_map, (0, pad_cols))
            else:
                pos_map = positive_map
        # ----------------------------------------------------------------

        indices = self.matcher(outputs_without_aux, targets, pos_map)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
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
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, pos_map, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(args):
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
    )

    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce": args.ce_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    if getattr(args, "masks", False):
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
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
    if getattr(args, "masks", False):
        losses += ["masks"]
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
        ).to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE).to(device)
    else:
        contrastive_criterion = None

    # ---- extra optim groups (learnable temperatures) -----------------------
    extra = []
    if hasattr(model, "logit_scale_cls") and model.logit_scale_cls is not None:
        extra.append(model.logit_scale_cls)
    if hasattr(model, "logit_scale_align") and model.logit_scale_align is not None:
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

    return model, criterion, contrastive_criterion, None, weight_dict
