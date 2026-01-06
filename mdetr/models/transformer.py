import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from .transformer_vanilla import VanillaTransformer


# ---- Try to import the CUDA op from MMCV -----------------------------------
_HAS_MMCV = False
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as _MMCV_MSDA

    _HAS_MMCV = True
except Exception:
    try:
        from mmcv.ops import MultiScaleDeformableAttention as _MMCV_MSDA

        _HAS_MMCV = True
    except Exception:
        _HAS_MMCV = False


def _mean_pool(last_hidden: Tensor, attn_mask: Tensor) -> Tensor:
    """
    Mean-pool token embeddings w.r.t. the attention mask.
    last_hidden: [B, T, H], attn_mask: [B, T] (1=valid, 0=pad)
    """
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    summed = (last_hidden * mask).sum(1)  # [B,H]
    denom = mask.sum(1).clamp_min(1e-6)  # [B,1]
    return summed / denom


class _MSDeformAttnWrapper(nn.Module):
    def __init__(self, d_model: int, n_levels: int, n_heads: int, n_points: int):
        super().__init__()
        if not _HAS_MMCV:
            raise ImportError(
                "MultiScaleDeformableAttention requires MMCV w/ CUDA ops. "
                "Install a wheel that matches your Torch/CUDA."
            )

        mod = None
        try:
            mod = _MMCV_MSDA(
                embed_dims=d_model,
                num_heads=n_heads,
                num_levels=n_levels,
                num_points=n_points,
                batch_first=True,
            )
            self.batch_first = True
        except TypeError:
            mod = _MMCV_MSDA(d_model, n_heads, n_levels, n_points)
            self.batch_first = getattr(mod, "batch_first", False)

        self.mod = mod

    def forward(
        self,
        query: Tensor,  # [S_q,B,C]
        reference_points: Tensor,  # [B,S_q,L,2]
        value: Tensor,  # [S_v,B,C]
        spatial_shapes: Tensor,  # [L,2]
        level_start_index: Tensor,  # [L]
        key_padding_mask: Optional[Tensor] = None,  # [B,S_v]
    ) -> Tensor:
        if self.batch_first:
            q = query.transpose(0, 1).contiguous()  # [B,S_q,C]
            v = value.transpose(0, 1).contiguous()  # [B,S_v,C]
            len_dim = 1
        else:
            q = query.contiguous()  # [S_q,B,C]
            v = value.contiguous()  # [S_v,B,C]
            len_dim = 0

        spatial_shapes = spatial_shapes.to(dtype=torch.long, device=v.device).contiguous()
        level_start_index = level_start_index.to(dtype=torch.long, device=v.device).contiguous()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=v.device).contiguous()

        reference_points = reference_points.to(device=v.device, dtype=v.dtype)
        reference_points = reference_points.clamp(1e-5, 1 - 1e-5).contiguous()

        s_total = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if v.shape[len_dim] != s_total:
            raise RuntimeError(
                f"MSDA precheck failed: value_len={v.shape[len_dim]} but sum(HxW)={s_total}; "
                f"spatial_shapes={spatial_shapes.tolist()}, batch_first={self.batch_first}"
            )

        try:
            out = self.mod(
                query=q,
                reference_points=reference_points,
                value=v,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )
        except TypeError:
            out = self.mod(q, reference_points, v, spatial_shapes, level_start_index, key_padding_mask)

        return out.transpose(0, 1).contiguous() if self.batch_first else out.contiguous()


class FeatureResizer(nn.Module):
    """Linear + LN + Dropout to map text hidden size -> d_model."""

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        if self.do_ln:
            x = self.layer_norm(x)
        return self.dropout(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Unsupported activation {activation}")


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def _get_valid_ratio(mask: Tensor) -> Tensor:
    B, H, W = mask.shape
    valid_H = (~mask).any(dim=2).sum(dim=1)
    valid_W = (~mask).any(dim=1).sum(dim=1)
    ratio_h = valid_H.float() / H
    ratio_w = valid_W.float() / W
    return torch.stack([ratio_w, ratio_h], dim=-1)  # [B,2]


def _get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device):
    B, L = valid_ratios.shape[0], spatial_shapes.shape[0]
    ref_list = []
    for lvl in range(L):
        H = int(spatial_shapes[lvl, 0].item())
        W = int(spatial_shapes[lvl, 1].item())
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_x.reshape(-1) / W, ref_y.reshape(-1) / H), dim=-1)  # [HW,2]
        ref_list.append(ref[None].repeat(B, 1, 1))
    reference_points = torch.cat(ref_list, dim=1)  # [B,S,2]
    reference_points = reference_points[:, :, None, :] * valid_ratios[:, None, :, :]  # [B,S,L,2]
    return reference_points.contiguous()


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, n_levels, n_heads, n_points, activation="relu"):
        super().__init__()
        self.self_attn = _MSDeformAttnWrapper(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.act = _get_activation_fn(activation)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask):
        q = k = src + pos
        src2 = self.self_attn(q, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        ff = self.linear2(self.dropout2(self.act(self.linear1(src))))
        src = self.norm2(src + self.dropout3(ff))
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask):
        out = src
        for layer in self.layers:
            out = layer(out, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return out


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ffn,
        dropout,
        n_levels,
        n_points,
        activation="relu",
        use_text_cross_attn: bool = True,
    ):
        super().__init__()
        self.use_text_cross = use_text_cross_attn

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn_img = _MSDeformAttnWrapper(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        if self.use_text_cross:
            self.cross_attn_txt = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
            self.dropout_txt = nn.Dropout(dropout)
            self.norm_txt = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

        self.act = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,  # [Q,B,C]
        query_pos: Tensor,  # [Q,B,C]
        reference_points: Tensor,  # [B,Q,L,2]
        memory: Tensor,  # [S_total,B,C]
        spatial_shapes: Tensor,  # [L,2]
        level_start_index: Tensor,  # [L]
        memory_key_padding_mask: Optional[Tensor],  # [B,S_total]
        text_memory: Optional[Tensor],  # [T,B,C]
        text_key_padding_mask: Optional[Tensor],  # [B,T]
        pos: Optional[Tensor],
    ):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=None, key_padding_mask=None)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.cross_attn_img(
            tgt + query_pos, reference_points, memory, spatial_shapes, level_start_index, memory_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        if self.use_text_cross and (text_memory is not None):
            tgt2 = self.cross_attn_txt(
                query=tgt + query_pos,
                key=text_memory,
                value=text_memory,
                attn_mask=None,
                key_padding_mask=text_key_padding_mask,
            )[0]
            tgt = self.norm_txt(tgt + self.dropout_txt(tgt2))

        ff = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = self.norm4(tgt + self.dropout4(ff))
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int, d_model: int, n_levels: int):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

        self.refine_head = _get_clones(nn.Linear(d_model, 2), num_layers)
        self._init_refine_head()
        self.ref_point_head = nn.Linear(d_model, 2)
        self.n_levels = n_levels
        self.return_intermediate = True

    def _init_refine_head(self):
        for lin in self.refine_head:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(
        self,
        tgt: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        text_memory: Optional[Tensor],
        text_key_padding_mask: Optional[Tensor],
        pos: Optional[Tensor],
    ) -> Tensor:
        output = tgt
        intermediate = []

        ref_points_norm = self.ref_point_head(query_pos).sigmoid()  # [Q,B,2]
        ref_points_norm = ref_points_norm.permute(1, 0, 2).contiguous()  # [B,Q,2]
        ref_points_norm = ref_points_norm.clamp(1e-5, 1 - 1e-5)

        ref_points = (ref_points_norm[:, :, None, :] * valid_ratios[:, None, :, :]).contiguous()

        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos,
                ref_points,
                memory,
                spatial_shapes,
                level_start_index,
                memory_key_padding_mask,
                text_memory,
                text_key_padding_mask,
                pos,
            )
            intermediate.append(self.norm(output))

            delta_xy = self.refine_head[lid](output)  # [Q,B,2]
            delta_xy = delta_xy.permute(1, 0, 2).contiguous()  # [B,Q,2]

            ref_points_norm = (inverse_sigmoid(ref_points_norm) + delta_xy).sigmoid().clamp(1e-5, 1 - 1e-5)
            ref_points = (ref_points_norm[:, :, None, :] * valid_ratios[:, None, :, :]).contiguous()

        return torch.stack(intermediate)  # [L,Q,B,C]


class Transformer(nn.Module):
    """Deformable Transformer used by your MDETR."""

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        contrastive_loss=False,
        num_feature_levels: int = 3,
        deform_num_points: int = 4,
        use_text_cross_attn: bool = True,
    ):
        super().__init__()

        if text_encoder_type.lower().endswith("sentence-transformers/all-minilm-l6-v2"):
            text_encoder_type = "sentence-transformers/all-MiniLM-L6-v2"

        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, use_fast=True)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.text_hidden_dim = int(self.text_encoder.config.hidden_size)
        self.expander_dropout = dropout
        self.resizer = FeatureResizer(
            input_feat_size=self.text_hidden_dim,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.multi_scale = True
        self.num_feature_levels = num_feature_levels
        self.nhead = nhead
        self.d_model = d_model
        self.deform_num_points = deform_num_points

        self.level_embed = nn.Parameter(torch.zeros(self.num_feature_levels, d_model))
        nn.init.normal_(self.level_embed, std=0.01)

        enc_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=deform_num_points,
            activation=activation,
        )
        self.encoder = DeformableTransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            n_heads=nhead,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_points=deform_num_points,
            activation=activation,
            use_text_cross_attn=use_text_cross_attn,
        )
        self.decoder = DeformableTransformerDecoder(dec_layer, num_decoder_layers, d_model, num_feature_levels)

        self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None
        self.pass_pos_and_query = pass_pos_and_query  # kept for interface parity

    def _flatten_multi_level(self, srcs: List[Tensor], poss: List[Tensor], masks: List[Tensor]):
        if not (len(srcs) == len(poss) == len(masks)):
            raise RuntimeError(
                f"_flatten_multi_level: len mismatch srcs={len(srcs)} poss={len(poss)} masks={len(masks)}"
            )
        B0 = srcs[0].shape[0]
        for t in srcs:
            assert t.shape[0] == B0, "Batch size mismatch across levels"

        spatial_shapes_list = []
        src_flatten, pos_flatten, mask_flatten = [], [], []
        valid_ratios = []

        for lvl, (src, pos, msk) in enumerate(zip(srcs, poss, masks)):
            B, C, H, W = src.shape
            assert pos.shape[-2:] == (H, W) and msk.shape[-2:] == (H, W), "pos/mask spatial mismatch with src"

            spatial_shapes_list.append(torch.tensor([H, W], device=src.device, dtype=torch.long))

            s_l = src.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
            p_l = pos.flatten(2).transpose(1, 2).contiguous()
            p_l = p_l + self.level_embed[lvl].view(1, 1, -1)
            m_l = msk.flatten(1).contiguous()  # [B, HW]

            src_flatten.append(s_l)
            pos_flatten.append(p_l)
            mask_flatten.append(m_l)
            valid_ratios.append(_get_valid_ratio(msk))

        spatial_shapes = torch.stack(spatial_shapes_list, dim=0)  # [L, 2]
        level_start_index = torch.cat(
            [
                torch.zeros((1,), device=spatial_shapes.device, dtype=torch.long),
                (spatial_shapes.prod(1).cumsum(0)[:-1]),
            ],
            dim=0,
        )  # [L]
        valid_ratios = torch.stack(valid_ratios, dim=1)  # [B, L, 2]

        src_flatten = torch.cat(src_flatten, dim=1).transpose(0, 1).contiguous()  # [S_total, B, C]
        pos_flatten = torch.cat(pos_flatten, dim=1).transpose(0, 1).contiguous()  # [S_total, B, C]
        mask_flatten = torch.cat(mask_flatten, dim=1).contiguous()  # [B, S_total]

        S_total = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        assert src_flatten.shape[0] == S_total, (
            f"S_total mismatch: spatial sum={S_total}, but src_flatten={src_flatten.shape[0]}"
        )
        assert mask_flatten.shape[1] == S_total, (
            f"Mask length mismatch: {mask_flatten.shape[1]} vs {S_total}"
        )
        return src_flatten, pos_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios

    def forward(
        self,
        src=None,  # list[Tensor] when encode_and_save=True
        mask=None,  # list[Tensor] of masks
        query_embed=None,
        pos_embed=None,  # list[Tensor]
        text=None,
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        valid_ratios: Optional[Tensor] = None,
    ):
        if encode_and_save:
            assert isinstance(src, (list, tuple)), (
                "Deformable transformer expects a *list* of multi-scale feature maps for 'src'"
            )
            device = src[0].device

            # Encode text
            if isinstance(text[0], str):
                tokenized = self.tokenizer.batch_encode_plus(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                encoded_text = self.text_encoder(**tokenized)
                last_hidden = encoded_text.last_hidden_state  # [B,T,H]

                text_mem = last_hidden.transpose(0, 1)  # [T,B,H]
                text_attn_mask = tokenized.attention_mask.ne(1).bool()  # [B,T] True=pad
                text_mem_resized = self.resizer(text_mem)  # [T,B,C]
                text_pooled = _mean_pool(last_hidden, tokenized.attention_mask)  # [B,H]
            else:
                text_attn_mask, text_mem_resized, tokenized = text
                text_mem = text_mem_resized
                text_pooled = text_mem_resized.transpose(0, 1).mean(1)  # [B,C]

            if not (len(src) == len(pos_embed) == len(mask)):
                raise RuntimeError(
                    f"Level count mismatch: src={len(src)}, pos={len(pos_embed)}, mask={len(mask)}."
                )

            src_flat, pos_flat, mask_flat, spatial_shapes, level_start_index, valid_ratios = self._flatten_multi_level(
                src, pos_embed, mask
            )

            ref_pts = _get_reference_points(spatial_shapes, valid_ratios, src_flat.device)
            memory = self.encoder(src_flat, pos_flat, ref_pts, spatial_shapes, level_start_index, mask_flat)

            if self.CLS is not None:
                img_pooled_op = (memory * (~mask_flat).float().transpose(0, 1)[:, :, None]).sum(0)
                denom = (~mask_flat).float().sum(1).clamp(min=1.0)[:, None]
                img_pooled_op = img_pooled_op / denom
            else:
                img_pooled_op = None

            return {
                "text_memory_resized": text_mem_resized,
                "text_memory": text_mem_resized,
                "img_memory": memory,
                "mask": mask_flat,
                "pos_embed": pos_flat,
                "query_embed": query_embed.unsqueeze(1).repeat(1, memory.shape[1], 1),
                "text_attention_mask": text_attn_mask,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
                "tokenized": tokenized,
                "text_pooled_op": (text_pooled if self.CLS is not None else None),
                "img_pooled_op": (img_pooled_op if self.CLS is not None else None),
            }

        # decode
        assert img_memory is not None and text_memory is not None
        assert (spatial_shapes is not None) and (level_start_index is not None) and (valid_ratios is not None)
        if query_embed is None:
            raise ValueError("query_embed is required for decoding")

        tgt = torch.zeros_like(query_embed)  # [Q,B,C]
        hs = self.decoder(
            tgt,
            query_embed,
            img_memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            memory_key_padding_mask=mask,
            text_memory=text_memory,
            text_key_padding_mask=text_attention_mask,
            pos=None,
        )
        return hs.transpose(1, 2)  # [L,Q,B,C] -> [L,B,Q,C]


def build_transformer(args):
    ttype = getattr(args, "transformer_type", "deformable").lower()
    if ttype == "vanilla":
        return VanillaTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            pass_pos_and_query=True,
            text_encoder_type=args.text_encoder_type,
            freeze_text_encoder=args.freeze_text_encoder,
            contrastive_loss=args.contrastive_loss,
            use_text_cross_attn=getattr(args, "use_text_cross_attn", True),
        )

    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        contrastive_loss=args.contrastive_loss,
        num_feature_levels=getattr(args, "num_feature_levels", 3),
        deform_num_points=getattr(args, "deform_num_points", 4),
        use_text_cross_attn=getattr(args, "use_text_cross_attn", True),
    )
