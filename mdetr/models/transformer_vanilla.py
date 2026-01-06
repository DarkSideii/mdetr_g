# transformer_vanilla.py
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer


def _get_activation_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise RuntimeError(f"Unsupported activation {name}")


def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _mean_pool(last_hidden: Tensor, attn_mask: Tensor) -> Tensor:
    """
    last_hidden: [B,T,H], attn_mask: [B,T] with 1 for valid, 0 for pad
    """
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    summed = (last_hidden * mask).sum(1)
    denom = mask.sum(1).clamp_min(1e-6)
    return summed / denom


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size: int, output_feat_size: int, dropout: float, do_ln: bool = True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.ln = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.do_ln:
            x = self.ln(x)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, normalize_before: bool):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, x: Tensor, pos: Optional[Tensor]):
        return x if pos is None else x + pos

    def forward_post(self, src, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

    def forward_pre(self, src, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return self.forward_post(src, src_key_padding_mask=src_key_padding_mask, pos=pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        normalize_before: bool,
        use_text_cross_attn: bool = True,
    ):
        super().__init__()
        self.normalize_before = normalize_before  # kept for interface parity
        self.use_text_cross = use_text_cross_attn

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn_img = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        if self.use_text_cross:
            self.cross_attn_txt = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_txt = nn.LayerNorm(d_model) if self.use_text_cross else None
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_txt = nn.Dropout(dropout) if self.use_text_cross else None
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, x: Tensor, pos: Optional[Tensor]):
        return x if pos is None else x + pos

    def forward(
        self,
        tgt: Tensor,                    # [Q,B,C]
        memory: Tensor,                 # [S,B,C] image memory
        text_memory: Optional[Tensor],  # [T,B,C]
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,   # [B,S]
        text_memory_key_padding_mask=None,  # [B,T]
        pos: Optional[Tensor] = None,   # [S,B,C]
        query_pos: Optional[Tensor] = None,  # [Q,B,C]
    ):
        # Self-attn on queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Cross-attn to image memory
        tgt2 = self.cross_attn_img(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Optional cross-attn to text memory
        if self.use_text_cross and (text_memory is not None):
            tgt2 = self.cross_attn_txt(
                query=self.with_pos_embed(tgt, query_pos),
                key=text_memory,
                value=text_memory,
                key_padding_mask=text_memory_key_padding_mask,
            )[0]
            tgt = self.norm_txt(tgt + self.dropout_txt(tgt2))

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module, return_intermediate: bool = True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        memory_key_padding_mask=None,
        text_memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        out = tgt
        intermediate = []

        for layer in self.layers:
            out = layer(
                out,
                memory,
                text_memory,
                memory_key_padding_mask=memory_key_padding_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(out))

        out = self.norm(out)
        if self.return_intermediate:
            intermediate[-1] = out
            return torch.stack(intermediate)
        return out


class VanillaTransformer(nn.Module):
    """
    Vanilla (no deformable attention) transformer with the *same forward API*
    as the deformable transformer:
      - encode_and_save=True returns a memory_cache dict
      - encode_and_save=False consumes it and returns hs of shape [L,B,Q,C]

    Designed to work when num_feature_levels=1.
    """

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
        use_text_cross_attn: bool = True,
    ):
        super().__init__()
        self.multi_scale = False
        self.pass_pos_and_query = pass_pos_and_query
        self.d_model = d_model
        self.nhead = nhead

        enc_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        enc_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, norm=enc_norm)

        dec_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            use_text_cross_attn=use_text_cross_attn,
        )
        self.decoder = TransformerDecoder(
            dec_layer,
            num_decoder_layers,
            norm=nn.LayerNorm(d_model),
            return_intermediate=return_intermediate_dec,
        )

        # Contrastive: add an image CLS token (like original MDETR)
        self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        # Text encoder (HF)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, use_fast=True)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        text_hidden = int(self.text_encoder.config.hidden_size)
        self.resizer = FeatureResizer(text_hidden, d_model, dropout=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,                      # list[Tensor] or Tensor
        mask=None,                     # list[Tensor] or Tensor
        query_embed=None,              # [Q,C]  (not repeated)
        pos_embed=None,                # list[Tensor] or Tensor
        text=None,                     # list[str] or already encoded
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        spatial_shapes: Optional[Tensor] = None,     # ignored (kept for API compatibility)
        level_start_index: Optional[Tensor] = None,  # ignored
        valid_ratios: Optional[Tensor] = None,       # ignored
    ):
        if encode_and_save:
            # Accept lists but only use last (expects num_feature_levels=1 anyway)
            if isinstance(src, (list, tuple)):
                src = src[-1]
            if isinstance(mask, (list, tuple)):
                mask = mask[-1]
            if isinstance(pos_embed, (list, tuple)):
                pos_embed = pos_embed[-1]

            assert src is not None and mask is not None and pos_embed is not None
            assert query_embed is not None

            bs, c, h, w = src.shape
            device = src.device

            # Flatten image: [B,C,H,W] -> [S,B,C]
            src_seq = src.flatten(2).permute(2, 0, 1).contiguous()
            pos_seq = pos_embed.flatten(2).permute(2, 0, 1).contiguous()
            mask_seq = mask.flatten(1).contiguous()  # [B,S]

            # Add CLS for contrastive if enabled
            if self.CLS is not None:
                cls_tok = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)  # [1,B,C]
                src_seq = torch.cat([cls_tok, src_seq], dim=0)
                pos_seq = torch.cat([torch.zeros(1, bs, self.d_model, device=device), pos_seq], dim=0)
                cls_pad = torch.zeros(bs, 1, dtype=torch.bool, device=device)
                mask_seq = torch.cat([cls_pad, mask_seq], dim=1)

            # Prepare query embeddings: [Q,C] -> [Q,B,C]
            query_embed_rep = query_embed.unsqueeze(1).repeat(1, bs, 1)

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed_rep)
            else:
                # legacy option
                src_seq = src_seq + 0.1 * pos_seq
                tgt = query_embed_rep
                query_embed_rep = None
                pos_seq = None

            # Encode text if needed
            if isinstance(text[0], str):
                tokenized = self.tokenizer.batch_encode_plus(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                encoded = self.text_encoder(**tokenized)
                last_hidden = encoded.last_hidden_state  # [B,T,H]

                text_mem = last_hidden.transpose(0, 1).contiguous()  # [T,B,H]
                text_kpm = tokenized.attention_mask.ne(1).bool().contiguous()  # [B,T] True=pad
                text_mem_resized = self.resizer(text_mem)  # [T,B,C]

                text_pooled = _mean_pool(last_hidden, tokenized.attention_mask)  # [B,H]
            else:
                # already encoded: (mask, resized_mem, tokenized)
                text_kpm, text_mem_resized, tokenized = text
                text_kpm = text_kpm.to(device)
                text_mem_resized = text_mem_resized.to(device)
                text_pooled = text_mem_resized.transpose(0, 1).mean(1)

            # Image encoder
            memory = self.encoder(src_seq, src_key_padding_mask=mask_seq, pos=pos_seq)

            # CLS pooled image representation (if enabled)
            img_pooled = memory[0] if self.CLS is not None else None  # [B,C]

            return {
                "text_memory_resized": text_mem_resized,  # [T,B,C]
                "text_memory": text_mem_resized,          # [T,B,C]
                "img_memory": memory,                     # [S,B,C]
                "mask": mask_seq,                         # [B,S]
                "pos_embed": pos_seq,                     # [S,B,C]
                "query_embed": query_embed_rep,           # [Q,B,C]
                "text_attention_mask": text_kpm,          # [B,T]
                # compatibility keys with deformable path
                "spatial_shapes": None,
                "level_start_index": None,
                "valid_ratios": None,
                "tokenized": tokenized,
                "text_pooled_op": (text_pooled if self.CLS is not None else None),  # [B,H_text]
                "img_pooled_op": (img_pooled if self.CLS is not None else None),    # [B,C]
            }

        # ---------------- decode ----------------
        assert img_memory is not None and text_memory is not None and query_embed is not None

        tgt = torch.zeros_like(query_embed) if self.pass_pos_and_query else query_embed

        hs = self.decoder(
            tgt=tgt,
            memory=img_memory,
            text_memory=text_memory,
            memory_key_padding_mask=mask,
            text_memory_key_padding_mask=text_attention_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2).contiguous()  # [L,Q,B,C] -> [L,B,Q,C]
