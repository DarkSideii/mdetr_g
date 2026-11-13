# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.

"""Vision backbones and positional-encoding adapters."""

import os
from collections import OrderedDict

import satlaspretrain_models as sat
import torch
import torch.nn.functional as F
import torchvision
from satlaspretrain_models.utils import SatlasPretrain_weights
from timm.models import create_model
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from mdetr.util.misc import NestedTensor

from .position_encoding import build_position_encoding


class SatlasAerialSwin(nn.Module):
    """Return the last K Satlas Swin feature maps as NestedTensors."""

    def __init__(
        self,
        model_id: str = "Aerial_SwinB_SI",
        use_fpn: bool = False,
        num_feature_levels: int = 4,
        train_backbone: bool = False,
    ):
        super().__init__()
        assert 1 <= num_feature_levels <= 4, "num_feature_levels must be in [1,4]"

        self.use_fpn = use_fpn
        self.num_levels = num_feature_levels

        weights_path = os.environ.get(
            "SATLAS_WEIGHTS_PATH",
            "checkpoints/aerial_swinb_si.pth",
        )

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Satlas pretrained weights not found: {weights_path}")

        print(f"[Satlas] Loading local pretrained weights: {weights_path}")

        model_info = SatlasPretrain_weights[model_id]

        weights = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False,
        )

        self.backbone = sat.Model(
            num_channels=model_info["num_channels"],
            multi_image=model_info["multi_image"],
            backbone=model_info["backbone"],
            fpn=use_fpn,
            weights=weights,
        )

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if use_fpn:
            full_channels = [128, 128, 128, 128]  # P3..P6
        else:
            full_channels = [128, 256, 512, 1024]  # C2..C5 (Swin-B)

        self.channels = full_channels[-self.num_levels :]
        self.num_channels = self.channels[-1]

    def forward(self, tensor_list: NestedTensor) -> OrderedDict:
        x = tensor_list.tensors
        mask = tensor_list.mask[None].float()

        feats_all = self.backbone(x)
        feats = list(feats_all)[-self.num_levels :]

        out = OrderedDict()
        for i, f in enumerate(feats):
            m = F.interpolate(mask, size=f.shape[-2:], mode="nearest").bool()[0]
            out[f"layer{i}"] = NestedTensor(f, m)
        return out


class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d variant with fixed affine params and running statistics."""

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class GroupNorm32(torch.nn.GroupNorm):
    """GroupNorm with 32 groups (common ResNet-GN setting)."""

    def __init__(self, num_channels: int, num_groups: int = 32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class GN_8(nn.Module):
    """8-group GroupNorm wrapper."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.gn = torch.nn.GroupNorm(8, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


def replace_bn(m: nn.Module, name: str = "") -> None:
    """Recursively replace BatchNorm2d with FrozenBatchNorm2d."""
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(target_attr.num_features)
            bn = getattr(m, attr_str)
            frozen.weight.data.copy_(bn.weight.data)
            frozen.bias.data.copy_(bn.bias.data)
            frozen.running_mean.data.copy_(bn.running_mean.data)
            frozen.running_var.data.copy_(bn.running_var.data)
            setattr(m, attr_str, frozen)
    for n, ch in m.named_children():
        replace_bn(ch, n)


class BackboneBase(nn.Module):
    """IntermediateLayerGetter wrapper that returns NestedTensor feature maps."""

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()

        for name, parameter in backbone.named_parameters():
            if not train_backbone or (
                "layer2" not in name and "layer3" not in name and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> OrderedDict:
        xs = self.body(tensor_list.tensors)
        image_mask = tensor_list.mask[None].float()
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(image_mask, size=x.shape[-2:], mode="nearest").bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """Torchvision ResNet backbone using FrozenBatchNorm2d."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class GroupNormBackbone(BackboneBase):
    """Load ResNet-GN weights from RESNET_GN_WEIGHTS_PATH."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        name_map = {
            "resnet50-gn": "resnet50",
            "resnet101-gn": "resnet101",
        }
        weights_path = os.environ.get("RESNET_GN_WEIGHTS_PATH")
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Pretrained weights for {name} were not found. "
                "Set RESNET_GN_WEIGHTS_PATH to a local checkpoint."
            )
        backbone = getattr(torchvision.models, name_map[name])(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
            norm_layer=GroupNorm32,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = {k[7:]: p for k, p in checkpoint["model"].items()}
        backbone.load_state_dict(state_dict)

        super().__init__(backbone, train_backbone, 2048, return_interm_layers)


def unfreeze_satlas_last2_and_norm(satlas_core: nn.Module) -> None:
    """Train Swin features.6, features.7 and final norm at any wrapper depth."""
    for p in satlas_core.parameters():
        p.requires_grad_(False)

    unfrozen_params = 0
    for name, p in satlas_core.named_parameters():
        parts = name.split(".")
        last_stages = any(
            parts[i] == "features" and parts[i + 1] in ("6", "7") for i in range(len(parts) - 1)
        )
        final_norm = (
            "features" not in parts
            and len(parts) >= 2
            and parts[-2] == "norm"
            and parts[-1] in ("weight", "bias")
        )
        if last_stages or final_norm:
            p.requires_grad_(True)
            unfrozen_params += p.numel()

    if unfrozen_params == 0:
        raise RuntimeError(
            "Satlas Swin final features.6/features.7/norm were not found; "
            "backbone fine-tuning would be silently disabled."
        )
    print(f"[Satlas] Trainable final Swin stages and norm: {unfrozen_params:,} parameters")


class TimmBackbone(nn.Module):
    """Return the last K timm feature stages as NestedTensors."""

    def __init__(self, name: str, num_feature_levels: int, return_interm_layers: bool):
        super().__init__()

        # Feature metadata does not require loading pretrained weights twice.
        tmp = create_model(name, pretrained=False, features_only=True)
        all_idx = list(range(len(tmp.feature_info.channels())))
        del tmp

        if return_interm_layers and num_feature_levels > 1:
            K = min(num_feature_levels, len(all_idx))
            out_indices = tuple(all_idx[-K:])
        else:
            out_indices = (all_idx[-1],)

        backbone = create_model(
            name, pretrained=True, in_chans=3, features_only=True, out_indices=out_indices
        )

        with torch.no_grad():
            replace_bn(backbone)

        self.body = backbone
        self.channels = self.body.feature_info.channels()
        self.num_channels = self.channels[-1]
        self.interm = len(self.channels) > 1

    def forward(self, tensor_list: NestedTensor) -> OrderedDict:
        feats = self.body(tensor_list.tensors)
        image_mask = tensor_list.mask[None].float()
        out = OrderedDict()
        for i, x in enumerate(feats):
            mask = F.interpolate(image_mask, size=x.shape[-2:], mode="nearest").bool()[0]
            out[f"layer{i}"] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    """Backbone wrapper that also returns positional encodings."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out, pos = [], []
        for x in xs.values():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def _normalize_backbone_name(name: str) -> str:
    """Map shorthand backbone names to timm model identifiers."""
    low = name.lower()
    if low == "convnext":
        return "convnext_small"
    if low.startswith("convnext_"):
        return low
    return name


def build_backbone(args):
    """Build the configured vision backbone + position encoding joiner."""
    position_embedding = build_position_encoding(args)
    train_backbone = getattr(args, "lr_backbone", 0.0) > 0.0

    ttype = getattr(args, "transformer_type", "deformable").lower()
    if ttype == "vanilla":
        k = 1
    else:
        k = int(getattr(args, "num_feature_levels", 1))
    return_interm_layers = k > 1

    if args.backbone.lower().startswith("convnext"):
        timm_name = _normalize_backbone_name(args.backbone)
        backbone = TimmBackbone(
            timm_name, num_feature_levels=k, return_interm_layers=return_interm_layers
        )

    elif args.backbone == "satlas_aerial_swinb":
        backbone = SatlasAerialSwin(
            model_id="Aerial_SwinB_SI",
            use_fpn=False,
            num_feature_levels=k,
            train_backbone=train_backbone,
        )

        if train_backbone:
            unfreeze_satlas_last2_and_norm(backbone.backbone)
        else:
            print("[Satlas] Backbone frozen")

    elif args.backbone in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation
        )

    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    if not train_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    if hasattr(backbone, "channels"):
        model.channels = backbone.channels
    return model
