from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from timm.models import create_model
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from mdetr.util.misc import NestedTensor
from .position_encoding import build_position_encoding

import satlaspretrain_models as sat


class SatlasAerialSwin(nn.Module):
    """Satlas Swin backbone adapter that returns the last K feature maps as NestedTensors."""
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

        self.weights = sat.Weights()
        self.backbone = self.weights.get_pretrained_model(model_id, fpn=use_fpn)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if use_fpn:
            full_channels = [128, 128, 128, 128]  # P3..P6
        else:
            full_channels = [128, 256, 512, 1024]  # C2..C5 (Swin-B)

        self.channels = full_channels[-self.num_levels:]
        self.num_channels = self.channels[-1]

    def forward(self, tensor_list: NestedTensor) -> OrderedDict:
        x = tensor_list.tensors
        mask = tensor_list.mask

        feats_all = self.backbone(x)
        feats = list(feats_all)[-self.num_levels:]

        out = OrderedDict()
        for i, f in enumerate(feats):
            m = F.interpolate(mask[None].float(), size=f.shape[-2:], mode="nearest").to(torch.bool)[0]
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
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()

        for name, parameter in backbone.named_parameters():
            if not train_backbone or ("layer2" not in name and "layer3" not in name and "layer4" not in name):
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> OrderedDict:
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:], mode="nearest").bool()[0]
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
    """ResNet backbone with GroupNorm weights loaded from a checkpoint."""
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        name_map = {
            "resnet50-gn": ("resnet50", "/checkpoint/szagoruyko/imagenet/22014122/checkpoint.pth"),
            "resnet101-gn": ("resnet101", "/checkpoint/szagoruyko/imagenet/22080524/checkpoint.pth"),
        }
        backbone = getattr(torchvision.models, name_map[name][0])(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
            norm_layer=GroupNorm32,
        )
        checkpoint = torch.load(name_map[name][1], map_location="cpu")
        state_dict = {k[7:]: p for k, p in checkpoint["model"].items()}
        backbone.load_state_dict(state_dict)

        num_channels = 512 if name_map[name][0] in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def unfreeze_satlas_last2_and_norm(satlas_core: nn.Module) -> None:
    """
    Freeze all Satlas Swin params, then unfreeze the last two stages and final norm.

    Unfreezes parameter names starting with:
      - "features.6."
      - "features.7."
      - "norm."
    """
    for p in satlas_core.parameters():
        p.requires_grad_(False)

    allow_prefixes = ("features.6.", "features.7.", "norm.")
    for n, p in satlas_core.named_parameters():
        if n.startswith(allow_prefixes):
            p.requires_grad_(True)


class TimmBackbone(nn.Module):
    """timm backbone in features_only mode, returning the last K stages as NestedTensors."""
    def __init__(self, name: str, num_feature_levels: int, return_interm_layers: bool):
        super().__init__()

        tmp = create_model(name, pretrained=True, features_only=True)
        all_idx = list(range(len(tmp.feature_info.channels())))

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
        out = OrderedDict()
        for i, x in enumerate(feats):
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:], mode="nearest").bool()[0]
            out[f"layer{i}"] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    """Backbone wrapper that also returns positional encodings."""
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out, pos = [], []
        for _, x in xs.items():
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
    return_interm_layers = (k > 1)

    backbone = None

    if args.backbone.lower().startswith("convnext") or args.backbone == "ConvNeXt":
        timm_name = _normalize_backbone_name(args.backbone)
        backbone = TimmBackbone(timm_name, num_feature_levels=k, return_interm_layers=return_interm_layers)

    elif args.backbone == "satlas_aerial_swinb":
        backbone = SatlasAerialSwin(
            model_id="Aerial_SwinB_SI",
            use_fpn=False,
            num_feature_levels=k,
            train_backbone=(getattr(args, "lr_backbone", 0.0) > 0.0),
        )

        if getattr(args, "lr_backbone", 0.0) > 0.0:
            unfreeze_satlas_last2_and_norm(backbone.backbone)
        else:
            for p in backbone.parameters():
                p.requires_grad_(False)
            print("Frozen backbone")
            backbone.eval()

    elif args.backbone in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    if getattr(args, "lr_backbone", 0.0) <= 0.0:
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    if hasattr(backbone, "channels"):
        model.channels = backbone.channels
    return model
