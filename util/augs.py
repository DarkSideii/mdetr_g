from __future__ import annotations

import io
import random
from typing import Any, Dict
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode
_EPS = 1e-6


# ---------------------- box utilities (xyxy, pixel) ----------------------
def _flip_boxes_h(boxes: torch.Tensor, w: int) -> torch.Tensor:
    """Horizontal flip boxes within width w."""
    if boxes.numel() == 0:
        return boxes
    x0, y0, x1, y1 = boxes.unbind(-1)
    fx0, fx1 = w - x1, w - x0
    fx0 = fx0.clamp(0, w - _EPS)
    fx1 = fx1.clamp(0, w - _EPS)
    return torch.stack([fx0, y0, fx1, y1], dim=-1)


def _flip_boxes_v(boxes: torch.Tensor, h: int) -> torch.Tensor:
    """Vertical flip boxes within height h."""
    if boxes.numel() == 0:
        return boxes
    x0, y0, x1, y1 = boxes.unbind(-1)
    fy0, fy1 = h - y1, h - y0
    fy0 = fy0.clamp(0, h - _EPS)
    fy1 = fy1.clamp(0, h - _EPS)
    return torch.stack([x0, fy0, x1, fy1], dim=-1)


def _rot90_boxes(boxes: torch.Tensor, w: int, h: int, k: int) -> torch.Tensor:
    """
    Rotate boxes by k * 90° CCW, return axis-aligned AABBs in the rotated image.
    Coordinates are pixel xyxy.
    """
    k = k % 4
    if k == 0 or boxes.numel() == 0:
        return boxes

    x0, y0, x1, y1 = boxes.unbind(-1)
    # 4 corners per box: [N,4,2]
    pts = torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x1, y1], dim=-1),
            torch.stack([x0, y1], dim=-1),
        ],
        dim=1,
    )

    if k == 1:
        # 90° CCW: (x, y) -> (y, w-1-x)
        xn, yn = pts[..., 1], (w - 1) - pts[..., 0]
        new_w, new_h = h, w
    elif k == 2:
        # 180°: (x, y) -> (w-1-x, h-1-y)
        xn, yn = (w - 1) - pts[..., 0], (h - 1) - pts[..., 1]
        new_w, new_h = w, h
    else:
        # 270° CCW: (x, y) -> (h-1-y, x)
        xn, yn = (h - 1) - pts[..., 1], pts[..., 0]
        new_w, new_h = h, w

    rpts = torch.stack([xn, yn], dim=-1)
    xmin = rpts[..., 0].min(dim=1).values.clamp(0, new_w - _EPS)
    xmax = rpts[..., 0].max(dim=1).values.clamp(0, new_w - _EPS)
    ymin = rpts[..., 1].min(dim=1).values.clamp(0, new_h - _EPS)
    ymax = rpts[..., 1].max(dim=1).values.clamp(0, new_h - _EPS)
    xmax = torch.maximum(xmax, xmin + _EPS)
    ymax = torch.maximum(ymax, ymin + _EPS)
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def _update_size(target: Dict[str, Any], new_h: int, new_w: int) -> None:
    """Update target['size'] while preserving dtype/device if it exists."""
    if "size" in target:
        old = target["size"]
        target["size"] = torch.as_tensor([new_h, new_w], dtype=old.dtype, device=old.device)


# ---------------------- optional mask helpers ----------------------
def _hflip_masks(m):
    if isinstance(m, torch.Tensor):  # [N,H,W]
        return torch.flip(m, dims=(-1,))
    return [F.hflip(mi) for mi in m]  # list of PIL masks


def _vflip_masks(m):
    if isinstance(m, torch.Tensor):  # [N,H,W]
        return torch.flip(m, dims=(-2,))
    return [F.vflip(mi) for mi in m]


def _rot90_masks(m, k: int):
    if isinstance(m, torch.Tensor):  # [N,H,W]
        return torch.rot90(m, k, dims=(-2, -1))
    # list of PIL masks
    if k == 1:
        op = Image.ROTATE_90
    elif k == 2:
        op = Image.ROTATE_180
    else:
        op = Image.ROTATE_270
    return [mi.transpose(op) for mi in m]


# ------------------------------ transforms ------------------------------
class RandomVerticalFlip:
    """Flip image (and boxes/masks) vertically with probability p."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() < self.prob:
            w, h = img.size
            img = F.vflip(img)
            if "boxes" in target:
                target["boxes"] = _flip_boxes_v(target["boxes"], h)
            if "masks" in target:  # optional
                target["masks"] = _vflip_masks(target["masks"])
        return img, target


class RandomRotate90:
    """Rotate by a random choice of {90, 180, 270} deg CCW with probability p."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() >= self.prob:
            return img, target

        k = random.choice([1, 2, 3])  # steps of 90° CCW
        w, h = img.size

        if k == 1:
            img = img.transpose(Image.ROTATE_90)
        elif k == 2:
            img = img.transpose(Image.ROTATE_180)
        else:
            img = img.transpose(Image.ROTATE_270)

        if "boxes" in target and target["boxes"].numel():
            target["boxes"] = _rot90_boxes(target["boxes"], w, h, k)
        if "masks" in target:  # optional
            target["masks"] = _rot90_masks(target["masks"], k)

        # Update *current* size; leave orig_size alone.
        new_w, new_h = (h, w) if k in (1, 3) else (w, h)
        _update_size(target, new_h, new_w)
        return img, target


class RandomJPEG:
    """Re-encode JPEG to add compression artifacts."""

    def __init__(self, prob: float = 0.3, qmin: int = 60, qmax: int = 95):
        self.prob, self.qmin, self.qmax = prob, qmin, qmax

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() >= self.prob:
            return img, target
        buf = io.BytesIO()
        q = random.randint(self.qmin, self.qmax)
        img.save(buf, format="JPEG", quality=q, optimize=False)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img, target


class RandomGaussianBlur:
    """Apply Gaussian blur with random radius in [rmin, rmax] with probability p."""

    def __init__(self, prob: float = 0.2, radius=(0.1, 1.2)):
        self.prob, self.rmin, self.rmax = prob, float(radius[0]), float(radius[1])

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() < self.prob:
            r = random.uniform(self.rmin, self.rmax)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
        return img, target


class RandomColorJitter:
    """Lightweight color jitter that keeps geometry intact."""

    def __init__(
        self,
        prob: float = 0.4,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.02,  # must be in [-0.5, 0.5]
    ):
        self.prob = prob
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() >= self.prob:
            return img, target
        img = F.adjust_brightness(img, 1.0 + random.uniform(-self.brightness, self.brightness))
        img = F.adjust_contrast(img, 1.0 + random.uniform(-self.contrast, self.contrast))
        img = F.adjust_saturation(img, 1.0 + random.uniform(-self.saturation, self.saturation))
        img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
        return img, target


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper function to create the inverse matrix for box transformations
    from torchvision.transforms.functional import _get_inverse_affine_matrix as get_matrix
    return get_matrix(center, angle, translate, scale, shear)


class RandomAffine:
    """Applies random affine transformations (shear/rotation) to image and boxes."""

    def __init__(self, prob: float = 0.5, degrees: float = 0, shear: float = 10):
        self.prob = prob
        self.degrees = degrees
        self.shear = shear

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() >= self.prob:
            return img, target

        w, h = img.size
        center = (w * 0.5, h * 0.5)  # ← define once, reuse for image + boxes

        # random params
        angle   = random.uniform(-self.degrees, self.degrees)
        shear_x = random.uniform(-self.shear,   self.shear)
        shear_y = random.uniform(-self.shear,   self.shear)

        # 1) transform image
        img = F.affine(
            img,
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=[shear_x, shear_y],
            interpolation=InterpolationMode.BICUBIC,
            center=center,
        )

        # 2) transform boxes
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"]
            dtype, device = boxes.dtype, boxes.device

            # inverse matrix from torchvision (maps output->input)
            inv_flat = _get_inverse_affine_matrix(
                center=center,
                angle=angle,
                translate=[0.0, 0.0],
                scale=1.0,
                shear=[shear_x, shear_y],
            )
            inv_2x3 = torch.tensor(inv_flat, dtype=dtype, device=device).view(2, 3)
            inv_3x3 = torch.vstack([inv_2x3, torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)])

            # forward matrix (input->output)
            fwd_3x3 = torch.linalg.inv(inv_3x3)
            M = fwd_3x3[:2, :]  # 2x3

            x0, y0, x1, y1 = boxes.unbind(-1)
            corners = torch.stack(
                [torch.stack([x0, y0], -1),
                 torch.stack([x1, y0], -1),
                 torch.stack([x1, y1], -1),
                 torch.stack([x0, y1], -1)], dim=1)  # (N,4,2)
            ones = torch.ones((*corners.shape[:-1], 1), dtype=dtype, device=device)
            corners_h = torch.cat([corners, ones], dim=-1)            # (N,4,3)
            new = torch.einsum('ij,nkj->nki', M, corners_h)          # (N,4,2)

            # axis-aligned AABB, clamp to image
            xmin = new[..., 0].min(dim=1).values.clamp(0, w - 1e-6)
            xmax = new[..., 0].max(dim=1).values.clamp(0, w - 1e-6)
            ymin = new[..., 1].min(dim=1).values.clamp(0, h - 1e-6)
            ymax = new[..., 1].max(dim=1).values.clamp(0, h - 1e-6)
            target["boxes"] = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

        return img, target


class RandomGaussianNoise:
    """Adds Gaussian noise to a PIL image."""

    def __init__(self, prob: float = 0.5, std: float = 15.0):
        self.prob = prob
        self.std = std

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() >= self.prob:
            return img, target

        # Convert PIL image to numpy array to perform numerical operations
        img_arr = np.array(img)

        # Generate Gaussian noise with a mean of 0 and the specified standard deviation
        noise = np.random.normal(0, self.std, img_arr.shape)

        # Add noise to the image and clip the values to the valid [0, 255] range
        noisy_img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

        # Convert the noisy numpy array back to a PIL image
        noisy_img = Image.fromarray(noisy_img_arr)

        return noisy_img, target


__all__ = [
    "RandomVerticalFlip",
    "RandomRotate90",
    "RandomGaussianBlur",
    "RandomColorJitter",
    "RandomJPEG",
    "RandomGaussianNoise",
    "RandomAffine",
]
