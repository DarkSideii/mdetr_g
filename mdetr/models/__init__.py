"""Model factory wrapper."""

from .mdetr import build as _build_mdetr

def build_model(args):
    """Build the MDETR model/criteria tuple using the canonical builder."""
    return _build_mdetr(args)
