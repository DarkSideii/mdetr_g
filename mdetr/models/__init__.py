from .mdetr import build as _build_mdetr  # original builder
def build_model(args):
    return _build_mdetr(args)