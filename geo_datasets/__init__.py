# geo_datasets/__init__.py

def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "dior_rsvg":
        from .dior_rsvg_preprocess import build_dior_rsvg
        return build_dior_rsvg(image_set, args)
    raise ValueError(f"Unknown dataset_file: {dataset_file}")


def get_evaluator(dataset_file: str):
    if dataset_file == "dior_rsvg":
        from .dior_rsvg_eval import DiorRSVGEvaluator
        return DiorRSVGEvaluator
    raise ValueError(f"Unknown dataset_file: {dataset_file}")
