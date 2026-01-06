"""
Dataset/evaluator factories for geo_datasets.
"""

def build_dataset(dataset_file: str, image_set: str, args):
    """Return the dataset for `dataset_file` and split name `image_set`."""
    if dataset_file == "dota":
        from .dota_preprocess import build_dota
        return build_dota(image_set, args)

    raise ValueError(f"Unknown dataset_file: {dataset_file}")


def get_evaluator(dataset_file: str):
    """Return the evaluator class for `dataset_file`."""
    if dataset_file == "dota":
        from .dota_eval import DotaEvaluator
        return DotaEvaluator

    raise ValueError(f"Unknown dataset_file: {dataset_file}")
