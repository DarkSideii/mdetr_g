# geo_datasets/__init__.py

def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "dota":
        from .dota_preprocess import build_dota
        if image_set == "test":
            image_set = "val"

        return build_dota(image_set, args)

    raise ValueError(f"Unknown dataset_file: {dataset_file}")


def get_evaluator(dataset_file: str):
    if dataset_file == "dota":
        from .dota_eval import DotaEvaluator 
        return DotaEvaluator

    raise ValueError(f"Unknown dataset_file: {dataset_file}")