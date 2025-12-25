def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "rareplanes":
        from .rp_preprocess import build_rareplanes
        return build_rareplanes(image_set, args)
    elif dataset_file == "dota":
        from .dota_preprocess import build_dota
        return build_dota(image_set, args)
    elif dataset_file in "hrsc":
        from .hrsc_preprocess import build_hrsc2016ms
        return build_hrsc2016ms(image_set, args)
    elif dataset_file == "fair1m":
        from .fair1m_preprocess import build_fair1m
        return build_fair1m(image_set, args)
    elif dataset_file == "dior":
        from .dior_preprocess import build_dior
        return build_dior(image_set, args)
    else:
        raise ValueError(f"Unknown dataset_file: {dataset_file}")


def get_evaluator(dataset_file: str):
    if dataset_file == "rareplanes":
        from .rp_eval import RarePlanesEvaluator
        return RarePlanesEvaluator
    elif dataset_file == "dota":
        from .dota_eval import DotaEvaluator
        return DotaEvaluator
    elif dataset_file in "hrsc":
        from .hrsc_eval import HRSC2016MSEvaluator
        return HRSC2016MSEvaluator
    elif dataset_file in "fair1m":
        from .fair1m_eval import FAIR1MEvaluator
        return FAIR1MEvaluator
    elif dataset_file == "dior":
        from .dior_eval import DiorEvaluator
        return DiorEvaluator
    else:
        raise ValueError(f"Unknown dataset_file: {dataset_file}")
