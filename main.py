# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright (c) 2026 Nicholas Harvey.
# Modified for MDETR-G to support geospatial/remote-sensing grounding, training, and evaluation.
#
# Licensed under the Apache License, Version 2.0.
"""Train or evaluate MDETR-G with dataset configuration and MLflow logging."""

import argparse
import datetime
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time
import warnings

# Set runtime options before importing CUDA/tokenizer dependencies.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import mdetr.util.misc as utils
from geo_datasets import build_dataset, get_evaluator
from mdetr.engine import evaluate, train_one_epoch
from mdetr.models import build_model
from mdetr.models.postprocessors import build_postprocessors


def get_args_parser():
    """Build shared training and evaluation arguments."""
    p = argparse.ArgumentParser("mdetr_g detector", add_help=False)

    # Run and dataset settings.
    p.add_argument("--run_name", default="", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--no_detection", action="store_true", help="Disable the detection objective")

    p.add_argument(
        "--dataset_config",
        default=None,
        nargs="?",
        const=None,
        help="JSON with local/S3 dataset paths and argument overrides",
    )
    p.add_argument(
        "--val_split_ratio",
        default=0.2,
        type=float,
        help="Fraction of training set held out for validation",
    )

    # Optimization.
    p.add_argument("--batch_size", default=2, type=int)
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--lr_backbone", default=1e-5, type=float)
    p.add_argument("--text_encoder_lr", default=5e-5, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--optimizer", default="adam", type=str)
    p.add_argument("--clip_max_norm", default=0.1, type=float)
    p.add_argument(
        "--schedule",
        default="linear_with_warmup",
        choices=(
            "step",
            "multistep",
            "linear_with_warmup",
            "all_linear_with_warmup",
            "all_cosine_with_warmup",
        ),
    )
    p.add_argument(
        "--lr_drop", default=35, type=int, help="Epoch milestone for *step* or *multistep* sched"
    )
    p.add_argument("--fraction_warmup_steps", default=0.01, type=float)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", default=0.9998, type=float)
    p.add_argument("--eval_skip", default=1, type=int, help="Validate every N epochs")

    # Vision and text encoders.
    p.add_argument("--frozen_weights", default=None, type=str)
    p.add_argument("--freeze_text_encoder", action="store_true")
    p.add_argument(
        "--text_encoder_type",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="HF model name for text encoder (e.g. roberta-base, sentence-transformers/all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--backbone", default="satlas_aerial_swinb", type=str, help="vision backbone selection"
    )
    p.add_argument("--dilation", action="store_true")
    p.add_argument("--position_embedding", default="sine", choices=("sine", "learned"))

    # Transformer.
    p.add_argument("--enc_layers", default=6, type=int)
    p.add_argument("--dec_layers", default=6, type=int)
    p.add_argument("--dim_feedforward", default=2048, type=int)
    p.add_argument("--hidden_dim", default=256, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--nheads", default=8, type=int)
    p.add_argument("--num_queries", default=100, type=int)
    p.add_argument("--pre_norm", action="store_true")
    p.add_argument("--no_pass_pos_and_query", dest="pass_pos_and_query", action="store_false")

    p.add_argument(
        "--num_feature_levels",
        type=int,
        default=3,
        help="How many backbone stages to use (1-4). Typically 3 (last 3 stages).",
    )
    p.add_argument(
        "--deform_num_points",
        type=int,
        default=4,
        help="Sampling points per head per level (4 or 8 are common)",
    )

    # Detection and alignment losses.
    p.add_argument("--set_cost_class", default=1, type=float)
    p.add_argument("--set_cost_bbox", default=5, type=float)
    p.add_argument("--set_cost_giou", default=2, type=float)
    p.add_argument("--no_aux_loss", dest="aux_loss", action="store_false")
    p.add_argument(
        "--set_loss", default="hungarian", choices=("sequential", "hungarian", "lexicographical")
    )
    p.add_argument("--ce_loss_coef", default=1.0, type=float)
    p.add_argument("--bbox_loss_coef", default=5.0, type=float)
    p.add_argument("--giou_loss_coef", default=2.0, type=float)
    p.add_argument("--eos_coef", default=0.1, type=float)

    p.set_defaults(contrastive_loss=True, contrastive_align_loss=True)
    p.add_argument("--no_contrastive_loss", dest="contrastive_loss", action="store_false")
    p.add_argument(
        "--no_contrastive_align_loss", dest="contrastive_align_loss", action="store_false"
    )
    p.add_argument("--contrastive_loss_hdim", default=64, type=int)
    p.add_argument("--temperature_NCE", default=0.07, type=float)
    p.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    p.add_argument("--contrastive_align_loss_coef", default=1.0, type=float)
    p.add_argument(
        "--logit_scale_lr",
        default=5e-6,
        type=float,
        help="LR for learnable logit scales (cls/align). Use a small value.",
    )

    # Runtime and checkpoints.
    p.add_argument("--output_dir", default="", type=str)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="resume from ckpt")
    p.add_argument("--load", default=None, help="load weights only")
    p.add_argument("--start-epoch", default=0, type=int)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--test_type", default="test", choices=("test", "testA", "testB"))
    p.add_argument(
        "--grad_accum_steps",
        default=1,
        type=int,
        help="Accumulate gradients over N micro-batches before optimizer.step()",
    )

    metrics_group = p.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--eval_phrase_box_metrics",
        dest="eval_phrase_box_metrics",
        action="store_true",
        help="Enable eval/test-time phrase->box Recall@1/5/10 and phrase->box ECE@0.50.",
    )
    metrics_group.add_argument(
        "--no_eval_phrase_box_metrics",
        dest="eval_phrase_box_metrics",
        action="store_false",
        help="Disable eval/test-time phrase->box Recall@1/5/10 and phrase->box ECE@0.50.",
    )
    p.set_defaults(eval_phrase_box_metrics=False)

    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--use_text_cross_attn",
        dest="use_text_cross_attn",
        action="store_true",
        help="Enable decoder cross-attention to text tokens",
    )
    group.add_argument(
        "--no_text_cross_attn",
        dest="use_text_cross_attn",
        action="store_false",
        help="Disable decoder cross-attention to text tokens",
    )
    p.set_defaults(use_text_cross_attn=True)

    p.add_argument(
        "--dataset_file", default="dota", choices=("dota",), help="Which dataset to use."
    )

    p.add_argument(
        "--transformer_type",
        default="deformable",
        choices=("deformable", "vanilla"),
        help="deformable = MSDeformAttn (MMCV); vanilla = standard MultiheadAttention (no deformable attn).",
    )
    p.add_argument(
        "--no_deformable_attn", action="store_true", help="Alias for --transformer_type vanilla"
    )
    p.add_argument(
        "--align_scale_mode",
        default="learnable",
        choices=("learnable", "hardcoded"),
        help="Ablation: learnable uses model logit_scale_align, hardcoded uses 1/temperature_NCE.",
    )
    p.add_argument(
        "--cls_scale_mode",
        default="learnable",
        choices=("learnable", "off"),
        help="Ablation: learnable multiplies pred_logits by exp(logit_scale_cls); off disables cls scaling (original MDETR-style).",
    )
    return p


def count_params_model(model):
    """Return (total_params, requires_grad_params)."""
    total = 0
    reqgrad = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            reqgrad += n
    return total, reqgrad


def report_trainable_by_group(optimizer, verbose: bool = True):
    """Report optimizer groups and unique trainable parameter counts."""
    rows = []
    seen = set()
    req_in_opt_dedup = 0
    all_in_opt_dedup = 0
    eff_in_opt_dedup = 0
    overlap = False

    for g in optimizer.param_groups:
        name = g.get("name", "group")
        lr = float(g.get("lr", 0.0))
        wd = float(g.get("weight_decay", 0.0))
        params = [p for p in g["params"] if p is not None]

        n_all = sum(p.numel() for p in params)
        n_req = sum(p.numel() for p in params if p.requires_grad)
        n_eff = sum(p.numel() for p in params if p.requires_grad and lr > 0.0)
        rows.append((name, n_all, n_req, n_eff, lr, wd))

        for p in params:
            pid = id(p)
            if pid in seen:
                overlap = True
            else:
                seen.add(pid)
                all_in_opt_dedup += p.numel()
                if p.requires_grad:
                    req_in_opt_dedup += p.numel()
                    if lr > 0.0:
                        eff_in_opt_dedup += p.numel()

    if verbose:
        print("\n── Optimizer param-groups ─────────────────────────────────")
        print(
            f"{'group':<22} {'total':>12} {'requires_grad':>15} {'effective(lr>0)':>18}   {'lr':>10} {'wd':>8}"
        )
        for name, n_all, n_req, n_eff, lr, wd in rows:
            print(f"{name:<22} {n_all:12,d} {n_req:15,d} {n_eff:18,d}   {lr:10.3g} {wd:8.3g}")
        if overlap:
            print("WARNING: some parameters appear in multiple param groups (overlap detected).")
        print("───────────────────────────────────────────────────────────\n")

    return {
        "in_optimizer_dedup_total": all_in_opt_dedup,
        "in_optimizer_dedup_requires_grad": req_in_opt_dedup,
        "in_optimizer_dedup_effective": eff_in_opt_dedup,
        "overlap": overlap,
        "rows": rows,
    }


def _sanitize_mlflow_metric_name(name: str) -> str:
    """Replace unsupported MLflow metric-name characters."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./: ")
    name = name.replace("@", "_at_")
    return "".join(ch if ch in allowed else "_" for ch in name)


def _metric_scalar(value):
    """Convert training summaries to plain scalars before logging."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return float(value.item()) if value.numel() == 1 else float(value.mean().item())
    return float(value) if isinstance(value, np.generic) else value


def _flatten_metrics_for_mlflow(d, prefix=""):
    """Flatten numeric metrics, summarizing arrays by mean/min/max."""
    flat = {}
    for k, v in d.items():
        raw_name = f"{prefix}{k}"
        name = _sanitize_mlflow_metric_name(raw_name)

        if isinstance(v, dict):
            flat.update(_flatten_metrics_for_mlflow(v, prefix=raw_name + "/"))
            continue

        if isinstance(v, (bool, np.bool_)):
            continue

        if torch.is_tensor(v):
            arr = v.detach().cpu().double().numpy().reshape(-1)
        elif isinstance(v, (int, float, np.number, list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
        else:
            continue

        if arr.size == 0:
            continue
        if not np.isfinite(arr).all():
            warnings.warn(f"Skipping non-finite MLflow metric: {raw_name}", stacklevel=2)
            continue
        if arr.size == 1:
            flat[name] = float(arr.item())
        else:
            flat[name + "/mean"] = float(arr.mean())
            flat[name + "/min"] = float(arr.min())
            flat[name + "/max"] = float(arr.max())

    return flat


def _restore_best_ap(checkpoint, checkpoint_path):
    """Recover legacy best AP from its run log; explicit None means unknown."""
    if "best_ap" in checkpoint:
        value = checkpoint["best_ap"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value) and value >= 0:
                return float(value)
        return None

    log_path = Path(checkpoint_path).parent / "log.txt"
    if not log_path.is_file():
        return None
    values = []
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            epoch = record.get("epoch")
            if (
                not isinstance(epoch, (int, float))
                or isinstance(epoch, bool)
                or not math.isfinite(epoch)
                or not 0 <= epoch <= checkpoint.get("epoch", -1)
            ):
                continue
            value = record.get("val_bbox_ap", record.get("val_ap"))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value) and value >= 0:
                    values.append(float(value))
    return max(values) if values else None


def _checkpoint_payload(model, model_ema, optimizer, epoch, args, best_ap):
    """Keep resume metadata and raw/EMA weights in one consistent payload."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "model_ema": model_ema.state_dict() if model_ema is not None else None,
        "best_ap": best_ap,
    }


def _validate_runtime_args(args):
    """Reject unsupported or invalid settings before loading models or data."""
    if args.frozen_weights is not None:
        raise ValueError("--frozen_weights is not supported; use --load for weights-only loading")
    if args.test_type != "test":
        raise ValueError("DOTA supports --test_type test only")
    if args.set_loss != "hungarian":
        raise ValueError("Only the Hungarian matcher is supported")
    for name in ("batch_size", "grad_accum_steps", "eval_skip"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    checkpoint_path = args.load or args.resume
    if checkpoint_path and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def main(args):
    # Dataset configuration retains precedence over CLI arguments.
    if args.dataset_config:
        with open(args.dataset_config, encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("Dataset configuration must be a JSON object")
        vars(args).update(config)

    _validate_runtime_args(args)

    if getattr(args, "no_deformable_attn", False):
        args.transformer_type = "vanilla"

    if getattr(args, "transformer_type", "deformable") == "vanilla":
        args.num_feature_levels = 1

    print("==== ARGS ====\n", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.set_deterministic(True)

    is_main = utils.is_main_process()

    eval_mode = bool(getattr(args, "eval", False)) or bool(getattr(args, "test", False))

    with mlflow.start_run(run_name=args.run_name or None) if is_main else nullcontext():
        if is_main:
            loggable_args = {}
            for k, v in vars(args).items():
                if k == "tokenizer":
                    continue
                if isinstance(v, (int, float, str, bool)):
                    loggable_args[k] = v
                elif isinstance(v, Path):
                    loggable_args[k] = str(v)
            mlflow.log_params(loggable_args)

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; check PyTorch/CUDA or use --device cpu"
            )

        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir is not None and is_main:
            out_dir.mkdir(parents=True, exist_ok=True)

        model, criterion, contrastive_criterion, weight_dict = build_model(args)
        model.to(device)
        args.tokenizer = model.tokenizer  # used by dataset builder

        # Evaluation never constructs an optimizer or enables parameter gradients.
        if eval_mode:
            for p in model.parameters():
                p.requires_grad_(False)

        model_ema = deepcopy(model) if getattr(args, "ema", False) else None
        if model_ema is not None:
            for p in model_ema.parameters():
                p.requires_grad_(False)
            print(
                f"[EMA] enabled (decay={args.ema_decay}). EMA weights will be used for eval/checkpoints."
            )

        optimizer = None
        effective_trainable = 0

        if not eval_mode:
            extra_groups = getattr(model, "extra_optim_groups", [])
            extra_param_ids = {
                id(p) for g in extra_groups for p in g.get("params", []) if p is not None
            }

            backbone_params, textenc_params, base_params = [], [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue

                if id(p) in extra_param_ids:
                    continue

                if n.startswith("backbone."):
                    backbone_params.append(p)
                elif n.startswith("transformer.text_encoder."):
                    textenc_params.append(p)
                else:
                    base_params.append(p)

            param_groups = [
                {
                    "name": "base",
                    "params": base_params,
                    "lr": args.lr,
                    "base_lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "name": "backbone",
                    "params": backbone_params,
                    "lr": args.lr_backbone,
                    "base_lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
            ]

            if textenc_params:
                param_groups.append(
                    {
                        "name": "text",
                        "params": textenc_params,
                        "lr": args.text_encoder_lr,
                        "base_lr": args.text_encoder_lr,
                        "weight_decay": args.weight_decay,
                    }
                )

            # Preserve group order for optimizer checkpoint compatibility.
            already = {id(p) for g in param_groups for p in g["params"]}
            for g in extra_groups:
                params = []
                for p in g.get("params", []):
                    if p is not None and p.requires_grad and id(p) not in already:
                        params.append(p)
                        already.add(id(p))
                if params:
                    param_groups.append(
                        {
                            "name": g.get("name", "extra"),
                            "params": params,
                            "lr": g.get("lr", args.lr),
                            "base_lr": g.get("base_lr", g.get("lr", args.lr)),
                            "weight_decay": g.get("weight_decay", 0.0),
                        }
                    )

            opt = str(getattr(args, "optimizer", "adamw")).lower()
            if opt == "sgd":
                optimizer = torch.optim.SGD(
                    param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
                )
            elif opt == "adam":
                optimizer = torch.optim.Adam(
                    param_groups, lr=args.lr, weight_decay=args.weight_decay
                )
            elif opt == "adamw":
                optimizer = torch.optim.AdamW(
                    param_groups, lr=args.lr, weight_decay=args.weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {args.optimizer!r}")

        total_params, reqgrad_params = count_params_model(model)
        if eval_mode:
            print(
                f"Model params: total={total_params:,}  requires_grad={reqgrad_params:,} (eval_mode: frozen)"
            )
        else:
            print(
                f"Model params: total={total_params:,}  trainable(requires_grad)={reqgrad_params:,}"
            )

        if is_main:
            mlflow.log_param("n_parameters_total", int(total_params))
            mlflow.log_param("n_parameters_requires_grad", int(reqgrad_params))

        if not eval_mode:
            opt_stats = report_trainable_by_group(optimizer, verbose=True)
            missing_from_optimizer = reqgrad_params - opt_stats["in_optimizer_dedup_requires_grad"]
            if missing_from_optimizer != 0:
                print(
                    f"WARNING: {missing_from_optimizer:,} trainable parameters are not in the optimizer param_groups."
                )
            effective_trainable = opt_stats["in_optimizer_dedup_effective"]
            print(f"Effective trainable (requires_grad & lr>0): {effective_trainable:,}")
            if is_main:
                mlflow.log_param(
                    "n_parameters_effective_trainable_lr_gt_0", int(effective_trainable)
                )
        elif is_main:
            mlflow.log_param("n_parameters_effective_trainable_lr_gt_0", 0)

        if eval_mode:
            ds_val = build_dataset(args.dataset_file, "test", args)
            print(f" eval images: {len(ds_val)}")
            dl_train = None
        else:
            ds_train = build_dataset(args.dataset_file, "train", args)
            ds_val = build_dataset(args.dataset_file, "val", args)
            print(f" train images: {len(ds_train)}\n val images: {len(ds_val)}")

            dl_train = DataLoader(
                ds_train,
                batch_sampler=torch.utils.data.BatchSampler(
                    RandomSampler(ds_train), args.batch_size, drop_last=True
                ),
                collate_fn=partial(utils.collate_fn, False),
                num_workers=4,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
            )

        dl_val = DataLoader(
            ds_val,
            batch_size=args.batch_size,
            sampler=SequentialSampler(ds_val),
            drop_last=False,
            collate_fn=partial(utils.collate_fn, False),
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
        )

        postprocessors = build_postprocessors(args, args.dataset_file)
        best_ap = 0.0
        protect_existing_best = False

        if args.load:
            ckpt = torch.load(args.load, map_location="cpu", weights_only=False)
            prefer_ema = bool(
                getattr(args, "ema", False)
                and ("model_ema" in ckpt)
                and (ckpt["model_ema"] is not None)
            )
            model.load_state_dict(ckpt["model_ema"] if prefer_ema else ckpt["model"], strict=False)
            if getattr(args, "ema", False) and (model_ema is not None):
                if prefer_ema:
                    model_ema.load_state_dict(ckpt["model_ema"], strict=False)
                else:
                    model_ema.load_state_dict(model.state_dict(), strict=False)
            print(f"Loaded weights from {args.load} (prefer_ema={prefer_ema})")

        if args.resume and not args.load:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            if "optimizer" in ckpt and (not eval_mode) and (optimizer is not None):
                optimizer.load_state_dict(ckpt["optimizer"])
            args.start_epoch = ckpt.get("epoch", 0) + 1

            if not eval_mode:
                restored_best = _restore_best_ap(ckpt, args.resume)
                if restored_best is not None:
                    best_ap = restored_best
                else:
                    protect_existing_best = (
                        out_dir is not None and (out_dir / "BEST_checkpoint.pth").is_file()
                    )
                    detail = (
                        "The existing BEST_checkpoint.pth will be preserved."
                        if protect_existing_best
                        else "Best-checkpoint tracking starts with this resumed run."
                    )
                    warnings.warn(f"Previous best AP is unknown. {detail}", stacklevel=2)

            if getattr(args, "ema", False) and (model_ema is not None):
                if "model_ema" in ckpt and ckpt["model_ema"] is not None:
                    model_ema.load_state_dict(ckpt["model_ema"], strict=False)
                else:
                    print(
                        "[EMA] resume: ema weights missing in checkpoint -> resetting from current model"
                    )
                    model_ema.load_state_dict(model.state_dict(), strict=False)

            print(
                f"Loaded checkpoint from {args.resume}"
                if eval_mode
                else f"Resumed training from {args.resume}"
            )

        EvaluatorCls = get_evaluator(args.dataset_file)
        if eval_mode:
            test_model = model_ema if model_ema is not None else model
            evaluator = EvaluatorCls(ds_val, use_cats=False)

            with torch.inference_mode():
                stats = evaluate(
                    test_model,
                    criterion,
                    contrastive_criterion,
                    weight_dict,
                    dl_val,
                    postprocessors,
                    [evaluator],
                    device,
                    args,
                )
            print(json.dumps({f"test_{k}": v for k, v in stats.items()}, indent=2))
            return

        print("==== START TRAINING ====")
        start = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model,
                criterion,
                contrastive_criterion,
                dl_train,
                weight_dict,
                optimizer,
                device,
                epoch,
                args,
                args.clip_max_norm,
                model_ema=(model_ema if getattr(args, "ema", False) else None),
            )

            improved = False
            if epoch % args.eval_skip == 0:
                eval_model = (
                    model_ema if (getattr(args, "ema", False) and model_ema is not None) else model
                )
                val_stats = evaluate(
                    eval_model,
                    criterion,
                    contrastive_criterion,
                    weight_dict,
                    dl_val,
                    postprocessors,
                    [EvaluatorCls(ds_val, use_cats=False)],
                    device,
                    args,
                )
                curr_ap = val_stats.get("bbox_ap", val_stats.get("ap", 0.0))
                print(f"[VAL] epoch {epoch} :: AP={curr_ap:.4f}")

                if not protect_existing_best and curr_ap > best_ap:
                    best_ap = curr_ap
                    improved = True
            else:
                val_stats = {}

            # Save after evaluation so resume metadata includes this epoch's best AP.
            if out_dir is not None and is_main:
                payload = _checkpoint_payload(
                    model,
                    model_ema,
                    optimizer,
                    epoch,
                    args,
                    None if protect_existing_best else best_ap,
                )
                ckpt_path = out_dir / "checkpoint.pth"
                torch.save(payload, ckpt_path)
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                if improved:
                    best_path = out_dir / "BEST_checkpoint.pth"
                    torch.save(payload, best_path)
                    mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

            if is_main:
                metrics_to_log = {f"train_{k}": _metric_scalar(v) for k, v in train_stats.items()}
                metrics_to_log.update({f"val_{k}": _metric_scalar(v) for k, v in val_stats.items()})
                safe_metrics = _flatten_metrics_for_mlflow(metrics_to_log)
                mlflow.log_metrics(safe_metrics, step=epoch)

            if out_dir is not None and is_main:
                with (out_dir / "log.txt").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                **{f"train_{k}": v for k, v in train_stats.items()},
                                **{f"val_{k}": v for k, v in val_stats.items()},
                                "epoch": epoch,
                                "n_parameters_total": int(total_params),
                                "n_parameters_requires_grad": int(reqgrad_params),
                                "n_parameters_effective_trainable_lr_gt_0": int(
                                    effective_trainable
                                ),
                            }
                        )
                        + "\n"
                    )

        total = str(datetime.timedelta(seconds=int(time.time() - start)))
        print(f"Training completed in {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mdetr_g training / evaluation", parents=[get_args_parser()])
    main(parser.parse_args())
