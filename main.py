import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic cuBLAS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", message=".*CUDA capability sm_120.*")

import argparse
import json
import random
import time
import datetime
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import mlflow

# ---------------------------------------------------------------------------
# Project import path (adjusts for 2-level nesting)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# ----------------- project utilities -----------------
import multimodal_framework.mdetr.util.misc as utils
from multimodal_framework.mdetr.engine import train_one_epoch, evaluate
from multimodal_framework.mdetr.models import build_model
from multimodal_framework.mdetr.models.postprocessors import build_postprocessors
from multimodal_framework.mdetr.util.misc import NestedTensor, collate_fn  # noqa: F401 (may be unused)

# ----------------- RarePlanes dataset ----------------
from geo_datasets import build_dataset, get_evaluator



# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def get_args_parser():
    p = argparse.ArgumentParser("mdetr_g detector", add_help=False)

    # 1) bookkeeping
    p.add_argument("--run_name", default="", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--no_detection", action="store_true", help="Whether to train the detector")

    # 2) dataset & splits
    p.add_argument("--dataset_config", default=None, nargs="?", const=None, help="JSON with S3 paths / extra flags")
    p.add_argument("--val_split_ratio", default=0.2, type=float, help="Fraction of training set held out for validation")

    # 3) optimisation & schedule
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
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup", "all_cosine_with_warmup"),
    )
    p.add_argument("--lr_drop", default=35, type=int, help="Epoch milestone for *step* or *multistep* sched")
    p.add_argument("--fraction_warmup_steps", default=0.01, type=float)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", default=0.9998, type=float)
    p.add_argument("--eval_skip", default=1, type=int, help="Validate every N epochs")

    # 4) model-level knobs
    p.add_argument("--frozen_weights", default=None, type=str)
    p.add_argument("--freeze_text_encoder", action="store_true")
    p.add_argument(
        "--text_encoder_type",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="HF model name for text encoder (e.g. roberta-base, sentence-transformers/all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--backbone",
        default="ConvNeXt",
        type=str,
        help="Backbone CNN, e.g. ConvNeXt, convnext_tiny/small/base/large, or timm_<name>",
    )
    p.add_argument("--dilation", action="store_true")
    p.add_argument("--position_embedding", default="sine", choices=("sine", "learned"))
    p.add_argument("--masks", action="store_true", help="Enable segmentation heads/losses if present")

    # 5) transformer
    p.add_argument("--enc_layers", default=6, type=int)
    p.add_argument("--dec_layers", default=6, type=int)
    p.add_argument("--dim_feedforward", default=2048, type=int)
    p.add_argument("--hidden_dim", default=256, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--nheads", default=8, type=int)
    p.add_argument("--num_queries", default=100, type=int)
    p.add_argument("--pre_norm", action="store_true")
    p.add_argument("--no_pass_pos_and_query", dest="pass_pos_and_query", action="store_false")

    # Deformable attention (multi-scale always inferred from num_feature_levels)
    p.add_argument("--num_feature_levels", type=int, default=3, help="How many backbone stages to use (1-4). Typically 3 (last 3 stages).")
    p.add_argument("--deform_num_points", type=int, default=4, help="Sampling points per head per level (4 or 8 are common)")

    # 6) losses
    p.add_argument("--set_cost_class", default=1, type=float)
    p.add_argument("--set_cost_bbox", default=5, type=float)
    p.add_argument("--set_cost_giou", default=2, type=float)
    p.add_argument("--no_aux_loss", dest="aux_loss", action="store_false")
    p.add_argument("--set_loss", default="hungarian", choices=("sequential", "hungarian", "lexicographical"))
    p.add_argument("--ce_loss_coef", default=1.0, type=float)
    p.add_argument("--bbox_loss_coef", default=5.0, type=float)
    p.add_argument("--giou_loss_coef", default=2.0, type=float)
    p.add_argument("--qa_loss_coef", default=1.0, type=float)
    p.add_argument("--eos_coef", default=0.1, type=float)

    # contrastive defaults
    p.set_defaults(contrastive_loss=True, contrastive_align_loss=True)
    p.add_argument("--no_contrastive_loss", dest="contrastive_loss", action="store_false")
    p.add_argument("--no_contrastive_align_loss", dest="contrastive_align_loss", action="store_false")
    p.add_argument("--contrastive_loss_hdim", default=64, type=int)
    p.add_argument("--temperature_NCE", default=0.07, type=float)
    p.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    p.add_argument("--contrastive_align_loss_coef", default=1.0, type=float)
    p.add_argument("--logit_scale_lr", default=5e-6, type=float, help="LR for learnable logit scales (cls/align). Use a small value.")

    # 7) runtime / I-O
    p.add_argument("--output_dir", default="", type=str)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="resume from ckpt")
    p.add_argument("--load", default=None, help="load weights only")
    p.add_argument("--start-epoch", default=0, type=int)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--test_type", default="test", choices=("test", "testA", "testB"))
    p.add_argument("--grad_accum_steps", default=1, type=int, help="Accumulate gradients over N micro-batches before optimizer.step()")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--use_text_cross_attn", dest="use_text_cross_attn", action="store_true", help="Enable decoder cross-attention to text tokens")
    group.add_argument("--no_text_cross_attn", dest="use_text_cross_attn", action="store_false", help="Disable decoder cross-attention to text tokens")
    p.set_defaults(use_text_cross_attn=True)
    p.add_argument("--dataset_file", default="rareplanes",   choices=("rareplanes", "dota", "hrsc", "fair1m"),
                   help="Which dataset to use")
    return p

def count_params_model(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def report_trainable_by_group(optimizer):
    """Print a clean table and return a dict with key counts.
    - totals are per-group (can double-count the same tensor across groups)
    - *_dedup are union across groups (no double counting)
    - effective = requires_grad and lr > 0
    """
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

        # dedup across groups
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

    # pretty print
    print("\n── Optimizer param-groups ─────────────────────────────────")
    print(f"{'group':<22} {'total':>12} {'requires_grad':>15} {'effective(lr>0)':>18}   {'lr':>10} {'wd':>8}")
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

def _flatten_metrics_for_mlflow(d, prefix=""):
    """
    Convert a nested dict of metrics into MLflow-safe flat scalars.
    - Scalars pass through.
    - 1-element tensors -> item()
    - Lists/arrays/tensors with >1 element -> mean/min/max summaries.
    - Dicts are flattened with key/subkey.
    Non-numeric entries are skipped.
    """
    flat = {}
    for k, v in d.items():
        name = f"{prefix}{k}" if prefix else k

        # Pure scalars
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[name] = float(v)
            continue

        # Torch tensors
        if torch.is_tensor(v):
            v_cpu = v.detach().cpu()
            if v_cpu.numel() == 1:
                flat[name] = float(v_cpu.item())
            else:
                arr = v_cpu.flatten().numpy().astype(float)
                flat[name + "/mean"] = float(arr.mean())
                flat[name + "/min"] = float(arr.min())
                flat[name + "/max"] = float(arr.max())
            continue

        # Numpy arrays / lists / tuples
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.array(v, dtype=float).flatten()
                if arr.size == 1:
                    flat[name] = float(arr.item())
                elif arr.size > 1:
                    flat[name + "/mean"] = float(arr.mean())
                    flat[name + "/min"] = float(arr.min())
                    flat[name + "/max"] = float(arr.max())
            except Exception:
                # skip non-numeric lists
                pass
            continue

        # Nested dicts
        if isinstance(v, dict):
            flat.update(_flatten_metrics_for_mlflow(v, prefix=name + "/"))
            continue

        # Everything else: skip
    return flat


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    # ---------------- config overrides ----------------
    if args.dataset_config:
        with open(args.dataset_config) as f:
            vars(args).update(json.load(f))

    print("==== ARGS ====\n", args)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch
        torch.set_deterministic(True)

    is_main = utils.is_main_process()  # ← for safe logging in DDP/multiprocess

    # ----------------- MLFLOW RUN -----------------
    with mlflow.start_run(run_name=args.run_name or None):
        # log params once (main process only)
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
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available")

        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        # ---------------- model ----------------
        model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
        model.to(device)
        args.tokenizer = model.tokenizer  # expose to dataset builder

        # --- EMA: create the shadow copy (frozen, no-grad) ----------------------
        model_ema = deepcopy(model) if getattr(args, "ema", False) else None
        if model_ema is not None:
            for p in model_ema.parameters():
                p.requires_grad_(False)
            print(f"[EMA] enabled (decay={args.ema_decay}). EMA weights will be used for eval/checkpoints.")

        # ---------------- optimiser ----------------
        backbone_params, textenc_params, base_params = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("backbone."):
                backbone_params.append(p)
            elif n.startswith("transformer.text_encoder."):
                textenc_params.append(p)  # will be empty if --freeze_text_encoder
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

        # only add text group if it actually has trainable params
        if len(textenc_params) > 0 and hasattr(args, "text_encoder_lr"):
            param_groups.append(
                {
                    "name": "text",
                    "params": textenc_params,
                    "lr": args.text_encoder_lr,
                    "base_lr": args.text_encoder_lr,
                    "weight_decay": args.weight_decay,
                }
            )

        # ---- append extra groups (e.g., learnable logit scales) LAST ----
        if hasattr(model, "extra_optim_groups"):
            already = {id(p) for g in param_groups for p in g["params"]}
            for g in model.extra_optim_groups:
                params = [p for p in g["params"] if p.requires_grad and id(p) not in already]
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

        # Normalize optimizer choice
        opt = str(getattr(args, "optimizer", "adamw")).lower()
        if opt == "sgd":
            optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif opt == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        elif opt == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer!r}")

        # Accurate counts & audit
        total_params, trainable_params = count_params_model(model)
        print(f"Model params: total={total_params:,}  trainable(requires_grad)={trainable_params:,}")
        if is_main:
            mlflow.log_param("n_parameters_total", int(total_params))
            mlflow.log_param("n_parameters_requires_grad", int(trainable_params))

        # Accurate counts & audit
        opt_stats = report_trainable_by_group(optimizer)

        missing_from_optimizer = trainable_params - opt_stats["in_optimizer_dedup_requires_grad"]
        if missing_from_optimizer != 0:
            print(f"WARNING: {missing_from_optimizer:,} trainable parameters are not in the optimizer param_groups.")

        effective_trainable = opt_stats["in_optimizer_dedup_effective"]
        print(f"Effective trainable (requires_grad & lr>0): {effective_trainable:,}")
        if is_main:
            mlflow.log_param("n_parameters_effective_trainable_lr_gt_0", int(effective_trainable))

        # ---------------- datasets ----------------
        ds_train = build_dataset(args.dataset_file, "train", args)
        ds_val = build_dataset(args.dataset_file, "val", args)
        print(f" train images: {len(ds_train)}\n val images: {len(ds_val)}")

        dl_train = DataLoader(
            ds_train,
            batch_sampler=torch.utils.data.BatchSampler(RandomSampler(ds_train), args.batch_size, drop_last=True),
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

        # evaluator & post-processofdjhbfsdjbfdsj
        postprocessors = build_postprocessors(args, args.dataset_file)

        # ---------------- checkpoints ----------------
        def _maybe_reset_ema_from_model():
            if getattr(args, "ema", False) and (model_ema is not None):
                model_ema.load_state_dict(model.state_dict(), strict=False)

        if args.load:
            ckpt = torch.load(args.load, map_location="cpu", weights_only=False)
            prefer_ema = bool(getattr(args, "ema", False) and ("model_ema" in ckpt) and (ckpt["model_ema"] is not None))
            model.load_state_dict(ckpt["model_ema"] if prefer_ema else ckpt["model"], strict=False)
            if getattr(args, "ema", False) and (model_ema is not None):
                if prefer_ema:
                    model_ema.load_state_dict(ckpt["model_ema"], strict=False)
                else:
                    _maybe_reset_ema_from_model()
            print(f"Loaded weights from {args.load} (prefer_ema={prefer_ema})")

        if args.resume and not args.load:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            if "optimizer" in ckpt and not args.eval:
                optimizer.load_state_dict(ckpt["optimizer"])
            args.start_epoch = ckpt.get("epoch", 0) + 1

            if getattr(args, "ema", False) and (model_ema is not None):
                if "model_ema" in ckpt and ckpt["model_ema"] is not None:
                    model_ema.load_state_dict(ckpt["model_ema"], strict=False)
                else:
                    print("[EMA] resume: ema weights missing in checkpoint -> resetting from current model")
                    _maybe_reset_ema_from_model()

            print(f"Resumed training from {args.resume}")

        # ---------------- evaluation only ----------------
        EvaluatorCls = get_evaluator(args.dataset_file)
        if args.eval:
            test_model = model_ema if model_ema is not None else model
            evaluator = EvaluatorCls(ds_val, use_cats=False)
            stats = evaluate(test_model, criterion, contrastive_criterion, weight_dict,
                             dl_val, postprocessors, [evaluator], device, args)
            print(json.dumps({f"test_{k}": v for k, v in stats.items()}, indent=2))
            return

        # ---------------- training loop ------------------
        print("==== START TRAINING ====")
        start = time.time()
        best_ap = 0.0

        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model,
                criterion,
                contrastive_criterion,
                qa_criterion,
                dl_train,
                weight_dict,
                optimizer,
                device,
                epoch,
                args,
                args.clip_max_norm,
                model_ema=(model_ema if getattr(args, "ema", False) else None),
            )

            # checkpoint every epoch (save raw, and EMA weights when enabled)
            if out_dir is not None:
                ckpt_payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "model_ema": (model_ema.state_dict() if getattr(args, "ema", False) and (model_ema is not None) else None),
                }
                ckpt_path = out_dir / "checkpoint.pth"
                torch.save(ckpt_payload, ckpt_path)
                if is_main:
                    mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

            def make_eval():
                return EvaluatorCls(ds_val, use_cats=False)

            # validation
            if epoch % args.eval_skip == 0:
                eval_model = model_ema if (getattr(args, "ema", False) and model_ema is not None) else model
                val_stats = evaluate(
                    eval_model,
                    criterion,
                    contrastive_criterion,
                    weight_dict,
                    dl_val,
                    postprocessors,
                    [make_eval()],
                    device,
                    args,
                )
                curr_ap = val_stats.get("bbox_ap", val_stats.get("ap", 0.0))
                print(f"[VAL] epoch {epoch} :: AP={curr_ap:.4f}")

                if curr_ap > best_ap and out_dir is not None:
                    best_ap = curr_ap
                    best_payload = {
                        "model": model.state_dict(),  # save RAW as best (original MDETR style)
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "model_ema": (model_ema.state_dict() if getattr(args, "ema", False) and (model_ema is not None) else None),
                    }
                    best_path = out_dir / "BEST_checkpoint.pth"
                    torch.save(best_payload, best_path)
                    if is_main:
                        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
            else:
                val_stats = {}

            # --- MLflow metrics (main process only) ---
            if is_main:
                def _to_float(v):
                    if torch.is_tensor(v):
                        return float(v.detach().cpu().item()) if v.numel() == 1 else float(v.detach().cpu().mean().item())
                    if isinstance(v, np.generic):
                        return float(v)
                    return v

                metrics_to_log = {f"train_{k}": _to_float(v) for k, v in train_stats.items()}
                metrics_to_log.update({f"val_{k}": _to_float(v) for k, v in val_stats.items()})
                safe_metrics = _flatten_metrics_for_mlflow(metrics_to_log)
                if utils.is_main_process():
                    mlflow.log_metrics(safe_metrics, step=epoch)

            # logging to file
            if out_dir is not None and utils.is_main_process():
                with (out_dir / "log.txt").open("a") as f:
                    f.write(json.dumps({
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"val_{k}": v for k, v in val_stats.items()},
                        "epoch": epoch,
                        "n_parameters_total": int(total_params),
                        "n_parameters_requires_grad": int(trainable_params),
                        "n_parameters_effective_trainable_lr_gt_0": int(effective_trainable),
                    }) + "\n")

        total = str(datetime.timedelta(seconds=int(time.time() - start)))
        print(f"Training completed in {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mdetr_g training / evaluation", parents=[get_args_parser()])
    main(parser.parse_args())