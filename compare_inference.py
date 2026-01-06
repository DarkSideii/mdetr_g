"""
Compare inference/eval performance across multiple checkpoints.

Supports:
- Parameter counts + rough module breakdown
- Optional evaluate() metrics
- Optional forward-only benchmark on a cached batch (to reduce data-loading variance)
"""

import os

# Reduce nondeterminism; disable tokenizer thread fan-out.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import json
import argparse
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler

# Repo root (used for sys.path).
def find_project_root(start: Path) -> Path:
    """Walk up from `start` until a directory containing `mdetr/` is found."""
    for p in [start] + list(start.parents):
        if (p / "mdetr").is_dir():
            return p
    return start

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

# Project imports (import torch first to avoid occasional loader issues).
import mdetr.util.misc as utils
from mdetr.models import build_model
from mdetr.models.postprocessors import build_postprocessors
from mdetr.engine import evaluate
from geo_datasets import build_dataset, get_evaluator


# Helpers
def torch_load(path: str):
    """Compatible torch.load across versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def apply_main_alias_rules(args):
    """Match main.py flag alias behavior (e.g., no_deformable_attn => vanilla)."""
    if getattr(args, "no_deformable_attn", False):
        args.transformer_type = "vanilla"
    if getattr(args, "transformer_type", "deformable") == "vanilla":
        args.num_feature_levels = 1
    return args


def maybe_apply_dataset_config(args):
    """If args.dataset_config points to a JSON file, merge it into args (best-effort)."""
    cfg = getattr(args, "dataset_config", None)
    if cfg:
        try:
            with open(cfg, "r") as f:
                vars(args).update(json.load(f))
        except FileNotFoundError:
            # Checkpoints may store machine-specific paths; allow CLI overrides.
            pass
    return args


def count_params(model) -> Tuple[int, int]:
    """Return (total_params, requires_grad_params)."""
    total = 0
    req = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            req += n
    return total, req


def param_breakdown(model) -> Dict[str, int]:
    """Rough parameter grouping by name prefix."""
    groups = {
        "backbone": 0,
        "text_encoder": 0,
        "transformer": 0,
        "other": 0,
    }
    for n, p in model.named_parameters():
        k = "other"
        if n.startswith("backbone."):
            k = "backbone"
        elif "text_encoder" in n:
            k = "text_encoder"
        elif n.startswith("transformer.") or ".transformer." in n:
            k = "transformer"
        groups[k] += p.numel()
    return groups


def move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors / tensor-like objects to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        # NestedTensor etc.
        try:
            return obj.to(device)
        except TypeError:
            pass
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj


def extract_samples_targets(batch):
    """Extract (samples, targets) from common batch formats."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]

    if isinstance(batch, dict):
        samples = batch.get("samples") or batch.get("images") or batch.get("inputs")
        targets = batch.get("targets") or batch.get("target") or batch.get("annotations")

        if samples is None or targets is None:
            raise RuntimeError(f"Unrecognized dict batch keys: {list(batch.keys())}")

        if isinstance(targets, tuple):
            targets = list(targets)

        return samples, targets

    raise RuntimeError(
        f"Unrecognized batch format: {type(batch)} / keys={list(batch.keys()) if isinstance(batch, dict) else 'n/a'}"
    )

def forward_model(model, samples, captions, targets=None):
    """Call the model using common MDETR forward signatures."""
    try:
        return model(samples, captions)
    except TypeError:
        pass
    try:
        return model(samples, captions=captions)
    except TypeError:
        pass
    if targets is not None:
        try:
            return model(samples, targets)
        except TypeError:
            pass
    return model(samples)


class LimitedLoader:
    """DataLoader wrapper that yields at most N batches (preserves len())."""
    def __init__(self, dl, n: int):
        self.dl = dl
        self.n = int(n)

    def __iter__(self):
        for i, b in enumerate(self.dl):
            if i >= self.n:
                break
            yield b

    def __len__(self):
        try:
            return min(self.n, len(self.dl))
        except TypeError:
            return self.n


def build_eval_dataloader(args, split: str, num_workers: int, prefetch_factor: int):
    """Build dataset + sequential DataLoader for an eval split."""
    ds = build_dataset(args.dataset_file, split, args)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=SequentialSampler(ds),
        drop_last=False,
        collate_fn=partial(utils.collate_fn, False),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return ds, dl


def benchmark_forward(
    model,
    dl,
    device: torch.device,
    iters: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark forward-only time on a cached batch (minimizes data-loading variance)."""
    model.eval()
    batch = next(iter(dl))
    samples, targets = extract_samples_targets(batch)
    captions = [t.get("caption", "") for t in targets]

    samples = move_to_device(samples, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = forward_model(model, samples, captions, targets=targets)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = forward_model(model, samples, captions, targets=targets)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    sec = (t1 - t0)
    ms_per_iter = 1000.0 * sec / max(1, iters)
    bs = int(getattr(dl, "batch_size", 1) or 1)
    imgs_per_s = (bs * iters) / sec if sec > 0 else float("inf")
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0**2)

    return {
        "bench_ms_per_iter": float(ms_per_iter),
        "bench_imgs_per_s": float(imgs_per_s),
        "bench_peak_mem_mb": float(peak_mem_mb),
    }


def run_eval_metrics(
    model,
    criterion,
    contrastive_criterion,
    weight_dict,
    postprocessors,
    ds,
    dl,
    device: torch.device,
    args,
    num_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run evaluate() metrics, optionally limiting to the first `num_batches`."""
    model.eval()
    EvaluatorCls = get_evaluator(args.dataset_file)
    evaluator = EvaluatorCls(ds, use_cats=False)

    eval_dl = dl
    if num_batches is not None and num_batches > 0:
        eval_dl = LimitedLoader(dl, num_batches)

    with torch.inference_mode():
        stats = evaluate(
            model,
            criterion,
            contrastive_criterion,
            weight_dict,
            eval_dl,
            postprocessors,
            [evaluator],
            device,
            args,
        )
    return stats


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    dataset_file: Optional[str],
    dataset_config: Optional[str],
    batch_size: Optional[int],
    prefer_ema: bool,
) -> Dict[str, Any]:
    """Build an eval-only model from a checkpoint and return components + metadata."""
    ckpt = torch_load(ckpt_path)

    if "args" not in ckpt or ckpt["args"] is None:
        raise RuntimeError(f"Checkpoint missing 'args': {ckpt_path}")

    args = deepcopy(ckpt["args"])

    # Re-attach tokenizer after model build.
    if hasattr(args, "tokenizer"):
        try:
            delattr(args, "tokenizer")
        except Exception:
            setattr(args, "tokenizer", None)

    # Optional dataset overrides so all models run on the same data.
    if dataset_file is not None:
        args.dataset_file = dataset_file
    if dataset_config is not None:
        args.dataset_config = dataset_config

    args = maybe_apply_dataset_config(args)

    # Eval-mode semantics.
    args.eval = True
    args.test = False
    args.device = str(device)

    if batch_size is not None:
        args.batch_size = int(batch_size)

    args = apply_main_alias_rules(args)

    model, criterion, contrastive_criterion, weight_dict = build_model(args)
    model.to(device)
    model.eval()

    # Dataset builders may rely on args.tokenizer.
    args.tokenizer = model.tokenizer

    for p in model.parameters():
        p.requires_grad_(False)

    state = ckpt.get("model", None)
    if prefer_ema and ckpt.get("model_ema", None) is not None:
        state = ckpt["model_ema"]

    if state is None:
        raise RuntimeError(f"Checkpoint missing model weights: {ckpt_path}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    arch_warning = (len(missing) > 0 or len(unexpected) > 0)

    postprocessors = build_postprocessors(args, args.dataset_file)

    total, req = count_params(model)
    breakdown = param_breakdown(model)

    name_guess = Path(ckpt_path).resolve().parent.name

    return {
        "name": name_guess,
        "ckpt_path": ckpt_path,
        "args": args,
        "model": model,
        "criterion": criterion,
        "contrastive_criterion": contrastive_criterion,
        "weight_dict": weight_dict,
        "postprocessors": postprocessors,
        "params_total": total,
        "params_requires_grad": req,
        "params_breakdown": breakdown,
        "arch_warning": arch_warning,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


# CLI entrypoint
def main():
    ap = argparse.ArgumentParser("Compare inference/eval across multiple checkpoints")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="2+ checkpoint paths")
    ap.add_argument("--names", nargs="*", default=None, help="Optional names (same length as checkpoints)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--dataset_file", default=None, help="Override dataset_file for ALL models (e.g. dota)")
    ap.add_argument("--dataset_config", default=None, help="Override dataset_config JSON for ALL models")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch_size for ALL models")
    ap.add_argument("--prefer_ema", action="store_true", help="If checkpoint has model_ema, use it")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=1)

    ap.add_argument("--run_eval", action="store_true", help="Run evaluate() metrics (AP etc.)")
    ap.add_argument("--eval_num_batches", type=int, default=0, help="Limit evaluate() to first N batches (0 = full)")

    ap.add_argument("--bench", action="store_true", help="Run forward-only benchmark on a cached batch")
    ap.add_argument("--bench_iters", type=int, default=100)
    ap.add_argument("--bench_warmup", type=int, default=10)

    args = ap.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("Need at least 2 checkpoints to compare.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")

    models = []
    for i, ckpt_path in enumerate(args.checkpoints):
        m = load_model_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
            dataset_file=args.dataset_file,
            dataset_config=args.dataset_config,
            batch_size=args.batch_size,
            prefer_ema=args.prefer_ema,
        )
        if args.names and i < len(args.names):
            m["name"] = args.names[i]
        models.append(m)

    print("\n==================== MODEL PARAM SUMMARY ====================")
    for m in models:
        bd = m["params_breakdown"]
        print(f"\n[{m['name']}]  ckpt={m['ckpt_path']}")
        print(f"  total params:        {m['params_total']:,}  ({m['params_total']/1e6:.2f} M)")
        print(f"  requires_grad:       {m['params_requires_grad']:,}  (eval frozen)")
        print(f"  breakdown:")
        print(f"    backbone:          {bd['backbone']:,}  ({bd['backbone']/1e6:.2f} M)")
        print(f"    transformer:       {bd['transformer']:,}  ({bd['transformer']/1e6:.2f} M)")
        print(f"    text_encoder:      {bd['text_encoder']:,}  ({bd['text_encoder']/1e6:.2f} M)")
        print(f"    other:             {bd['other']:,}  ({bd['other']/1e6:.2f} M)")

        if m["arch_warning"]:
            print("  [WARNING] state_dict mismatches detected (likely CLI/config mismatch vs checkpoint):")
            print(f"    missing keys:    {len(m['missing_keys'])}")
            print(f"    unexpected keys: {len(m['unexpected_keys'])}")

    results = []

    for m in models:
        print(f"\n==================== RUN: {m['name']} ====================")
        ckpt_args = m["args"]

        ds, dl = build_eval_dataloader(
            ckpt_args,
            split="test",
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        print(f"  dataset size (test split): {len(ds)}")
        print(f"  batch_size: {ckpt_args.batch_size}")

        row = {
            "name": m["name"],
            "ckpt": m["ckpt_path"],
            "params_total": m["params_total"],
            "params_total_m": m["params_total"] / 1e6,
        }

        if args.run_eval:
            nb = args.eval_num_batches if args.eval_num_batches > 0 else None
            t0 = time.perf_counter()
            stats = run_eval_metrics(
                model=m["model"],
                criterion=m["criterion"],
                contrastive_criterion=m["contrastive_criterion"],
                weight_dict=m["weight_dict"],
                postprocessors=m["postprocessors"],
                ds=ds,
                dl=dl,
                device=device,
                args=ckpt_args,
                num_batches=nb,
            )
            t1 = time.perf_counter()
            row["eval_seconds"] = float(t1 - t0)

            print("  eval stats:")
            for k in sorted(stats.keys()):
                v = stats[k]
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v}")
                    row[f"eval_{k}"] = float(v)
            print(f"  eval wall time: {row['eval_seconds']:.2f}s")

        if args.bench:
            bench = benchmark_forward(
                model=m["model"],
                dl=dl,
                device=device,
                iters=args.bench_iters,
                warmup=args.bench_warmup,
            )
            row.update(bench)
            print("  forward benchmark (cached batch):")
            print(f"    {bench['bench_ms_per_iter']:.2f} ms/iter")
            print(f"    {bench['bench_imgs_per_s']:.2f} imgs/s")
            if device.type == "cuda":
                print(f"    {bench['bench_peak_mem_mb']:.1f} MB peak allocated")

        results.append(row)

    print("\n==================== SUMMARY ====================")
    keys = ["name", "params_total_m"]
    if any("bench_ms_per_iter" in r for r in results):
        keys += ["bench_ms_per_iter", "bench_imgs_per_s"]
    for metric in ["eval_bbox_ap", "eval_ap", "eval_loss"]:
        if any(metric in r for r in results):
            keys.append(metric)

    def fmt(k, v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}" if "ap" in k else f"{v:.2f}"
        return str(v)

    colw = {k: max(len(k), max(len(fmt(k, r.get(k))) for r in results)) for k in keys}
    header = "  ".join(k.ljust(colw[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for r in results:
        line = "  ".join(fmt(k, r.get(k)).ljust(colw[k]) for k in keys)
        print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
