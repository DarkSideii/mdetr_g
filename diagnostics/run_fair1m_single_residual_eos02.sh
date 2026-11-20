#!/usr/bin/env bash
# From-scratch FAIR1M ablation: canonical single-residual attention + eos_coef=0.2.
# Use ONLY after installing the matching transformer.py from this branch.
# Preserves original config, original checkpoint, source data, and validation split.
set -euo pipefail

cd /home/nick/mdetr_g
source venv/bin/activate

BASE_CONFIG="configs/fair1m_config.json"
NEW_CONFIG="configs/fair1m_config_single_residual_eos02.json"
RUN_NAME="mdetr_g_fair1m704_single_residual_eos02"
OUT_DIR="runs/${RUN_NAME}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 1
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output directory: ${OUT_DIR}" >&2
  exit 1
fi

python - "${BASE_CONFIG}" "${NEW_CONFIG}" "${RUN_NAME}" "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import mmcv
import mdetr.models.transformer as transformer

if mmcv.__version__ != "2.2.0":
    raise RuntimeError(
        f"This single-residual variant was written for MMCV 2.2.0; got {mmcv.__version__}"
    )

source = Path(transformer.__file__).read_text(encoding="utf-8")
if "identity=torch.zeros_like(q)" not in source or "dropout=0.0" not in source:
    raise RuntimeError(
        "Single-residual transformer not installed. Download and replace "
        "mdetr/models/transformer.py before running this experiment."
    )

base_path, new_path, run_name, out_dir = map(Path, sys.argv[1:])
original = json.loads(base_path.read_text(encoding="utf-8"))
if not isinstance(original, dict):
    raise ValueError("Expected a JSON object in the dataset config")
modified = dict(original)
modified.update(
    eos_coef=0.2,
    run_name=str(run_name),
    output_dir=str(out_dir),
    dataset_file="fair1m",
    seed=42,
    resume=None,
    load=None,
)
new_content = json.dumps(modified, indent=2, sort_keys=True) + "\n"
if new_path.exists():
    if json.loads(new_path.read_text(encoding="utf-8")) != modified:
        raise RuntimeError(
            f"Refusing to overwrite different config at {new_path}"
        )
else:
    new_path.write_text(new_content, encoding="utf-8")

print("MMCV:", mmcv.__version__)
print("Transformer:", transformer.__file__)
print("Single residual: verified by source guard")
print("Training change: eos_coef=0.2 (baseline 0.1)")
print("New config:", new_path)
print("New run:", out_dir)
print("Original FAIR1M data and original run are untouched.")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: config prepared, training not started."
  exit 0
fi

python -m py_compile mdetr/models/transformer.py
mkdir -p "${OUT_DIR}"
cp "${NEW_CONFIG}" "${OUT_DIR}/effective_dataset_config.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python -u main.py \
  --dataset_file fair1m \
  --dataset_config "${NEW_CONFIG}" \
  --device cuda --seed 42 \
  --backbone satlas_aerial_swinb \
  --transformer_type deformable \
  --num_feature_levels 3 --deform_num_points 4 \
  --enc_layers 3 --dec_layers 3 --hidden_dim 256 --num_queries 300 \
  --use_text_cross_attn --freeze_text_encoder \
  --text_encoder_type sentence-transformers/all-MiniLM-L6-v2 \
  --no_contrastive_loss \
  --batch_size 8 --grad_accum_steps 8 \
  --epochs 200 --lr 3e-5 --lr_backbone 3e-6 \
  --logit_scale_lr 3e-7 --weight_decay 1e-4 \
  --optimizer adamw --clip_max_norm 0.1 \
  --schedule step --lr_drop 180 \
  --eos_coef 0.2 \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/train_console.log"
