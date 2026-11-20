#!/usr/bin/env bash
# Controlled, from-scratch FAIR1M MDETR-G experiment:
# only eos_coef changes from 0.1 to 0.2.
# Do NOT run until you have reviewed the query-score audit.
set -euo pipefail

cd /home/nick/mdetr_g
source venv/bin/activate

BASE_CONFIG="configs/fair1m_config.json"
NEW_CONFIG="configs/fair1m_config_eos02.json"
RUN_NAME="mdetr_g_fair1m704_eos02"
OUT_DIR="runs/${RUN_NAME}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing original config: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ -e "${OUT_DIR}/checkpoint.pth" || -e "${OUT_DIR}/BEST_checkpoint.pth" ]]; then
  echo "Refusing to overwrite an existing training run: ${OUT_DIR}" >&2
  exit 1
fi

# Preserve the original config and all dataset files. Use a sibling config
# so any relative paths retain the same base directory.
python - "${BASE_CONFIG}" "${NEW_CONFIG}" "${RUN_NAME}" "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

base_path, new_path, run_name, out_dir = map(Path, sys.argv[1:])
original = json.loads(base_path.read_text(encoding="utf-8"))
if not isinstance(original, dict):
    raise ValueError("Expected a JSON object in the original dataset config")

modified = dict(original)
modified["eos_coef"] = 0.2
modified["run_name"] = str(run_name)
modified["output_dir"] = str(out_dir)
modified["dataset_file"] = "fair1m"
modified["seed"] = 42

new_content = json.dumps(modified, indent=2, sort_keys=True) + "\n"
if new_path.exists():
    if json.loads(new_path.read_text(encoding="utf-8")) != modified:
        raise RuntimeError(
            f"{new_path} already exists with different contents; refusing overwrite"
        )
else:
    new_path.write_text(new_content, encoding="utf-8")

print("Original config:", base_path)
print("New config:", new_path)
print("New run:", out_dir)
print("Only intended training hyperparameter change: eos_coef 0.1 -> 0.2")
print("No resume/load: training starts from scratch.")
PY

mkdir -p "${OUT_DIR}"
cp "${NEW_CONFIG}" "${OUT_DIR}/effective_dataset_config.json"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: prepared the new config; training not started."
  exit 0
fi

# The config loader applies JSON overrides after CLI parsing. The new JSON
# explicitly sets eos_coef=0.2 so it cannot silently revert to 0.1.
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
  --output_dir "${OUT_DIR}"
