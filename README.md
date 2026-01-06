# MDETR-G (Modulated Detection Transformer — Geospatial)

MDETR-G is a geospatially adapted variant of [**MDETR**](https://github.com/ashkamath/mdetr) for **text-conditioned object detection** in overhead / geospatial imagery. 
It keeps MDETR’s end-to-end “detect what the text describes” formulation, but updates the vision + attention stack to better handle high-resolution 
geospatial scenes and small targets.


## What’s different vs. baseline MDETR?

MDETR-G modifies the original MDETR design in a few key ways:

- **Deformable attention** for more efficient multi-scale spatial reasoning (especially helpful in high-resolution imagery).
- **Swin Transformer backbone** tuned for aerial/overhead imagery (stronger hierarchical feature extraction than natural-image backbones). 
- **Learnable contrastive temperature (τ)** in the text–image alignment objective (rather than a fixed constant).
- **Learnable classification logit temperature (τ_cls) (inverse-temperature / logit-scale on the decoder’s class logits)** to improve confidence calibration and stabilize optimization under soft token targets.
- **Shallower transformer (3 encoder / 3 decoder layers)** to improve training/inference efficiency.

Under the hood, MDETR-G still uses:
- Hungarian matching (set prediction),
- box regression losses (L1 + GIoU),
- soft token prediction,
- contrastive alignment between token and region embeddings.

## Core capability

Given an image and a natural-language query, MDETR-G predicts bounding boxes corresponding to the referenced objects.

## Data used

[DOTA Phrase Grounding](https://drive.google.com/drive/folders/10sYbpxucNDF-EJZAs58-ff9CjNMPdeb9?usp=sharing)

The dataset is a curated subset of the DOTA-v1.5 benchmark, restricted specifically to ship and plane objects found within tiled aerial imagery. To enable phrase grounding, each image is paired with synthetic, object-centric captions that explicitly reference every annotated instance in the scene.

### Dataset structure
- **train/val**: contains the tiles, annotations, and captions used to train/validate MDETR-G.
- **test**: contains the tiles, annotations, and captions to test MDETR-G on unseen data. 

## Training environment

All experiments reported for MDETR-G were run on an NVIDIA DGX Spark (Grace Blackwell) system and an RTX 5000 Ada workstation.

**System A (DGX Spark / Grace Blackwell)**
- GPU: NVIDIA GB10 (Blackwell architecture, sm_121 / compute capability 12.1)
- Memory: 128 GB coherent unified system memory (shared between CPU and GPU)
  - Note: tooling may report slightly less usable “GPU memory” due to reservations/overhead.
- Use requirements.txt for Spark specific library imports
- See GX Spark specific configurations section for more information
  
**System B (Workstation)**
- GPU: NVIDIA RTX 5000 Ada Generation
- Memory: 32 GB device (VRAM) (typical RTX 5000 Ada configuration).
- A requirementsv2,txt is given for specific libraries needed.
  
Quick sanity checks:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), 'GB')"
```
### DGX Spark specific configurations

An venv was created using pyenv’s 3.11.9.

#### Deformable attention (MMCV) on DGX Spark

The deformable transformer path uses MMCV’s CUDA op:
mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttention.

**On DGX Spark (Linux aarch64 + CUDA 13.x + sm_121), what was done:**

- install a CUDA-enabled PyTorch build compatible with Spark (cu130 wheels), and
- build MMCV from source with CUDA ops enabled (otherwise mmcv._ext is missing and deformable attention won’t load).

##### Summary

- Created a venv with python 3.11.9
- Installed PyTorch from the CUDA 13 wheel index (cu130)
- Cloned MMCV and built/install with ops enabled, targeting GB10 (TORCH_CUDA_ARCH_LIST="12.1")
- Disabled pip build isolation so MMCV can “see” the installed torch when compiling ops

**Repo Commands**
```bash
# 1) PyTorch (CUDA 13)
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.9.1+cu130 torchvision==0.24.1 torchaudio==2.9.1

# 2) Install the requirements
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple

# 3) Build MMCV w/ CUDA ops (example: local clone kept gitignored as ./mmcv)
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.1"
export MAX_JOBS="$(nproc)"
pip install -e . -v --no-build-isolation

# 4) Verify compiled extension + deformable attention op loads
python -c "import mmcv; import mmcv._ext; print('mmcv._ext OK')"
python -c "from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention; print('MSDA OK')"
```
## Python version

All experiments were ran with **Python 3.11**.

**Why 3.11?**
- On DGX Spark (Ubuntu 24.04 defaults to Python 3.12), the ML stack we used (**PyTorch cu130 + MMCV CUDA ops**) was validated end-to-end with Python 3.11.
- Python 3.11 generally has the broadest support across the ecosystem for compiled ML/CV dependencies.
- Workstation used the same version of python.
- 
**Setup note (pyenv):**
If you install Python via `pyenv`, make sure `liblzma-dev` is installed *before* building Python, otherwise the stdlib `lzma` module may be missing (`ModuleNotFoundError: No module named '_lzma'`).

## S3 Configurations

This dataset loader is configured to read DOTA images + label TXT files + a captions CSV from Amazon S3, and it uses an S3 client with a botocore.config.Config for retries/timeouts. It also caches parsed annotations locally to speed up future runs (writes dota_annotations.parquet + dota_classes.json).

```json
{
  "bucket": "<object-store-bucket-or-container-name>",
  "images_prefix": "<path/to/images/>",
  "labels_prefix": "<path/to/labels/>",
  "csv_key": "<path/to/metadata.csv>",
  "val_split_ratio": 0.1
}
```

## How to run the code examples

**Train**
```py
python main.py \
  --dataset_file dota \
  --dataset_config configs/dota_config.json \
  --output_dir runs/mdetr_g \
  --run_name mdetr_g \
  --seed 42 \
  --device cuda \
  --backbone satlas_aerial_swinb \
  --transformer_type deformable \
  --num_feature_levels 3 \
  --deform_num_points 4 \
  --no_text_cross_attn \
  --text_encoder_type sentence-transformers/all-MiniLM-L6-v2 \
  --freeze_text_encoder \
  --optimizer adamw \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --weight_decay 1e-4 \
  --schedule step \
  --epochs 40 \
  --lr_drop 35 \
  --batch_size 64 \
  --enc_layers 3 \
  --dec_layers 3 \
  --align_scale_mode learnable \
  --cls_scale_mode learnable \
  --logit_scale_lr 5e-6 \
  --no_contrastive_loss
```

**Test**
```py
python main.py \
  --eval \
  --dataset_file dota \
  --dataset_config configs/dota_config.json \
  --device cuda \
  --backbone satlas_aerial_swinb \
  --transformer_type deformable \
  --num_feature_levels 3 \
  --deform_num_points 4 \
  --no_text_cross_attn \
  --text_encoder_type sentence-transformers/all-MiniLM-L6-v2 \
  --freeze_text_encoder \
  --enc_layers 3 \
  --dec_layers 3 \
  --align_scale_mode learnable \
  --cls_scale_mode learnable \
  --no_contrastive_loss \
  --batch_size 16 \
  --load runs/mdetr_g/BEST_checkpoint.pth
```



