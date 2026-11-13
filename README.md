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
- A requirementsv2.txt is given for specific libraries needed.
  
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
- On DGX Spark (Ubuntu 24.04 defaults to Python 3.12), the ML stack used (**PyTorch cu130 + MMCV CUDA ops**) was validated end-to-end with Python 3.11.
- Python 3.11 generally has the broadest support across the ecosystem for compiled ML/CV dependencies.
- Workstation used the same version of python.
  
**Setup note (pyenv):**
If you install Python via `pyenv`, make sure `liblzma-dev` is installed *before* building Python, otherwise the stdlib `lzma` module may be missing (`ModuleNotFoundError: No module named '_lzma'`).

## Local and S3 configurations

The dataset loader can read the same prepared DOTA images + label TXT files + a captions CSV from **local files** or **Amazon S3**. Set `dataset_format` to `"local"` or `"s3"`; the remaining source keys describe directories/files in local mode and prefixes/object keys in S3 mode.

Save the configuration you use as `configs/dota_config.json`, or pass a different filename with `--dataset_config`. The paths below are examples—replace them with the locations of your data. Settings in the JSON take precedence over matching command-line settings.

### Local files

```json
{
  "dataset_format": "local",
  "bucket": "",
  "images_prefix": "data/dota/trainval/images",
  "labels_prefix": "data/dota/trainval/labels",
  "csv_key": "data/dota/trainval/captions.csv",
  "test_images_prefix": "data/dota/test/images",
  "test_labels_prefix": "data/dota/test/labels",
  "test_csv_key": "data/dota/test/captions.csv",
  "val_split_ratio": 0.1,
  "augment_train": false
}
```

- Local dataset paths can be absolute or relative to the **repository root**, not the config file's directory. `~` and environment variables in these paths are expanded.
- `bucket` is ignored in local mode. Images, labels, and captions are read from disk; no S3 client is opened.
- The CSV's `Image` entries can retain their original S3 references. Local mode matches them to local image and label files by filename stem; it does not fetch those URLs.

### Amazon S3

```json
{
  "dataset_format": "s3",
  "bucket": "<your-s3-bucket>",
  "images_prefix": "dota/trainval/images/",
  "labels_prefix": "dota/trainval/labels/",
  "csv_key": "dota/trainval/captions.csv",
  "test_images_prefix": "dota/test/images/",
  "test_labels_prefix": "dota/test/labels/",
  "test_csv_key": "dota/test/captions.csv",
  "val_split_ratio": 0.1
}
```

- Use the bucket name without `s3://`. Prefixes and CSV keys are relative to that bucket; keep the trailing `/` on directory-like prefixes.
- The CSV's `Image` entries should identify the image object key or its `s3://<bucket>/<key>` URI.
- S3 uses the normal boto3 credential configuration and retries/timeouts. Do not put AWS access keys in the dataset JSON or commit them to the repository.

The local example sets `augment_train` to `false` to use existing captions without new caption-generation API calls. S3 training keeps its existing default: optional caption generation is enabled. Add `"augment_train": false` to the S3 config to disable those calls too. This setting controls caption generation, **not** training image transforms, and does not remove existing cached captions.

### Data format, splits, and caching

The shipped loader uses the prepared **ship/plane HBB** data described above:

- The captions CSV needs `Image` and `Description` columns, with one row per chip. Numbered captions such as `1. ... 2. ...` belong in the same `Description` field.
- Images, CSV entries, and `.txt` labels must share a unique filename stem. Local image and label directories are searched recursively.
- Label rows use pixel coordinates in the form `ship x1 y1 x2 y2` or `plane x1 y1 x2 y2`. These are horizontal boxes, not raw eight-coordinate DOTA polygons. Changing `dataset_format` does not change the annotation format or supported classes.
- **Train/val** use the unsuffixed source keys. Validation is held out using `val_split_ratio` and `seed`, grouped by chip filename stem so caption copies stay in the same split.
- **Training** expands the numbered captions into separate image-caption samples. **Validation and test** use one consistent caption per chip: the first numbered caption, or the original caption when it is not numbered.
- **Test** uses the separate `test_*` source keys and the entire test dataset. There is no fallback to training paths. Both `--eval` and `--test` in `main.py` select this independent test data.

**Cache note:** S3 mode caches parsed annotations, **not image bytes**. Train/val use `dota_cache/dota_annotations.parquet` + `dota_cache/dota_classes.json`; test annotations use a separate source-specific cache. Images are still fetched from S3 when samples are loaded. For repeated runs without dataset requests, download the images, labels, and CSV once, then use `"dataset_format": "local"`.

Local mode does not reuse the S3 annotation cache. If you change S3 training sources, captions, or the caption-generation policy, move aside/rebuild the old train/val cache so it does not retain stale data. Test cache names depend on source paths, not object contents; replacing data at the same S3 paths also requires rebuilding its cache.

## How to run the code examples

Run these commands from the repository root, using the environment where the dependencies and MMCV were installed. The train/test examples use the selected `configs/dota_config.json` for either local or S3 data.

**Pretrained model files:** `satlas_aerial_swinb` loads `checkpoints/aerial_swinb_si.pth` locally, or the file selected by `SATLAS_WEIGHTS_PATH`. This file must exist even when evaluating a trained checkpoint because the backbone is constructed before the trained checkpoint is loaded. For a different location:

```bash
export SATLAS_WEIGHTS_PATH=/absolute/path/to/aerial_swinb_si.pth
```

Local dataset loading does not make every dependency offline: the text encoder/tokenizer may still need a first-time Hugging Face download. The offline integrity tests below require those real-model assets to be available locally when a checkpoint is supplied.

**Train**

```bash
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

Use the same model settings as training, including the query count and text cross-attention setting. This evaluates the independent `test_*` sources, not the training run's held-out validation split.

```bash
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

## Model integrity tests

The tests check that the model's components work together as expected: attention sampling, decoder references, token spans, matching, losses, gradients, and evaluation. Run them **before training** and again **after training** with a checkpoint. They also provide a quick regression check after code changes.

The suite uses Python's `unittest` runner through `tests/run.py`; no separate pytest command is needed. The synthetic cases use generic captions/classes rather than requiring a particular DOTA image or class list.

### What each test file does

All paths in this table are relative to `tests/`.

| File | What it does |
| --- | --- |
| `run.py` | Selects CPU/CUDA, applies offline safeguards, discovers the suite, and reports pass/fail. A checkpoint enables the additional real-data check. |
| `suite/__init__.py` | Makes the suite an importable Python package. |
| `suite/common.py` | Shared generic captions/targets, a locally built fast tokenizer, tiny backbone/text fixtures, an independent attention sampler, and the network guard. |
| `suite/test_attention.py` | Checks bilinear/multi-level sampling and wrapper layouts; separately compares native MMCV CUDA forward/backward results with a PyTorch reference. |
| `suite/test_encoder_decoder.py` | Checks feature shapes, padding/valid ratios, image conditioning, and that supervised box centers seed the following decoder layer's references. |
| `suite/test_matcher_losses.py` | Checks Hungarian costs against independent formulas, fractional token targets, CE/L1/GIoU, empty targets, gradients, and mixed-precision stability. Saturated FP16 logits are allowed to have zero classification gradients. |
| `suite/test_full_model.py` | Runs production model wiring with tiny fixtures: encode/decode, postprocessing, auxiliary/alignment losses, gradient paths, optimizer updates, and learned-scale stability. |
| `suite/test_engine.py` | Checks the training loop's gradient accumulation against equivalent batches, including unequal object counts, partial final windows, and empty images. |
| `suite/test_evaluation.py` | Checks that perfect synthetic predictions receive perfect class-agnostic COCO AP, even with nonstandard category IDs. This tests the evaluator, not model accuracy. |
| `suite/test_spans.py` | Checks class phrases, aliases/plurals, word boundaries, per-object positive-token maps, and rejection of swapped or empty token supervision. |
| `suite/test_real_checkpoint.py` | Loads an actual checkpoint and checks sampled local images/captions, box normalization, token maps, decoder references, predictions, and finite losses. Reports informational IoU hits. |

### Before training

**No checkpoint, dataset download, or JSON file is required for the pre-training tests.** They exercise the production matcher/loss/encoder/decoder with small synthetic inputs and tiny visual/text fixtures. Native MMCV CUDA checks are separate from the reference sampler.

From the repository root, activate your environment and run:

```bash
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -u tests/run.py \
  --repo . \
  --device cuda \
  --require-native-msda
```

Choose the GPU index appropriate for your system. `--require-native-msda` makes missing CUDA/MMCV support a failure rather than silently skipping the native attention checks. The real-checkpoint test is expected to skip at this stage.

For a CPU-only code check:

```bash
CUDA_VISIBLE_DEVICES="" python -u tests/run.py \
  --repo . \
  --device cpu
```

The CPU run uses the reference sampler for synthetic integration and skips native CUDA checks; it does not validate the compiled GPU extension. An optional production phrase-helper check also skips when that helper is not installed. Read the reported skip reasons rather than treating every skip as a failure.

### After training

Add a **full checkpoint** containing both model weights and training arguments. This repeats the synthetic tests and adds checks of the real model, tokenizer, and a sample of the actual dataset. It does not train or modify the supplied checkpoint; the short optimizer tests operate on disposable synthetic models.

The runner is strictly offline. Use a local dataset configuration and make sure the Satlas weights and text model/tokenizer files required by model construction are already on disk. An S3 annotation cache alone is not sufficient because the S3 loader still fetches image bytes. If training used S3, save a local version of the configuration and pass its filename with `--dataset-config`.

Example using `configs/dota_config.json` set to `"dataset_format": "local"`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tests/run.py \
  --repo . \
  --device cuda \
  --require-native-msda \
  --checkpoint runs/mdetr_g/BEST_checkpoint.pth \
  --dataset-config configs/dota_config.json \
  --split val \
  --samples 8
```

- `--dataset-config` overrides dataset settings stored in the checkpoint; omit it when those stored settings already point to the correct local data. Keep the original validation split settings when checking the same held-out data.
- `--split val` checks the held-out validation subset. Change it to `--split test` to check the independent test data using the configured `test_*` paths.
- `--samples 8` checks up to eight image-caption samples, not the entire dataset.
- The test runner uses `--dataset-config` with a hyphen; `main.py` uses `--dataset_config` with an underscore.

**What passing means:** the checked code paths and data/model contracts behave as expected. It does not prove convergence or a particular AP. The checkpoint check reports best-of-all-query IoU@0.50 hits as a geometry diagnostic, not COCO AP. Use the full **Test** evaluation command above to measure held-out detection performance.

## License

MDETR-G like MDETR is released under the Apache 2.0 license. Please see the [LICENSE](mdetr_g/LICENSE) file for more information.



