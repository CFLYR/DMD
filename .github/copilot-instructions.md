# Copilot Instructions for DMD

## Project Overview

**Decoupled Multimodal Distilling for Emotion Recognition** - CVPR 2023 Highlight Paper (10% of accepted papers)

Research implementation of a multimodal emotion recognition system using graph-based knowledge distillation across language, acoustic, and visual modalities. The core innovation is decoupling multimodal representations into modality-irrelevant (common) and modality-exclusive (specific) spaces for specialized knowledge transfer.

**Paper:** [arXiv:2303.13802](https://arxiv.org/abs/2303.13802)  
**Official Code:** https://github.com/mdswyz/DMD

## Quick Commands

### Training
```bash
# Train on specific dataset (dataset name set in train.py)
python train.py

# Reproduce all experiments
python reproduce_experiments.py --all --gpu 0

# Run specific experiment configuration
python reproduce_experiments.py --dataset mosi --aligned --glove
python reproduce_experiments.py --dataset mosei --aligned --bert
```

### Testing
```bash
# Test trained model (set model path in run.py line 174)
python test.py
```

### Results Collection
```bash
# View experiment summary
python collect_results.py --summary

# Compare with paper results
python collect_results.py --compare

# Export results to CSV
python collect_results.py --export csv --output results.csv
```

## Architecture Overview

### Core Components

1. **Feature Decoupling** (`trains/singleTask/model/dmd.py`)
   - Each modality (L/A/V) is split into two parts:
     - **Modality-irrelevant (common)**: `c_l`, `c_v`, `c_a` - shared semantic information
     - **Modality-exclusive (specific)**: `s_l`, `s_v`, `s_a` - modality-unique features
   - Implemented via parallel encoders:
     - `encoder_c`: Common space encoder (Conv1d)
     - `encoder_s_l/v/a`: Specific space encoders (Conv1d)
   - Self-reconstruction decoders: `decoder_l/v/a`

2. **Graph Distillation Units** (`trains/singleTask/distillnets/`)
   - **HomoGD** (Homogeneous Graph Distillation): Distills knowledge in the modality-irrelevant space
     - Dynamic graph where each vertex = modality, each edge = distillation weight
     - Weights learned automatically via `get_distillation_kernel_homo.py`
   - **HeteroGD** (Heterogeneous Graph Distillation): Distills knowledge in the modality-exclusive space
     - Uses multimodal transformers for cross-modal attention
     - Weights learned via `get_distillation_kernel.py`

3. **Multimodal Transformers** (`trains/subNets/transformers_encoder/`)
   - Cross-modal attention: `trans_l_with_a`, `trans_l_with_v`, `trans_a_with_l`, etc.
   - Self-attention: `self_attentions_c_l/v/a`
   - Memory transformers: `trans_l_mem`, `trans_a_mem`, `trans_v_mem`

4. **BERT Text Encoder** (`trains/subNets/BertTextEncoder.py`)
   - Optional BERT fine-tuning for text features (768-dim)
   - Alternative: GloVe embeddings (300-dim)

### Training Pipeline

Entry point: `run.py` → `DMD_run()` → `_run()` → `trains/singleTask/DMD.py`

**Training components:**
- Model: 3-part tuple `[DMD_model, HomoGD, HeteroGD]`
- Optimizer: Adam with ReduceLROnPlateau scheduler
- Loss functions:
  - L1Loss: Main regression loss
  - MSE: Reconstruction loss
  - HingeLoss: Similarity constraint
  - CosineEmbeddingLoss: Alignment loss

**Key hyperparameters:**
- `learning_rate`: 0.0001 (both MOSI/MOSEI)
- `batch_size`: 16
- `early_stop`: 10 epochs
- `patience`: 5 (for LR scheduler)
- Dataset-specific dropout rates in `config/config.json`

## Configuration System

### Config File Structure (`config/config.json`)

**Critical flags:**
- `need_data_aligned`: `true` → load aligned features (aligned_50.pkl), `false` → unaligned
- `use_bert`: `true` → 768-dim BERT features, `false` → 300-dim GloVe
- `use_finetune`: `true` → BERT parameters trainable during training
- `need_model_aligned`: `true` → model applies internal alignment constraints
- `attn_mask`: `true` → use attention masks for variable-length sequences

**Feature dimensions:**
- MOSI: `[300/768, 5, 20]` (Text GloVe/BERT, Acoustic, Visual)
- MOSEI: `[300/768, 74, 35]`

**Sequence lengths:**
- MOSI Aligned: `[50, 50, 50]` for L/V/A
- MOSI Unaligned: `[50, 500, 375]`
- MOSEI Aligned: `[50, 50, 50]`
- MOSEI Unaligned: `[50, 500, 500]`

### Experiment Configurations

6 main experiments reproduce Tables 1 & 2 from the paper:

| Dataset | Aligned | Features | Config Flags | Expected ACC7 |
|---------|---------|----------|--------------|---------------|
| MOSI | ✓ | GloVe | `need_data_aligned=true, use_bert=false` | 41.4% |
| MOSI | ✓ | BERT* | `need_data_aligned=true, use_bert=true` | 45.6% |
| MOSI | ✗ | GloVe | `need_data_aligned=false, use_bert=false` | 41.9% |
| MOSEI | ✓ | GloVe | `need_data_aligned=true, use_bert=false` | 53.7% |
| MOSEI | ✓ | BERT* | `need_data_aligned=true, use_bert=true` | 54.5% |
| MOSEI | ✗ | GloVe | `need_data_aligned=false, use_bert=false` | 54.6% |

*Note: Asterisk (*) indicates BERT features in paper tables*

## Data Loading

**Data format:** Pickle files containing train/valid/test splits

**Key arrays in pickle:**
- `text` / `text_bert`: Language features (GloVe/BERT)
- `vision`: Visual features
- `audio`: Acoustic features
- `raw_text`: Original text
- `id`: Sample IDs

**Data loader:** `data_loader.py` → `MMDataLoader` → `MMDataset`
- Automatically selects BERT vs GloVe based on `use_bert` flag
- Supports custom feature overrides via `feature_T/A/V` paths

## Key Conventions

### Model Structure Conventions

**3-component model tuple:**
```python
model = [DMD_model, Homo_GD_Unit, Hetero_GD_Unit]
```
All three are passed together to training/testing functions and optimized jointly.

**Feature naming:**
- `c_*`: Common/shared features (modality-irrelevant space)
- `s_*`: Specific/exclusive features (modality-specific space)
- `_l`, `_v`, `_a`: Language, Visual, Acoustic suffixes
- `Xcom`, `Xprt`: Alternative names for common and private spaces in comments

**Distillation graph edges:**
- Format: `L→A` means "Language distills knowledge to Acoustic"
- Weights are learned dynamically, not manually assigned

### File Organization

**Output directories:**
- `pt/`: Saved model checkpoints (`.pth` files)
- `log/`: Training logs (`dmd-{dataset}.log`)
- `result/normal/`: CSV results (`mosi.csv`, `mosei.csv`)
- `dataset/`: Input pickle files (download separately)

**Code structure:**
- `run.py`: Main entry point with `DMD_run()` function
- `train.py` / `test.py`: Simple wrappers that call `DMD_run()`
- `trains/singleTask/DMD.py`: Training/validation/testing logic
- `trains/singleTask/model/dmd.py`: Core DMD architecture
- `trains/subNets/`: Reusable components (BERT, Transformers, AlignNets)

### Metrics

**Evaluation metrics** (via `utils/metricsTop.py`):
- **ACC7**: 7-class accuracy (primary metric)
- **ACC2**: Binary classification accuracy
- **F1**: F1-score
- **MAE**: Mean Absolute Error (regression)
- **Corr**: Correlation coefficient

**Success criteria:**
- < 2% difference from paper: ✓ Full reproduction
- 2-5% difference: ⚠️ Partial reproduction (likely due to randomness)
- > 5% difference: ✗ Needs investigation

### CUDA and Reproducibility

**Environment setup:**
```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
```

**Seeds:** Default is `[1111, 1112, 1113, 1114, 1115]` for multiple runs
- Set via `setup_seed()` in `utils/functions.py`
- Results averaged across all seeds

**GPU assignment:** `assign_gpu(gpu_ids)` in `utils/functions.py`
- Supports single or multi-GPU training
- Format: `--gpu 0` or `--gpu 0 1 2`

## Common Pitfalls

1. **Config vs Data mismatch:** Ensure `need_data_aligned` matches the actual pickle file loaded (aligned_50.pkl vs unaligned_50.pkl)

2. **BERT path:** The `pretrained` path in config.json may need updating to local BERT model location

3. **Feature dimension consistency:** When using `use_bert=true`, text feature dim automatically becomes 768, not 300

4. **Model save path:** Testing requires setting the correct model path in `run.py` line 174 (not configurable via args)

5. **GD distillation flag:** Training uses `is_distill=True`, testing uses `is_distill=False`

6. **Dataset selection:** Must manually change dataset in `train.py` before running (line 6)

## Prerequisites

- Python 3.8
- PyTorch 1.9.0
- CUDA 11.4

Dependencies in `requirements.txt`:
- easydict, numpy, pandas
- transformers (for BERT)
- scikit-learn, tqdm, pynvml
