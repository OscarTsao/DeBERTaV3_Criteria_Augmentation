# Quickstart Guide: DeBERTaV3 Criteria Matching Classifier

**Feature**: Binary classification for Reddit post-criterion matching using DeBERTaV3-base
**Updated**: 2025-11-12
**Status**: Production Ready

## Table of Contents

1. [Installation](#installation)
2. [Data Setup](#data-setup)
3. [Baseline Training](#baseline-training)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
5. [MLflow Monitoring](#mlflow-monitoring)
6. [Reproducibility Workflows](#reproducibility-workflows)
7. [Configuration Overrides](#configuration-overrides)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended, bf16 or fp16 support)
- Git
- 16GB+ RAM, 8GB+ VRAM recommended

### Environment Setup

```bash
# Clone repository
cd /path/to/DeBERTaV3_Criteria_Augmentation

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets
pip install hydra-core omegaconf
pip install optuna
pip install mlflow
pip install scikit-learn pandas numpy
pip install matplotlib seaborn
pip install tqdm

# Optional: For 8-bit optimizers
pip install bitsandbytes

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Data Setup

### Data Directory Structure

Ensure your data is organized as follows:

```
data/
├── redsm5/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── dsm5/
    └── MDD_Criteira.json
```

### Data Format

**ReDSM5 CSV** (`data/redsm5/*.csv`):
- Columns: `post_id`, `text`, `criterion_id`, `status` (0 or 1)

**DSM-5 Criteria JSON** (`data/dsm5/MDD_Criteira.json`):
- Format: `{"criterion_id": "criterion_description_text", ...}`

### Data Validation

```bash
# Check data files exist
ls -lh data/redsm5/
ls -lh data/dsm5/

# Quick data preview
head -n 5 data/redsm5/train.csv
python -c "import json; print(json.load(open('data/dsm5/MDD_Criteira.json')))"
```

---

## Baseline Training

### Default Configuration

The baseline configuration (FR-017) includes:
- **Head**: 1 hidden layer (768→256→2), ReLU activation, dropout=0.1, CLS pooling
- **Optimizer**: AdamW, LR=2e-5, weight_decay=0.01
- **Training**: batch_size=16, 5 epochs, warmup_ratio=0.1
- **Max length**: 512 tokens
- **Optimizations**: bf16 AMP (if available), TF32, SDPA attention

### Run Baseline Training

```bash
# From repo root
cd /path/to/DeBERTaV3_Criteria_Augmentation

# Run with default config
python scripts/train.py

# Monitor progress in terminal (shows progress bars, metrics)
```

### Expected Output

```
================================================================================
Training Configuration
================================================================================
[Hydra config printed here...]

Using device: cuda
✓ TF32 enabled for matmul and cuDNN
Loading tokenizer: microsoft/deberta-v3-base
Building datasets...
Train samples: 1200
Val samples: 150
Test samples: 150
✓ Dry-run successful. Loss: 0.6931

================================================================================
Epoch 1/5
================================================================================
Training: 100%|██████████| 75/75 [00:45<00:00,  1.65it/s]
Evaluating: 100%|██████████| 10/10 [00:03<00:00,  3.21it/s]
Epoch 1: Train Loss=0.5123, Val Loss=0.4567, Val F1=0.7234
✓ New best model saved (F1=0.7234)

[... epochs 2-5 ...]

================================================================================
Final Test Set Evaluation
================================================================================
Test Results: {'loss': 0.4123, 'accuracy': 0.8312, 'macro_f1': 0.7456, ...}

✓ Training complete!
Best Val F1: 0.7654
Test F1: 0.7456
Test Accuracy: 0.8312
```

### Artifacts Saved

- **MLflow Run**: Logged to `sqlite:///mlflow.db`
- **Best Model**: `outputs/<timestamp>/best_model/`
- **Confusion Matrix**: `outputs/<timestamp>/confusion_matrix.png`
- **Config Snapshot**: Logged as MLflow artifact

---

## Hyperparameter Optimization

### Run HPO with Optuna

```bash
# Run 30 trials (default from FR-018)
python scripts/hpo.py

# Run fewer trials for testing
python scripts/hpo.py hpo.n_trials=5

# Override search space
python scripts/hpo.py \
    hpo.search_space.learning_rate.low=1e-5 \
    hpo.search_space.learning_rate.high=5e-5
```

### HPO Search Space (FR-018)

- **Head layers**: 0-3
- **Hidden dim**: 64-1024 (log scale)
- **Pooling**: CLS, mean, max
- **Activation**: ReLU, GELU, tanh
- **Learning rate**: 1e-6 to 1e-4 (log scale)
- **Batch size**: 4, 8, 16, 32, 64
- **Dropout**: 0.0-0.5
- **Warmup ratio**: 0.0-0.3
- **Weight decay**: 0.0-0.1
- **Scheduler**: linear, cosine, constant

### Expected HPO Output

```
================================================================================
Hyperparameter Optimization with Optuna
================================================================================

Starting 30 trials...

[I 2025-11-12 15:23:41] Trial 0 finished with value: 0.7234 (...)
[I 2025-11-12 15:28:13] Trial 1 finished with value: 0.7512 (...)
[I 2025-11-12 15:31:02] Trial 2 pruned at epoch 2
[I 2025-11-12 15:35:47] Trial 3 finished with value: 0.7689 (...)
...

================================================================================
HPO Results
================================================================================
Number of finished trials: 30
Best trial: 12
Best value (macro-F1): 0.7834
Best params: {'head_layers': 2, 'hidden_dim': 384, 'pooling_strategy': 'mean', ...}

✓ Best trial config saved to best_trial_config.yaml
✓ Visualizations generated and logged
✓ HPO complete!
```

### Use Best HPO Config

```bash
# Load best config and retrain
python scripts/train.py --config-name best_trial_config
```

---

## MLflow Monitoring

### Launch MLflow UI

```bash
# From repo root
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Open browser to http://localhost:5000
```

### MLflow Interface

**Experiment View**:
- Experiment name: `debertav3-criteria`
- Filter runs by tags: `git.commit`, `git.branch`, `stage`
- Sort by metrics: `test_macro_f1`, `test_accuracy`

**Run Details**:
- **Parameters**: All hyperparameters logged
- **Metrics**: Training/validation/test metrics per epoch
- **Artifacts**:
  - `config/config.yaml` - Resolved Hydra config
  - `confusion_matrix.png` - Test set confusion matrix
  - `classification_report.json` - Detailed per-class metrics
  - `model/` - Saved PyTorch model

**Tags**:
- `git.commit` - Git SHA at training time
- `git.branch` - Git branch
- `git.dirty` - Whether working tree had uncommitted changes
- `version.python`, `version.torch`, `version.transformers` - Software versions
- `stage` - `train`, `hpo`, or `reproducibility`

**Lineage Metadata** (FR-009, US3):
- Git commit SHA
- Software versions
- Optimization flags (`opt.use_amp`, `opt.amp_dtype`, etc.)
- Dataset hashes (`data.hash.*`)

---

## Reproducibility Workflows

### Overview

Every training run logs comprehensive lineage metadata enabling exact reproduction:
- Git commit SHA and dirty status
- Python, PyTorch, transformers versions
- Dataset file hashes
- Optimization flags
- Resolved Hydra configuration

### Workflow 1: Find and Inspect Runs

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or use Python API
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('debertav3-criteria')

runs = mlflow.search_runs(order_by=['metrics.test_macro_f1 DESC'], max_results=5)
print(runs[['run_id', 'metrics.test_macro_f1', 'tags.git.commit']])
"
```

### Workflow 2: Reproduce a Run

**Automatic Reproduction** (SC-003: ±0.5 pp tolerance):

```bash
# Get run ID from MLflow UI
RUN_ID="abc123def456..."

# Reproduce evaluation only (fast)
python scripts/reproduce.py --run_id $RUN_ID

# With custom tolerance (default 0.005 = 0.5 pp)
python scripts/reproduce.py --run_id $RUN_ID --tolerance 0.01

# Expected output:
# ================================================================================
# Metric Comparison (SC-003: ±0.5 pp tolerance)
# ================================================================================
# ✓ test_accuracy       : Original=0.8312, Replayed=0.8305, Delta=0.0007
# ✓ test_macro_f1       : Original=0.7456, Replayed=0.7461, Delta=0.0005
# ✓ test_weighted_f1    : Original=0.8234, Replayed=0.8229, Delta=0.0005
# ✓ test_precision      : Original=0.7623, Replayed=0.7618, Delta=0.0005
# ✓ test_recall         : Original=0.7289, Replayed=0.7295, Delta=0.0006
#
# ================================================================================
# ✓ REPRODUCIBILITY CHECK PASSED
# All metrics within ±0.5 pp tolerance
# ================================================================================
```

**Manual Reproduction**:

```bash
# 1. Download config from MLflow UI or Python
python -c "
import mlflow
from omegaconf import OmegaConf

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()

run_id = 'abc123def456...'
config_path = client.download_artifacts(run_id, 'config/config.yaml')

cfg = OmegaConf.load(config_path)
print(OmegaConf.to_yaml(cfg))

# Save for later use
OmegaConf.save(cfg, 'reproduced_config.yaml')
"

# 2. Retrain with exact config
python scripts/train.py --config-path . --config-name reproduced_config
```

### Workflow 3: Compare Multiple Runs

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('debertav3-criteria')

# Get all runs with test metrics
runs = mlflow.search_runs(
    filter_string="metrics.test_macro_f1 > 0",
    order_by=["metrics.test_macro_f1 DESC"]
)

# Select columns
comparison = runs[[
    'run_id',
    'tags.git.commit',
    'params.learning_rate',
    'params.batch_size',
    'params.pooling_strategy',
    'metrics.test_macro_f1',
    'metrics.test_accuracy'
]]

print(comparison.head(10))
```

### Workflow 4: Audit Lineage

```python
import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()

run_id = "abc123..."
run = client.get_run(run_id)

# Print lineage metadata
print("Git Commit:", run.data.tags.get('git.commit'))
print("Git Branch:", run.data.tags.get('git.branch'))
print("Git Dirty:", run.data.tags.get('git.dirty'))
print("\nSoftware Versions:")
for key, value in run.data.tags.items():
    if key.startswith('version.'):
        print(f"  {key}: {value}")

print("\nOptimization Flags:")
for key, value in run.data.params.items():
    if key.startswith('opt.'):
        print(f"  {key}: {value}")

print("\nDataset Hashes:")
for key, value in run.data.tags.items():
    if key.startswith('data.hash.'):
        print(f"  {key}: {value}")
```

### Troubleshooting Reproducibility

**Metrics Don't Match**:
- Check git.dirty tag - uncommitted changes may affect results
- Verify dataset hashes match
- Check CUDA/cuDNN version differences
- Ensure same GPU architecture (Ampere, Ada, etc.)

**Config Download Fails**:
- Verify run ID is correct
- Check MLflow tracking URI
- Ensure artifacts were logged (older runs may lack config artifacts)

**Model Download Fails**:
- Use `reproduce.py --mode train` to retrain from config instead
- Check artifact store path is accessible

---

## Configuration Overrides

### Using Hydra CLI

```bash
# Override single parameter
python scripts/train.py train.learning_rate=1e-5

# Override multiple parameters
python scripts/train.py \
    train.learning_rate=3e-5 \
    train.batch_size=32 \
    train.epochs=10

# Override nested parameters
python scripts/train.py \
    model.head.num_layers=2 \
    model.head.hidden_dims=[512,256] \
    model.head.activation=gelu

# Change pooling strategy
python scripts/train.py model.head.pooling_strategy=mean

# Disable optimizations
python scripts/train.py \
    train.optimizations.use_amp=false \
    train.optimizations.use_compile=false
```

### Common Override Scenarios

**Faster Prototyping**:
```bash
python scripts/train.py \
    train.epochs=2 \
    train.batch_size=8 \
    data.dataloader.num_workers=0
```

**Maximum Performance**:
```bash
python scripts/train.py \
    train.batch_size=64 \
    train.gradient_accumulation_steps=2 \
    train.optimizations.use_amp=true \
    train.optimizations.amp_dtype=bf16 \
    train.optimizations.use_compile=true \
    train.optimizations.compile_mode=max-autotune
```

**Debugging**:
```bash
python scripts/train.py \
    train.epochs=1 \
    train.batch_size=2 \
    data.dataloader.num_workers=0 \
    train.optimizations.use_compile=false
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error during training

**Solutions**:
```bash
# Reduce batch size
python scripts/train.py train.batch_size=8

# Enable gradient accumulation
python scripts/train.py \
    train.batch_size=8 \
    train.gradient_accumulation_steps=2  # Effective batch=16

# Enable gradient checkpointing
python scripts/train.py \
    train.optimizations.gradient_checkpointing=true

# Reduce sequence length
python scripts/train.py model.tokenizer.max_length=256

# Use 8-bit optimizer
python scripts/train.py train.optimizer_type=adamw_bnb_8bit
```

#### 2. Slow Training

**Symptoms**: <1 it/s training speed

**Solutions**:
```bash
# Enable mixed precision
python scripts/train.py \
    train.optimizations.use_amp=true \
    train.optimizations.amp_dtype=bf16  # or fp16

# Enable torch.compile
python scripts/train.py train.optimizations.use_compile=true

# Use fused optimizer
python scripts/train.py train.optimizer_type=adamw_torch_fused

# Increase dataloader workers
python scripts/train.py data.dataloader.num_workers=4

# Enable pinned memory
python scripts/train.py data.dataloader.pin_memory=true
```

#### 3. Low F1 Score

**Symptoms**: Validation F1 < 0.70 (SC-002 threshold)

**Solutions**:
```bash
# Run HPO to find better hyperparameters
python scripts/hpo.py

# Try different pooling strategies
python scripts/train.py model.head.pooling_strategy=mean

# Increase model capacity
python scripts/train.py \
    model.head.num_layers=2 \
    model.head.hidden_dims=[512,256]

# Adjust learning rate
python scripts/train.py train.learning_rate=5e-5

# Train longer
python scripts/train.py train.epochs=10
```

#### 4. MLflow Connection Errors

**Symptoms**: `Connection refused` or `No such file or directory: mlflow.db`

**Solutions**:
```bash
# Ensure tracking URI is correct
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Create mlflow.db if missing
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); mlflow.create_experiment('debertav3-criteria')"

# Check file permissions
ls -l mlflow.db mlruns/
```

#### 5. Reproducibility Failures

**Symptoms**: `reproduce.py` reports metrics outside tolerance

**Possible Causes**:
- Different CUDA/cuDNN versions
- Different GPU architecture
- Dataset file changes (check hashes)
- Uncommitted code changes (check `git.dirty` tag)

**Solutions**:
```bash
# Check git status
git status

# Verify dataset hasn't changed
sha256sum data/redsm5/*.csv data/dsm5/*.json

# Use larger tolerance if acceptable
python scripts/reproduce.py --run_id <ID> --tolerance 0.01

# Check for determinism issues
python scripts/train.py train.seed=42  # Try different seeds
```

---

## Performance Benchmarks

### Baseline Performance (FR-017, SC-002)

**Target**: Val macro-F1 ≥ 0.70

**Typical Results** (V100 16GB):
- Training time: ~45 min (5 epochs, batch_size=16)
- Val macro-F1: 0.72-0.75
- Test macro-F1: 0.70-0.73
- Test accuracy: 0.82-0.85

### HPO Performance (SC-004)

**Target**: Best trial improves baseline by ≥2.0 pp absolute

**Typical Results** (30 trials, A100 40GB):
- HPO time: ~8 hours
- Baseline F1: 0.73
- Best trial F1: 0.76-0.78 (2-5 pp improvement ✓)
- Pruned trials: 10-15 / 30

### Optimization Impact

| Optimization | Speed-up | Memory | Notes |
|--------------|----------|--------|-------|
| bf16 AMP | 1.8x | -30% | Recommended for Ampere+ |
| fp16 AMP | 1.6x | -25% | Use if bf16 unavailable |
| TF32 | 1.3x | 0% | Free on Ampere+, enabled by default |
| SDPA | 1.2x | -10% | Default attention backend |
| torch.compile | 1.1-1.3x | +5% | Static shapes only |
| Gradient checkpointing | -20% | -40% | Trade speed for memory |
| Fused AdamW | +5% | 0% | Slight speed improvement |

---

## Next Steps

1. **Train Baseline**: `python scripts/train.py`
2. **Monitor MLflow**: `mlflow ui` → http://localhost:5000
3. **Run HPO**: `python scripts/hpo.py`
4. **Reproduce Best Run**: `python scripts/reproduce.py --run_id <BEST_RUN_ID>`
5. **Deploy Model**: Load best model from MLflow for inference

For questions or issues, refer to:
- Feature spec: `specs/001-debertav3-criteria-classifier/spec.md`
- Implementation plan: `specs/001-debertav3-criteria-classifier/plan.md`
- Data model: `specs/001-debertav3-criteria-classifier/data-model.md`
