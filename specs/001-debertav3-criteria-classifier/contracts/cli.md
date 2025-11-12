# CLI Contracts: DeBERTaV3 Criteria Classifier

**Created**: 2025-11-12
**Status**: Finalized
**Responsible**: MLOps/Platform | **Accountable**: Tech Lead | **Consulted**: Modeling + QA | **Informed**: PM + SRE

## Overview

This document specifies the command-line interfaces for training, hyperparameter optimization, and reproducibility workflows.

---

## General Conventions

### Hydra Configuration System

All CLIs use [Hydra](https://hydra.cc/) for configuration management.

**Config Location**: `conf/` directory (relative to repo root)

**Main Config**: `conf/config.yaml`

**Config Groups**:
- `model/`: Model architecture settings
- `data/`: Dataset and dataloader settings
- `train/`: Training hyperparameters and optimizations
- `hpo/`: Optuna HPO settings
- `mlflow/`: Experiment tracking configuration

**Override Syntax**:
```bash
# Single override
python scripts/train.py train.learning_rate=3e-5

# Multiple overrides
python scripts/train.py \
    train.learning_rate=3e-5 \
    train.batch_size=32

# Nested overrides
python scripts/train.py \
    model.head.num_layers=2 \
    model.head.hidden_dims=[512,256]

# Boolean overrides
python scripts/train.py train.optimizations.use_compile=true

# List overrides
python scripts/train.py model.head.hidden_dims=[768,384,192]
```

### Output Directories

Hydra creates timestamped output directories:
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml        # Resolved config
        │   ├── overrides.yaml     # Applied overrides
        │   └── hydra.yaml         # Hydra internal config
        ├── train.log              # Training logs
        └── best_model/            # Saved model checkpoint
```

---

## 1. Baseline Training CLI

### Script

`scripts/train.py`

### Purpose

Train a DeBERTaV3-based binary classifier for post-criterion matching with full MLflow logging.

### Usage

```bash
python scripts/train.py [OVERRIDES...]
```

### Required Config Keys

All required keys have defaults in `conf/config.yaml`:

```yaml
model:
  checkpoint: str                    # HuggingFace model ID
  head:
    num_layers: int                  # 0-3
    hidden_dims: list[int]           # Per-layer dims
    activation: str                  # relu, gelu, tanh
    dropout: float                   # 0.0-0.5
    pooling_strategy: str            # cls, mean, max
  tokenizer:
    max_length: int                  # Max sequence length
    truncation: str                  # Truncation strategy
    padding: str                     # Padding strategy

data:
  redsm5_dir: str                    # Path to ReDSM5 CSVs
  dsm5_criteria_path: str            # Path to criteria JSON
  train_ratio: float                 # 0.0-1.0
  val_ratio: float                   # 0.0-1.0
  test_ratio: float                  # 0.0-1.0
  dataloader:
    num_workers: int                 # Dataloader workers
    pin_memory: bool                 # Pinned memory
    persistent_workers: bool         # Keep workers alive

train:
  seed: int                          # Random seed
  epochs: int                        # Number of epochs
  batch_size: int                    # Training batch size
  eval_batch_size: int               # Evaluation batch size
  learning_rate: float               # Optimizer learning rate
  weight_decay: float                # Weight decay
  warmup_ratio: float                # Warmup fraction
  gradient_accumulation_steps: int   # Gradient accumulation
  scheduler: str                     # linear, cosine, constant
  optimizer_type: str                # adamw, adamw_torch_fused, adamw_bnb_8bit
  early_stopping_patience: int       # Epochs before early stopping
  optimizations:
    use_amp: bool                    # Enable mixed precision
    amp_dtype: str                   # bf16 or fp16
    enable_tf32: bool                # Enable TF32
    attention_implementation: str    # sdpa or flash_attention_2
    gradient_checkpointing: bool     # Enable activation checkpointing
    use_compile: bool                # Enable torch.compile
    compile_mode: str                # default, reduce-overhead, max-autotune

mlflow:
  tracking_uri: str                  # MLflow tracking URI
  experiment_name: str               # MLflow experiment name
```

### Examples

**1. Default Training**:
```bash
python scripts/train.py
```

**2. Custom Learning Rate and Batch Size**:
```bash
python scripts/train.py \
    train.learning_rate=5e-5 \
    train.batch_size=32
```

**3. 2-Layer Head with Mean Pooling**:
```bash
python scripts/train.py \
    model.head.num_layers=2 \
    model.head.hidden_dims=[512,256] \
    model.head.pooling_strategy=mean
```

**4. Fast Prototyping (2 epochs, small batch)**:
```bash
python scripts/train.py \
    train.epochs=2 \
    train.batch_size=8 \
    data.dataloader.num_workers=0
```

**5. Maximum Performance**:
```bash
python scripts/train.py \
    train.batch_size=64 \
    train.gradient_accumulation_steps=2 \
    train.optimizations.use_amp=true \
    train.optimizations.amp_dtype=bf16 \
    train.optimizations.use_compile=true \
    train.optimizations.compile_mode=max-autotune
```

**6. Memory-Constrained Setup**:
```bash
python scripts/train.py \
    train.batch_size=8 \
    train.optimizations.gradient_checkpointing=true \
    train.optimizer_type=adamw_bnb_8bit
```

### MLflow Logging

**Logged Parameters** (FR-005):
- All model hyperparameters (checkpoint, head config, tokenizer settings)
- All training hyperparameters (LR, batch size, epochs, optimizer, scheduler)
- Optimization flags (AMP, TF32, compile, etc.)
- Seed

**Logged Metrics** (per epoch):
- `train_loss`, `train_accuracy`, `train_macro_f1`, `train_weighted_f1`
- `val_loss`, `val_accuracy`, `val_macro_f1`, `val_weighted_f1`
- `test_loss`, `test_accuracy`, `test_macro_f1`, `test_weighted_f1`, `test_precision`, `test_recall`

**Logged Artifacts** (FR-012):
- `config/config.yaml` - Resolved Hydra config
- `model/` - Best model checkpoint
- `confusion_matrix.png` - Test set confusion matrix
- `classification_report.json` - Per-class metrics

**Logged Tags** (Lineage, FR-009, US3):
- `git.commit`, `git.branch`, `git.dirty`
- `version.python`, `version.torch`, `version.transformers`, etc.
- `data.hash.*` - Dataset file hashes
- `stage` - "train"

### Exit Codes

- `0`: Success (training completed, test F1 logged)
- `1`: Failure (error during training, check logs)

---

## 2. Hyperparameter Optimization CLI

### Script

`scripts/hpo.py`

### Purpose

Run Optuna-based hyperparameter search with MedianPruner and log all trials to MLflow.

### Usage

```bash
python scripts/hpo.py [OVERRIDES...]
```

### Required Config Keys

In addition to base training config:

```yaml
hpo:
  study_name: str                    # Optuna study name
  n_trials: int                      # Number of trials to run
  pruner:
    type: str                        # MedianPruner
    n_startup_trials: int            # Trials before pruning starts
  search_space:
    head_layers:
      low: int
      high: int
    hidden_dim:
      low: int
      high: int
      log: bool
    pooling_strategy:
      choices: list[str]
    activation:
      choices: list[str]
    dropout:
      low: float
      high: float
    learning_rate:
      low: float
      high: float
      log: bool
    batch_size:
      choices: list[int]
    warmup_ratio:
      low: float
      high: float
    weight_decay:
      low: float
      high: float
    scheduler:
      choices: list[str]
```

### Examples

**1. Default HPO (30 trials)**:
```bash
python scripts/hpo.py
```

**2. Quick HPO (5 trials for testing)**:
```bash
python scripts/hpo.py hpo.n_trials=5
```

**3. Extended HPO (100 trials)**:
```bash
python scripts/hpo.py hpo.n_trials=100
```

**4. Narrow Search Space (LR only)**:
```bash
python scripts/hpo.py \
    hpo.search_space.learning_rate.low=1e-5 \
    hpo.search_space.learning_rate.high=5e-5
```

**5. Custom Study Name**:
```bash
python scripts/hpo.py hpo.study_name=experiment-001
```

### MLflow Logging

**Parent Run**:
- Tags: Lineage metadata, study name
- Params: n_trials, pruner type, sampler type
- Metrics: best_trial_number, best_trial_value
- Artifacts: `best_trial_config.yaml`, `optimization_history.png`, `param_importances.png`

**Nested Trial Runs**:
- Tags: trial_number, trial_status (completed/pruned)
- Params: All suggested hyperparameters (10 params per FR-018)
- Metrics (per epoch): train_loss, val_loss, val_macro_f1, val_accuracy
- Metric (final): best_val_macro_f1

### Output Files

- `best_trial_config.yaml` - Config for best trial (can be used with train.py)
- `optimization_history.png` - F1 progression across trials
- `param_importances.png` - Feature importance of hyperparameters

### Exit Codes

- `0`: Success (all trials completed, best config saved)
- `1`: Failure (error during HPO)

---

## 3. Reproducibility CLI

### Script

`scripts/reproduce.py`

### Purpose

Reproduce an MLflow run from its saved config and compare metrics.

### Usage

```bash
python scripts/reproduce.py --run_id RUN_ID [OPTIONS]
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--run_id` | str | Yes | - | MLflow run ID to reproduce |
| `--tracking_uri` | str | No | `sqlite:///mlflow.db` | MLflow tracking URI |
| `--experiment` | str | No | `debertav3-criteria` | MLflow experiment name |
| `--tolerance` | float | No | `0.005` | Metric tolerance (0.005 = 0.5 pp) |
| `--mode` | str | No | `eval` | `eval` (fast) or `train` (full retrain) |

### Examples

**1. Reproduce with Default Tolerance (±0.5 pp)**:
```bash
python scripts/reproduce.py --run_id abc123def456
```

**2. Custom Tolerance (±1.0 pp)**:
```bash
python scripts/reproduce.py \
    --run_id abc123def456 \
    --tolerance 0.01
```

**3. Full Retraining Mode** (not yet implemented):
```bash
python scripts/reproduce.py \
    --run_id abc123def456 \
    --mode train
```

### MLflow Logging

**Reproduced Run**:
- Tags: `original_run_id`, `stage=reproducibility`, `reproducibility_status` (passed/failed)
- Params: original_run_id, mode, tolerance
- Metrics: `replayed_test_*` (accuracy, F1, etc.), `delta_test_*`, `original_test_*`

### Output

```
================================================================================
Metric Comparison (SC-003: ±0.5 pp tolerance)
================================================================================
✓ test_accuracy       : Original=0.8312, Replayed=0.8305, Delta=0.0007
✓ test_macro_f1       : Original=0.7456, Replayed=0.7461, Delta=0.0005
✓ test_weighted_f1    : Original=0.8234, Replayed=0.8229, Delta=0.0005
✓ test_precision      : Original=0.7623, Replayed=0.7618, Delta=0.0005
✓ test_recall         : Original=0.7289, Replayed=0.7295, Delta=0.0006

================================================================================
✓ REPRODUCIBILITY CHECK PASSED
All metrics within ±0.5 pp tolerance
================================================================================
```

### Exit Codes

- `0`: Success (metrics within tolerance)
- `1`: Failure (metrics exceeded tolerance or error)

---

## Configuration Reference

### Default Values (FR-017 Baseline)

```yaml
model:
  checkpoint: microsoft/deberta-v3-base
  head:
    num_layers: 1
    hidden_dims: [256]
    activation: relu
    dropout: 0.1
    pooling_strategy: cls

train:
  learning_rate: 2.0e-5
  batch_size: 16
  eval_batch_size: 16
  epochs: 5
  warmup_ratio: 0.1
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  scheduler: linear
  optimizer_type: adamw_torch_fused
  early_stopping_patience: 3
  optimizations:
    use_amp: true
    amp_dtype: bf16
    enable_tf32: true
    attention_implementation: sdpa
    gradient_checkpointing: false
    use_compile: false
```

### HPO Search Space (FR-018)

```yaml
hpo:
  search_space:
    head_layers: {low: 0, high: 3}
    hidden_dim: {low: 64, high: 1024, log: true}
    pooling_strategy: {choices: [cls, mean, max]}
    activation: {choices: [relu, gelu, tanh]}
    dropout: {low: 0.0, high: 0.5}
    learning_rate: {low: 1.0e-6, high: 1.0e-4, log: true}
    batch_size: {choices: [4, 8, 16, 32, 64]}
    warmup_ratio: {low: 0.0, high: 0.3}
    weight_decay: {low: 0.0, high: 0.1}
    scheduler: {choices: [linear, cosine, constant]}
```

---

## Error Handling

### Common Errors

**1. Missing Data Files**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/redsm5/train.csv'
```
**Solution**: Ensure data files exist at specified paths in config

**2. MLflow Connection Error**:
```
mlflow.exceptions.MlflowException: Connection refused
```
**Solution**: Check `mlflow.tracking_uri` is correct

**3. CUDA Out of Memory**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solution**: Reduce `train.batch_size`, enable `gradient_checkpointing`, or use `adamw_bnb_8bit`

**4. Invalid Config Override**:
```
omegaconf.errors.ConfigAttributeError: Key 'invalid_key' not in ...
```
**Solution**: Check config key exists in `conf/config.yaml` structure

### Validation Errors

All scripts perform validation before training:

- **Dry-run batch** (FR-008): Validates tokenization and tensor shapes
- **Split validation**: Checks no post leakage across splits
- **Config validation**: Ensures all required keys present

---

## Testing CLIs

### Unit Tests

See `tests/unit/test_dataset.py`, `tests/unit/test_model.py`

### Integration Tests

See `tests/integration/test_train_cli.py`, `tests/integration/test_hpo.py`, `tests/integration/test_reproducibility.py`

**Run Tests**:
```bash
# All tests
pytest tests/

# Integration only
pytest tests/integration/ -v

# Specific test
pytest tests/integration/test_reproducibility.py::test_reproducibility_workflow -v -s
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-13 | Initial CLI specification |

---

**Last Updated**: 2025-11-13
**Maintained By**: MLOps Team
