# Data Model: DeBERTaV3 Criteria Matching Classifier

**Created**: 2025-11-12
**Status**: Finalized
**Responsible**: Data Engineering | **Accountable**: Tech Lead | **Consulted**: Modeling + QA | **Informed**: PM + MLOps

## Overview

This document defines the core data entities, their relationships, and the data flow for the DeBERTaV3 Criteria Matching Classifier.

---

## Entity Definitions

### 1. Criterion

Represents a single DSM-5 diagnostic criterion.

**Source**: `data/dsm5/MDD_Criteira.json`

**Schema**:
```python
{
    "criterion_id": str,      # e.g., "MDD_1", "MDD_2"
    "text": str,              # Full criterion description
    "category": str,          # "MDD" (Major Depressive Disorder)
}
```

**Example**:
```json
{
    "criterion_id": "MDD_1",
    "text": "Depressed mood most of the day, nearly every day, as indicated by either subjective report or observation made by others.",
    "category": "MDD"
}
```

**Cardinality**: ~9 criteria (MDD-specific)

**Lifecycle**: Static, loaded once at dataset construction

---

### 2. Post

Represents a Reddit post from the ReDSM5 dataset.

**Source**: `data/redsm5/*.csv`

**Schema**:
```python
{
    "post_id": str,           # Unique identifier
    "text": str,              # Full post text
    "metadata": {
        "split": str,         # "train", "val", or "test"
        "num_tokens": int,    # Token count (after tokenization)
    }
}
```

**Example**:
```python
{
    "post_id": "abc123",
    "text": "I've been feeling really down lately. Nothing seems to interest me anymore...",
    "metadata": {
        "split": "train",
        "num_tokens": 287
    }
}
```

**Cardinality**: ~1,500 posts

**Constraints**:
- `text` length: 1-10,000 characters (typical: 200-1,000)
- `split` assigned via stratified group splitting (grouped by `post_id`)

---

### 3. PairSample

Represents a (post, criterion) pair with binary label.

**Source**: Derived from ReDSM5 annotations + criterion descriptions

**Schema**:
```python
{
    "post_id": str,           # Foreign key → Post
    "criterion_id": str,      # Foreign key → Criterion
    "label": int,             # 0 (does not match) or 1 (matches)
    "split": str,             # "train", "val", or "test" (inherited from post)
    "class_weight": float,    # Weight for loss computation (inverse frequency)
}
```

**Example**:
```python
{
    "post_id": "abc123",
    "criterion_id": "MDD_1",
    "label": 1,
    "split": "train",
    "class_weight": 5.67
}
```

**Cardinality**: ~13,500 pairs (1,500 posts × 9 criteria)

**Label Aggregation Logic** (FR-016):
```python
# Post is positive for a criterion IF:
# - ANY sentence in the post has status=1 for that criterion
# Otherwise negative (including unannotated pairs)

def aggregate_labels_to_post_level(annotations):
    post_labels = annotations.groupby(['post_id', 'criterion_id'])['status'].apply(
        lambda x: 1 if (x == 1).any() else 0
    )
    return post_labels
```

**Split Assignment** (FR-014):
- All pairs from the same `post_id` assigned to the same split (train/val/test)
- Stratified by label (maintains ~15% positive rate in each split)
- Ratios: 80% train, 10% val, 10% test

**Class Weight Computation** (FR-015):
```python
# Inverse frequency weighting
positive_weight = total_samples / (2 * num_positive)
negative_weight = total_samples / (2 * num_negative)

# Typical: [1.0, 5.67] for [negative, positive]
```

---

### 4. RunConfig

Represents the complete configuration for a training or HPO run.

**Source**: Hydra configuration + overrides

**Schema**:
```python
{
    "run_id": str,            # MLflow run ID
    "config": dict,           # Resolved OmegaConf config
    "overrides": list[str],   # CLI overrides applied
    "timestamp": datetime,    # Run start time
    "git_commit": str,        # Git SHA at run time
    "software_versions": {
        "python": str,
        "torch": str,
        "transformers": str,
        # ...
    },
    "optimization_flags": {
        "use_amp": bool,
        "amp_dtype": str,
        "use_compile": bool,
        # ...
    },
    "dataset_hashes": {
        "redsm5/train.csv": str,
        "dsm5/MDD_Criteira.json": str,
        # ...
    }
}
```

**Example**:
```python
{
    "run_id": "abc123def456",
    "config": {
        "model": {"checkpoint": "microsoft/deberta-v3-base", ...},
        "train": {"learning_rate": 2e-5, "batch_size": 16, ...}
    },
    "overrides": ["train.learning_rate=2e-5"],
    "timestamp": "2025-11-13T02:30:15",
    "git_commit": "401720a",
    "software_versions": {
        "python": "3.11.5",
        "torch": "2.4.0+cu121",
        "transformers": "4.44.0"
    },
    "optimization_flags": {
        "use_amp": true,
        "amp_dtype": "bf16",
        "use_compile": false
    },
    "dataset_hashes": {
        "redsm5/train.csv": "a3b2c1d4e5f6...",
        "dsm5/MDD_Criteira.json": "f1e2d3c4b5a6..."
    }
}
```

**Storage**: MLflow artifact `config/config.yaml` + run tags/params

**Retrieval**: `get_run_config(run_id)` from `mlflow_utils.py`

---

### 5. TrialResult

Represents the outcome of a single Optuna trial during HPO.

**Source**: Optuna study + MLflow nested runs

**Schema**:
```python
{
    "trial_number": int,      # Optuna trial index
    "trial_id": str,          # Optuna internal ID
    "mlflow_run_id": str,     # Nested MLflow run ID
    "params": dict,           # Trial hyperparameters
    "value": float,           # Objective metric (validation macro-F1)
    "state": str,             # "COMPLETE", "PRUNED", "FAIL"
    "duration": float,        # Trial duration (seconds)
    "intermediate_values": {  # Metrics per epoch
        1: float,
        2: float,
        # ...
    }
}
```

**Example**:
```python
{
    "trial_number": 5,
    "trial_id": "optuna_trial_5",
    "mlflow_run_id": "def456ghi789",
    "params": {
        "head_layers": 2,
        "hidden_dim": 384,
        "pooling_strategy": "mean",
        "learning_rate": 3.2e-5,
        "batch_size": 32
    },
    "value": 0.7612,          # Best validation F1
    "state": "COMPLETE",
    "duration": 285.3,
    "intermediate_values": {
        1: 0.6834,
        2: 0.7291,
        3: 0.7512,
        4: 0.7589,
        5: 0.7612
    }
}
```

**Relationships**:
- One parent HPO run → Many trial runs (nested in MLflow)
- Each trial has associated RunConfig (via mlflow_run_id)

---

## Entity Relationships

```
┌──────────────┐
│  Criterion   │
│ (9 criteria) │
└──────┬───────┘
       │
       │ 1:N
       ▼
┌──────────────┐         ┌──────────────┐
│     Post     │ 1:N     │  PairSample  │
│ (1,500 posts)├────────▶│ (13,500 pairs)│
└──────────────┘         └──────┬───────┘
                                │
                                │ Split Assignment
                                ▼
                         ┌──────────────┐
                         │ Train/Val/   │
                         │ Test Splits  │
                         └──────────────┘

┌──────────────┐         ┌──────────────┐
│  RunConfig   │ 1:1     │ MLflow Run   │
│              ├────────▶│              │
└──────────────┘         └──────┬───────┘
                                │
                                │ Nested Runs
                                ▼
                         ┌──────────────┐
                         │ TrialResult  │
                         │  (HPO only)  │
                         └──────────────┘
```

---

## Data Flow

### 1. Dataset Construction

```
1. Load Criteria
   ├── Read data/dsm5/MDD_Criteira.json
   └── Create Criterion entities (9 criteria)

2. Load Posts
   ├── Read data/redsm5/*.csv
   ├── Parse post_id, text, criterion_id, status
   └── Create Post entities (~1,500 posts)

3. Aggregate Labels (FR-016)
   ├── Group by (post_id, criterion_id)
   ├── Label = 1 IF any sentence has status=1
   └── Create PairSample entities (~13,500 pairs)

4. Compute Splits (FR-014)
   ├── Stratified group split by post_id
   ├── Ratios: 80/10/10 (train/val/test)
   └── Assign split to each PairSample

5. Compute Class Weights (FR-015)
   ├── Calculate inverse frequency per split
   └── Attach class_weight to each PairSample

6. Create Dataset
   └── Return {train: Dataset, val: Dataset, test: Dataset}
```

### 2. Training Run

```
1. Setup
   ├── Load config (Hydra)
   ├── Seed RNGs (FR-009)
   └── Configure MLflow (FR-006)

2. Start MLflow Run
   └── Log lineage metadata (git SHA, versions, hashes, opt flags)

3. Build Datasets
   ├── Call build_criterion_dataset()
   └── Create dataloaders (train/val/test)

4. Create Model
   ├── DeBERTaV3Classifier (FR-001, FR-003)
   └── Move to device

5. Training Loop
   ├── For each epoch:
   │   ├── train_epoch() → train metrics
   │   ├── evaluate(val_loader) → val metrics
   │   ├── Log metrics to MLflow
   │   └── Early stopping check
   └── Save best model

6. Test Evaluation
   ├── Load best model
   ├── evaluate(test_loader) → test metrics
   ├── Log metrics, confusion matrix, classification report
   └── Log model artifact

7. End MLflow Run
   └── Config snapshot saved as artifact
```

### 3. HPO Run

```
1. Setup (same as training)

2. Build Datasets ONCE
   └── Reuse across all trials

3. Start Parent MLflow Run
   └── Log parent lineage metadata

4. Create Optuna Study
   ├── MedianPruner (FR-007)
   └── TPESampler

5. For Each Trial:
   ├── Start Nested MLflow Run
   ├── Suggest hyperparameters (FR-018)
   ├── Create trial config
   ├── Log trial lineage metadata
   ├── Train with pruning:
   │   ├── For each epoch:
   │   │   ├── Train + Evaluate
   │   │   ├── Report to Optuna
   │   │   └── Check pruning (may raise TrialPruned)
   │   └── Return best validation F1
   └── End nested run

6. After All Trials:
   ├── Log best trial info
   ├── Save best_trial_config.yaml
   ├── Generate visualizations
   └── End parent run
```

### 4. Reproducibility Run

```
1. Input: run_id (from MLflow UI)

2. Download Config
   ├── get_run_config(run_id)
   └── Load resolved Hydra config

3. Reproduce Run
   ├── Seed RNGs with same seed
   ├── Build datasets with same seed
   ├── Load model from original run (or retrain)
   └── Evaluate on test set

4. Compare Metrics
   ├── Original vs Reproduced
   ├── Compute deltas
   └── Assert within ±0.5 pp tolerance (SC-003)

5. Log Comparison
   └── New MLflow run with comparison results
```

---

## Data Validation

### Post-Level Validation

```python
def validate_post(post: Post):
    assert post.post_id is not None and len(post.post_id) > 0
    assert post.text is not None and len(post.text) > 0
    assert len(post.text) <= 10000, "Post too long"
    # No empty or whitespace-only posts
    assert post.text.strip() != "", "Empty post after strip"
```

### Pair-Level Validation

```python
def validate_pair(pair: PairSample):
    assert pair.post_id in valid_post_ids
    assert pair.criterion_id in valid_criterion_ids
    assert pair.label in {0, 1}
    assert pair.split in {"train", "val", "test"}
    assert pair.class_weight > 0
```

### Split Validation

```python
def validate_splits(datasets):
    # Check no post appears in multiple splits
    train_posts = set(d['post_id'] for d in datasets['train'])
    val_posts = set(d['post_id'] for d in datasets['val'])
    test_posts = set(d['post_id'] for d in datasets['test'])

    assert len(train_posts & val_posts) == 0, "Post leak: train/val"
    assert len(train_posts & test_posts) == 0, "Post leak: train/test"
    assert len(val_posts & test_posts) == 0, "Post leak: val/test"

    # Check class distribution maintained
    for split_name, dataset in datasets.items():
        positive_rate = sum(d['label'] for d in dataset) / len(dataset)
        assert 0.10 <= positive_rate <= 0.20, f"{split_name} imbalance"
```

---

## Schema Evolution

### Version 1.0 (Current)

- Initial schema
- 9 MDD criteria
- Post-level labels
- Stratified group splitting
- Inverse frequency class weights

### Future Considerations

1. **Multi-Label Extension**:
   - Change `label` from binary to multi-hot vector
   - Support multiple criteria per post simultaneously

2. **Sentence-Level Annotations**:
   - Add Sentence entity
   - Link PairSample to specific sentences (explainability)

3. **Additional Disorders**:
   - Extend beyond MDD to GAD, PTSD, etc.
   - Add disorder_type to Criterion

4. **Temporal Annotations**:
   - Add timestamp to Post for temporal analysis

---

**Last Updated**: 2025-11-13
**Schema Version**: 1.0
