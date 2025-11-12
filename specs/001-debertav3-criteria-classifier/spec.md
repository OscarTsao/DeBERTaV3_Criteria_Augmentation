# Feature Specification: DeBERTaV3 Criteria Matching Classifier

**Feature Branch**: `[001-debertav3-criteria-classifier]`  
**Created**: 2025-11-12  
**Status**: Draft  
**Input**: User description: "Use microsoft/deberta-v3-base with a configurable classification head to perform binary classification: determine whether a post matches a criterion description. Inputs are encoded strictly in this order: [CLS] <post> [SEP] <criterion> [SEP]. Manage all configuration with Hydra; perform hyperparameter optimization with Optuna including the classification head architecture (layers, hidden dims, activation, dropout, pooling) and training parameters (optimizer, scheduler, LR, weight decay, warmup, epochs, train/eval batch sizes, gradient accumulation, max_length). Track all runs with MLflow using sqlite:///mlflow.db and the local mlruns/ artifact store. Implement all optimizations listed in Optimization_Examples and Optimization_List."

## Clarifications

### Session 2025-11-12

- Q: Which dataset and pairing approach for the criterion description should we use? → A: Use ReDSM5 posts under `data/redsm5/`; the criterion description text comes from `data/dsm5/` (canonical DSM‑5 symptom descriptions). Positives are (post, symptom) with `status=1`; negatives are pairs where the symptom is not present.
- Q: Data splitting strategy to avoid leakage? → A: Split by `post_id` so all (post, criterion) pairs from the same post stay in the same split; stratify by label.
- Q: Metric and class weighting policy? → A: Optimize macro‑F1; enable class weights.
- Q: HPO budget and improvement threshold? → A: 30 trials; ≥ 2% absolute macro‑F1 gain over baseline.
- Q: Train/val/test split ratios? → A: 80/10/10.
- Q: Negative sampling and label aggregation policy? → A: Post is positive for a symptom iff any sentence has `status=1`; otherwise negative. Negatives include explicit `status=0` and posts/symptoms lacking annotations (implicit negatives).
- Q: Confirm input granularity and contents? → A: Use post‑level inputs only; encode strictly as `[CLS] <post> [SEP] <criterion> [SEP]` (no sentence text appended).
- Q: What should FR-010's optimization requirement refer to? → A: Implement optimizations specified in the existing `Optimization_Examples` and `Optimization_List` files at repo root (includes mixed precision, SDPA/Flash attention, fused optimizers, gradient checkpointing, torch.compile, dataloader optimizations).
- Q: What pooling strategies should be supported for the classification head? → A: [CLS] token as default; HPO searches: [CLS], mean pooling, max pooling.
- Q: What should the baseline configuration include? → A: Standard baseline: 1 hidden layer (768→256→2), AdamW optimizer, LR=2e-5, batch_size=16, warmup=10%, 5 epochs.
- Q: Which Optuna pruner should be used for HPO? → A: MedianPruner with default settings (stops if trial underperforms median at any step).
- Q: What HPO search space ranges should be used? → A: Wide search: head layers [0,3], hidden_dim [64,1024], LR [1e-6,1e-4], batch [4,64], dropout [0,0.5].

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train & Evaluate Baseline (Priority: P1)

As a practitioner, I can train a DeBERTaV3‑base binary classifier that consumes `[CLS] <post> [SEP] <criterion> [SEP]` inputs and outputs match vs non‑match, with configuration managed by Hydra and metrics (accuracy, F1) computed on a validation split.

**Why this priority**: Establishes a working, measurable baseline and validates the end‑to‑end data and model pipeline.

**Independent Test**: Running the training entrypoint with default Hydra config completes an epoch and logs accuracy and F1 to MLflow; a saved model produces predictions for a held‑out batch.

**Acceptance Scenarios**:

1. Given a tokenized batch with paired inputs (post, criterion), When I run one training epoch, Then MLflow records parameters, metrics, and artifacts (config snapshot, model weights).
2. Given a saved model, When I run evaluation on the validation set, Then F1 and accuracy are computed and logged; a confusion matrix artifact is saved.

---

### User Story 2 - Hyperparameter Optimization (Priority: P2)

As an experimenter, I can launch Optuna to search both the classification head architecture and training hyperparameters, selecting the best trial based on validation F1 and logging full trial details to MLflow.

**Why this priority**: Improves model quality and formalizes a repeatable experiment process.

**Independent Test**: Running the HPO CLI for a small budget (e.g., 5 trials) completes successfully, produces a best‑trial config, and logs trial metrics/params to MLflow.

**Acceptance Scenarios**:

1. Given an HPO config with a pruner, When I run N trials, Then pruned trials stop early and the best trial is selected by F1.
2. Given the best trial, When I export the resolved config, Then I can retrain the model with the exact params to reproduce comparable validation metrics.

---

### User Story 3 - Reproducibility & Lineage (Priority: P3)

As a developer, I can reproduce any prior run using the saved Hydra config and seed settings and trace all artifacts and metrics in MLflow.

**Why this priority**: Ensures trustworthy comparisons and auditability.

**Independent Test**: Re‑running training with the same config and seed yields metrics within a small tolerance; MLflow shows linked artifacts and source control info (when available).

**Acceptance Scenarios**:

1. Given a past run’s config artifact, When I rerun training with the same seed, Then validation metrics differ by ≤ 0.5% absolute.
2. Given a run ID, When I inspect MLflow, Then I see parameters, metrics, model, and config snapshot artifacts.

---

### Edge Cases

- Very long post or criterion requiring truncation; ensure truncation strategy preserves critical tokens.
- Empty or near‑empty post/criterion; skip/validate input.
- Class imbalance; report per‑class metrics and consider class weighting.
- Non‑ASCII or unusual punctuation; tokenizer handles gracefully.
- Mismatched or missing labels in dataset; validation/strict error handling.

## Requirements *(mandatory)*

### Functional Requirements

- FR-001: System MUST use `microsoft/deberta-v3-base` tokenizer and encoder.
- FR-002: Inputs MUST be encoded in order: `[CLS] <post> [SEP] <criterion> [SEP]`.
- FR-003: Classification head MUST be configurable (layers, hidden dims, activation, dropout, pooling strategy) via Hydra. Pooling options: [CLS] token (default), mean pooling, max pooling. HPO MUST search across all three pooling strategies.
- FR-004: Training parameters (optimizer, scheduler, LR, weight decay, warmup, epochs, train/eval batch sizes, grad accumulation, max_length) MUST be Hydra‑configurable.
- FR-005: System MUST log parameters, metrics (accuracy, F1), and artifacts (model, config snapshot, evaluation report) to MLflow.
- FR-006: MLflow MUST use `sqlite:///mlflow.db` as tracking URI and `./mlruns/` as artifact store.
- FR-007: Optuna MUST search head architecture and training parameters using MedianPruner (default settings) for early stopping of underperforming trials; objective metric is validation macro‑F1.
- FR-008: A single‑batch dry‑run MUST validate tokenization and tensor shapes before full training.
- FR-009: Random seeds and library versions MUST be recorded for reproducibility.
- FR-010: System MUST implement applicable optimizations from `Optimization_Examples` and `Optimization_List` files (repo root), including: mixed precision (bf16/fp16 AMP with appropriate scaling), efficient attention (SDPA or FlashAttention), fused optimizers (adamw_torch_fused), gradient checkpointing, torch.compile, and dataloader optimizations (pinned memory, multi-worker loading). Implementation MUST be Hydra-configurable to enable/disable individual optimizations.
- FR-011: System MUST provide a CLI for baseline training and for HPO runs with Hydra overrides.
- FR-012: Evaluation MUST output a confusion matrix artifact and a classification report.
- FR-013: Truncation strategy MUST be explicit (e.g., `longest_first`) and documented.
- FR-014: Dataset source and pairing MUST use posts under `data/redsm5/` and canonical criterion descriptions under `data/dsm5/`. Split strategy: group by `post_id` so all (post, criterion) pairs from the same post remain in the same split; apply label stratification; ratios: 80/10/10 (train/val/test).
- FR-015: Metric policy: optimize macro‑F1; enable class weights in the loss (log both macro‑F1 and weighted‑F1 for analysis).
- FR-016: Model inputs MUST be post‑level only (no sentence‑level text concatenation); labels MUST be aggregated to post‑level from sentence annotations. Label policy: a post is positive for a symptom iff any sentence for that post has `status=1`; otherwise negative. Negatives include explicit `status=0` and unannotated (implicit) pairs.
- FR-017: Baseline configuration for comparison MUST be: classification head with 1 hidden layer (768→256→2 units), [CLS] token pooling, ReLU activation, dropout=0.1; training with AdamW optimizer, learning_rate=2e-5, batch_size=16, warmup_ratio=0.1, num_epochs=5, gradient_accumulation_steps=1, max_length=512.
- FR-018: HPO search space MUST include: num_head_layers [0,3] (0=direct linear projection), hidden_dim_per_layer [64,1024] (log scale), learning_rate [1e-6,1e-4] (log scale), batch_size [4,64] (categorical: 4,8,16,32,64), dropout [0.0,0.5], pooling_strategy {cls, mean, max} (categorical), activation {relu, gelu, tanh} (categorical), warmup_ratio [0.0,0.3], weight_decay [0.0,0.1], scheduler {linear, cosine, constant} (categorical).

### Key Entities *(include if feature involves data)*

- Example: TrainingSample — fields: `post: str`, `criterion: str`, `label: {0,1}`.
- Example: TrialConfig — head architecture (num_layers, hidden_dims per layer, activation function, dropout rate, pooling strategy: {cls, mean, max}) + training params (optimizer, scheduler, LR, weight_decay, batch_sizes, epochs).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- SC-001: A baseline training run completes and logs ≥ 10 parameters and ≥ 3 metrics to MLflow; artifacts include model weights and config snapshot.
 - SC-002: Validation macro‑F1 ≥ 0.70 (or baseline +5% absolute, whichever is higher) on the defined dataset split.
 - SC-003: Re‑running with the same config and seed produces validation macro‑F1 within ±0.5 percentage points of the original.
 - SC-004: An HPO run of at least 30 trials completes with pruning enabled and logs all trial details to MLflow; best trial improves baseline macro‑F1 by ≥ 2.0 percentage points (absolute).
