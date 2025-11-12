# Implementation Plan: DeBERTaV3 Criteria Matching Classifier

**Branch**: `[001-debertav3-criteria-classifier]` | **Date**: 2025-11-12 | **Spec**: `specs/001-debertav3-criteria-classifier/spec.md`
**Input**: Feature specification refined via `/speckit.specify` + `/speckit.clarify`

## Summary

Train and evaluate a binary classifier built on `microsoft/deberta-v3-base` that ingests `[CLS] <post> [SEP] <criterion> [SEP]` pairs and predicts whether the Reddit post satisfies a DSM-5 criterion. Hydra owns all configuration, Optuna explores the search space (including the classification head architecture and training hyperparameters), and MLflow logs every run to the local `mlflow.db` + `mlruns/`. Data pairs come from `data/redsm5/*.csv` plus DSM-5 criterion text in `data/dsm5/MDD_Criteira.json`, aggregated at the post level with 80/10/10 splits grouped by `post_id`. Baseline training (US1), HPO (US2), and reproducibility/lineage (US3) drive the implementation order while ensuring every optimization listed in `Optimization_Examples` and `Optimization_List` is implemented.

## Technical Context

**Language/Version**: Python 3.11 (repo default) with CUDA-capable PyTorch 2.4+  
**Primary Dependencies**: `transformers` (DeBERTaV3-base), `torch`, `datasets`/custom dataloader, `hydra-core`, `omegaconf`, `optuna`, `mlflow`, `scikit-learn` (metrics), `pandas`, `numpy`, `accelerate` (optional for distributed), `bitsandbytes` (optional optimizers)  
**Storage**: Local CSV + JSON data under `data/redsm5/` and `data/dsm5/`; model artifacts tracked via MLflow in `mlruns/`  
**Testing**: `pytest` with GPU-marked tests where needed; smoke CLI invocations under `tests/integration/` plus config snapshot validation  
**Target Platform**: Linux workstation/server with NVIDIA GPU (bf16 preferred); fallback CPU path for sanity checks  
**Project Type**: Single Python project (`src/Project/SubProject`) plus Hydra config tree and scripts  
**Performance Goals**: Baseline validation macro-F1 ≥0.70; HPO best trial improves macro-F1 ≥2 pp; throughput optimizations (AMP, SDPA/Flash attn, gradient checkpointing, `torch.compile`, pinned dataloaders) per Optimization lists  
**Constraints**: Deterministic seeds logged; Hydra config overrides must reproduce runs; tracking URI fixed to `sqlite:///mlflow.db`; artifact dir `./mlruns/`; no hard-coded hyperparameters  
**Scale/Scope**: Dataset ~1.5k posts; HPO 30 trials; single-GPU training with potential expansion to multi-GPU later

## Constitution Check

| Gate | Plan Compliance |
| --- | --- |
| Model & Input Contract | DeBERTaV3-base + tokenizer locked via Hydra `model/base=deberta_v3_base`. Input pipeline builds `[CLS] post [SEP] criterion [SEP]` exactly; tokenizer config (max_length=512, truncation=`longest_first`, padding=`max_length`) stored in `conf/model/tokenizer.yaml`. |
| Configuration & Reproducibility | Hydra config tree (`conf/config.yaml`) with groups: `model/`, `data/`, `train/`, `hpo/`, `mlflow/`, `optimizations/`. Seeds (Python/NumPy/Torch/CUDA) and determinism toggles in `conf/train/seed.yaml`. CLI entrypoints save resolved config to `outputs/<timestamp>/config.yaml` and upload to MLflow. |
| Experiment Tracking | `mlflow.set_tracking_uri("sqlite:///mlflow.db")` + `mlflow.set_experiment("debertav3-criteria")`. Logging plan: parameters, metrics (loss, accuracy, macro-F1, weighted-F1), confusion matrix image, classification report JSON, Optuna trial params, best-model artifact, tokenizer files, Hydra config artifact. |
| Hyperparameter Optimization | Optuna study defined via Hydra `hpo/optuna.yaml` with `n_trials=30`, pruner=`MedianPruner`, sampler=`TPESampler`, direction=`maximize`. Search space covers classification head (layers 1–3, hidden dims 256–1024, dropout 0.1–0.4, activation GELU/SiLU, pooling CLS vs mean), optimizer (adamw_torch_fused / adamw_bnb_8bit), LR (1e-6–5e-5), scheduler (linear/cosine), warmup ratio, weight decay, batch sizes, grad accumulation, max_length. Trials log to MLflow via Optuna callback. |
| Optimization Coverage | Implementation plan includes explicit tasks to enable each item in `Optimization_List` (bf16/fp16 AMP, TF32, SDPA/FlashAttn selector, gradient checkpointing, fused/8-bit optimizers, LoRA readiness, `torch.compile`, CUDA graphs evaluation, dataloader optimizations, sequence packing/length bucketing). Completion recorded in research + tasks. |

Re-check after Phase 1 ensures design artifacts still satisfy all gates.

## Project Structure

### Documentation (this feature)

```text
specs/001-debertav3-criteria-classifier/
├── spec.md
├── plan.md                # this document
├── research.md            # Phase 0 (data audit, optimization decisions, risk log)
├── data-model.md          # Entity definitions + relationships (PostSample, Criterion, Pair, RunConfig, TrialResult)
├── quickstart.md          # Step-by-step: setup env, prepare data, run baseline/HPO, inspect MLflow
├── contracts/
│   └── cli.md             # CLI interface + config contract for train.py / hpo.py
└── tasks.md               # Generated later via /speckit.tasks
```

### Source Code & Config Layout

```text
repo root
├── conf/                          # Hydra config tree (new)
│   ├── config.yaml                # defaults list (model, data, train, hpo, mlflow, optimizations)
│   ├── model/
│   │   ├── base.yaml              # HF checkpoint + tokenizer settings
│   │   └── head.yaml              # classification head search space definitions
│   ├── data/
│   │   └── redsm5.yaml            # paths, split ratios, post-level aggregation toggles
│   ├── train/
│   │   ├── defaults.yaml          # epochs, AMP, grad clip, compile, logging freq
│   │   └── optimizations.yaml     # toggle list aligning with Optimization_List items
│   ├── hpo/
│   │   └── optuna.yaml            # study params, search space ranges, pruner, sampler
│   └── mlflow/
│       └── local.yaml             # tracking URI + experiment names
├── data/
│   ├── redsm5/                    # raw CSVs (posts + annotations)
│   └── data/dsm5/MDD_Criteira.json # criterion descriptions
├── scripts/
│   ├── train.py                   # Hydra-enabled training CLI (US1/US3)
│   └── hpo.py                     # Optuna driver logging to MLflow (US2)
├── src/Project/SubProject/
│   ├── data/dataset.py            # extend to load post+criterion pairs with splits & class weights
│   ├── engine/train_engine.py     # baseline trainer (AMP, grads, logging hooks)
│   ├── engine/eval_engine.py      # evaluation + metrics logging, confusion matrix export
│   ├── models/model.py            # wraps DeBERTaV3 + configurable head
│   ├── utils/
│   │   ├── mlflow_utils.py       # tracking helpers (update to enforce config logging)
│   │   ├── seed.py               # deterministic seeding utilities
│   │   └── log.py                # structured logging (Hydra friendly)
└── tests/
    ├── unit/test_head.py         # verifies configurable head wiring
    ├── unit/test_dataset.py      # ensures pairing + splits group by post_id
    └── integration/test_cli.py   # smoke test Hydra + MLflow logging
```

**Structure Decision**: Keep the existing `src/Project/SubProject` package but augment it with Hydra-aware config loading, new CLI scripts, and tests. Add a top-level `conf/` tree and `scripts/` entrypoints to cleanly separate configuration from code. Documentation + artifacts live under `specs/001-debertav3-criteria-classifier/` per workflow.

## Phase 0: Outline & Research

1. **Data & Compliance Audit**
   - Verify availability/licensing for `data/redsm5` (gated) and DSM-5 text; document usage constraints in `research.md`.
   - Confirm preprocessing needs: lowercasing, whitespace cleanup, handling long posts (>512 tokens) via truncation policy.
   - Decide on class imbalance handling (class weights vs resampling) beyond the clarifications, note rationale.
2. **Optimization Mapping**
   - For each item in `Optimization_List`, note the enabling mechanism (Hydra flag, code change, dependency) and risks.
   - Evaluate hardware support (bf16/TF32, FlashAttention, bitsandbytes) and document fallbacks.
3. **Experiment Tracking Design**
   - Define MLflow experiment hierarchy (baseline vs HPO) and artifact naming conventions.
   - Specify config snapshot format (Hydra `.yaml` + git hash) and retention.
4. **Deliverables**
   - `research.md` summarizing findings, open risks, and decisions (e.g., gradient checkpointing default off unless memory constrained).
5. **Ownership & Exit Gate**
   - `research.md`: Responsible = Modeling, Accountable = Tech Lead, Consulted = PM & Clinical reviewer, Informed = QA & MLOps.
   - `data-model.md`: Responsible = Data Engineering, Accountable = Tech Lead, Consulted = Modeling & QA, Informed = PM & MLOps.
   - `contracts/cli.md`: Responsible = MLOps/Platform, Accountable = Tech Lead, Consulted = Modeling & QA, Informed = PM & SRE.
   - These three artifacts MUST exist (at least in draft form) before Phase 2 work begins.

## Phase 1: Design & Contracts

1. **Data Model (`data-model.md`)**
   - Entities: `Criterion` (id, text), `Post` (post_id, text, metadata), `PairSample` (post_id, criterion_id, label, split, class_weight), `RunConfig`, `TrialResult`.
   - Relationships: PairSample references Post + Criterion; RunConfig links to Hydra config snapshot; TrialResult references Optuna trial + MLflow run.
2. **Contracts/**
   - `contracts/cli.md`: Document Hydra CLI usage (`python scripts/train.py model.head.num_layers=2 train.batch_size=16`), required config keys, and MLflow logging expectations.
   - Define Optuna callback contract (trial -> MLflow run correlation) and failure handling.
3. **Quickstart (`quickstart.md`)**
   - Environment setup (poetry/pip, GPU drivers), data prep (validate CSV paths, create splits cache), run baseline, run HPO, inspect MLflow UI, reproduce best trial.
4. **Agent Context**
   - Run `.specify/scripts/bash/update-agent-context.sh codex` after design docs capture stack additions (Hydra, Optuna, MLflow specifics).
5. **Re-check Gates**
   - Ensure design artifacts reiterate base model, Hydra config coverage, MLflow wiring, Optuna search space, and optimization list mapping.

## Complexity Tracking

No constitution violations expected; we remain within single-project scope and reuse existing code structure (table intentionally left blank).

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| _None_ | – | – |

## Implementation Strategy (Preview)

- **Phase 1 (Setup)**: Create Hydra `conf/` tree, wire MLflow defaults, add CLI scripts, enforce deterministic seeding, set up logging + config snapshot artifacts.
- **Phase 2 (Foundational)**: Extend dataset loader for post+criterion pairs with grouped splits and class weights; implement configurable classification head; integrate Optimization_List toggles (AMP, SDPA/FlashAttn, gradient checkpointing, fused/8-bit optimizers, `torch.compile`, sequence packing, pinned memory, length bucketing, LoRA readiness, CUDA graphs evaluation, dataloader improvements).
- **Phase 3+ (User Stories)**:
  - **US1**: Baseline training + evaluation CLI, MLflow logging, confusion matrix artifact, deterministic reproducibility.
  - **US2**: Optuna integration with MLflow callbacks, search space for head/training params, pruner, best-trial export.
  - **US3**: Reproducibility workflows (config snapshot restore, deterministic reruns, MLflow lineage, comparison reports) and verification tests.
- **Polish**: Documentation updates, final optimization checklist confirmation, config linting, Hydra sweep templates, packaging.

## Dependencies & Risks

- **Hardware**: Need GPU with bf16 or fp16 support; fallback to fp16 + GradScaler if bf16 unavailable. Document detection logic.
- **Dataset Access**: Ensure gating requirements satisfied; store ReDSM5 data securely (handled locally).
- **Token Length**: Some posts exceed 512 tokens—decide on truncation vs sliding window (research task).
- **Optuna/MLflow Coupling**: Guarantee each trial logs to a unique MLflow run; handle restarts gracefully (resume study from RDB or allow sqlite).
- **Optimization Feature Flags**: Some features (FlashAttention, LoRA, bitsandbytes) require optional dependencies—Hydra config must allow disabling when unavailable without violating constitution (document fallback logic).

## Next Steps

1. Complete Phase 0 research tasks and capture in `research.md`.
2. Draft `data-model.md`, `contracts/cli.md`, and `quickstart.md` per Phase 1.
3. Re-run Constitution check; then invoke `/speckit.tasks` to generate the actionable task list.
