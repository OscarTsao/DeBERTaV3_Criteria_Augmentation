---

description: "Task list template for feature implementation"
---

# Tasks: DeBERTaV3 Criteria Matching Classifier

**Input**: Design documents from `/specs/001-debertav3-criteria-classifier/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single project layout rooted at `src/Project/SubProject/`
- Hydra configs live in `conf/`
- CLI entrypoints under `scripts/`
- Tests live in `tests/`

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Draft `specs/001-debertav3-criteria-classifier/research.md` (R=Modeling, A=Tech Lead, C=PM+Clinical reviewer, I=QA+MLOps) capturing dataset audit, optimization mapping, and risks; this doc is a Phase 1 exit gate.
- [ ] T002 Draft `specs/001-debertav3-criteria-classifier/data-model.md` (R=Data Engineering, A=Tech Lead, C=Modeling+QA, I=PM+MLOps) describing Post/Criterion/PairSample/RunConfig/TrialResult entities + relationships; required before Phase 2.
- [ ] T003 Draft `specs/001-debertav3-criteria-classifier/contracts/cli.md` (R=MLOps/Platform, A=Tech Lead, C=Modeling+QA, I=PM+SRE) documenting CLI parameters, Hydra groups, and MLflow expectations; required before Phase 2.
- [ ] T004 Create Hydra config tree (`conf/config.yaml`) with defaults referencing `model/`, `data/`, `train/`, `hpo/`, `mlflow/`, and `optimizations/` groups.
- [ ] T005 Author MLflow defaults in `conf/mlflow/local.yaml` and update `src/Project/SubProject/utils/mlflow_utils.py` to set `sqlite:///mlflow.db` + `./mlruns/`, experiment names, and config snapshot uploads.
- [ ] T006 [P] Bootstrap Hydra-enabled CLI entrypoints `scripts/train.py` and `scripts/hpo.py` that parse configs, seed RNGs (`src/Project/SubProject/utils/seed.py`), and emit resolved configs to `outputs/`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T007 Implement post+criterion dataset builder in `src/Project/SubProject/data/dataset.py` to load `data/redsm5/*.csv` + `data/dsm5/MDD_Criteira.json`, aggregate labels to post level, compute 80/10/10 splits grouped by `post_id`, and attach class weights.
- [ ] T008 [P] Define Hydra data settings in `conf/data/redsm5.yaml` (paths, split ratios, caching, truncation strategy) and wire them into the dataset factory.
- [ ] T009 Create configurable classification head module in `src/Project/SubProject/models/model.py` plus `conf/model/head.yaml` covering layers, hidden dims, activation, dropout, pooling, and link it to the DeBERTaV3 backbone.
- [ ] T010 Update training configuration files `conf/train/defaults.yaml` + `conf/train/optimizations.yaml` to expose toggles for AMP (bf16/fp16), TF32, gradient checkpointing, fused/8-bit optimizers, LoRA readiness, `torch.compile`, CUDA graphs, sequence packing, length bucketing, and dataloader worker settings per `Optimization_List`.
- [ ] T011 Extend `src/Project/SubProject/engine/train_engine.py` to honor all optimization toggles (AMP contexts, TF32 flags, optimizer selection, gradient checkpointing, compile/cuda graphs) and emit telemetry for each choice.
- [ ] T012 [P] Add data pipeline performance features in `src/Project/SubProject/data/dataset.py` + `engine/train_engine.py`: pinned memory, persistent workers, sequence packing + length bucketing, and validation hooks to ensure single-batch dry-run success.

---

## Phase 3: User Story 1 - Train & Evaluate Baseline (Priority: P1) 🎯 MVP

**Goal**: Baseline training CLI that logs metrics/artifacts via MLflow and produces reproducible checkpoints.

**Independent Test**: Running `python scripts/train.py` completes an epoch, logs accuracy + macro/weighted F1 + confusion matrix, and saves the resolved Hydra config artifact.

### Implementation

- [ ] T013 [US1] Wire `scripts/train.py` to orchestrate dataset, model, and `train_engine.py`, ensuring Hydra overrides flow through and resolved config snapshots upload to MLflow.
- [ ] T014 [P] [US1] Implement evaluation pipeline in `src/Project/SubProject/engine/eval_engine.py` to compute accuracy, macro/weighted F1, classification report JSON, and confusion matrix image, then attach results to the MLflow run.
- [ ] T015 [US1] Add deterministic single-batch dry-run + tensor-shape validation utility in `src/Project/SubProject/utils/seed.py` + `scripts/train.py`, failing fast if tokenization or device placement is incorrect.
- [ ] T016 [P] [US1] Create automated tests: `tests/unit/test_dataset.py` (split grouping + label policy) and `tests/integration/test_train_cli.py` (one-epoch smoke run mocked to CPU) to guard regressions.
- [ ] T017 [US1] Document baseline procedure, metrics table, and troubleshooting steps in `specs/001-debertav3-criteria-classifier/quickstart.md`.

---

## Phase 4: User Story 2 - Hyperparameter Optimization (Priority: P2)

**Goal**: Optuna study that searches head architecture + training hyperparameters for ≥2 pp macro-F1 gain and logs every trial to MLflow.

**Independent Test**: Running `python scripts/hpo.py hpo.n_trials=5` completes successfully, records trial metrics/params/artifacts, and exports the best-trial config.

### Implementation

- [ ] T018 [US2] Implement Optuna study driver in `scripts/hpo.py` that instantiates Hydra configs, attaches MedianPruner/TPESampler, and links each trial to an MLflow run with serialized params and metrics.
- [ ] T019 [P] [US2] Encode search space + budget in `conf/hpo/optuna.yaml` (classification head layers/dims/dropout/pooling, optimizer/scheduler/LR/weight_decay, warmup ratio, epochs, train/eval batch sizes, grad accumulation, max_length, AMP/optimization toggles).
- [ ] T020 [US2] Update `src/Project/SubProject/models/model.py` and `engine/train_engine.py` to accept trial-suggested hyperparameters at runtime, saving the best-trial Hydra config + tokenizer/model artifacts to MLflow.
- [ ] T021 [P] [US2] Add Optuna→MLflow callback module in `src/Project/SubProject/utils/mlflow_utils.py` to log intermediate metrics, pruner decisions, and feature importance plots.
- [ ] T022 [US2] Create `tests/integration/test_hpo_cli.py` (small dataset stub) to verify a two-trial run logs metrics and selects a best trial deterministically.

---

## Phase 5: User Story 3 - Reproducibility & Lineage (Priority: P3)

**Goal**: Provide auditable run lineage so any MLflow run can be replayed with the stored config and yield comparable metrics.

**Independent Test**: Invoking the reproducibility CLI with a past MLflow run ID replays training and produces macro-F1 within ±0.5 pp while attaching lineage artifacts.

### Implementation

- [ ] T023 [US3] Enhance `src/Project/SubProject/utils/mlflow_utils.py` to log git SHA, Hydra config artifact, optimization flags, dataset hashes, and software versions for every run.
- [ ] T024 [P] [US3] Build a replay CLI `scripts/reproduce.py` that downloads the MLflow config artifact, restores the exact Hydra overrides, reruns evaluation, and compares metrics to tolerance.
- [ ] T025 [US3] Implement lineage summary reporting in `specs/001-debertav3-criteria-classifier/quickstart.md` (how to find runs, reproduce them, interpret deltas).
- [ ] T026 [P] [US3] Add `tests/integration/test_reproducibility.py` to simulate saving a config and rerunning to assert macro-F1 deltas ≤0.5 percentage points.

---

## Phase N: Polish & Cross-Cutting Concerns

- [ ] T027 [P] Finalize `specs/001-debertav3-criteria-classifier/research.md`, `data-model.md`, and `contracts/cli.md` with decisions made during implementation (dataset audit, entity diagrams, CLI contracts).
- [ ] T028 Confirm every optimization from `Optimization_Examples` + `Optimization_List` is wired and documented (checklist entry in `specs/001-debertav3-criteria-classifier/research.md`).
- [ ] T029 [P] Add developer documentation to `README.md` + `specs/.../quickstart.md` for environment setup, Hydra overrides, MLflow UI usage, and troubleshooting common GPU/Optuna issues.
- [ ] T030 Run end-to-end validation: baseline train, HPO (≥5 trials), reproducibility replay, then capture results + metric tables under `specs/.../quickstart.md`.
- [ ] T031 Prepare release checklist (open PR description) summarizing macro-F1 targets hit, MLflow experiment URIs, and next steps for deployment.

---

## Dependencies & Execution Order

- Phase 1 documents (T001–T003) are exit gates and MUST be complete before Phase 2 work begins.
- Phase 1 infrastructure tasks (T004–T006) set up configs/CLI before dataset/model work.
- Phase 2 (T007–T012) is blocking: dataset, config, and optimization wiring must be stable before running baseline or HPO tasks.
- User Story phases can proceed sequentially (US1 → US2 → US3) or partially in parallel once their prerequisites are satisfied; ensure US1 baseline is complete before US2/US3 verification.
- Tests for each phase should run before moving on to the next phase.

## Parallel Opportunities

- Tasks marked [P] manipulate independent files or modules and can run concurrently.
- Dataset config (T008) can proceed while main dataset builder (T007) is in progress if contract is well-defined.
- Evaluation pipeline (T014) can be developed in parallel with training orchestration (T013).
- Optuna config (T019) and MLflow callback (T021) can progress separately once the HPO driver scaffold exists.

## Implementation Strategy

1. Complete all Phase 1 exit-gate documents plus Hydra/MLflow scaffolding.
2. Finish Phase 2 foundational work to establish a deterministic, optimized training stack.
3. Deliver US1 baseline end-to-end (data → train → eval → MLflow artifacts) and lock in success criteria.
4. Layer US2 HPO workflow and ensure ≥2 pp macro-F1 improvement with 30-trial budget.
5. Build US3 reproducibility tooling so any MLflow run can be replayed automatically.
6. Finish documentation, optimization checklist, and release notes before handoff.

