# Experiment Tracking & Reproducibility Checklist: DeBERTaV3 Criteria Matching Classifier

**Purpose**: Peer-review checklist ensuring requirements for MLflow logging, Hydra config lineage, Optuna HPO, and non-functional gates (latency/retries/lineage) are complete and testable before approval.
**Created**: 2025-11-12
**Feature**: specs/001-debertav3-criteria-classifier/spec.md

## Experiment Tracking & Logging

- [ ] CHK001 Does FR-005 clearly enumerate every parameter, metric (accuracy, macro-F1, weighted-F1), and artifact (model weights, config snapshot, evaluation report) that each MLflow run must log, leaving no ambiguity about the minimum set? [Completeness][Spec FR-005]
- [ ] CHK002 Are the mandated tracking URI (`sqlite:///mlflow.db`) and artifact store (`./mlruns/`) captured in both the spec and Hydra config plan so reviewers can verify them without inspecting code? [Consistency][Spec FR-006][Plan §Technical Context]
- [ ] CHK003 Do the acceptance scenarios for US1 explicitly require confusion matrix and classification report uploads, matching the plan’s constitution gate for artifacts? [Coverage][Spec US1][Plan §Constitution Check]
- [ ] CHK004 Is there a documented requirement describing how failed MLflow logging should be retried or reported (e.g., retry budget, alert), or is this a missing non-functional expectation? [Gap][Reliability]

## Reproducibility & Lineage

- [ ] CHK005 Do the requirements state that resolved Hydra configs must be versioned (saved alongside run artifacts) and linked to MLflow run IDs so a reviewer can trace configuration lineage? [Completeness][Plan §Constitution Check]
- [ ] CHK006 Is deterministic seeding (Python/NumPy/Torch/CUDA) documented with enough precision (exact config keys, default values) for peers to confirm reproducibility claims? [Clarity][Spec FR-009][Plan §Technical Context]
- [ ] CHK007 Does US3 (and SC-003) describe the expected tolerance (±0.5 pp macro-F1) and the procedure for replaying a run, including how to obtain the original dataset split and config? [Completeness][Spec US3][Spec SC-003]
- [ ] CHK008 Are git SHA, dataset hash, and optimization-flag recording requirements documented (not just implied by tasks T023/T024), or should a new requirement be added to cover lineage metadata explicitly? [Gap][Plan §Phase 5][Tasks T023]

## HPO & Config Search Controls

- [ ] CHK009 Is the Optuna objective (validation macro-F1) and success criteria (≥2 pp gain, ≥30 trials) spelled out in both FR-007 and SC-004 so reviewers can trace the HPO gate? [Consistency][Spec FR-007][Spec SC-004]
- [ ] CHK010 Do the clarifications fully describe the search space (head layers, hidden dims, pooling strategies, LR, batch sizes, dropout, warmup, AMP toggles) so missing ranges can be spotted during review? [Completeness][Clarifications §HPO]
- [ ] CHK011 Are pruner/sampler expectations (MedianPruner + TPE) recorded in requirements (not only plan prose) so a reviewer can confirm compliance without reading implementation? [Gap][Plan §Constitution Check]
- [ ] CHK012 Does the spec or plan explain how Optuna + MLflow runs are correlated (one MLflow run per trial, best-trial export), including failure/retry behavior when a trial aborts? [Gap][Plan §Phase 4]

## Non-Functional Gates (Performance, Resilience, Lineage)

- [ ] CHK013 Do the requirements specify any latency/throughput targets or training-duration expectations that justify the mandated optimization list (FR-010), or is a measurable performance goal missing? [Gap][Spec FR-010]
- [ ] CHK014 Are data-pipeline dry-run and truncation safeguards (FR-008, FR-013) sufficient to cover failure scenarios such as tokenizer overflows or invalid pairs, or should additional retry/backoff behavior be captured? [Coverage][Spec FR-008][Spec FR-013]
- [ ] CHK015 Does the quickstart/plan describe operational runbooks for MLflow outages or Optuna study restarts (resume from RDB, clean retries), ensuring reviewers can verify resilience requirements? [Gap][Plan §Dependencies & Risks]

## Notes

- Check items off only when the underlying requirement text (spec/plan) satisfies the question.
- Use `[Gap]` items to track missing requirements; resolve them before implementation proceeds.
- Attach references (Spec/Plan/Tasks sections) in PR descriptions to show evidence for each checked item.
