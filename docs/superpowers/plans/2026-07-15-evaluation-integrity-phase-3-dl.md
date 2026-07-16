# Evaluation Integrity Phase 3 DL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove split-before-transform leakage from DL training and make DL inference reuse persisted train-only preprocessing while preserving old `.pt` models.

**Architecture:** Reuse `TabularPreprocessor` inside a versioned `DLPreprocessingArtifact` that also owns the classification target encoder. Save it beside the existing network and scaler as `{model}.preprocessor.joblib`. New inference loads this sidecar and transforms raw rows directly; legacy models without it continue using `prepare_prediction_frame(training_df, ...)`.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, PyTorch, joblib, pytest.

## Global Constraints

- Keep `.pt` checkpoint and `.scaler.joblib` formats backward compatible.
- Do not change DL network architecture, optimizer behavior, metrics, prediction fields, or encoded prediction values.
- Fit categorical mappings, missing-value statistics, target encoding, and StandardScaler only from the training split.
- Unknown inference categories remain explicit validation errors.
- Preserve unrelated working-tree changes.

---

### Task 1: DL preprocessing artifact

**Files:**
- Modify: `ml_platform/app/core/model_artifact.py`
- Create: `ml_platform/tests/test_dl_preprocessing.py`

**Interfaces:**
- Produces: `DLPreprocessingArtifact`, `fit_dl_preprocessing_artifact(X, y, task_kind)`.
- Consumes: `TabularPreprocessor` and `LabelEncoder`.

- [x] Add failing tests for train-only numeric fills, classification labels mapped to contiguous zero-based ids, decoded labels, joblib round trip, and unsupported version rejection.
- [x] Implement the minimal artifact with `transform_features`, `encode_target`, and `decode_predictions`.
- [x] Run `cd ml_platform && python -m pytest tests/test_dl_preprocessing.py -q` and verify all artifact tests pass.

### Task 2: Raw DL split and train-only transforms

**Files:**
- Modify: `ml_platform/app/services/dl_service.py`
- Modify: `ml_platform/tests/test_dl_preprocessing.py`

**Interfaces:**
- Produces: `_prepare_dl_data(...) -> (X_train, X_val, y_train, y_val, preprocessing_artifact, resolved_task_type)`.
- Consumes: `prepare_raw_training_frame` and `fit_dl_preprocessing_artifact`.

- [x] Add a failing test showing `_prepare_dl_data` fits its fill value from the deterministic training rows rather than the full dataset.
- [x] Resolve `task_type="auto"` from raw targets before encoding; object/category/bool targets are classification, integer-like low-cardinality numeric targets remain classification.
- [x] Split raw rows first, fit the artifact on train only, transform train/validation, and return float32 feature matrices plus encoded targets.
- [x] Use artifact feature columns in checkpoint metadata so derived target columns cannot reappear.

### Task 3: Sidecar persistence

**Files:**
- Modify: `ml_platform/app/core/dl_trainer.py`
- Modify: `ml_platform/app/services/dl_service.py`
- Modify: `ml_platform/tests/test_dl_preprocessing.py`

**Interfaces:**
- Produces: `{path}.preprocessor.joblib`; `load_for_inference()` returns `preprocessing_artifact` or `None`.
- Consumes: `DLPreprocessingArtifact` from Task 1.

- [x] Add a failing lightweight trainer save/load test proving the sidecar survives a round trip.
- [x] Add `preprocessing_artifact` state to `BaseDLTrainer`; save/load it without changing existing scaler handling.
- [x] Include the preprocessor sidecar in object-storage artifact uploads.
- [x] Verify a missing sidecar loads as `None` for legacy checkpoints.

### Task 4: Direct and deployed inference compatibility

**Files:**
- Modify: `ml_platform/app/services/dl_service.py`
- Modify: `ml_platform/tests/test_dl_preprocessing.py`

**Interfaces:**
- Produces: `_prepare_dl_prediction_input(meta, training_df, rows, target_column)`.
- Consumes: persisted artifact when present; legacy `prepare_prediction_frame` otherwise.

- [x] Add failing tests proving new inference does not call legacy preprocessing and old metadata still does.
- [x] Route both `predict_dl_deployment` and `predict_dl_task_direct` through the shared helper.
- [x] Preserve existing encoded predictions, probability dictionaries, and feature-column response fields.

### Task 5: Verification and roadmap status

**Files:**
- Modify: `doc/优化蓝图.md`
- Modify: `docs/superpowers/plans/2026-07-15-evaluation-integrity-phase-3-dl.md`

**Interfaces:**
- Produces: verified B0 completion status.

- [x] Run DL preprocessing/prediction tests and the complete backend suite.
- [x] Run `git diff --check` and inspect the scoped diff.
- [x] Mark plan steps complete and close B0 only if both classic ML and DL paths now meet the train-only transformation contract.
