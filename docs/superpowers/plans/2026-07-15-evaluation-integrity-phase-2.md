# Evaluation Integrity Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make classic ML feature transformations fold-local and persist the fitted preprocessing state with the estimator while preserving legacy joblib prediction, deployment, visualization, and SHAP behavior.

**Architecture:** Add a versioned `TabularModelArtifact` containing a strict fitted feature transformer, estimator, and optional target encoder. Production ML training splits raw rows first, fits a separate artifact inside every CV fold, then saves the final train-only artifact. Consumers branch on the artifact type; legacy plain estimators retain their existing data-preparation path.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, joblib, pytest.

## Global Constraints

- Classic ML only; do not change DL `.pt` or scaler artifact formats in this phase.
- Unknown categorical values remain a validation error, matching the current prediction contract.
- New artifacts are self-contained and must not refit preprocessing during inference.
- Plain legacy estimator joblib files remain readable and keep their existing behavior.
- Do not change public HTTP response fields or existing metric keys.
- Preserve unrelated working-tree changes, including `ml_platform/requirements.txt`.

---

### Task 1: Versioned tabular artifact

**Files:**
- Create: `ml_platform/app/core/model_artifact.py`
- Create: `ml_platform/tests/test_model_artifact.py`

**Interfaces:**
- Produces: `TabularPreprocessor.fit/transform`, `TabularModelArtifact.predict/predict_proba`, `fit_tabular_artifact`, `is_tabular_artifact`.
- Consumes: pandas DataFrames and sklearn-compatible estimators.

- [x] Write failing tests proving numeric fill values and categorical mappings come only from the supplied fit frame, unknown categories fail, target labels decode, and a joblib round trip preserves predictions.
- [x] Run `cd ml_platform && python -m pytest tests/test_model_artifact.py -q` and verify failure because the module does not exist.
- [x] Implement the minimal versioned artifact. `TabularPreprocessor.transform` must validate required columns, preserve fit-time column order, coerce numeric values, and reject unknown categories.
- [x] Run the artifact tests and verify they pass.

### Task 2: Raw split and fold-local classic ML training

**Files:**
- Modify: `ml_platform/app/services/prediction_service.py`
- Modify: `ml_platform/app/services/training_service.py`
- Modify: `ml_platform/app/core/trainer.py`
- Modify: `ml_platform/app/core/regression_trainers.py`
- Modify: `ml_platform/tests/test_training_service_prepare_data.py`
- Modify: `ml_platform/tests/test_evaluation_integrity.py`

**Interfaces:**
- Produces: `prepare_raw_training_frame(df, target_column)` and DataFrame/Series outputs from `_prepare_data`.
- Consumes: `fit_tabular_artifact` for each CV fold and final fit.

- [x] Add failing tests showing `_prepare_data` retains raw categorical values/missing values and a validation-only extreme cannot alter the final artifact's fill state.
- [x] Change `_prepare_data` to remove target/derived-target columns and split raw rows before any fitted transform.
- [x] In classification and regression trainers, use a fresh `fit_tabular_artifact` per CV fold when inputs are DataFrames; keep NumPy compatibility for direct legacy callers.
- [x] Fit and assign one final artifact on all outer training rows, evaluate it on outer validation rows, and save it through the existing `trainer.save` path.
- [x] Verify trainer, preparation, model, and artifact tests.

### Task 3: Prediction and deployment compatibility

**Files:**
- Modify: `ml_platform/app/services/prediction_service.py`
- Modify: `ml_platform/app/services/deploy_service.py`
- Modify: `ml_platform/tests/test_prediction_service.py`
- Modify: `ml_platform/tests/test_deploy_service.py`

**Interfaces:**
- Produces: one `predict_with_model(model, training_df, rows, target_column, include_probabilities)` adapter returning decoded predictions, class labels, and optional probabilities.
- Consumes: self-contained artifacts or legacy estimators.

- [x] Add failing tests proving new artifacts predict without refitting from `training_df`, while legacy estimators still use `prepare_prediction_frame`.
- [x] Implement the shared adapter and route both `predict_rows` and `run_inference` through it.
- [x] Verify unknown/missing feature errors and probability labels remain stable.

### Task 4: Resolver, visualization, and SHAP compatibility

**Files:**
- Modify: `ml_platform/app/services/resolver.py`
- Modify: `ml_platform/app/services/shap_service.py`
- Modify: `ml_platform/tests/v3/test_shap_service.py`

**Interfaces:**
- Produces: artifact-aware evaluation data preparation that returns transformed matrices and the underlying estimator to explainers.
- Consumes: `TabularModelArtifact.transform_features`, `encode_target`, and `estimator`.

- [x] Add an artifact-backed SHAP integration test with categorical input.
- [x] Load the model before preparing evaluation data; for new artifacts, split raw rows with the same random seed, transform using persisted state, and unwrap the estimator for Tree/Kernel/permutation explainers.
- [x] Keep the legacy resolver and SHAP path unchanged for plain estimator files.
- [x] Run visualization and SHAP tests.

### Task 5: Verification and roadmap status

**Files:**
- Modify: `doc/优化蓝图.md`
- Modify: `docs/superpowers/plans/2026-07-15-evaluation-integrity-phase-2.md`

**Interfaces:**
- Produces: truthful B0 progress and a separate DL follow-up.

- [x] Run `cd ml_platform && python -m pytest tests/ -q`.
- [x] Run `git diff --check` and inspect the scoped diff.
- [x] Mark completed checkboxes and update B0 progress without claiming DL preprocessing is fixed.
