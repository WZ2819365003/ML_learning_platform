# Evaluation Integrity Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the outer validation split from participating in classification or regression cross-validation while preserving existing metric keys and model artifact behavior.

**Architecture:** Keep the current trainer API unchanged. Both trainer implementations run CV exclusively over `X_train/y_train`, then fit once on all training rows and evaluate once on `X_val/y_val`. Persisted models and response fields remain backward compatible; fold-local preprocessing and sealed final-test semantics are delivered in later B0/B1 phases.

**Tech Stack:** Python 3, NumPy, scikit-learn, pytest.

## Global Constraints

- Do not change the public `BaseTrainer.train(...)` or `RegressionMixin.train(...)` signatures.
- Keep existing raw final metric keys, `cv_avg_*`, `cv_std_*`, and `cv_folds` in this phase.
- Do not change joblib artifact structure in this phase.
- Add regression tests before production changes and verify that they fail because validation rows enter CV.
- Preserve all unrelated working-tree changes.

---

### Task 1: Lock the train-only CV contract with failing tests

**Files:**
- Create: `ml_platform/tests/test_evaluation_integrity.py`

**Interfaces:**
- Consumes: `BaseTrainer.train(...)` and `RegressionMixin.train(...)`.
- Produces: Regression tests asserting fold fit sizes are derived only from the training split.

- [x] **Step 1: Add recording estimators and classification/regression tests**

```python
class RecordingEstimator:
    def __init__(self):
        self.fit_sizes = []

    def fit(self, X, y):
        self.fit_sizes.append(len(X))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_classification_cv_never_fits_validation_rows():
    trainer = RecordingClassifierTrainer()
    trainer.configure({})
    trainer.train(X_train, y_train, X_val, y_val, eval_metrics=["accuracy"], cv_folds=2)
    assert trainer.model.fit_sizes == [4, 4, 8]


def test_regression_cv_never_fits_validation_rows():
    trainer = RecordingRegressionTrainer()
    trainer.configure({})
    trainer.train(X_train, y_train, X_val, y_val, eval_metrics=["rmse"], cv_folds=2)
    assert trainer.model.fit_sizes == [4, 4, 8]
```

- [x] **Step 2: Run the new tests and verify RED**

Run: `cd ml_platform && python -m pytest tests/test_evaluation_integrity.py -q`

Expected: both tests fail with observed fit sizes `[5, 5, 8]`, proving the outer validation rows currently enter the two CV loops.

### Task 2: Restrict classification and regression CV to training rows

**Files:**
- Modify: `ml_platform/app/core/trainer.py:34`
- Modify: `ml_platform/app/core/regression_trainers.py:34`
- Test: `ml_platform/tests/test_evaluation_integrity.py`

**Interfaces:**
- Consumes: unchanged `X_train/y_train/X_val/y_val` arguments.
- Produces: CV metrics from training-only folds and raw final metrics from validation-only evaluation.

- [x] **Step 1: Change classification fold generation**

Replace the combined `X_full/y_full` inputs with the training inputs:

```python
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
```

- [x] **Step 2: Change regression fold generation**

```python
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_f_tr, X_f_val = X_train[train_idx], X_train[val_idx]
    y_f_tr, y_f_val = y_train[train_idx], y_train[val_idx]
```

- [x] **Step 3: Run the new tests and verify GREEN**

Run: `cd ml_platform && python -m pytest tests/test_evaluation_integrity.py -q`

Expected: `2 passed`.

- [x] **Step 4: Run focused compatibility tests**

Run: `cd ml_platform && python -m pytest tests/test_models.py tests/test_prediction_service.py tests/v3/test_run_diagnosis.py -q`

Expected: all tests pass; existing metric keys remain available.

- [x] **Step 5: Run the full backend suite**

Run: `cd ml_platform && python -m pytest tests/ -q`

Expected: all backend tests pass. Pre-existing environment warnings may remain, but no test failures are accepted.

### Task 3: Review the phase boundary

**Files:**
- Review: `doc/优化蓝图.md`
- Review: `docs/superpowers/plans/2026-07-15-evaluation-integrity-phase-1.md`

**Interfaces:**
- Consumes: verified phase-1 behavior.
- Produces: A clear handoff to B0 phase 2 without overstating leakage protection.

- [x] **Step 1: Confirm this phase does not claim fold-local preprocessing**

Verify the implementation only guarantees that outer validation rows do not enter trainer CV. Feature encoding/imputation is still currently fit before the split and remains explicitly scheduled for B0 phase 2.

- [x] **Step 2: Inspect the final diff**

Run: `git diff --check && git diff -- doc/优化蓝图.md docs/superpowers/plans/2026-07-15-evaluation-integrity-phase-1.md ml_platform/app/core/trainer.py ml_platform/app/core/regression_trainers.py ml_platform/tests/test_evaluation_integrity.py`

Expected: no whitespace errors and no unrelated file changes in the scoped diff.
