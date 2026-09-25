# Evaluation Integrity Phase 4 Selection Metrics Plan

**Goal:** Stop Bayesian tuning, leaderboards, strategy comparison, and automatic Top-K actions from ranking ML trials by the repeatedly observed hold-out score.

**Architecture:** Add one backend metric-semantics helper that resolves an objective into a selection value and a final-test value. Trainers emit explicit `selection_cv_mean_*`, `selection_cv_std_*`, and `final_test_*` aliases while retaining legacy `cv_avg_*`, `cv_std_*`, and raw metric keys. All model-selection consumers use the shared resolver. Existing API `objective_value` remains as a compatibility alias for the selection value; responses add explicit provenance fields.

**Non-goal:** This phase does not yet implement a physically sealed hold-out evaluated only once after candidate selection. It removes hold-out values from automated selection and establishes the contract needed for that follow-up.

### Task 1: Metric semantics contract

- [x] Add failing unit tests for canonical aliases, legacy fallback, missing values, and min/max-independent resolution.
- [x] Implement the shared resolver without database or frontend dependencies.

### Task 2: Trainer output aliases

- [x] Add failing classification and regression tests for `selection_cv_mean_*`, `selection_cv_std_*`, and `final_test_*`.
- [x] Emit aliases without removing existing metric keys.
- [x] Add DL final-test aliases during V3 metric normalization; DL tuning remains out of scope.

### Task 3: Selection consumers

- [x] Make Bayesian Optuna feedback use the shared selection resolver.
- [x] Make task leaderboard, task runs, global runs, strategy comparison, task summary, and SHAP Top-K use the same selection resolver.
- [x] Preserve `objective_value` as the selection-value compatibility alias and add explicit metric/value fields where run DTOs are built.

### Task 4: Verification and roadmap

- [x] Run targeted tuning/modeling/trainer tests and the complete backend suite.
- [x] Run `git diff --check` and inspect the scoped diff.
- [x] Mark B1 as in progress and document that sealed winner-only evaluation remains.

**Verification:** backend `262 passed`; frontend unit `6 passed`; production build passed. Scoped frontend lint passed. Full frontend lint remains blocked by pre-existing repository errors outside this phase.
