"""DL B1 — selection mode seals the outer hold-out; finalize reopens it once.

Covers the three layers added for DL evaluation credibility:
1. _prepare_dl_data(selection) trains/validates strictly inside the outer
   training portion — the sealed hold-out rows never appear.
2. Run write-back maps DL val metrics to selection_val_* (no pre-stamped
   final_test_*), and the objective resolver picks them up.
3. _evaluate_dl_artifact replays the canonical outer split and scores the
   winner on the sealed hold-out with its own preprocessing sidecar.
"""
import numpy as np
import pandas as pd
import torch.nn as nn

from app.core.dl_trainer import BaseDLTrainer
from app.core.evaluation_metrics import resolve_objective_metrics
from app.core.model_artifact import fit_dl_preprocessing_artifact
from app.services import final_evaluation_service as fes
from app.services.dl_service import _prepare_dl_data, split_raw_holdout
from app.services.tuning_service import _normalise_run_metrics


class TinyDLTrainer(BaseDLTrainer):
    def build_model(self, input_dim: int, output_dim: int, arch_config: dict):
        return nn.Linear(input_dim, output_dim)


def _make_dataset(tmp_path, n=100):
    # row_id doubles as the single feature so transformed rows stay traceable.
    df = pd.DataFrame({
        "row_id": np.arange(n, dtype=float),
        "label": (np.arange(n) % 2).astype(int),
    })
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return path


def test_selection_mode_never_touches_sealed_holdout(tmp_path):
    path = _make_dataset(tmp_path)
    X_train, X_val, _, _, _, _ = _prepare_dl_data(
        str(path), "label", 0.2, "classification", evaluation_mode="selection"
    )
    holdout_X, _, _ = split_raw_holdout(str(path), "label", 0.2, "classification")

    seen = set(np.asarray(X_train)[:, 0]) | set(np.asarray(X_val)[:, 0])
    sealed = set(holdout_X["row_id"].to_numpy())
    assert seen.isdisjoint(sealed), "selection 训练/验证行泄漏了封存 hold-out"
    # inner split partitions the outer-train portion completely
    assert len(seen) + len(sealed) == 100


def test_standard_mode_still_validates_on_outer_holdout(tmp_path):
    path = _make_dataset(tmp_path)
    _, X_val, _, _, _, _ = _prepare_dl_data(
        str(path), "label", 0.2, "classification", evaluation_mode="standard"
    )
    holdout_X, _, _ = split_raw_holdout(str(path), "label", 0.2, "classification")
    assert set(np.asarray(X_val)[:, 0]) == set(holdout_X["row_id"].to_numpy())


def test_normalise_run_metrics_selection_maps_val_to_selection_val():
    metrics = _normalise_run_metrics(
        {"val_acc": 0.9, "val_f1_macro": 0.88, "val_loss": 0.3},
        evaluation_mode="selection",
    )
    assert metrics["selection_val_accuracy"] == 0.9
    assert metrics["selection_val_f1"] == 0.88
    assert not any(k.startswith("final_test_") for k in metrics), \
        "selection 模式不得预盖 final_test_*"


def test_normalise_run_metrics_standard_keeps_legacy_aliases():
    metrics = _normalise_run_metrics({"val_acc": 0.9}, evaluation_mode="standard")
    assert metrics["accuracy"] == 0.9
    assert metrics["final_test_accuracy"] == 0.9


def test_resolver_picks_selection_val_for_dl_runs():
    resolved = resolve_objective_metrics(
        {"selection_val_accuracy": 0.91, "val_acc": 0.91}, "accuracy"
    )
    assert resolved.selection_metric_key == "selection_val_accuracy"
    assert resolved.selection_value == 0.91
    assert resolved.final_test_value is None  # 未终评前无终评分


def test_evaluate_dl_artifact_scores_sealed_holdout(tmp_path, monkeypatch):
    path = _make_dataset(tmp_path)

    # Train-side: replicate selection-mode preprocessing and save a tiny net
    # with its sidecar, exactly as _run_dl_sync would.
    X_train, _, y_train, _, artifact, _ = _prepare_dl_data(
        str(path), "label", 0.2, "classification", evaluation_mode="selection"
    )
    trainer = TinyDLTrainer()
    trainer.model = trainer.build_model(1, 2, {})
    trainer.num_classes = 2
    model_path = tmp_path / "dl_model.pt"
    trainer.save(
        str(model_path),
        input_dim=1,
        task_type="classification",
        feature_columns=artifact.feature_names,
        preprocessing_artifact=artifact,
    )

    import app.core.dl_registry as dl_registry
    monkeypatch.setattr(dl_registry, "get_dl_trainer", lambda _mt: TinyDLTrainer())

    computed = fes._evaluate_dl_artifact(
        dataset_path=path,
        model_path=model_path,
        target_column="label",
        test_size=0.2,
        model_type="tiny_dl",
        eval_metrics=["accuracy", "f1"],
    )
    # Untrained linear net → correctness of VALUE is irrelevant; the contract
    # is metric keys/rounding matching the ML path and holdout-sized scoring.
    assert set(computed) == {"accuracy", "f1"}
    assert all(isinstance(v, float) for v in computed.values())
