from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss
import joblib

from app.core.model_artifact import fit_tabular_artifact

# Callback type: callback(step: int, total_steps: int, metrics: dict)
MetricsCallback = Callable[[int, int, dict], None] | None

class BaseTrainer(ABC):
    def __init__(self):
        self.model = None
        self.model_type: str = ""

    @abstractmethod
    def configure(self, hyperparameters: dict):
        """Configure model with hyperparameters"""
        pass

    def train(self, X_train, y_train, X_val, y_val,
              eval_metrics: list[str] = None,
              cv_folds: int = 5,
              callback: MetricsCallback = None) -> dict:
        """
        Train the model. For sklearn models, use cross-validation folds as "epochs/steps".
        Each fold result is sent to callback for real-time progress.

        Returns final metrics dict.
        """
        eval_metrics = eval_metrics or ["accuracy"]

        # Use StratifiedKFold for cross-validation as "training steps"
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        tabular_input = isinstance(X_train, pd.DataFrame)
        base_model = self.model

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = _take_rows(X_train, train_idx), _take_rows(X_train, val_idx)
            y_fold_train, y_fold_val = _take_rows(y_train, train_idx), _take_rows(y_train, val_idx)

            # Train on fold
            fold_model = (
                fit_tabular_artifact(
                    base_model,
                    X_fold_train,
                    y_fold_train,
                    task_kind="classification",
                )
                if tabular_input
                else self.model.fit(X_fold_train, y_fold_train)
            )

            # Evaluate
            y_pred = fold_model.predict(X_fold_val)
            y_pred_proba = None
            if hasattr(fold_model, 'predict_proba'):
                try:
                    y_pred_proba = fold_model.predict_proba(X_fold_val)
                except Exception:
                    pass

            fold_metrics = self._compute_metrics(y_fold_val, y_pred, y_pred_proba, eval_metrics)
            fold_metrics["fold"] = fold_idx + 1
            fold_results.append(fold_metrics)

            if callback:
                callback(fold_idx + 1, cv_folds, fold_metrics)

        # Final training on all data
        if tabular_input:
            self.model = fit_tabular_artifact(
                base_model,
                X_train,
                y_train,
                task_kind="classification",
            )
        else:
            self.model.fit(X_train, y_train)

        # Final evaluation on validation set
        final_metrics = {}
        if X_val is not None and len(X_val) > 0:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_val_proba = self.model.predict_proba(X_val)
                except Exception:
                    pass
            final_metrics = self._compute_metrics(y_val, y_val_pred, y_val_proba, eval_metrics)

        # Average across folds
        avg_metrics = {}
        for key in fold_results[0]:
            if key == "fold":
                continue
            values = [fr[key] for fr in fold_results if fr[key] is not None]
            if values:
                mean_value = round(np.mean(values), 4)
                std_value = round(np.std(values), 4)
                avg_metrics[f"cv_avg_{key}"] = mean_value
                avg_metrics[f"cv_std_{key}"] = std_value
                avg_metrics[f"selection_cv_mean_{key}"] = mean_value
                avg_metrics[f"selection_cv_std_{key}"] = std_value

        final_metrics.update({
            f"final_test_{key}": value for key, value in final_metrics.items()
        })
        final_metrics.update(avg_metrics)
        final_metrics["cv_folds"] = fold_results

        return final_metrics

    def _compute_metrics(self, y_true, y_pred, y_pred_proba, metric_names: list[str]) -> dict:
        metrics = {}
        for name in metric_names:
            try:
                if name == "accuracy":
                    metrics[name] = round(accuracy_score(y_true, y_pred), 4)
                elif name == "f1":
                    metrics[name] = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                elif name == "precision":
                    metrics[name] = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                elif name == "recall":
                    metrics[name] = round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                elif name == "roc_auc":
                    if y_pred_proba is not None:
                        if y_pred_proba.shape[1] == 2:
                            metrics[name] = round(roc_auc_score(y_true, y_pred_proba[:, 1]), 4)
                        else:
                            metrics[name] = round(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'), 4)
                    else:
                        metrics[name] = None
                elif name == "log_loss":
                    if y_pred_proba is not None:
                        metrics[name] = round(log_loss(y_true, y_pred_proba), 4)
                    else:
                        metrics[name] = None
            except Exception:
                metrics[name] = None
        return metrics

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)


def _take_rows(values, indices):
    return values.iloc[indices] if hasattr(values, "iloc") else values[indices]


class RandomForestTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "random_forest"

    def configure(self, hyperparameters: dict):
        from sklearn.ensemble import RandomForestClassifier
        params = {
            "n_estimators": hyperparameters.get("n_estimators", 100),
            "max_depth": hyperparameters.get("max_depth", None),
            "min_samples_split": hyperparameters.get("min_samples_split", 2),
            "min_samples_leaf": hyperparameters.get("min_samples_leaf", 1),
            "random_state": hyperparameters.get("random_state", 42),
            "class_weight": hyperparameters.get("class_weight", None),
            "n_jobs": -1,
        }
        self.model = RandomForestClassifier(**params)


class XGBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "xgboost"

    def configure(self, hyperparameters: dict):
        from xgboost import XGBClassifier
        params = {
            "n_estimators": hyperparameters.get("n_estimators", 100),
            "learning_rate": hyperparameters.get("learning_rate", 0.1),
            "max_depth": hyperparameters.get("max_depth", 6),
            "subsample": hyperparameters.get("subsample", 0.8),
            "colsample_bytree": hyperparameters.get("colsample_bytree", 0.8),
            "reg_alpha": hyperparameters.get("reg_alpha", 0),
            "reg_lambda": hyperparameters.get("reg_lambda", 1),
            "scale_pos_weight": hyperparameters.get("scale_pos_weight", 1),
            "random_state": hyperparameters.get("random_state", 42),
            "eval_metric": "logloss",
        }
        self.model = XGBClassifier(**params)


class LightGBMTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "lightgbm"

    def configure(self, hyperparameters: dict):
        from lightgbm import LGBMClassifier
        params = {
            "n_estimators": hyperparameters.get("n_estimators", 100),
            "learning_rate": hyperparameters.get("learning_rate", 0.1),
            "num_leaves": hyperparameters.get("num_leaves", 31),
            "max_depth": hyperparameters.get("max_depth", -1),
            "min_child_samples": hyperparameters.get("min_child_samples", 20),
            "colsample_bytree": hyperparameters.get("colsample_bytree", 1.0),
            "reg_alpha": hyperparameters.get("reg_alpha", 0),
            "reg_lambda": hyperparameters.get("reg_lambda", 1),
            "is_unbalance": hyperparameters.get("is_unbalance", False),
            "random_state": hyperparameters.get("random_state", 42),
            "verbose": -1,
        }
        self.model = LGBMClassifier(**params)


class LogisticRegressionTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "logistic_regression"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import LogisticRegression
        penalty = hyperparameters.get("penalty", "l2")
        params = {
            "C": hyperparameters.get("C", 1.0),
            "penalty": penalty,
            "max_iter": hyperparameters.get("max_iter", 1000),
            "solver": hyperparameters.get("solver", "lbfgs"),
            "class_weight": hyperparameters.get("class_weight", None),
            "random_state": hyperparameters.get("random_state", 42),
        }
        # `l1_ratio` is only valid when penalty='elasticnet'; passing it under
        # other penalties triggers sklearn's UserWarning on every fit. Only
        # forward the user-supplied value when it actually applies.
        if penalty == "elasticnet":
            params["l1_ratio"] = hyperparameters.get("l1_ratio", 0.5)
        self.model = LogisticRegression(**params)


class SVMTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "svm"

    def configure(self, hyperparameters: dict):
        from sklearn.svm import SVC
        params = {
            "C": hyperparameters.get("C", 1.0),
            "kernel": hyperparameters.get("kernel", "rbf"),
            "gamma": hyperparameters.get("gamma", "scale"),
            "class_weight": hyperparameters.get("class_weight", None),
            "probability": True,  # needed for predict_proba and roc_auc
            "random_state": hyperparameters.get("random_state", 42),
        }
        self.model = SVC(**params)


class MLPTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "mlp"

    def configure(self, hyperparameters: dict):
        from sklearn.neural_network import MLPClassifier
        hidden_layers = hyperparameters.get("hidden_layer_sizes", [100, 50])
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(hidden_layers)
        params = {
            "hidden_layer_sizes": hidden_layers,
            "activation": hyperparameters.get("activation", "relu"),
            "learning_rate": hyperparameters.get("learning_rate", "adaptive"),
            "learning_rate_init": hyperparameters.get("learning_rate_init", 0.001),
            "max_iter": hyperparameters.get("max_iter", 500),
            "alpha": hyperparameters.get("alpha", 0.0001),
            "early_stopping": hyperparameters.get("early_stopping", False),
            "random_state": hyperparameters.get("random_state", 42),
        }
        self.model = MLPClassifier(**params)


# Model factory
TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "random_forest": RandomForestTrainer,
    "xgboost": XGBoostTrainer,
    "lightgbm": LightGBMTrainer,
    "logistic_regression": LogisticRegressionTrainer,
    "svm": SVMTrainer,
    "mlp": MLPTrainer,
}


def detect_task_type(model_type: str) -> str:
    """Return 'regression' or 'classification' based on model_type string."""
    from app.core.regression_trainers import REGRESSION_TRAINER_REGISTRY
    return "regression" if model_type in REGRESSION_TRAINER_REGISTRY else "classification"


def get_trainer(model_type: str) -> BaseTrainer:
    from app.core.regression_trainers import REGRESSION_TRAINER_REGISTRY
    combined = {**TRAINER_REGISTRY, **REGRESSION_TRAINER_REGISTRY}
    trainer_cls = combined.get(model_type)
    if trainer_cls is None:
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {list(combined.keys())}")
    return trainer_cls()


def list_available_models() -> list[str]:
    from app.core.regression_trainers import REGRESSION_TRAINER_REGISTRY
    return list(TRAINER_REGISTRY.keys()) + list(REGRESSION_TRAINER_REGISTRY.keys())
