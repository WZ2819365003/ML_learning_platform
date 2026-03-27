from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss
import joblib
from pathlib import Path

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

        fold_results = []
        X_full = np.vstack([X_train, X_val]) if X_val is not None and len(X_val) > 0 else X_train
        y_full = np.concatenate([y_train, y_val]) if y_val is not None and len(y_val) > 0 else y_train

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_fold_train, X_fold_val = X_full[train_idx], X_full[val_idx]
            y_fold_train, y_fold_val = y_full[train_idx], y_full[val_idx]

            # Train on fold
            self.model.fit(X_fold_train, y_fold_train)

            # Evaluate
            y_pred = self.model.predict(X_fold_val)
            y_pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_pred_proba = self.model.predict_proba(X_fold_val)
                except Exception:
                    pass

            fold_metrics = self._compute_metrics(y_fold_val, y_pred, y_pred_proba, eval_metrics)
            fold_metrics["fold"] = fold_idx + 1
            fold_results.append(fold_metrics)

            if callback:
                callback(fold_idx + 1, cv_folds, fold_metrics)

        # Final training on all data
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
                avg_metrics[f"cv_avg_{key}"] = round(np.mean(values), 4)
                avg_metrics[f"cv_std_{key}"] = round(np.std(values), 4)

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
        params = {
            "C": hyperparameters.get("C", 1.0),
            "l1_ratio": hyperparameters.get("l1_ratio", 0),
            "max_iter": hyperparameters.get("max_iter", 1000),
            "solver": hyperparameters.get("solver", "lbfgs"),
            "random_state": hyperparameters.get("random_state", 42),
        }
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

def get_trainer(model_type: str) -> BaseTrainer:
    trainer_cls = TRAINER_REGISTRY.get(model_type)
    if trainer_cls is None:
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {list(TRAINER_REGISTRY.keys())}")
    return trainer_cls()

def list_available_models() -> list[str]:
    return list(TRAINER_REGISTRY.keys())
