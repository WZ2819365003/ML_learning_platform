"""Regression trainers using KFold CV and regression metrics (MSE/RMSE/MAE/R²/MAPE)."""
from typing import Any, Callable
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.core.trainer import BaseTrainer

MetricsCallback = Callable[[int, int, dict], None] | None


class RegressionMixin:
    """
    Mixin that overrides BaseTrainer.train() with KFold (not StratifiedKFold)
    and computes regression metrics (mse, rmse, mae, r2, mape).

    Multiple-inheritance usage:
        class FooRegressor(RegressionMixin, BaseTrainer): ...
    MRO ensures RegressionMixin.train() is found before BaseTrainer.train().
    """

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        eval_metrics: list[str] = None,
        cv_folds: int = 5,
        callback: MetricsCallback = None,
    ) -> dict:
        eval_metrics = eval_metrics or ["rmse", "mae", "r2"]

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        X_full = np.vstack([X_train, X_val]) if X_val is not None and len(X_val) > 0 else X_train
        y_full = np.concatenate([y_train, y_val]) if y_val is not None and len(y_val) > 0 else y_train

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            X_f_tr, X_f_val = X_full[train_idx], X_full[val_idx]
            y_f_tr, y_f_val = y_full[train_idx], y_full[val_idx]

            self.model.fit(X_f_tr, y_f_tr)
            y_pred = self.model.predict(X_f_val)

            fold_metrics = self._compute_regression_metrics(y_f_val, y_pred, eval_metrics)
            fold_metrics["fold"] = fold_idx + 1
            fold_results.append(fold_metrics)

            if callback:
                callback(fold_idx + 1, cv_folds, fold_metrics)

        # Final fit on training split only
        self.model.fit(X_train, y_train)

        # Final eval on validation set
        final_metrics = {}
        if X_val is not None and len(X_val) > 0:
            y_val_pred = self.model.predict(X_val)
            final_metrics = self._compute_regression_metrics(y_val, y_val_pred, eval_metrics)

        # CV averages
        avg_metrics: dict[str, Any] = {}
        metric_keys = [k for k in fold_results[0] if k != "fold"]
        for key in metric_keys:
            values = [fr[key] for fr in fold_results if fr[key] is not None]
            if values:
                avg_metrics[f"cv_avg_{key}"] = round(float(np.mean(values)), 4)
                avg_metrics[f"cv_std_{key}"] = round(float(np.std(values)), 4)

        final_metrics.update(avg_metrics)
        final_metrics["cv_folds"] = fold_results
        return final_metrics

    @staticmethod
    def _compute_regression_metrics(y_true, y_pred, metric_names: list[str]) -> dict:
        metrics: dict[str, Any] = {}
        for name in metric_names:
            try:
                if name == "mse":
                    metrics[name] = round(float(mean_squared_error(y_true, y_pred)), 4)
                elif name == "rmse":
                    metrics[name] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
                elif name == "mae":
                    metrics[name] = round(float(mean_absolute_error(y_true, y_pred)), 4)
                elif name == "r2":
                    metrics[name] = round(float(r2_score(y_true, y_pred)), 4)
                elif name == "mape":
                    # Avoid division by zero
                    mask = y_true != 0
                    if mask.sum() > 0:
                        metrics[name] = round(
                            float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 4
                        )
                    else:
                        metrics[name] = None
            except Exception:
                metrics[name] = None
        return metrics


# ---------------------------------------------------------------------------
# Concrete regression trainers
# ---------------------------------------------------------------------------

class RandomForestRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "random_forest_regressor"

    def configure(self, hyperparameters: dict):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            max_depth=hyperparameters.get("max_depth", None),
            min_samples_split=hyperparameters.get("min_samples_split", 2),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", 1),
            random_state=hyperparameters.get("random_state", 42),
            n_jobs=-1,
        )


class XGBoostRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "xgboost_regressor"

    def configure(self, hyperparameters: dict):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            max_depth=hyperparameters.get("max_depth", 6),
            subsample=hyperparameters.get("subsample", 0.8),
            colsample_bytree=hyperparameters.get("colsample_bytree", 0.8),
            random_state=hyperparameters.get("random_state", 42),
        )


class LightGBMRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "lightgbm_regressor"

    def configure(self, hyperparameters: dict):
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            num_leaves=hyperparameters.get("num_leaves", 31),
            max_depth=hyperparameters.get("max_depth", -1),
            random_state=hyperparameters.get("random_state", 42),
            verbose=-1,
        )


class LinearRegressionTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "linear_regression"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(
            fit_intercept=hyperparameters.get("fit_intercept", True),
        )


class RidgeTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "ridge"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import Ridge
        self.model = Ridge(
            alpha=hyperparameters.get("alpha", 1.0),
            max_iter=hyperparameters.get("max_iter", 2000),
        )


class LassoTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "lasso"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import Lasso
        self.model = Lasso(
            alpha=hyperparameters.get("alpha", 1.0),
            max_iter=hyperparameters.get("max_iter", 2000),
        )


class ElasticNetTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "elasticnet"

    def configure(self, hyperparameters: dict):
        from sklearn.linear_model import ElasticNet
        self.model = ElasticNet(
            alpha=hyperparameters.get("alpha", 1.0),
            l1_ratio=hyperparameters.get("l1_ratio", 0.5),
            max_iter=hyperparameters.get("max_iter", 2000),
        )


class SVRTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "svr"

    def configure(self, hyperparameters: dict):
        from sklearn.svm import SVR
        self.model = SVR(
            C=hyperparameters.get("C", 1.0),
            kernel=hyperparameters.get("kernel", "rbf"),
            gamma=hyperparameters.get("gamma", "scale"),
            epsilon=hyperparameters.get("epsilon", 0.1),
        )


class MLPRegressorTrainer(RegressionMixin, BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "mlp_regressor"

    def configure(self, hyperparameters: dict):
        from sklearn.neural_network import MLPRegressor
        hidden_layers = hyperparameters.get("hidden_layer_sizes", [100, 50])
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(hidden_layers)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=hyperparameters.get("activation", "relu"),
            learning_rate=hyperparameters.get("learning_rate", "adaptive"),
            learning_rate_init=hyperparameters.get("learning_rate_init", 0.001),
            max_iter=hyperparameters.get("max_iter", 500),
            random_state=hyperparameters.get("random_state", 42),
        )


REGRESSION_TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "random_forest_regressor": RandomForestRegressorTrainer,
    "xgboost_regressor": XGBoostRegressorTrainer,
    "lightgbm_regressor": LightGBMRegressorTrainer,
    "linear_regression": LinearRegressionTrainer,
    "ridge": RidgeTrainer,
    "lasso": LassoTrainer,
    "elasticnet": ElasticNetTrainer,
    "svr": SVRTrainer,
    "mlp_regressor": MLPRegressorTrainer,
}
