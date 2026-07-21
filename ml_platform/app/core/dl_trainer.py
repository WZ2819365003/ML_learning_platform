"""Base deep-learning trainer (PyTorch).

All DL trainers inherit from BaseDLTrainer and implement build_model().
The base class handles:
  - Feature normalisation (StandardScaler)
  - DataLoader construction
  - Optimizer + LR scheduler creation
  - Full train/eval loop with early stopping + grad clipping
  - Per-epoch callback for real-time WebSocket streaming
  - save() / load() (state-dict .pt + scaler .joblib)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve as sk_roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from app.core.model_artifact import DLPreprocessingArtifact

logger = logging.getLogger(__name__)

EpochCallback = Callable[[int, dict], None] | None


class BaseDLTrainer(ABC):
    """
    Abstract base for all PyTorch DL trainers.

    Subclass contract: implement build_model(input_dim, output_dim, arch_config).
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.scaler: StandardScaler | None = None
        self.model_type: str = ""
        self.num_classes: int = 1
        self.preprocessing_artifact: DLPreprocessingArtifact | None = None

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def build_model(self, input_dim: int, output_dim: int, arch_config: dict) -> nn.Module:
        """Instantiate and return the nn.Module (not yet moved to device)."""

    # ── Optimizer factory ─────────────────────────────────────────────────────

    def build_optimizer(self, model: nn.Module, opt_config: dict) -> torch.optim.Optimizer:
        name = str(opt_config.get("optimizer", "adam")).lower()
        lr   = float(opt_config.get("learning_rate", 1e-3))
        wd   = float(opt_config.get("weight_decay", 0.0))
        b1   = float(opt_config.get("beta1", 0.9))
        b2   = float(opt_config.get("beta2", 0.999))
        mom  = float(opt_config.get("momentum", 0.9))

        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
        if name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        raise ValueError(f"Unknown optimizer '{name}'. Choose: adam / adamw / sgd / rmsprop")

    # ── Scheduler factory ─────────────────────────────────────────────────────

    def build_scheduler(self, optimizer, train_config: dict):
        name   = str(train_config.get("scheduler", "none")).lower()
        epochs = int(train_config.get("epochs", 50))
        if name == "step":
            step = max(1, epochs // 5)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
        return None

    # ── Data pipeline ─────────────────────────────────────────────────────────

    def _make_loaders(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
        batch_size: int, task_type: str,
    ) -> tuple[DataLoader, DataLoader]:
        self.scaler = StandardScaler()
        X_tr = self.scaler.fit_transform(X_train.astype(np.float32))
        X_va = self.scaler.transform(X_val.astype(np.float32))

        Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
        Xva_t = torch.tensor(X_va, dtype=torch.float32)

        if task_type == "regression":
            ytr_t = torch.tensor(y_train.astype(np.float32), dtype=torch.float32).unsqueeze(1)
            yva_t = torch.tensor(y_val.astype(np.float32),   dtype=torch.float32).unsqueeze(1)
        else:
            ytr_t = torch.tensor(y_train.astype(np.int64), dtype=torch.long)
            yva_t = torch.tensor(y_val.astype(np.int64),   dtype=torch.long)

        # BatchNorm cannot train on a final batch containing one sample. Drop
        # only that pathological tail; keep all other partial batches.
        drop_single_tail = len(Xtr_t) > batch_size and len(Xtr_t) % batch_size == 1
        train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                                  batch_size=batch_size, shuffle=True, drop_last=drop_single_tail)
        val_loader   = DataLoader(TensorDataset(Xva_t, yva_t),
                                  batch_size=batch_size * 2, shuffle=False)
        return train_loader, val_loader

    def _make_criterion(self, task_type: str) -> nn.Module:
        if task_type == "regression":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        arch_config:  dict,
        opt_config:   dict,
        train_config: dict,
        task_type:    str = "classification",
        epoch_callback: EpochCallback = None,
    ) -> dict:
        epochs   = int(train_config.get("epochs", 50))
        bs       = int(train_config.get("batch_size", 32))
        patience = int(train_config.get("early_stopping_patience", 10))
        grad_clip = float(train_config.get("grad_clip", 1.0))

        input_dim  = X_train.shape[1]
        if task_type == "regression":
            output_dim = 1
            self.num_classes = 1
        else:
            output_dim = int(len(np.unique(y_train)))
            self.num_classes = output_dim

        logger.info("DL train | device=%s input=%d output=%d task=%s",
                    self.device, input_dim, output_dim, task_type)

        self.model = self.build_model(input_dim, output_dim, arch_config).to(self.device)
        optimizer  = self.build_optimizer(self.model, opt_config)
        scheduler  = self.build_scheduler(optimizer, train_config)
        criterion  = self._make_criterion(task_type).to(self.device)

        train_loader, val_loader = self._make_loaders(
            X_train, y_train, X_val, y_val, bs, task_type)

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        history: list[dict] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(optimizer, criterion, train_loader, grad_clip)
            val_metrics = self._eval_epoch(criterion, val_loader, task_type)
            val_loss = val_metrics["val_loss"]
            current_lr = optimizer.param_groups[0]["lr"]

            # scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # early stopping / best-state tracking
            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            ep_data: dict = {
                "epoch": epoch,
                "total": epochs,
                "train_loss": round(float(train_loss), 6),
                "lr": round(current_lr, 8),
                **{k: round(float(v), 6) for k, v in val_metrics.items()},
            }
            history.append(ep_data)

            if epoch_callback:
                epoch_callback(epoch, ep_data)

            if no_improve >= patience:
                logger.info("Early stopping at epoch %d/%d", epoch, epochs)
                break

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        final = self._eval_final(criterion, val_loader, task_type)
        result: dict = {
            "best_val_loss": round(float(best_val_loss), 6),
            "final_epoch":   len(history),
            "history":       history,
        }
        for k, v in final.items():
            if k == "confusion_matrix":
                result[k] = v           # list-of-lists — keep as-is
            elif v is not None:
                try:
                    result[k] = round(float(v), 6)
                except (TypeError, ValueError):
                    result[k] = v
        return result

    # ── Epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(self, optimizer, criterion, loader: DataLoader, grad_clip: float) -> float:
        self.model.train()
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            out = self.model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * len(Xb)
        return total / max(1, len(loader.dataset))

    def _eval_epoch(self, criterion, loader: DataLoader, task_type: str) -> dict:
        """Lightweight per-epoch evaluation: loss + primary metric + F1 (classification)."""
        self.model.eval()
        total = 0.0
        preds_list, labels_list = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                out  = self.model(Xb)
                loss = criterion(out, yb)
                total += loss.item() * len(Xb)
                if task_type == "classification":
                    preds_list.append(out.argmax(1).cpu().numpy())
                    labels_list.append(yb.cpu().numpy())
                else:
                    preds_list.append(out.cpu().numpy().flatten())
                    labels_list.append(yb.cpu().numpy().flatten())

        avg_loss = total / max(1, len(loader.dataset))
        preds  = np.concatenate(preds_list)
        labels = np.concatenate(labels_list)

        if task_type == "classification":
            acc = float(np.mean(preds == labels))
            f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
            return {"val_loss": avg_loss, "val_acc": acc, "val_f1_macro": f1}
        else:
            rmse = float(mean_squared_error(labels, preds) ** 0.5)
            mae  = float(mean_absolute_error(labels, preds))
            r2   = float(r2_score(labels, preds))
            return {"val_loss": avg_loss, "val_rmse": rmse, "val_mae": mae, "val_r2": r2}

    def _eval_final(self, criterion, loader: DataLoader, task_type: str) -> dict:
        """Comprehensive post-training evaluation with confusion matrix, AUC-ROC, etc."""
        self.model.eval()
        total = 0.0
        preds_list, labels_list, proba_list = [], [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                out  = self.model(Xb)
                loss = criterion(out, yb)
                total += loss.item() * len(Xb)
                if task_type == "classification":
                    proba_list.append(torch.softmax(out, dim=1).cpu().numpy())
                    preds_list.append(out.argmax(1).cpu().numpy())
                    labels_list.append(yb.cpu().numpy())
                else:
                    preds_list.append(out.cpu().numpy().flatten())
                    labels_list.append(yb.cpu().numpy().flatten())

        avg_loss = total / max(1, len(loader.dataset))
        preds  = np.concatenate(preds_list)
        labels = np.concatenate(labels_list)

        if task_type == "classification":
            acc          = float(np.mean(preds == labels))
            f1_macro     = float(f1_score(labels, preds, average="macro",    zero_division=0))
            f1_weighted  = float(f1_score(labels, preds, average="weighted", zero_division=0))
            precision    = float(precision_score(labels, preds, average="macro", zero_division=0))
            recall       = float(recall_score(labels, preds,    average="macro", zero_division=0))
            cm           = sk_confusion_matrix(labels, preds).tolist()

            result: dict = {
                "val_loss":        avg_loss,
                "val_acc":         acc,
                "val_f1_macro":    f1_macro,
                "val_f1_weighted": f1_weighted,
                "val_precision":   precision,
                "val_recall":      recall,
                "confusion_matrix": cm,
            }

            # AUC-ROC + ROC curve data (binary classification only)
            if proba_list:
                probas = np.concatenate(proba_list)
                try:
                    if self.num_classes == 2:
                        result["val_auc_roc"] = float(roc_auc_score(labels, probas[:, 1]))
                        # Downsample ROC curve to ≤200 points for storage
                        fpr, tpr, _ = sk_roc_curve(labels, probas[:, 1])
                        n_pts = min(len(fpr), 200)
                        idx = np.linspace(0, len(fpr) - 1, n_pts, dtype=int)
                        result["val_roc_fpr"] = [round(float(v), 6) for v in fpr[idx]]
                        result["val_roc_tpr"] = [round(float(v), 6) for v in tpr[idx]]
                    else:
                        result["val_auc_roc"] = float(
                            roc_auc_score(labels, probas, multi_class="ovr", average="macro")
                        )
                except Exception as e:
                    logger.warning("AUC-ROC computation skipped: %s", e)

                # Prediction confidence distribution (max-probability per sample)
                try:
                    conf = probas.max(axis=1)
                    sample_n = min(len(conf), 500)
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(conf), sample_n, replace=False)
                    result["val_confidence_dist"] = [round(float(v), 6) for v in conf[idx]]
                except Exception as e:
                    logger.warning("Confidence distribution skipped: %s", e)
            return result

        else:  # regression
            rmse = float(mean_squared_error(labels, preds) ** 0.5)
            mae  = float(mean_absolute_error(labels, preds))
            r2   = float(r2_score(labels, preds))
            result = {"val_loss": avg_loss, "val_rmse": rmse, "val_mae": mae, "val_r2": r2}
            # MAPE — safe against zero-valued labels
            nonzero = np.abs(labels) > 1e-8
            if nonzero.sum() > 0:
                mape = float(np.mean(np.abs((labels[nonzero] - preds[nonzero]) / labels[nonzero])))
                result["val_mape"] = mape
            # Predicted vs actual scatter (max 500 points)
            try:
                sample_n = min(len(preds), 500)
                rng = np.random.default_rng(42)
                idx = rng.choice(len(preds), sample_n, replace=False)
                result["val_scatter"] = {
                    "actual":    [round(float(v), 6) for v in labels[idx]],
                    "predicted": [round(float(v), 6) for v in preds[idx]],
                }
            except Exception as e:
                logger.warning("Scatter data computation skipped: %s", e)
            return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str, arch_config: dict | None = None, input_dim: int = 0,
             task_type: str = "", feature_columns: list[str] | None = None,
             preprocessing_artifact: DLPreprocessingArtifact | None = None):
        """Save checkpoint plus optional scaler and preprocessing sidecars."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "model_type":      self.model_type,
            "num_classes":     self.num_classes,
            "arch_config":     arch_config or {},
            "input_dim":       input_dim,
            "task_type":       task_type,
            "feature_columns": feature_columns or [],
        }, p)
        if self.scaler is not None:
            joblib.dump(self.scaler, str(p) + ".scaler.joblib")
        self.preprocessing_artifact = preprocessing_artifact
        if preprocessing_artifact is not None:
            joblib.dump(preprocessing_artifact, str(p) + ".preprocessor.joblib")
        logger.info("DL model saved → %s", p)

    def load(self, path: str):
        """Load model weights (model must already be built)."""
        p = Path(path)
        ckpt = torch.load(p, map_location=self.device, weights_only=True)
        if self.model is not None:
            self.model.load_state_dict(ckpt["model_state"])
        self.model_type  = ckpt.get("model_type",  self.model_type)
        self.num_classes = ckpt.get("num_classes", self.num_classes)
        scaler_path = str(p) + ".scaler.joblib"
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        preprocessing_path = str(p) + ".preprocessor.joblib"
        self.preprocessing_artifact = (
            joblib.load(preprocessing_path)
            if Path(preprocessing_path).exists()
            else None
        )

    def load_for_inference(self, path: str) -> dict:
        """Build model from saved checkpoint metadata + load weights + scaler.

        Returns a dict with task_type and feature_columns for callers.
        """
        p = Path(path)
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        arch_config     = ckpt.get("arch_config", {})
        input_dim       = ckpt.get("input_dim", 0)
        self.num_classes = ckpt.get("num_classes", 1)
        self.model_type  = ckpt.get("model_type", self.model_type)
        task_type        = ckpt.get("task_type", "classification")
        feature_columns  = ckpt.get("feature_columns", [])

        output_dim = self.num_classes if task_type == "classification" else 1
        if input_dim > 0:
            self.model = self.build_model(input_dim, output_dim, arch_config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        scaler_path = str(p) + ".scaler.joblib"
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)

        preprocessing_path = str(p) + ".preprocessor.joblib"
        self.preprocessing_artifact = (
            joblib.load(preprocessing_path)
            if Path(preprocessing_path).exists()
            else None
        )

        return {
            "task_type": task_type,
            "feature_columns": feature_columns,
            "preprocessing_artifact": self.preprocessing_artifact,
        }

    def predict(self, X: np.ndarray, task_type: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Run inference on pre-processed input X.  Returns (predictions, probabilities)."""
        self.model.eval()
        if self.scaler is not None:
            X = self.scaler.transform(X.astype(np.float32))
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(Xt)
        if task_type == "classification":
            probas = torch.softmax(out, dim=1).cpu().numpy()
            preds  = out.argmax(1).cpu().numpy()
            return preds, probas
        else:
            preds = out.cpu().numpy().flatten()
            return preds, None
