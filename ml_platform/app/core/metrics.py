"""Supported metrics and their descriptions."""

SUPPORTED_METRICS = {
    "accuracy": "Classification accuracy",
    "f1": "F1 score (weighted)",
    "precision": "Precision (weighted)",
    "recall": "Recall (weighted)",
    "roc_auc": "ROC AUC score",
    "log_loss": "Log loss / cross-entropy",
}

def get_supported_metrics() -> dict[str, str]:
    return SUPPORTED_METRICS.copy()
