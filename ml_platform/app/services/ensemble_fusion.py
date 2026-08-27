"""Pure fusion maths for weighted multi-model deployments.

Kept free of the database and of model loading so the part that is easy to get
silently wrong can be tested directly.

The classification path is the reason this module exists. `predict_with_model`
returns ``probabilities`` as positional rows straight out of ``predict_proba``,
with the meaning of each column carried separately in ``class_labels``. Two
members trained on the same data can order their classes differently — each
fits its own LabelEncoder — so averaging column 0 against column 0 can silently
mix "cat" into "dog". Every member's row is therefore keyed by its own labels
before anything is added together.
"""

from __future__ import annotations

from typing import Any

# Weights below this are treated as absent rather than as a vanishing vote.
_WEIGHT_EPSILON = 1e-12


def normalise_weights(weights: list[float]) -> list[float]:
    """Scale weights to sum to 1, keeping their ratios.

    All-zero (or empty) input falls back to an equal split: a deployment whose
    weights cancelled out should still predict, and equal weighting is the
    honest default rather than an arbitrary winner.
    """
    usable = [max(float(w or 0.0), 0.0) for w in weights]
    total = sum(usable)
    if not usable:
        return []
    if total <= _WEIGHT_EPSILON:
        return [1.0 / len(usable)] * len(usable)
    return [w / total for w in usable]


def fuse_regression(member_predictions: list[list[float]], weights: list[float]) -> list[float]:
    """Weighted average of each member's numeric predictions, row by row."""
    if not member_predictions:
        raise ValueError("至少需要一个成员的预测结果")
    row_counts = {len(p) for p in member_predictions}
    if len(row_counts) != 1:
        raise ValueError(f"各成员返回的行数不一致: {sorted(row_counts)}")

    w = normalise_weights(weights)
    fused: list[float] = []
    for row_idx in range(row_counts.pop()):
        total = 0.0
        for member_idx, preds in enumerate(member_predictions):
            total += w[member_idx] * float(preds[row_idx])
        fused.append(total)
    return fused


def fuse_classification(
    member_results: list[dict[str, Any]],
    weights: list[float],
) -> dict[str, Any]:
    """Weighted average of per-class probabilities, aligned by class label.

    ``member_results`` items carry ``class_labels`` and ``probabilities``
    (positional rows). A member without usable probabilities contributes a
    one-hot vote from its hard prediction instead of being dropped — dropping
    it would silently change the weighting the caller configured.
    """
    if not member_results:
        raise ValueError("至少需要一个成员的预测结果")

    row_counts = {len(m.get("predictions") or []) for m in member_results}
    if len(row_counts) != 1:
        raise ValueError(f"各成员返回的行数不一致: {sorted(row_counts)}")
    n_rows = row_counts.pop()

    w = normalise_weights(weights)

    # Union of every member's labels, in first-seen order, so a member that
    # never saw a rare class does not erase it from the fused output.
    labels: list[str] = []
    for member in member_results:
        for label in member.get("class_labels") or []:
            if str(label) not in labels:
                labels.append(str(label))
    if not labels:
        raise ValueError("成员未提供类别标签，无法融合分类结果")

    fused_rows: list[dict[str, float]] = []
    for row_idx in range(n_rows):
        acc = {label: 0.0 for label in labels}
        for member_idx, member in enumerate(member_results):
            weight = w[member_idx]
            member_labels = [str(x) for x in (member.get("class_labels") or [])]
            probs = member.get("probabilities")
            row_probs = probs[row_idx] if probs and row_idx < len(probs) else None

            if row_probs is not None and len(row_probs) == len(member_labels):
                # Key by this member's own labels before adding — the whole
                # point of the exercise.
                for label, p in zip(member_labels, row_probs):
                    acc[label] += weight * float(p)
            else:
                # No probabilities (or a shape that does not match its labels):
                # treat the hard prediction as a full vote for that class.
                predicted = str((member.get("predictions") or [None])[row_idx])
                if predicted in acc:
                    acc[predicted] += weight
                else:
                    acc[predicted] = weight
                    labels.append(predicted)

        total = sum(acc.values())
        if total > _WEIGHT_EPSILON:
            acc = {k: v / total for k, v in acc.items()}
        fused_rows.append(acc)

    predictions = [max(row.items(), key=lambda kv: kv[1])[0] for row in fused_rows]
    return {
        "predictions": predictions,
        "class_labels": labels,
        "probabilities": fused_rows,
    }
