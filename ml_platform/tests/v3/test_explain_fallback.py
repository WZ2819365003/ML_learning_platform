import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from app.services.explain_service import _fallback_feature_importances


def test_fallback_uses_permutation_importance_for_models_without_native_importance():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])

    model = SVC(kernel="rbf")
    model.fit(frame, y)

    importances, method = _fallback_feature_importances(model, frame, pd.Series(y))

    assert method == "permutation_importance"
    assert set(importances) == {"f1", "f2", "f3", "f4"}
    assert all(isinstance(value, float) for value in importances.values())


def test_fallback_without_target_raises_for_models_without_native_importance():
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    model = SVC(kernel="rbf")
    model.fit(frame, y)

    with pytest.raises(ValueError, match="fallback feature importance"):
        _fallback_feature_importances(model, frame)
