import numpy as np

from app.core.dl_registry import get_dl_trainer


def test_training_loader_drops_single_sample_tail_for_batch_norm_models():
    trainer = get_dl_trainer("mlp_dl")
    X_train = np.arange(33 * 4, dtype=np.float32).reshape(33, 4)
    y_train = np.linspace(0, 1, 33, dtype=np.float32)
    X_val = X_train[:8]
    y_val = y_train[:8]

    train_loader, _ = trainer._make_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=32,
        task_type="regression",
    )

    assert train_loader.drop_last is True
    assert [len(batch_X) for batch_X, _ in train_loader] == [32]


def test_training_loader_keeps_non_single_tail():
    trainer = get_dl_trainer("mlp_dl")
    X_train = np.arange(34 * 4, dtype=np.float32).reshape(34, 4)
    y_train = np.linspace(0, 1, 34, dtype=np.float32)

    train_loader, _ = trainer._make_loaders(
        X_train,
        y_train,
        X_train[:8],
        y_train[:8],
        batch_size=32,
        task_type="regression",
    )

    assert train_loader.drop_last is False
    assert sorted(len(batch_X) for batch_X, _ in train_loader) == [2, 32]
