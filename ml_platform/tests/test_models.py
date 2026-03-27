"""测试模型功能"""

import pytest
from app.core.trainer import get_trainer, list_available_models
import numpy as np


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost", "lightgbm", "logistic_regression", "svm", "mlp"])
def test_get_trainer(model_type):
    """测试获取不同类型的训练器"""
    trainer = get_trainer(model_type)
    assert trainer is not None
    assert trainer.model_type == model_type


def test_list_available_models():
    """测试列出可用模型"""
    models = list_available_models()
    assert isinstance(models, list)
    assert len(models) == 6
    expected_models = ["random_forest", "xgboost", "lightgbm", "logistic_regression", "svm", "mlp"]
    for model in expected_models:
        assert model in models


@pytest.mark.parametrize("model_type", ["random_forest", "logistic_regression"])
def test_trainer_configure(model_type):
    """测试训练器配置"""
    trainer = get_trainer(model_type)
    hyperparameters = {
        "n_estimators": 50,
        "max_depth": 5
    }
    trainer.configure(hyperparameters)
    assert trainer.model is not None


@pytest.mark.parametrize("model_type", ["random_forest", "logistic_regression"])
def test_trainer_train(model_type):
    """测试训练器训练"""
    trainer = get_trainer(model_type)
    trainer.configure({"n_estimators": 10, "max_depth": 3})
    
    # 创建简单的训练数据
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([0, 1, 0, 1, 0])
    X_val = np.array([[11, 12], [13, 14]])
    y_val = np.array([1, 0])
    
    # 训练模型
    metrics = trainer.train(X_train, y_train, X_val, y_val, 
                          eval_metrics=["accuracy", "f1"],
                          cv_folds=2)
    
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "cv_avg_accuracy" in metrics
    assert "cv_avg_f1" in metrics
