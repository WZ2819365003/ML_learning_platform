from __future__ import annotations

from pathlib import Path

import yaml

from app.core.dl_registry import DL_MODEL_REGISTRY, get_dl_trainer_registry
from app.core.model_registry import MODEL_REGISTRY
from app.core.regression_trainers import REGRESSION_TRAINER_REGISTRY
from app.core.trainer import TRAINER_REGISTRY


REGISTRY_DIR = Path(__file__).resolve().parents[1] / "registry"


def _load_yaml(name: str) -> dict:
    with (REGISTRY_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_ml_model_registry_matches_trainers_and_tuning_spaces():
    model_specs = {item["id"]: item for item in MODEL_REGISTRY}
    trainers = {**TRAINER_REGISTRY, **REGRESSION_TRAINER_REGISTRY}
    tuning_spaces = _load_yaml("tuning_spaces.yaml")

    for model_id, spec in model_specs.items():
        assert model_id in trainers, (
            f"模型 {model_id!r} 已配置 MODEL_REGISTRY，但漏配 trainer registry"
        )
        trainers[model_id]()
        for task_type in spec["task_types"]:
            assert model_id in tuning_spaces.get(task_type, {}), (
                f"模型 {model_id!r} 声明 task_type={task_type!r}，"
                "但漏配 tuning_spaces.yaml"
            )

    for task_type, models in tuning_spaces.items():
        for model_id in models:
            assert model_id in model_specs, (
                f"模型 {model_id!r} 已配置 tuning_spaces.yaml/{task_type}，"
                "但漏配 MODEL_REGISTRY"
            )
            assert task_type in model_specs[model_id]["task_types"], (
                f"模型 {model_id!r} 在 tuning_spaces.yaml/{task_type} 中，"
                f"但 MODEL_REGISTRY.task_types={model_specs[model_id]['task_types']!r} 不匹配"
            )


def test_automl_candidates_reference_compatible_ml_models():
    model_specs = {item["id"]: item for item in MODEL_REGISTRY}
    candidates = _load_yaml("automl_candidates.yaml")

    for task_type in ("classification", "regression"):
        for candidate in candidates.get(task_type, []):
            model_id = candidate["model_type"]
            assert model_id in model_specs, (
                f"模型 {model_id!r} 已被 automl_candidates.yaml/{task_type} 引用，"
                "但漏配 MODEL_REGISTRY"
            )
            assert task_type in model_specs[model_id]["task_types"], (
                f"模型 {model_id!r} 被 automl_candidates.yaml/{task_type} 引用，"
                f"但 MODEL_REGISTRY.task_types={model_specs[model_id]['task_types']!r} 不匹配"
            )


def test_dl_model_registry_matches_trainer_registry_both_ways():
    model_specs = {item["id"]: item for item in DL_MODEL_REGISTRY}
    trainers = get_dl_trainer_registry()

    for model_id in model_specs:
        assert model_id in trainers, (
            f"模型 {model_id!r} 已配置 DL_MODEL_REGISTRY，但漏配 DL trainer registry"
        )
        trainers[model_id]()

    for model_id in trainers:
        assert model_id in model_specs, (
            f"模型 {model_id!r} 已配置 DL trainer registry，但漏配 DL_MODEL_REGISTRY"
        )
