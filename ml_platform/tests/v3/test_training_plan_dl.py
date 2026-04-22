"""
Phase 1 — TrainingPlan ML/DL/mixed validation tests.

Covers the behaviour added in V3 Phase 1:
  * `model_family` enum + persistence
  * `selected_models` must agree with `model_family`:
     - family='ml'  → all tokens must resolve to the ML registry
     - family='dl'  → all tokens must resolve to the DL registry
     - family='mixed' → both allowed, unknown tokens rejected
  * `dl_config` is backfilled from registry defaults for any DL token missing
    an entry (baseline usability: user can leave it blank)
  * Round-trip through create/update preserves new fields
  * Unknown tokens still raise HTTP 422
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.dl_registry import DL_MODEL_REGISTRY
from app.core.model_registry import MODEL_REGISTRY
from app.services import training_plan_service as svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_ml_token(task_type: str = "classification") -> str:
    for m in MODEL_REGISTRY:
        if task_type in m.get("task_types", []):
            return m["id"]
    pytest.skip("No ML token available for task_type")


def _first_dl_token(task_type: str = "classification") -> str:
    for m in DL_MODEL_REGISTRY:
        if task_type in m.get("task_types", []):
            return m["id"]
    pytest.skip("No DL token available for task_type")


# ---------------------------------------------------------------------------
# Create — family + selected_models combinations
# ---------------------------------------------------------------------------

async def test_create_ml_only_plan(db):
    ml_token = _first_ml_token()
    plan = await svc.create_plan(db, {
        "name": "ml-only",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "ml",
        "selected_models": [ml_token],
    })
    assert plan["model_family"] == "ml"
    assert plan["selected_models"] == [ml_token]
    # No DL tokens → dl_config stays empty / null
    assert not plan.get("dl_config")


async def test_create_dl_only_plan_backfills_defaults(db):
    dl_token = _first_dl_token()
    plan = await svc.create_plan(db, {
        "name": "dl-only",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "dl",
        "selected_models": [dl_token],
        # dl_config intentionally omitted — service should backfill
    })
    assert plan["model_family"] == "dl"
    assert plan["selected_models"] == [dl_token]
    cfg = plan["dl_config"]
    assert dl_token in cfg, "dl_config must contain an entry for the selected DL token"
    entry = cfg[dl_token]
    # Every backfilled entry carries arch/opt/train sections
    assert set(entry.keys()) >= {"arch", "opt", "train"}
    assert isinstance(entry["opt"], dict) and entry["opt"]  # non-empty


async def test_create_mixed_plan_allows_both(db):
    ml_token = _first_ml_token()
    dl_token = _first_dl_token()
    plan = await svc.create_plan(db, {
        "name": "mixed-1",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "mixed",
        "selected_models": [ml_token, dl_token],
    })
    assert plan["model_family"] == "mixed"
    assert set(plan["selected_models"]) == {ml_token, dl_token}
    assert dl_token in (plan.get("dl_config") or {})
    assert ml_token not in (plan.get("dl_config") or {})


# ---------------------------------------------------------------------------
# Create — rejection paths
# ---------------------------------------------------------------------------

async def test_create_ml_family_rejects_dl_token(db):
    dl_token = _first_dl_token()
    with pytest.raises(HTTPException) as exc:
        await svc.create_plan(db, {
            "name": "bad-ml",
            "task_type": "classification",
            "strategy_type": "baseline",
            "model_family": "ml",
            "selected_models": [dl_token],
        })
    assert exc.value.status_code == 422
    assert "DL model" in exc.value.detail


async def test_create_dl_family_rejects_ml_token(db):
    ml_token = _first_ml_token()
    with pytest.raises(HTTPException) as exc:
        await svc.create_plan(db, {
            "name": "bad-dl",
            "task_type": "classification",
            "strategy_type": "baseline",
            "model_family": "dl",
            "selected_models": [ml_token],
        })
    assert exc.value.status_code == 422
    assert "ML model" in exc.value.detail


async def test_create_rejects_unknown_token(db):
    with pytest.raises(HTTPException) as exc:
        await svc.create_plan(db, {
            "name": "bad-unknown",
            "task_type": "classification",
            "strategy_type": "baseline",
            "model_family": "mixed",
            "selected_models": ["not_a_real_model_xyz"],
        })
    assert exc.value.status_code == 422
    assert "Unknown model" in exc.value.detail


async def test_create_rejects_invalid_family(db):
    ml_token = _first_ml_token()
    with pytest.raises(HTTPException) as exc:
        await svc.create_plan(db, {
            "name": "bad-family",
            "task_type": "classification",
            "strategy_type": "baseline",
            "model_family": "quantum",
            "selected_models": [ml_token],
        })
    assert exc.value.status_code == 422
    assert "Invalid model_family" in exc.value.detail


# ---------------------------------------------------------------------------
# Update — re-validates when family / selected_models change
# ---------------------------------------------------------------------------

async def test_update_switch_to_dl_backfills_and_validates(db):
    ml_token = _first_ml_token()
    dl_token = _first_dl_token()
    plan = await svc.create_plan(db, {
        "name": "mutable",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "ml",
        "selected_models": [ml_token],
    })
    updated = await svc.update_plan(db, plan["id"], {
        "model_family": "dl",
        "selected_models": [dl_token],
    })
    assert updated["model_family"] == "dl"
    assert updated["selected_models"] == [dl_token]
    assert dl_token in (updated.get("dl_config") or {})


async def test_update_rejects_mismatched_change(db):
    ml_token = _first_ml_token()
    dl_token = _first_dl_token()
    plan = await svc.create_plan(db, {
        "name": "mutable-2",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "ml",
        "selected_models": [ml_token],
    })
    # Swap in DL token while keeping family=ml → should 422
    with pytest.raises(HTTPException) as exc:
        await svc.update_plan(db, plan["id"], {
            "selected_models": [dl_token],
        })
    assert exc.value.status_code == 422


async def test_update_preserves_existing_dl_config_overrides(db):
    dl_token = _first_dl_token()
    plan = await svc.create_plan(db, {
        "name": "dl-ovr",
        "task_type": "classification",
        "strategy_type": "baseline",
        "model_family": "dl",
        "selected_models": [dl_token],
        "dl_config": {dl_token: {"train": {"epochs": 7}}},
    })
    # Registry defaults include epochs=50 — the user's override must survive.
    assert plan["dl_config"][dl_token]["train"]["epochs"] == 7

    # Bumping description shouldn't clobber the override
    updated = await svc.update_plan(db, plan["id"], {"description": "note"})
    assert updated["dl_config"][dl_token]["train"]["epochs"] == 7
