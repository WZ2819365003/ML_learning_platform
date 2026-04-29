from app.core.ts_registry import (
    TS_MODEL_REGISTRY,
    get_ts_model_spec,
    get_ts_model_ids,
    list_available_models,
)


def test_5_models_registered():
    ids = get_ts_model_ids()
    assert ids == ["arima", "ets", "lstm_forecaster", "tcn_forecaster", "timesfm_1"]


def test_each_model_has_required_fields():
    required = {"id", "display_name", "category", "task_types", "params",
                "supports_intervals", "supports_exogenous"}
    for m in TS_MODEL_REGISTRY:
        missing = required - m.keys()
        assert not missing, f"{m['id']} missing: {missing}"


def test_arima_supports_intervals_and_exog():
    m = get_ts_model_spec("arima")
    assert m["supports_intervals"] is True
    assert m["supports_exogenous"] is True


def test_lstm_does_not_support_intervals_baseline():
    m = get_ts_model_spec("lstm_forecaster")
    assert m["supports_intervals"] is False


def test_unknown_id_returns_none():
    assert get_ts_model_spec("not_a_model") is None


def test_list_available_marks_timesfm_status():
    avail = list_available_models()
    timesfm = next(m for m in avail if m["id"] == "timesfm_1")
    assert "available" in timesfm
    assert isinstance(timesfm["available"], bool)
