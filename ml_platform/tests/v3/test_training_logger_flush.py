"""Buffered log entries must reach the DB *during* a run, not only at its end.

`flush_to_db()` used to be called once, from `_run_training_sync` after
training finished. Until then `training_logs` held nothing, so opening the log
panel mid-training showed an empty history and a crash left the entries only in
the on-disk .log file. The buffer now also flushes on a size or age threshold.
"""
import time
from unittest import mock

import pytest

from app.core.logger import TrainingLogger


@pytest.fixture
def logger_factory(tmp_path, monkeypatch):
    """TrainingLogger writing into a tmp dir, with flush_to_db stubbed out.

    The tests below pin *when* a flush is triggered; flush_to_db's own DB write
    is covered by the code path that already existed.
    """
    from app.core import logger as logger_module

    # Settings is a frozen dataclass, so swap the accessor rather than a field.
    stub_settings = mock.Mock()
    stub_settings.storage_logs = tmp_path
    monkeypatch.setattr(logger_module, "get_settings", lambda: stub_settings)

    def _make(**kwargs):
        tl = TrainingLogger(task_id="task-under-test", model_type="random_forest", **kwargs)
        tl.flush_calls = 0

        def _counting_flush():
            tl.flush_calls += 1
            with tl._buffer_lock:
                drained = len(tl._db_buffer)
                tl._db_buffer = []
                tl._last_flush_at = time.monotonic()
            return drained

        tl.flush_to_db = _counting_flush
        return tl

    return _make


def test_flushes_once_the_buffer_reaches_the_size_threshold(logger_factory):
    tl = logger_factory()
    for i in range(TrainingLogger._FLUSH_EVERY_N_ENTRIES - 1):
        tl.log("INFO", f"line {i}")
    assert tl.flush_calls == 0, "flushed before the threshold was reached"

    tl.log("INFO", "the line that crosses the threshold")
    assert tl.flush_calls == 1


def test_flushes_once_the_buffer_is_old_enough(logger_factory):
    """A quiet run must still surface its few lines without waiting for the end."""
    tl = logger_factory()
    tl.log("INFO", "first line")
    assert tl.flush_calls == 0

    # Pretend the last flush happened longer ago than the interval, rather than
    # sleeping for it.
    tl._last_flush_at = time.monotonic() - TrainingLogger._FLUSH_INTERVAL_SECONDS - 0.1
    tl.log("INFO", "second line")
    assert tl.flush_calls == 1


def test_persist_to_db_false_never_buffers_or_flushes(logger_factory):
    """DL tasks are not rows in training_tasks, whose id column the FK targets.

    Buffering them would make every flush fail the constraint, push the entries
    back, and retry forever.
    """
    tl = logger_factory(persist_to_db=False)
    for i in range(TrainingLogger._FLUSH_EVERY_N_ENTRIES * 2):
        tl.log("INFO", f"line {i}")
    assert tl.flush_calls == 0
    assert tl._db_buffer == []


def test_file_and_event_bus_are_unaffected_when_db_persistence_is_off(logger_factory):
    """Turning off DB buffering must not turn off the live stream or the file."""
    tl = logger_factory(persist_to_db=False)
    with mock.patch("app.core.logger.event_bus") as bus:
        tl.log("INFO", "still streamed")
    bus.publish.assert_called_once()
    assert "still streamed" in tl.log_file.read_text(encoding="utf-8")


def test_a_failing_flush_keeps_the_backlog_bounded(logger_factory, monkeypatch):
    """A broken DB must not turn logging into unbounded memory growth."""
    tl = logger_factory()
    del tl.flush_to_db          # exercise the real flush, which will fail
    monkeypatch.setattr(
        "app.models.database.sync_session_factory",
        mock.Mock(side_effect=RuntimeError("database is down")),
        raising=False,
    )
    for i in range(TrainingLogger._MAX_BUFFERED_ENTRIES + 200):
        tl.log("INFO", f"line {i}")

    assert len(tl._db_buffer) <= TrainingLogger._MAX_BUFFERED_ENTRIES
    # The cap keeps the newest entries, which are the ones worth having.
    assert tl._db_buffer[-1]["message"].endswith(
        str(TrainingLogger._MAX_BUFFERED_ENTRIES + 199)
    )
