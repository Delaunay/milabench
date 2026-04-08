from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmate.profiler import JaxProfilerWindow, jax_profiler


def test_jax_profiler_window_uses_start_and_window_observations():
    with patch.dict(
        "os.environ",
        {
            "MILABENCH_PROFILE": "1",
            "MILABENCH_PROFILE_START_OBS": "12",
            "MILABENCH_PROFILE_WINDOW_OBS": "7",
        },
        clear=False,
    ):
        window = JaxProfilerWindow(trace_dir="/tmp/trace")

    assert window.active
    assert window.start_obs == 12
    assert window.stop_obs == 19


def test_jax_profiler_window_starts_and_stops_trace_once():
    fake_profiler = SimpleNamespace(start_trace=Mock(), stop_trace=Mock())
    fake_jax = SimpleNamespace(profiler=fake_profiler)

    with patch.dict(
        "os.environ",
        {
            "MILABENCH_PROFILE": "1",
            "MILABENCH_PROFILE_START_OBS": "3",
            "MILABENCH_PROFILE_WINDOW_OBS": "2",
        },
        clear=False,
    ):
        window = JaxProfilerWindow(trace_dir="/tmp/trace")

    with patch.dict("sys.modules", {"jax": fake_jax}):
        with patch("benchmate.profiler.os.makedirs") as makedirs:
            window.maybe_advance(2)
            fake_profiler.start_trace.assert_not_called()

            window.maybe_advance(3)
            makedirs.assert_called_once_with("/tmp/trace", exist_ok=True)
            fake_profiler.start_trace.assert_called_once_with("/tmp/trace")
            fake_profiler.stop_trace.assert_not_called()

            window.maybe_advance(4)
            fake_profiler.stop_trace.assert_not_called()

            window.maybe_advance(5)
            fake_profiler.stop_trace.assert_called_once_with()

            window.maybe_advance(8)
            fake_profiler.start_trace.assert_called_once_with("/tmp/trace")
            fake_profiler.stop_trace.assert_called_once_with()


def test_jax_profiler_skips_full_run_trace_when_window_is_enabled():
    @contextmanager
    def fail_trace(_trace_dir):
        raise AssertionError("whole-run trace should be skipped")
        yield

    fake_jax = SimpleNamespace(profiler=SimpleNamespace(trace=fail_trace))

    with patch.dict(
        "os.environ",
        {
            "MILABENCH_PROFILE": "1",
            "MILABENCH_PROFILE_START_OBS": "5",
            "MILABENCH_PROFILE_WINDOW_OBS": "5",
        },
        clear=False,
    ):
        with patch.dict("sys.modules", {"jax": fake_jax}):
            with jax_profiler(trace_dir="/tmp/trace"):
                pass
