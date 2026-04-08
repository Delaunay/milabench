import os
from contextlib import contextmanager

from .toggles import _get_flag


_DEFAULT_TRACE_DIR = "/tmp/milabench-trace"


def is_profiling_enabled():
    return _get_flag("MILABENCH_PROFILE", int, 0) != 0


def _get_trace_dir():
    return _get_flag("MILABENCH_PROFILE_DIR", str, _DEFAULT_TRACE_DIR)


def _get_optional_int_flag(name):
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return int(value)


def _get_trace_window():
    start_obs = _get_optional_int_flag("MILABENCH_PROFILE_START_OBS")
    stop_obs = _get_optional_int_flag("MILABENCH_PROFILE_STOP_OBS")
    window_obs = _get_optional_int_flag("MILABENCH_PROFILE_WINDOW_OBS")

    if start_obs is not None and stop_obs is None and window_obs is not None:
        stop_obs = start_obs + max(window_obs, 1)

    return start_obs, stop_obs


class JaxProfilerWindow:
    def __init__(self, trace_dir=None, enabled=None):
        if enabled is None:
            enabled = is_profiling_enabled()

        if trace_dir is None:
            trace_dir = _get_trace_dir()

        self.enabled = enabled
        self.trace_dir = trace_dir
        self.start_obs, self.stop_obs = _get_trace_window()
        self.started = False
        self.stopped = False

    @property
    def active(self):
        return (
            self.enabled
            and self.start_obs is not None
            and self.stop_obs is not None
            and self.stop_obs > self.start_obs
        )

    def maybe_advance(self, observation):
        if not self.active or self.stopped:
            return

        import jax

        if not self.started and observation >= self.start_obs:
            os.makedirs(self.trace_dir, exist_ok=True)
            print(
                "[profiler] Window tracing to "
                f"{self.trace_dir} over observations "
                f"[{self.start_obs}, {self.stop_obs})"
            )
            jax.profiler.start_trace(self.trace_dir)
            self.started = True

        if self.started and observation >= self.stop_obs:
            jax.profiler.stop_trace()
            self.stopped = True

    def close(self):
        if self.started and not self.stopped:
            import jax

            jax.profiler.stop_trace()
            self.stopped = True


@contextmanager
def jax_profiler(trace_dir=None, enabled=None):
    """Context manager that optionally activates JAX's XLA profiler trace.

    When enabled, produces a trace viewable in TensorBoard or Perfetto
    (ui.perfetto.dev). When disabled, this is a complete no-op.

    Named scopes (``jax.named_scope``) placed in the training loop are
    always free; they only become visible when a trace is active.

    Enable via:
      - ``enabled=True`` argument, or
      - ``MILABENCH_PROFILE=1`` environment variable

    The trace directory defaults to ``/tmp/milabench-trace`` and can be
    overridden with ``MILABENCH_PROFILE_DIR``.
    """
    if enabled is None:
        enabled = is_profiling_enabled()

    if not enabled:
        yield
        return

    trace_window = JaxProfilerWindow(trace_dir=trace_dir, enabled=enabled)
    if trace_window.active:
        yield
        return

    import jax

    if trace_dir is None:
        trace_dir = _get_trace_dir()

    os.makedirs(trace_dir, exist_ok=True)
    print(f"[profiler] Tracing to {trace_dir}")

    with jax.profiler.trace(trace_dir):
        yield
