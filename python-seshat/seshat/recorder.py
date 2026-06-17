"""Cross-language recorder using ``sys.setprofile``.

``sys.setprofile`` is the key: its callback fires on every Python ``call`` /
``return`` **and** on every ``c_call`` / ``c_return`` — i.e. every time Python
crosses into a native (C / Rust-extension) function. That single hook gives us,
in pure stdlib Python:

* **Real Python frames** (``co_name`` / file / line) → ``FrameKind::Py``.
* **The Python↔native boundary** (``c_call``) → an ``FfiBoundary`` frame plus a
  ``BoundaryCross`` edge. A PyO3 Rust function is one kind of native callee, so
  this is exactly the Py↔Rust seam Seshat exists to measure.

A Sample is emitted on each ``call`` / ``c_call`` (the stack at that moment), so
the flamegraph is weighted by inclusive call count. Because the call sequence of
a deterministic Python program is fixed, the resulting trace is itself
deterministic — a stronger property than a wall-clock sampler.

Scope: main thread only (``sys.setprofile`` is per-thread); no Python-heap
allocation capture yet (a future ``tracemalloc`` integration). Wall-clock total
is recorded as advisory metadata only.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import List

from .format import TraceWriter, DOMAIN, DOMAIN_PYHEAP, KIND_ASYNC, KIND_FFI, KIND_PY

# Directory of the seshat package — frames from here are the recorder's own and
# must not appear in the trace.
_SESHAT_DIR = os.path.dirname(os.path.abspath(__file__))

# CPython code-object flags marking a frame as an async task.
_CO_COROUTINE = 0x80
_CO_ASYNC_GENERATOR = 0x200
_CO_ITERABLE_COROUTINE = 0x100
_ASYNC_FLAGS = _CO_COROUTINE | _CO_ASYNC_GENERATOR | _CO_ITERABLE_COROUTINE

# "Transparent" C calls that merely *run Python code* (the script bootstrap and
# the import machinery). Counting them as boundary crossings would make the
# whole program look like it lives at the boundary, so they are skipped — but
# their Python children are still recorded.
_TRANSPARENT_C = frozenset(
    {
        "builtins.exec",
        "builtins.eval",
        "builtins.compile",
        "builtins.__import__",
        "sys.setprofile",
        "sys.settrace",
    }
)

# The recorder currently driving capture, so `zone()` can reach its writer.
_ACTIVE = None


class Recorder:
    """Records the current thread's Python execution into a :class:`TraceWriter`.

    Two modes:

    * ``"sampling"`` (default) — ``sys.setprofile`` cheaply *maintains* the call
      stack and the Py↔native boundary, while a background thread samples that
      stack at a fixed interval. Samples are **time-weighted** (a slow function
      gets more samples) and the event count is tiny. This is the right mode for
      "where is my wall-clock going?". Nondeterministic (timing-based).
    * ``"calls"`` — a Sample is emitted per call, giving a call-count-weighted
      profile that is **deterministic** (a fixed program → identical trace).
      Use for CI ``diff``/``variance`` gates. Higher overhead.

    Frame interning is cached by code-object / native-label id so the
    per-call hook stays cheap.
    """

    def __init__(self) -> None:
        self.w = TraceWriter()
        # Each entry is a frame id (recorded) or None (a skipped bootstrap /
        # recorder frame). The None placeholders keep push/pop balanced.
        self._stack: List = []
        self._t0 = 0
        self._internal_cache = {}
        self._trace_memory = False
        self._coro_suspend = {}
        self._mode = "sampling"
        self._interval = 0.002
        self._sampling = False
        self._sampler = None
        self._pyframe_cache = {}  # id(code) -> (fid, is_async)
        self._boundary_cache = {}  # native label -> fid
        self._sampled_frame = 0
        self._prev_mem = 0

    def start(self, trace_memory: bool = False, mode: str = "sampling", interval_ms: float = 2.0) -> None:
        global _ACTIVE
        _ACTIVE = self
        self._mode = "calls" if mode == "calls" else "sampling"
        self._interval = max(0.0005, interval_ms / 1000.0)
        self._trace_memory = trace_memory
        if trace_memory:
            try:
                import tracemalloc

                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                self._prev_mem = tracemalloc.get_traced_memory()[0]
            except Exception:
                self._trace_memory = False
        # fallback frame for memory attribution when the stack is empty
        self._sampled_frame = self.w.intern_frame(KIND_PY, "<sampled>", "<sampler>", 0)
        self._t0 = time.perf_counter_ns()
        sys.setprofile(self._hook)
        if self._mode == "sampling":
            self._sampling = True
            self._sampler = threading.Thread(
                target=self._sample_loop, name="seshat-sampler", daemon=True
            )
            self._sampler.start()

    def stop(self) -> TraceWriter:
        global _ACTIVE
        self._sampling = False
        sys.setprofile(None)
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
            self._sampler = None
        if self._trace_memory and self._mode == "calls":
            self._emit_tracemalloc()  # end-of-run snapshot (site attribution)
        if self._trace_memory:
            try:
                import tracemalloc

                if tracemalloc.is_tracing():
                    tracemalloc.stop()
            except Exception:
                pass
        self.w.wall_ns = max(0, time.perf_counter_ns() - self._t0)
        _ACTIVE = None
        return self.w

    def _sample_loop(self) -> None:
        """Background sampler (no ``setprofile`` here, so it is invisible to the
        hook). Each tick: emit a time-weighted Sample of the live stack, and a
        PyHeap alloc/free delta (temporal memory) attributed to the running
        frame. Reads the stack via an atomic slice copy — safe under the GIL
        without a lock."""
        try:
            import tracemalloc
        except Exception:
            tracemalloc = None
        while self._sampling:
            time.sleep(self._interval)
            try:
                snapshot = self._stack[:]  # atomic copy under the GIL
                live = [f for f in snapshot if f is not None]
                top = live[-1] if live else self._sampled_frame
                if live:
                    self.w.sample(live)
                if self._trace_memory and tracemalloc is not None and tracemalloc.is_tracing():
                    cur = tracemalloc.get_traced_memory()[0]
                    delta = cur - self._prev_mem
                    self._prev_mem = cur
                    if delta > 0:
                        self.w.alloc(DOMAIN_PYHEAP, delta, top)
                    elif delta < 0:
                        self.w.free(DOMAIN_PYHEAP, -delta, top)
            except Exception:
                pass

    def _emit_tracemalloc(self) -> None:
        """`calls`-mode only: append live Python-heap sites as PyHeap allocs."""
        try:
            import tracemalloc

            if not tracemalloc.is_tracing():
                return
            snapshot = tracemalloc.take_snapshot()
            for stat in snapshot.statistics("lineno")[:64]:
                if stat.size <= 0 or not stat.traceback:
                    continue
                frame_info = stat.traceback[0]
                if self._is_internal(frame_info.filename):
                    continue
                fid = self.w.intern_frame(
                    KIND_PY, "<pyalloc>", os.path.basename(frame_info.filename), frame_info.lineno
                )
                self.w.alloc(DOMAIN_PYHEAP, stat.size, fid)
        except Exception:
            pass

    def _is_internal(self, filename: str) -> bool:
        cached = self._internal_cache.get(filename)
        if cached is not None:
            return cached
        try:
            result = os.path.abspath(filename).startswith(_SESHAT_DIR)
        except Exception:
            result = False
        self._internal_cache[filename] = result
        return result

    def _live(self) -> List[int]:
        return [f for f in self._stack if f is not None]

    def _hook(self, frame, event, arg):
        # A profiler must never crash the program it observes.
        try:
            if event == "call":
                code = frame.f_code
                cached = self._pyframe_cache.get(id(code))
                if cached is None:
                    if self._is_internal(code.co_filename):
                        cached = (None, False)
                    else:
                        is_async = bool(code.co_flags & _ASYNC_FLAGS)
                        fid = self.w.intern_frame(
                            KIND_ASYNC if is_async else KIND_PY,
                            code.co_name,
                            os.path.basename(code.co_filename),
                            frame.f_lineno,
                        )
                        cached = (fid, is_async)
                    self._pyframe_cache[id(code)] = cached
                fid, is_async = cached
                self._stack.append(fid)
                if fid is None:
                    return
                if self._mode == "calls":
                    self.w.sample(self._live())
                if is_async:
                    suspended_at = self._coro_suspend.pop(id(frame), None)
                    if suspended_at is not None:
                        self.w.await_resume(fid, self.w.event_count - suspended_at)
            elif event == "return":
                if self._stack:
                    self._stack.pop()
                code = frame.f_code
                if (code.co_flags & _ASYNC_FLAGS) and not self._is_internal(code.co_filename):
                    self._coro_suspend[id(frame)] = self.w.event_count
            elif event == "c_call":
                if self._is_internal(frame.f_code.co_filename):
                    self._stack.append(None)
                    return
                label = _c_label(arg)
                if label in _TRANSPARENT_C:
                    self._stack.append(None)
                    return
                fid = self._boundary_cache.get(label)
                if fid is None:
                    fid = self.w.intern_frame(KIND_FFI, label, "<native>", 0)
                    self._boundary_cache[label] = fid
                self._stack.append(fid)
                # In sampling mode the boundary frame on the stack is what the
                # sampler captures → a *time-weighted* boundary share (no per-call
                # event). Only `calls` mode emits a crossing edge + a sample.
                if self._mode == "calls":
                    self.w.boundary_cross(fid)
                    self.w.sample(self._live())
            elif event in ("c_return", "c_exception"):
                if self._stack:
                    self._stack.pop()
        except Exception:
            # Swallow — instrumentation errors must not propagate.
            pass


def mark_copy(from_domain: str, to_domain: str, nbytes: int) -> None:
    """Record a cross-domain copy live (e.g. a Rust→NumPy buffer handoff).

    Domains: pyheap, rustheap, mmap, numpy, arrow, tensor, gpu, nativeext.
    No-op if no recording is active or a domain name is unknown.
    """
    rec = _ACTIVE
    if rec is None:
        return
    f = DOMAIN.get(str(from_domain).lower())
    t = DOMAIN.get(str(to_domain).lower())
    if f is None or t is None:
        return
    fid = rec.w.intern_frame(KIND_FFI, "seshat_marker", "<marker>", 0)
    rec.w.copy(f, t, int(nbytes), fid)


def mark_boundary(name: str) -> None:
    """Record an explicit FFI / Py↔Rust boundary crossing live. No-op if no
    recording is active. (Most boundaries are detected automatically via
    ``sys.setprofile``; this is for ones you want to name yourself.)"""
    rec = _ACTIVE
    if rec is None:
        return
    fid = rec.w.intern_frame(KIND_FFI, str(name), "<marker>", 0)
    rec.w.boundary_cross(fid)


def _c_label(arg) -> str:
    """A readable name for a C/Rust callee, e.g. ``json.loads`` or ``builtins.len``."""
    name = getattr(arg, "__name__", None) or "<c_call>"
    module = getattr(arg, "__module__", None)
    if module:
        return f"{module}.{name}"
    # built-in methods expose __self__ for the owning type
    owner = getattr(getattr(arg, "__self__", None), "__class__", None)
    if owner is not None and getattr(owner, "__name__", None):
        return f"{owner.__name__}.{name}"
    return name


# NB: `zone` and `record` are implemented as classes (not `@contextmanager`
# generators) on purpose — their `__enter__`/`__exit__` live in *this* module,
# which the hook skips as internal, so the context-manager machinery never leaks
# into the trace. A `@contextmanager` generator runs in `contextlib.py` and would
# pollute every recording with `__enter__`/`helper` frames.


class zone:
    """Mark a pipeline stage. Samples taken inside are attributed to ``name`` by
    the analyzer's data-pipeline profiler. No-op if no recording is active.

    ::

        with seshat.record("run.seshat"):
            with seshat.zone("parse"):
                ...
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.handle = None

    def __enter__(self):
        rec = _ACTIVE
        if rec is not None:
            self.handle = rec.w.zone_start(self.name)
        return self

    def __exit__(self, *exc):
        rec = _ACTIVE
        if rec is not None and self.handle is not None:
            rec.w.zone_stop(self.handle)
        return False


class record:
    """Record the enclosed block and write ``out_path`` on exit.

    ::

        import seshat
        with seshat.record("run.seshat"):
            ...  # your Python + Rust-extension code

    Then analyze with the Rust CLI:  ``seshat analyze run.seshat``
    """

    def __init__(self, out_path: str, trace_memory: bool = True, mode: str = "sampling") -> None:
        self.out_path = out_path
        self.trace_memory = trace_memory
        self.mode = mode
        self.rec = None

    def __enter__(self):
        self.rec = Recorder()
        self.rec.start(trace_memory=self.trace_memory, mode=self.mode)
        return self.rec

    def __exit__(self, *exc):
        w = self.rec.stop()
        with open(self.out_path, "wb") as fh:
            fh.write(w.to_bytes())
        return False


def run_path(script_path: str, out_path: str, trace_memory: bool = True, mode: str = "sampling") -> int:
    """Execute a Python script under recording; write the trace; return #events."""
    with open(script_path, "r", encoding="utf-8") as fh:
        source = fh.read()
    code = compile(source, script_path, "exec")
    globals_dict = {
        "__name__": "__main__",
        "__file__": script_path,
        "__builtins__": __builtins__,
    }
    rec = Recorder()
    rec.start(trace_memory=trace_memory, mode=mode)
    try:
        exec(code, globals_dict)  # noqa: S102 — running the user's own script by design
    finally:
        w = rec.stop()
        with open(out_path, "wb") as fh:
            fh.write(w.to_bytes())
    return w.event_count
