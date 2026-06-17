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

Beyond the per-thread stack + boundary, the recorder also:

* **Auto-discovers cross-domain copies** (Gap A): copy-inducing native callees
  (``ndarray.copy``, ``np.ascontiguousarray``, ``Tensor.clone``, Arrow
  ``to_pandas`` …) emit a ``Copy`` edge with a best-effort byte size — no manual
  ``mark_copy`` needed. See ``_COPY_FUNCS`` / ``_estimate_copy_bytes``.
* **Captures all threads** (Gap B): ``threading.setprofile`` installs the hook on
  every thread; stacks are per-OS-thread and the sampler emits one Sample per
  live thread per tick, stamped with a stable logical thread id (main = 0).
* **Infers GIL-wait** (Gap C, *heuristic*): a thread frozen at the same leaf
  across ticks while another progresses is labelled ``GilWait``. Approximate —
  there is no exact pure-Python GIL signal. See ``_classify_states``.

Wall-clock total is recorded as advisory metadata only (excluded from the
analyzer's content hash).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import List

from .format import (
    TraceWriter,
    DOMAIN,
    DOMAIN_PYHEAP,
    KIND_ASYNC,
    KIND_FFI,
    KIND_PY,
    STATE_GIL_WAIT,
    STATE_RUNNING,
)

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

# Native callees that *materialize a buffer* — i.e. induce a copy. The `c_call`
# hook already sees every one of these, so when a call's `_c_label` is a key
# here we emit a `Copy` edge automatically; no manual `mark_copy` needed. The
# value is the (from_domain, to_domain) ownership guess.
#
# These are HEURISTICS, deliberately conservative to honour the copy detector's
# "no false positives" contract:
#   * Only calls that copy *by definition* are listed. Ambiguous builtins like
#     `bytes`/`bytearray` (which allocate for an int arg but copy for a buffer
#     arg, indistinguishable from the hook) are intentionally excluded.
#   * Some listed calls can be runtime no-ops (e.g. `ascontiguousarray` on an
#     already-contiguous array); they are surfaced as candidates.
#   * Byte counts are best-effort (see `_estimate_copy_bytes`): exact for bound
#     methods, 0 ("unknown") for free functions whose operand the hook can't see.
_COPY_FUNCS = {
    # NumPy — materialize / re-layout into a NumPy buffer.
    "numpy.array": ("pyheap", "numpy"),
    "numpy.asarray": ("pyheap", "numpy"),
    "numpy.ascontiguousarray": ("numpy", "numpy"),
    "numpy.asfortranarray": ("numpy", "numpy"),
    "ndarray.copy": ("numpy", "numpy"),
    "ndarray.astype": ("numpy", "numpy"),
    "ndarray.tobytes": ("numpy", "pyheap"),
    # PyTorch — clone / contiguous materialize a tensor buffer.
    "Tensor.clone": ("tensor", "tensor"),
    "Tensor.contiguous": ("tensor", "tensor"),
    "Tensor.numpy": ("tensor", "numpy"),
    # Apache Arrow — combine_chunks / to_pandas materialize across the seam.
    "Table.to_pandas": ("arrow", "pyheap"),
    "ChunkedArray.combine_chunks": ("arrow", "arrow"),
}

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
        # Per-thread call stacks, keyed by OS thread id (``threading.get_ident()``).
        # Each entry is a frame id (recorded) or None (a skipped bootstrap /
        # recorder frame); the None placeholders keep push/pop balanced. Each OS
        # thread mutates only its own stack; the sampler slice-copies each
        # (GIL-atomic), so no per-stack lock is needed.
        self._stacks = {}  # ident -> List
        # ident -> stable logical u32 thread id, assigned in first-seen order.
        # The main thread is always 0, so single-threaded traces are unchanged.
        self._thread_logical = {}
        self._next_thread_id = 0
        self._thread_lock = threading.Lock()  # guards first-seen id assignment only
        self._main_ident = None
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
        # When True the hook is a no-op (no push/pop, no events) — used to make
        # the recorder's own `threading.setprofile` bootstrap call invisible.
        self._suspend_hook = False
        # GIL heuristic (Gap C): previous sampler tick's leaf frame per thread.
        # Written only by the sampler thread, so no lock is needed.
        self._prev_tops = {}  # ident -> frame id

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
        # Pre-register the main thread as logical id 0 so single-threaded traces
        # are byte-identical to the pre-multithread recorder.
        self._main_ident = threading.get_ident()
        self._stacks[self._main_ident] = []
        self._thread_logical[self._main_ident] = 0
        self._next_thread_id = 1
        self._t0 = time.perf_counter_ns()
        sys.setprofile(self._hook)
        if self._mode == "sampling":
            self._sampling = True
            self._sampler = threading.Thread(
                target=self._sample_loop, name="seshat-sampler", daemon=True
            )
            # Start the sampler BEFORE threading.setprofile so the sampler thread
            # itself is never hooked — it must stay invisible to the profiler it
            # drives (it samples other threads' stacks; it has no stack of its own
            # in `self._stacks`).
            self._sampler.start()
        # Install the hook on USER threads spawned *after* start() too (setprofile
        # is per-thread). `threading.setprofile` registers it for every future
        # `threading.Thread`, giving multi-thread capture (Gap B). It is a
        # *pure-Python* function (unlike the transparent C `sys.setprofile`), so
        # suspend the hook across it — otherwise its own `call` frame would be
        # recorded into every trace and perturb the deterministic calls-mode path.
        self._suspend_hook = True
        try:
            threading.setprofile(self._hook)
        finally:
            self._suspend_hook = False

    def stop(self) -> TraceWriter:
        global _ACTIVE
        self._sampling = False
        sys.setprofile(None)
        threading.setprofile(None)
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
        hook). Each tick: emit one time-weighted Sample per live thread (Gap B —
        multi-thread), classifying each thread's state with the GIL heuristic
        (Gap C), plus a process-global PyHeap alloc/free delta attributed to the
        main thread's leaf. Each thread's stack is read via an atomic slice copy
        — safe under the GIL without a per-stack lock."""
        try:
            import tracemalloc
        except Exception:
            tracemalloc = None
        while self._sampling:
            time.sleep(self._interval)
            try:
                # Snapshot every thread's live stack this tick. `list()` first:
                # the dict may gain a key when a new thread is first seen, and
                # iterating it directly would raise "changed size during
                # iteration".
                tick = []  # (ident, tid, live_stack)
                for ident, stack in list(self._stacks.items()):
                    live = [f for f in stack[:] if f is not None]  # atomic copy
                    if live:
                        tid = self._thread_logical.get(ident, 0)
                        tick.append((ident, tid, live))
                states = self._classify_states(tick)  # Gap C: GIL-wait heuristic
                for ident, tid, live in tick:
                    self.w.sample(live, thread=tid, state=states[ident])
                if self._trace_memory and tracemalloc is not None and tracemalloc.is_tracing():
                    # Memory is process-global; attribute the delta to the main
                    # thread's leaf frame (preserves single-thread behavior).
                    main = self._stacks.get(self._main_ident)
                    live_main = [f for f in main[:] if f is not None] if main else []
                    top = live_main[-1] if live_main else self._sampled_frame
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

    @staticmethod
    def _live(stack) -> List[int]:
        return [f for f in stack if f is not None]

    def _cur_thread(self):
        """Return ``(stack, logical_id)`` for the calling OS thread, registering
        it on first sight (Gap B). The fast path is a single dict lookup; the
        lock is taken only the first time a thread is seen, so the per-event
        hot path stays lock-free. Safe to call from inside the hook — the
        ``get_ident``/dict C-calls it makes fire no nested ``c_call`` (CPython
        suppresses the profile callback while it runs)."""
        ident = threading.get_ident()
        st = self._stacks.get(ident)
        if st is not None:
            return st, self._thread_logical[ident]
        with self._thread_lock:
            st = self._stacks.get(ident)
            if st is None:
                tid = self._next_thread_id
                self._next_thread_id += 1
                st = []
                self._thread_logical[ident] = tid
                self._stacks[ident] = st
            else:
                tid = self._thread_logical[ident]
        return st, tid

    def _hook(self, frame, event, arg):
        # A profiler must never crash the program it observes.
        if self._suspend_hook:
            return  # no push/pop, no events → balanced; used during bootstrap
        try:
            st, tid = self._cur_thread()
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
                st.append(fid)
                if fid is None:
                    return
                if self._mode == "calls":
                    self.w.sample(self._live(st), thread=tid)
                if is_async:
                    suspended_at = self._coro_suspend.pop(id(frame), None)
                    if suspended_at is not None:
                        self.w.await_resume(fid, self.w.event_count - suspended_at)
            elif event == "return":
                if st:
                    st.pop()
                code = frame.f_code
                if (code.co_flags & _ASYNC_FLAGS) and not self._is_internal(code.co_filename):
                    self._coro_suspend[id(frame)] = self.w.event_count
            elif event == "c_call":
                if self._is_internal(frame.f_code.co_filename):
                    st.append(None)
                    return
                label = _c_label(arg)
                if label in _TRANSPARENT_C:
                    st.append(None)
                    return
                fid = self._boundary_cache.get(label)
                if fid is None:
                    fid = self.w.intern_frame(KIND_FFI, label, "<native>", 0)
                    self._boundary_cache[label] = fid
                st.append(fid)
                # Auto-discover cross-domain copies (Gap A). A known copy-inducing
                # native callee emits a `Copy` edge attributed to this boundary
                # frame — in BOTH modes, because a Copy edge is a discrete,
                # deterministic event (tied to the call), not a timing sample.
                if label in _COPY_FUNCS:
                    self._maybe_emit_copy(label, arg, fid)
                # In sampling mode the boundary frame on the stack is what the
                # sampler captures → a *time-weighted* boundary share (no per-call
                # event). Only `calls` mode emits a crossing edge + a sample.
                if self._mode == "calls":
                    self.w.boundary_cross(fid)
                    self.w.sample(self._live(st), thread=tid)
            elif event in ("c_return", "c_exception"):
                if st:
                    st.pop()
        except Exception:
            # Swallow — instrumentation errors must not propagate.
            pass

    def _classify_states(self, tick):
        """GIL-wait heuristic (Gap C) — returns ``{ident: state}`` for this tick.

        **This is a heuristic, not an exact measurement.** CPython exposes no
        pure-Python signal for GIL acquisition, so we infer it: across two
        consecutive sampler ticks, if *some* thread made progress (its leaf frame
        changed) while *another* thread's leaf is frozen at the frame it had last
        tick, the frozen thread is most likely blocked waiting for the GIL →
        labelled :data:`GilWait`. With fewer than two active threads there is no
        contention, so everything stays :data:`Running`. A genuinely-running
        thread in a tight call-free loop can be misread as frozen — the report
        labels this share as approximate."""
        cur_tops = {ident: live[-1] for (ident, _tid, live) in tick}
        progressed = any(
            ident in self._prev_tops and self._prev_tops[ident] != top
            for ident, top in cur_tops.items()
        )
        multi = len(tick) >= 2
        states = {}
        for ident, top in cur_tops.items():
            frozen = ident in self._prev_tops and self._prev_tops[ident] == top
            if multi and progressed and frozen:
                states[ident] = STATE_GIL_WAIT
            else:
                states[ident] = STATE_RUNNING
        self._prev_tops = cur_tops
        return states

    def _maybe_emit_copy(self, label: str, arg, frame_id: int) -> bool:
        """Emit a `Copy` edge for a copy-inducing native callee (Gap A).

        Returns True iff an edge was emitted. Safe to call from inside the hook:
        CPython suppresses the profile callback while it runs, so the
        ``len()``/``element_size()`` probes inside :func:`_estimate_copy_bytes`
        fire no nested ``c_call`` events. Factored out of ``_hook`` so it can be
        unit-tested directly with a fake callee (no NumPy required)."""
        domains = _COPY_FUNCS.get(label)
        if domains is None:
            return False
        try:
            f_dom = DOMAIN[domains[0]]
            t_dom = DOMAIN[domains[1]]
            nbytes = _estimate_copy_bytes(arg)
            self.w.copy(f_dom, t_dom, int(nbytes), frame_id)
            return True
        except Exception:
            return False


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


def _estimate_copy_bytes(arg) -> int:
    """Best-effort byte size of the buffer a copy-inducing native call moved.

    ``sys.setprofile``'s ``c_call`` hook is handed the *callee*, never its
    arguments — so for a **free function** (``numpy.array(x)``) the source
    operand is invisible and we return 0 ("unknown", recorded honestly). For a
    **bound method** (``ndarray.copy()``, ``Tensor.clone()``, ``Table.to_pandas()``)
    the callee's ``__self__`` *is* the source operand, so we can read its exact
    size: ``.nbytes`` (NumPy / Arrow / newer torch), else ``element_size() *
    nelement()`` (older torch), else ``len()`` (buffer-like). Every probe is
    guarded — a profiler must never raise into the program it observes."""
    operand = getattr(arg, "__self__", None)
    if operand is None:
        return 0  # free function: operand not visible to the profile hook
    try:
        nbytes = getattr(operand, "nbytes", None)
        if isinstance(nbytes, int) and nbytes >= 0:
            return nbytes
    except Exception:
        pass
    try:
        es, ne = operand.element_size(), operand.nelement()
        if isinstance(es, int) and isinstance(ne, int) and es >= 0 and ne >= 0:
            return es * ne
    except Exception:
        pass
    try:
        n = len(operand)
        if isinstance(n, int) and n >= 0:
            return n
    except Exception:
        pass
    return 0


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
