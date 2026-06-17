"""Pure-Python tests for the writer + recorder. No Rust needed.

Run:  cd python-seshat && python -m pytest tests/   (or  python tests/test_format.py)
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seshat.format import (  # noqa: E402
    TraceWriter,
    KIND_PY,
    KIND_FFI,
    KIND_ASYNC,
    DOMAIN,
    STATE_RUNNING,
    STATE_GIL_WAIT,
)
from seshat.recorder import Recorder, mark_copy, _estimate_copy_bytes  # noqa: E402


def _sample_threads_states(w):
    """Decode every Sample event → list of ``(thread_id, state)``.

    Sample = ``u8 tag=0, u32 thread, u8 state, u64 len, u32*len frames``."""
    out = []
    for ev in w._events:
        if ev and ev[0] == 0:
            (thread,) = struct.unpack_from("<I", ev, 1)
            out.append((thread, ev[5]))
    return out


def _copy_edges(w):
    """Decode every auto/manual Copy edge in a writer's event stream.

    A Copy event is ``u8 tag=6, u8 kind=3, u8 from, u8 to, u64 bytes, u32 frame``
    (see ``TraceWriter.copy``). Returns a list of ``(from_tag, to_tag, bytes)``."""
    out = []
    for ev in w._events:
        if len(ev) >= 4 and ev[0] == 6 and ev[1] == 3:
            (nbytes,) = struct.unpack_from("<Q", ev, 4)
            out.append((ev[2], ev[3], nbytes))
    return out


def _decode_header(data):
    """Minimal reader: returns (n_strings, strings, n_frames, frames)."""
    assert data[:8] == b"SESHAT01"
    p = 8
    p += 16  # wall_ns + run_id
    (n_strings,) = struct.unpack_from("<Q", data, p)
    p += 8
    strings = []
    for _ in range(n_strings):
        (ln,) = struct.unpack_from("<Q", data, p)
        p += 8
        strings.append(data[p : p + ln].decode())
        p += ln
    (n_frames,) = struct.unpack_from("<Q", data, p)
    p += 8
    frames = []
    for _ in range(n_frames):
        kind = data[p]
        p += 1
        name, file, line = struct.unpack_from("<III", data, p)
        p += 12
        frames.append((kind, strings[name], strings[file], line))
    return n_strings, strings, n_frames, frames


def test_writer_round_trips_structure():
    w = TraceWriter(run_id=7)
    f = w.intern_frame(KIND_PY, "main", "app.py", 10)
    b = w.intern_frame(KIND_FFI, "ext.call", "<native>", 0)
    w.sample([f])
    w.boundary_cross(b)
    w.sample([f, b])
    data = w.to_bytes()

    _, strings, _, frames = _decode_header(data)
    assert "main" in strings and "ext.call" in strings
    assert (KIND_PY, "main", "app.py", 10) in frames
    assert (KIND_FFI, "ext.call", "<native>", 0) in frames


def test_interning_dedups_and_ids_are_stable():
    w = TraceWriter()
    a = w.intern_str("x")
    b = w.intern_str("x")
    assert a == b
    f1 = w.intern_frame(KIND_PY, "f", "a.py", 1)
    f2 = w.intern_frame(KIND_PY, "f", "a.py", 1)
    assert f1 == f2
    assert w.intern_frame(KIND_PY, "f", "a.py", 2) != f1


def test_recorder_captures_python_frame_and_boundary():
    import json

    def work():
        total = 0
        for i in range(4):
            total += json.loads('{"n": %d}' % i)["n"]
        return total

    rec = Recorder()
    rec.start()
    work()
    w = rec.stop()

    kinds = {fr[0] for fr in w._frames}
    names = {w._strings[fr[1]] for fr in w._frames}
    assert KIND_PY in kinds, "captured a Python frame"
    assert KIND_FFI in kinds, "captured a native boundary frame"
    assert "work" in names, "the user function appears, correctly named"
    # the re-entrancy bug would have mislabeled `work`; guard against regression
    work_frames = [fr for fr in w._frames if fr[0] == KIND_PY and w._strings[fr[1]] == "work"]
    assert len(work_frames) == 1


def test_tracemalloc_captures_pyheap_allocs():
    rec = Recorder()
    rec.start(trace_memory=True, mode="calls")  # end-snapshot path interns <pyalloc>
    data = [bytes(128) for _ in range(2000)]  # real Python-heap allocations
    w = rec.stop()
    keep = len(data)  # keep `data` alive past the snapshot
    assert keep == 2000
    pyalloc = [fr for fr in w._frames if fr[0] == KIND_PY and w._strings[fr[1]] == "<pyalloc>"]
    assert pyalloc, "expected tracemalloc PyHeap allocation frames"


def test_sampling_mode_is_far_lighter_than_calls():
    """The headline overhead fix: sampling emits orders of magnitude fewer
    events than per-call recording for the same work."""

    def helper(x):
        return x * x

    def work():
        total = 0
        for i in range(30000):
            total += helper(i)
        return total

    rc = Recorder()
    rc.start(mode="calls")
    work()
    wc = rc.stop()

    rs = Recorder()
    rs.start(mode="sampling", interval_ms=2)
    work()
    ws = rs.stop()

    assert wc.event_count > 1000, f"calls mode should emit one event per call, got {wc.event_count}"
    assert ws.event_count * 5 < wc.event_count, (
        f"sampling should be far lighter: sampling={ws.event_count} calls={wc.event_count}"
    )


def test_mark_copy_records_a_copy():
    rec = Recorder()
    rec.start()  # sets the active recorder so mark_copy reaches it
    mark_copy("rustheap", "numpy", 4096)
    w = rec.stop()
    assert any(w._strings[fr[1]] == "seshat_marker" for fr in w._frames), "copy marker frame"


def test_copy_estimate_bytes_bound_vs_free():
    """Byte estimation: exact via a bound method's __self__, 0 for free funcs."""

    class _FakeArr:
        nbytes = 4096

    class _FakeBoundMethod:  # mimics e.g. ndarray.copy / Tensor.clone
        __self__ = _FakeArr()

    assert _estimate_copy_bytes(_FakeBoundMethod()) == 4096
    # a free function (no __self__) → operand invisible to the hook → 0
    assert _estimate_copy_bytes(lambda: None) == 0

    # older-torch fallback: element_size() * nelement()
    class _FakeTensorMeth:
        class _T:
            def element_size(self):
                return 4

            def nelement(self):
                return 100

        __self__ = _T()

    assert _estimate_copy_bytes(_FakeTensorMeth()) == 400

    # buffer-like fallback: len()
    class _FakeBufMeth:
        __self__ = b"abcdef"

    assert _estimate_copy_bytes(_FakeBufMeth()) == 6


def test_copy_autodiscovery_registry_with_fake_callee():
    """`_maybe_emit_copy` emits exactly one Copy edge for a registered callee,
    with the registry's domains and the estimated size — and nothing for an
    unregistered one. No NumPy required (a fake bound method drives it)."""

    class _FakeArr:
        nbytes = 4096

    class _FakeNdarrayCopy:
        __self__ = _FakeArr()

    rec = Recorder()
    fid = rec.w.intern_frame(KIND_FFI, "ndarray.copy", "<native>", 0)
    assert rec._maybe_emit_copy("ndarray.copy", _FakeNdarrayCopy(), fid) is True
    assert rec._maybe_emit_copy("totally.not.a.copy", _FakeNdarrayCopy(), fid) is False

    copies = _copy_edges(rec.w)
    assert len(copies) == 1, f"exactly one auto Copy edge, got {copies}"
    from_tag, to_tag, nbytes = copies[0]
    assert from_tag == DOMAIN["numpy"] and to_tag == DOMAIN["numpy"]
    assert nbytes == 4096


def test_copy_autodiscovery_numpy_live():
    """End-to-end: recording real NumPy copy-inducing calls auto-emits Copy
    edges, with an exact size for the bound-method `.copy()`. Skipped when NumPy
    is unavailable (the feature itself never imports numpy)."""
    try:
        import numpy as np
    except Exception:
        print("  (skipped numpy live copy test — numpy not installed)")
        return

    def work():
        a = np.zeros((64, 64), dtype=np.float64)  # 64*64*8 = 32768 bytes
        b = np.ascontiguousarray(a)               # free fn → copy, size unknown(0)
        c = b.copy()                              # bound method → exact nbytes
        return float(c.sum())

    rec = Recorder()
    rec.start(mode="calls")
    work()
    w = rec.stop()

    sizes = [nbytes for (_, _, nbytes) in _copy_edges(w)]
    assert sizes, "expected at least one auto-detected NumPy Copy edge"
    assert 32768 in sizes, f"expected exact 32768-byte copy from ndarray.copy, got {sizes}"


def test_multithread_capture_distinct_thread_ids():
    """Gap B: with `threading.setprofile` + per-thread stacks, the sampler emits
    Samples for worker threads too, each stamped with its own stable logical id
    (main = 0). Asserts ≥2 distinct logical thread ids appear."""
    import threading as _t

    def step(x):
        return (x * 1315423911 + 12345) & 0xFFFFFFFF

    def busy(n):
        acc = 0
        for i in range(n):
            acc = step(acc + i)  # function calls → stack churn the sampler sees
        return acc

    rec = Recorder()
    rec.start(mode="sampling", interval_ms=1)
    workers = [_t.Thread(target=busy, args=(400000,)) for _ in range(2)]
    for th in workers:
        th.start()
    for th in workers:
        th.join()
    w = rec.stop()

    thread_ids = {thr for (thr, _s) in _sample_threads_states(w)}
    assert len(thread_ids) >= 2, f"expected >=2 logical thread ids, got {sorted(thread_ids)}"


def test_gil_classify_states_unit():
    """Gap C heuristic logic, exercised deterministically (no threads/timing).

    `_classify_states` takes a tick = list of (ident, logical_id, live_stack)."""
    rec = Recorder()
    # Tick 1 — first sighting of both threads → Running (no history yet).
    s1 = rec._classify_states([(101, 0, [9, 1]), (202, 1, [9, 2])])
    assert s1 == {101: STATE_RUNNING, 202: STATE_RUNNING}
    # Tick 2 — thread 101 progressed (leaf 1→3); thread 202 frozen (leaf 2) while
    # another progressed → GilWait.
    s2 = rec._classify_states([(101, 0, [9, 3]), (202, 1, [9, 2])])
    assert s2[101] == STATE_RUNNING
    assert s2[202] == STATE_GIL_WAIT
    # Tick 3 — nobody progressed (both frozen) → no GilWait (can't attribute).
    s3 = rec._classify_states([(101, 0, [9, 3]), (202, 1, [9, 2])])
    assert s3 == {101: STATE_RUNNING, 202: STATE_RUNNING}

    # A single active thread can never be GIL-starved → always Running.
    solo = Recorder()
    solo._classify_states([(101, 0, [9, 1])])
    assert solo._classify_states([(101, 0, [9, 1])]) == {101: STATE_RUNNING}


def test_gil_wait_heuristic_live_contention():
    """Gap C end-to-end (heuristic): two CPU-bound threads contend for the GIL;
    some of the starved thread's samples should be labelled GilWait. Loose — it
    is an approximation, and the deterministic proof is the unit test above."""
    import threading as _t

    def step(x):
        return (x * 2654435761 + 1) & 0xFFFFFFFFFFFFFFFF

    def churn(n):
        acc = 0
        for i in range(n):
            acc = step(acc + i)
        return acc

    rec = Recorder()
    rec.start(mode="sampling", interval_ms=1)
    workers = [_t.Thread(target=churn, args=(1500000,)) for _ in range(2)]
    for th in workers:
        th.start()
    for th in workers:
        th.join()
    w = rec.stop()

    states = [s for (_thr, s) in _sample_threads_states(w)]
    assert STATE_GIL_WAIT in states, (
        f"expected at least one heuristic GilWait sample under 2-thread "
        f"contention; saw {len(states)} samples, states={set(states)}"
    )


def test_async_frames_are_tagged():
    import asyncio

    async def coro():
        await asyncio.sleep(0)
        return 1

    rec = Recorder()
    rec.start()
    asyncio.run(coro())
    w = rec.stop()
    kinds = {fr[0] for fr in w._frames}
    assert KIND_ASYNC in kinds, "coroutine frames should be tagged async"
    names = {w._strings[fr[1]] for fr in w._frames if fr[0] == KIND_ASYNC}
    assert "coro" in names


if __name__ == "__main__":
    test_writer_round_trips_structure()
    test_interning_dedups_and_ids_are_stable()
    test_recorder_captures_python_frame_and_boundary()
    test_tracemalloc_captures_pyheap_allocs()
    test_sampling_mode_is_far_lighter_than_calls()
    test_mark_copy_records_a_copy()
    test_copy_estimate_bytes_bound_vs_free()
    test_copy_autodiscovery_registry_with_fake_callee()
    test_copy_autodiscovery_numpy_live()
    test_multithread_capture_distinct_thread_ids()
    test_gil_classify_states_unit()
    test_gil_wait_heuristic_live_contention()
    test_async_frames_are_tagged()
    print("ok: all python-seshat format/recorder tests passed")
