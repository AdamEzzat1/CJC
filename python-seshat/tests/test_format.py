"""Pure-Python tests for the writer + recorder. No Rust needed.

Run:  cd python-seshat && python -m pytest tests/   (or  python tests/test_format.py)
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seshat.format import TraceWriter, KIND_PY, KIND_FFI, KIND_ASYNC  # noqa: E402
from seshat.recorder import Recorder, mark_copy  # noqa: E402


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
    test_async_frames_are_tagged()
    print("ok: all python-seshat format/recorder tests passed")
