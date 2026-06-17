"""Pure-Python writer for the ``.seshat`` binary trace format (``SESHAT01``).

This is the *only* thing the Python side needs: the trace file is the
cross-language interface. A Python program records frames/events with
:class:`TraceWriter`, writes the bytes, and the Rust ``cjc-seshat`` analyzer
reads them — no PyO3, no maturin, no compilation, no Python headers.

The layout here mirrors ``crates/cjc-seshat/src/serialize.rs`` exactly. All
integers are little-endian; strings are length-prefixed UTF-8. The Rust reader
bounds-checks every id, so the writer must only emit valid ones (the interning
methods below guarantee that).
"""

from __future__ import annotations

import struct
import threading
from typing import Dict, List, Tuple

MAGIC = b"SESHAT01"

# Frame kinds — must match cjc_seshat::trace::FrameKind::tag()
KIND_PY = 0
KIND_RUST = 1
KIND_NATIVE = 2
KIND_FFI = 3
KIND_ASYNC = 4

# Thread states — must match ThreadState::tag()
STATE_RUNNING = 0
STATE_GIL_WAIT = 1
STATE_LOCK_WAIT = 2
STATE_CHANNEL_WAIT = 3
STATE_IO_WAIT = 4
STATE_ASYNC_IDLE = 5

# Ownership domains — must match OwnershipDomain::tag()
DOMAIN_PYHEAP = 0
DOMAIN = {
    "pyheap": 0,
    "rustheap": 1,
    "mmap": 2,
    "numpy": 3,
    "arrow": 4,
    "tensor": 5,
    "gpu": 6,
    "nativeext": 7,
}


def _u8(v: int) -> bytes:
    return struct.pack("<B", v & 0xFF)


def _u32(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF)


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


class TraceWriter:
    """Builds a ``.seshat`` byte stream. Strings and frames are interned.

    Thread-safe: every mutating method holds an internal re-entrant lock, so the
    multi-thread recorder (a per-thread ``sys.setprofile`` hook on each worker
    plus the sampler thread) can intern and emit events concurrently without
    corrupting the interning tables. The lock only serializes access — it never
    changes the emitted bytes, so single-threaded output is unaffected. (``RLock``
    rather than ``Lock`` because ``intern_frame`` calls ``intern_str``.)"""

    def __init__(self, run_id: int = 0) -> None:
        self._strings: List[str] = []
        self._str_idx: Dict[str, int] = {}
        self._frames: List[Tuple[int, int, int, int]] = []  # (kind, name, file, line)
        self._frame_idx: Dict[Tuple[int, int, int, int], int] = {}
        self._events: List[bytes] = []
        self.wall_ns: int = 0
        self.run_id: int = run_id
        self._next_handle: int = 1
        self.event_count: int = 0
        self._lock = threading.RLock()

    # ── interning ──
    def intern_str(self, s: str) -> int:
        with self._lock:
            i = self._str_idx.get(s)
            if i is not None:
                return i
            # Append first, then read the index — robust even if a nested call
            # mutates the table before this returns.
            self._strings.append(s)
            i = len(self._strings) - 1
            self._str_idx[s] = i
            return i

    def intern_frame(self, kind: int, name: str, file: str, line: int) -> int:
        with self._lock:
            key = (kind, self.intern_str(name), self.intern_str(file), int(line) & 0xFFFFFFFF)
            i = self._frame_idx.get(key)
            if i is not None:
                return i
            self._frames.append(key)
            i = len(self._frames) - 1
            self._frame_idx[key] = i
            return i

    # ── events (each appends its pre-encoded bytes) ──
    def sample(self, stack: List[int], thread: int = 0, state: int = STATE_RUNNING) -> None:
        buf = _u8(0) + _u32(thread) + _u8(state) + _u64(len(stack))
        for f in stack:
            buf += _u32(f)
        with self._lock:
            self._events.append(buf)
            self.event_count += 1

    def alloc(self, domain: int, nbytes: int, frame: int) -> None:
        with self._lock:
            self._events.append(_u8(1) + _u8(domain) + _u64(nbytes) + _u32(frame))
            self.event_count += 1

    def free(self, domain: int, nbytes: int, frame: int) -> None:
        with self._lock:
            self._events.append(_u8(2) + _u8(domain) + _u64(nbytes) + _u32(frame))
            self.event_count += 1

    def zone_start(self, name: str) -> int:
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._events.append(_u8(4) + _u32(self.intern_str(name)) + _u64(handle))
            self.event_count += 1
            return handle

    def zone_stop(self, handle: int) -> None:
        with self._lock:
            self._events.append(_u8(5) + _u64(handle))
            self.event_count += 1

    def copy(self, from_domain: int, to_domain: int, nbytes: int, frame: int) -> None:
        # Edge tag 6, edge-kind tag 3 (Copy)
        buf = _u8(6) + _u8(3) + _u8(from_domain) + _u8(to_domain) + _u64(nbytes) + _u32(frame)
        with self._lock:
            self._events.append(buf)
            self.event_count += 1

    def boundary_cross(self, frame: int) -> None:
        # Edge tag 6, edge-kind tag 4 (BoundaryCross)
        with self._lock:
            self._events.append(_u8(6) + _u8(4) + _u32(frame))
            self.event_count += 1

    def await_resume(self, task_frame: int, waited_ticks: int) -> None:
        # Edge tag 6, edge-kind tag 1 (AwaitResume): a coroutine resumed after
        # being parked `waited_ticks` logical ticks at an `await`.
        with self._lock:
            self._events.append(_u8(6) + _u8(1) + _u32(task_frame) + _u64(waited_ticks))
            self.event_count += 1

    # ── finalize ──
    def to_bytes(self) -> bytes:
        with self._lock:
            return self._to_bytes_locked()

    def _to_bytes_locked(self) -> bytes:
        out = bytearray()
        out += MAGIC
        out += _u64(self.wall_ns)
        out += _u64(self.run_id)
        out += _u64(len(self._strings))
        for s in self._strings:
            out += _string(s)
        out += _u64(len(self._frames))
        for (kind, name, file, line) in self._frames:
            out += _u8(kind) + _u32(name) + _u32(file) + _u32(line)
        out += _u64(len(self._events))
        for ev in self._events:
            out += ev
        return bytes(out)
