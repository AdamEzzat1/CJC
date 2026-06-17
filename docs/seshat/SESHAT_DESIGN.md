# Seshat — Design Notes

Status: **deterministic core shipped** (2026-06-17). Live collectors + Python
bridge deferred. See `SESHAT_IMPLEMENTATION_PROMPT.md` for the full multi-phase
brief and `SESHAT_DETERMINISM.md` for the invariant list.

## 1. Goal

A cross-language causal profiler for Python/Rust systems: one merged timeline
explaining CPU, memory, copies, async stalls, GIL/lock contention, performance
variance, and thermal effects — grounded in a recorded trace, not heuristics.

## 2. Architecture

```
COLLECTION (nondeterministic, feature `collect-live`, quarantined)
   probes → canonical, logical-clock-stamped events
        │
        ▼
TRACE MODEL  — the `.seshat` format (deterministic, content-addressed)
   crates/cjc-seshat/src/trace.rs + serialize.rs + hash.rs
        │
        ▼  pure, deterministic, BTreeMap, integer reductions
ANALYSIS ENGINE  — the 12 features
   crates/cjc-seshat/src/analyze.rs + report.rs
        │
        ▼
RENDERERS  — json / text / svg (crates/cjc-seshat/src/render.rs)
```

The split mirrors `cjc-cana`'s passive-observer contract: collection is
best-effort and may drop samples; analysis is an exact function of whatever was
recorded.

## 3. The trace model (`trace.rs`)

- **`FrameKind`** — `Py | Rust | Native | FfiBoundary | AsyncTask`. The whole
  point: these coexist in one stack.
- **`OwnershipDomain`** — `PyHeap | RustHeap | Mmap | NumPy | Arrow | Tensor |
  Gpu | NativeExt`. `zero_copy_compatible(a, b)` defines avoidable copies.
- **`ThreadState`** — `Running | GilWait | LockWait | ChannelWait | IoWait |
  AsyncIdle`. Drives contention/async analyses from sample *counts*.
- **`Event`** — `Sample | Alloc | Free | Counter | ZoneStart | ZoneStop |
  Edge`. The event's logical `seq` is its index in `Trace::events` — there is no
  timestamp field by design.
- **`CausalEdge`** — `Wakeup | AwaitResume | GilHandoff | Copy | BoundaryCross`.
  These make the profile *causal*.
- Strings and frames are interned (deduped). `Trace::frame_label` resolves a
  frame to `"kind:name (file:line)"`, the key analyses group by — which is also
  what makes report hashes independent of interning-id values.

## 4. The 12 analyses (`analyze.rs`)

Each is a pure reduction over `&Trace` returning an owned report struct with a
`hash_into` method. Percentages are integer **milli-percent** (`100% = 100_000`)
so nothing floats through the hashed path.

| # | Function | Key math |
|---|----------|----------|
| 1 | `flamegraph` | merge sample stacks into a `BTreeMap`-keyed tree (sorted children → order-invariant). `sum(self_count) == total_samples`. |
| 2 | `boundary` | samples whose stack contains/leafs an `FfiBoundary` frame + `BoundaryCross` edges. |
| 3 | `copy` | aggregate `Copy` edges by `(from,to)`; `avoidable` iff zero-copy compatible. **Soundness: reported ⊆ recorded.** |
| 4 | `contention` | count `Sample.state`: GIL vs lock/channel vs io. |
| 5 | `async_stall` | per-`AsyncTask` `AsyncIdle` samples + `Wakeup`/`AwaitResume` edges. |
| 6 | `ownership` | per-domain alloc/free/live + copy transfers. **Partition: `sum(per_domain) == total`.** |
| 7 | `peak` | walk live-set over `seq`, snapshot top `(domain,frame)` contributors at the max. |
| 8 | `recommend` (report.rs) | fixed-order rule table: predicate over the report → templated message + evidence. |
| 9 | `variance` | flamegraph per run; flag frames whose share spread across runs ≥ threshold; classify cause. |
| 10 | `thermal` | `Counter` events: frequency drop ≥10% → throttle; logical-thread count > cores → oversubscription. |
| 11 | `pipeline` | attribute each sample to the innermost open zone. |
| 12 | `diff` | per-frame share delta between two traces; `diff(t,t)` → empty. |

## 5. The `.cjcl` surface (`dispatch.rs`)

A thin set of **write-only** `seshat_*` builtins emit events into a thread-local
`TraceBuilder` sink. Routed through a satellite `dispatch_seshat()` (the
`cjc-ad`/`cjc-quantum`/`cjc-locke` precedent) to avoid a `cjc-runtime →
cjc-seshat` cycle, and wired into `cjc-eval` and `cjc-mir-exec` so both
executors share one sink per thread. Zone handles come from a monotonic counter
→ identical across executors (parity).

## 6. Serialization (`serialize.rs`)

A compact `SESHAT01` binary format. `replay` is fully bounds-checked: arbitrary
bytes return `DecodeError`, never panic, and nothing pre-allocates from an
untrusted count. Round-trip preserves `content_hash`.

## 7. Live capture (feature `collect-live`)

Shipped: a minimal, zero-dependency, in-process **Rust** collector
(`collect/alloc.rs` + `collect/recorder.rs`) and the `seshat` CLI
(`src/bin/seshat.rs`).

- **`SeshatAlloc`** — a `#[global_allocator]` shim recording real heap alloc/free
  (size, `RustHeap`), made reentrancy-safe by a thread-local guard (the
  dhat/Memray technique): while recording, the allocator forwards straight to the
  system allocator without re-entering, so `alloc → record → alloc …` can neither
  recurse nor re-lock the buffer.
- **`Recorder` + `zone`** — a session with a background thread that samples the
  currently-open zone stack every interval. Sample *counts* per zone are the
  count-based time proxy; the wall-clock interval is advisory. The sampler
  records *zone* frames, not native unwound frames (zero-dep, cross-platform) —
  this is the honest limitation.
- **`mark_copy` / `mark_boundary`** — live causal-edge markers.
- **`Recorder::finish`** re-interns the captured `RawEvent` stream into a
  deterministic `Trace`. The recording is nondeterministic; the *analysis of the
  recorded trace* is byte-identical (verified: a separate default-build `seshat
  analyze` reproduces the recording run's report hash exactly).

Tested by `tests/collect.rs` (gated) for **well-formedness** — captures real
allocations, round-trips, analyzes without panic — never for timing values.

## 8. Cross-language capture — the Python recorder (`python-seshat/`)

Shipped: a **pure-stdlib Python** recorder. The key architectural insight is that
the `.seshat` *file* is the cross-language interface — the Python side only needs
to **write** the format (re-implemented in ~150 lines of Python in
`python-seshat/seshat/format.py`), so there is **no PyO3, no maturin, no Python
headers, no compilation**.

- `sys.setprofile` fires on every Python `call`/`return` **and** every
  `c_call`/`c_return` — so one hook captures real Python frames *and* the
  Python→native boundary. A PyO3 Rust function is one kind of native callee, so
  this is exactly the Py↔Rust seam.
- Bootstrap (`exec`/`eval`/`import`) and the recorder's own frames are skipped via
  a `None`-placeholder stack (keeps push/pop balanced). C calls made *by* the
  recorder's own code are also skipped — without that, interning re-enters
  mid-call and corrupts string ids (a bug found and fixed during bring-up; see
  `python-seshat/tests/test_format.py::test_recorder_captures_python_frame_and_boundary`).
- `seshat.zone(...)` emits pipeline stages, so the data-pipeline profiler works
  from Python too.
- Determinism: the call sequence of a deterministic Python program is fixed, so
  the produced trace is deterministic (call-structured, like `cProfile`). Proven
  end-to-end: a Python-recorded fixture (`crates/cjc-seshat/tests/fixtures/python_demo.seshat`)
  is analyzed by Rust in `tests/python_bridge.rs`, no Python needed at build time.

Three further capture capabilities were added on top (all **pure stdlib**, no
dependency — the recorder stays zero-dep):

- **Copy auto-discovery** (was manual `mark_copy`). A `_COPY_FUNCS` registry maps
  copy-inducing native callees (`ndarray.copy`, `np.ascontiguousarray`,
  `Tensor.clone`, Arrow `to_pandas`/`combine_chunks`, …) to a `(from, to)`
  ownership-domain guess; the `c_call` hook emits a `Copy` edge when it sees one.
  Byte counts are best-effort: **exact** when the callee is a bound method (the
  source operand is its `__self__`, so `.nbytes` is readable), **0 = unknown**
  for free functions whose operand the hook cannot see. Heuristic and deliberately
  conservative — ambiguous builtins like `bytes`/`bytearray` (alloc vs copy) are
  excluded to keep the copy detector's "no false positives" contract.
- **Multi-thread capture.** `threading.setprofile` installs the hook on every
  future thread; stacks are per-OS-thread (`get_ident()` → list), and the sampler
  emits one Sample per live thread per tick, stamped with a stable logical thread
  id (main = 0). The `TraceWriter` gained an `RLock` so concurrent interning from
  many threads can't corrupt the tables (the lock never changes emitted bytes, so
  single-threaded output is unaffected). The sampler thread is started *before*
  `threading.setprofile`, so it is never hooked; and the `threading.setprofile`
  bootstrap call is made with the hook suspended so it leaves no frame in the
  trace (keeping calls-mode fixtures byte-identical).
- **GIL-wait detection — heuristic.** With multi-thread sampling, a thread whose
  leaf frame is frozen across consecutive ticks while another thread progressed is
  labelled `GilWait` (`_classify_states`). **This is approximate, not an exact
  GIL-acquisition signal** (CPython exposes none from pure Python); the report's
  `SES-GIL-BOUND` recommendation and the README say so explicitly. A genuinely
  running thread in a tight call-free loop can be misread — accepted, and labelled.

## 9. Still deferred and why

- **Native Rust-frame unwinding inside the extension** (Gap D1) — the Python side
  sees the boundary, not the Rust functions past it. Decision pending: manual
  `collect::zone` brackets + a `seshat merge` step (zero-dep) vs the `backtrace`
  crate for automatic unwinding (a dependency, to be scoped strictly to
  `collect-live`). Not yet taken — needs the merge/correlation-token design first.
- **Thermal / perf counters** (Gap D2, thermal mode's live data) — no portable
  stdlib source. Decision pending: an optional Python extra (`seshat[thermal]` →
  `psutil.cpu_freq()`) or a Linux-only `perf_event` reader behind `collect-live`.
  Either is an opt-in dependency, never a core requirement.
- **Exact (non-heuristic) GIL detection** — needs a C extension or out-of-process
  interpreter-state read; deferred pending an explicit waiver of the zero-dep rule.
- **Tokio `tracing` bridge**, and **time-weighted Python sampling** refinements —
  straightforward extensions.

Decisions recorded: Gaps A (copy auto-discovery), B (multi-thread), and C (GIL
heuristic) were all closed **with zero new dependencies** (pure stdlib). The two
remaining gaps (D1 native unwinding, D2 thermal) each *want* a dependency and are
left open with the decision deferred, per the zero-dep-by-default rule. None of
the shipped work changes any gated output; it only widens what can be *captured*.

## 10. Dogfooding target

The repo's own `python/` (`cjc-locke-py`) is a live PyO3/maturin Python↔Rust
seam. With the Python recorder now in place, profiling it under `seshat.record`
would show the `cjc_locke` PyO3 calls as boundary crossings directly — the
natural next acceptance workload (structure asserted; timing advisory).
