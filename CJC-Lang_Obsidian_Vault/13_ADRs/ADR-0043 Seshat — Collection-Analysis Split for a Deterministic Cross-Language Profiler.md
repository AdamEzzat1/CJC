# ADR-0043 Seshat — Collection/Analysis Split for a Deterministic Cross-Language Profiler

- **Status:** Accepted (2026-06-17)
- **Crate:** `cjc-seshat` (new, publish = false) + wiring into `cjc-eval`, `cjc-mir-exec`
- **Companion docs:** `docs/seshat/SESHAT_IMPLEMENTATION_PROMPT.md`, `docs/seshat/SESHAT_DESIGN.md`, `docs/seshat/SESHAT_DETERMINISM.md`; precedent [[ADR-0014 MIR Analysis Infrastructure]] (cana passive-observer pattern)
- **Tests:** 51 new (17 unit + 17 integration + 8 proptest + 4 bolero + 4 executor-parity + 1 doctest), zero regressions across the workspace build + abng (629) + executor units (56).

## Context

CJC-Lang wanted a profiler that understands the seam between Python and Rust —
the awkward, under-tooled boundary where modern projects glue Rust speed onto
Python via PyO3/maturin. Existing tools (perf/samply/pprof on the Rust side,
Scalene/Memray/py-spy on the Python side) each see only one language. The
unique, resume-worthy capability is **one merged timeline** spanning Python,
Rust, native, async, and FFI frames, plus a *causal* explanation (copies,
GIL/lock contention, ownership transfer, peak attribution) rather than a flat
self-time list.

The hard constraint: **a profiler measures time, and time is nondeterministic,
but CJC-Lang's prime directive is bit-identical output.** A naive profiler keyed
on nanosecond timestamps cannot satisfy the determinism contract.

## Decision

Adopt the same shape `cjc-cana` uses for its passive observer: **split the system
into a quarantined nondeterministic collection layer and a pure deterministic
analysis layer**, joined by a stable, content-addressed data model — the
`.seshat` trace.

### 1. The central determinism decision

- **Ordering and structure use a LOGICAL clock.** An event's "time" is its index
  (`seq: u64`) in `Trace::events`. There is no timestamp field. Analyses order,
  group, and hash by `seq` and interned ids.
- **Reproducible attribution uses SAMPLE COUNTS, not durations.** "`process_batch`
  is 38% of the profile" is a sample-count ratio, stable for a recorded trace,
  and is in the determinism gate.
- **Wall-clock is a single advisory scalar** (`Trace::wall_ns_total`), recorded
  for display but excluded from the content hash.
- **A recorded `.seshat` trace is the unit of reproducibility.** Live re-recording
  is *not* expected to be byte-identical — that drift is what feature 9
  (variance profiling) measures.

### 2. Integer-only hashed path

Percentages are integer **milli-percent** (`100% = 100_000`); counts and bytes
are exact integers. **No float ever enters the hashed path**, so there is no
summation-order concern and no need for Kahan/Binned accumulators in the
analysis engine. `BTreeMap`/`BTreeSet` throughout; FNV-1a content addressing
(fixed little-endian, length-prefixed strings) — never `DefaultHasher`.

### 3. The trace model (`.seshat`)

`FrameKind` (Py/Rust/Native/FfiBoundary/AsyncTask) · `OwnershipDomain`
(PyHeap/RustHeap/Mmap/NumPy/Arrow/Tensor/Gpu/NativeExt) · `ThreadState`
(Running/GilWait/LockWait/ChannelWait/IoWait/AsyncIdle) · `Event`
(Sample/Alloc/Free/Counter/ZoneStart/ZoneStop/Edge) · `CausalEdge`
(Wakeup/AwaitResume/GilHandoff/Copy/BoundaryCross). Frames/strings interned;
analyses group by the resolved `"kind:name (file:line)"` label, making report
hashes independent of interning-id values.

### 4. Twelve analyses as pure reductions

flamegraph · boundary · copy (no false positives) · contention · async-stall ·
ownership (partition invariant) · peak · recommendations (fixed-order rule
table) · variance · thermal · pipeline · regression-diff. Each returns an owned
report; the aggregate `SeshatReport::content_hash` is the gated digest.

### 5. The `.cjcl` surface — write-only satellite dispatch

Eight `seshat_*` builtins (zone_start/stop, mark_boundary, mark_copy, alloc_tag,
event_count, dump_trace, reset) emit events into a per-thread `TraceBuilder`
sink. They are **write-only** — they never return analysis back into program
state — exactly like the existing `profile_zone_*` builtins they generalize.
Routed through a satellite `dispatch_seshat()` to avoid a `cjc-runtime →
cjc-seshat` cycle (the `cjc-ad`/`cjc-quantum`/`cjc-locke` precedent), and wired
into both executors after the other satellites. Zone handles come from a
monotonic counter → identical across `cjc-eval` and `cjc-mir-exec` (parity gate).

### 6. Collectors and Python bridge deferred (feature `collect-live`)

The live OS-level probes (Rust `GlobalAlloc` shim, CPython frame walk,
perf_event counters) and the PyO3/maturin `seshat` Python package are
platform-specific and nondeterministic. They only *produce* traces, so the
entire analysis surface is testable today via the synthetic `Trace::builder`.
They are feature-gated so the default build/test surface stays pure.

## Consequences

- **Positive:** a profiler with a real determinism story — pinned golden report
  hash, order-invariant flamegraph, sound copy detector, bounds-checked
  `replay`. Reuses the cana observer contract and the satellite-dispatch
  precedent, so it touches no IR and preserves `Value` enum layout (HARD RULE).
- **Honest limitation:** count-based attribution is gated; wall-clock millisecond
  numbers are advisory. We do not dress a nondeterministic number as a
  guarantee. This matches the project's "infrastructure gates pass / quality
  numbers honest" precedent.
- **Deferred work:** live capture and the Python bridge. The acceptance test for
  that phase is dogfooding the repo's own `python/` (cjc-locke-py) PyO3 seam.

## Alternatives rejected

- **Timestamp-keyed trace (perf/pprof style).** Breaks determinism at the root;
  rejected per the prime directive.
- **Builtins that return analysis into program state.** Would let profiling
  perturb control flow and break parity/NoGC; rejected in favor of write-only
  markers + opaque integer handles.
- **`cjc-runtime` calling into `cjc-seshat` directly.** Dependency cycle;
  rejected for satellite dispatch.

## Addendum — recorder capture gaps A/B/C closed (pure stdlib, zero-dep)

Three Python-recorder capture capabilities were added *entirely on the collection
side*, leaving the deterministic analysis engine (and the golden report hash
`0xa4cda1369275d1ff`) untouched — the collection⟂analysis split doing exactly its
job:

- **A — copy auto-discovery.** A `_COPY_FUNCS` registry + bound-method `__self__`
  byte estimation emits `Copy` edges for numpy/torch/arrow copy calls, so the
  copy detector no longer needs manual `mark_copy`. Conservative (ambiguous
  `bytes`/`bytearray` excluded) to keep the detector's no-false-positives contract.
- **B — multi-thread capture.** `threading.setprofile` + per-OS-thread stacks +
  per-tick per-thread samples with stable logical thread ids. The `.seshat`
  `Sample.thread` field already existed; only capture was wired. `TraceWriter`
  gained an `RLock` (output bytes unchanged) for concurrent interning safety.
- **C — GIL-wait heuristic.** Frozen-leaf-while-another-progresses → `GilWait`.
  Explicitly **approximate** (no exact pure-Python GIL signal), labelled as such
  in the `SES-GIL-BOUND` recommendation and the README.

**Decision:** A/B/C honored the zero-dep rule (no dependency added). Gap D (D1
native Rust-frame unwinding, D2 thermal counters) each *wants* a dependency and is
deferred with the decision still open — see `SESHAT_DESIGN.md` §9. No gated output
changed; the `Value` enum / MIR remain untouched.
