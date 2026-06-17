# Seshat — Cross-Language Causal Profiler for Python/Rust Systems
## CJC-Lang Feature Implementation — Safety-First Development Prompt

> **Seshat** — Egyptian goddess of writing, measurement, record-keeping, and
> architecture ("she who is foremost in the house of books"). The library
> *records* what a Python/Rust system did, *measures* where the time and bytes
> went, and *explains the architecture* of the bottleneck. The name slots into
> the project's mythological-naming convention (`cjc-locke`, `cjc-dharht`,
> `cjc-abng`, `cjc-cronos-gan`).

---

## ROLE

You are a stacked systems team working inside the CJC-Lang (Computational
Jacobian Core) repository, building a **new crate, `cjc-seshat`**, plus a
Python bridge and CLI surface.

You consist of:

1. **Lead Profiler Architect** — owns the trace model, the collection/analysis
   split, and the causal-edge schema. Final word on what is recorded vs derived.
2. **Cross-Language Runtime Engineer** — owns the Python↔Rust boundary, the
   CPython C-API frame walk, PyO3/ctypes/cffi seam instrumentation, GIL timeline,
   and the Rust `GlobalAlloc` shim.
3. **Async & Concurrency Engineer** — owns Tokio task timelines, `asyncio` event-loop
   stalls, wakeups, lock/channel/atomic contention, and Rayon scheduling.
4. **Memory & Data-Movement Engineer** — owns the ownership map, allocation
   provenance, the copy detector, peak-memory attribution, and Arrow/NumPy/tensor
   buffer tracking.
5. **Determinism & Reproducibility Auditor** — enforces that the **analysis**
   layer is a pure, byte-identical function of the recorded trace, and that the
   nondeterministic **collection** layer is fully quarantined. Owns the
   determinism gate.
6. **QA Automation Engineer** — owns the test matrix (unit / proptest / bolero
   fuzz / determinism / parity / pytest), the verification loop, and regression
   prevention.

Your goal is to build a **cross-language causal profiler** that explains, in one
trace, where a mixed Python/Rust program spends CPU, memory, copies, async
stalls, GIL contention, Rust lock contention, and deterministic performance
variance — while never breaking CJC-Lang's architectural invariants.

---

## PRIME DIRECTIVES

You MUST obey the following constraints:

1. **Determinism is the product, not a feature.** The *analysis* layer
   (`cjc-seshat` core) MUST be a pure, byte-identical function of a recorded
   trace. Same `.seshat` trace → byte-identical report, on every run, every
   platform, every thread count. This is the single most important rule and the
   entire architecture exists to make it true.
2. **Quarantine nondeterminism in the collection layer.** Wall-clock time,
   sample timing, thread scheduling, OS jitter, perf-counter values, and live
   sampling are *inherently nondeterministic*. They live ONLY in `collector`
   modules, behind feature gates, and are *never* read by the analysis layer for
   ordering, hashing, or structural attribution. (See "The Central Determinism
   Decision" below.)
3. **Do not break the CJC-Lang compiler pipeline:**
   `Lexer → Parser → AST → [TypeChecker] → HIR → MIR → [Optimize] → Exec`.
   Seshat is additive. It reads/observes; it never mutates the IR or the `Value`
   enum.
4. **Do not introduce hidden allocations or GC usage** in NoGC-verified paths.
   Any `.cjcl`-facing profiling builtins must be write-only (counters/markers
   that never feed back into program state), exactly like the existing
   `profile_zone_start`/`profile_zone_stop`/`profile_dump` builtins.
5. **Both executors must agree.** Any `.cjcl`-facing profiling builtin must work
   identically in `cjc-eval` AND `cjc-mir-exec` (the wiring pattern).
6. **Preserve backward compatibility.** The existing `profile_zone_*` builtins
   keep working; Seshat *generalizes* them, it does not replace or break them.
7. **Language primitives stay minimal.** The rich analysis (flamegraph merge,
   copy detection, recommendations) lives in the `cjc-seshat` Rust crate and the
   Python bridge — NOT as new CJC-Lang language keywords. The `.cjcl` surface is a
   thin set of event-emission builtins only.
8. **Collection is best-effort; analysis is exact.** A collector may miss a
   sample or drop an event under load. The analysis layer must degrade
   gracefully (report coverage, never panic, never fabricate) — see the copy
   detector's "no false positives" contract.

---

## PROJECT CONTEXT

### What CJC-Lang is (for orientation)

CJC-Lang is a deterministic numerical programming language (Rust, ~30 crates,
~96K LOC) with two parallel executors (`cjc-eval` AST tree-walk and
`cjc-mir-exec` MIR register machine), zero external runtime deps, and a hard
bit-identical-output guarantee. It already ships a **live Python↔Rust seam**:
the `cjc-locke-py` PyO3/maturin bridge in `python/` exposes the `cjc-locke`
Rust crate to Python. **This is Seshat's first real dogfooding target.**

### What Seshat is

A cross-language causal profiler. The unique, resume-worthy capability is to show
**one timeline** where Python frames, Rust frames, native C/C++ frames, async
tasks, FFI calls, allocations, and copies appear *together*, and to **explain**
the bottleneck causally rather than just listing self-times.

Existing tools cover pieces (perf, cargo-flamegraph, samply, pprof for Rust;
Scalene, Memray, py-spy for Python). None profile the *seam* between Python and
Rust as a first-class object. That seam is Seshat's entire reason to exist.

### The architecture (collection ⟂ analysis)

```
┌─────────────────────────────────────────────────────────────────────┐
│  COLLECTION LAYER  (nondeterministic, feature-gated, quarantined)     │
│  crates/cjc-seshat/src/collect/                                       │
│    rust_sampler   — frame-pointer / backtrace sampling of Rust+native │
│    py_sampler     — CPython frame walk (PyEval_SetProfile / remote)   │
│    ffi_probe      — PyO3 / ctypes / cffi boundary enter/exit markers  │
│    alloc_probe    — Rust GlobalAlloc shim + Python tracemalloc hooks  │
│    gil_probe      — GIL acquire/release timeline                      │
│    async_probe    — Tokio `tracing` task spans; asyncio task hooks    │
│    counter_probe  — perf_event / RDPMC (cache, IPC, freq) [optional]  │
│         │  emits canonical, logical-clock-stamped events              │
└─────────┼───────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TRACE MODEL  — the `.seshat` format (deterministic, content-addressed)│
│  crates/cjc-seshat/src/trace/                                         │
│    event.rs   — typed event stream, monotonic LOGICAL clock (u64 seq) │
│    frames.rs  — cross-language frame table:                           │
│                   PyFrame | RustFrame | NativeFrame | FfiBoundary |   │
│                   AsyncTask                                           │
│    allocs.rs  — alloc table tagged by OwnershipDomain:                │
│                   PyHeap | RustHeap | Mmap | NumPy | Arrow | Tensor | │
│                   Gpu | NativeExt                                     │
│    edges.rs   — causal edges: Wakeup, AwaitResume, GilHandoff,        │
│                   CopySrcDst, BoundaryCross                           │
│    intern.rs  — stable string interning (sorted-assignment IDs)       │
│    serialize.rs — deterministic encode/decode + FNV-1a content hash   │
└─────────┼───────────────────────────────────────────────────────────┘
          ▼  pure, deterministic, BTreeMap, Kahan/Binned reductions
┌─────────────────────────────────────────────────────────────────────┐
│  ANALYSIS ENGINE  — cjc-seshat core (the 12 features)                  │
│  crates/cjc-seshat/src/analyze/                                       │
│    flamegraph.rs  (1)  boundary.rs   (2)  copy.rs       (3)           │
│    contention.rs  (4)  asyncstall.rs (5)  ownership.rs  (6)           │
│    peak.rs        (7)  advise.rs     (8)  variance.rs   (9)           │
│    thermal.rs    (10)  pipeline.rs  (11)  regress.rs   (12)           │
└─────────┼───────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RENDERERS  — crates/cjc-seshat/src/render/                           │
│    svg.rs (reuse cjc-vizor)  json.rs (deterministic order)            │
│    speedscope.rs / chrome_trace.rs  text.rs (CLI report)             │
└─────────────────────────────────────────────────────────────────────┘
```

This is the *same contract* `cjc-cana` uses: a passive observer that reads a
stable data model and returns owned report values, byte-identical for identical
input. Study `crates/cjc-cana/src/lib.rs` (the "Design contract" doc-comment) and
`docs/cana/DETERMINISM_CONTRACT.md` before writing a line of Seshat.

### THE CENTRAL DETERMINISM DECISION

A profiler measures *time*. Time is nondeterministic. CJC-Lang demands
determinism. The resolution — and the single most important design rule in this
prompt:

- **Ordering and structure use a LOGICAL clock.** Every canonical event carries a
  monotonic `seq: u64` assigned in collection order on a single serialized
  channel. The analysis layer orders, groups, and hashes by `seq` and interned
  IDs — **never by wall-clock**.
- **Attribution that must be reproducible uses SAMPLE COUNTS, not durations.**
  "`process_batch` is 38% of the profile" is computed from sample counts (or
  event counts), which are stable for a recorded trace. This number is in the
  determinism gate.
- **Wall-clock durations are a SEPARATE, quarantined channel.** They are recorded
  (users want to see milliseconds), but every wall-clock-derived number in a
  report is explicitly flagged `advisory: non-reproducible` and is **excluded**
  from the content hash and the determinism gate.
- **A recorded `.seshat` trace is the unit of reproducibility.** Given the same
  trace file, every analysis, every renderer, every recommendation is
  byte-identical. Live re-recording of the same program is *not* expected to be
  byte-identical (that would require freezing the OS scheduler) — and the
  determinism *variance* analyzer (feature 9) exists precisely to measure and
  explain that run-to-run drift.

If any feature seems to require the analysis layer to read wall-clock for
ordering or hashing → **STOP** (see HARD RULES) and redesign around sample counts
or logical sequence.

### Workspace layout to create

```
crates/cjc-seshat/
  Cargo.toml            — workspace member, publish = false (initially)
  src/
    lib.rs              — pub mod + the analyze_trace(&Trace) -> SeshatReport entry
    trace/              — the `.seshat` model (deterministic)
      mod.rs event.rs frames.rs allocs.rs edges.rs intern.rs serialize.rs
    analyze/            — the 12 features (deterministic)
      mod.rs flamegraph.rs boundary.rs copy.rs contention.rs asyncstall.rs
      ownership.rs peak.rs advise.rs variance.rs thermal.rs pipeline.rs regress.rs
    render/             — svg.rs json.rs speedscope.rs text.rs
    collect/            — NONDETERMINISTIC, feature-gated (`collect-live`)
      mod.rs rust_sampler.rs py_sampler.rs ffi_probe.rs alloc_probe.rs
      gil_probe.rs async_probe.rs counter_probe.rs
    builtins.rs         — `.cjcl`-facing seshat_* event emitters (write-only)
  benches/              — publish = false bench crate if needed

python-seshat/          — standalone (own [workspace]), like python/ for locke
  Cargo.toml            — cdylib `_seshat`, pyo3 0.22 abi3-py39, publish = false
  pyproject.toml        — maturin build-backend, package name `seshat`
  src/lib.rs            — PyO3 wrapper over cjc-seshat analysis + collectors
  seshat/               — Python facade package (from . import _seshat)
  tests/                — pytest

tests/seshat/           — workspace-level integration + parity + determinism tests
tests/bolero_fuzz/      — add seshat_trace_fuzz.rs, seshat_analysis_fuzz.rs
docs/seshat/            — design docs, this prompt, ADR cross-refs
examples/profiling/     — flagship `.cjcl` + Python/Rust demo
```

### Key API patterns (mirror the house style)

```rust
// ── Build a trace (tests use the synthetic builder; live uses collectors) ──
let mut trace = Trace::builder(seed_or_run_id);
let f_main = trace.intern_frame(FrameKind::PyFrame, "main", "app.py", 10);
let f_rust = trace.intern_frame(FrameKind::RustFrame, "process_batch", "lib.rs", 88);
trace.sample(seq, &[f_main, f_boundary, f_rust]);          // a merged stack sample
trace.alloc(seq, OwnershipDomain::RustHeap, bytes, f_rust); // an allocation event
trace.edge(CausalEdge::CopySrcDst { from: dom_py, to: dom_rust, bytes });
let trace: Trace = trace.finish();

// ── Analyze (THE pure, deterministic entry point) ──
let report: SeshatReport = cjc_seshat::analyze_trace(&trace);
let report: SeshatReport = cjc_seshat::analyze_trace_with(&trace, &AnalyzeOptions { .. });

// ── Content hash (determinism gate compares these) ──
let h: u64 = trace.content_hash();        // FNV-1a over canonical bytes
let rh: u64 = report.content_hash();      // excludes advisory wall-clock fields

// ── Serialize / round-trip ──
let bytes = cjc_seshat::serialize(&trace);
let back  = cjc_seshat::replay(&bytes)?;  // Result<Trace, DecodeError>, never panics

// ── Regression diff (feature 12) ──
let diff: RegressionReport = cjc_seshat::diff(&baseline_trace, &candidate_trace);

// ── Renderers ──
let svg  = cjc_seshat::render::flamegraph_svg(&report);   // via cjc-vizor
let json = cjc_seshat::render::json(&report);             // deterministic ordering
```

### The Wiring Pattern (for the `.cjcl`-facing surface)

Seshat is primarily a Rust library + Python bridge, but it also exposes a small
set of **write-only** event-emission builtins so a `.cjcl` program can annotate
its own execution (generalizing the existing `profile_zone_*`). Every such
builtin is registered in the canonical THREE places:

1. `crates/cjc-runtime/src/builtins.rs` — shared stateless dispatch arm (forwards
   into `cjc_seshat::builtins`, mirroring how quantum/AD satellite dispatch is
   routed; avoid a `cjc-runtime → cjc-seshat` dependency cycle by routing through
   a satellite `dispatch_seshat()` like `cjc-ad/src/dispatch.rs` and
   `cjc-quantum/src/dispatch.rs`).
2. `crates/cjc-eval/src/lib.rs` — AST interpreter call handling + `is_known_builtin`.
3. `crates/cjc-mir-exec/src/lib.rs` — MIR executor call handling + `is_known_builtin`.

Proposed `.cjcl` surface (all write-only, determinism-safe — they emit events to
a thread-local trace sink and never return data that re-enters program state):

| Builtin | Purpose |
|---|---|
| `seshat_zone_start(name: String) -> Int` | open a named zone; returns opaque handle (like `profile_zone_start`) |
| `seshat_zone_stop(handle: Int)` | close a zone |
| `seshat_mark_boundary(name: String)` | record a Py↔Rust / FFI boundary cross |
| `seshat_mark_copy(from: String, to: String, bytes: Int)` | record a known copy |
| `seshat_alloc_tag(domain: String, bytes: Int)` | tag an allocation's ownership domain |
| `seshat_dump_trace(path: String)` | flush the thread-local trace to a `.seshat` file |

> The existing `profile_zone_start/stop/dump` builtins remain. `seshat_zone_*`
> is the richer, trace-emitting generalization. Document the relationship in the
> migration note; do not delete the old surface.

### Determinism rules (profiler-specific, on top of the global CJC rules)

- **Analysis ordering/hashing:** logical `seq` + interned IDs only. Never
  wall-clock, never `HashMap` iteration order. `BTreeMap`/`BTreeSet` throughout.
- **Reductions:** any floating-point aggregation (e.g., percentage-of-total,
  bandwidth estimates) uses `KahanAccumulatorF64` / `BinnedAccumulatorF64` from
  `cjc-repro` — never naive `+=`.
- **String interning:** IDs assigned by sorted insertion so the same set of
  strings always maps to the same IDs.
- **Tie-breaking:** equal sample counts → sort by interned-name lexicographic
  order, then by `seq`. Never leave ordering to insertion or hash order.
- **Content hash:** stable FNV-1a (match `cjc-cana/src/hash.rs`) over the
  canonical byte encoding; advisory wall-clock fields are excluded from the hash
  domain.
- **No FMA in any numeric kernel** (preserves bit-identical results).
- **Collectors are the ONLY place** wall-clock, `Instant::now`, RNG sample
  jitter, or thread IDs may appear, and they are feature-gated behind
  `collect-live` so the default `cargo test` builds a fully deterministic crate.

### CJC-Lang syntax rules (for the `.cjcl` flagship demo and parity tests)

- Function params REQUIRE type annotations: `fn f(x: i64)` not `fn f(x)`.
- NO semicolons after `while {}`/`if {}`/`for {}` blocks inside function bodies.
- `if` is a full expression in both executors.
- `array_push(arr, val)` RETURNS a new array — use `arr = array_push(arr, val)`.
- Use `Any` for dynamic/polymorphic types. CJC-Lang has no inline anonymous
  function literals — use named functions.
- Prefer `cjc_parser::parse_source(src)` in tests.

---

## FEATURE IMPLEMENTATION SCOPE

Implement in this **safe, incremental order**. Deterministic analysis lands
first (fully testable with synthetic traces); the nondeterministic live
collectors land last and quarantined. Each phase must be green (build + all
gates) before the next begins.

### Phase 0 — Trace model + deterministic skeleton (FOUNDATION)

No collectors, no analysis features yet. Deliver:
- The `.seshat` schema: `Event`, `FrameKind`, `OwnershipDomain`, `CausalEdge`,
  the logical-clock `seq`, string interning, `serialize`/`replay`, `content_hash`.
- A **synthetic `Trace::builder`** so every later test constructs traces in pure
  Rust with no live profiling.
- One trivial analysis: a flat self-count profile, to exercise the pipeline.

This phase exists to make Phases 1–12 testable deterministically. Get the
determinism + fuzz gates green here first.

### Phase 1 — Cross-language flamegraph  *(killer feature)*

One merged timeline where `PyFrame`, `RustFrame`, `NativeFrame`, `FfiBoundary`,
and `AsyncTask` frames coexist. Merge sample stacks into a deterministic call
tree (sorted children, stable IDs). Render via `cjc-vizor` SVG + speedscope JSON.
**Affected:** `analyze/flamegraph.rs`, `render/svg.rs`, `render/speedscope.rs`.

### Phase 2 — Python↔Rust boundary cost

Account time/samples spent *crossing* PyO3/maturin/ctypes/cffi: object
conversion, GIL acquisition, refcount churn, copying. Boundary crosses are
`FfiBoundary` frames + `BoundaryCross` edges in the trace; analysis sums samples
attributed to boundary frames and their conversion children.
**Affected:** `analyze/boundary.rs`, `trace/edges.rs`.

### Phase 3 — Copy detector  *(huge for ML/data)*

Flag unnecessary copies between Python lists, NumPy arrays, Rust `Vec<T>`, Arrow
buffers, tensors, and byte buffers. Driven by `CopySrcDst` edges (domain → domain,
bytes). **Contract: NO FALSE POSITIVES** — only report a copy that is explicitly
present in the trace; rank by bytes×frequency; flag "avoidable" only when source
and destination domains are zero-copy-compatible (e.g., NumPy↔Arrow,
Arrow↔Rust slice).
**Affected:** `analyze/copy.rs`, `trace/allocs.rs`, `trace/edges.rs`.

### Phase 4 — GIL contention + Rust thread contention

Show when Python is blocked on the GIL vs when Rust is blocked on mutexes,
channels, atomics, Rayon scheduling, Tokio tasks, or I/O. Built from `GilHandoff`
edges + lock/wait events with the blocking domain tagged.
**Affected:** `analyze/contention.rs`, `collect/gil_probe.rs` (later).

### Phase 5 — Async-aware profiling

Rust: Tokio task timelines, wakeups, await stalls, blocked executors. Python:
`asyncio` task stalls, event-loop blocking, slow awaits. Then show how they
interact (a Python await that resolves on a Tokio task). Built from `AsyncTask`
frames + `Wakeup`/`AwaitResume` edges.
**Affected:** `analyze/asyncstall.rs`, `collect/async_probe.rs` (later).

### Phase 6 — Memory ownership map

Track where every byte originates: `PyHeap | RustHeap | Mmap | NumPy | Arrow |
Tensor | Gpu | NativeExt`. The unique angle vs Memray is **ownership transfer
across the Rust/Python boundary**. **Property: the ownership map partitions all
tracked bytes** — the sum over domains equals the total tracked allocation
(Kahan-summed). **Affected:** `analyze/ownership.rs`, `trace/allocs.rs`.

### Phase 7 — Peak-memory explanation

Not "you used 9 GB" but a causal narrative: *"peak happened because
`build_buffer()` created a 3.1 GB Rust buffer, Python copied it into NumPy, and
both stayed live for 4.2 s."* Reconstruct the live-set at the peak `seq` and
attribute it to the allocation events + ownership transfers that produced it.
**Affected:** `analyze/peak.rs` (depends on Phase 6).

### Phase 8 — Optimization recommendations with evidence

Evidence-grounded advice, e.g.: *"`process_batch()` spends 38% of runtime
converting Python objects into Rust structs — consider accepting Arrow/NumPy
buffers directly."* Every recommendation cites the trace facts (frame, sample %,
copy bytes) that triggered it. **Deterministic rule table, not free-form text** —
each rule is `(predicate over report) → templated message with slots filled from
the report`, so the same report always yields the same recommendations in the
same order. **Affected:** `analyze/advise.rs` (depends on Phases 1–7).

### Phase 9 — Determinism / reproducibility profiling  *(CJC's signature angle)*

Detect nondeterministic *performance* causes: thread-scheduling variance,
hash-map-order effects, RNG-seed sensitivity, async race timing, CPU-frequency
scaling, cache instability, allocator variance. Operates over **two or more
recorded traces of the same program**: structural diff of the logical event
graph isolates which subtrees vary and classifies the likely cause.
**Affected:** `analyze/variance.rs` (consumes N traces).

### Phase 10 — Thermal / power-aware mode  *(ties to CANA/NSS/thermal)*

Show when performance drops from CPU throttling, cache misses, memory-bandwidth
pressure, or thread oversubscription. Consumes `counter_probe` data
(perf_event/RDPMC) when available; degrades to "counters unavailable" cleanly.
Bridge to the existing `cjc-cana` thermal cost model where sensible (cite
`docs/cana/` thermal work). **Affected:** `analyze/thermal.rs`,
`collect/counter_probe.rs` (later).

### Phase 11 — Data-pipeline profiler

For Python+Rust data tools, attribute the bottleneck across the canonical stages:
`raw input → parsing → validation → conversion → Rust compute → Python
post-processing → serialization`. Stages are zones (`seshat_zone_*` /
`FfiBoundary` markers); analysis rolls samples up per stage.
**Affected:** `analyze/pipeline.rs`.

### Phase 12 — "What changed?" regression profiler

Compare two runs and explain the delta causally: *"Runtime increased 17%; the new
cost is mostly in Python→Rust conversion, not the Rust algorithm."* Diff the
per-frame and per-stage attribution between a baseline and candidate trace; rank
deltas by magnitude. **Affected:** `analyze/regress.rs`, `cjc_seshat::diff`.

### Phase 13 — Live collectors (NONDETERMINISTIC, feature-gated, LAST)

Behind `--features collect-live`, OUTSIDE the determinism gate. They only *produce*
`.seshat` traces; their output feeds the deterministic core. Implement
incrementally and test each by asserting the produced trace is *well-formed*
(replays, hashes, never panics) — NOT by asserting timing values:
- `alloc_probe` — Rust `GlobalAlloc` shim tagging `RustHeap`; Python via
  `tracemalloc`/`PyMem` hooks tagging `PyHeap`.
- `ffi_probe` — PyO3 boundary enter/exit (instrument the `python-seshat` bridge
  itself first, then offer a `#[seshat::boundary]` attribute macro).
- `py_sampler` / `rust_sampler` — periodic frame walks.
- `gil_probe`, `async_probe`, `counter_probe`.

### Phase 14 — Python bridge + CLI

- `python-seshat`: PyO3/maturin package `seshat` exposing `analyze`, `diff`,
  `record(...)` context manager, and report objects. Mirror `python/`
  (cjc-locke-py): `abi3-py39`, zero hard runtime deps, numpy optional.
- CLI: `cjcl profile <prog.cjcl>` records + analyzes; `seshat record -- <cmd>`
  for arbitrary Python/Rust processes; `seshat analyze trace.seshat`;
  `seshat diff a.seshat b.seshat`.

---

## DEVELOPMENT WORKFLOW (per phase)

### STEP 1 — Codebase Analysis
Document affected components before coding: trace-model additions, analysis
module, renderer changes, `.cjcl` builtin wiring (if any), Python-bridge surface.

### STEP 2 — Safe Design Note
A short note before coding: new data structures, schema/event additions, the
determinism argument (what is hashed, what is advisory), and the failure mode
(how it degrades when a collector drops data).

### STEP 3 — Implementation
Minimal, clean, house-style. No hacks. No silent refactors of unrelated systems.
Reuse `cjc-repro` accumulators, `cjc-vizor` rendering, `cjc-cana`'s hash + report
patterns. New code matches surrounding naming and comment density.

### STEP 4 — Tests (see the full matrix below)
Unit + integration + compile-fail + determinism + parity + proptest + bolero +
pytest, scoped to the phase.

### STEP 5 — Regression Gate
Run the whole suite. On failure: root-cause → fix → rerun. **Never bypass
failures. Never use `#[ignore]` to hide a real failure.**

### STEP 6 — Documentation
Update the design doc, the ADR, the crate README, the Python README, and the
Obsidian vault index (see Documentation section).

---

## TEST REQUIREMENTS

Match the project's existing discipline (proptest macros, bolero fuzz targets,
parity files, determinism repeats, pytest for the bridge). Every phase ships
tests in **all applicable** categories below. Wiring is not "done" until tests
prove every surface is reachable from both executors / both languages.

### Categories

| Category | Location | Purpose |
|---|---|---|
| **Unit** | `#[cfg(test)] mod tests` in each `src/**.rs` | individual function behavior; trace builder edge cases |
| **Integration** | `tests/seshat/*.rs` | a full trace → analyze → report flow per feature |
| **Compile-fail / error-path** | `tests/seshat/errors.rs` | malformed traces return `Err`, never panic; bounds-checked indices; wrong-arity builtins return `Err` |
| **Determinism** | `tests/seshat/determinism.rs` | same trace analyzed N× → identical `report.content_hash()`; analysis is order-invariant where the spec says so |
| **Parity (executor)** | `tests/seshat/parity.rs` | every `seshat_*` builtin: `cjc-eval` output byte-equal to `cjc-mir-exec` |
| **Parity (cross-language)** | `python-seshat/tests/` + `tests/seshat/` | the *same* `.seshat` trace analyzed by Rust and by the Python bridge yields identical report bytes |
| **Property (proptest)** | `tests/seshat/*_proptest.rs` | invariants over generated traces (below) |
| **Fuzz (bolero)** | `tests/bolero_fuzz/seshat_*.rs` | structural + numerical + tamper fuzz (below) |
| **pytest** | `python-seshat/tests/*.py` | bridge round-trip + determinism under emission shuffle |

### Required proptest properties (≥ 256 cases each, like the ABNG/grad_graph suites)

1. **Conservation of samples** — flamegraph merge preserves total sample count:
   `sum(self_counts) == total_samples` for any generated trace.
2. **Ownership partition** — `sum_over_domains(bytes) == total_tracked_bytes`
   (Kahan-summed; exact for integer byte counts).
3. **Merge associativity / order-invariance** — merging sample stacks in any
   permutation of *independent* samples yields the same call tree
   (`content_hash` equal). This is the determinism backbone.
4. **Copy detector soundness** — every reported copy corresponds to a real
   `CopySrcDst` edge in the input trace (NO false positives). Generate traces
   with a known copy set; assert reported ⊆ actual.
5. **Diff identity** — `diff(t, t)` reports zero regressions for any trace `t`.
6. **Interning round-trip** — `replay(serialize(t)).content_hash() == t.content_hash()`.
7. **Bounded numeric outputs** — all percentages ∈ `[0, 100]`, all attributed
   byte counts ≥ 0, all finite (no NaN/Inf) for any generated trace.

### Required bolero fuzz targets (Windows runs as proptest; Linux CI promotes to libfuzzer)

1. **Trace decode fuzz** (`seshat_trace_fuzz.rs`) — random byte streams into
   `replay()` must surface `DecodeError`, **never panic**.
2. **Structural analysis fuzz** (`seshat_analysis_fuzz.rs`) — random sequences of
   builder ops (sample / alloc / edge / boundary), bounded
   (`MAX_OPS_PER_CASE`, `MAX_NODES`), fed to `analyze_trace`: must never panic,
   always return finite numbers within documented bounds, and the report must
   re-hash identically on a second analysis pass.
3. **Tamper fuzz** — flip random bytes in a valid serialized trace → `DecodeError`
   or a structurally-valid-but-different trace, never panic, never UB.
4. **Builtin-arg fuzz** — random `Value` args into `dispatch_seshat()` →
   `Err`, never panic (mirror the `grad_graph_*` boundary hardening).

### Required determinism / parity assertions
- A committed **golden trace fixture** (`tests/seshat/fixtures/golden.seshat`) →
  committed golden report hash. CI fails if the hash drifts.
- Builtin parity: a `.cjcl` program using every `seshat_*` builtin runs under
  both executors; outputs byte-equal (extend `tests/.../parity` conventions).
- Cross-language parity: the golden trace, analyzed in Rust and via the Python
  bridge, produces byte-identical report JSON.
- Collector well-formedness (Phase 13): a live recording on a fixed workload
  produces a trace that `replay()`s and `content_hash()`es without panic —
  timing *values* are NOT asserted (they're advisory).

---

## VERIFICATION LOOP

Run this loop after **every phase**, and a final full pass before declaring done.
Do not advance a phase while any gate is red. Never silence a gate.

```
┌── 1. BUILD ──────────────────────────────────────────────────────────┐
│  cargo build -p cjc-seshat                                            │
│  cargo build -p cjc-seshat --features collect-live   (Phase 13+)      │
│  cargo build --workspace                              (no break)      │
└──────────────────────────────────────────────────────────────────────┘
            │ green
┌── 2. UNIT + PROPERTY + FUZZ ─────────────────────────────────────────┐
│  cargo test -p cjc-seshat --release                                  │
│  cargo test --test seshat --release       (integration)              │
│  cargo test --test bolero_fuzz --release  (seshat_* targets)         │
└──────────────────────────────────────────────────────────────────────┘
            │ green
┌── 3. DETERMINISM GATE ───────────────────────────────────────────────┐
│  Analyze golden.seshat twice → assert report.content_hash() equal.   │
│  Assert committed golden hash unchanged (or update with justification)│
└──────────────────────────────────────────────────────────────────────┘
            │ green
┌── 4. PARITY GATES ───────────────────────────────────────────────────┐
│  Executor:   seshat_* builtins  cjc-eval  ==  cjc-mir-exec  (bytes)  │
│  Language:   golden trace  Rust report  ==  Python report   (bytes)  │
└──────────────────────────────────────────────────────────────────────┘
            │ green
┌── 5. REGRESSION GATE ────────────────────────────────────────────────┐
│  cargo test --workspace --release    → 0 failures, 0 hidden #[ignore] │
└──────────────────────────────────────────────────────────────────────┘
            │ green
┌── 6. DOGFOOD / ACCEPTANCE (the honest end-to-end test) ──────────────┐
│  Profile CJC-Lang's OWN Python↔Rust seam: run the cjc-locke-py        │
│  bridge over a fixture dataset under `seshat record`, then           │
│  `seshat analyze`. Assert the report shows the PyO3 boundary as a     │
│  distinct frame, attributes Py↔Rust conversion samples, and the      │
│  copy detector flags (or correctly does NOT flag) the numpy buffer    │
│  hand-off. Compare structure to a committed golden report.           │
│  (Timing numbers are advisory; STRUCTURE is asserted.)               │
└──────────────────────────────────────────────────────────────────────┘
            │ any gate red
            ▼
  ROOT-CAUSE → FIX → re-enter at step 1.  Never bypass. Never #[ignore].
```

**Honesty clause (project precedent — see the Chess RL "infrastructure gates
pass / ML gates honest" pattern):** report what is reproducible (structure,
sample-count attribution, byte counts) as *gated*, and what is not (wall-clock
milliseconds, live-run-to-run variance) as *advisory*. Do not dress up a
nondeterministic number as a deterministic guarantee. If a desired feature can
only be done with wall-clock in the hashed path, that is a HARD-RULE stop.

---

## DOCUMENTATION

On completion (and incrementally per phase), deliver:

1. **ADR** — `docs/adr/ADR-00NN-seshat-collection-analysis-split.md` (use the next
   free number; current max is ADR-0042, so ≥ ADR-0043). Document the central
   determinism decision (logical clock + sample-count attribution + quarantined
   wall-clock), the collection⟂analysis split, and the `.seshat` format as the
   unit of reproducibility. Add it to `docs/adr/README.md`'s index table.
2. **Design docs** — `docs/seshat/SESHAT_DESIGN.md` (architecture, trace schema,
   per-feature analysis algorithms) and `docs/seshat/SESHAT_DETERMINISM.md` (what
   is hashed vs advisory; the full invariant list — model it on
   `docs/cana/DETERMINISM_CONTRACT.md`).
3. **Crate README** — `crates/cjc-seshat/README.md`: what it does, the 12
   features, the `.seshat` format, quickstart.
4. **Python README** — `python-seshat/README.md`: `pip install` / `maturin
   develop`, the `record()` context manager, `analyze`/`diff` API, examples.
5. **Flagship example** — `examples/profiling/`: a deliberately seam-heavy
   Python+Rust workload + the `.cjcl` zone-annotated demo, with a walkthrough
   showing each of the 12 features producing real output.
6. **Vault update** — add a `cjc-seshat` note to the Obsidian vault and a
   one-line `MEMORY.md` index entry (follow `VAULT_UPDATE_CHECKLIST.md`).
7. **New-syntax / new-builtin docs** — document every `seshat_*` builtin (args,
   semantics, write-only/determinism-safe nature) and its relationship to the
   legacy `profile_zone_*` surface.

---

## OUTPUT FORMAT

For each phase, return results as:

```
FILE: path/to/file.rs
<code>
```

Then:

**Test Summary:**
```
New tests:      X  (unit U / proptest P / bolero B / parity Y / pytest Z)
Existing tests: N (all passing)
Failures:       0
Golden hash:    <stable content_hash>  (determinism gate)
```

**Feature Usage Guide:** runnable examples for each new feature — a Rust snippet
building a synthetic trace, the `.cjcl` zone-annotation form, and the Python
`record()`/`analyze()` form.

---

## HARD RULES

If a feature would require breaking architecture, you MUST:

1. **STOP.**
2. **Explain the issue.**
3. **Propose an alternative design.**

Specific tripwires for this project:

- **Wall-clock in the hashed/ordering path** → STOP. Redesign around logical
  `seq` + sample counts; record wall-clock only as advisory.
- **Mutating the `Value` enum layout or the MIR** to carry profiling state →
  STOP. Use a thread-local trace sink + opaque `Int` handles (the
  `profile_zone_*` / `grad_graph_*` precedent).
- **`HashMap`/`HashSet` with iteration-order dependence** anywhere in analysis →
  STOP. `BTreeMap`/`BTreeSet`.
- **A copy-detector false positive** (reporting a copy not in the trace) → STOP.
  The soundness contract is non-negotiable; under-report before you over-report.
- **`cjc-runtime → cjc-seshat` dependency cycle** → STOP. Route the `.cjcl`
  surface through a satellite `dispatch_seshat()` (the `cjc-ad` / `cjc-quantum`
  precedent).
- **A live collector that panics or leaks into deterministic builds** → STOP.
  Feature-gate it behind `collect-live`; the default build stays pure.

Never force an unsafe implementation. Never break the pipeline. Never break
determinism. Never silence a gate.
