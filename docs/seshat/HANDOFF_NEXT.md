# Seshat — Handoff for the Next Session

Purpose: pick up Seshat (cross-language Python/Rust causal profiler) and tackle
the remaining capability gaps. Everything you need to start cold is here.

> **UPDATE (this session).** The previously-uncommitted Seshat core was first
> committed as a clean snapshot, then **Gaps A, B, and C were implemented**
> (all pure stdlib, zero new dependencies):
> - **A — copy auto-discovery** (`recorder.py` `_COPY_FUNCS`/`_estimate_copy_bytes`/`_maybe_emit_copy`).
> - **B — multi-thread capture** (`threading.setprofile` + per-thread stacks; `TraceWriter` made thread-safe with an `RLock`).
> - **C — GIL-wait heuristic** (`_classify_states`; labelled approximate in the report + docs).
>
> Proven: Python suite green (incl. new copy/thread/GIL tests), Rust `cjc-seshat`
> green with the **golden hash `0xa4cda1369275d1ff` unchanged**, `seshat_parity`
> green, committed fixtures **content-identical** (only advisory `wall_ns` moves),
> and a 3-thread recording analyses to `running:159 gil_wait:50` through the CLI.
>
> **Gap D — partly DONE (next session):**
> - **`seshat merge`** (feature 13, `merge.rs` + CLI) — unifies a Python and a Rust
>   `.seshat` into one trace (Rust grafted under the Py↔Rust boundary). Zero-dep,
>   deterministic, no format change. Proven e2e: merged report shows py + ffi +
>   rust frames together.
> - **Automatic Rust alloc-site unwinding** (`CaptureConfig{alloc_stacks}` / CLI
>   `--unwind`) — allocations attributed to real Rust functions (e.g.
>   `seshat::cmd_record_demo (seshat.rs:266)`) via `backtrace`, **scoped to
>   `collect-live`** (default build still dependency-free).
> - **Leftovers pass (best-effort, all green):** thermal capture DONE
>   (`seshat[thermal]` → psutil `cpu_freq` → Counter events → throttle detection);
>   explicit token merge correlation DONE (`collect::mark_host` + multi-native
>   fold); synchronous native CPU sampling DONE (`collect::native_sample()`); GIL
>   heuristic HARDENED (2-consecutive-tick streak). Golden hash unchanged.
> - **Genuinely still open:** *automatic* cross-thread CPU sampling
>   (SIGPROF/SuspendThread — unsafe/platform), *automatic* token injection across
>   the seam, cache-miss/IPC counters (psutil = freq only), exact GIL (needs a C
>   ext that defeats the pure-stdlib design — heuristic hardened instead).

---

## 0. Status snapshot (read this first)

Seshat is **working end-to-end**: a Python program records a `.seshat` trace, the
Rust CLI analyzes it into a cross-language report. Current capabilities for a
running Python process:

- 🟢 time-weighted flamegraph, Py↔native boundary (time-weighted), pipeline
  stages, async-stall measurement, Python-heap memory (temporal),
  **copy detection (manual marker + auto-discovery, Gap A)**,
  **multi-thread capture (Gap B)**, **GIL-wait heuristic (Gap C)**,
  recommendations, variance/diff (CLI), **Python+Rust trace merge (`seshat merge`)**,
  **automatic Rust alloc-site unwinding (`--unwind`, `collect-live`)**,
  low overhead (~py-spy), deterministic mode for CI.
- 🟡 **Gap D partly done** — merge + alloc-site unwinding shipped; CPU-time native
  sampling, thermal, and exact GIL remain (see the UPDATE banner + §5).

**Where things live**

| Thing | Path |
|---|---|
| Rust analysis engine (deterministic core) | `crates/cjc-seshat/src/{trace,analyze,report,serialize,hash,render}.rs` |
| Rust live collector (alloc shim + zone sampler), feature `collect-live` | `crates/cjc-seshat/src/collect/{alloc,recorder}.rs` |
| Rust CLI (`analyze`/`diff`/`variance`/`record-demo`) | `crates/cjc-seshat/src/bin/seshat.rs` |
| `.cjcl`-facing builtins (`seshat_*`) | `crates/cjc-seshat/src/dispatch.rs` (wired into `cjc-eval` + `cjc-mir-exec`) |
| Python recorder (pure stdlib) | `python-seshat/seshat/{recorder,format}.py` |
| Python CLI + entry point | `python-seshat/seshat/__main__.py`, `python-seshat/pyproject.toml` |
| Rust tests | `crates/cjc-seshat/tests/{integration,prop,fuzz,collect,python_bridge}.rs` + `tests/seshat/parity.rs` (root) |
| Python tests | `python-seshat/tests/test_format.py` |
| Committed cross-language fixtures | `crates/cjc-seshat/tests/fixtures/python_{demo,async}.seshat` (regen via `python-seshat/examples/gen_*.py`, `mode="calls"`) |
| Design / determinism / prompt | `docs/seshat/{SESHAT_DESIGN,SESHAT_DETERMINISM,SESHAT_IMPLEMENTATION_PROMPT}.md` |
| ADR | `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0043 …` |

**Test counts:** Rust `cjc-seshat` ≈ 49 (+1 gated `collect` test under
`--features collect-live`), Python 7, root `seshat_parity` 4. All green.

---

## 1. Build & test commands

This is a git worktree; the parent checkout's target dir is reused to avoid
recompiling the world. Run from the worktree root.

```bash
# Rust analysis core + CLI (default features)
CARGO_TARGET_DIR=<parent>/target cargo build -p cjc-seshat
CARGO_TARGET_DIR=<parent>/target cargo test  -p cjc-seshat

# Rust live collector (nondeterministic, quarantined)
CARGO_TARGET_DIR=<parent>/target cargo test -p cjc-seshat --features collect-live --test collect
CARGO_TARGET_DIR=<parent>/target cargo run  -p cjc-seshat --features collect-live --bin seshat -- record-demo out.seshat

# executor parity (root)
CARGO_TARGET_DIR=<parent>/target cargo test --test seshat_parity

# Python recorder
cd python-seshat && python tests/test_format.py
python -m seshat run examples/demo.py --out demo.seshat        # sampling (time-weighted)
cargo run -p cjc-seshat --bin seshat -- analyze demo.seshat    # the report

# regenerate committed fixtures (deterministic, calls mode)
cd python-seshat && PYTHONPATH=. python examples/gen_fixture.py       ../crates/cjc-seshat/tests/fixtures/python_demo.seshat
cd python-seshat && PYTHONPATH=. python examples/gen_async_fixture.py ../crates/cjc-seshat/tests/fixtures/python_async.seshat
```

---

## 2. Architecture invariants — DO NOT break these

1. **Collection ⟂ analysis.** The analysis engine (`analyze.rs`, `report.rs`) is a
   pure, byte-identical function of a recorded `.seshat`. All nondeterminism
   (wall-clock, sampling, threads) lives in collectors, quarantined behind feature
   gates / in the Python recorder. New capture work goes in collectors, never in
   the analysis engine.
2. **The `.seshat` file is the cross-language interface.** Python writes it
   (`python-seshat/seshat/format.py`), Rust reads it (`serialize.rs`). They must
   stay byte-compatible — any new event/edge type goes in BOTH, with matching
   tags. See `SESHAT_DETERMINISM.md` for tags.
3. **Determinism of the hashed path.** Integers only (milli-percent, counts,
   bytes), `BTreeMap`/`BTreeSet`, FNV-1a, logical clock (event index). The golden
   report hash `0xa4cda1369275d1ff` (`integration.rs::golden_report_hash_is_stable`)
   guards drift — if you change the engine's hashed output, update it deliberately.
4. **Value enum / MIR untouched.** `seshat_*` builtins route through the satellite
   `dispatch_seshat` (no `cjc-runtime → cjc-seshat` cycle).
5. **Zero external dependencies** is the CJC-Lang rule. Three of the four gaps
   below *want* a dependency or C code — each is flagged with the decision you
   must make. Default to honoring zero-dep; if you break it, scope the dep to the
   `collect-live` feature (Rust) or an optional `extras` (Python), never the
   deterministic core.

---

## 3. Hard-won gotchas (the bug log — re-read before touching the recorder)

- **`exec()` is a C call.** `run_path` execs the script, so a `builtins.exec`
  boundary frame would sit at the root of every sample. Fix in place: a
  `_TRANSPARENT_C` set (`exec`/`eval`/`compile`/`__import__`/`setprofile`) that is
  skipped (push `None`, keep children).
- **Don't record the recorder.** Frames from the seshat package dir are skipped
  (`_is_internal`). Critically, **C calls made BY the recorder's own Python code
  must also be skipped** — otherwise `intern_str`'s internal `dict.get`/
  `list.append` fire `c_call` events that re-enter interning mid-call and corrupt
  string ids. (Interning also computes its index *after* append for safety.)
- **`@contextmanager` leaks `contextlib.py` frames.** `record`/`zone` are plain
  classes so their `__enter__`/`__exit__` live in the recorder module (skipped).
- **Push/pop must stay balanced.** Skipped frames push `None` placeholders so the
  stack depth tracks correctly; `_live()` filters them.
- **Sampler thread safety is GIL-based.** The sampler reads the stack via an
  atomic slice copy `self._stack[:]`; only the main thread interns. No lock. The
  sampler thread has no `setprofile`, so it is invisible to the hook.
- **Sampling needs runtime.** CPython's GIL switch interval (~5 ms,
  `sys.getswitchinterval()`) floors the effective sample rate; sub-100 ms scripts
  get a handful of samples. Use `mode="calls"` for short/deterministic runs.

---

## 4. The two recording modes (you'll touch both)

`python-seshat/seshat/recorder.py`:
- `mode="sampling"` (default): `setprofile` only maintains the stack + boundary;
  a daemon thread samples it → **time-weighted**, low overhead, nondeterministic.
- `mode="calls"`: a sample per call → call-weighted, **deterministic** (used for
  committed fixtures and CI diffing).

New capture features generally need to work in BOTH (or be clearly mode-scoped).

---

## 5. The remaining work (the four gaps)

Suggested order: **#4 copy auto-discovery** (pure stdlib, self-contained, highest
ratio of value to risk) → **#1 multi-thread sampling** (pure stdlib, unblocks #2)
→ **#2 GIL heuristic** (rides on #1) → **#3 native unwinding** and **thermal**
(both need a dependency decision).

> Status: **Gaps A, B, C are DONE** (this session) — see the UPDATE banner above.
> Only **Gap D** below remains.

### Gap A — Copy auto-discovery ✅ DONE (currently manual `mark_copy`)

**Problem.** Today you must call `seshat.mark_copy(...)`; it doesn't *find* copies.
Auto-discovery is most of Memray's value.

**Why it's tractable here.** The `c_call` hook already SEES every native call,
including the copy-inducing ones (`numpy.array`, `np.ascontiguousarray`,
`ndarray.copy`, `bytes`, `bytearray`, `torch.from_numpy`/`.clone()`, Arrow
`to_pandas`/`combine_chunks`, `pandas.DataFrame.to_numpy(copy=True)`). The `arg`
passed to the hook is the callee; in many cases you can inspect the operand's
`.nbytes` to estimate copied bytes.

**Realistic approach (pure stdlib).** Add a `_COPY_FUNCS` registry mapping known
native-callee labels → a `(from_domain, to_domain)` guess. In the `c_call`
handler, when `label in _COPY_FUNCS`, try to read the size from the call (the
hook's `frame` gives the caller; `frame.f_locals` or the arg's `__len__`/`.nbytes`
where reachable) and emit a `Copy` edge. Bytes will be an estimate — mark it
honestly. This stays zero-dep (you only inspect objects that are already there;
you do NOT import numpy).

**Files.** `python-seshat/seshat/recorder.py` (c_call branch + registry).
**Test/proof.** A Python test that does `np.ascontiguousarray(np.zeros(...))` (skip
if numpy absent) and asserts a `Copy` edge appears; plus a unit test of the
registry with a fake callee. Update the committed demo to show auto-detected
copies alongside the marked one.
**Decision.** None — pure stdlib. Be explicit in docs that byte counts are
estimates.

### Gap B — Multi-thread Python sampling ✅ DONE (prerequisite for GIL)

**Problem.** `setprofile` is per-thread, and there is a single `self._stack`, so
only the main thread is captured.

**Realistic approach (pure stdlib).** (1) Call `threading.setprofile(self._hook)`
so spawned threads install the hook. (2) Make the stack per-thread: key by
`threading.get_ident()` (e.g. `self._stacks: dict[int, list]`), assign each OS
thread a stable logical `u32` thread id in first-seen order. (3) In the sampler,
iterate all known threads and emit one Sample per thread per tick, stamping
`Sample.thread`. The `Event::Sample{thread,…}` field already exists — wire it.
Watch thread-safety: each thread mutates only its own stack; the sampler
slice-copies each.

**Files.** `python-seshat/seshat/recorder.py`. The engine already supports
per-thread samples (`thread` field in `analyze::contention`/`thermal`).
**Test/proof.** A test spawning 2 worker threads; assert ≥2 distinct logical
thread ids appear in the trace.
**Decision.** None — pure stdlib. Note it raises overhead (hook on every thread).

### Gap C — GIL-wait detection ✅ DONE (heuristic, rides on Gap B)

**Problem.** `setprofile` cannot see GIL acquisition; there is no pure-Python
exact signal.

**Realistic approach.** With multi-thread sampling (Gap B), use a **heuristic**:
across consecutive sampler ticks, a thread whose top frame is unchanged while a
*different* thread made progress is likely GIL-starved → emit its samples with
`ThreadState::GilWait` instead of `Running`. This is approximate, not exact —
**label it as a heuristic in the report and docs.** The engine's `contention`
analysis already splits `GilWait`; you only need the Python side to set the state.
**Exact alternative (needs a dep / C):** a tiny C extension or `greenlet`/
`py-spy`-style out-of-process read of interpreter state. Do NOT pursue unless the
zero-dep rule is explicitly waived.
**Files.** `python-seshat/seshat/recorder.py` (sampler state classification).
**Test/proof.** Two CPU-bound threads contending; assert some `GilWait` samples
appear. Keep the assertion loose (it's a heuristic).
**Decision.** Heuristic = zero-dep, ship it labeled as approximate. Exact = needs
sign-off to break zero-dep.

### Gap D — Native Rust-frame unwinding + Thermal (both need a dependency call)

**D1 — Native Rust frames inside the extension.**
**Problem.** The Python side sees "the boundary," not the Rust function names
executing past it.
**Realistic path (the honest one).** The Rust extension self-instruments with
`cjc-seshat`'s collector: it links `cjc-seshat` (feature `collect-live`), installs
`SeshatAlloc`, and brackets work in `collect::zone(...)` (or a small
`#[seshat::frame]` proc-macro you add) — producing its own `.seshat`. Then the two
traces are **merged on a shared timeline**: when Python crosses the boundary it
records a correlation token; the Rust side records the same token at entry; a
merge step (new `seshat merge a.seshat b.seshat` CLI subcommand) slots the Rust
subtree under the Python boundary frame. The merge + correlation token is the real
design work — sketch it before coding.
**True automatic unwinding** (no manual zones) needs frame-pointer unwinding or
the `backtrace` crate → **breaks zero-dep**. If chosen, scope `backtrace` strictly
to the `collect-live` feature (it never touches the deterministic core).
**Files.** `crates/cjc-seshat/src/collect/`, a new merge in `bin/seshat.rs`, maybe a
proc-macro crate. **Decision required:** manual-zones (zero-dep) vs `backtrace`
(dep, but quarantined).

**D2 — Thermal / power.**
**Problem.** Need CPU frequency + cache misses; no portable stdlib source.
**Realistic approach.** Optional: a Python extra (`seshat[thermal]` → `psutil`)
that samples `psutil.cpu_freq()` in the sampler thread and emits `Counter` events;
or, Rust side, a Linux-only perf_event reader behind `collect-live`. The engine's
`thermal` analysis already consumes `Counter` events — only capture is missing.
**Decision required:** thermal needs a dependency (`psutil`) or OS-specific code.
Honor zero-dep by making it an explicit opt-in extra, not a core requirement.

---

## 6. What "done" looks like for each

For every gap: the capture lands in the collector/recorder (never the engine);
there's a test proving the new events flow into the existing analysis; the report
shows the new signal; docs (`python-seshat/README.md` capture table + this file)
are updated; and any dependency is scoped to a feature/extra with the decision
recorded in `SESHAT_DESIGN.md` §9 and ADR-0043 (or a new ADR). Keep the
deterministic core's golden hash stable unless you deliberately change hashed
output.
