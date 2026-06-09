# Option B — Real MIR-exec Instrumentation (Design Doc)

**Date:** 2026-06-09
**Status:** Design only — no implementation in this session
**Scope:** §3.1 of `HANDOFF_NEXT_SESSION_v3.md`
**Estimated effort:** 2-3 weeks for one engineer, full-stack

This doc describes WHAT needs to change to replace `NssPressurePredictor`'s
synthetic-trace projection (Option A) with real per-event telemetry from
an instrumented `cjc-mir-exec` run (Option B), and WHY each piece needs
to look the way it does.

---

## 1. Why Option B is the only way forward

Under Option A, `NssPressurePredictor::predict_thermal/memory/cpu`
synthesizes events from static `CanaFeatures` (no executor run):

```rust
// crates/cjc-cana-nss/src/lib.rs::synthesize_events (current Option A)
register_pressure: (ff.memory.expr_count as f64 / 256.0).min(1.0),
heap_bytes_in_use: (ff.memory.alloc_sites as u64).saturating_mul(4096),
call_depth:        ff.cfg.max_loop_depth.saturating_add(1),
branch_taken:      (i % 2 == 0) && ff.cfg.branch_count > 0,
instruction_count: 8,  // hardcoded
```

Every event's `register_pressure` and (after §6.6 Option (a) wiring)
`thermal_intensity` derive from the same `expr_count/256` formula. The
substrate distinction at the NSS layer is real (CPU vs Thermal axes are
plumbed correctly), but the underlying values are degenerate — they
land at identical magnitudes on every function.

**Consequence:** `ThermalAwareCostModel`'s penalty is structurally
inert. The bench `cana_ab_pinn` confirms this: it reports
`Three-way output byte-identical: ✓ NssPressurePredictor empty
thermal map → ThermalAwareCostModel is a no-op vs base cost model`.
The thermal-aware path is wired correctly all the way down, but
nothing it could observe is informative.

**Option B fixes this by sourcing every event field from runtime
observation of the actual MIR-exec dispatch path.** Then:

- `register_pressure` reflects real live-SSA-value counts at the event
- `thermal_intensity` reflects real FP-op density (proxy for FPU
  utilisation)
- `heap_bytes_in_use` reflects actual live tensor + buffer bytes
- `call_depth` reflects the executor's actual call-stack depth
- `branch_taken` reflects which branch the program actually took
- `io_event` / `gc_event` reflect real syscalls and GC pauses

Once values diverge per function, the thermal penalty starts producing
recommendation differences and PINN's hot kernels see real drops.

---

## 2. What stays the same

`crates/cjc-nss/src/mir_adapter.rs` — the `MirTraceEvent` struct and
`adapt_mir_trace_to_cluster_trajectory` function — is downstream of the
event source. They consume `Vec<MirTraceEvent>` regardless of where
the events came from. **No changes needed.**

The same goes for everything from the adapter onward:
- `ClusterTrajectory` → `ClusterSystemState` → pressure magnitudes
- `ThermalAwareCostModel` wrapping logic
- The ranker's consumption of penalised benefits

The whole Option A → Option B change is bounded to the event-source
end of the pipe.

---

## 3. Architecture

### 3.1 Three components

```
+--------------------+      +-------------------+     +-------------------+
| Instrumented       | ---> | TraceCollector    | --> | NssPressurePred.  |
| MIR-exec           |      | (per-thread)      |     | (consumes Vec<… >)|
+--------------------+      +-------------------+     +-------------------+
       │                              │                         │
       │ emits events via             │ owns the                │ same surface as
       │ TraceCollector::emit()       │ Vec<MirTraceEvent>      │ Option A — only
       │ at instrumentation sites     │ buffer                  │ the constructor
       │                              │                         │ changes
```

### 3.2 New types (cjc-mir-exec)

```rust
// crates/cjc-mir-exec/src/trace.rs (new file, ~300 LOC)

/// Per-thread event collector. Lives in TLS like the GradGraph arena.
/// One instance per executor invocation.
pub struct TraceCollector {
    events: Vec<MirTraceEvent>,
    tick: u64,
    enabled: bool,
}

impl TraceCollector {
    pub fn new(capacity_hint: usize) -> Self { … }
    pub fn enable(&mut self) { self.enabled = true; }
    pub fn disable(&mut self) { self.enabled = false; }
    pub fn emit(&mut self, ev: MirTraceEvent) {
        if self.enabled {
            self.events.push(ev);
            self.tick += 1;
        }
    }
    pub fn take(self) -> Vec<MirTraceEvent> { self.events }
}

thread_local! {
    static TRACE_COLLECTOR: RefCell<TraceCollector> =
        RefCell::new(TraceCollector::new(0));
}
```

### 3.3 Per-event signal sources

Each `MirTraceEvent` field needs a source in the dispatch path:

| Field | Source | Instrumentation point |
|---|---|---|
| `tick` | `TraceCollector::tick` | auto-incremented on each `emit()` |
| `block_id` | Current MIR basic block ID | Executor tracks this for the CFG path; expose via `Executor::current_block_id()` |
| `register_pressure` | Live SSA value count | `Executor::frame_live_slots()` — count populated frame slots |
| `heap_bytes_in_use` | Sum of all live tensor/buffer Rc strong counts × size | Walk the executor's reachable Value graph and sum sizes (expensive — sample, don't compute per event) |
| `call_depth` | `Executor::call_stack().len()` | already tracked for tracebacks |
| `branch_taken` | True iff last `If`/`While` cond evaluated true | emit in the cond-eval path |
| `io_event` | True iff a builtin in `cjc-runtime::io::*` was called | emit at builtin dispatch |
| `gc_event` | True iff a `Buffer::release_oldest_dead` call freed memory | emit at GC sweep |
| `instruction_count` | Number of MIR stmts executed since last event | counted by executor between events |
| `thermal_intensity` | FP-op count in last block / 8 (or similar normalization) | counted at each FP arithmetic dispatch |

The two most expensive to compute are `heap_bytes_in_use` (requires
graph walk) and `register_pressure` (requires per-slot poll). Both
should be **sampled** — compute once per N events rather than every
event. The adapter aggregates over ticks anyway, so per-event
sampling resolution doesn't change pressure magnitudes much.

### 3.4 Instrumentation sites

Roughly **9 sites** in `cjc-mir-exec/src/lib.rs`:

1. **Basic block entry** — emit event with current block_id, sampled
   register_pressure and heap_bytes_in_use, instruction_count since
   last event, branch_taken (the branch that brought us here)
2. **Call entry** — emit, increment call_depth
3. **Call exit** — emit, decrement call_depth
4. **FP arithmetic dispatch** — increment per-block FP counter (read
   into `thermal_intensity` at next block emit)
5. **Builtin dispatch (io family)** — set io_event = true for next emit
6. **Builtin dispatch (alloc family)** — adjust heap_bytes_in_use proxy
7. **GC sweep** — set gc_event = true for next emit
8. **Loop iteration boundary** — emit at top of `while` loop body
9. **Match/If branch resolution** — emit branch_taken at branch dispatch

Each site costs ~5-15 LOC of conditional `TRACE_COLLECTOR.with(|c|
c.borrow_mut().emit(...))`. Total ~150 LOC of instrumentation. The
TraceCollector itself + plumbing is another ~200 LOC.

---

## 4. The determinism wall

This is the single highest-risk part of Option B.

### 4.1 Hard constraints

CLAUDE.md states: "**Maintain deterministic execution — same seed =
bit-identical output.**" If the instrumented path produces different
program output than the un-instrumented path, the entire architecture
breaks. The MIR-exec parity tests will fire and the fix loop becomes a
nightmare.

### 4.2 What can break determinism

- **`Instant::now()`, `SystemTime`** in TraceCollector → reads system
  clock → not deterministic. **MUST NOT USE.** All "timing" derives
  from `tick`, which is event-count-based.
- **`Atomic*` operations** → ordering relative to other threads is
  non-deterministic. **MUST NOT USE.** TraceCollector is TLS, no
  contention.
- **`Rc::strong_count`** depends on which order other variables go
  out of scope → potentially non-deterministic across optimizer
  passes. Use Buffer-side accounting (track allocations explicitly)
  instead.
- **`HashMap` iteration** during event aggregation. The existing
  Option A code already uses BTreeMap everywhere — preserve that.
- **Sampling thresholds** based on `Instant` elapsed → must be
  event-count-based (`if self.tick % 16 == 0 { sample_heap(); }`).

### 4.3 What's safe

- Counting events (deterministic by definition — same program, same
  events).
- Reading `Executor::call_stack().len()` (the executor's call stack
  is already deterministic).
- Reading `frame_live_slots()` (slot allocation is deterministic).
- BTreeMap aggregation of events into ticks.

### 4.4 The parity test

Add `tests/cana_nss/test_instrumented_determinism.rs`:

```rust
#[test]
fn instrumented_run_produces_identical_output_to_uninstrumented() {
    let prog = build_pinn_program();
    let seed = 42;

    let (out_uninst, _) = cjc_mir_exec::run_program_with_executor(
        &prog, seed
    ).unwrap();

    let (out_inst, events) = cjc_mir_exec::run_program_instrumented(
        &prog, seed
    ).unwrap();

    assert_eq!(out_uninst.stdout, out_inst.stdout,
        "instrumentation must not change program output");
    assert!(!events.is_empty(),
        "instrumented run must produce events");
}
```

This is the **gate that must pass** before Option B is anywhere near
done. Run it on the PINN demo, the chess RL training program, the
PINN heat-equation fixture, and the AST/MIR parity fixture corpus.

---

## 5. Configuration

### 5.1 Feature flag vs config struct

**Recommendation:** runtime config struct, not a Cargo feature flag.

Reasoning:
- Feature flag would require recompiling cjc-mir-exec to enable
  instrumentation. That's a 10-minute incremental compile every time
  we want to A/B.
- A runtime config struct (`ExecutorConfig { trace: bool }`) lets
  the same binary do both an uninstrumented and an instrumented run
  in the same process — useful for the parity test above.
- The instrumentation cost when disabled is one boolean check per
  event site (the `if self.enabled` in `emit`). LLVM should inline
  it to a no-op fast path.

### 5.2 New entry point

```rust
// crates/cjc-mir-exec/src/lib.rs (new)
pub fn run_program_instrumented(
    program: &MirProgram,
    seed: u64,
) -> Result<(Value, Executor, Vec<MirTraceEvent>), MirExecError> {
    TRACE_COLLECTOR.with(|c| c.borrow_mut().enable());
    let result = run_program_with_executor(program, seed)?;
    let events = TRACE_COLLECTOR.with(|c| {
        c.borrow_mut().disable();
        let collector = std::mem::take(&mut *c.borrow_mut());
        collector.take()
    });
    Ok((result.0, result.1, events))
}
```

Existing `run_program_with_executor` callers are unchanged.

### 5.3 NssPressurePredictor consumes Vec\<MirTraceEvent\>

```rust
// crates/cjc-cana-nss/src/lib.rs (new)
impl NssPressurePredictor {
    pub fn from_recorded_events(
        events: Vec<MirTraceEvent>,
    ) -> Self {
        Self { events, mode: PredictorMode::Recorded }
    }
}
```

The existing constructor (synthesizing from features) stays for
backward compatibility and for the cana_ab_pinn bench's intentional
no-op-verification role.

---

## 6. Rollout sequence

Five distinct PRs, each independently mergeable:

### PR 1: TraceCollector infrastructure (~300 LOC, 1-2 days)
- New `crates/cjc-mir-exec/src/trace.rs`
- `TraceCollector` struct + TLS slot
- `run_program_instrumented` entry point
- Empty event set when enabled but no instrumentation sites yet
- Test: instrumented run produces empty event vec + identical output

### PR 2: Block-level instrumentation (~200 LOC, 2-3 days)
- Instrument basic-block entry only (most informative single site)
- Wire register_pressure (slot-count sampling) + instruction_count
- Test: instrumented PINN run produces N events where N == basic-block-execution-count
- Parity gate: stdout identical to uninstrumented

### PR 3: FP-op + call_depth + branch_taken (~150 LOC, 2-3 days)
- Add the per-FP-op counter
- Hook call entry / exit
- Set branch_taken in cond-eval path
- Test: thermal_intensity values diverge from register_pressure on a known FP-heavy workload

### PR 4: Heap + IO + GC (~150 LOC, 3-4 days)
- Buffer-side heap accounting (track allocations/frees explicitly)
- IO event flag at io-family builtin dispatch
- GC event flag at sweep
- Test: heap_bytes_in_use rises and falls as expected on a known
  alloc-heavy workload

### PR 5: NssPressurePredictor integration (~100 LOC, 1-2 days)
- New `from_recorded_events` constructor
- Wire `cana_ab_pinn` (or a new bench) to use it
- Verify: `cana_pinn_thermal_probe` reports thermal != register_pressure
  on `main` (the validation the handoff calls out)
- Update `cana_ab_pinn` to optionally use the recorded predictor and
  document that "trained vs thermal" diverges under that mode

**Total:** ~900 LOC, 9-14 days of focused work. The handoff's "2-3
weeks" budget allows for code review iteration, integration debugging,
and any architectural surprises the parity tests surface.

---

## 7. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Instrumentation breaks AST/MIR parity | **Critical** | Mandatory parity gate after each PR; revert immediately if it fires |
| Heap tracking is non-deterministic | **High** | Explicit Buffer-side counters, not Rc::strong_count |
| Performance regression on optimized builds | Medium | Conditional `if self.enabled` is single branch; bench to confirm < 5% overhead when disabled |
| Event volume overwhelms downstream adapter | Medium | Sample heavy events (heap walk) every N ticks; PR-2's test surfaces this |
| Cross-thread interactions confuse TLS arena | Low | TraceCollector is per-thread; no cross-thread sharing |
| Multiple instrumented runs on same thread pollute each other | Medium | `run_program_instrumented` clears the collector at entry and at exit |

---

## 8. Validation

After PR 5 lands, the §3.1 success criteria are:

1. **`cana_pinn_thermal_probe`** reports `thermal_intensity !=
   register_pressure` on `main` (PINN's hot kernel). Under Option A
   they're identical at 1.0; under Option B they should diverge
   because main's FP-op density differs from its slot-allocation
   density.

2. **`cana_ab_pinn` with recorded predictor** produces
   `trained vs thermal: N function(s) differ` where N > 0. The
   thermal penalty was previously inert; under Option B it actively
   drops `loop_unroll` (or future thermally-aggressive passes) on
   hot kernels.

3. **AST/MIR parity gate** continues to pass byte-identical for every
   fixture program — instrumentation MUST be transparent to program
   semantics.

4. **`cjc-runtime` and `cjc-mir-exec` test suites** continue to pass
   at their pre-Option-B counts (777/777 and ~equivalent for mir-exec).

---

## 9. What this design intentionally does NOT cover

- **Trace persistence to disk.** Events live in memory only. Persistent
  traces (for offline analysis) are a separate feature on top of this.
- **Online prediction updates.** The recorded events feed
  `NssPressurePredictor` once after a representative run; we don't
  re-run the predictor mid-compilation. Adding online updates would
  require feedback loops the current architecture doesn't have.
- **Cross-process or distributed instrumentation.** Single-process,
  single-thread for now. CJC-Lang doesn't have multi-process
  semantics anyway.
- **Replacing the synthesize_events path.** Option A stays for the
  cana_ab_pinn no-op-verification bench, which is documented as an
  intentional no-op confirmation. Option B is a parallel path.

---

## 10. Decision summary

| Question | Answer |
|---|---|
| Why now? | The substrate distinction at the NSS layer is structurally degenerate under Option A. Option B is the only fix. |
| Where? | New `crates/cjc-mir-exec/src/trace.rs` + ~9 instrumentation sites in the executor. The downstream NSS adapter is already ready. |
| Risk? | Determinism — the parity gate is the only thing standing between Option B and silent data corruption. |
| Cost? | ~900 LOC across 5 PRs, 9-14 days focused work, 2-3 weeks calendar with review. |
| Reversibility? | High — each PR is independent and the entry point `run_program_instrumented` is opt-in. Reverting is one commit per PR. |

---

*Generated as part of the §3.2 / §3.3 / §4.1 / §3.1 sweep. This is a
design doc only — no implementation in this session. The Option B work
is genuinely 2-3 weeks and should not be attempted under time pressure;
the parity-gate cost of getting it wrong is too high.*
