# §4B.2 — `cjc-cana-nss` Bridge Crate: Design Options

**Status:** decision pending. Authored 2026-06-08 after Track A
validation completed. Predecessor: `CANA_PHASE_4_NSS_INTEGRATION_DESIGN.md`
(architectural overview) and `HANDOFF_NEXT_SESSION.md` §4B.2.

---

## The decision

`crates/cjc-cana/src/pressure.rs` defines a trait:

```rust
pub trait PressurePredictor: std::fmt::Debug {
    fn predict_thermal(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64>;
    fn predict_memory_peak(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64>;
    fn predict_cpu_saturation(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64>;
    fn identify_structural_hot_kernels(&self, program: &MirProgram, features: &CanaFeatures)
        -> Vec<String>;
    fn name(&self) -> &'static str;
    fn version(&self) -> u32;
}
```

The contract: given a **static** `MirProgram` and pre-computed
`CanaFeatures`, return per-function pressure predictions in `[0.0, 1.0]`.
Deterministic. Compile-time.

NSS's actual API surface (from
[`cjc-nss/src/mir_adapter.rs`](../../crates/cjc-nss/src/mir_adapter.rs)
and [`cluster_nss.rs`](../../crates/cjc-nss/src/cluster_nss.rs)) is
**runtime-trajectory-shaped**:

```rust
pub fn adapt_mir_trace_to_cluster_trajectory(
    events: &[MirTraceEvent],
    cfg: &MirAdapterConfig,
) -> Result<AdapterOutput, NssError>;

pub struct MirTraceEvent {
    pub instruction_count: u64,
    pub register_pressure: f64,    // [0, 1]
    pub heap_bytes_in_use: u64,
    pub call_depth: u32,
    pub branch_taken: bool,
    pub io_event: bool,
    pub gc_event: bool,
    // ...
}
```

The adapter expects a stream of `MirTraceEvent`s **collected from a real
execution** of an instrumented MIR executor. The
[NSS handoff §6.1](../nss/HANDOFF_PHASE_5_COMPILER_INTEGRATION.md)
explicitly says the instrumentation in `cjc-mir-exec` does not yet exist
and would be 200-400 LOC of unscoped work.

So: **CANA wants static predictions, NSS wants runtime traces.** The
bridge must reconcile that mismatch. Four options follow, ordered from
smallest to largest implementation surface.

---

## Option C — Structural-only (smallest, my recommendation)

Implement `cjc_cana_nss::NssPressurePredictor` against NSS's structural
capabilities only. Don't synthesize traces. Don't instrument MIR-exec.

**Method status:**

| Method | Implementation |
|---|---|
| `identify_structural_hot_kernels` | Uses NSS's `LegalityVerifier` + structural CFG analysis. Returns functions whose proposed pass sequence has high "legality concern" (cooldown violations, oscillation risk). NSS's verifier is content-addressed and deterministic — works without runtime traces. |
| `predict_thermal` | Returns `BTreeMap::new()` (empty). Documents "trace required, not available in static mode". Caller (`ThermalAwareCostModel`) interprets empty map as zero adjustment. |
| `predict_memory_peak` | Same. Empty map. |
| `predict_cpu_saturation` | Same. Empty map. |
| `name`, `version` | `"nss_structural"`, `1`. |

**Code sketch:**

```rust
// crates/cjc-cana-nss/src/lib.rs (Option C, ~150 LOC total)
use cjc_cana::pressure::PressurePredictor;
use cjc_cana::features::CanaFeatures;
use cjc_mir::MirProgram;
use cjc_nss::{LegalityVerifier, LegalityConfig};
use std::collections::BTreeMap;

pub struct NssPressurePredictor {
    legality: LegalityVerifier,
}

impl NssPressurePredictor {
    pub fn new() -> Result<Self, cjc_nss::NssError> {
        let cfg = LegalityConfig {
            min_action_cooldown: 1,
            min_active_nodes: 1,
            allow_aggressive_actions: true,
            max_reversals_per_node: 1,
            initial_active_nodes: 16,  // arbitrary — verifier handles dynamic
        };
        Ok(Self { legality: LegalityVerifier::new(cfg)? })
    }
}

impl PressurePredictor for NssPressurePredictor {
    fn identify_structural_hot_kernels(
        &self, program: &MirProgram, features: &CanaFeatures,
    ) -> Vec<String> {
        // Pure structural — no traces needed.
        // Functions with high loop_depth × branch_count × call_depth are hot.
        // Cross-check with NSS's legality verifier on a hypothetical
        // optimization script: functions whose proposed pass sequence
        // would oscillate per the verifier are also surfaced.
        program.functions.iter()
            .filter(|f| {
                let Some(fn_feat) = features.per_fn.get(&f.name) else { return false };
                fn_feat.cfg.max_loop_depth >= 2 && fn_feat.cfg.branch_count >= 2
            })
            .map(|f| f.name.clone())
            .collect()
    }

    fn predict_thermal(&self, _: &MirProgram, _: &CanaFeatures)
        -> BTreeMap<String, f64> { BTreeMap::new() }
    fn predict_memory_peak(&self, _: &MirProgram, _: &CanaFeatures)
        -> BTreeMap<String, f64> { BTreeMap::new() }
    fn predict_cpu_saturation(&self, _: &MirProgram, _: &CanaFeatures)
        -> BTreeMap<String, f64> { BTreeMap::new() }
    fn name(&self) -> &'static str { "nss_structural" }
    fn version(&self) -> u32 { 1 }
}
```

**Tests required (~6-10):**
- `identify_structural_hot_kernels` returns expected functions on
  hand-crafted MIR.
- Empty-map methods return empty maps unconditionally.
- Determinism: same input → byte-identical output across runs.
- `ThermalAwareCostModel<LinearCostModel, NssPressurePredictor>`
  composes cleanly — empty thermal map produces zero cost-model
  adjustment.

**Effort:** 1-2 days (1 day implementation + 1 day tests +
ADR write-up).

**Pros:**
- No MIR-exec instrumentation needed.
- No synthetic-trace approximation.
- Doesn't change the `PressurePredictor` trait contract.
- Honest about what NSS can and cannot do in static mode.
- Unblocks §4B.3 (`--thermal-aware` CLI flag) — the flag wires up the
  ThermalAwareCostModel even when the predictor returns empty maps; the
  composition is just a no-op.
- Future-proof: option A or B can be layered on top later without
  rewriting C.

**Cons:**
- Doesn't actually use NSS's predictor. The bridge crate exists but
  only exercises NSS's structural-analysis surface.
- "Bridge in name only" — exposed methods don't deliver NSS predictions.

**Why I recommend it:**

§3A.2 demonstrated that the **base** cost model isn't yet producing
behavioural differences on real workloads (trained vs default both
skip everything on PINN). Adding a thermal layer on top of an inactive
base is solving a problem we don't yet have. Option C ships the bridge
surface, gets `--thermal-aware` wireable, and defers the
synthetic-vs-instrumentation choice until we have a workload where the
base ranker actually makes decisions worth refining.

---

## Option A — Synthetic trace from CANA features

Implement `predict_*` methods by **synthesizing** a `Vec<MirTraceEvent>`
from `CanaFeatures` at predict time, then feeding it through NSS's
adapter and predictor.

**Synthesis sketch:**

```rust
fn synthesize_trace(
    program: &MirProgram, features: &CanaFeatures,
) -> Vec<MirTraceEvent> {
    let mut events = Vec::new();
    let mut tick = 0u64;
    for func in &program.functions {
        let Some(fn_feat) = features.per_fn.get(&func.name) else { continue };
        // One synthesized event per "estimated basic block window":
        let n_blocks = (fn_feat.memory.expr_count / 16).max(1);
        for _ in 0..n_blocks {
            events.push(MirTraceEvent {
                instruction_count: tick,
                register_pressure: synth_reg_pressure(fn_feat),
                heap_bytes_in_use: synth_heap(fn_feat),
                call_depth: fn_feat.cfg.max_call_depth.into(),
                branch_taken: tick % 2 == 0,  // arbitrary
                io_event: false,
                gc_event: false,
            });
            tick += 16;
        }
    }
    events
}

fn synth_reg_pressure(f: &FnFeatures) -> f64 {
    // Heuristic: expr_count_in_function / max_window
    (f.memory.expr_count as f64 / 64.0).min(1.0)
}

fn synth_heap(f: &FnFeatures) -> u64 {
    // Heuristic: alloc_sites × tensor_default_size + base
    (f.memory.alloc_sites as u64) * 4096 + 1024
}
```

Then `predict_thermal` calls:

```rust
let trace = synthesize_trace(program, features);
let adapter_out = adapt_mir_trace_to_cluster_trajectory(&trace, &cfg)?;
let mut thermal = BTreeMap::new();
for func in &program.functions {
    // Get the corresponding NSS NodeId, call nss.predict_next(...),
    // extract thermal pressure component.
    let nss_node_id = self.fn_to_node_id(&func.name);
    let prediction = self.nss.predict_next(&adapter_out.trajectory, nss_node_id);
    thermal.insert(func.name.clone(), prediction.thermal_pressure);
}
thermal
```

**Pros:**
- Self-contained — no MIR-exec changes.
- Trait contract unchanged.
- Deterministic by construction.
- Exercises NSS's predictor (some real value).

**Cons:**
- Predictions are **predictions of predictions**. The synthetic trace
  is a guess; NSS's predictions on a guess are at best heuristic.
- Loses NSS's biggest advantage — real runtime signal.
- The synthesis heuristics need tuning to match real traces. Without
  ground truth (which Option B provides), there's no way to validate
  the synthesis is faithful.
- Risk of "validated nothing": both the synthesis and the predictor
  could be wrong in complementary ways and the user has no way to
  tell.

**Effort:** 3-5 days. Most of the time is iteration on synthesis
heuristics + tests + bench against a known-output workload.

**When to choose A:** if you want NSS visible in the compiler today,
have a use case for thermal predictions specifically, and accept that
the predictions will be heuristic until Option B lands.

---

## Option B — Instrument cjc-mir-exec, get real traces

Add `MirTraceCollector` to `cjc-mir-exec`, emit events from the
instruction-dispatch loop, then change `PressurePredictor` to take an
execution context (or run a profiling pass internally) to collect real
traces.

**Required work:**

1. **Trace collector trait + impl** in `cjc-mir-exec` (~150 LOC).
2. **Instrumentation hooks** in `cjc-mir-exec`'s instruction loop —
   call collector on each `MirInst::*` (~100 LOC, hot-path discipline).
3. **Determinism guards** — collector output must be byte-identical
   across runs/platforms. Substream salt per executor instance,
   BTreeMap iteration, no FMA. (~50 LOC + tests.)
4. **Trait redesign** in `cjc-cana::pressure`:
   ```rust
   fn predict_thermal(
       &self,
       program: &MirProgram,
       features: &CanaFeatures,
       profile: &MirExecutionProfile,  // ← NEW: real trace
   ) -> BTreeMap<String, f64>;
   ```
   Or: keep current signature but have `NssPressurePredictor::new`
   take a callback that produces a profile on demand.
5. **Bridge implementation** — feed real `Vec<MirTraceEvent>` through
   the adapter, call NSS, return predictions. (~100 LOC.)
6. **Tests** — round-trip from a small CJC-Lang program through
   collector → adapter → NSS → predictions. (~10-15 tests.)

**Pros:**
- Real runtime signal — NSS's intended use.
- Aligns NSS's actual API design with its CANA consumer.
- Captures everything synthesis can't: IO bursts, GC spikes, branch
  unpredictability.
- Once landed, every NSS feature (predictor, advisor, autonomous
  optimizer) becomes available to CANA, not just predictions.

**Cons:**
- 200-400 LOC of instrumentation in `cjc-mir-exec` — touches the
  hottest path in the entire compiler. Every instruction-loop iteration
  has to call (or skip) the collector. Performance overhead must be
  measured.
- Trait contract changes — every existing `PressurePredictor` impl
  (currently just `NullPressurePredictor`) must update.
- Calling the trait method now triggers an execution. Slow,
  side-effecting. Compile-time predictions become compile-time
  predictions-after-profiling.
- "Compile-time decisions inform runtime decisions" feedback loop —
  if the predictor's profiling-run goes through optimization, you have
  a circular dependency between the cost model and the profile.
  Solvable (use a no-opt profiling run) but non-trivial.

**Effort:** 2-3 weeks. Most of the time is the instrumentation
hot-path discipline + determinism testing + trait redesign migration.

**When to choose B:** if NSS's value to the compiler depends on real
runtime signal (it largely does), if you have time for the larger
investment, and if the eventual hot/warm/cool kernel codegen (§4B.4)
is genuinely on the roadmap (which needs real pressure data).

---

## Option D — Hybrid: synthetic for predict, NSS verifier for kernels

Use Option A's synthetic trace for `predict_thermal/memory/cpu`. Use
Option C's structural + legality-verifier approach for
`identify_structural_hot_kernels`.

**Pros:**
- Lower commitment than B, more capability than C.
- Two NSS surfaces exposed (predictor + verifier).
- Honest about what's heuristic (the predict methods) vs structural
  (the kernel identification).

**Cons:**
- Same synthesis weakness for predictions as Option A.
- More moving parts to maintain.

**Effort:** 5-7 days. Combines A's synthesis cost with C's structural
cost.

**When to choose D:** if you want a "middle path" that exercises more
of NSS than C without committing to B's full instrumentation lift.

---

## Decision matrix

| Option | LOC | Days | Real signal? | Affects mir-exec? | Trait change? |
|---|---|---|---|---|---|
| C — Structural only | ~150 | 1-2 | No (structural) | No | No |
| A — Synthetic trace | ~500 | 3-5 | Heuristic | No | No |
| D — Hybrid | ~600 | 5-7 | Heuristic + structural | No | No |
| B — Real instrumentation | ~1500 | 14-21 | Yes | Yes (200-400 LOC) | Yes (likely) |

---

## My recommendation: Option C, with stated plan to extend later

Three reasons, ordered by weight:

### 1. §3A.2 just demonstrated the base cost model is inactive on real workloads.

The PINN AB test showed that trained_ranker and default_ranker
**agree** to skip every candidate pass on a real workload. The
underlying ranker isn't yet making decisions that need refining. Adding
a thermal-aware layer on top of a layer that produces no decisions is
solving a problem we don't yet have.

When the base ranker *does* start producing differential decisions
(more diverse corpus, more diverse workloads, different skip
thresholds), then a thermal-aware refinement layer becomes valuable.
Until then, Option C ships the surface, defers the heavy lift.

### 2. Option C's structural method delivers real value standalone.

`identify_structural_hot_kernels` is genuinely useful even without
trace-based predictions. It surfaces functions that are
structurally-likely-to-be-hot (high loop depth + branch density) — the
same information you'd want to drive `KernelVariant::select` in
§4B.4's hot/warm/cool variants, derived from CFG analysis you already
have.

The other three methods returning empty maps is honest. The contract
in `pressure.rs` already accommodates empty returns —
`ThermalAwareCostModel` interprets empty as "no adjustment" rather
than failing.

### 3. Option B is the right long-term answer, but premature.

The 200-400 LOC of cjc-mir-exec instrumentation is the right
investment **when**:
- The base cost model is producing useful but improvable decisions.
- A specific workload demonstrates that runtime pressure matters
  (e.g. chess RL with large tensors → memory pressure matters).
- The §4B.4 hot/warm/cool kernel codegen has a consumer.

None of those are true today. Building B today is building
infrastructure for a load that doesn't exist yet, and 2-3 weeks of work
is enough cost to want certainty about the use case first.

### Concrete next steps for Option C

If you agree with this recommendation:

1. **Scaffold `crates/cjc-cana-nss/`** (1 day):
   - `Cargo.toml` with dependencies on `cjc-cana`, `cjc-nss`, `cjc-mir`.
   - `src/lib.rs` with `NssPressurePredictor` struct + trait impl per
     the sketch above.
   - Hook into NSS's `LegalityVerifier` for structural validation of
     hot-kernel candidates.

2. **Tests** (1 day):
   - Determinism: same MIR program → byte-identical hot-kernel list.
   - Composition: `ThermalAwareCostModel<LinearCostModel,
     NssPressurePredictor>` produces a `PassRanker` that ranks
     identically to `default_ranker()` (because the empty thermal map
     is a no-op).
   - Bench: run on PINN, verify hot-kernels list matches structural
     expectation (e.g. `forward` is identified as hot due to nested
     loops).

3. **§4B.3 CLI flag** (1 day, follows §4B.2):
   - Add `--thermal-aware` to `cjcl run` / `cjcl compile`.
   - When set, replace `default_ranker()` with `PassRanker::new(
     ThermalAwareCostModel::new(LinearCostModel::trained(),
     NssPressurePredictor::new()?), DefaultLegalityGate::new())`.
   - On PINN, this flag is a no-op (per §3A.2). Documented as
     "wired but inactive until base ranker exercises decisions."

4. **ADR** (~1 hour):
   - Write ADR-XXXX recording the decision: chose C because the base
     ranker is inactive on real workloads, deferred A/B until that
     changes.

**Total: ~3 days for §4B.2 + §4B.3 under Option C.**

When the base ranker starts making differential decisions (which
either §3A.2 follow-up on chess RL or a corpus expansion will reveal),
re-evaluate whether to layer A, B, or D on top of C.

---

## What if you'd rather pick A, B, or D?

- **If A:** I'd recommend starting with `identify_structural_hot_kernels`
  done structurally (cheapest method, real value) plus
  `predict_thermal` done synthetically. Get to a working end-to-end
  flow on the simplest method, then expand.

- **If B:** Begin with the cjc-mir-exec instrumentation as a separate
  PR before touching cjc-cana-nss at all. The instrumentation is the
  load-bearing work; the bridge crate is downstream. Land the
  instrumentation, validate it's deterministic, *then* build the bridge
  against real traces.

- **If D:** Build C first (1-2 days), then add A's synthesis for the
  predict methods. That sequencing keeps C's value (structural
  hot-kernel) intact even if A's synthesis turns out to be
  unfaithful.

---

## What this doc does NOT cover

- **§4B.5 second-NN composition.** The NSS handoff §5 lists three
  composition points (audit trace, pressure interlingua, learned
  feature extractor). Picking one of those depends on what the second
  NN's role is — that's a separate conversation. None of A/B/C/D
  preclude any of the §4B.5 choices.

- **§4B.4 hot/warm/cool kernel codegen.** That's the largest scope on
  the roadmap (estimated 1 week per fused primitive). It needs runtime
  pressure data, which means it needs Option B eventually. Tracking it
  as a downstream dependency, not a parallel decision.

---

*Next action: pick A / B / C / D and either drop me a one-liner or
re-open this with the choice. C is implementable today; A and D are
implementable this session if you want to push; B is multi-session
work.*
