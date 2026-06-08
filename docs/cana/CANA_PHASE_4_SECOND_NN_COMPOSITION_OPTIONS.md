# §4B.5 — Second-NN Composition with NSS: Design Options

**Status:** **partially shipped** — see §0 below. Authored 2026-06-08,
revised same day after a sharp reader observation forced a re-framing.
Sibling docs:
[`CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md`](CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md)
(§4B.2 bridge crate) and
[`CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md`](CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md)
(§4B.4 kernel variants).

---

## 0. Already shipped: Option 5 — Downstream-consumer composition (CANA ← NSS)

**The most obvious composition was missing from the original four
options.** The first draft of this document enumerated three options
where the "second NN" was a separate entity *producing* outputs into
NSS (audit-trace consumer, pressure-field interlingua, learned-feature
extractor) plus a fourth alternative with no NSS coupling. None of
those describe what we've already built. Option 5 — the
**downstream-consumer** pattern — does:

```
Compile  →  MIR  →  CANA features
                       ↓
                  NSS Pressure-                  ↓
                  Predictor                  CANA's
                       ↓                  LinearCostModel
                  pressure_map  →  ThermalAwareCostModel
                                            ↓
                                       adjusted_cost
                                            ↓
                                       CANA's PassRanker
                                            ↓
                                         PassPlan
```

The second NN (CANA's trained `LinearCostModel`) is **downstream** of
NSS in the data flow. NSS produces pressure predictions; CANA's
`ThermalAwareCostModel` reads those predictions and uses them to adjust
its own per-pass benefit estimates; CANA's `PassRanker` then produces
the `PassPlan` consumed by `cjc-mir-exec`.

This is what `cjc-cana-nss::NssPressurePredictor` +
`cjc_cana::thermal_cost_model::ThermalAwareCostModel` already implement
end-to-end, wired through `cjcl run --thermal-aware` as of §4B.3 and
validated three-way byte-identical in §15.

### What "second NN" reads naturally

CANA's `LinearCostModel` is a learned model:

- Coefficients fit by OLS gradient descent on a 73-program corpus
  (`bench/cana_train_cost_model/`).
- Takes 4 features per `(function, pass)` pair (expr_count, loop_depth,
  branch_count, alloc_sites).
- Produces per-pass benefit predictions in `[0, 1]`.
- Is not a *deep* neural network (no hidden layers, no nonlinearities)
  but is unambiguously a trained statistical model.

If "the second NN" in the NSS handoff §5 refers to CANA — and based on
the architectural fact that CANA is the existing compile-time learned
model wired to NSS — then the composition pattern has been chosen and
shipped. It's Option 5.

### What's open under this reading

The composition *pattern* is fixed. What's not yet active is the
**signal flow**:

- `NssPressurePredictor` in Option C mode returns empty thermal maps.
- `ThermalAwareCostModel.predict_thermal(...).get(name).unwrap_or(0.0)`
  evaluates to 0.0 for every function.
- So the thermal adjustment is a no-op today, exactly matching the
  §3A.2 and §15 byte-identity findings.

The composition activates when:

1. `NssPressurePredictor` migrates from Option C to Option A (synthetic
   trace, see §4B.2 doc) or Option B (real MIR-exec instrumentation),
   AND
2. The base ranker starts producing differential decisions on real
   workloads (corpus expansion or threshold tuning),

after which the existing `ThermalAwareCostModel` plumbing will start
producing measurably different `PassPlan`s under `--thermal-aware` vs
`--mir-opt`. No further composition-architecture work is needed at
that point; the wiring is already correct.

### When the remaining options below DO apply

The four options in §1-§4 below describe how to compose a
**genuinely additional** learned model with NSS — one that isn't
CANA's `LinearCostModel`. Examples:

- A deep network trained on PINN-shaped programs to predict per-pass
  benefit better than CANA's 4-feature linear model.
- A learned static-analysis model that predicts thermal pressure
  directly from MIR shape (effectively Option A for §4B.2's
  `NssPressurePredictor`, but with the prediction sourced from an ML
  model instead of a hand-coded synthesis heuristic).
- An ML-driven kernel-variant selector that replaces
  `KernelVariant::select_for_budget` (§4B.4 Option α's
  `PressureAwareSelector`) with a learned decision boundary.

In any of those scenarios, the four options below are the design
choices for how that **new** model composes with the existing
NSS + CANA pipeline. So they're still relevant — they just don't
describe the CANA ↔ NSS bridge, which is already done.

---

The four options that follow describe **genuinely separate**
neural networks (not CANA) and how each would compose with NSS. If
you're not planning to add such a model, the only thing this doc
contributes beyond §0 is forward-compatible thinking for a possible
future.

---

## The decision

The NSS Phase-5 handoff doc
([`docs/nss/HANDOFF_PHASE_5_COMPILER_INTEGRATION.md`](../nss/HANDOFF_PHASE_5_COMPILER_INTEGRATION.md))
mentions:

> §5. Companion network — what the next session should know
>
> You mentioned working on **another neural network** that will improve
> the CJC-Lang compiler alongside NSS. A few integration points NSS
> leaves available...

This is §4B.5 from the CANA handoff: how does a *second* neural
network (the one you're building alongside NSS) compose with NSS at
the compiler-integration layer?

The NSS handoff names three composition points; this doc gives each
one a concrete code sketch + the trade-offs + the test surface they
imply, so you can pick.

### What I need to know from you

Before §4B.5 can be implemented, I need answers to:

1. **What does the second NN actually predict?** Compile-time decisions
   (which passes to run)? Runtime decisions (which kernel variant to
   call)? Something orthogonal (failure prediction, memory layout,
   pipeline schedule)?
2. **What's its input shape?** Raw MIR? CANA features? NSS prediction
   audit trails? Something else?
3. **What's its output shape?** A vector? A categorical label? A
   pressure-domain projection? A pass-plan modification proposal?
4. **Determinism constraint?** Is the second NN trained on
   deterministic data and required to produce deterministic outputs?
   Or is it OK for it to be non-deterministic as long as its outputs
   are captured in NSS's audit trail?

The composition point that's right for you depends on those four
answers. The three options below correspond to three plausible
combinations.

---

## Option 1 — Audit-trace consumer (lowest coupling, recommended starting point)

The second NN **reads** NSS's `DecisionRecord` stream as a feature
input. NSS produces predictions + audit traces; the second NN consumes
those traces and produces its own predictions downstream.

### How it composes

```
Compile  →  MirTraceEvent[]  →  NSS  →  AdvisoryRanking +
                                        DecisionRecord[]  →  Second NN
                                                              ↓
                                                         (your output)
```

The second NN doesn't touch NSS internals. It reads NSS's
content-addressed audit log and treats each `DecisionRecord` as a
training example or a runtime feature.

### What's in a `DecisionRecord`

Per NSS's handoff §5.1, each record carries:

- `run_id` — content-addressed input fingerprint
- `recommended` action + `recommended_collapse` probability
- `baseline_collapse` (DoNothing baseline) for delta computation
- `confidence_margin`
- `outcome` (Applied / Skipped / NoOp)
- `skip_reason` (interpretable text reason)
- `applied_interventions` (the actual interventions that fired)

The second NN can train to predict **the safety-layer's decision** —
useful for learning when to trust NSS's recommendation.

### Code sketch

```rust
// crates/cjc-cana-nss-companion/src/lib.rs (new crate, ~250 LOC)
pub trait CompanionPredictor: std::fmt::Debug {
    /// Given a recent NSS audit trail, produce a downstream prediction.
    fn predict(&self, audit: &[cjc_nss::DecisionRecord]) -> CompanionVerdict;
}

pub struct CompanionVerdict {
    pub trust_nss: f64,  // [0, 1] — how much to weight NSS's recommendation
    pub adjustment: Option<NssActionOverride>,
}
```

A wrapper around `SchedulerAdvisor` consults the companion before
applying recommendations:

```rust
let nss_ranking = scheduler_advisor.recommend(...);
let verdict = companion.predict(&recent_audit_buffer);
let action = if verdict.trust_nss > 0.5 {
    nss_ranking.top_choice()
} else {
    verdict.adjustment.unwrap_or_else(|| AdvisoryAction::DoNothing)
};
```

### Pros

- **Lowest coupling.** Second NN doesn't depend on NSS's encoder
  architecture; it just reads serializable output.
- **Easy to swap.** Multiple companion models can be A/B tested
  against the same NSS deployment.
- **Determinism boundary is clean.** Second NN can be
  non-deterministic (e.g. uses HashMap iteration, system RNG) — its
  outputs become an *input* to NSS-like decisions, captured in NSS's
  next audit step. The non-determinism is isolated to the second NN.
- **Bridge crate is small.** ~250 LOC for the companion trait +
  default adapter + tests.

### Cons

- **The second NN is downstream of NSS's recommendations.** It can
  only override or scale NSS verdicts — it can't influence what
  NSS itself sees.
- **Audit data is lossy.** `DecisionRecord` summarizes a tick's
  worth of NSS reasoning; if your second NN needs raw trajectory
  context, this is the wrong layer.

### When to choose Option 1

If the second NN is essentially a **policy/meta-learner** that
decides when to trust NSS, or a **post-NSS adjustment layer** that
refines NSS's output for a specific workload.

**Effort:** 1-2 weeks.

---

## Option 2 — Pressure-field interlingua (medium coupling)

The second NN's output projects onto NSS's `PressureKind` substrate
the same way `MirAdapter` projects MIR events onto pressures. NSS then
treats the second NN's output as an additional input source.

### How it composes

```
Compile  →  MirTraceEvent[]  →  MirAdapter ──┐
                                              ├──→  NSS  →  AdvisoryRanking
Other source  →  Second NN  →  PressureKind ──┘
```

The second NN runs in parallel with `MirAdapter` and produces values
keyed by `PressureKind` (CPU / Memory / Thermal / IO / GC / etc).
NSS's encoder concatenates both projections.

### Code sketch

```rust
// crates/cjc-cana-nss-companion/src/lib.rs (different shape from Option 1)
pub struct PressureFieldOutput {
    pub cpu: f64,
    pub memory: f64,
    pub thermal: f64,
    pub io: f64,
    pub gc: f64,
    // Match cjc-nss's PressureKind enum exactly.
}

pub trait CompanionAdapter: std::fmt::Debug {
    fn project(&self, input: &CompanionInput) -> PressureFieldOutput;
}

// In NSS integration:
let companion_pressure = companion.project(&companion_input);
let mir_pressure = MirAdapter::project(&mir_events);
let combined = combine_pressure_fields(mir_pressure, companion_pressure);
let prediction = nss.predict_next(combined);
```

### Determinism constraint

This is the strict path: per NSS handoff §5.3, if the second NN is
non-deterministic, it'll **break NSS's audit-trace contract**. You
either:

- Make the second NN deterministic (same `NssSeed::substream` pattern
  + BTreeMap iteration + Kahan reductions + no FMA), OR
- Treat the second NN's output as an *input* to NSS, captured in NSS's
  next `input_hash`. Same effective constraint, different framing.

### Pros

- **Second NN's signal reaches NSS directly.** No reliance on NSS's
  recommendation surface as a downstream filter.
- **Multiple second-NNs compose.** Two or three companions can each
  produce `PressureFieldOutput` and combine via averaging or learned
  weighting.
- **Pressure-field interlingua is closed at the kind level.** No
  custom enum variants needed.

### Cons

- **Strict determinism contract** on the second NN. Either make it
  deterministic from the start, or accept the constraint of capturing
  its non-determinism in the audit hash.
- **Encoder layer needs modification.** NSS's `SystemEncoder` is
  built for one pressure-projection source today. Adding a second
  source requires extending `EncoderConfig`.
- **Effort is higher.** ~3-5 weeks because of the encoder work +
  determinism testing.

### When to choose Option 2

If the second NN produces a **first-class compile-time signal** that
NSS should weigh alongside MIR-trace pressure, and you're willing to
make the second NN deterministic. Example: a learned static-analysis
model that predicts thermal pressure from MIR structure better than
the synthetic trace from Option A of §4B.2.

**Effort:** 3-5 weeks.

---

## Option 3 — Learned feature extractor (highest coupling)

The second NN is wrapped in a `LearnedPressureField` adapter that
plugs directly into NSS's encoder layer. NSS's encoder concatenates
the second NN's learned feature vector with the standard
`PressureField`.

### How it composes

```
Compile  →  MirTraceEvent[]  →  MirAdapter  →  PressureField
                                                     ↓
                                                NSS encoder
                                                     ↑
                          Second NN  →  Learned feature vector
```

### Code sketch

```rust
// In cjc-nss (would need to land there, not in cjc-cana-nss-companion).
pub struct LearnedPressureField {
    canonical: PressureField,
    learned: SmallVec<[f64; 16]>,  // Second NN's fixed-size output.
}

// EncoderConfig gains:
pub struct EncoderConfig {
    pub use_learned_features: bool,
    pub learned_feature_dim: usize,
    // ...
}
```

### Pros

- **Tightest integration.** Second NN becomes part of NSS's
  representation pipeline.
- **End-to-end learning.** With appropriate plumbing, gradients could
  flow through the encoder back into the second NN.
- **No serialization between models.** Second NN's output is a fixed-
  size vector, processed in-memory.

### Cons

- **Changes NSS's internal contracts.** `LearnedPressureField` is a
  new variant of NSS's pressure substrate. Touches `nss.rs`,
  `encoder.rs`, `cluster_nss.rs`, `mir_adapter.rs`. Breaks NSS's
  current "no new pressure-kind variants" invariant from §4 of the
  NSS handoff.
- **Determinism contract is strict and load-bearing.** Same as Option
  2 but more so — the second NN's gradients (if used) need to be
  reproducible across runs.
- **Architecture risk.** If the second NN's design changes (e.g. its
  output dimension changes from 16 to 32), this requires re-validating
  every NSS test.

### When to choose Option 3

If the second NN is being **co-designed with NSS** as a single
combined system, and you're committed to a long-term integration
where both networks evolve together.

**Effort:** 6-10 weeks. This is genuinely architectural work.

---

## Decision matrix

| Option | LOC | Weeks | Couples to | Determinism constraint |
|---|---|---|---|---|
| 1 — Audit-trace consumer | ~250 | 1-2 | NSS public output only | None (non-det second NN OK) |
| 2 — Pressure-field interlingua | ~700 | 3-5 | NSS encoder config | Strict |
| 3 — Learned feature extractor | ~1500 | 6-10 | NSS encoder internals | Strict + gradient-stable |

---

## My recommendation: Option 1, with a question

Same pattern as §4B.2 and §4B.4: ship the lowest-coupling option,
defer higher-coupling decisions until they're driven by evidence.

For §4B.5 specifically, Option 1 has an extra advantage: it lets the
second NN ship **without** the second NN being deterministic. That
matters because most neural networks in development today are not
fully deterministic — between PyTorch's stochastic CUDA kernels, the
loss-of-determinism in optimization gradients, and the cost of
deterministic training, a brand-new model is unlikely to meet
NSS's strict determinism contract from day one.

Option 1 isolates that non-determinism behind a clean serialization
boundary. The second NN can be the messy reality-of-ML model it
wants to be, and NSS-side determinism stays clean.

### But this only applies if you're adding a NEW model

Per §0 above, if "second NN" means CANA's existing `LinearCostModel`,
the composition is done. The four options 1-3 (plus 4 as the no-NSS
alternative) describe how a **future, additional** learned model would
compose. If no such model is being added, this entire doc beyond §0
is forward-compatible thinking, not a pending decision.

If you ARE planning an additional model on top of CANA + NSS:

1. What does the new model predict?
2. What does it consume as input?
3. Is it deterministic (BTreeMap iteration, no FMA, fixed RNG)?
4. Is it co-trained with NSS or developed separately?

The right composition point genuinely depends on those answers. With
them I can write a focused implementation design doc the same way
§4B.2 went. Without them, scaffolding would be a guess.

### Next-step proposal

If you don't have a separate model in flight: **§4B.5 is done as
Option 5**. The composition pattern is shipped. Activation happens
automatically when §4B.2 migrates from Option C to A or B. No further
work on this item is needed today.

If you DO have a separate model:

1. **You send a short note** answering the four questions above.
2. **I write a focused §4B.5 design doc** for the matched composition
   point, with code sketches grounded in your model's actual
   input/output/determinism story.
3. **Decide whether to scaffold** (Option 1's bridge crate is
   probably 1-2 days; Option 2's encoder work is genuinely
   multi-session; Option 3 is a project).

---

## Open architectural question

There's an alternative I'm not formally listing as Option 4 because
it sidesteps NSS entirely:

> **Option 4 — Independent second-NN, no NSS coupling at all.**
>
> The second NN runs as its own compiler-side analysis pass, reading
> MIR + CANA features, producing pass-plan modifications that compose
> with — but don't go through — NSS. The two networks are independent
> consumers of MIR; their outputs are combined at the pass-ranker
> level.

This is the "two parallel rankers" architecture. It's the simplest
to ship (no NSS integration work) but loses the audit-trail
composition. If the second NN's role is "another ranker" rather than
"a complement to NSS's predictions," Option 4 might be the cleanest
answer.

Mentioning here so it's on the table even though the NSS handoff
didn't formally enumerate it.

---

*Next action: send me the four answers, and I'll write the matched
implementation design doc. Or pick Option 1 / 2 / 3 / 4 directly if
you've already decided.*
