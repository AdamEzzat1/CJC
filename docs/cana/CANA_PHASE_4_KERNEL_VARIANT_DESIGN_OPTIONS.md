# §4B.4 — Hot/Warm/Cool Kernel Variant Codegen: Design Options

**Status:** decision pending. Authored 2026-06-08 after §4B.3 landed
`cjcl run --thermal-aware`. Predecessor:
[`CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md`](CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md)
(the §4B.2 bridge crate design doc, which this mirrors).

---

## The decision

[`crates/cjc-cana/src/kernel_variant.rs`](../../crates/cjc-cana/src/kernel_variant.rs)
defines the `KernelVariant` enum (Hot / Warm / Cool) and a
`select_for_budget` heuristic. Three fused primitives shipped in
Phase 3.5 — `fused_matmul_dot`, `fused_matmul_norm`,
`fused_matmul_matmul` — currently exist in only one form (Hot, fully
fused).

The §4B.4 plan from the handoff calls for emitting all three variants
per primitive and letting the runtime select at call time based on
NSS-observed pressure. Concretely:

| Variant | Implementation |
|---|---|
| **Hot**  | The fully-fused kernel shipping today. Peak performance, peak heat. |
| **Warm** | Partially fused: keep the matmul as a primitive call but skip the intermediate `Tensor` wrapper for the second op. ~30% slower than Hot, ~30% cooler. |
| **Cool** | MIR-walked: the unfused chain `matmul(a, w)` then `norm(h)`. Slowest, lowest thermal pressure. |

The handoff's effort estimate: **1 week per fused primitive**, or
~3 weeks for all three. This is genuinely multi-session work — the
biggest item on the remaining roadmap.

### Two architectural unknowns

Before per-primitive Warm/Cool implementations land, two architectural
questions need answers:

1. **Where does the selector live?** In `cjc-cana` next to
   `KernelVariant`? In `cjc-runtime` next to the kernel dispatch
   tables? In `cjc-cana-nss` next to the predictor?
2. **How does the selector reach the runtime call site?** A new
   thread-local? A parameter on every fused primitive's dispatch
   arm? A capability passed into `MirExecutor::new`?

This doc enumerates the smaller-scope decisions (the trait surface
that answers question 1) and defers question 2 to the first
implementation phase.

Four options follow, ordered from smallest to largest implementation
surface.

---

## Option α — Trait scaffolding only (smallest, my recommendation)

Ship the `KernelVariantSelector` trait plus an `AlwaysHotSelector`
default impl. No per-primitive Warm/Cool implementations. Runtime
dispatch stays unchanged.

**What gets built:**

```rust
// crates/cjc-cana/src/kernel_variant.rs (extended)

pub trait KernelVariantSelector: std::fmt::Debug {
    /// Pick which variant of `kernel_name` to call. `predicted_thermal`
    /// is in [0.0, 1.0]. Implementations should be deterministic for a
    /// given (kernel_name, predicted_thermal) — selector outputs feed
    /// into compile-time decisions (when the codegen tier exists) AND
    /// runtime dispatch (when the runtime tier exists).
    fn select(&self, kernel_name: &str, predicted_thermal: f64) -> KernelVariant;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AlwaysHotSelector;
impl KernelVariantSelector for AlwaysHotSelector {
    fn select(&self, _: &str, _: f64) -> KernelVariant { KernelVariant::Hot }
    fn name(&self) -> &'static str { "always_hot" }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PressureAwareSelector;
impl KernelVariantSelector for PressureAwareSelector {
    fn select(&self, _kernel: &str, predicted_thermal: f64) -> KernelVariant {
        KernelVariant::select_for_budget(1.0 - predicted_thermal)
    }
    fn name(&self) -> &'static str { "pressure_aware" }
}
```

**Tests required (~5):**
- `AlwaysHotSelector` returns `Hot` regardless of input.
- `PressureAwareSelector` returns `Hot` when thermal_pressure ≤ 0.7,
  `Warm` when in `(0.7, 0.9]`, `Cool` when > 0.9. Derived from
  `select_for_budget`'s thresholds.
- Determinism: same `(kernel_name, predicted_thermal)` → byte-identical
  output for both selectors.
- Audit labels are stable strings.

**Effort:** 1 day (most of which is doc + trait + a few unit tests).

**Pros:**
- The trait surface exists in the workspace. Future Warm/Cool
  implementations have a place to plug in.
- `PressureAwareSelector` is a real implementation that takes NSS
  thermal predictions and maps them to variants. Today
  `NssPressurePredictor::predict_thermal` returns empty maps, so
  in composition the selector falls back to `select_for_budget(1.0)
  → Hot` for every call — behavioural no-op vs today.
- Composition with `NssPressurePredictor` already validated by §4B.2/§15.
- No runtime changes. Every existing fused primitive still routes to
  its Hot kernel.

**Cons:**
- Doesn't actually emit Warm or Cool kernels. The selector machinery
  produces verdicts that no consumer uses.
- "Trait in name only" — same critique as Option C for §4B.2.

**Why I recommend it:**

Same reason as the §4B.2 Option C recommendation, with extra force:

1. **The §3A.2/§15 result.** The base ranker is currently inactive on
   real workloads. The thermal-aware path is wired but produces zero
   adjustment. Building the Cool kernel for a primitive that's already
   producing optimal Hot code in 100% of compiles is solving a problem
   we don't yet have.
2. **NSS isn't predicting thermal pressure yet.** Until
   `NssPressurePredictor` migrates from Option C (empty maps) to A or
   B (real predictions), there's no signal for the runtime to dispatch
   on. The kernel variant infrastructure that calls the selector at
   runtime would always see `predicted_thermal = 0.0` and always pick
   Hot — same as no infrastructure.
3. **Per-primitive parity testing is the load-bearing cost.** Each
   Warm/Cool variant must produce byte-identical output to Hot across
   the full input space — hundreds of property tests per primitive
   per variant. That's where the multi-week estimate comes from, not
   the kernel code itself. Doing this work before there's a consumer
   means writing tests for code paths that no production caller
   exercises.

### Concrete next steps for Option α

1. **Extend `kernel_variant.rs`** with the trait + two impls (1 day).
2. **One ADR** recording the decision: chose α because the upstream
   thermal signal isn't yet flowing.
3. **Cross-link** from `CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md` so
   future readers see both deferrals as a coherent strategy.

**Total: ~1 day for §4B.4 scaffolding under Option α.**

Re-evaluate to β/γ/δ when:
- `NssPressurePredictor` returns non-empty thermal maps (Option A or B
  for §4B.2 lands), AND
- A workload demonstrates that thermal pressure matters (e.g. chess RL
  with sustained tensor allocation triggers thermal throttling on a
  bench machine), AND
- A specific primitive shows runtime benefit from Cool variant on
  that workload.

---

## Option β — Cool variant for one primitive (proof-of-concept)

Implement the Cool (fully unfused) variant for `fused_matmul_norm`
only. The Cool variant is the easiest by construction: it's literally
the unfused chain that existed before Phase 3.5 fusion. The
runtime dispatches via the trait surface from Option α.

**What gets built:**

Everything in Option α, plus:
- `crates/cjc-runtime/src/accumulator.rs`: a `fused_matmul_norm_cool`
  function that calls `matmul(a, w)` then `norm(h)` directly.
- `crates/cjc-runtime/src/builtins.rs`: the existing `fused_matmul_norm`
  arm becomes a thin wrapper that consults a (thread-local or args-
  passed) selector and routes to either the existing Hot kernel or
  the new Cool kernel.
- Parity tests: ~100 random inputs verify Hot vs Cool produce
  byte-identical output.

**Pros:**
- One working end-to-end demonstration of the variant infrastructure.
- The Cool variant is the cheapest possible — pre-fusion code.
- Surfaces the runtime-selector-plumbing question (#2 above) on a
  real example before the architectural pattern locks in.

**Cons:**
- Doesn't ship Warm. Two-tier dispatch (Hot/Cool) is less expressive
  than three-tier.
- The selector still has no real-world consumer (empty thermal maps).
- The per-primitive parity test infrastructure is built but not
  amortized across multiple primitives.

**Effort:** 3-5 days (Cool impl + parity tests + selector routing).

**When to choose β:** if you want to validate the selector-routing
architecture before scaling to all primitives, and you have a clear
near-term plan to layer Warm on top.

---

## Option γ — All three variants for one primitive

Implement Hot (already shipping), Warm, and Cool for
`fused_matmul_norm`. Same selector routing as β.

**What's new vs β:**

- Warm variant: partially fused. The handoff sketch: "keep the matmul
  as a primitive call but skip the intermediate `Tensor` wrapper for
  the second op." So `fused_matmul_norm_warm` calls
  `matmul_raw(a_data, w_data) -> Vec<f64>` (no Tensor wrapping)
  then computes the L2 norm directly over that vector, skipping the
  intermediate `Tensor::from_vec` allocation.
- ~100 more parity tests covering Hot vs Warm and Warm vs Cool.

**Effort:** 1 week (per the handoff estimate).

**Pros:**
- One primitive fully covered. The architectural pattern is exercised
  in its final form.
- Future primitives (`fused_matmul_dot`, `fused_matmul_matmul`) follow
  the same template.
- The 3-tier dispatch is a richer surface for thermal-aware decisions.

**Cons:**
- Still no real-world consumer until the §4B.2 predictor migrates.
- 1 week of effort for code that doesn't change any user-observable
  behavior today.

**When to choose γ:** if you have a planned consumer in the next
2-4 weeks (e.g. you're committing to Option A for §4B.2 and want
the variants ready when the predictor activates).

---

## Option δ — All three variants for all three primitives

The handoff's stated end state: Hot/Warm/Cool for `fused_matmul_dot`,
`fused_matmul_norm`, and `fused_matmul_matmul`. All routed through
the selector.

**Effort:** 3 weeks (3 × 1 week per primitive).

**Pros:**
- Complete §4B.4 deliverable.
- All three Phase 3.5 fused primitives now have thermal-aware variants.
- The variant infrastructure is fully amortized across primitives.

**Cons:**
- 3 weeks of multi-session work.
- Same "no real-world consumer" problem as β/γ until §4B.2 migrates.
- Risk: if the selector routing pattern turns out to be wrong after
  the first primitive lands, the second and third primitive's
  implementations need rework.

**When to choose δ:** if §4B.2 has migrated to Option A or B, real
NSS thermal predictions are flowing, and a specific workload
demonstrates that thermal pressure varies enough to warrant runtime
variant selection.

---

## Decision matrix

| Option | LOC | Days | Real codegen? | Affects cjc-runtime? | Runtime selector needed? |
|---|---|---|---|---|---|
| α — Trait scaffolding | ~100 | 1 | No | No | No (returns Hot, dispatch unchanged) |
| β — Cool for one primitive | ~400 | 3-5 | Yes (Cool only) | Yes (one arm) | Yes |
| γ — All 3 variants, one primitive | ~700 | 7 | Yes | Yes | Yes |
| δ — Full implementation | ~2000 | 21 | Yes | Yes | Yes |

---

## My recommendation: Option α, with stated trigger for β

Same reasoning as §4B.2's Option C: build the surface, defer the
implementation, set a clear trigger for advancement.

Specifically:

- Land Option α this session (~1 day, fits cleanly into the current
  arc).
- Document the trigger for moving to β: when §4B.2 migrates to Option
  A or B AND a workload shows thermal-pressure variance, the first
  Cool variant becomes worth building.
- Until that trigger fires, the kernel-variant infrastructure stays
  as a trait surface plus the existing `select_for_budget` heuristic.

### What I'm shipping in this session

The accompanying commit lands Option α:

1. Extends `crates/cjc-cana/src/kernel_variant.rs` with the
   `KernelVariantSelector` trait, `AlwaysHotSelector`, and
   `PressureAwareSelector`.
2. Adds unit tests proving deterministic dispatch + label stability.
3. Re-exports the trait from `cjc-cana` lib root for downstream use.

Total: ~120 LOC of additions, no runtime changes, no per-primitive
work. All forward-compatible with future Warm/Cool variants.

---

## What this doc does NOT cover

- **The runtime selector plumbing question.** When Warm/Cool actually
  get implemented, the question of *how the selector reaches the
  runtime call site* matters: thread-local, MirExecutor field, args
  passed through dispatch? β/γ/δ should address this before any code
  lands.
- **Codegen of Warm and Cool.** Option α explicitly ships zero
  codegen. Future phases will need to decide:
  - Should Warm/Cool be hand-written in `cjc-runtime`?
  - Should they be MIR-emitted from `fusion_rewrite` as alternative
    candidates?
  - Should the executor select at runtime, or should the compiler
    bake-in one variant per call site?

These are real architectural questions that belong in the β-or-later
design doc, not here.

---

*Next action: pick α / β / γ / δ. α is implementable today; β-δ are
multi-session.*
