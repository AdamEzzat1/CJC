# ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML

- **Status:** Accepted (2026-06-02) — **v0.1 estimator surface complete**. Sessions 1 (`PropensityScoreMatcher`), 2 (`IVRegression` with HC1 sandwich SE + Stock-Yogo F-stat → E9100), and 3 (`DoubleMLEstimator` with K-fold cross-fitted linear nuisances) all shipped on `feat/cjc-causal-scaffolding`. v0.1 ships **linear nuisances** for DML; MLP nuisances via `cjc_ad::GradGraph` deferred to v0.2 (deliberate decision — see §What's-deferred). `publish = false` retained until v0.2.x release engineering pass
- **Crate:** `cjc-causal` (new)
- **Companion docs:** [[New Crate Stack — Cronos, Causal, Tempest]] (handoff §2), [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]], [[ADR-0042 Locke v0.8 — Str-to-Float Auto-Promotion + E9070 Wiring]]
- **Reserved error-code range:** **E9100..=E9199**

## Context

[[ADR-0028 Locke Data Skepticism Layer]] explicitly left causal inference out of scope:

> ABNG already ships `drift`, `audit`, `merkle`, `stats` modules, but ABNG is **tree-node-oriented** (belief radix graph) — not the column/row-oriented model needed for tabular skepticism. … 5 causal warning kinds + observational-only mode + user-declared claims.

Locke ships **causal warnings** (no warning kind is named "Cause" or "Effect"; the [`causal`](../../crates/cjc-locke/src/causal.rs) module emits hint-level findings about correlation-as-causation risk). Locke does *not* compute treatment effects. Per the [[Locke v0.1.10 — An Epistemic Layer for Data Validation|v0.8 retrospective]] the framing is "Locke is *where to look*, not *what to do*."

cjc-causal closes that gap as a dedicated workspace crate. It depends on cjc-locke (consumes [`LockeReport`](../../crates/cjc-locke/src/report.rs) and emits [`ValidationFinding`](../../crates/cjc-locke/src/report.rs)) but reverses the framing: cjc-causal commits to *what to do* — formal estimation of treatment effects under explicitly-declared identification assumptions, with byte-identical reproducibility.

The Python causal-inference ecosystem (DoWhy, EconML, CausalML, doubleML-Python) does not provide structural determinism. That gap is the publishable claim.

## Decisions

### 1. New workspace crate `cjc-causal`

Workspace member under `crates/cjc-causal/`. Standard `version.workspace = true` inheritance. **`publish = false`** until v0.1 implementation lands — the scaffolding compiles but is not yet shipped to crates.io.

Path-deps: `cjc-data`, `cjc-ad`, `cjc-runtime`, `cjc-repro`, `cjc-locke`. No external runtime dependencies.

### 2. Method-specific estimator structs (not a unified trait) for v0.1

Three estimators, each its own struct with its own builder-style configuration:

```rust
pub struct PropensityScoreMatcher { /* ... */ }
pub struct IVRegression { /* ... */ }
pub struct DoubleMLEstimator { /* ... */ }
```

Each exposes `.estimate(df, treatment, outcome, covariates, assumptions, locke_report) -> Result<EffectEstimate, CausalError>`.

**Rejected alternative**: unified `pub trait CausalEstimator { fn estimate(...) }` — too many method-specific knobs (caliper for matching, instruments for IV, fold count for DML) to fit cleanly behind a single trait. Revisit in v0.2 if a real cross-method abstraction emerges from usage.

### 3. `IdentificationAssumption` is a closed enum with no defaults

Every estimator's `estimate()` takes `&[IdentificationAssumption]`. There is **no default assumption set**. The caller must declare:

```rust
pub enum IdentificationAssumption {
    Unconfoundedness,
    Positivity,
    ExcludabilityOfInstrument,
    MonotonicityOfInstrument,
    ParallelTrends,
    LocalRandomization,
    NoInterference,
}
```

Rationale: the declared assumption set is part of the [`EffectEstimate.identifier`](../../crates/cjc-causal/src/estimate.rs) content hash, so two estimates from the same data with different assumption sets carry different IDs. A typo as a string would be a silently-different analysis; a closed enum makes it a compile error.

**Rejected alternative**: `&[&str]` for ergonomics. Loses the typo-catches-at-compile-time benefit. Rejected.

### 4. `EffectEstimate` is content-addressed via `cjc_locke::id::FingerprintId`

Re-exported from cjc-locke (no new ID infrastructure). The identifier is a SplitMix64-derived 64-bit fingerprint over the canonical byte representation of:

- Estimator type (`PropensityScoreMatching`, `IVRegression`, `DoubleML`)
- Treatment column name
- Outcome column name
- Covariate column names sorted ascending
- Assumptions sorted ascending
- Seed
- Point estimate bits (`f64::to_bits`)
- Std error bits

Two runs that produce the same canonical inputs produce byte-identical identifiers. This is the publishable reproducibility claim.

### 5. Caller passes the Locke report (not estimator-calls-Locke internally)

```rust
matcher.estimate(&df, "treatment", "outcome", &covariates, &assumptions, &locke_report)
```

Rationale: the caller owns Locke's configuration (custom detectors, sentinel sets, auto-promotion toggles per [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]] + [[ADR-0042 Locke v0.8 — Str-to-Float Auto-Promotion + E9070 Wiring]]). Hiding the Locke call inside the estimator would mean either (a) re-validating with the wrong config, or (b) needing to pass the config through, defeating the point of the abstraction.

Estimators inspect the report for **refusal-grade findings**:

- `E9001` missingness ≥ 0.30 on treatment or outcome column
- `E9009` continuous covariate not promoted to `Column::Float`
- `E9060` strong leakage detected on a declared covariate

On match, the estimator returns `CausalError::DataQualityRefusal { findings }`. Thresholds are revisable in the implementation session if the empirical cost-of-refusal is too aggressive.

### 6. v0.1 estimator surface

| Estimator | Algorithm | Session | Key references |
| --- | --- | --- | --- |
| `PropensityScoreMatcher` | IRLS logistic + nearest-neighbor matching with caliper | 1 | Austin (2011) caliper default 0.2σ; reuse `cjc_runtime::hypothesis::logistic_regression` |
| `IVRegression` | 2SLS with HC1 sandwich SE + Stock-Yogo F-stat | 2 | Stock-Yogo (2005) F < 10 → E9100 |
| `DoubleMLEstimator` | Orthogonal moments + K-fold cross-fitting + GradGraph MLP nuisances | 3 | Chernozhukov et al. (2018); K=5 default; nuisances via [[ADR-0016 Language-Level GradGraph Primitives]] |

### 7. Reserved error-code range E9100..=E9199 (cjc-causal)

Fits between Locke's E9001..=E9112 (validation + drift + leakage + temporal) and the custom-detector range E9500..=E9999 from [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]]. Initial assignments:

| Code  | Severity | Trigger |
| ----- | -------- | ------- |
| E9100 | Error    | Weak instrument detected (F-statistic < 10) |
| E9101 | Error    | Positivity violation (propensity near 0 or 1) |
| E9102 | Warning  | Covariate imbalance post-match (\|SMD\| > 0.10) |
| E9103 | Warning  | Overlap failure (treated unit no control within caliper) |
| E9104 | Warning  | Orthogonality violation in double ML |

E9100..=E9199 reserved for the v0.1 surface; v0.2 estimators (RD, DiD) draw codes from this range.

### 8. Library-only Rust API for v0.1 — no language-level builtins

cjc-causal is invoked through `cargo run` or via library linkage. **No `.cjcl` source-level access in v0.1.** Rationale: the [[CLAUDE|wiring pattern]] requires three-place registration (`cjc-runtime`, `cjc-eval`, `cjc-mir-exec`) per builtin and parity tests for byte-identical AST↔MIR output on every entry point. That's ~9 wiring sites and ~10+ parity tests for the three estimators — appropriate work but distinct from the math implementation. Defer to v0.2 as a focused session.

## Determinism contract

All five workspace-wide rules apply unmodified:

1. **All float reductions** through `cjc_repro::KahanAccumulatorF64`. Raw `.iter().sum()` banned.
2. **All map iteration** through `BTreeMap` / `BTreeSet`. No `HashMap`.
3. **All randomness** through `cjc_repro::Rng` (SplitMix64) with the seed threaded explicitly from the caller. Never `rand::thread_rng()`.
4. **No FMA**. `RUSTFLAGS` must not enable `target-feature=+fma`. `f64::mul_add` is banned.
5. **Cross-platform parity** on Linux + macOS + Windows for the same `EffectEstimate.identifier`.

Three cjc-causal-specific determinism contracts:

1. **Tie-breaking in nearest-neighbor matching**: when two control units have logits within `f64::EPSILON`, the lower row index wins. Never hash order, never iteration order from an unsorted collection.
2. **`logit()` / `expit()` numerical guards**: clamp inputs to `[1e-10, 1 - 1e-10]` before taking the logit (mirroring `cjc-runtime/src/hypothesis.rs::logistic_regression`). Prevents `inf` propagation.
3. **Seed threading for sample-splitting in DML**: K-fold split is RNG-driven via `cjc_runtime::ml::kfold_indices`. Seed is part of the `EffectEstimate.identifier` content hash.

## Test surface (handoff §6.1 floors)

| Bucket          | Floor | Location |
| --------------- | ----- | -------- |
| Unit            | ≥ 25  | `crates/cjc-causal/src/lib.rs` (in-module `tests`) |
| Integration     | ≥ 12  | `tests/causal/` (multiple files per estimator) |
| Proptest        | ≥ 5   | `tests/causal/causal_proptest.rs` (256 cases each) |
| Bolero fuzz     | ≥ 3   | `tests/causal/causal_fuzz.rs` |
| Determinism     | (headline) | `tests/causal/causal_determinism.rs` — required on Linux + macOS + Windows |
| Locke parity    | (1)   | `tests/causal/causal_locke_parity.rs` |

Required proptest properties (handoff §2.6):

1. Same input + same seed ⇒ byte-identical `EffectEstimate`.
2. `EffectEstimate.identifier` stable across two runs.
3. Adding a constant to the outcome shifts the estimate by exactly that constant.
4. Adding a constant to a covariate does not change the estimate.
5. Doubling the sample size never decreases the absolute number of matched pairs.

## Alternatives considered

### A — Embed cjc-causal inside cjc-locke

Pros: zero dependency boundary; analyst can call `report.estimate(...)` after `validate(...)`.
Cons: pollutes Locke's "where to look" identity. Locke's public surface grows to include estimators, which forces every downstream of Locke to ship with causal-inference types in scope. Rejected.

### B — A pure Rust library outside the workspace

Pros: clean isolation; published independently.
Cons: loses workspace `Cargo.lock` reproducibility; can't share `cjc-repro` / `cjc-runtime` helpers without external versioning; CI gating becomes awkward. Same reasoning as Locke's ADR-0028 §C. Rejected.

### C — Build on top of an existing causal-inference Python crate via PyO3

Pros: leverages existing implementations (econml, doubleML).
Cons: defeats the publishable determinism claim (downstream code is non-deterministic); loses cross-platform parity; introduces a Python runtime dependency. Rejected.

### D (chosen) — New workspace crate `cjc-causal`, mirroring cjc-locke

Pros: matches the [[ADR-0028 Locke Data Skepticism Layer|cjc-locke precedent]]; clear ownership boundary; shared workspace lockfile; Locke composes via report-passing without a circular dep. Cons: one more crate to publish.

## What's deferred to v0.2+

| Capability | Defer to | Rationale |
| --- | --- | --- |
| Regression discontinuity | v0.2 | Bandwidth-selection literature (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik) is its own rabbit hole; v0.1 ships without it cleanly. |
| Difference-in-differences | v0.2 | Modern DiD needs the staggered-adoption fixes from Callaway-Sant'Anna 2021; significant scope; doing it half-right is worse than waiting. |
| Mediation analysis | v0.2 | Sits between matching and SEM; pick a side after RD + DiD land. |
| `.cjcl` language-level builtins | v0.2 | 9 wiring sites + parity tests; focused session. |
| Python bindings via PyO3 | v0.1.1 / v0.2 | Mirrors the cjc-locke-py pattern from the [[Locke v0.1.10 — An Epistemic Layer for Data Validation|v0.8 release]]. Does not block the Rust v0.1 ship. |
| Do-calculus / Pearl identification | v0.3 | Full graph-surgery semantics is a workshop's worth of work. |
| Structural equation modelling | v0.3+ | Orthogonal scope; no SEM story in CJC-Lang yet. |
| Synthetic control method | indefinitely | Specialised use case; defer until a real demand surfaces. |
| Causal forests / generalised random forests | indefinitely | Same. |

## Scope-discipline summary

v0.1 ships *the three estimators econometrics teaching tracks most often cover first* — propensity score matching (treatment-effect-as-comparison), IV (treatment-effect-as-natural-experiment), and double ML (treatment-effect-with-flexible-nuisances) — plus the data-quality-refusal hook into cjc-locke. v0.2 expands to RD + DiD + cjcl builtins + Python bindings. v0.3 introduces do-calculus.

Per the [[CLAUDE|Prime Directives]]: directives 3 (determinism) and 5 (no silent refactors of unrelated systems) are the most load-bearing. The byte-identical-reproducibility story is what makes cjc-causal uniquely CJC-shaped; if you sacrifice it for a perceived performance win, you have given up the publishable claim.
