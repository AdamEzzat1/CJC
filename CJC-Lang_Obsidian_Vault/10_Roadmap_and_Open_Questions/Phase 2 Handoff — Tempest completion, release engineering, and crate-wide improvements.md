---
title: "Phase 2 Handoff — Tempest completion, release engineering, and crate-wide improvements"
tags: [handoff, planning, roadmap, tempest, release, locke, tidyview, autodiff, cjc-ad]
status: 📋 Planned — five workstreams queued
date: 2026-06-02
date-modified: 2026-06-02
---

# Phase 2 Handoff — Tempest completion, release engineering, and crate-wide improvements

Follows the [[New Crate Stack — Cronos, Causal, Tempest|Phase 1 handoff]] now that cjc-causal v0.1, cjc-cronos v0.1, and cjc-tempest Session 1 have all shipped. This document covers the next five workstreams:

1. **cjc-tempest Sessions 2-4** — HMC, NUTS, Vehtari R-hat + ESS
2. **cjc-causal + cjc-cronos v0.1.x release engineering** — lift `publish = false`, ship to crates.io
3. **cjc-locke v0.9 roadmap** — improvements identified by audit
4. **TidyView roadmap** — window functions, string ops, ergonomics
5. **cjc-ad autodiff roadmap** — higher-order AD, JVP, determinism audit, composition with HMC

Each workstream is independently shippable. Read §0 for the TL;DR, §8 for the recommended execution order.

## 0. TL;DR

| Workstream | Headline | Scope | Suggested branch |
|---|---|---|---|
| **cjc-tempest S2: HMC** | Leapfrog + reverse-mode AD via cjc-ad GradGraph for log-posterior gradient | ~2 sessions | `feat/cjc-tempest-scaffolding` |
| **cjc-tempest S3: NUTS** | Hoffman & Gelman 2014 Algorithm 6 with dual averaging | ~2 sessions | `feat/cjc-tempest-scaffolding` |
| **cjc-tempest S4: Diagnostics** | Vehtari et al. 2021 split-rank-normalised R-hat + bulk/tail ESS | ~1 session | `feat/cjc-tempest-scaffolding` |
| **causal + cronos release** | crates.io publication, README write-ups, version bumps, CI matrix | ~1-2 sessions | `release/0.1.12` (new) |
| **Locke v0.9** | Inter-crate finding routing, evidence-weighted penalties, count-min streaming, per-column belief, confounder hints | ~3-4 sessions | `feat/locke-v0.9` (new) |
| **TidyView roadmap** | Window functions, string mutate ops, column-selector helpers, nested groupby | ~3 sessions | `feat/tidyview-windows` (new) |
| **cjc-ad roadmap** | Native double-backward, JVP/batched jacobian, determinism audit, language-level Hessian | ~4-5 sessions | `feat/cjc-ad-phase-3d` (new) |

**Critical interdependency**: cjc-tempest's HMC determinism contract requires **cjc-ad's determinism audit (§5.7) to land first**, or HMC's byte-identical-posteriors claim is at risk. The audit is small (~1 session) and should ship before HMC code starts.

## 1. cjc-tempest v0.1 completion (Sessions 2-4)

### 1.1 Pre-flight: cjc-ad determinism audit (BLOCKING)

Before HMC code starts, audit cjc-ad's backward pass for non-Kahan reductions. The audit findings from this handoff §5.7 list specific sites:

- `accumulate_grad()` in backward uses `Tensor::add_unchecked()` — does NOT mandate reduction order. On platforms with different SIMD widths (Linux x86_64 AVX2 vs macOS arm64), accumulation order may vary, producing ULP-level differences. **This breaks HMC's byte-identical-posteriors claim on cross-platform reproducibility.**
- `Sum` backward, `Mean` backward expand and add grads in graph order; no Kahan compensation.
- `softmax`, `cross_entropy`, `layer_norm`, `batch_norm`, `clip_grad_norm` already use Kahan ✓
- `pinn_mlp_eval_grid`, `pinn_l2_max_errors` already use Kahan ✓

**Action**: Wrap all gradient accumulation in `KahanAccumulatorF64`, OR mandate that `Tensor::add_assign_unchecked()` uses Kahan internally (move responsibility into cjc-runtime). Either path is ~1 session. Add `GradGraph::verify_determinism(other_graph)` method that checks bit-identical forward+backward results across two runs.

### 1.2 Session 2: Hamiltonian Monte Carlo

**Scope**: `crates/cjc-tempest/src/hmc.rs` (~450 LOC) + ~25 unit + ~12 integration tests.

**Architecture** (per [[ADR-0045 cjc-tempest v0.1 — Metropolis, HMC, NUTS with byte-identical posteriors|ADR-0045]] §4.4):
- Leapfrog integrator with caller-specified step size `epsilon` and trajectory length `L`
- Identity mass matrix in v0.1 (mass adaptation deferred to v0.2)
- Reverse-mode AD via `cjc_ad::GradGraph`: log-posterior closure is called once per leapfrog step; gradient via `GradGraph::backward()`
- Energy error tracking: divergence declared when `|H_new - H_initial| > 1000`
- All accumulations Kahan-summed (Kahan in leapfrog is non-negotiable — even small per-step rounding compounds over a trajectory)

**Public API**:
```rust
pub struct HamiltonianMonteCarlo {
    epsilon: f64,            // leapfrog step size
    trajectory_length: usize, // number of leapfrog steps L
    divergence_threshold: f64, // default 1000.0
}

impl HamiltonianMonteCarlo {
    pub fn new(epsilon: f64, trajectory_length: usize) -> Self;
    pub fn with_divergence_threshold(mut self, t: f64) -> Self;
    pub fn run<F, G>(
        &self,
        log_posterior: F,
        log_posterior_grad: G,  // ∇ log π(θ|y)
        initial_state: &[f64],
        n_chains: usize,
        n_warmup: usize,
        n_iter: usize,
        base_seed: u64,
    ) -> Result<PosteriorSamples, TempestError>
    where F: Fn(&[f64]) -> f64, G: Fn(&[f64]) -> Vec<f64>;
}
```

**Why caller supplies the gradient** rather than auto-deriving from `log_posterior`: HMC closure invocations are the hot loop (`L` per HMC step × `n_warmup + n_iter` steps × `n_chains`). Forcing the gradient through `GradGraph::backward()` per call is correct but slow for v0.1. Allowing the caller to pass an analytic gradient OR an auto-built `GradGraph` closure is both faster and more honest about the determinism contract.

**Convenience constructor** that builds the closure via cjc-ad:
```rust
impl HamiltonianMonteCarlo {
    pub fn with_grad_graph_log_posterior<F>(
        &self,
        log_posterior_builder: F,  // builds the GradGraph once
        initial_state: &[f64],
        ...
    ) -> Result<PosteriorSamples, TempestError>
    where F: Fn(&mut cjc_ad::GradGraph, &[usize]) -> usize  // returns loss node idx
```

**Required tests**:
- HMC on a 1D Gaussian recovers mean = 0, variance = 1 (within 5%)
- HMC on a 2D banana density samples cover the known density
- Divergence detection: degenerate `epsilon = 0.001` on a steep posterior triggers `AcceptResult::Divergent`
- Same seed + same model + same gradient closure ⇒ byte-identical posterior `content_hash`
- Cross-platform parity (run on Linux + macOS + Windows in CI; assert content_hash matches)
- Gradient closure called exactly `L × (n_warmup + n_iter)` times (count assertion)
- HMC vs Metropolis on the same posterior: HMC has higher acceptance rate at large step counts

**Determinism contract** (specific to HMC):
1. Momentum draw uses `cjc_repro::Rng::next_normal()` (Box-Muller from two uniforms)
2. Leapfrog integrator accumulates with Kahan
3. Energy `H = potential + kinetic` accumulator is Kahan
4. Acceptance coin flip ALWAYS consumes one uniform regardless of decision (same invariant as Metropolis Session 1)
5. Per-chain seed stretch identical to Metropolis: `base_seed + chain_idx * SPLITMIX_STEP`

### 1.3 Session 3: No-U-Turn Sampler (NUTS)

**Scope**: `crates/cjc-tempest/src/nuts.rs` (~600 LOC) + ~25 unit + ~12 integration tests + the headline determinism extension.

**Architecture** (per ADR-0045 §4.4):
- Hoffman & Gelman 2014 Algorithm 6 (efficient NUTS with dual averaging for step-size adaptation)
- Recursive tree building with no-U-turn termination criterion
- Dual averaging during warm-up adapts `epsilon` to target acceptance rate (default 0.8)
- Reuses HMC's leapfrog integrator and gradient interface

**Public API**:
```rust
pub struct NoUTurnSampler {
    max_tree_depth: usize,       // default 10 (2^10 = 1024 leapfrog steps max)
    target_accept_prob: f64,     // default 0.8
    divergence_threshold: f64,
}

impl NoUTurnSampler {
    pub fn new() -> Self;
    pub fn with_max_tree_depth(mut self, d: usize) -> Self;
    pub fn with_target_accept_prob(mut self, p: f64) -> Self;
    pub fn run<F, G>(&self, log_posterior: F, log_posterior_grad: G, ...) -> Result<PosteriorSamples, TempestError>;
}
```

**Algorithm 6 in pseudocode**:
```
for each iteration:
    sample slice variable u ~ Uniform(0, exp(H_initial))
    build a NUTS tree by doubling in a random direction (left/right)
        - each leaf is a leapfrog step
        - terminate when U-turn detected: dot(p_left, q_right - q_left) < 0
          OR dot(p_right, q_right - q_left) < 0
        - or when energy error exceeds divergence_threshold
        - or when tree depth reaches max_tree_depth
    select a sample uniformly from acceptable leaf states (slice u < exp(-H))
```

**Determinism contract** (NUTS-specific additions):
1. **Direction RNG**: each tree expansion consumes one uniform to pick left vs right. Must consume regardless of which branch is taken.
2. **Slice sample**: `u ~ Uniform(0, exp(H_initial))` — one uniform per iteration.
3. **Sample selection**: choosing the new state from acceptable leaves consumes one uniform; the multinomial weights are deterministic functions of the tree.
4. **Dual averaging RNG**: NONE — dual averaging is deterministic given the trajectory.
5. **Adaptation-phase RNG SEPARATE from sampling-phase RNG** (ADR-0045 §determinism rule 1.6). Implement via SplitMix64 sub-stream: `adaptation_seed = base_seed ^ 0x_ADAPT_SALT_`, `sampling_seed = base_seed ^ 0x_SAMPLE_SALT_`.

**Required tests**:
- NUTS on 8-Schools hierarchical model (the canonical PPL test) — posterior of `tau` matches Stan's output direction (not byte identity; Stan and NUTS use different RNGs)
- Same seed + same model ⇒ byte-identical posterior `content_hash`
- Max tree-depth flagged via `ConvergenceDiagnostics::n_max_treedepth`
- Divergent transitions counted in `ConvergenceDiagnostics::divergences`
- Step-size adaptation converges to target acceptance rate within warmup
- Cross-platform parity on Linux + macOS + Windows

### 1.4 Session 4: Vehtari et al. 2021 R-hat + ESS

**Scope**: `crates/cjc-tempest/src/diagnostics.rs` (~250 LOC) + ~20 unit + ~10 integration tests.

**Algorithm**:
- **Split-rank-normalised R-hat** (Vehtari 2021): for each parameter, split each chain into halves (so 4 chains → 8 split-chains), rank-normalise across all split-chains, compute classical R-hat on the ranked values
- **Bulk ESS**: 4 × (number of chains × samples per chain) / (1 + 2·Σρ) where ρ are autocorrelations summed via Geyer's initial monotone sequence estimator
- **Tail ESS**: same formula but on a quantile transformation of the chains (5%/95% quantiles)
- All reductions Kahan-summed
- No HashMap; all per-parameter aggregates in `Vec<f64>` or `BTreeMap` (R-hat per parameter index)

**Public API**:
```rust
impl PosteriorSamples {
    /// Compute split-rank-normalised R-hat per parameter.
    pub fn compute_r_hat(&self) -> Vec<f64>;
    /// Bulk ESS per parameter.
    pub fn compute_ess_bulk(&self) -> Vec<f64>;
    /// Tail ESS per parameter.
    pub fn compute_ess_tail(&self) -> Vec<f64>;
    /// Compute all diagnostics in one pass.
    pub fn compute_diagnostics(&self) -> ConvergenceDiagnostics;
}
```

**E-code emission**:
- `E9301` Error: R-hat > 1.01 on any parameter (convergence failure)
- `E9302` Warning: bulk-ESS < 400 (insufficient effective sample size)
- `E9303` Warning: NUTS hit max tree depth (already populated in `ConvergenceDiagnostics::n_max_treedepth`)
- New helper: `pub fn convergence_findings(samples: &PosteriorSamples) -> Vec<ValidationFinding>`

**Required tests**:
- R-hat for a single chain equals 1.0 by definition (definitional property)
- R-hat for two non-mixing chains far apart > 2.0 (sanity)
- ESS ≤ n_samples_per_chain × n_chains (definitional)
- For an AR(1) process with known autocorrelation ρ, ESS matches closed-form `(1-ρ)/(1+ρ) · n_samples`
- Vehtari's split-rank trick is sensitive to the "stuck chain" failure mode; specific synthetic test where one chain is stuck and R-hat correctly flags it
- Multi-chain doubling: doubling n_chains halves the R-hat threshold violation distance (sensitivity check)
- Cross-platform parity on diagnostic values

### 1.5 Headline cross-platform CI

After Sessions 2-4 ship, the **headline cross-platform determinism test cannot be deferred**. Add to `.github/workflows/`:

```yaml
- name: cjc-tempest determinism cross-platform
  strategy:
    matrix:
      os: [ubuntu-latest, macos-latest, windows-latest]
  runs-on: ${{ matrix.os }}
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - name: Run determinism headline tests
      run: cargo test --test tempest tempest_determinism --release
    - name: Hash posteriors and upload
      run: |
        cargo run --release --example dump_posteriors -- --seed 42 > posteriors-${{ matrix.os }}.txt
    - uses: actions/upload-artifact@v4
      with:
        name: posteriors-${{ matrix.os }}
        path: posteriors-${{ matrix.os }}.txt
- name: Compare across platforms
  needs: cjc-tempest determinism cross-platform
  runs-on: ubuntu-latest
  steps:
    - uses: actions/download-artifact@v4
    - name: Assert byte-identity across OS
      run: diff posteriors-ubuntu-latest/posteriors-ubuntu-latest.txt posteriors-macos-latest/posteriors-macos-latest.txt
      # Repeat for windows-latest pair
```

**If this matrix fails on any platform**, the offending PR does NOT merge. Document this explicitly in `.github/PULL_REQUEST_TEMPLATE.md`.

### 1.6 What v0.1 explicitly does NOT do (locked in ADR-0045 §What's-deferred)

- Variational inference (ADVI / SVI) — v0.2
- Mass-matrix adaptation (diagonal or dense) — v0.2; identity mass for v0.1
- Posterior predictive sampling — v0.1.1 small extension
- Reversible-jump MCMC, sequential Monte Carlo — indefinitely
- `model { ... }` DSL block in `.cjcl` — v0.3+ (separate ADR)
- Bayes factor / marginal likelihood — v0.2 or later
- Python bridge via PyO3 — v0.1.1 / v0.2

## 2. cjc-causal + cjc-cronos v0.1.x release engineering

Both crates have feature-complete v0.1 surfaces. This workstream lifts `publish = false`, bumps versions, ships READMEs, and publishes to crates.io.

### 2.1 Pre-flight checklist

Per `[[CLAUDE]]` Prime Directives and the existing publish-process precedent (`v0.1.9` release published 2026-05-20):

- [ ] All tests pass on Linux + macOS + Windows in CI
- [ ] `cargo doc --workspace --no-deps` builds without warnings on the new crates
- [ ] Each crate has a `README.md` with: tagline, quick-start, link to ADR, link to docs.rs after publish
- [ ] Each crate's `Cargo.toml` has `description`, `repository`, `homepage`, `keywords`, `categories`, `license` fields complete
- [ ] No `unwrap()` / `expect()` in non-test code that could trigger on user input (cjc-causal audit done; cjc-cronos needs same audit)
- [ ] `unsafe` blocks: none introduced in cjc-causal or cjc-cronos (verify)
- [ ] Determinism audit per ADR-0043 §determinism and ADR-0044 §determinism: BTreeMap-only, Kahan-only, SplitMix64-only, no FMA

### 2.2 Workspace version bump strategy

Current workspace version: `0.1.11`. Two options:

**Option A**: Bump workspace to `0.1.12`. All publishable crates ship at `0.1.12`. cjc-causal and cjc-cronos become `0.1.12` on crates.io.

**Option B**: cjc-causal and cjc-cronos publish at their own version (`0.1.0`) while the rest of the workspace stays at `0.1.11`. Requires hard-coding `version = "0.1.0"` in the crate's `Cargo.toml` instead of `version.workspace = true`.

**Recommendation**: **Option A**. Aligns the release log with the actual workspace version. Matches the [[CLAUDE]] memory note: "Internal crate names remain `cjc-*` ... workspace version bumps in lockstep." Adds the cost of touching every workspace crate's CHANGELOG, but that's a one-time mechanical edit.

### 2.3 README write-ups

Each crate needs a top-level `README.md` (~200-300 lines):

**Template structure** (cjc-causal example):
```markdown
# cjc-causal — formal causal inference for CJC-Lang

Deterministic, content-addressed treatment-effect estimates with three
v0.1 estimators: PropensityScoreMatcher, IVRegression, DoubleMLEstimator.

## Why cjc-causal

Two runs of `PropensityScoreMatcher::estimate(df, ..., seed = 42)` on the
same data produce the same `EffectEstimate.identifier`. Across runs, across
platforms, across library upgrades. Published causal analyses can cite the
identifier; anyone re-running gets the same identifier.

## Quick start

```rust
use cjc_causal::{IdentificationAssumption, PropensityScoreMatcher};

let estimate = PropensityScoreMatcher::new()
    .with_seed(42)
    .estimate(
        &df,
        "treatment",
        "outcome",
        &["age", "income"],
        &[IdentificationAssumption::Unconfoundedness,
          IdentificationAssumption::Positivity,
          IdentificationAssumption::NoInterference],
        &locke_report,
    )?;

println!("ATT: {} ± {} (id: {})", estimate.point, estimate.std_error, estimate.identifier);
```

## v0.1 estimator surface
- PropensityScoreMatcher (greedy NN matching + bootstrap CI)
- IVRegression (2SLS + HC1 sandwich SE + Stock-Yogo F-stat → E9100)
- DoubleMLEstimator (Chernozhukov 2018 orthogonal moments + K-fold)

## See also
- [ADR-0043](https://github.com/AdamEzzat1/CJC/blob/master/CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0043%20cjc-causal%20v0.1%20%E2%80%94%20Propensity%20Score%2C%20IV%2C%20Double%20ML.md)
- [Determinism contract](https://github.com/AdamEzzat1/CJC/blob/master/CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0043%20cjc-causal%20v0.1%20%E2%80%94%20Propensity%20Score%2C%20IV%2C%20Double%20ML.md#determinism-contract)
- [docs.rs](https://docs.rs/cjc-causal)

## License
MIT
```

cjc-cronos README follows the same pattern: tagline, why-deterministic, quick-start (ETS + ARIMA + Kalman + STL each one-liner), ADR-0044 link.

### 2.4 CI matrix expansion

Current CI runs on `ubuntu-latest` only (per the workspace `.github/workflows/`). Before publishing v0.1.x:

- [ ] Add `macos-latest` to the CI matrix for both crates' test suites
- [ ] Add `windows-latest` similarly
- [ ] Ensure release-profile build also runs on the matrix (LTO + codegen-units=1 has historically uncovered cross-platform issues)
- [ ] Add a `cargo doc` job that fails on doc warnings

### 2.5 Crates.io metadata

Each crate's `Cargo.toml` needs:

```toml
[package]
keywords = ["causal-inference", "statistics", "deterministic", "reproducible"]
categories = ["science", "mathematics"]
readme = "README.md"
```

(cjc-cronos uses `["time-series", "forecasting", "statistics", "deterministic", "reproducible"]`.)

### 2.6 Publication order

1. cjc-causal v0.1.12 (smaller, simpler dep tree, already feature-complete)
2. cjc-cronos v0.1.12 (no new deps; same Locke composition pattern)
3. Wait for crates.io indexing + docs.rs build
4. Update root `cjc-lang` `[package].keywords` to mention the new crates (so cargo search surfaces them)
5. Blog post for `adamezzat1.github.io/blog/posts/` announcing the two crates with the LendingClub + ETS-on-synthetic demos

### 2.7 Python bridges (deferred but planned)

Both crates need `python/cjc_causal/` and `python/cjc_cronos/` mirrors via maturin, matching the cjc-locke-py pattern. Deferred to v0.1.1+ each (handoff §5.4 of the original).

## 3. cjc-locke v0.9 roadmap

Per the audit findings, Locke v0.8 is feature-rich and well-tested (740 tests across 30 modules) but has five concrete improvement priorities ready for v0.9.

### 3.1 Current capability summary

Locke v0.8 is a deterministic data-skepticism engine: schema validation, missingness/drift/leakage/duplicate detection, per-axis belief score composition algebra, findings (E9001..E9112 core + custom E9500..E9999), lineage DAGs, governance policies, text drift via BPE tokenizer, per-value lineage. Routes to sibling decision-layer crates (cjc-causal E9100..E9199, cjc-cronos E9200..E9299, cjc-tempest E9300..E9399) via the open extension layer.

### 3.2 Top 5 v0.9 priorities

#### P1: Seamless inter-crate finding routing & composition

**Why**: cjc-causal (live), cjc-cronos (in progress), cjc-tempest (Session 1 shipped) each extend Locke's finding namespace. Today, composition is manual — callers hand-stitch belief scores together.

**Sketch**:
- Introduce `FindingCodeRange` enum + trait to declare namespace ownership (Locke owns E9001..E9112, custom E9500..E9999, causal E9100..E9199, etc.)
- Add `BeliefAxisAssignments::from_finding_source(crate_name)` to auto-populate axis routes
- Export `compose_belief_reports(reports: &[LockeReport], rules: &BeliefAxisRules) -> BeliefReport` from algebra — currently each crate re-invents this
- Wire sibling crates to emit `LockeReport` subsets (findings + assumptions only, no `column_reports`) and document the composition contract

**Complexity**: Medium (~30-40 LOC per crate, one new algebra helper, minor report schema tweak).

#### P2: Automated threshold tuning & evidence-weighted penalties

**Why**: All 60+ detection thresholds are hard-coded. The LendingClub demo suggests E9060/E9061 are conservative. No mechanism to tune without forking `ValidationConfig`. Evidence-weighted penalties (scale penalty by violation magnitude, not just binary presence) would let high-confidence findings drive belief scores harder than marginal ones.

**Sketch**:
- Add `ThresholdProfile` enum (Conservative / Balanced / Aggressive) + per-finding optional override in `ValidationConfig`
- Implement `FindingEvidence::weight() -> f64` (ratio fields map directly, count fields normalize to 0..1 by cardinality); use in `penalty_from_findings_with_model` to scale penalty contribution instead of fixed per-code increments
- Ship 2-3 pre-tuned profiles based on real datasets (diabetes-130, LC, financial time-series)

**Complexity**: Medium (~50-80 LOC + 3 new profiles + proptest law verification that scaling preserves monotonicity).

#### P3: Streaming finding construction & per-column belief

**Why**: `StreamingValidator` today is O(RAM) memory on duplicates and does sample truncation past `sample_cap`, making distribution-shape findings lossy for large datasets. v0.4 planned a count-min sketch fallback but it never shipped. Per-column belief scores (e.g., "this column is 95% reliable, that one 60%") are valuable for downstream ML pipelines but not exposed.

**Sketch**:
- Implement deterministic count-min sketch (3-5 hash functions, seeded from crate version) for exact-duplicate tracking above `duplicate_hash_cap`. Document byte-identity caveat: exact up to cap, probabilistic above
- Export `per_column_belief_scores(report: &LockeReport) -> BTreeMap<String, BeliefScore>` by reducing per-column findings to 8-tuple (schema, missingness, etc.) independently. Reuse algebra composition for consistency
- Extend `LockeReport` JSON schema with optional `per_column_beliefs` field (v0.9 schema_version bump) default off for backward compat

**Complexity**: Medium (count-min sketch ~100 LOC, per-column belief ~60 LOC, 15+ new tests).

#### P4: Causal edge detection & confounding hints

**Why**: Locke flags correlation-as-causation risk but does not detect confounder patterns. `CausalConfig::assumed_dag` works but places the burden on domain experts. cjc-causal consumes `LockeReport` and would benefit from confounder flags inside Locke.

**Sketch**:
- Add `detect_confounder_hints(df, causal_cfg)` in `causal.rs` that:
  - Computes partial correlations `corr(X, Y | Z)` for triples of numeric columns
  - Flags when `corr(X, Y) >> corr(X, Y | Z)` (Z confounds X-Y)
  - Routes to E9114..E9119 (claim new codes in causal block)
- Integrate into `causal_guardrail` as opt-in `detect_confounders: bool` (default false to preserve v0.8 byte-identity)
- Test against simulated DAGs

**Complexity**: Large (partial correlation math ~150 LOC, integration ~50 LOC, 20+ tests including proptest on random DAGs).

#### P5: PII re-identification via quasi-identifier combinations

**Why**: E9090..E9093 detect exact PII (email, SSN, phone, API key). They don't flag quasi-identifiers (age + postal + gender re-identifies 87% of US population per Sweeney 2000).

**Sketch**:
- Add `detect_quasi_identifier_cardinality(df)` that identifies low-cardinality column combinations whose product cardinality is high (two 100-value columns → 10K combos on 5K rows = suspicious)
- Emit E9094 findings (extend PII range) with severity tied to `(combination_cardinality / n_rows)`
- Surface combinations in finding evidence as sorted column-names list + joint cardinality
- Integrate with `policy.rs` so teams can declare quasi-ID sets as intentional

**Complexity**: Medium (cardinality estimation ~80 LOC via count-distinct; careful combinations to avoid O(2^n)).

### 3.3 Pre-existing TODOs (top items not in P1-P5)

| TODO | Module | Status | Recommended action |
|---|---|---|---|
| Deterministic count-min sketch for streaming dup tracking | `streaming.rs:21` | v0.4 planned, never shipped | Part of P3 above |
| Thrift Parquet metadata decoder | `parquet_reader.rs:127` | Deferred indefinitely | Mark non-goal; document alternative path |
| Concurrent custom-detector execution | ADR-0041 open Q | Deferred to v0.9 | Belongs in P1's expansion |
| YAML/TOML rule DSL | ADR-0041 open Q | Deferred to v0.9 | Standalone v0.9 item, ~50 LOC |
| JSON emit of `custom_axis_assignments` | ADR-0041 open Q | Deferred to v0.9 | Small; pair with v0.9 schema bump |
| Per-column belief scores | Locke Roadmap | Deferred under "Belief upgrades" | Part of P3 |
| Ontology / taxonomy consistency | Locke Roadmap | Deferred indefinitely | Document non-goal |

### 3.4 Polish items (non-urgent UX)

1. **No `crates/cjc-locke/README.md`** — generate one from `lib.rs` + Roadmap on `cargo doc`, or commit a separate `.md` to the crate root
2. **Centralized E9xxx code registry** — `const` table in `lib.rs` mapping code → module + severity + one-line description
3. **`BeliefAxisSet::any(&[axis1, axis2, ...])` constructor** — match the Rust bool-flag pattern instead of bitwise OR
4. **`LockeReport::findings_in_range(E9001..=E9099)`** — filter helper for tooling that treats custom E9500+ separately
5. **Canonical `FindingEvidence::Display` impl** or `human_readable()` to standardize "85.3% match", "1.2e-4 p-value" across tools

## 4. TidyView roadmap

TidyView lives in `cjc-data`. The Adaptive TidyView Engine v2 (ADR-0017) shipped a sophisticated `AdaptiveSelection` mask representation (Empty/All/SelectionVector/VerbatimMask/Hybrid). The surface verbs are solid (filter, select, mutate, arrange, group_by, summarise, distinct, slice, joins, basic pivots) but several critical dplyr/pandas operations are missing.

### 4.1 Current capability summary

dplyr-inspired lazy-evaluation data DSL with fluent method chaining, columnar predicate evaluation, deterministic output, and adaptive mask representation that picks among Empty/All/SelectionVector/VerbatimMask/Hybrid by density threshold. Core verbs: filter, select, mutate (scalar numeric), arrange, group_by, summarise, distinct, slicing (head/tail/sample), and joins (left/inner; full/right partial).

### 4.2 Top 5 v0.x priorities

#### T1: Window functions (lag, lead, cumulative, rolling)

**Why**: Critical missing dplyr/pandas verbs. lag/lead enable time-series and sequential analysis; cumsum/cummax enable running aggregates; rolling windows enable smoothing. **All are required for cjc-cronos integration** (TimeSeries construction often follows from lag/lead computations).

**Sketch**:
- Extend `DExpr` enum with `Lag(String, i64)`, `Lead(String, i64)`, `CumSum(String)`, `RollingSum/Mean/Var(String, usize)`
- Materialize the selection to a row index list; evaluate window functions in single pass
- For rolling windows, use deque-based accumulator (Welford for variance, Kahan for sum)
- Emit results as new columns via `mutate`

**Complexity**: Medium. Execution is straightforward but needs careful edge-row handling and NaN propagation.

#### T2: String operations & regex in mutate

**Why**: Current mutate only supports scalar numeric expressions. String transformations (trim, substring, case, regex match/replace) are present in the language layer but not wired to TidyView; blocks real ETL workflows.

**Sketch**:
- Add `DExpr::FnCall("str_trim" | "str_upper" | "str_lower" | "str_sub" | "str_replace" | "str_detect", ...)` in the evaluator
- Reuse cjc-regex for complex patterns
- Implement column-wise evaluation (avoid row-wise looping)

**Complexity**: Small to Medium. Most functions are simple; regex linking is the only external dep.

#### T3: Column selector helpers (starts_with, ends_with, matches, all_numeric)

**Why**: Users must explicitly list columns by name. dplyr's `select(starts_with("x_"))` is far more ergonomic for large schemas or dynamic workflows.

**Sketch**:
- Introduce `ColumnSelector` enum: `AllNamed`, `Regex(String)`, `Predicate(fn(&str, &Column) -> bool)`, `Type(String)`
- Wire into `select()`, `drop_cols()`, `mutate_across()`, `summarise_across()`
- Provide sugar: `df.select(starts_with("id_"), ends_with("_total"))`

**Complexity**: Small. No algorithmic challenge; mostly API design and plumbing.

#### T4: Nested groupby & multi-key joins

**Why**: Current `group_by(["a", "b"])` is supported, but no support for within-group aggregates referencing other groups, or cross-group joins. Needed for cohort analysis and causal workflows.

**Sketch**:
- Extend `summarise` to accept computed aggregates that reference group-specific subframes
- Add `cross_join`, `right_join`, `full_join` (full_join exists but needs testing)
- Implement multi-key join indices (currently single-key, vectorised)

**Complexity**: Medium. Careful index construction; correctness proofs.

#### T5: API ergonomics (overloading, defaults, builders)

**Why**: Current API requires explicit column-name arrays even for single columns. Some verbs could accept `impl Into<Vec<String>>` or builders.

**Sketch**:
- Introduce builder patterns: `join_builder.on("id").left_col("left_id").right_col("right_id").run()`
- Varargs-like sugar: `select("a", "b", "c")` → `select(&["a", "b", "c"])`
- Expose fluent error context: `view.filter(...).context("customer filter")`

**Complexity**: Small.

### 4.3 Specific verbs missing from v0.1 (priority order)

| Verb | Frequency in dplyr | Priority | Notes |
|---|---|---|---|
| `lag(col, n)`, `lead(col, n)` | Very high | **URGENT** | Blockers for time-series; partially sketched in `DExpr` |
| `rolling_sum`, `rolling_mean`, `rolling_var`, `rolling_sd`, `rolling_min`, `rolling_max` | High | **URGENT** | Present in `DExpr`, absent from API |
| `str_trim`, `str_upper`, `str_lower`, `str_sub`, `str_replace`, `str_detect` | High | **HIGH** | Wired in language, TidyView mutate can't invoke |
| `cumsum`, `cumprod`, `cummax`, `cummin` | High | **HIGH** | In `DExpr`; need column-at-a-time evaluation |
| `rank`, `dense_rank`, `row_number` | Medium | HIGH | Partly sketched |
| `across(selector, transform)` | Medium | MEDIUM | Half-implemented (`mutate_across`, `summarise_across`) |
| `relocate(cols, position)` | Medium | MEDIUM | Already implemented, not in dispatch |
| `case_when(condition1 → value1, ...)` | Medium | MEDIUM | Needs extended DExpr |
| `first(col)`, `last(col)` | Medium | MEDIUM | Already in `TidyAgg` enum |
| `right_join`, `full_join` | Medium | MEDIUM | Exist as methods; need dispatch wiring |
| `unnest(col)` | Low | LOW | No nested-col type yet |
| `nest(cols)` | Low | LOW | Same |

### 4.4 AdaptiveSelection design — pros and cons

**Pros**:
- **Space efficiency**: Empty/All/SelectionVector avoids 8-byte-per-bit densities for sparse results. Sparse threshold of nrows/1024 is empirically sound
- **Cache locality**: Hybrid chunking (4096-row blocks) keeps per-chunk dense bitmasks in L3
- **No re-classification overhead in chains**: intersect/union recompute density and pick the optimal arm
- **Determinism**: Pure integer arithmetic; bit-stable across platforms

**Cons**:
- **Hybrid complexity**: 16-way per-chunk shape table in intersect/union adds cognitive load (~400 lines)
- **Materialization cost**: Hybrid → mask for fallback paths forces a full pass, defeating lazy evaluation
- **Sparse iteration over large nrows**: SelectionVector holds u32 per selected row; on 1B-row frames with 0.01% sparsity, 1M allocations
- **No predicate pushdown**: AdaptiveSelection doesn't track which predicates produced it

### 4.5 Polish items

1. **Explicit error messages with context**: `.filter(...).context("removing nulls")`
2. **Column selector sugar**: `select(cols!["a", "b", "c"])` macro or tuple syntax
3. **Fluent aggregation builder**: `group_by(...).summarise().sum("revenue", "total_revenue").mean("qty", "avg_qty").collect()`
4. **TidyView → decision-layer adapters**: `to_time_series(time_col, value_col)` for cronos; `to_design_matrix(target, features)` for causal/tempest. Lightweight wrappers over existing materialize + gather ops

## 5. cjc-ad autodiff roadmap

cjc-ad is the load-bearing dependency for cjc-tempest's HMC. The audit identified a critical determinism issue that **must land before HMC**.

### 5.1 Current capability summary

Reverse-mode AD via `GradGraph` (arena-based: flat `ops`, `tensors`, `param_grads` arrays). 40+ ops (arithmetic, transcendental, activation, normalization, loss). Fused MLP layers (collapse transpose + matmul + bias-add + activation into one node). PINN support module (`pinn.rs`) with harmonic loss training, dual-numbers for forward-mode scalars. Language-level access via dispatch builtins (Phase 3c shipped 24 `grad_graph_*` primitives).

### 5.2 GradGraph arena layout

Three parallel `Vec<>` arrays for zero-copy node access: `ops[idx]` holds operation type (Add, MatMul, MlpLayer, etc.), `tensors[idx]` stores forward-pass results, `param_grads[idx]` accumulates parameter gradients. Each node indexed by `NodeIdx` (u32 newtype from Phase 2), eliminating type confusion. Node construction is append-only.

The `reforward(start..=end)` method recomputes forward-pass tensors for a contiguous range without rebuilding the graph — PINN-specific optimization that allows `set_tensor()` on a parameter, reforward affected downstream nodes, backward again without graph reconstruction. Fused `mlp_layer(input, weight, bias, activation)` collapses four nodes into one, reducing graph size 3× per layer.

### 5.3 Top 5 v0.x priorities

#### A1: Determinism audit + Kahan everywhere (BLOCKING HMC)

**Why**: cjc-tempest's HMC byte-identical-posteriors claim depends on cjc-ad producing bit-identical gradients across platforms. Current backward pass uses `Tensor::add_unchecked()` which doesn't mandate reduction order — on platforms with different SIMD widths or thread interleaving, accumulation order may vary, yielding ULP-level differences. This is the single highest-priority cjc-ad item.

**Sketch**:
- Audit all `accumulate_grad()` and similar in `lib.rs` backward
- Wrap all gradient accumulation in `KahanAccumulatorF64` at the element level, OR mandate `Tensor::add_assign_unchecked()` uses Kahan internally (move responsibility into cjc-runtime)
- Add `GradGraph::verify_determinism(other_graph) -> bool` method that checks bit-identical forward+backward across two runs
- Cross-platform CI matrix on the verify_determinism test

**Complexity**: Low-medium. Mostly mechanical audit; testing is the hard part. **Must ship before tempest HMC Session 2 starts.**

#### A2: Native double-backward (Phase 3d) for HMC + Hessian

**Why**: HMC needs the Hessian of log π(θ|y) for proposal calibration. Currently `hessian()` is O(p²) forward-backward pairs. Native double-backward would enable mixed-mode AD (reverse-on-reverse) and cut cost to O(1) forward + O(1) reverse.

**Sketch**:
- Extend `GradOp` to track intermediate Jacobians
- Implement `GradOp::Grad` representing a backward node; run backward on the backward graph
- Add `grad_graph_double_backward` builtin so users can request `H @ v` from `.cjcl` source
- JAX and PyTorch use this pattern

**Complexity**: High (subtle bugs in mixed-mode accumulation are common; careful management of gradient-of-gradient graph structure).

#### A3: JVP / forward-mode batched jacobian (vmap-like)

**Why**: Current `Dual` is scalar-only; `jacobian()` runs backward once per output element. Vectorized forward-mode would compute `J·v` for arbitrary `v` in a single pass. cjc-causal's deferred MLP nuisances benefit from fast batch Jacobian.

**Sketch**:
- Implement dual-number arithmetic at the tensor level: each element carries a tangent vector
- Fuse multiple forward passes into a single forward pass with tangent bundles
- Add `forward_mode_grad()` builtin

**Complexity**: Medium-high (tensor API redesign; avoid explosion of intermediate memory).

#### A4: Convolution and strided operations

**Why**: No conv1d/conv2d/transpose_conv limits applicability to signal/image domains. PINNs for image data, vision transformers, denoising diffusion all need this.

**Sketch**:
- Add `GradOp::Conv2d(input, kernel, bias, stride, padding)`, `TransposeConv2d(...)`
- Forward via im2col or FFT (if cjc-runtime permits); backward via dual convolutions

**Complexity**: Medium (convolution backward is well-known; main cost is integration with Tensor's stride layout).

#### A5: Sparse gradients and masked operations

**Why**: Graphs with many zeroed parameters (masked attention, pruning) accumulate dense gradients inefficiently. cjc-causal's matching stage outputs sparse propensity weights; sparse-aware AD makes parameter updates effectively free.

**Sketch**:
- Introduce `SparseGrad` wrapper storing (indices, values) pairs
- Accumulation becomes sparse-sparse or sparse-dense add
- Add `mask()` op that zeros gradients outside a mask

**Complexity**: Medium.

### 5.4 v0.2 vs v0.3 vs deferred

**v0.2 (next, after the determinism audit)**:
- `hessian_full()` memory-efficient (currently O(p²) space; use incremental assembly)
- `sparse_grad()` on-demand sparsity wrapper
- `forward_mode_grad()` batched tangent for small p
- Determinism audit + Kahan everywhere + verification test

**v0.3 (Phase 3e)**:
- Native double-backward with `grad_graph_double_backward` builtin
- Conv1d/Conv2d ops (forward and backward)
- `vmap()` / batched jacobian without loop unrolling
- Attention (scaled dot-product) as a fused op

**Deferred (Phase 4+)**:
- Sparse AD with automatic format selection
- JAX-style `jit` compilation (graph optimization + lowering)
- Quantized arithmetic (int8/int4 forward, float32 backward)
- Custom user-defined ops with autodiff registration

### 5.5 Composition gaps with decision-layer crates

**HMC (cjc-tempest) needs**:
- The Hessian of log π(θ|y). Currently `hessian()` is O(p²) forward passes; native double-backward makes HMC feasible for p > 100
- **Missing**: `backward_collect_hessian(loss, param_idx) -> [p, p] matrix`
- **Action**: Add the method; implement via finite-difference jacobian of gradient (current fallback) but optimize the loop

**DML (cjc-causal deferred MLP nuisances) needs**:
- Fast batch Jacobian J of MLP output w.r.t. inputs (for moment equations)
- **Missing**: `jacobian()` loops over output elements; no vectorized path for multiple RHS
- **Action**: Extend `jacobian()` to accept optional RHS matrix; use forward-mode batching

**KdV / Allen-Cahn PINNs need**:
- Automatic second derivatives via AD (not just finite-difference inside `pinn_harmonic_train`)
- **Missing**: No `grad_graph_hessian()` builtin for language-level access
- **Action**: Add dispatch builtins: `grad_graph_jacobian()`, `grad_graph_hessian()`, `grad_graph_hessian_diag()`

### 5.6 Existing TODOs

- `Phase 3d`: native higher-order AD — A2 above
- `Phase 3e`: language-level Hessian + conv — A2 + A4 above
- `pinn.rs`: harmonic loss is well-tested; biharmonic and tri-harmonic deferred
- `dispatch.rs`: 24 `grad_graph_*` builtins shipped in Phase 3c; double-backward + Hessian require new builtins (Phase 3d)

### 5.7 Determinism audit findings (critical for HMC)

**Kahan summation status**:
- ✓ Used in: `softmax()`, `cross_entropy()`, `layer_norm()`, `batch_norm()`, `clip_grad_norm()`, `pinn_mlp_eval_grid()`, `pinn_l2_max_errors()`
- ✗ **Missing in**: `Sum` backward (expands and adds grads in graph order), `Mean` backward (ditto), gradient accumulation in reductions without explicit `KahanAccumulator`
- ✗ Backward pass in `backward_with_seed()` and `backward()` do NOT use Kahan for gradient accumulation — rely on raw `add_assign_unchecked()` or `add_unchecked()` on Tensors

**Platform dependency risk**: `accumulate_grad()` does `grads[idx].add_unchecked(&grad)` which is tensor-wise element addition. `Tensor::add_unchecked()` does not mandate a reduction order. On CPU with different SIMD widths or GPU with thread interleaving, accumulation order may vary, yielding ULPs of error. **This violates cjc-tempest's determinism contract if HMC relies on gradient reproducibility.**

**Recommended actions**:
1. Either wrap all backward gradient accumulation in `KahanAccumulatorF64` at the element level
2. Or mandate that `Tensor::add_assign_unchecked()` uses Kahan internally (move responsibility into cjc-runtime)
3. Add `GradGraph::verify_determinism(other)` API to assert bit-identical fwd+bwd across two runs
4. Add cross-platform CI matrix test on the verifier

### 5.8 Ergonomics improvements

1. **Gradient history / trace format**: `GradientFormat` enum (Dense, Sparse, Log) + `GradGraph::set_gradient_format()` to trade space for speed
2. **Graph debugging / introspection**: `grad_graph_viz()` builtin that emits DOT-format graph or human-readable node list
3. **Composite loss helper**: `PinnLossBuilder` struct that accepts callbacks for each component (data + physics + boundary) and returns a single scalar loss node

## 6. Cross-crate composition opportunities

### 6.1 Locke ↔ all decision-layer crates

Pattern shipped in cjc-causal: `estimate(df, ..., locke_report: &LockeReport)`. cjc-cronos and cjc-tempest follow.

**v0.9 evolution**: Locke P1 (Inter-crate finding routing) standardizes the composition — sibling crates emit `LockeReport` subsets and Locke's algebra composes them. The result: a caller running `validate` + `forecast` + `estimate` + `sample` gets ONE unified `BeliefReport` with axis-by-axis attribution.

### 6.2 cjc-ad ↔ cjc-causal (DML MLP nuisances)

Currently cjc-causal v0.1 ships linear OLS nuisances. v0.2 adds MLP nuisances via `cjc_ad::GradGraph`. The composition needs:
- A clean `MlpNuisance::fit(x_train, y_train) -> MlpNuisance` builder that wraps GradGraph + adam_step
- `MlpNuisance::predict(x_test) -> Vec<f64>` for cross-fitting
- Byte-identical reproducibility (depends on cjc-ad A1: determinism audit)

### 6.3 cjc-ad ↔ cjc-tempest (HMC + NUTS)

HMC and NUTS both need `∇ log π(θ|y)`. Two paths:
- **Caller supplies analytic gradient**: faster, simpler, but user must compute by hand
- **GradGraph wrapper**: cjc-tempest exposes `with_grad_graph_log_posterior(builder)` that takes a closure returning the loss node; auto-derives gradient via backward

The `with_grad_graph_log_posterior` path needs cjc-ad determinism (A1). NUTS additionally needs cjc-ad A2 (native double-backward) for adaptive proposals in v0.2.

### 6.4 cjc-data TidyView ↔ all consumers

cjc-data is the universal substrate. TidyView verbs flow into:
- cjc-locke `validate(df)` — already wired
- cjc-causal `estimate(df, treatment, outcome, covariates, ...)` — already wired
- cjc-cronos `TimeSeries::from_dataframe(df, time_col, value_col)` — already wired
- cjc-tempest — not yet wired (tempest accepts a closure, not a DataFrame; v0.2 may add a `data_log_likelihood(df, model)` adapter)

TidyView T1 (window functions) + T2 (string ops) + T3 (column selector helpers) make this flow much cleaner.

## 7. Definition of done per workstream

### 7.1 cjc-tempest v0.1 surface complete

- [ ] cjc-ad determinism audit (A1) lands first
- [ ] HMC ships on `feat/cjc-tempest-scaffolding` with ≥25 unit + ≥12 integration + headline determinism test
- [ ] NUTS ships on same branch with ≥25 unit + ≥12 integration + headline cross-platform determinism on Linux/macOS/Windows CI
- [ ] Vehtari R-hat + ESS ships with ≥20 unit + ≥10 integration including E9301/E9302/E9303 emission tests
- [ ] ADR-0045 status: Accepted for all four sampler items
- [ ] `publish = false` lifted; tempest ships to crates.io at workspace version

### 7.2 cjc-causal + cjc-cronos v0.1.x release

- [ ] CI matrix expanded to Linux + macOS + Windows
- [ ] READMEs written for both crates with tagline + quick-start + ADR link
- [ ] `cargo doc --workspace --no-deps` clean
- [ ] Workspace version bumped to 0.1.12 (Option A from §2.2)
- [ ] `publish = false` lifted from both crates
- [ ] cjc-causal v0.1.12 + cjc-cronos v0.1.12 published to crates.io
- [ ] docs.rs build succeeds for both
- [ ] Blog post drafted (not necessarily posted) for `adamezzat1.github.io/blog/posts/`

### 7.3 Locke v0.9

- [ ] P1 (inter-crate finding routing) lands, with cjc-causal/cronos/tempest emitting composable `LockeReport` subsets
- [ ] P2 (evidence-weighted penalties + threshold profiles) lands with 3 pre-tuned profiles
- [ ] P3 (count-min sketch streaming + per-column belief) lands; `LockeReport` schema_version bumped
- [ ] P4 (confounder hints) lands as opt-in via `CausalConfig::detect_confounders`
- [ ] P5 (quasi-identifier PII) lands with E9094 emission
- [ ] `crates/cjc-locke/README.md` written
- [ ] E9xxx code registry added to `lib.rs`
- [ ] Locke v0.9 ships at workspace version bump

### 7.4 TidyView v0.x

- [ ] T1 (window functions: lag, lead, cumsum, cummax, cummin, rolling_*) lands
- [ ] T2 (string operations in mutate: trim, upper, lower, sub, replace, detect) lands
- [ ] T3 (column selector helpers: starts_with, ends_with, matches, all_numeric, etc.) lands
- [ ] T4 (nested groupby + multi-key joins + right_join/full_join wired to dispatch) lands
- [ ] T5 (API ergonomics: varargs sugar, error context, fluent aggregation) lands
- [ ] `to_time_series()` + `to_design_matrix()` adapters land
- [ ] Bench results documented (lag/lead on 10M-row frame, rolling mean on 10M-row frame)

### 7.5 cjc-ad

- [ ] A1 (determinism audit + Kahan everywhere + `verify_determinism()`) lands FIRST — blocking
- [ ] A2 (native double-backward Phase 3d) lands
- [ ] A3 (JVP / batched jacobian) lands
- [ ] A4 (Conv1d/Conv2d) lands (optional for v0.2; required for vision PINNs)
- [ ] A5 (sparse gradients + masked ops) lands (optional for v0.2)
- [ ] `grad_graph_jacobian()`, `grad_graph_hessian()`, `grad_graph_hessian_diag()` language-level builtins land
- [ ] Cross-platform CI matrix passes on `verify_determinism` test

## 8. Recommended execution order

The dependency graph forces an order:

```
                               ┌─────────────────────────┐
                               │ cjc-ad A1: determinism  │
                               │ audit + Kahan + verify  │
                               └──────────┬──────────────┘
                                          │ BLOCKING
                                          ▼
                  ┌───────────────────────────────────────┐
                  │ cjc-tempest S2: HMC                   │
                  │ (uses cjc-ad backward gradient)       │
                  └──────────────────┬────────────────────┘
                                     │
                  ┌──────────────────▼────────────────────┐
                  │ cjc-tempest S3: NUTS                  │
                  │ (extends HMC's leapfrog + adds tree)  │
                  └──────────────────┬────────────────────┘
                                     │
                  ┌──────────────────▼────────────────────┐
                  │ cjc-tempest S4: Vehtari diagnostics   │
                  │ + headline cross-platform CI matrix   │
                  └──────────────────┬────────────────────┘
                                     │
                  ┌──────────────────▼────────────────────┐
                  │ cjc-tempest publish to crates.io      │
                  └───────────────────────────────────────┘

Independent parallel tracks:
  cjc-causal + cjc-cronos release engineering  ──── can ship any time
  Locke v0.9 P1-P5                              ──── can ship any time
  TidyView T1-T5                                ──── can ship any time
  cjc-ad A2-A5                                  ──── after A1
```

**Recommended sprint plan**:

**Sprint 1 (1 session)**: cjc-ad A1 (determinism audit). **Must finish before HMC starts.**

**Sprint 2 (2 sessions, parallel)**:
- Track A: cjc-tempest S2 HMC
- Track B: cjc-causal + cjc-cronos release engineering (READMEs, CI matrix, publish)

**Sprint 3 (2 sessions)**: cjc-tempest S3 NUTS

**Sprint 4 (1 session)**: cjc-tempest S4 Vehtari diagnostics + cross-platform CI

**Sprint 5+ (parallel ongoing)**: Locke v0.9, TidyView roadmap, cjc-ad A2-A5

### 8.1 Branch strategy

| Workstream | Branch |
|---|---|
| cjc-ad A1 | `feat/cjc-ad-determinism-audit` (new, from origin/master) |
| cjc-tempest S2-S4 | `feat/cjc-tempest-scaffolding` (existing) |
| cjc-causal + cjc-cronos release | `release/0.1.12` (new) |
| Locke v0.9 | `feat/locke-v0.9` (new) |
| TidyView roadmap | `feat/tidyview-windows` (new) |
| cjc-ad A2-A5 | `feat/cjc-ad-phase-3d` (new, after A1 lands) |

Each branch is independent; the dependency graph determines which branches can merge in which order.

## 9. Open questions for the next session driver

These are explicit questions for the next session(s) to decide and document:

1. **cjc-tempest HMC closure interface** — Caller supplies analytic gradient OR `GradGraph` builder closure: should both be supported in v0.1, or pick one and make the other v0.2?
2. **cjc-tempest NUTS adaptation seed split** — `base_seed ^ ADAPT_SALT` vs `base_seed ^ SAMPLE_SALT`: confirm both constants are bit-distinct hex literals (no weak domain separation)
3. **Cross-platform CI cost** — Adding Linux + macOS + Windows to every PR roughly triples CI minutes. Acceptable for release-engineering PRs; should regular PRs only test ubuntu-latest?
4. **Locke v0.9 schema_version bump** — Going to 2 breaks JSON-parsing of v1 reports. Strict break or migration path?
5. **TidyView `lag(col, n)` semantics on group_by** — Within group only, or across the entire frame? (dplyr's lag respects groups.)
6. **cjc-ad determinism contract scope** — Does the byte-identity claim extend to thread-parallel BLAS via cjc-runtime, or only to single-threaded backward? Document the boundary.
7. **cjc-causal v0.2 MLP nuisances** — Land after cjc-ad A1 + A2, or earlier with linear-nuisance-only fallback when GradGraph is unavailable?

## 10. Final reminder for the session drivers

Re-read the [[CLAUDE|Prime Directives]] before opening any source file:

1. Do not break the compiler pipeline.
2. Do not introduce hidden allocations or GC usage in NoGC-verified paths.
3. Maintain deterministic execution — same seed = bit-identical output.
4. Preserve backward compatibility unless explicitly impossible.
5. Never silently refactor unrelated systems — scope changes to the feature being implemented.
6. Language primitives must stay minimal — higher-level functionality belongs in libraries.
7. Both executors must agree.

For cjc-tempest specifically, directive 3 (determinism) is non-negotiable — that's the entire reason the crate exists. The cjc-ad A1 audit lands FIRST not because HMC code wouldn't work without it, but because the byte-identical-posteriors claim cannot survive if cjc-ad's backward pass uses non-Kahan reductions.

For the v0.1.x releases (cjc-causal, cjc-cronos), directive 5 (no silent refactors) is the most load-bearing: the release pass should ONLY touch READMEs, version bumps, and CI config. Any code changes that sneak into the release commit make the release notes lie about what changed.

For Locke v0.9, directive 4 (backward compat) is the chief constraint — Locke is now consumed by three sibling crates (cjc-causal + cjc-cronos + cjc-tempest) plus the LendingClub demo. Schema changes need a v9-on-v8 migration path documented in the ADR.

For TidyView, directive 1 (compiler pipeline) is the chief constraint — TidyView verbs are exposed through `cjc-cli`'s `cjcl tidy` command and through MIR-execution paths. Any new verb must work in both AST-eval and MIR-exec.

For cjc-ad, directive 2 (no hidden allocations) is the chief constraint — `GradGraph` is the NoGC-verified path that PINN training relies on for memory predictability. New ops must not introduce hidden boxes / Rc / RefCell.
