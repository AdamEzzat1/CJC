# ADR-0045 cjc-tempest v0.1 — Metropolis, HMC, NUTS with byte-identical posteriors

- **Status:** Proposed (2026-06-02) — scaffolding committed, implementation pending across multiple sessions
- **Crate:** `cjc-tempest` (new)
- **Companion docs:** [[New Crate Stack — Cronos, Causal, Tempest]] (handoff §4), [[ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML]] (sister crate), [[ADR-0044 cjc-cronos v0.1 — ETS, ARIMA, Kalman, STL]] (sister crate), [[ADR-0016 Language-Level GradGraph Primitives]] (HMC's gradient dependency)
- **Reserved error-code range:** **E9300..=E9399**

## Context

The [[New Crate Stack — Cronos, Causal, Tempest|handoff]] §0 frames cjc-tempest as the *uncertainty leg* of the decision-layer triad. Where cjc-causal answers *"what would happen if I intervened?"* and cjc-cronos answers *"what will happen next?"*, cjc-tempest answers *"how uncertain am I, and why?"*.

**This is the most ambitious of the three crates** because the headline claim — byte-identical posterior chains — is structurally unattainable by any existing PPL. PyMC's JAX backend derives RNG from system entropy by default; Stan's NUTS uses a thread-local RNG that varies per platform; Turing.jl relies on Julia's `Random.GLOBAL_RNG` which is unsuitable for cross-platform reproducibility. None of them produce bit-identical posterior bytes when re-run.

**That gap is the publishable claim**: Bayesian model selection, posterior predictive checks, sensitivity analysis, and Bayes-factor comparisons all become reproducible. Published analyses can cite a `content_hash`; anyone re-running gets the same hash. This is structurally what makes cjc-tempest uniquely CJC-shaped.

## Decisions

### 1. New workspace crate `cjc-tempest`

Workspace member under `crates/cjc-tempest/`, mirroring the cjc-causal and cjc-cronos scaffolding precedent. `publish = false` until v0.1 implementation lands.

Path-deps: `cjc-ad` (HMC's reverse-mode gradient via [`GradGraph`](../../crates/cjc-ad/src/lib.rs)), `cjc-repro` (SplitMix64), `cjc-runtime`, `cjc-locke` (refusal check), `cjc-snap` (posterior chain serialization — chains can be 10000 samples × 50 params × 4 chains = 16MB; deserve content-addressed disk storage).

### 2. Closure-based model representation for v0.1

```rust
pub trait Sampler {
    fn label(&self) -> &'static str;
    fn step(&mut self, state: &mut [f64], rng: &mut Rng) -> AcceptResult;
}
```

The user supplies a log-posterior closure `Fn(&[f64]) -> f64`. The sampler internally constructs a [`GradGraph`](../../crates/cjc-ad/src/lib.rs) over this closure when reverse-mode AD is needed (HMC, NUTS).

**Rejected alternatives**:
- **DAG-based via cjc-abng** — composes well with existing tooling but adds setup overhead. v0.1 keeps the closure interface; cjc-abng wrappers come in v0.2.
- **`model { ... }` DSL block in `.cjcl`** — cleanest UX but requires new AST node + HIR lowering + MIR codegen. Separate ADR (v0.3+).

### 3. `PosteriorSamples.content_hash` is the headline output

```rust
pub struct PosteriorSamples {
    pub chains: Vec<Vec<Vec<f64>>>,
    pub n_chains: usize,
    pub n_samples_per_chain: usize,
    pub n_dim: usize,
    pub diagnostics: ConvergenceDiagnostics,
    pub content_hash: FingerprintId,
}
```

`content_hash` is content-addressed over `(sampler_label, model_hash, seed, n_chains, n_samples_per_chain, every sample bit-pattern, diagnostics)`. The hashing is the publishable claim: two runs with bit-identical inputs produce the same `content_hash`.

### 4. v0.1 sampler surface

| Sampler | Algorithm | Session | References |
| --- | --- | --- | --- |
| `MetropolisHastings` | Symmetric proposal + Welford-adaptive covariance during warmup | 1 | Roberts-Gelman-Gilks 1997; serves as determinism warmup |
| `HamiltonianMonteCarlo` | Leapfrog integrator + identity mass matrix | 2-3 | Neal 2011 Chapter 5 |
| `NoUTurnSampler` (NUTS) | Hoffman & Gelman 2014 Algorithm 6 + dual averaging | 3-4 | Hoffman-Gelman 2014 |

Convergence diagnostics (Vehtari et al. 2021 split-rank-normalised R-hat + bulk-ESS + tail-ESS via Geyer's initial monotone sequence estimator) ship alongside the samplers — likely Session 4 if NUTS spills over.

### 5. Reserved error-code range E9300..=E9399

| Code  | Severity | Trigger |
| ----- | -------- | ------- |
| E9300 | Warning  | Divergent transitions detected (HMC / NUTS energy error) |
| E9301 | Error    | R-hat > 1.01 on any parameter (convergence failure) |
| E9302 | Warning  | Bulk-ESS < 400 (insufficient effective sample size) |
| E9303 | Warning  | NUTS max tree-depth hit (tree expansion stopped before U-turn) |

Implementation sessions may add E9304..=E9399 codes.

### 6. Mass-matrix adaptation deferred to v0.2

v0.1 ships HMC and NUTS with **identity mass matrix** only. Mass adaptation (the diagonal-or-dense Welford accumulator that Stan and PyMC use) significantly increases scope (adaptation phase RNG, separate seed bookkeeping, byte-identity proof across the adaptation/sampling boundary). v0.1's "byte-identical posteriors" claim is unchanged by deferring this — identity mass still produces fully deterministic chains.

### 7. Library-only Rust API for v0.1 — no `.cjcl` `model { ... }` block

Per the handoff §4.7 deferral list, the cjcl-language `model { ... }` block is v0.3+. Its own ADR.

## Determinism contract (the entire reason this crate exists)

Per the [[CLAUDE|Prime Directives]] determinism rules. Tempest-specific contracts on top:

1. **Every RNG draw routes through [`cjc_repro::Rng`](../../crates/cjc-repro/src/lib.rs) (SplitMix64).** Auditor maintains a list of every site:
   - Initial state per chain (one `Rng` seeded per chain via SplitMix64 stretch from the caller's base seed).
   - Momentum draw per leapfrog kick (HMC, NUTS).
   - Acceptance/rejection coin flip per proposal.
   - Direction choice per NUTS tree expansion (left vs right).
   - Slice sample for NUTS (Hoffman & Gelman 2014 §3.1).
   - **Adaptation-phase RNG (warmup) is SEPARATE from sampling-phase RNG** — both seeded from the base seed but along different SplitMix64 branches so adaptation rounds don't perturb sampling-phase bytes.
2. **Reproducibility lock**: same seed + same model + same input data ⇒ byte-identical `PosteriorSamples.content_hash`. This is the headline proptest.
3. **Cross-platform parity**: same posterior content hash on Linux + macOS + Windows. The math is the hardest of any of the three crates to keep cross-platform stable; expect debug iterations.
4. **No HashMap** in any internal data structure. Adaptation history (step size, mass diagonal) goes in `BTreeMap` or `Vec` with explicit ordering.
5. **Leapfrog integrator accumulates energy errors with [`KahanAccumulatorF64`](../../crates/cjc-repro/src/kahan.rs)**. Even small per-step rounding compounds over a trajectory.
6. **Seed-flow diagram documented in `src/lib.rs` module documentation**. A future contributor MUST be able to trace where every random number comes from in < 10 minutes.

## Test surface (handoff §6.1 floors)

| Bucket          | Floor | Location |
| --------------- | ----- | -------- |
| Unit            | ≥ 25  | `crates/cjc-tempest/src/lib.rs` (in-module `tests`) |
| Integration     | ≥ 10  | `tests/tempest/` (multiple files per sampler) |
| Proptest        | ≥ 5   | `tests/tempest/tempest_proptest.rs` |
| Bolero fuzz     | ≥ 3   | `tests/tempest/tempest_fuzz.rs` |
| **Determinism (HEADLINE)** | (1, non-negotiable) | `tests/tempest/tempest_determinism.rs` — required on Linux + macOS + Windows. **If this test fails on any platform, the release does NOT ship.** |
| Reference parity (Stan)    | (optional)          | `tests/tempest/tempest_stan_reference.rs` — compare posterior means against Stan's 8-Schools output (direction-of-effect validation, not byte identity) |

Required proptest properties (handoff §4.6):

1. Same seed + same model + same data ⇒ byte-identical `content_hash`.
2. Samples are always in the model's support (positive constraints stay positive).
3. `ess_bulk ≤ n_samples_per_chain` (definitional).
4. R-hat for a single chain equals `1.0` by definition.
5. Doubling `n_chains` does not change per-chain content hashes.

## Alternatives considered

### A — Embed sampling inside cjc-ad

Pros: zero new crate; GradGraph already there.
Cons: bloats cjc-ad with sampler logic; cjc-ad is the autodiff layer, not the inference layer. Same separation cjc-causal / cjc-cronos enforced. Rejected.

### B — A pure Rust library outside the workspace

Pros: clean isolation.
Cons: loses workspace `Cargo.lock` reproducibility, can't share `cjc-ad::GradGraph` or `cjc-repro::Rng` without external versioning. Same reasoning as Locke ADR-0028 §C. Rejected.

### C (chosen) — New workspace crate `cjc-tempest`, mirroring cjc-causal + cjc-cronos

Pros: clean ownership boundary, shared lockfile, mirrors the established precedent for consistency, GradGraph + adam_step already shipped from [[Phase 3c — Language-Level GradGraph Primitives]]. Cons: one more crate to publish.

## What's deferred to v0.2+

| Capability | Defer to | Rationale |
| --- | --- | --- |
| Variational inference (ADVI / SVI) | v0.2 | Different inference paradigm; deserves its own surface. |
| Posterior predictive sampling | v0.1.1 / v0.2 | Small extension once samplers ship. |
| Reversible-jump MCMC | indefinitely | Orthogonal; complex implementation. |
| Sequential Monte Carlo | indefinitely | Different inference paradigm. |
| Mass-matrix adaptation (diagonal + dense) | v0.2 | Significantly increases scope; identity mass works for v0.1's headline claim. |
| `model { ... }` DSL block in `.cjcl` | v0.3+ | Separate ADR; largest single piece of compiler work the stack implies. |
| Marginal likelihood (Bayes factor) computation | v0.2 or later | Bridge sampling is its own discipline. |
| Python bindings via PyO3 | v0.1.1 / v0.2 | Mirrors the cjc-locke-py pattern. |

## Scope-discipline summary

v0.1 ships *the three samplers Bayesian-stats teaching tracks cover first* — Metropolis-Hastings (the simplest), HMC (the bridge to gradient-based), NUTS (the headline) — plus deterministic R-hat + ESS diagnostics + the byte-identity reproducibility contract. v0.2 expands to mass adaptation + variational inference + posterior predictive. v0.3 considers the `.cjcl` `model { ... }` block.

Per the [[CLAUDE|Prime Directives]]: directives 3 (determinism) and 5 (no silent refactors of unrelated systems) are the most load-bearing. **The byte-identical-posterior-chains story is THE reason cjc-tempest exists**; if you sacrifice it for a perceived performance win, you have given up the publishable claim — and unlike cjc-causal and cjc-cronos, there's no fallback claim that survives. The Determinism Auditor role is the gatekeeper for every Tempest PR; the seed-flow audit is the first thing a reviewer reads.
