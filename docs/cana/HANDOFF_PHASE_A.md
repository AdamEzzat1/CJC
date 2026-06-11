# Handoff — Phase A Complete: Data + Safety Foundation, Measured

**Date:** 2026-06-11
**Branch:** `claude/hopeful-elion-72485e` (worktree
`C:\Users\adame\CJC\.claude\worktrees\hopeful-elion-72485e`)
**Supersedes as entry point:** `docs/cana/HANDOFF_NEXT_ARC.md` (Phase A
fully executed; Phases B–F remain — sequenced there, evidenced in
`NEXT_ARC_RESEARCH.md`, with §7 of that doc now logging settled
hypotheses)

---

## 0. THE RULE, NOW PAID FOR ELEVEN TIMES

**Verify every handoff/design claim against the code before building on
it.** Prior arcs caught nine doc-vs-code drifts; Phase A added two more
findings of the same species — this time *instrument-vs-reality* drifts
that no amount of doc reading could catch, only experiments:

10. The thermal LABEL was tensor-blind, not just the feature basis
    (handoff §6 hypothesized feature-side blindness; the probe measured
    label-side too — "head matches label" on tensor programs was two
    blind instruments agreeing).
11. The research doc's 1.5× code-size bound contradicted the
    optimizer's DESIGNED behavior on first contact: the ranked
    baseline plan fully unrolls countable 8-trip loops at ~6.2× node
    growth (`grad_f10_d2_n64`: 97 → 605). Bound re-set to 16× with a
    measured corpus-max report printed every regen.

**`git log` and the code are authoritative; this doc is advice.**

## 1. What Phase A shipped (all measured, all gates green)

### A1 — Tensor blindness: measured, fixed dual-side, retrained (PINN_V2_DESIGN.md §7)

- `bench/cana_tensor_probe` (committed, re-runnable): recorded thermal
  read **0.0000** on programs running 409,600+ analytic FP ops; trace
  totals equalled the scalar subset to 0.00% error; static
  `est_float_ops` carried builder-scalar only; `tensor_heavy_ops`
  missed element-wise binops AND method forms (`.sum()`).
- Fix: runtime `trace_fp_ops` (u32→u64) prices tensor binops by element
  count, matmul by `2·m·k·n` from runtime shapes, curated FP-heavy
  builtins/methods at free-call + method dispatch choke points;
  `TypeMix` propagates tensor-ness (tensor binops EXCLUDED from scalar
  float count — runtime arm precedence); `MemoryProxy` classifies
  method callees; `build_physical_query` prices tensor sites at
  `TENSOR_FP_PER_OP = 128`; density feature capped at 1.0
  (`PINN_V2_MODEL_VERSION` 2→3).
- 9-program `tensor_` corpus family (4 hot-loop shapes + graded
  `tensor_tg_k{0..4}`); eval↔MIR parity locked BEFORE harness entry.
- Retrained head (v3): **shadow MAE held-out 0.0319 vs v1 0.2150
  (6.7×), corr +0.98, PROMOTE**; train↔regen fixed point verified
  byte-for-byte.

### A2+A3 — Schema v3: per-function labels + loop features (one bump)

- `FnProfile` per-function records in `CompilationProfile.per_function`:
  6 workload estimates + `countable_loop_count`/`max_loop_depth`
  (existed in `CfgMetrics`, never reached rows) + per-function
  cpu/memory/thermal labels (the per-program MAX remains as the
  aggregate fields). `PROFILE_SCHEMA_VERSION = 3`, no migration,
  regenerate.
- Tests per the discipline contract: round-trip units (0/1/many fns,
  hash sensitivity), proptest round-trip ∀ per-function contents,
  bolero decoder fuzz (arbitrary bytes → Err-not-panic).

### A4 — Memory-gradient family + the honest label finding

- **The memory label is structurally blind to Rc memory** (the A1
  pattern, third instance): `heap_bytes_in_use = gc_live×4096 +
  arena_alloc_count×64`, where `gc_live` moves ONLY via the explicit
  `gc_alloc` builtin and `arena_alloc_count` counts executed
  arena-classified `Let`s — flat 64 bytes, size-blind, cumulative.
  Arrays/tensors/strings (the real memory) are invisible. THIS is why
  memory std was 0.0007 — not corpus composition.
- `mem_grad_a{1..5}` (×4 arena-let executions per step) measures what
  the label CAN show; the measured spread is recorded in §3 below.
  The research-doc expectation "std 0.0007 → ~0.1+" was **not
  achievable against this label** — Phase F must start with a
  label-side instrumentation fix (per-window allocation-bytes counter
  at creation sites, or Rc-lifecycle tracking), not with programs.

### A5 — Verifiers wired into the harness

- Every optimized program in every config now passes
  `cjc_mir::nogc_verify::verify_nogc` + `cjc_mir::verify::
  verify_mir_legality` (hard panic = regen gate failure) plus the 16×
  code-size cap with a measured corpus-max report.

### A6 — Cross-profile determinism canary: PASS

- Dev-profile and release-profile training produce **byte-identical**
  CPB0 bundles (SHA256 verified) despite release's `lto = true` +
  `codegen-units = 1`. The bit-reproducibility claim now spans build
  profiles.

### A7 — Frozen holdout set

- 10 `holdout_`-prefixed programs (new shapes: collatz, gcd grid,
  logistic map, mandelbrot cell, EMA mix, rect matmul, tensor scale
  chain, deep calls, alloc pulse, FP/int alternation). The trainer
  excludes them from BOTH train and FNV test split
  (`is_holdout_program`); shadow mode reports them as a separate
  cohort. They exist so promotion gates keep one cohort that never
  erodes.

## 2. Verification state at handoff

The §4 test-discipline loop (handoff_next_arc) ran fully green after
A1-fix, and again after schema v3 + families (numbers in §3). New
gates: `tests/test_tensor_fp_accounting.rs` (6),
`tests/test_tensor_typemix_props.rs` (4 incl. bolero), cjc-cana lib
195, cjc-cana-compress lib 181 + proptest/bolero additions.

## 3. Final corpus + training numbers (measured this session)

- Corpus: **3,160 rows = 158 programs × 20 configs**
  (134 prior + 9 `tensor_` + 5 `mem_grad_` + 10 `holdout_`).
  Parity 100%; row-hash double-run byte-identical; NoGC + MIR-legality
  verifiers green on all 3,160 optimized programs; corpus-max
  code-size ratio reported each regen (designed full-unroll measured
  at 6.24×; cap 16×).
- Per-function labels (schema v3): **360** (vs 158 per-program);
  fn-thermal std 0.3214, fn-cpu 0.0128, fn-memory 0.0008.
- Memory label (A4): max 0.0078 (`mem_grad_a5`), std 0.0009 —
  mechanism-exact; the std > 0.05 target is unreachable against the
  Rc-blind label (see §1 A4, §5).
- Retrain (114/34/10 split): R²(train) 0.9587, **R²(test) 0.9552**,
  density coefficient +0.7212; train↔regen fixed point verified.
- Shadow: v2(v3) MAE — train 0.0315 (corr +0.9865), FNV held-out
  0.0314 (+0.9820), **frozen holdout 0.1885 (+0.8107)** vs v1's
  0.4761 (−0.1038), overall 0.0414 (+0.9690). **PROMOTE.**
  Quote the FROZEN-HOLDOUT line externally; the FNV split erodes.
- Plan-level consequence: 38/158 plans differ between
  `full_pinn_v2_rec` and `full_pinn_rec`.

## 4. TEST DISCIPLINE CONTRACT — unchanged

Wiring → unit → proptest → bolero → verification loop → docs, for
every feature. The verification command list from
`HANDOFF_NEXT_ARC.md` §4 stands, plus:

```bash
cargo test --test test_tensor_fp_accounting --release   # 6
cargo test --test test_tensor_typemix_props --release   # 4
cargo run --release -p cana-tensor-probe                # probe: must report NOT BLIND
```

## 5. Traps (verified this session — will bite)

All §5 traps from `HANDOFF_NEXT_ARC.md` remain live, plus:

- **The trainer selects label rows by `config_id.contains("_rec")`**
  (`bench/cana_train_pinn/main.rs`, two sites). Currently safe (no
  non-recorded config contains the substring) but it's the same
  pattern family as the `ends_with` bug. If a new config name ever
  contains `_rec`, labels mislabel silently. Convert to membership
  lists when touching that code.
- **`thermal_intensity` caps at 1.0 per window** — Σ intensity·instr
  recovers only cap-clipped FP (~77% on FP-dense scalar statements).
  The harness energy formula inherits this. Phase B's energy target
  must either accept the clip or widen the event schema.
- **Code-size cap is 16×, not 1.5×** — full unroll of countable
  8-trip loops is ~6.2× by design (measured). The regen prints the
  corpus max each run; tighten from evidence only.
- **`Tensor` is a valid param/let type annotation** in CJC-Lang test
  programs, and method calls lower to `Field`-callee `Call` nodes —
  static classifiers must handle BOTH free-call and method forms (the
  A1 probe caught `tensor_heavy_ops` missing the latter).
- **The memory label cannot be moved >~0.01 by any honest program**
  under the current heap proxy (see A4). Don't write programs to chase
  the old std>0.05 exit criterion; fix the label first (Phase F).

## 6. Hypotheses vs facts (updated ledger)

| Claim | Status |
|---|---|
| Thermal head tensor-blind | **Measured + FIXED** (was dual; §1 A1) |
| v3 head beats v1 on held-out + overall + frozen holdout | **Measured** (PROMOTE) |
| Training bit-reproducible across build profiles | **Measured** (A6 PASS) |
| 1.5× size bound sane | **Refuted by measurement** → 16× evidence-based |
| Memory label can reach std > 0.05 with new programs | **Refuted by code + measurement** (label is Rc-blind) |
| Ridge + loop features reaches R² 0.65–0.75 on energy | **Hypothesis** (Phase B; schema v3 now carries the loop features) |
| Selector finds energy wins on real programs | **Hypothesis** (Phase C) |
| Quantum layer zero effect is structural | **Measured + code-verified** (unchanged) |

## 7. Phase B — EXECUTED same session (2026-06-11 PM): energy head PROMOTE

Full record: `PINN_V2_DESIGN.md` §9. Summary: sanity-energy settled
the R² hypothesis (0.8207 > the 0.65–0.75 band; −32-class failure
replicated at −16.98 under the old recipe); the regret-vs-R² fork was
the load-bearing finding (the R²-best fit loses to the always-baseline
heuristic on selector regret; the all-rows fit reaches +0.0014 test
regret, 32/34 exact-best, 10/10 frozen holdout); shipped
`pinn_energy_v1` (CPB1, vocabulary-carrying basis,
`crates/cjc-cana/src/pinn_energy_v1.rs` +
`crates/cjc-cana-compress/src/energy_bundle.rs` + trainer modes
`train-energy`/`shadow-energy` + `tests/test_energy_head.rs`).
Shadow verdict through the persisted artifact: **PROMOTE**.

**Next session: Phase C** (PassPlanSelector — research doc §1): 10
deterministic candidates/function, the trained energy head as
criterion, selector gates (legality-of-selected, determinism, on/off
parity, never-worse-than-baseline), ablation config, corpus re-run.
MANDATORY first step: re-validate selector regret on the REAL
10-candidate-per-function space (Phase B's regret used the 20-config
plan space as proxy). Traps: plan-absence semantics
(`cjc-mir/src/optimize.rs:342-356` — insert explicit per-function
entries); selector identity must join report hashes.

Open debt queued: serializer/replay 5 pre-existing failures
(`docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md`, chip task_a41b1c8d
raised); cjc-quantum wide-matrix SVD upstream fix; embed default
weights as compiled-in const; Locke/LendingClub peak-RSS macro A/B.

## 8. File map (Phase A surface, additive to prior maps)

```
bench/cana_tensor_probe/main.rs              A1 probe (label/feature/analytic 3-way)
crates/cjc-mir-exec/src/lib.rs               tensor FP accounting (binop/unary/call/method sites), u64 counter
crates/cjc-cana/src/type_mix.rs              tensor-ness propagation + tensor_binop_count
crates/cjc-cana/src/memory_proxy.rs          method-call classification (TENSOR_HEAVY_METHODS)
crates/cjc-cana/src/physical_cost.rs         TENSOR_FP_PER_OP pricing in float_ops_estimate
crates/cjc-cana/src/pinn_thermal_v2.rs       density cap + model v3
crates/cjc-cana-compress/src/profile_db.rs   schema v3 (FnProfile per-function records)
bench/cana_ablation/main.rs                  tensor_/mem_grad_/holdout_ families; verifier gates; size-cap report
bench/cana_train_pinn/main.rs                holdout cohort; per-fn label sanity section
tests/test_tensor_fp_accounting.rs           6 wiring tests
tests/test_tensor_typemix_props.rs           3 proptest + 1 bolero
docs/cana/PINN_V2_DESIGN.md                  §7 A1 record (+§8 Phase A numbers)
docs/cana/NEXT_ARC_RESEARCH.md               §7 settled-hypotheses log
```

## 9. Determinism invariants — unchanged, non-negotiable

BTreeMap everywhere; Kahan/Binned FP reductions; no FMA; SplitMix64
seeded RNG only; FNV-1a; `f64::total_cmp`; no wall-clock/sensors in
decision paths or hashes; model_id+version in report hashes; training
offline only; shadow before activation; legality gate final authority.
