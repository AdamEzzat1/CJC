# Handoff — After the D→I Arc: What's Done, What's Next

**Date:** 2026-06-13
**Branch:** `claude/stupefied-liskov-83b258` — HEAD `8c1c09a` (Phase I).
**Merge state (verified):** branch is **4 commits ahead of master**
(F1 `c32d474`, G `a784dc6`, H `670ceb4`, I `8c1c09a`); master is at
`8f1c655` and has **1 commit the branch lacks** — the
`cana_pass_ordering` ranker-type fix (the spun-off chip task, landed on
master independently). So master↔branch have DIVERGED: merging is a
2-parent merge (or merge master into the branch first to linearize).
D+E+F0 were merged to master earlier (`21bb1f7` lineage). Re-verify with
`git log --oneline master..HEAD` and `HEAD..master` before merging.
**Supersedes as entry point:** `docs/cana/HANDOFF_PHASE_D.md` (its banner
chronicles D+E+F0; THIS doc consolidates through I).
**Numbers ledger:** `docs/cana/PINN_V2_DESIGN.md` §11–§17.
**Determinism gate:** `docs/cana/DETERMINISM_CONTRACT.md` — the
10-invariant checklist every optimization is checked against. READ IT
before any runtime/executor change.

---

## 0. THE RULE (paid for again this arc)

Verify every claim against the code before building on it. This arc's
additions to the casualty list: the design panel's "contained" speed/
memory wins were mostly refuted on contact (closure-env already a move;
Tensor-Rc is 250 sites; view micro-opt <2%) — see Phase I. And Phase H
inverted Phase G (margin gating became counterproductive once the head
was fixed). `git log` and code are authoritative; this doc is advice.

## 1. What the D→I arc shipped (one line each)

| Phase | Commit | Result |
|---|---|---|
| D | `ce0ea0e` | Selector wall-clock wins **CONFIRMED on silicon** (5/6 whole-band, 2.7–3.5×, frozen holdout); energy formula underprices allocations |
| E | `48ba825` | Compression measured: traces **35–43×** lossless, disk 8.3×, checkpoint 1.38× (honest miss) |
| F0 | `21bb1f7` | Memory label variance unblocked **120×** (std 0.0009→0.108) — runtime allocation now observable |
| F1 | `c32d474` | Memory head **PROMOTE** (R²(test) 0.019→0.088; beats both baselines); existing heads byte-identical |
| G | `a784dc6` | Margin gating: selector regressions **16→7** (τ=0.02 dominated *then*) |
| H | `670ceb4` | Exploration configs: ungated selector regressions **16→1**; **margin gating SUPERSEDED** (τ=0 now best) |
| I | `8c1c09a` | `Value` slimmed **88→72 B** (boxed SparseCsr); verified perf roadmap; determinism contract doc |

**Honest project status:** the selector is now **6 wins / 1 regression**
(modeled energy) — the closest to default-on it has ever been, from
"ablation-grade, 16 regressions" three phases ago. Still NOT default-on
(1 residual regression; modeled-not-yet-stopwatch on the gated net
effect). Phase D remains the only SILICON-proven win.

## 2. The selector's current state (important — Phase H inverted G)

- **Calibrated default is now τ = 0** (no margin gating). Phase H's
  exploration configs fixed the energy head directly, so the ungated
  selector (`selector_rec`) is best: 6 wins, 1 regression. The
  `selector_mg_rec_t*` configs now show *more* regressions — gating
  re-introduces them by suppressing the improved head's correct
  switches.
- `PassPlanSelector::with_margin` is KEPT (byte-identical no-op at τ=0;
  correct tool for a future noisier head) — do not rip it out.
- **The margin-gating sweep configs could be retired** from the default
  corpus now that τ=0 won (they remain as the audit trail of WHY). Low
  priority.

## 3. What's next (priority order)

### 3.1 The 1 residual selector regression — non-linear head
The last regression is a *confident* head misprediction that anchors
can't reach (Phase H §4.3). The lever is model class, not more data: a
tiny MLP energy head (GradGraph + `adam_step`, offline, seeded — the
project already trains MLPs this way) OR a pairwise-interaction feature
(licm×unroll, cf×dce products) added to the linear basis. Either is a
FeatureHash/bundle change → regen → retrain → re-shadow. Scoped, modest.

### 3.2 Performance roadmap — `PERFORMANCE_ROADMAP.md` (dedicated sessions)
The "improve speed/memory as much as possible" continuation. Each is a
focused session because the blast radius forbids rushing:
- **P1 — `Tensor.shape/strides` → `Rc<[usize]>`** (memory + speed,
  highest leverage): shrinks Tensor 64→~40 B AND elides Vec allocations
  in the hot view ops (transpose/reshape/slice/broadcast). ~250 sites in
  `tensor.rs` + `builtins.rs` + `cjc-ad`. Measure: `alloc_bytes` on the
  `tensor_*` corpus + `cana_diagnostics` wall-clock on `tensor_mm`.
- **P2 — `Value` cold-variant boxing** (memory): after P1, box
  `Enum`/`Closure`/`Struct`/`Regex` toward `Value` ~40 B. Pervasive match
  sites; profile first (don't box a hot variant).
- **P3 — non-escaping array/tuple literal elision** (memory + speed):
  the Phase-D allocation-churn continuation — frame-arena-back literals
  escape analysis proves don't escape. Highest care (correctness of the
  escape proof); heavy proptest/bolero on escape edges. Measure on
  `mem_grad_*` + `holdout_alloc_pulse`.
- **P4 — softmax/layer-norm scratchpad pooling** (memory): ACCURACY-
  GATED — must preserve Kahan accumulation order; only with a
  bit-identical chess-RL weight-hash proof.

### 3.3 Energy-formula recalibration (Phase D §4.1)
Phase D found the formula underprices allocation statements ~2–3× (why
measured wins exceeded modeled). Adding an allocation-statement weight
beside `FP_ENERGY_WEIGHT` would make the selector's objective truer —
and Phase F0/F1's `creation_alloc` signal is exactly the input. Relabel
→ retrain energy → reselect → re-measure (Phase D stopwatch). Larger arc.

### 3.4 Stopwatch-confirm the post-H selector
Phase H's 16→1 is MODELED. `bench/cana_diagnostics` (Phase D's harness)
would put the ungated post-H selector on the wall clock — confirm the
net effect (wins real, gated-away regressions were real). Natural,
self-contained.

### 3.5 Housekeeping
- **Merge to master**: the whole D→I arc. One merge from the main
  worktree.
- Margin-gating sweep configs retirement (§2, low priority).
- The committed corpus is now **4,740 rows** (Phase H added 6 exploration
  configs). Energy bundle is the Phase-H retrained one; thermal +
  memory bundles unchanged.

## 4. Test discipline contract (unchanged) + the verification loop

Wiring → unit → proptest → bolero → verification loop → docs. The loop:
```
cargo test --test fixtures --release                 # THE parity gate (invariant 7)
cargo test --test test_parity_stress --release       # 50-seed stress (FP determinism)
cargo test -p cjc-cana --release --lib               # 223
cargo test -p cjc-cana-compress --release            # codecs + bundles
cargo test -p cana-diagnostics --release             # 23 (D harness, determinism gates)
cargo test -p cjc-runtime --test value_size_guard    # Phase I size ceiling
cargo run --release -p cana-ablation                 # regen (~5 min) + asserts
cargo run --release -p cana-train-pinn -- shadow-energy   # selector head regret gate
```
After any corpus-affecting change: regen → retrain (all heads) →
re-shadow → verify bundle fixed points byte-identically.

## 5. Traps (accumulated; verified this arc)

- **Phase H inverted Phase G** — don't reason about the selector from G's
  conclusions; τ=0 is now best (§2).
- **`.contains("_rec")` substring trap** (trainer): the new `force_*`
  exploration configs contain no `_rec`, so they're correctly in energy
  training and out of the thermal/memory program datasets. Any new
  *selector*-driven config must go in `ENERGY_EXCLUDED_CONFIGS`; any new
  *forced* config must NOT.
- **`Value` size ceiling = 72 B** (`value_size_guard.rs`): a new large
  variant re-bloats every value; box its payload (the boxed-SparseCsr
  precedent). Don't box hot variants (Tensor/Enum).
- **MirTraceEvent / FnFeatures / schema versions**: any field change
  ripples FeatureHash → every row hash → full regen → retrain both/all
  heads → re-shadow. PROFILE schema is v4 (Phase F1).
- **Determinism contract is final authority**: no FMA, BTreeMap-only,
  Kahan/Binned reductions, total_cmp, FNV-1a, SplitMix64. A faster-but-
  nondeterministic change is a regression — redesign, don't force.
- PowerShell 5.1 mangles inner double-quotes in native-exe args — use
  here-strings without inner `"` for git commit messages.

## 6. Determinism invariants — unchanged, non-negotiable

BTreeMap everywhere; Kahan/Binned FP reductions; no FMA; SplitMix64;
FNV-1a; `f64::total_cmp`; no wall-clock/sensors in decision paths or
hashes (diagnostics are post-hoc, byte-equality-gated first); parity
(AST-eval ≡ MIR-exec) is the final authority. Full enforcement map:
`DETERMINISM_CONTRACT.md`.

---

*Closing artifact of the 2026-06-13 stacked-role optimization arc
(Phases G+H+I on top of the D+E+F session). The selector went from
ablation-grade (16 regressions) to 6 wins / 1 regression; `Value`
slimmed 18%; the larger perf wins are scoped, measured-first, and
determinism-gated for the sessions that will execute them.*
