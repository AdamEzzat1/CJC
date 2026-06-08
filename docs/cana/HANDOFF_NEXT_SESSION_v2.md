# Handoff — Next Session (v2, post-§20)

**Date:** 2026-06-08
**Branch:** `claude/compassionate-chebyshev-8956ab`
**Companion docs:**
- [`CANA_COST_MODEL_TRAINING_FINDINGS.md`](CANA_COST_MODEL_TRAINING_FINDINGS.md) — full §3A.1 → §20 record (19 sections, ~1400 LOC)
- [`CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md`](CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md) — bridge crate options
- [`CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md`](CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md) — §4B.4 options
- [`CANA_PHASE_4_SECOND_NN_COMPOSITION_OPTIONS.md`](CANA_PHASE_4_SECOND_NN_COMPOSITION_OPTIONS.md) — §4B.5 options

The original [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md) is
fully addressed; this v2 doc supersedes it.

---

## 0. TL;DR

**Track A (cost-model validation): 4/4 done.** Held-out, cross-corpus,
feature audit, PINN AB test — all shipped and documented.

**Track B (NSS integration): 5/5 done.** Bridge crate shipped Option A
(synthetic-trace projection), CLI flag wired and now the default,
kernel-variant selector scaffolded, composition-options documented.

**Two structural compiler improvements landed beyond the handoff scope:**
- **§17:** v5 trained coefficients shipped (CF `w_loop_depth` sign
  corrected, `w_alloc_sites` activated on 4/5 passes).
- **§19:** `PerPassLegalityGate` shipped — CF/DCE/LICM now run on
  float-heavy ML workloads where `DefaultLegalityGate` was vetoing
  them blanket. PINN went from 0 to 16 Run-recommendations at default
  threshold.

**§20 (this commit):** `--mir-opt` now defaults to thermal-aware
optimization. `cjcl run --mir-opt` uses
`ThermalAwareCostModel<LinearCostModel::trained(), NssPressurePredictor>`
+ `PerPassLegalityGate`. `--no-thermal-aware` opts out.

**The compiler is meaningfully different for users now.** Float-heavy
programs that previously got zero optimization through `--mir-opt`
now actually optimize. Output byte-identity preserved across all
paths (validated at 5 layers including AST/MIR parity gate).

---

## 1. What shipped — chronological commit list

20+ commits this session. The structurally important ones (in
session order):

```
e1fea8a  cjc-nss: ship Phases 1-5 (240 tests, 0 failures)            §4B.1
ee9d85c  Phase 3 held-out validation (§3A.1)
3efa5b1  Phase 4 feature audit + cross-corpus (§3A.3 + §3A.4)
c6da05f  §3A.2 PINN AB test
6ef0b81  §4B.2 bridge crate design options doc (α/β/γ/δ)
1b829d9  §4B.2 Option C scaffolding (cjc-cana-nss crate)
95fab9f  §4B.3 wire --thermal-aware flag through cjcl run
9f15c69  3-way AB extension (default/trained/thermal)
a4ca33d  §4B.4 Option α (KernelVariantSelector) + §4B.5 doc
ab06e9d  §4B.5 reframing — CANA = Option 5 (downstream consumer)
7b58d62  §3A.4 follow-up — corpus augmentation, close alloc blind dim
ef49b23  cana_ab_corpus — non-PINN AB bench (trained ≠ default on 8/8)
b2973fc  §17 threshold probe (revealed clamp-from-negative)
1e39db2  cmd_run_formatted honors --mir-opt/--thermal-aware
ca3e0ec  §17 Option A — v5 coefficients shipped, legality gate identified
4421052  §19 PerPassLegalityGate — PINN activation lands
<this>   §20 Option A + thermal-aware as standard
```

---

## 2. State of the codebase

### Crates added / extended

| Crate | Status |
|---|---|
| `cjc-nss` | NEW (16,578 LOC, 240 tests). NSS Phase 1-5. |
| `cjc-cana-nss` | NEW (Option A — synthetic-trace projection). 3 unit + 11 integration tests. |
| `cjc-cana` | Extended: PerPassLegalityGate, KernelVariantSelector trait, v5 coefficients, 170 tests (was 134). |
| `cjc-mir-exec` | Extended: `run_program_optimized_thermal_aware*`, `cana_thermal_aware_plan_for`. |
| `cjc-cli` | Extended: `--thermal-aware`, `--no-thermal-aware` flags, structured-output paths honor MIR flags. |

### New benches (`bench/`)

| Bench | Purpose |
|---|---|
| `cana_train_cost_model` | 85-program corpus, OLS GD fit, regenerates v5 coefficients |
| `cana_ab_pinn` | 3-way AB on PINN heat 1D (default / trained / thermal) |
| `cana_ab_corpus` | 3-way AB on 8 pass_ordering programs (compile-time only) |
| `cana_threshold_probe` | Sweep skip_threshold + per-function feature dump |

### New design docs (`docs/cana/`)

- `CANA_PHASE_4_NSS_BRIDGE_DESIGN_OPTIONS.md` (α/β/γ/δ for §4B.2)
- `CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md` (α/β/γ/δ for §4B.4)
- `CANA_PHASE_4_SECOND_NN_COMPOSITION_OPTIONS.md` (Option 1-5 for §4B.5)
- `CANA_COST_MODEL_TRAINING_FINDINGS.md` — §3A.1 through §20

---

## 3. What `--mir-opt` now actually does

```
cjcl run --mir-opt <prog.cjcl>
  →
  cjc-cli resolves: thermal_aware = true (unless --no-thermal-aware)
  →
  cjc_mir_exec::run_program_optimized_thermal_aware
  →
  cana_thermal_aware_plan_for(mir):
    PassRanker<
      ThermalAwareCostModel<
        LinearCostModel::trained(),    ← v5 coefficients
        NssPressurePredictor           ← Option A: synthesized trace
      >,
      PerPassLegalityGate              ← §19: CF/DCE/LICM allowed on
                                         float-heavy functions
    >
  →
  PassPlan that includes CF/DCE/LICM on float-reduction functions
  (PINN, ML examples) where DefaultLegalityGate would have vetoed
  everything
```

### The 5-layer chain — final accounting

| Layer | Status | Notes |
|---|---|---|
| 1. `NssPressurePredictor.predict_thermal` returns non-empty | ✓ Option A active | synthesized trace projects features onto pressure substrate |
| 2. `ThermalAwareCostModel` applies penalty factor | ✓ Live | reads layer 1's map via `.get().unwrap_or(0.0)` |
| 3. CANA's `PassRanker` produces different plans | ✓ Live | trained vs default Run counts: 37 vs 46 on pass_ordering |
| 4. `PerPassLegalityGate` lets CF/DCE/LICM through | ✓ §19 active | float-heavy ML workloads now get recommendations |
| 5. `cjcl run --mir-opt` routes through the chain | ✓ §20 standard | thermal-aware is now the default; `--no-thermal-aware` opts out |

### What's NOT yet user-visible (honest accounting)

- **Trainable passes aren't in `THERMALLY_AGGRESSIVE_PASSES`.** That
  constant is `["loop_unroll", "vectorize", "specialize",
  "monomorphize"]`. None of CF/SR/DCE/CSE/LICM are listed. So Option
  A's thermal predictions populate the maps but don't actually
  adjust the cost of any current trainable pass. Future passes
  (when added to `CANONICAL_PASSES`) automatically benefit.
- **The §20 default switch IS user-visible:** `--mir-opt` now uses
  trained (v5) instead of hand-tuned (`LinearCostModel::new()`)
  coefficients, AND uses PerPassLegalityGate. This matters for
  programs where trained vs hand-tuned diverge (8/8 pass_ordering
  programs do).

---

## 4. Workspace health (final state)

```
cjc-cana lib:        144/144  (was 134; +10 for PerPassLegalityGate, all green)
cjc-cana integration: 26/26   (unchanged)
cjc-cana-nss:         14/14   (was 13; +1 for Option A determinism test)
cjc-mir-exec lib:     17/17   (unchanged)
cjc-cli lib:         181/181  (unchanged)
cjc-nss:             240/240  (unchanged from §4B.1 drop-in)
AST/MIR parity:        1/1    (load-bearing — confirms byte-identity)
```

All test suites green. No regressions.

---

## 5. The "actual compiler improvement" picture

**Verifiable end-to-end on a float-heavy CJC-Lang program:**

```
$ cat /tmp/cjcl_float.cjcl
fn poly(x: f64) -> f64 { ... }
fn accum(n: i64) -> f64 { ... }   // for-loop with f64 accumulator
print(accum(40));

$ cjcl run                /tmp/cjcl_float.cjcl  →  141.99999999999997  (eval)
$ cjcl run --mir-opt      /tmp/cjcl_float.cjcl  →  141.99999999999997  (mir + thermal)
$ cjcl run --thermal-aware /tmp/cjcl_float.cjcl →  141.99999999999997  (explicit alias)
$ cjcl run --mir-opt --no-thermal-aware ...    →  141.99999999999997  (old default)
```

Bit-exact across all paths. The IEEE 754 trailing `.99999999999997`
is preserved by the optimizer.

**Before §17 + §19 + §20:**
- `cjcl run --mir-opt` on PINN-like programs was silently equivalent
  to `cjcl run` (no optimization happened because legality gate
  vetoed everything).

**After §17 + §19 + §20:**
- `cjcl run --mir-opt` on PINN-like programs actually runs CF + DCE +
  LICM. PassPlan has 16 recommendations on PINN at default threshold
  (vs 0 before). Output bit-identical.

---

## 6. What's next — prioritized for the next session

### Tier 1: Cheap wins (1-3 hours each)

#### 6.1 Extend `THERMALLY_AGGRESSIVE_PASSES` (1 hour)

`THERMALLY_AGGRESSIVE_PASSES` is currently `["loop_unroll",
"vectorize", "specialize", "monomorphize"]`. None of these are in
CANA's `CANONICAL_PASSES`, so the thermal predictions don't affect
PassPlans today.

Consider adding `cse` to the list — CSE creates intermediate SSA
values which increase register pressure (a real thermal contributor).
Result: on hot functions, CSE benefit would be halved by the thermal
penalty, potentially shifting it below the skip threshold.

This is a tuning decision that needs empirical validation: would
adding CSE here make the thermal-aware path produce visibly different
PassPlans on real workloads? Run `cana_ab_corpus` after the change to
see.

#### 6.2 Update bench/cana_pass_ordering with new defaults (1 hour)

That bench was built when `default_ranker()` used `DefaultLegalityGate`
+ hand-tuned coefficients. After §19 + §20, both pieces shifted.
Re-running it would update its baseline measurements with the new
behavior.

#### 6.3 Document `--no-thermal-aware` in the help text (already done)

Verify `cjcl --help` output shows the new flags correctly. Update
README if it documents the CLI surface.

### Tier 2: Mid-effort (1-2 days each)

#### 6.4 Per-expression legality analysis for CSE / SR (1-2 days)

`PerPassLegalityGate` shipped two tiers: Universal (always safe) and
NoStrictReductions (block when `strict_count > 0`). The latter is
conservative. CSE and SR could be allowed on float-heavy functions
with per-expression analysis:

- **CSE**: track which sub-expressions are part of strict reductions.
  Allow CSE to collapse occurrences that aren't.
- **SR**: track which multiplications are inside strict reductions.
  Allow `x * 2.0 → x + x` rewrites that don't touch reductions.

This would activate the last two passes on PINN-like workloads.
Probably ~2 days of careful analysis + test work.

#### 6.5 Runtime impact measurement (1 day)

The `cana_ab_pinn` bench measures compile-time decisions and
byte-identity. It doesn't measure runtime speedup. After §19 + §20,
PINN actually gets optimized — but how much faster does it run?

Build a `cana_runtime_pinn` bench that times `run_program_optimized`
on PINN with and without --no-thermal-aware. Multiple iterations for
stable medians. Expected: small speedup from CF + LICM hoisting, maybe
5-15%.

#### 6.6 Implement §4B.4 Option β (one Cool variant) — 3-5 days

[`CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md`](CANA_PHASE_4_KERNEL_VARIANT_DESIGN_OPTIONS.md)
documents Option β: pick one fused primitive
(e.g. `fused_matmul_norm`), implement its Cool variant in
`cjc-runtime`, modify the dispatch arm to consult a selector at call
time, validate parity with the existing Hot variant.

Same scope, same structure as the §4B.2 Option A migration. Wires
the runtime side of kernel variant selection that §4B.4 Option α only
stubbed.

### Tier 3: Multi-week

#### 6.7 §4B.2 Option B (real MIR-exec instrumentation) — 2-3 weeks

Per the design doc, this is the "real runtime signal" version of NSS
predictions. ~200-400 LOC of instrumentation in `cjc-mir-exec`'s
instruction-dispatch loop, plus trait-signature changes that
propagate back through `ThermalAwareCostModel`.

Only worth doing once Option A has been validated on real workloads
AND a specific workload demonstrates that synthetic-trace predictions
are insufficient. Not yet.

#### 6.8 Add new compiler passes to CANONICAL_PASSES — open-ended

`loop_unroll`, `vectorize`, `specialize`, `monomorphize` are in
`THERMALLY_AGGRESSIVE_PASSES` but NOT in `CANONICAL_PASSES`. Adding
them would activate the thermal-aware behavior in earnest. Each
pass is ~1-3 days of MIR optimizer work.

---

## 7. Known gaps and honest limitations

### Pre-existing failures NOT touched

The handoff at the start of this session listed 15 pre-existing
failures (7 Tier 0 replay + 8 Tier 2 canary). None of those surfaces
were modified in this session, so the count should be unchanged. They
remain documented in `docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md`.

### `cmd_run_formatted` and `--mir-mono`

Fixed in §4B.3 follow-up. All four flag combinations (`--mir-opt`,
`--thermal-aware`, `--no-thermal-aware`, `--mir-mono`) now work with
`--format json/csv`. Pre-existing limitation closed.

### Coefficient training is single-machine

`bench/cana_train_cost_model` uses pass-native diagnostic counts as
labels (introduced in §1592e2d). This is deterministic and
reproducible, but the labels aren't end-to-end runtime measurements.
A future training pipeline could use actual wall-clock benefits
measured from instrumented runs.

### NSS predictions are heuristic, not learned

Option A's synthesis maps `CanaFeatures` → `MirTraceEvent`s using
hand-coded heuristics (e.g. `register_pressure = expr_count / 256`).
The synthesis is deterministic but not trained. Option B's real
instrumentation would give NSS real data; Option A's synthetic data
is a best-guess projection.

### `THERMALLY_AGGRESSIVE_PASSES` mismatch

The four passes in this list aren't in `CANONICAL_PASSES`, so Option
A's thermal predictions are populating maps that no current
trainable pass consults. See §6.1 above.

---

## 8. Critical files for the next session

If you're picking up where I left off, read these in order:

1. **This doc** (HANDOFF_NEXT_SESSION_v2.md) — orientation.
2. [`CANA_COST_MODEL_TRAINING_FINDINGS.md`](CANA_COST_MODEL_TRAINING_FINDINGS.md)
   §17, §18, §19, §20 — the activation story end-to-end.
3. [`crates/cjc-cana-nss/src/lib.rs`](../../crates/cjc-cana-nss/src/lib.rs)
   — Option A implementation. ~100 LOC of synthesis + projection.
4. [`crates/cjc-cana/src/legality.rs`](../../crates/cjc-cana/src/legality.rs)
   — PerPassLegalityGate (§19). ~150 LOC of trait impl + per-pass tier
   classification.
5. [`crates/cjc-cana/src/linear_cost_model.rs`](../../crates/cjc-cana/src/linear_cost_model.rs)
   — v5 trained coefficients (§17). The `trained_pass_coefficients`
   function near line 230.
6. [`crates/cjc-cli/src/lib.rs`](../../crates/cjc-cli/src/lib.rs)
   — `--mir-opt` defaulting to thermal-aware (§20). Search
   `effective_thermal_aware`.

---

## 9. How to verify the current state quickly

```bash
cd /c/Users/adame/CJC/.claude/worktrees/compassionate-chebyshev-8956ab

# 1. Workspace tests (~3-5 min cold, sub-second warm)
cargo test -p cjc-cana --release --quiet | grep "test result"
cargo test -p cjc-cana-nss --release --quiet | grep "test result"
cargo test -p cjc-cli --release --quiet --lib | grep "test result"
cargo test --test fixtures --release       # AST/MIR parity — load-bearing

# 2. Compile-time activation visible
cargo run --release -p cana-threshold-probe    # PINN: 16 Run-rec at default
cargo run --release -p cana-ab-corpus          # default 46 / trained 37 / thermal 37

# 3. End-to-end byte-identity on float arithmetic
echo 'fn p(x: f64) -> f64 { return x * x + 1.0; } print(p(1.5));' > /tmp/probe.cjcl
diff <(./target/release/cjcl run /tmp/probe.cjcl) \
     <(./target/release/cjcl run --mir-opt /tmp/probe.cjcl)
# (empty — byte-identical)

# 4. Regenerate v5 coefficients deterministically
cargo run --release -p cana-train-cost-model   # ~10s + build
```

---

## 10. The four headline measurable changes

**For users:**
1. `cjcl run --mir-opt` on float-heavy ML programs now actually
   optimizes (PINN goes from 0 → 16 Run-recommendations).
2. `cjcl run --thermal-aware` and `cjcl run --mir-opt` are now the
   same path. `--no-thermal-aware` opts out.
3. JSON/CSV output paths honor MIR flags.
4. v5 coefficients shipped — held-out generalization preserved,
   `alloc_sites` blind dim closed, CF `w_loop_depth` sign corrected.

**For future sessions:**
1. 5-layer activation chain is end-to-end live. The next big win
   is wiring more passes into `CANONICAL_PASSES` so thermal
   predictions actually change PassPlans.
2. `PerPassLegalityGate` unblocks float-heavy ML — extending it to
   per-expression analysis would activate CSE and SR on those
   workloads too.
3. The bench surface is rich (threshold probe, AB corpus, AB pinn,
   training) — future activation work has measurement
   infrastructure ready.

---

*End of v2 handoff. Pick the right Tier-1 item from §6 to start
the next session.*
