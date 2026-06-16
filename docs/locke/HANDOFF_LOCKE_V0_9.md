# Handoff — Locke v0.9 belief-correctness + capability arc

**Date:** 2026-06-15/16 · **Branch:** `claude/happy-jennings-d673b9`
**Entry point for the next Locke session.** Companion design doc:
`docs/locke/BELIEF_AXIS_ROUTING.md`. Determinism contract (unchanged):
byte-identical `LockeReport.to_json()`, `BTreeMap`/`BTreeSet` only, no
FMA, `f64::total_cmp`, SplitMix64 IDs.

## 0. One-line status

Six Locke/cjc-data improvements shipped this arc, **+47 tests**, full
regression green, **determinism contract intact (no `SCHEMA_VERSION`
bump)**. The belief score now means what it says, two new detections
landed, a real security leak is closed, and the CSV reader is RFC-4180.

## 1. What shipped (with the commit)

| # | Change | Where | Tests |
|---|---|---|---|
| 1 | **Belief leakage/drift axes wired** — built-in E9060/61/63/64 (leakage), E9018/19/30–39/110–112 (drift) now move their axes; stale assumption text made conditional | `api.rs` | belief_routing_tests |
| 2 | **Routing registry + exhaustiveness guard** — code→axis is data (`belief_routing.rs`); a forgotten detector fails the guard | `belief_routing.rs` | guard ×4 |
| 3 | **Noisy-OR penalty** — `penalty + p − penalty·p` (byte-identical at ≤1 finding, monotone at the bad end); Info 0.02→0.01 < Notice | `belief.rs` | curve ×4 + proptest ×5 |
| 4 | **Categorical-feature leakage E9065** — Cramér's V over a contingency table; routed to leakage via the registry | `leakage.rs` | categorical_leakage_tests (+11) |
| 5 | **PII secrets-in-memory** — sample values masked at collection (`123-45-6789`→`***-**-****`) | `pii.rs` | pii_tests |
| 6 | **cjc-data RFC-4180 CSV** — whole-input quote-aware record tokenizer; fixes the LC `desc`-shatters blocker | `cjc-data/src/csv.rs` | rfc4180_tests (+13) |

Plus: a **`#[ignore]`d diabetes-130 Locke leakage audit** wired and ready
(`tests/locke/diabetes_leakage_audit.rs`) — see §4.

**Regression (all flat/green):** cjc-data 269 · cjc-locke lib 435 ·
`--test locke` 339 (8 ignored) · `--test abng` 629 · cjc-cli 181 · lendingclub-demo 12.

**Bonus the bolero fuzz earned:** a pre-existing latent CSV panic (data
row with more fields than the header indexed past the column buffers) was
surfaced and fixed (`col_types.truncate(ncols)`).

## 2. Determinism notes (for the auditor pass)

- Belief score is **derived, never serialized** into `LockeReport` JSON →
  report bytes + `run_id` unchanged → no schema bump, all snapshots green.
- Noisy-OR accumulates `mul` then `sub` separately — **no FMA**; fixed
  finding order → deterministic.
- Cramér's V: `BTreeMap` contingency, Kahan χ², `total_cmp` — deterministic.
- CSV tokenizer: single deterministic pass, no `HashMap`, no FP.

## 3. What's NOT done (pick up here, priority order)

**#4 capability remainder**
- **Extend E9065 to Int-coded low-cardinality categoricals.** Today it is
  `Str`-only; many real categoricals (incl. diabetes `discharge_disposition_id`)
  are integer codes, caught only by the numeric AUC / per-level paths. A
  low-distinct-Int branch in `detect_categorical_target_leakage` would
  close this — highest-value, cohesive next step.
- **Functional dependency** (`zip → state`): redundancy + leakage vector;
  the per-level machinery is most of the scaffolding.
- **PII Luhn (credit card) / IBAN / IPv4-6** — deterministic, zero-dep,
  fit the `looks_like_*` family.

**#5 perf/security remainder**
- **Whole-report PII scrub** — the *bigger* secrets issue the test
  surfaced: non-PII categorical detectors (E9016 rare-category, E9080…)
  still echo raw values. Needs PII-aware redaction across categorical
  detectors or a report-level scrub. (B6.12 closed only the PII detector's
  own samples.)
- **C6 keep-or-revert** — `PERF_C6_REVERT_NOTES.md`; needs a cool-machine
  A/B (bit-identical, so pure perf hygiene).
- Byte-identical perf long-tail (B6.5/B6.8–B6.11, dispatch binary search).

**#6 adoption remainder**
- **Demo B (NYC Taxi streaming/drift)** — needs the ~100M-row download
  (skipped this session per the user). The CSV reader is now ready for it.

**Registry upgrade candidates** (advisory today, in `belief_routing.rs`):
`E9072`→leakage, `E9051`/`E9052`→leakage, `E9070`→missingness. Each needs
a false-positive analysis before promotion; the guard makes it one line.

## 4. The diabetes-130 audit (how to get a real number)

The dataset is **untracked** (Kaggle `diabetes_130`, like LendingClub).
The audit is wired and `#[ignore]`d. To run it:

```text
# place diabetic_data.csv at:
tests/data/diabetes_130/diabetic_data.csv
cargo test --test locke --release diabetes130_locke_leakage_audit -- --ignored --nocapture
```

It parses the full CSV via the fixed reader, runs the leakage family
(E9063 multiclass / E9064 per-level / E9065 categorical) against
`readmitted`, splices them, and prints findings + the 8-axis belief score.
**Honest framing:** the published diabetes "AUC 0.6645" is the *ABNG
model's* predictive perf, not Locke — Locke is the validator. This audit
measures Locke's *detection*, not a model AUC.

## 5. Customization surface (for reference / docs)

Eight layers: custom detectors (ADR-0041, E9500–E9999) · ValidateOptions ·
ValidationConfig · per-detector configs (Leakage/Drift/Pii/Shape/… +
v0.9 Cramér's-V knobs) · BeliefPenalty · BeliefWeights · BeliefAxisRules ·
Policy (suppressions/requirements/gate). Plus the new belief-routing
registry as a code→axis customization point. Full table in the session
notes / BELIEF_AXIS_ROUTING.md.

## 6. Test discipline (unchanged contract)

wiring → unit → proptest → bolero → regression loop → docs. The loop:
```
cargo test -p cjc-data --release                       # 269 (CSV + RFC-4180)
cargo test -p cjc-locke --lib --release                # 435
cargo test --test locke --release                      # 339 passed / 8 ignored
cargo test --test abng --release                       # 629 (determinism canaries)
cargo test -p cjc-cli --lib --release                  # 181
cargo test -p lendingclub-demo --release               # 12
```
