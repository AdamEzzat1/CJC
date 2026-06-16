# Locke v0.9 — the belief-correctness arc

**Date:** 2026-06-15
**Scope:** make the 8-axis `BeliefScore` *mean what it says*. Three
coupled changes in `belief.rs` / `api.rs` / the new `belief_routing.rs`,
plus the test suite that locks them. Produced by the stacked-role group
(Architect / Pipeline / Numerical / Determinism-Auditor / QA).
**Determinism gate:** `docs/cana/DETERMINISM_CONTRACT.md` + Locke's own
contract (byte-identical `LockeReport.to_json()`, `BTreeMap` everywhere,
no FMA, `f64::total_cmp`). **Both hold — no `SCHEMA_VERSION` bump.**

## 0. The bug class (why this was the highest-leverage fix)

The silent-failures audit (`SILENT_FAILURES_AUDIT.md`) was created because
a 96.9%-missing column scored `missingness_score = 1.0` ("perfect"). The
same class of bug was still live on the **leakage** axis:

> `validate()` → splice `detect_target_leakage` findings into the report
> (exactly what `demos/lendingclub/src/lib.rs::run_locke_audit` does) →
> `belief_report_from_locke`. A report could carry an **E9060** finding
> (`|AUC| ≥ 0.95`, **Error** — near-deterministic leakage) and still
> report `leakage_score = 1.0`, with the assumption text claiming *"Locke
> v0 does not infer leakage automatically"* — which contradicted the
> E9060 sitting in the same `findings` list.

Root cause (verified against the code, not the research summary): the
per-axis penalty closures in `api::belief_axis_scores_from_report` wired
**zero built-in codes** to the drift / leakage / lineage axes — only
*custom-detector* findings could move them. The detector layer produced
leakage findings the belief layer then ignored.

## 1. The three changes

### #1 — Wire the built-in codes into the axes (the headline fix)
The drift and leakage axes now consume their detector findings:
- **leakage** ← E9060, E9061, E9063, E9064 (the AUC + per-level family).
  E9062 ("target not binary — skipped") stays advisory: it is a
  diagnostic, not leakage evidence.
- **drift** ← E9018, E9019, E9030–E9039 (numeric/categorical drift) and
  E9110–E9112 (text drift).

The stale assumption text is now **conditional**: the "axis = 1.0 by
default" caveat is emitted only when the report genuinely contains no
finding routed to that axis. Once a leakage/drift finding is present the
score reflects it and the caveat is suppressed, so the assumptions can no
longer contradict the findings.

### #2 — A routing registry with an exhaustiveness guard
The hand-maintained `||` chains became **data** in `belief_routing.rs`:
`builtin_axis_for(code) -> BeliefAxisSet`, plus `ALL_BUILTIN_CODES`
(every emitted code, sorted) and `ADVISORY_CODES` (codes deliberately
routed to no axis). The api now derives every axis penalty through one
helper: `builtin_axis_for(code).contains(axis) || custom_contains(code, axis)`.

The guard (`tests/locke/belief_routing_tests.rs::every_builtin_code_is_routed_or_advisory`)
asserts **every** code in `ALL_BUILTIN_CODES` is *either* routed *xor*
advisory. A future detector whose code is added to the registry but
forgotten in both places fails the guard — it can no longer silently gate
CI without ever moving a belief score. This guard would itself have caught
bug #1.

### #3 — Noisy-OR penalty curve + Info < Notice
The old aggregation was `Σ pᵢ` clamped at 1.0, which **saturated**: 4
Errors (4 × 0.25) or 10 Warnings drove an axis to exactly `0.0`, and every
worse dataset then scored identically. Replaced with **incremental
noisy-OR**:

```
penaltyₖ = penaltyₖ₋₁ + pₖ − penaltyₖ₋₁·pₖ        ( = 1 − Π(1 − pᵢ) )
axis_score = 1 − penalty                          ( = Π(1 − pᵢ) )
```

Each finding is an independent "defect event" with probability `pᵢ`; the
axis score is the probability it survives all of them. Properties:
- **Discrimination at the bad end** — strictly monotone in count: 40
  findings score strictly below 4 (the old model tied them at 0).
- **Byte-identical at ≤ 1 finding** — `0 + p − 0·p = p` exactly, so
  single-finding axes do not move (the incremental form is chosen over
  `1 − Π(1−pᵢ)` precisely to avoid the `1.0 − 0.99 ≠ 0.01` double-rounding).
- **Bounded by construction**, no reliance on a hard clamp.

`BeliefPenalty::default()` Info dropped `0.02 → 0.01` so it is strictly
below Notice, restoring the severity rank the rest of the system keeps.

## 2. Determinism / accuracy verification (the auditor pass)

- **`LockeReport` JSON is unchanged.** The belief score is *derived*
  (`belief_report_from_locke`), never serialized into the report (grep
  `json_emit.rs` for `belief`/`score` → none). So `to_json()` bytes, the
  content-addressed `run_id`, and every snapshot/round-trip test are
  byte-identical. **No `SCHEMA_VERSION` bump needed.**
- **No FP-order / FMA / RNG / iteration-order change.** The noisy-OR
  accumulates `penalty * p` then subtracts as **separate** ops (no
  `mul_add`), in the already-pinned finding order. Determinism invariants
  1–6 untouched; `BTreeMap` routing, `total_cmp` unaffected.
- **Behaviour change is contained and intentional.** For every existing
  test over a single-DF `validate()` report, the leakage/drift axes have
  no findings → still 1.0 → unchanged. The new scores appear only when
  leakage/drift findings are present (the demo splice pattern, or a
  compare). Single-finding axes are byte-identical; multi-finding axes
  shift within the loose thresholds existing tests use.

## 3. Test coverage (QA pass) — all in `tests/locke/`

`tests/locke/belief_routing_tests.rs` (**+22**, integration target
305 → 327):
- **Unit / integration (14):** the exhaustiveness guard (4: routed-xor-
  advisory, advisory⊆all, sorted+unique, leakage/drift-actually-routed),
  the headline fix (spliced E9060 → leakage < 1.0; end-to-end via
  `detect_target_leakage`; drift; caveat suppression/presence), and the
  curve (single-finding bit-identity, bad-end ordering = 0.75⁴,
  Info < Notice, unit-interval, determinism).
- **Property (5):** penalty ∈ [0,1]; monotone in count; single-finding ==
  linear bit-for-bit; deterministic; leakage axis never perfect with an
  E9060 present.
- **Bolero fuzz (3):** penalty bounded+finite on arbitrary severity
  multisets; `builtin_axis_for` total (never panics on any `&str`);
  arbitrary spliced findings keep all 8 axes in [0,1].

`crates/cjc-locke/src/belief.rs` lib test updated:
`default_penalty_v09_values_with_info_below_notice` (was the v0.2-values
pin) — now asserts the strictly-increasing severity ladder.

**Regression gate:** cjc-locke lib 435, `--test locke` 327 (7 ignored),
`--test abng` canaries flat, cjc-cli, LendingClub demo. (See the commit
for the exact run.)

## 4. Follow-on increment (themes #4 + #5, partial — shipped same session)

After the belief-correctness core, two contained, fully-tested pieces of
the larger list landed:

### #4 — Categorical-feature target leakage (E9065, Cramér's V)
`leakage::detect_categorical_target_leakage` fills the gap between the
numeric AUC path (never sees a `Str` column) and E9072 (only ID-like
cardinality): a *low-to-moderate-cardinality* categorical column that
nearly determines the target (e.g. `discharge_code = 11 ⇒ readmitted=no`).
Cramér's V = `sqrt(χ²/(n·min(r−1,c−1)))` over a deterministic contingency
table (`BTreeMap` counts, Kahan χ², no FMA). Error at V ≥ 0.9, Warning at
≥ 0.7; guards skip the target, non-`Str`, constant, high-cardinality
(> `categorical_max_distinct`, → E9072/E9017), and sparse (< 2·min_class)
cases. **E9065 routes to the leakage axis via the registry — a one-line,
guard-checked addition, exactly the payoff #2 was built for.** Tests:
`tests/locke/categorical_leakage_tests.rs` (+11: 9 unit incl. perfect/
independent/high-card/constant/float/multiclass + axis-feed + determinism,
1 proptest, 1 bolero).

### #5 — Secrets-in-memory: redact PII samples (B6.12)
`pii::detect_all_pii` stored the **raw** matched value (incl. SSNs) in its
sample maps → into the finding evidence → into the report. Now masked at
collection by `redact_pii_sample` (ASCII-alphanumeric → `*`, structure +
length kept: `123-45-6789` → `***-**-****`), so the *shape* stays
recognisable while the secret never enters the report. Distinct secrets
collapse to one masked pattern — the right behaviour for a PII finding.
Test: `pii_tests::pii_samples_are_redacted_no_raw_secret_reaches_the_report`.

**Honest scope limit (surfaced by that test):** the PII-sample redaction
is *necessary but not sufficient* for a whole-report "no secrets"
guarantee. Non-PII categorical detectors still echo raw values — e.g.
**E9016 rare-category** names a once-seen value, which for a PII column is
the secret. Closing that needs PII-aware redaction across the categorical
detectors (or a report-level scrub) and is a larger follow-up than B6.12;
the test is therefore scoped to the PII detector's own evidence (what this
fix actually closes) and this gap is recorded here rather than papered over.

## 5. Still not done (honestly scoped)

- **#4 remainder** — functional-dependency (`zip → state`), multi-class
  leakage beyond binary + the `multiclass_max_classes = 20` silent skip,
  PII Luhn/IBAN/IP patterns. Each its own tested detector.
- **#5 remainder** — the **whole-report PII scrub** above; the C6
  single-pass helper still needs a cool-machine A/B before keep-or-revert
  (`PERF_C6_REVERT_NOTES.md`); the byte-identical perf long-tail.
- **#6 adoption** — Demo B (NYC Taxi streaming/drift) needs a ~100M-row
  external download (not feasible in this environment). **The upstream
  RFC-4180 CSV-quoting fix is DONE** (`cjc-data/src/csv.rs`): a whole-input
  record tokenizer now honors quoted fields containing the delimiter,
  newlines, and `""` escapes — so a free-text column with commas (LC's
  `desc`) no longer shatters and shifts the joint-application columns (the
  ADR-0042 #1 blocker). +13 tests (8 unit, 2 proptest, 1 bolero) and the
  bolero pass **surfaced + fixed a pre-existing latent panic** (a data row
  with more fields than the header indexed past the column buffers). The
  streaming aggregators are quote-aware per-row; embedded newlines in
  quoted fields remain a documented streaming-only limit.

### Documented upgrade candidates for the registry (deliberate advisory today)
`E9072` (ID-like cardinality) → leakage: a real ID-leakage hint, but
fires on every legitimate primary key, so routing it would penalise clean
datasets. `E9051`/`E9052` (temporal overlap / future timestamp) → leakage.
`E9070` (conditional missingness) → missingness. Each needs its own
false-positive analysis before promotion; the registry makes adding them a
one-line, guard-checked change.
