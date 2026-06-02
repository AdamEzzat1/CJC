# ADR-0041 Locke v0.8 — Custom Detector Extension Layer

- **Status:** Accepted (2026-06-01)
- **Crate:** `cjc-locke` (extends ADRs 0028–0040); `cjc-locke-py` (Python bridge)
- **Companion docs:** [[Locke Roadmap]] §v0.8, [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0036 Locke v0.7 — Per-axis BeliefScore Composition Algebra]]
- **Demo proof:** `demos/lendingclub/` — same audit run with a 30-line custom detector closes the LC honest-model AUC gap from `0.9995 → 0.7388`, within 0.0006 of the hand-curated `0.7394` baseline.

## Context

The LendingClub demo (`demos/lendingclub/`, shipped 2026-06-01) made the gap concrete. Locke's built-in `detect_target_leakage` (E9061, |AUC| ≥ 0.85) flagged 3 columns. Refitting a logistic regression with only those 3 columns excluded left the model's |AUC| at **0.9995** — practically unchanged from the naive baseline. Only after excluding the *domain-curated* 12-column post-origination list did AUC collapse to **0.7394**, within the Bao et al. (2019) credit-risk literature band.

The conclusion the demo forced: **Locke's |AUC| ≥ 0.85 heuristic catches the strongest leakage but not all leakage.** The analyst's domain knowledge — "any column starting with `total_*` is post-origination, regardless of AUC" — must be applied somewhere. Two prior options were unsatisfactory:

1. **Run the domain check next to Locke**, in user code. The check's findings live in a separate code path. They don't compose with Locke's belief algebra, don't get fingerprinted, don't show up in the canonical JSON, don't get caught by Locke's regression gates. The analyst's experience accumulates as bespoke code per project.
2. **Fork Locke** and add the detector inline. Loses upgrade path; doubles maintenance burden.

Neither option matches the value proposition that Locke is selling. The demo's analysis (cross_validate.md §3.3) named the right framing: **"Locke shows you where to look; the analyst still does domain triage."** ADR-0041 makes that triage a first-class part of Locke.

Inspirations: Great Expectations (declarative test framework for data), Pandera (schema validation), `pytest` plugins. None match Locke's determinism contract or belief-axis algebra; the API here is intentionally narrower than GE because it preserves both.

## Decisions

### 1. New module `crates/cjc-locke/src/custom_detector.rs` (~540 LOC)

Three pieces:

```rust
pub struct BeliefAxisSet(pub u16);  // bitset over 8 belief axes
pub struct FindingSink { /* code, axes, pending findings */ }
pub trait CustomDetector: Send + Sync + Debug {
    fn code(&self) -> &'static str;          // E9500..=E9999
    fn belief_axes(&self) -> BeliefAxisSet;  // axes this detector affects
    fn name(&self) -> &str { self.code() }
    fn run(&self, df: &DataFrame, sink: &mut FindingSink);
}
```

The sink exposes one mutation method, `emit(severity, message, column, row_range, evidence, sample_size)`. All canonicalization happens inside `emit` — the detector cannot construct findings directly, which is what makes the determinism contract *structural* (you can't write a non-deterministic detector that produces non-deterministic output, even if you try).

### 2. Code namespace: E9500..=E9999 (500 slots)

Locke's built-in codes occupy E9001..=E9112. The custom range starts safely above them and leaves room for built-in growth. Codes outside the range are rejected at registration with a clear `CustomDetectorError::CodeOutOfNamespace`.

The validator `validate_custom_code(&str) -> Result<u32, CustomDetectorError>` is the single point of truth. The Python bridge calls it at detector-registration time; the Rust runtime calls it again inside `run_custom_detectors` as a defense-in-depth check.

### 3. Belief-axis declaration

A detector declares which axes its findings affect via `belief_axes()`. The framework records this mapping in `LockeReport.custom_axis_assignments: BTreeMap<String, BeliefAxisSet>` and the belief composition consults it:

```rust
let custom_contains = |code, axis| -> bool {
    report.custom_axis_assignments
        .get(code)
        .map(|axes| axes.contains(axis))
        .unwrap_or(false)
};

let leakage_score = 1.0 - penalty_from_findings_with_model(
    &report.findings,
    |code| custom_contains(code, BeliefAxisSet::LEAKAGE),
    penalty,
);
```

`BeliefAxisSet::NONE` is the "advisory" declaration: the detector produces findings that appear in the report but do not affect any belief score. To enforce this, the sink rejects non-`Info` severities when the detector declared no axes — a Warning or Error finding that affects no score would be a self-contradiction.

### 4. Determinism guarantees

Three structural properties:

- **Detector order is canonical.** `run_custom_detectors` sorts the detector slice by `code` before invocation. Registration order does not affect output.
- **Emission order inside `run()` is irrelevant.** The framework sorts the sink's emitted findings by `sort_key()` after `run()` returns. The Python proptest `test_determinism_under_emission_shuffle` exercises this: a detector emits findings in a `random.Random`-shuffled order each call, and the resulting JSON is byte-identical.
- **Duplicate codes are dropped after the first.** If two detectors register the same code, only the first runs; the second is recorded in `registration_errors` and the user sees an assumption surfacing the rejection.

What we explicitly do *not* prevent:

- A detector that uses `Instant::now()` or `Math.random()` to gate emission. The user can do this, but the report bytes will not be deterministic — Locke surfaces a warning in `assumptions`; it does not block the run. The contract is **structural**, not **behavioural**.

### 5. New `ValidateOptions::custom_detectors`

```rust
pub struct ValidateOptions {
    // ... existing fields ...
    pub custom_detectors: Vec<Arc<dyn CustomDetector>>,
}
```

Default empty so existing callers are byte-identical to pre-v0.8. The integration point in `api::validate` is one new block:

```rust
let custom_outcome = crate::custom_detector::run_custom_detectors(df, &opts.custom_detectors);
findings.extend(custom_outcome.findings.iter().cloned());
// ... then the existing sort merges built-ins and customs canonically.
report.custom_axis_assignments = custom_outcome.axis_assignments;
```

### 6. Python bridge: `cjc_locke.CustomDetector` ABC + `validate(custom_detectors=[...])`

Two pieces on the Rust side (`python/src/lib.rs`):

- `PyDetectorDataFrame` (`#[pyclass]`) — read-only view of the DataFrame for Python consumption. Methods: `n_rows`, `n_cols`, `column_names()`, `column_type(name)`, `get_float(name)`, `get_str(name)`. Holds an `Arc<DataFrame>` cloned once per detector invocation so the view can outlive the Rust call without lifetime gymnastics.
- `PyDetectorSink` (`#[pyclass]`) — exposes one method `emit(severity, message, column=None, row_range=None, sample_size=0)`. Internally collects pending emissions; after Python returns, the Rust adapter drains them into the real `FindingSink`.

`PyCustomDetectorAdapter` is a Rust struct that holds the Python instance and implements the Rust `CustomDetector` trait. Its `run()` method enters the GIL, builds the view + sink, calls Python's `run(df, sink)`, then drains. Exceptions in the Python detector are logged and swallowed — one detector's bug does not abort the audit.

Python facade (`cjc_locke/__init__.py`):

```python
class CustomDetector:
    def code(self) -> str: ...
    def belief_axes(self) -> list[str]: ...
    def run(self, df, sink) -> None: ...

def validate(data, label="dataset", options=None, custom_detectors=None):
    detectors = list(custom_detectors) if custom_detectors is not None else None
    return _cjc_locke.validate_dataframe(_ensure_dict(data), label, options, detectors)
```

Axis names cross the Python boundary as strings (`"leakage"`, `"schema"`, etc.) and are validated via `BeliefAxisSet::from_names(&[..])`. Unknown names raise `ValueError` at registration.

### 7. Tests shipped

| File                                                       | Tests | Kind                              |
| ---------------------------------------------------------- | ----- | --------------------------------- |
| `crates/cjc-locke/src/custom_detector.rs` (in-module)      | 15    | Unit                              |
| `tests/locke/custom_detector_tests.rs`                     | 14    | Integration (8) + proptest (3) + bolero fuzz (3) |
| `python/tests/test_custom_detectors.py`                    | 14    | Pytest end-to-end                 |
| `demos/lendingclub/src/lib.rs` (in-module)                 | 2     | Unit: `PostOriginationByName` detector |

**Total: 45 new tests** (29 Rust + 14 Python + 2 demo). Determinism is exercised by 3 dedicated tests (Rust proptest, Python emission-shuffle, demo regression-gate hash). Workspace stayed at 277/277 passing Locke integration tests (no regressions).

### 8. Demo proof: LendingClub before/after

The LC demo now ships with:
- `demos/lendingclub/src/lib.rs::PostOriginationByNameDetector` — ~30 LOC custom detector that flags any LC column whose name matches `total_*`, `last_pymnt_*`, `last_fico_range_*`, `out_prncp*`, `recoveries`, `collection_recovery_fee`, `hardship_*`, `settlement_*`, `debt_settlement_*`.
- `demos/lendingclub/src/main.rs --use-custom-detectors` flag that registers the detector and emits the augmented report.
- `demos/lendingclub/src/bin/honest_model.rs --from-report <path>` flag that extracts the E9500+E9061 flagged columns from the report and uses them as the exclusion set for a 4th honest-model variant.

**Measured 2026-06-01 (200K-row sample, seed=42, test_fraction=0.3):**

| Variant            | Exclusion set                                              | p  | Test \|AUC\| | IRLS iter |
| ------------------ | ---------------------------------------------------------- | -- | ------------ | --------- |
| Pre-Locke (naive)  | `target_default`, `id`, `member_id`                        | 87 | 0.9993       | 100 (cap) |
| Locke-filtered     | naive + 3 E9061 columns                                    | 84 | 0.9995       | 100 (cap) |
| **Locke + custom det.** | **naive + 39 E9500∪E9061 columns**                    | **68** | **0.7388**   | **17**    |
| Domain-honest      | naive + handoff §3.3 12 post-origination cols              | 75 | 0.7394       | 14        |

The "Locke + custom det." variant lands within 0.0006 of the hand-curated domain-honest baseline, using a 39-column flag set derived inside Locke. The analyst's domain knowledge — previously kept in a separate Python exclusion list — is now composable, deterministic Locke configuration.

The "Locke + custom det." set is more aggressive (39 vs 12 cols) but achieves essentially the same AUC, suggesting the extra exclusions didn't carry useful signal. The custom-detector layer captured the multicollinear post-origination cluster cleanly: IRLS convergence went from 100-iter cap (multicollinear) to 17 iters (well-conditioned).

## Trade-offs explicitly accepted

- **Python detectors can be non-deterministic in behavior.** We can't prevent a user from calling `random.shuffle` or `time.time()` inside `run()`. We *can* and *do* prevent that from affecting the report bytes (via canonical sort after emission). The contract is structural.
- **`PyDetectorDataFrame` clones the DataFrame.** O(data) per Python detector invocation. Acceptable for the demo case (column-name-pattern detection) where data is never read. Detectors that need bulk data access pay the clone cost; this can be optimized in a future ADR if real workloads demand it.
- **`Box::leak` on the code string at registration.** Custom detector code strings are heap-leaked so the `&'static str` trait return works with `ValidationFinding`'s existing field type. Detectors are typically registered once per process lifetime; the leak is bounded.
- **E9060 (built-in error threshold at 0.95) is not changed in this ADR.** The LC demo identified the threshold as potentially too conservative, but changing it requires more datasets to validate. Tracked as a separate v0.8.1 follow-up. The custom-detector layer ships in v0.8 because it solves the problem the threshold change would only partially solve.

## Open questions deferred

- **Concurrent execution of custom detectors.** The trait is `Send + Sync` so `Arc<dyn CustomDetector>` is safe across threads, but the framework currently runs detectors sequentially. Parallelization is a future ADR — the determinism contract would need to be restated under parallel execution (the canonical sort makes it work but the inputs to the sort are nondeterministically interleaved with built-in finding production).
- **YAML/TOML rule DSL for declarative custom detectors.** A common pattern is "if column matches pattern, emit finding". A small declarative DSL on top of the trait would lower the barrier. Deferred — the trait API is the foundation; a DSL is a layer.
- **JSON emit of `custom_axis_assignments`.** Currently in-memory only, not in the canonical report JSON. Adding it would change report bytes, which would break the determinism-hash regression tests across versions. Will be added in v0.9 with a schema_version bump.

## Vault updates

- New ADR-0041 (this file).
- [[Locke Roadmap]] v0.8 row marked shipped 2026-06-01.
- [[ADR Index]] entry.
- [[LendingClub Demo Notes]] updated with the 4-way AUC result.

## File inventory

```
crates/cjc-locke/src/custom_detector.rs                  +540
crates/cjc-locke/src/api.rs                              ~30  (integration)
crates/cjc-locke/src/report.rs                           +12  (custom_axis_assignments field)
crates/cjc-locke/src/lib.rs                              +1   (module declaration)
crates/cjc-locke/src/column_summary.rs                   +1   (test fixture update)
tests/locke/custom_detector_tests.rs                     +330 (14 tests)
tests/locke/mod.rs                                       +1   (module declaration)
python/src/lib.rs                                        +260 (PyDetectorDataFrame + PyDetectorSink + PyCustomDetectorAdapter)
python/cjc_locke/__init__.py                             +100 (CustomDetector ABC + validate() extension)
python/tests/test_custom_detectors.py                    +320 (14 tests)
demos/lendingclub/src/lib.rs                             +130 (PostOriginationByNameDetector + 2 unit tests)
demos/lendingclub/src/main.rs                            +35  (--use-custom-detectors flag)
demos/lendingclub/src/bin/honest_model.rs                +100 (--from-report flag + JSON walker)
demos/lendingclub/cross_validate.md                      ~80  (4-way table + closed-gap story)
demos/lendingclub/README.md                              ~25  (new flag + custom detector reference)
```

Total: ~1900 LOC (incl. tests and docs).
