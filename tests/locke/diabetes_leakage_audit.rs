//! `#[ignore]`d diabetes-130 Locke leakage audit — wired and ready.
//!
//! The diabetes-130 CSV is untracked (like LendingClub). The moment it is
//! present at `tests/data/diabetes_130/diabetic_data.csv`, running
//!
//! ```text
//! cargo test --test locke --release diabetes130_locke_leakage_audit -- --ignored --nocapture
//! ```
//!
//! parses the full dataset through cjc-data's (now RFC-4180-correct) CSV
//! reader, runs Locke's leakage detector family against `readmitted`, and
//! prints the findings + the 8-axis belief score — a real number on real
//! data, not a synthetic fixture.
//!
//! `discharge_disposition_id` (codes 11/13/14 ⇒ death/hospice ⇒
//! `readmitted = NO` by construction) is the canonical leakage column;
//! string categoricals such as `medical_specialty` are where the v0.9
//! E9065 Cramér's-V detector adds reach beyond the numeric AUC path.

use std::path::Path;

use cjc_data::{CsvConfig, CsvReader};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::leakage::{
    detect_categorical_target_leakage, detect_per_level_target_leakage,
    detect_target_leakage_multiclass, LeakageConfig, PerLevelLeakageConfig,
};

const DATASET_REL_PATH: &str = "tests/data/diabetes_130/diabetic_data.csv";
const TARGET: &str = "readmitted";

#[test]
#[ignore = "needs untracked tests/data/diabetes_130/diabetic_data.csv; run with --ignored"]
fn diabetes130_locke_leakage_audit() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATASET_REL_PATH);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!(
            "[skip] diabetes130_locke_leakage_audit: {} absent",
            path.display()
        );
        return;
    };

    let df = CsvReader::new(CsvConfig::default())
        .parse(&bytes)
        .expect("diabetes-130 CSV must parse");
    eprintln!("diabetes-130: {} rows x {} cols", df.nrows(), df.ncols());
    assert!(df.nrows() > 0 && df.ncols() > 0, "non-empty frame expected");
    assert!(
        df.get_column(TARGET).is_some(),
        "expected a `{TARGET}` column in the diabetes-130 CSV"
    );

    // Baseline single-DF report (schema / missingness / duplication / …).
    let mut report = validate(
        &df,
        &ValidateOptions {
            dataset_label: "diabetes130".into(),
            ..Default::default()
        },
    );
    let leakage_before = belief_report_from_locke(&report).score.leakage_score;
    eprintln!("leakage_score (pre-splice) = {leakage_before:.4}");

    // The leakage detector family, spliced exactly as the LendingClub
    // demo's `run_locke_audit` does. `readmitted` is 3-class, so the
    // multiclass one-vs-rest AUC (E9063), per-level deterministic-outcome
    // (E9064), and categorical-association Cramér's V (E9065) detectors
    // apply; the binary AUC path correctly skips (E9062).
    let mut leak = Vec::new();
    leak.extend(detect_target_leakage_multiclass(
        &df,
        TARGET,
        &LeakageConfig::default(),
    ));
    leak.extend(detect_per_level_target_leakage(
        &df,
        TARGET,
        &PerLevelLeakageConfig::default(),
    ));
    leak.extend(detect_categorical_target_leakage(
        &df,
        TARGET,
        &LeakageConfig::default(),
    ));

    eprintln!("leakage findings: {}", leak.len());
    for f in &leak {
        eprintln!(
            "  {} [{:?}] column={}",
            f.code,
            f.severity,
            f.column.as_deref().unwrap_or("-")
        );
    }

    report.findings.extend(leak.iter().cloned());
    let belief = belief_report_from_locke(&report);
    eprintln!(
        "belief: overall={:.4} schema={:.4} missingness={:.4} drift={:.4} leakage={:.4} \
         sample={:.4} duplication={:.4} constraint={:.4}",
        belief.score.overall,
        belief.score.schema_score,
        belief.score.missingness_score,
        belief.score.drift_score,
        belief.score.leakage_score,
        belief.score.sample_score,
        belief.score.duplication_score,
        belief.score.constraint_score,
    );

    // Structural invariants (always hold on real data).
    for axis in [
        belief.score.overall,
        belief.score.leakage_score,
        belief.score.missingness_score,
    ] {
        assert!(axis.is_finite() && (0.0..=1.0).contains(&axis), "axis out of range: {axis}");
    }
    // The v0.9 fix in action: once any leakage finding is present, the
    // leakage axis must reflect it (pre-v0.9 it stayed a false 1.0).
    if !leak.is_empty() {
        assert!(
            belief.score.leakage_score < 1.0,
            "leakage findings present but leakage_score stayed 1.0"
        );
    }
}
