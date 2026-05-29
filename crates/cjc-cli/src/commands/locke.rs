//! `cjcl locke` — evidence-aware analytics on tabular data.
//!
//! Subcommands:
//!
//! * `cjcl locke validate <data.csv>`  — run all v0 validators
//! * `cjcl locke drift <train.csv> <test.csv>` — induction-risk report
//! * `cjcl locke belief <data.csv>`    — emit a `BeliefReport` summary
//! * `cjcl locke lineage <data.csv>`   — emit a minimal lineage graph
//! * `cjcl locke causal <data.csv>`    — emit a causal-guardrail report
//! * `cjcl locke trace-value <data.csv> <column> <value>` — per-value
//!   lineage chain showing the canonicalisation stages that would apply
//!   to a single distinct value (v0.7+ A2).
//! * `cjcl locke policy apply <data.csv> --policy <.cjcl-locke.toml>` —
//!   apply a governance policy (suppressions + owner annotations +
//!   required-finding checks) to the validation output (v0.7+ A3).
//! * `cjcl locke gate <ref.json> <current> [--policy <.cjcl-locke.toml>]` —
//!   gate diff also honours required-finding policy when supplied (v0.7+ A3).
//!
//! All output is deterministic. The default format is a stable, indented
//! text emit suitable for snapshot tests; `--json` swaps in newline-
//! separated JSON-ish records (no `serde_json` dependency — Locke is
//! zero-dep by design).

use std::fs;
use std::path::PathBuf;
use std::process;

use cjc_data::{CsvConfig, CsvReader, DataFrame};
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::belief::{sample_score_from_n, BeliefScore};
use cjc_locke::causal::{CausalConfig, CausalMode};
use cjc_locke::drift::DriftConfig;
use cjc_locke::lineage::emit_lineage_text;
use cjc_locke::validation::ValidationConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockeFormat {
    Text,
    Json,
}

#[derive(Debug, Clone)]
pub enum LockeSubcommand {
    Validate {
        data: PathBuf,
        label: Option<String>,
        save_json: Option<PathBuf>,
        save_html: Option<PathBuf>,
        // v0.5 extensions:
        time_col: Option<String>,
        max_timestamp: Option<i64>,
        gap_threshold: Option<i64>,
        target: Option<String>,
        primary_key: Option<String>,
    },
    Drift { train: PathBuf, test: PathBuf },
    Belief { data: PathBuf },
    Lineage { data: PathBuf, label: Option<String>, mermaid: bool },
    Causal { data: PathBuf, target: Option<String>, observational_only: bool },
    Gate { reference: PathBuf, current: PathBuf, policy: Option<PathBuf> },
    /// v0.6: run `validate` N times and assert byte-identical reports.
    /// Exits 0 if all runs match, 3 if any divergence.
    Verify { data: PathBuf, runs: u32 },
    /// v0.7+ A2: per-value lineage trace for a single `(column, value)`
    /// pair. Emits the canonicalisation chain that would apply if the
    /// user adopted Locke's suggested normalisations.
    TraceValue { data: PathBuf, column: String, value: String },
    /// v0.7+ A3: apply a governance policy to a freshly-validated
    /// dataset and emit a `PolicyResult`. Exits non-zero if any
    /// required-finding rule fails.
    PolicyApply { data: PathBuf, policy: PathBuf },
}

#[derive(Debug, Clone)]
pub struct LockeArgs {
    pub subcommand: LockeSubcommand,
    pub format: LockeFormat,
    pub fail_on_severity: Option<String>,
}

fn dispatch(args: LockeArgs) -> i32 {
    let res = match args.subcommand.clone() {
        LockeSubcommand::Validate {
            data,
            label,
            save_json,
            save_html,
            time_col,
            max_timestamp,
            gap_threshold,
            target,
            primary_key,
        } => cmd_validate(
            &data,
            label.as_deref(),
            save_json.as_deref(),
            save_html.as_deref(),
            ValidateExtensions {
                time_col,
                max_timestamp,
                gap_threshold,
                target,
                primary_key,
            },
            args.format,
        ),
        LockeSubcommand::Drift { train, test } => cmd_drift(&train, &test, args.format),
        LockeSubcommand::Belief { data } => cmd_belief(&data, args.format),
        LockeSubcommand::Lineage { data, label, mermaid } => {
            cmd_lineage(&data, label.as_deref(), args.format, mermaid)
        }
        LockeSubcommand::Causal { data, target, observational_only } => {
            cmd_causal(&data, target.as_deref(), observational_only, args.format)
        }
        LockeSubcommand::Gate { reference, current, policy } => cmd_gate(
            &reference,
            &current,
            args.fail_on_severity.as_deref(),
            policy.as_deref(),
        ),
        LockeSubcommand::Verify { data, runs } => cmd_verify(&data, runs),
        LockeSubcommand::TraceValue { data, column, value } => {
            cmd_trace_value(&data, &column, &value)
        }
        LockeSubcommand::PolicyApply { data, policy } => {
            cmd_policy_apply(&data, &policy)
        }
    };
    match res {
        Ok(report) => {
            print!("{}", report.text);
            // Severity gate: caller can ask Locke to exit non-zero on warnings.
            let exit = if let Some(sev) = &args.fail_on_severity {
                worst_at_least(&report.worst, sev) as i32
            } else {
                0
            };
            exit
        }
        Err(e) => {
            eprintln!("locke error: {}", e);
            2
        }
    }
}

fn worst_at_least(actual: &str, threshold: &str) -> bool {
    let rank = |s: &str| match s {
        "info" => 0,
        "notice" => 1,
        "warning" => 2,
        "error" => 3,
        _ => 0,
    };
    rank(actual) >= rank(threshold)
}

struct CmdOutput {
    text: String,
    worst: String,
}

/// v0.3: read any supported format (CSV/TSV/JSONL) via the
/// `crate::formats` adapters, then convert to a `cjc_data::DataFrame`.
fn read_csv(path: &PathBuf) -> Result<DataFrame, String> {
    use crate::formats::{detect_format, load_tabular, DataFormat};
    let fmt = detect_format(path);
    match fmt {
        DataFormat::Csv | DataFormat::Tsv => {
            // Use the existing CsvReader for CSV/TSV — type-infers, faster.
            let bytes = fs::read(path)
                .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            let mut cfg = CsvConfig::default();
            if fmt == DataFormat::Tsv {
                cfg.delimiter = b'\t';
            }
            CsvReader::new(cfg)
                .parse(&bytes)
                .map_err(|e| format!("CSV/TSV parse error in {}: {:?}", path.display(), e))
        }
        DataFormat::Jsonl => {
            // Route through the JSONL loader, then promote columns to
            // cjc_data::Column with type inference per column.
            let tabular = load_tabular(path, None, true)?;
            tabular_to_dataframe(tabular)
        }
        DataFormat::Parquet => {
            // v0.4: improved diagnostic — inspect file structure and
            // surface either "not Parquet" or "Parquet but unsupported."
            match cjc_locke::parquet_reader::inspect_parquet_file(path) {
                Err(e) => Err(format!("{}: {}", path.display(), e)),
                Ok(_) => unreachable!("v0.4 inspect_parquet_file never returns Ok"),
            }
        }
        DataFormat::ArrowIpc => Err(format!(
            "{}: Arrow IPC support is metadata-only in v0.3; convert to CSV/JSONL first or wait for v0.4 Arrow decoder",
            path.display()
        )),
        DataFormat::Sqlite => Err(format!(
            "{}: SQLite support is metadata-only; export a CSV first",
            path.display()
        )),
        DataFormat::Unknown => Err(format!(
            "{}: unknown format; supported: csv, tsv, jsonl/ndjson",
            path.display()
        )),
        other => Err(format!(
            "{}: format {:?} not yet supported by Locke; supported: csv, tsv, jsonl/ndjson",
            path.display(),
            other
        )),
    }
}

/// Convert a `formats::TabularData` (string-rows) into a typed
/// `cjc_data::DataFrame`. Type inference is conservative: a column is
/// `Int` if every value parses as i64; `Float` if every value parses as
/// f64 with at least one decimal; `Bool` if every value is `true`/
/// `false` / `0` / `1`; else `Str`.
fn tabular_to_dataframe(t: crate::formats::TabularData) -> Result<DataFrame, String> {
    use cjc_data::Column;
    let n_cols = t.ncols();
    let n_rows = t.nrows();
    let mut columns: Vec<(String, Column)> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let name = t.headers.get(c).cloned().unwrap_or_else(|| format!("col_{}", c));
        let raw: Vec<&str> = (0..n_rows).map(|r| t.rows[r].get(c).map(|s| s.as_str()).unwrap_or("")).collect();
        let col = infer_column_type(&raw);
        columns.push((name, col));
    }
    DataFrame::from_columns(columns).map_err(|e| format!("from_columns: {:?}", e))
}

fn infer_column_type(raw: &[&str]) -> cjc_data::Column {
    use cjc_data::Column;
    // Try Bool first (most restrictive).
    let all_bool = raw.iter().all(|s| {
        let t = s.trim();
        t == "true" || t == "false" || t == "0" || t == "1"
    });
    if all_bool && !raw.is_empty() {
        let bools: Vec<bool> = raw
            .iter()
            .map(|s| matches!(s.trim(), "true" | "1"))
            .collect();
        return Column::Bool(bools);
    }
    // Try Int.
    let all_int = raw.iter().all(|s| s.trim().parse::<i64>().is_ok());
    if all_int && !raw.is_empty() {
        let ints: Vec<i64> = raw.iter().map(|s| s.trim().parse().unwrap()).collect();
        return Column::Int(ints);
    }
    // Try Float. Empty-string cell → NaN (null-as-NaN convention).
    let any_dot = raw.iter().any(|s| s.contains('.') || s.contains("e") || s.contains("E"));
    let all_float_or_empty = raw.iter().all(|s| {
        let t = s.trim();
        t.is_empty() || t.parse::<f64>().is_ok()
    });
    if all_float_or_empty && any_dot {
        let floats: Vec<f64> = raw
            .iter()
            .map(|s| {
                let t = s.trim();
                if t.is_empty() {
                    f64::NAN
                } else {
                    t.parse().unwrap()
                }
            })
            .collect();
        return Column::Float(floats);
    }
    // Fallback: Str.
    Column::Str(raw.iter().map(|s| (*s).to_string()).collect())
}

#[derive(Debug, Clone, Default)]
pub struct ValidateExtensions {
    pub time_col: Option<String>,
    pub max_timestamp: Option<i64>,
    pub gap_threshold: Option<i64>,
    pub target: Option<String>,
    pub primary_key: Option<String>,
}

fn cmd_validate(
    data: &PathBuf,
    label: Option<&str>,
    save_json: Option<&std::path::Path>,
    save_html: Option<&std::path::Path>,
    ext: ValidateExtensions,
    format: LockeFormat,
) -> Result<CmdOutput, String> {
    let df = read_csv(data)?;
    let dataset_label = label
        .map(String::from)
        .unwrap_or_else(|| data.display().to_string());
    let opts = ValidateOptions {
        dataset_label: dataset_label.clone(),
        config: ValidationConfig::default(),
        primary_key: ext.primary_key.clone(),
        ..Default::default()
    };
    let mut report = validate(&df, &opts);

    // v0.5: extension validators that need extra arguments beyond
    // ValidateOptions. These run *after* the standard validate() and
    // their findings are merged in. We rebuild the report from the
    // combined finding set so the run_id is content-addressed over
    // everything.
    let mut extra: Vec<cjc_locke::ValidationFinding> = Vec::new();
    if let Some(tc) = &ext.time_col {
        let mut tcfg = cjc_locke::temporal::TimeColumnConfig::default();
        tcfg.max_timestamp = ext.max_timestamp;
        tcfg.gap_threshold = ext.gap_threshold;
        extra.extend(cjc_locke::temporal::detect_temporal_issues(&df, tc, &tcfg));
    }
    if let Some(tg) = &ext.target {
        let lcfg = cjc_locke::leakage::LeakageConfig::default();
        extra.extend(cjc_locke::leakage::detect_target_leakage(&df, tg, &lcfg));
        // v0.6.3: multi-class target-leakage AUC (one-vs-rest).
        // Returns empty if the target is binary or has > max_classes
        // distinct values; safe to always call.
        extra.extend(cjc_locke::leakage::detect_target_leakage_multiclass(
            &df, tg, &lcfg,
        ));
        extra.extend(cjc_locke::detect_imbalanced_target(&df, tg, 0.05));
        // v0.6.4 — per-level deterministic-outcome leakage (E9064).
        // Complements E9063: catches level-by-level leakage that
        // column-wide ROC AUC misses (motivating example: diabetes-130
        // discharge_disposition_id death codes → readmitted=NO).
        let per_level_cfg = cjc_locke::PerLevelLeakageConfig::default();
        extra.extend(cjc_locke::detect_per_level_target_leakage(
            &df, tg, &per_level_cfg,
        ));
    }
    // Always-on v0.5 additions:
    // v0.6.4 — auto-detect string sentinels so the conditional-missingness
    // pairwise check sees `?`, `NA`, `NULL`, etc. on Str columns. Without
    // this, a column like diabetes-130's `weight` (96.9% `?`) is invisible
    // to the implication detector even when the user supplied a target.
    let cm_cfg = cjc_locke::ConditionalMissingnessConfig::default();
    let (auto_masks_cm, _) =
        cjc_locke::detect_string_sentinels(&df, &cjc_locke::ValidationConfig::default());
    let cm_masks =
        cjc_locke::merge_null_mask_maps(&cjc_locke::NullMaskMap::new(), &auto_masks_cm);
    extra.extend(cjc_locke::detect_conditional_missingness(&df, &cm_cfg, &cm_masks));
    extra.extend(cjc_locke::leakage::detect_id_like_columns(
        &df,
        &cjc_locke::leakage::LeakageConfig::default(),
    ));
    if let Some(pk) = &ext.primary_key {
        extra.extend(cjc_locke::detect_duplicate_key_conditioning(&df, pk));
    }

    if !extra.is_empty() {
        let mut all = report.findings.clone();
        all.extend(extra);
        report = cjc_locke::LockeReport::new(
            report.input.clone(),
            all,
            report.column_reports.clone(),
            report.assumptions.clone(),
        );
    }

    let worst = report.worst_severity().as_str().to_string();

    if let Some(path) = save_json {
        let json = cjc_locke::emit_locke_report_json(&report);
        fs::write(path, json).map_err(|e| {
            format!("failed to write report JSON to {}: {}", path.display(), e)
        })?;
    }
    if let Some(path) = save_html {
        // v0.5: use the with-DataFrame variant so the correlation
        // matrix renders.
        let html = cjc_locke::emit_locke_report_html_with_df(&report, &df);
        fs::write(path, html).map_err(|e| {
            format!("failed to write report HTML to {}: {}", path.display(), e)
        })?;
    }

    let text = match format {
        LockeFormat::Text => render_report_text(&report),
        LockeFormat::Json => render_report_json(&report),
    };
    Ok(CmdOutput { text, worst })
}

/// v0.4: `cjcl locke gate <reference.json> <current>` — diff the
/// stored reference report against a fresh validation of `current`
/// (or, if `current` is also a `.json` file, parse it as a stored
/// report).
///
/// v0.7+ A3: when `policy` is supplied, the policy is also applied
/// to the current report. The gate fails if either the appeared-
/// severity threshold is met OR any required-finding rule fails.
fn cmd_gate(
    reference: &PathBuf,
    current: &PathBuf,
    fail_on: Option<&str>,
    policy: Option<&std::path::Path>,
) -> Result<CmdOutput, String> {
    use cjc_locke::{diff_reports, emit_diff_text, parse_locke_report_json};

    // Load the reference report from JSON.
    let ref_json = fs::read_to_string(reference)
        .map_err(|e| format!("failed to read reference {}: {}", reference.display(), e))?;
    let ref_report = parse_locke_report_json(&ref_json)?;

    // Current: if it's .json, parse; otherwise, validate as data.
    let cur_report = if current.extension().and_then(|s| s.to_str()) == Some("json") {
        let txt = fs::read_to_string(current)
            .map_err(|e| format!("failed to read current {}: {}", current.display(), e))?;
        parse_locke_report_json(&txt)?
    } else {
        let df = read_csv(current)?;
        let opts = ValidateOptions {
            dataset_label: current.display().to_string(),
            config: ValidationConfig::default(),
            ..Default::default()
        };
        validate(&df, &opts)
    };

    let diff = diff_reports(&ref_report, &cur_report);
    // v0.7+ A3 — attach a policy if supplied via --policy.
    let diff = if let Some(path) = policy {
        let policy_src = fs::read_to_string(path)
            .map_err(|e| format!("failed to read policy {}: {}", path.display(), e))?;
        let parsed = parse_policy_toml(&policy_src)
            .map_err(|e| format!("policy parse error: {}", e))?;
        diff.with_policy(&parsed, &cur_report)
    } else {
        diff
    };
    let text = emit_diff_text(&diff);

    // Worst severity is computed from *appeared* findings only — that's
    // the v0.4 gate semantics. Disappeared findings are informational.
    // v0.7+ A3: also escalate worst to "error" when a required-finding
    // policy fails, so a `--fail-on error` from the caller correctly
    // fires on policy violations.
    let worst = match fail_on {
        Some(t) => {
            let threshold = parse_severity(t)?;
            if diff.gate_fails(threshold) || diff.policy_gate_fails() {
                if diff.policy_gate_fails() {
                    "error".into()
                } else {
                    diff.appeared_worst_severity().as_str().to_string()
                }
            } else {
                "info".into()
            }
        }
        None => {
            // Without an explicit threshold, a policy gate failure still
            // surfaces as "error" so the CLI can decide.
            if diff.policy_gate_fails() {
                "error".into()
            } else {
                "info".into()
            }
        }
    };

    Ok(CmdOutput { text, worst })
}

/// v0.7+ A3 — apply a governance policy to a freshly-validated dataset.
/// Emits the `PolicyResult` as canonical text. Exit code `0` when all
/// requirements satisfied, escalates to "error" worst-severity otherwise
/// so `--fail-on error` fires on policy violations.
fn cmd_policy_apply(
    data: &PathBuf,
    policy: &PathBuf,
) -> Result<CmdOutput, String> {
    let df = read_csv(data)?;
    let opts = ValidateOptions {
        dataset_label: data.display().to_string(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);

    let policy_src = fs::read_to_string(policy)
        .map_err(|e| format!("failed to read policy {}: {}", policy.display(), e))?;
    let parsed = parse_policy_toml(&policy_src)
        .map_err(|e| format!("policy parse error: {}", e))?;

    let result = cjc_locke::apply_policy(&report, &parsed);
    let text = cjc_locke::emit_policy_result_text(&result);
    let worst = if result.gate_fails() {
        "error".into()
    } else {
        "info".into()
    };
    Ok(CmdOutput { text, worst })
}

/// v0.7+ A3 — parse a `.cjcl-locke.toml` policy file into a `Policy`.
///
/// Schema (subset):
///
/// ```toml
/// [[suppress]]
/// code = "E9082"
/// column = "weight"          # optional
/// reason = "..."             # required
///
/// [[owner]]
/// team = "team-data-platform"
/// column = "patient_nbr"     # optional
/// code = "E9001"             # optional
///
/// [[require]]
/// code = "E9004"
/// operator = "=="            # one of: ==0, <=, >=, <, >, ==
/// threshold = 0
/// owner = "..."              # optional
/// ```
///
/// Returns a friendly error string when the file is malformed. The
/// caller (typically a CLI command) wraps that into a non-zero exit.
fn parse_policy_toml(src: &str) -> Result<cjc_locke::Policy, String> {
    use cjc_locke::{
        OwnerRule, Policy, RequiredFindingRule, RequirementOperator, SuppressionRule,
    };

    let doc = crate::toml_min::parse(src).map_err(|e| e.to_string())?;

    let get_str = |t: &crate::toml_min::TomlTable, key: &str| -> Option<String> {
        t.iter()
            .find_map(|(k, v)| if k == key { v.as_str().map(String::from) } else { None })
    };
    let get_int = |t: &crate::toml_min::TomlTable, key: &str| -> Option<i64> {
        t.iter()
            .find_map(|(k, v)| if k == key { v.as_int() } else { None })
    };

    let mut policy = Policy::default();

    for entry in doc.array_tables("suppress") {
        let code = get_str(entry, "code")
            .ok_or_else(|| "suppress entry missing required `code`".to_string())?;
        let column = get_str(entry, "column");
        let reason = get_str(entry, "reason")
            .ok_or_else(|| "suppress entry missing required `reason`".to_string())?;
        policy.suppressions.push(SuppressionRule {
            code,
            column,
            reason,
        });
    }

    for entry in doc.array_tables("owner") {
        let team = get_str(entry, "team")
            .ok_or_else(|| "owner entry missing required `team`".to_string())?;
        let column = get_str(entry, "column");
        let code = get_str(entry, "code");
        policy.owners.push(OwnerRule { team, column, code });
    }

    for entry in doc.array_tables("require") {
        let code = get_str(entry, "code")
            .ok_or_else(|| "require entry missing required `code`".to_string())?;
        let op_str = get_str(entry, "operator")
            .ok_or_else(|| "require entry missing required `operator`".to_string())?;
        let operator = RequirementOperator::from_label(&op_str).ok_or_else(|| {
            format!(
                "require entry has unknown operator {:?} (one of: ==0, <=, >=, <, >, ==)",
                op_str
            )
        })?;
        let threshold_raw = get_int(entry, "threshold").ok_or_else(|| {
            "require entry missing required `threshold` (non-negative integer)".to_string()
        })?;
        let threshold = u64::try_from(threshold_raw).map_err(|_| {
            format!(
                "require entry has invalid threshold {} (must be non-negative)",
                threshold_raw
            )
        })?;
        let owner = get_str(entry, "owner");
        policy.requirements.push(RequiredFindingRule {
            code,
            operator,
            threshold,
            owner,
        });
    }

    Ok(policy)
}

fn parse_severity(s: &str) -> Result<cjc_locke::FindingSeverity, String> {
    use cjc_locke::FindingSeverity::*;
    match s {
        "info" => Ok(Info),
        "notice" => Ok(Notice),
        "warning" => Ok(Warning),
        "error" => Ok(Error),
        other => Err(format!("unknown severity: {}", other)),
    }
}

fn cmd_drift(train: &PathBuf, test: &PathBuf, format: LockeFormat) -> Result<CmdOutput, String> {
    let t = read_csv(train)?;
    let s = read_csv(test)?;
    let report = cjc_locke::drift::compare(&t, &s, &DriftConfig::default());
    let worst = report.worst_severity().as_str().to_string();
    let mut text = String::new();
    text.push_str("# Locke Induction-Risk Report\n");
    text.push_str(&format!("n_train: {}\n", report.n_train));
    text.push_str(&format!("n_test:  {}\n", report.n_test));
    text.push_str(&format!("shared_columns: {}\n", report.shared_columns.join(", ")));
    text.push_str(&format!("train_only:     {}\n", report.train_only_columns.join(", ")));
    text.push_str(&format!("test_only:      {}\n", report.test_only_columns.join(", ")));
    text.push_str("findings:\n");
    for f in &report.findings {
        match format {
            LockeFormat::Text => text.push_str(&render_finding_text(f, 2)),
            LockeFormat::Json => text.push_str(&render_finding_json(f)),
        }
    }
    Ok(CmdOutput { text, worst })
}

fn cmd_belief(data: &PathBuf, format: LockeFormat) -> Result<CmdOutput, String> {
    let df = read_csv(data)?;
    let opts = ValidateOptions {
        dataset_label: data.display().to_string(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let belief = cjc_locke::api::belief_report_from_locke(&report);
    let worst = report.worst_severity().as_str().to_string();
    let mut text = String::new();
    text.push_str("# Locke Belief Report\n");
    text.push_str(&format!("dataset: {}\n", report.input.dataset_label));
    text.push_str(&format!("n_rows: {}\n", report.input.n_rows));
    text.push_str(&format!("sample_score (n={}): {:.3}\n", report.input.n_rows, sample_score_from_n(report.input.n_rows)));
    text.push_str("\n");
    text.push_str(&belief.score.explain());
    text.push_str("\nassumptions:\n");
    for a in &belief.assumptions {
        text.push_str(&format!("  - {}\n", a));
    }
    text.push_str("\nrecommended next steps:\n");
    for s in &belief.recommended_next_steps {
        text.push_str(&format!("  - {}\n", s));
    }
    if format == LockeFormat::Json {
        // Stable JSON-ish single-line.
        text = format!(
            "{{\"dataset\":\"{}\",\"n_rows\":{},\"overall\":{:.6},\"schema\":{:.6},\"missingness\":{:.6},\"drift\":{:.6},\"leakage\":{:.6},\"lineage\":{:.6},\"sample\":{:.6},\"duplication\":{:.6},\"constraint\":{:.6}}}\n",
            json_escape(&report.input.dataset_label),
            report.input.n_rows,
            belief.score.overall,
            belief.score.schema_score,
            belief.score.missingness_score,
            belief.score.drift_score,
            belief.score.leakage_score,
            belief.score.lineage_score,
            belief.score.sample_score,
            belief.score.duplication_score,
            belief.score.constraint_score,
        );
    }
    Ok(CmdOutput { text, worst })
}

fn cmd_lineage(
    data: &PathBuf,
    label: Option<&str>,
    _format: LockeFormat,
    mermaid: bool,
) -> Result<CmdOutput, String> {
    let df = read_csv(data)?;
    let lbl = label.unwrap_or_else(|| data.to_str().unwrap_or("dataset"));
    let g = cjc_locke::api::lineage_for_dataset(lbl, &df);
    let text = if mermaid {
        cjc_locke::lineage::emit_lineage_mermaid(&g)
    } else {
        emit_lineage_text(&g)
    };
    Ok(CmdOutput { text, worst: "info".into() })
}

/// v0.6 — re-run `validate` N times and assert all reports are byte-identical.
///
/// Returns a short success-or-divergence summary in `text`. Exits with code
/// 3 (caller's responsibility) when any byte differs between runs.
fn cmd_verify(data: &PathBuf, runs: u32) -> Result<CmdOutput, String> {
    if runs < 2 {
        return Err("--runs must be >= 2".into());
    }
    let df = read_csv(data)?;
    let opts = ValidateOptions {
        dataset_label: data.to_str().unwrap_or("dataset").into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let first = validate(&df, &opts);
    let first_bytes = cjc_locke::json_emit::emit_locke_report_json(&first);
    let mut divergent_runs: Vec<u32> = Vec::new();
    for i in 1..runs {
        let r = validate(&df, &opts);
        let bytes = cjc_locke::json_emit::emit_locke_report_json(&r);
        if bytes != first_bytes {
            divergent_runs.push(i);
        }
    }
    let mut text = String::new();
    text.push_str("# Locke Reproducibility Verifier\n");
    text.push_str(&format!("dataset: {}\n", first.input.dataset_label));
    text.push_str(&format!("runs: {}\n", runs));
    text.push_str(&format!("run_id: {}\n", first.run_id));
    text.push_str(&format!("findings: {}\n", first.findings.len()));
    text.push_str(&format!("report_bytes: {}\n", first_bytes.len()));
    if divergent_runs.is_empty() {
        text.push_str("status: REPRODUCIBLE — all runs byte-identical\n");
        Ok(CmdOutput { text, worst: "info".into() })
    } else {
        text.push_str(&format!(
            "status: DIVERGENT — {}/{} runs differed from run 0 (indices: {:?})\n",
            divergent_runs.len(),
            runs,
            divergent_runs
        ));
        // Worst severity "error" so a `--fail-on error` would also fire.
        Ok(CmdOutput { text, worst: "error".into() })
    }
}

/// v0.7+ A2 — `cjcl locke trace-value <data> <column> <value>` emits the
/// per-value canonicalisation lineage for a single `(column, value)`
/// pair. Surfaces "what would happen to this value if Locke's
/// suggested normalisations were adopted."
///
/// Exit semantics:
/// - 0 when the value is present and its lineage was emitted.
/// - 2 (caller-mapped from the returned `Err`) when the column or value
///   is missing.
fn cmd_trace_value(
    data: &PathBuf,
    column: &str,
    value: &str,
) -> Result<CmdOutput, String> {
    let df = read_csv(data)?;
    let cfg = cjc_locke::PerValueLineageConfig::default();
    let lineage = cjc_locke::trace_value(&df, &cfg, column, value).ok_or_else(|| {
        format!(
            "value {:?} not found in column {:?} (or column is not categorical)",
            value, column
        )
    })?;
    let text = cjc_locke::emit_value_trace_text(&lineage);
    Ok(CmdOutput { text, worst: "info".into() })
}

fn cmd_causal(
    data: &PathBuf,
    target: Option<&str>,
    observational_only: bool,
    format: LockeFormat,
) -> Result<CmdOutput, String> {
    let df = read_csv(data)?;
    let mut cfg = CausalConfig::default();
    if observational_only {
        cfg.mode = CausalMode::ObservationalOnly;
    }
    let report = cjc_locke::api::causal_guardrail(&df, target, &cfg, None, false);
    let mut text = String::new();
    text.push_str("# Locke Causal-Guardrail Report\n");
    text.push_str(&format!("disclaimer: {}\n", report.disclaimer));
    text.push_str(&format!("mode: {:?}\n", report.mode));
    text.push_str(&format!("n_correlations_inspected: {}\n", report.correlations.len()));
    text.push_str("warnings:\n");
    for w in &report.warnings {
        match format {
            LockeFormat::Text => text.push_str(&format!(
                "  - {:?}: {} [a={}, b={}]\n",
                w.kind, w.message, w.a, w.b
            )),
            LockeFormat::Json => text.push_str(&format!(
                "  {{\"kind\":\"{:?}\",\"a\":\"{}\",\"b\":\"{}\",\"message\":\"{}\"}}\n",
                w.kind,
                json_escape(&w.a),
                json_escape(&w.b),
                json_escape(&w.message)
            )),
        }
    }
    if !report.confounder_hints.is_empty() {
        text.push_str("confounder_hints:\n");
        for h in &report.confounder_hints {
            text.push_str(&format!(
                "  - candidate={} feature={} target={} r_with_feature={:.3} r_with_target={:.3}\n",
                h.candidate, h.feature, h.target, h.r_with_feature, h.r_with_target,
            ));
        }
    }
    Ok(CmdOutput { text, worst: "info".into() })
}

// ─── Renderers ────────────────────────────────────────────────────────────────

fn render_report_text(r: &cjc_locke::LockeReport) -> String {
    let mut s = String::new();
    s.push_str("# Locke Validation Report\n");
    s.push_str(&format!("schema_version: {}\n", r.schema_version));
    s.push_str(&format!("dataset: {}\n", r.input.dataset_label));
    s.push_str(&format!("n_rows: {}\n", r.input.n_rows));
    s.push_str(&format!("n_cols: {}\n", r.input.n_cols));
    s.push_str(&format!("run_id: {}\n", r.run_id));
    s.push_str(&format!(
        "severity_counts: info={} notice={} warning={} error={}\n",
        r.severity_counts.info, r.severity_counts.notice, r.severity_counts.warning, r.severity_counts.error
    ));
    s.push_str("assumptions:\n");
    for a in &r.assumptions {
        s.push_str(&format!("  - {}\n", a));
    }
    s.push_str("findings:\n");
    for f in &r.findings {
        s.push_str(&render_finding_text(f, 2));
    }
    s
}

fn render_finding_text(f: &cjc_locke::ValidationFinding, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let mut s = String::new();
    let column = f.column.as_deref().unwrap_or("-");
    s.push_str(&format!(
        "{}- code={} severity={} column={} id={}\n",
        pad, f.code, f.severity, column, f.id
    ));
    s.push_str(&format!("{}  message: {}\n", pad, f.message));
    if !f.evidence.is_empty() {
        s.push_str(&format!("{}  evidence:\n", pad));
        for e in &f.evidence {
            s.push_str(&format!("{}    - {}\n", pad, render_evidence(e)));
        }
    }
    if !f.assumptions.is_empty() {
        s.push_str(&format!("{}  assumptions:\n", pad));
        for a in &f.assumptions {
            s.push_str(&format!("{}    - {}\n", pad, a));
        }
    }
    if !f.suggested_next_checks.is_empty() {
        s.push_str(&format!("{}  next_checks:\n", pad));
        for c in &f.suggested_next_checks {
            s.push_str(&format!("{}    - {}\n", pad, c));
        }
    }
    s
}

fn render_evidence(e: &cjc_locke::FindingEvidence) -> String {
    use cjc_locke::FindingEvidence::*;
    match e {
        Count { label, value } => format!("{}={}", label, value),
        Ratio { label, value } => format!("{}={:.6}", label, value),
        Range { label, min, max } => format!("{}=[{:.6}, {:.6}]", label, min, max),
        Metric { label, value } => format!("{}={:.6}", label, value),
        Sample { label, value } => format!("{}={:?}", label, value),
    }
}

fn render_report_json(r: &cjc_locke::LockeReport) -> String {
    // One finding per line, plus a header. Stable order from canonical sort.
    let mut s = String::new();
    s.push_str(&format!(
        "{{\"kind\":\"header\",\"dataset\":\"{}\",\"n_rows\":{},\"n_cols\":{},\"run_id\":\"{}\"}}\n",
        json_escape(&r.input.dataset_label),
        r.input.n_rows,
        r.input.n_cols,
        r.run_id
    ));
    for f in &r.findings {
        s.push_str(&render_finding_json(f));
    }
    s
}

fn render_finding_json(f: &cjc_locke::ValidationFinding) -> String {
    format!(
        "{{\"kind\":\"finding\",\"code\":\"{}\",\"severity\":\"{}\",\"column\":\"{}\",\"id\":\"{}\",\"message\":\"{}\"}}\n",
        f.code,
        f.severity,
        f.column.as_deref().unwrap_or(""),
        f.id,
        json_escape(&f.message)
    )
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[allow(dead_code)]
pub fn _link_belief_score_into_binary(_: &BeliefScore) {
    // Forces the linker to keep BeliefScore symbols if the binary doesn't
    // otherwise reference them. Cheap no-op.
}

/// Parse `cjcl locke <sub> ...` arguments. `sub_args` is everything
/// AFTER the `locke` token (so `["validate", "data.csv"]`). Returns
/// `None` for unknown subcommands or missing positionals.
pub fn try_parse_sub(sub_args: &[String]) -> Option<LockeArgs> {
    if sub_args.is_empty() {
        return None;
    }
    let sub = sub_args[0].as_str();
    let rest = &sub_args[1..];
    let mut format = LockeFormat::Text;
    let mut fail_on_severity: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut target: Option<String> = None;
    let mut observational_only = false;
    let mut label: Option<String> = None;
    let mut save_json: Option<PathBuf> = None;
    let mut save_html: Option<PathBuf> = None;
    // v0.5 extensions
    let mut time_col: Option<String> = None;
    let mut max_timestamp: Option<i64> = None;
    let mut gap_threshold: Option<i64> = None;
    let mut primary_key: Option<String> = None;
    // v0.6 flags
    let mut mermaid = false;
    let mut runs: u32 = 3;
    // v0.7+ A3 flag
    let mut policy_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < rest.len() {
        let a = &rest[i];
        match a.as_str() {
            "--json" => format = LockeFormat::Json,
            "--text" => format = LockeFormat::Text,
            "--fail-on" => {
                i += 1;
                fail_on_severity = rest.get(i).cloned();
            }
            "--target" => {
                i += 1;
                target = rest.get(i).cloned();
            }
            "--observational-only" => observational_only = true,
            "--label" => {
                i += 1;
                label = rest.get(i).cloned();
            }
            "--save-json" => {
                i += 1;
                save_json = rest.get(i).map(PathBuf::from);
            }
            "--html" | "--save-html" => {
                i += 1;
                save_html = rest.get(i).map(PathBuf::from);
            }
            // v0.5 flags
            "--time-col" => {
                i += 1;
                time_col = rest.get(i).cloned();
            }
            "--max-timestamp" => {
                i += 1;
                max_timestamp = rest.get(i).and_then(|s| s.parse::<i64>().ok());
            }
            "--gap-threshold" => {
                i += 1;
                gap_threshold = rest.get(i).and_then(|s| s.parse::<i64>().ok());
            }
            "--primary-key" => {
                i += 1;
                primary_key = rest.get(i).cloned();
            }
            "--mermaid" => mermaid = true,
            "--runs" => {
                i += 1;
                if let Some(n) = rest.get(i).and_then(|s| s.parse::<u32>().ok()) {
                    runs = n;
                }
            }
            // v0.7+ A3 — policy file path; valid for `gate` and `policy apply`.
            "--policy" => {
                i += 1;
                policy_path = rest.get(i).map(PathBuf::from);
            }
            _ => positional.push(a.clone()),
        }
        i += 1;
    }

    let subcommand = match sub {
        "validate" => {
            let data = positional.first()?.into();
            LockeSubcommand::Validate {
                data,
                label,
                save_json,
                save_html,
                time_col,
                max_timestamp,
                gap_threshold,
                target,
                primary_key,
            }
        }
        "drift" => {
            let train = positional.first()?.into();
            let test = positional.get(1)?.into();
            LockeSubcommand::Drift { train, test }
        }
        "belief" => {
            let data = positional.first()?.into();
            LockeSubcommand::Belief { data }
        }
        "lineage" => {
            let data = positional.first()?.into();
            LockeSubcommand::Lineage { data, label, mermaid }
        }
        "causal" => {
            let data = positional.first()?.into();
            LockeSubcommand::Causal { data, target, observational_only }
        }
        "gate" => {
            let reference = positional.first()?.into();
            let current = positional.get(1)?.into();
            LockeSubcommand::Gate {
                reference,
                current,
                policy: policy_path.clone(),
            }
        }
        "verify" => {
            let data = positional.first()?.into();
            LockeSubcommand::Verify { data, runs }
        }
        "trace-value" => {
            // trace-value <data.csv> <column> <value>
            let data = positional.first()?.into();
            let column = positional.get(1)?.clone();
            let value = positional.get(2)?.clone();
            LockeSubcommand::TraceValue { data, column, value }
        }
        // v0.7+ A3 — `policy apply <data> --policy <policy.toml>`.
        // The `apply` action is the only one shipped in A3.1; future
        // actions (lint, show, etc.) will surface as further nested
        // verbs.
        "policy" => {
            // Sub-action must be "apply".
            let action = positional.first().map(String::as_str)?;
            if action != "apply" {
                return None;
            }
            let data = positional.get(1)?.into();
            let policy = policy_path?;
            LockeSubcommand::PolicyApply { data, policy }
        }
        _ => return None,
    };

    Some(LockeArgs {
        subcommand,
        format,
        fail_on_severity,
    })
}

/// Entry point matching the existing CLI convention. Parses `sub_args`
/// (everything after `locke`) and dispatches.
pub fn run(sub_args: &[String]) {
    let args = match try_parse_sub(sub_args) {
        Some(a) => a,
        None => {
            print_help();
            process::exit(2);
        }
    };
    let code = dispatch(args);
    if code != 0 {
        process::exit(code);
    }
}


pub fn print_help() {
    eprintln!(
        "cjcl locke — evidence-aware analytics

usage:
  cjcl locke validate <data.csv> [--label NAME] [--json] [--fail-on SEV] [--save-json PATH] [--html PATH]
                                 [--time-col COL] [--max-timestamp N] [--gap-threshold N]
                                 [--target COL] [--primary-key COL]
  cjcl locke drift    <train.csv> <test.csv>      [--json] [--fail-on SEV]
  cjcl locke belief   <data.csv>                  [--json]
  cjcl locke lineage  <data.csv> [--label NAME] [--mermaid]
  cjcl locke causal   <data.csv> [--target COL] [--observational-only] [--json]
  cjcl locke gate     <reference.json> <current>  [--fail-on SEV] [--policy FILE]
  cjcl locke verify   <data.csv> [--runs N]
  cjcl locke trace-value <data.csv> <column> <value>
                                  emit the per-value canonicalisation
                                  lineage chain for a single distinct
                                  value (v0.7+ A2)
  cjcl locke policy apply <data.csv> --policy <.cjcl-locke.toml>
                                  apply a governance policy
                                  (suppressions + owner annotations +
                                  required-finding checks) to the
                                  validation output (v0.7+ A3)

flags:
  --json                 emit one JSON-ish record per line
  --text                 emit indented text (default)
  --fail-on SEV          exit non-zero if worst finding severity >= SEV
                         (one of: info, notice, warning, error)
  --label NAME           override the dataset label in the report
  --target COL           target column for confounder hints (causal only)
  --observational-only   declare data is observational; raises warning severity
  --mermaid              emit lineage as a Quarto/Markdown Mermaid block
  --runs N               number of validate runs to compare (verify; default 3)
  --policy FILE          path to a .cjcl-locke.toml policy file
                         (gate, policy apply)

determinism:
  every report is deterministic — repeated runs over the same data
  produce byte-identical IDs and bytes.
"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_validate_simple() {
        let parsed = try_parse_sub(&args(&["validate", "data.csv"])).expect("validate parse");
        match parsed.subcommand {
            LockeSubcommand::Validate {
                data,
                save_json,
                save_html,
                time_col,
                target,
                primary_key,
                ..
            } => {
                assert_eq!(data.to_str().unwrap(), "data.csv");
                assert!(save_json.is_none());
                assert!(save_html.is_none());
                assert!(time_col.is_none());
                assert!(target.is_none());
                assert!(primary_key.is_none());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_drift_with_json_flag() {
        let parsed = try_parse_sub(&args(&["drift", "t.csv", "s.csv", "--json"])).expect("drift parse");
        assert!(matches!(parsed.subcommand, LockeSubcommand::Drift { .. }));
        assert_eq!(parsed.format, LockeFormat::Json);
    }

    #[test]
    fn parse_causal_with_target_and_obs() {
        let parsed = try_parse_sub(&args(&[
            "causal",
            "data.csv",
            "--target",
            "y",
            "--observational-only",
        ]))
        .expect("causal parse");
        match parsed.subcommand {
            LockeSubcommand::Causal { target, observational_only, .. } => {
                assert_eq!(target.as_deref(), Some("y"));
                assert!(observational_only);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_subcommand_returns_none() {
        assert!(try_parse_sub(&args(&["wat"])).is_none());
    }

    #[test]
    fn parse_validate_with_save_json() {
        let parsed = try_parse_sub(&args(&[
            "validate",
            "data.csv",
            "--save-json",
            "report.json",
        ]))
        .expect("validate --save-json parse");
        match parsed.subcommand {
            LockeSubcommand::Validate { save_json, .. } => {
                assert_eq!(save_json.as_ref().and_then(|p| p.to_str()), Some("report.json"));
            }
            _ => panic!("expected Validate"),
        }
    }

    #[test]
    fn parse_gate_subcommand() {
        let parsed = try_parse_sub(&args(&["gate", "ref.json", "data.csv"])).expect("gate parse");
        match parsed.subcommand {
            LockeSubcommand::Gate { reference, current, policy } => {
                assert_eq!(reference.to_str().unwrap(), "ref.json");
                assert_eq!(current.to_str().unwrap(), "data.csv");
                assert!(policy.is_none());
            }
            _ => panic!("expected Gate"),
        }
    }

    #[test]
    fn worst_at_least_thresholds() {
        assert!(worst_at_least("warning", "info"));
        assert!(worst_at_least("error", "warning"));
        assert!(!worst_at_least("info", "warning"));
    }

    #[test]
    fn json_escape_handles_quotes_and_backslashes() {
        let s = json_escape("a\"b\\c\n");
        assert_eq!(s, "a\\\"b\\\\c\\n");
    }

    // ── v0.6: lineage --mermaid, verify --runs ──────────────────────────

    #[test]
    fn parse_lineage_with_mermaid_flag() {
        let parsed = try_parse_sub(&args(&["lineage", "data.csv", "--mermaid"]))
            .expect("lineage --mermaid parse");
        match parsed.subcommand {
            LockeSubcommand::Lineage { mermaid, .. } => assert!(mermaid),
            _ => panic!("expected Lineage"),
        }
    }

    #[test]
    fn parse_lineage_default_no_mermaid() {
        let parsed =
            try_parse_sub(&args(&["lineage", "data.csv"])).expect("lineage default parse");
        match parsed.subcommand {
            LockeSubcommand::Lineage { mermaid, .. } => assert!(!mermaid),
            _ => panic!("expected Lineage"),
        }
    }

    #[test]
    fn parse_verify_subcommand_with_runs() {
        let parsed =
            try_parse_sub(&args(&["verify", "data.csv", "--runs", "5"])).expect("verify parse");
        match parsed.subcommand {
            LockeSubcommand::Verify { data, runs } => {
                assert_eq!(data.to_str().unwrap(), "data.csv");
                assert_eq!(runs, 5);
            }
            _ => panic!("expected Verify"),
        }
    }

    #[test]
    fn parse_verify_default_runs_is_three() {
        let parsed = try_parse_sub(&args(&["verify", "data.csv"])).expect("verify default parse");
        match parsed.subcommand {
            LockeSubcommand::Verify { runs, .. } => assert_eq!(runs, 3),
            _ => panic!(),
        }
    }

    // ── v0.7+ A2: trace-value subcommand ─────────────────────────────────

    #[test]
    fn parse_trace_value_subcommand() {
        let parsed = try_parse_sub(&args(&[
            "trace-value",
            "data.csv",
            "race",
            "Caucasian",
        ]))
        .expect("trace-value parse");
        match parsed.subcommand {
            LockeSubcommand::TraceValue { data, column, value } => {
                assert_eq!(data.to_str().unwrap(), "data.csv");
                assert_eq!(column, "race");
                assert_eq!(value, "Caucasian");
            }
            _ => panic!("expected TraceValue"),
        }
    }

    #[test]
    fn parse_trace_value_missing_positional_returns_none() {
        // Only two positionals supplied — missing the value arg.
        assert!(
            try_parse_sub(&args(&["trace-value", "data.csv", "race"])).is_none(),
            "trace-value with only 2 positionals should fail to parse"
        );
    }

    #[test]
    fn parse_trace_value_accepts_question_mark_value() {
        // Real-world case: tracing the diabetes-130 `?` sentinel.
        let parsed = try_parse_sub(&args(&[
            "trace-value",
            "diabetes.csv",
            "weight",
            "?",
        ]))
        .expect("trace-value with ? value");
        match parsed.subcommand {
            LockeSubcommand::TraceValue { value, .. } => assert_eq!(value, "?"),
            _ => panic!("expected TraceValue"),
        }
    }

    // ── v0.7+ A3: policy + gate --policy ────────────────────────────────

    #[test]
    fn parse_policy_apply_subcommand() {
        let parsed = try_parse_sub(&args(&[
            "policy", "apply", "data.csv",
            "--policy", "p.toml",
        ]))
        .expect("policy apply parse");
        match parsed.subcommand {
            LockeSubcommand::PolicyApply { data, policy } => {
                assert_eq!(data.to_str().unwrap(), "data.csv");
                assert_eq!(policy.to_str().unwrap(), "p.toml");
            }
            _ => panic!("expected PolicyApply"),
        }
    }

    #[test]
    fn parse_policy_apply_requires_policy_flag() {
        assert!(
            try_parse_sub(&args(&["policy", "apply", "data.csv"])).is_none(),
            "policy apply without --policy should not parse"
        );
    }

    #[test]
    fn parse_policy_with_unknown_action_returns_none() {
        assert!(
            try_parse_sub(&args(&[
                "policy", "show", "data.csv", "--policy", "p.toml"
            ])).is_none(),
            "policy with unknown action should fail to parse"
        );
    }

    #[test]
    fn parse_gate_with_policy_flag() {
        let parsed = try_parse_sub(&args(&[
            "gate", "ref.json", "current.csv",
            "--policy", "p.toml",
        ]))
        .expect("gate --policy parse");
        match parsed.subcommand {
            LockeSubcommand::Gate { reference, current, policy } => {
                assert_eq!(reference.to_str().unwrap(), "ref.json");
                assert_eq!(current.to_str().unwrap(), "current.csv");
                assert_eq!(policy.as_ref().and_then(|p| p.to_str()), Some("p.toml"));
            }
            _ => panic!("expected Gate"),
        }
    }

    // ── policy TOML parser ───────────────────────────────────────────────

    #[test]
    fn parse_policy_toml_minimal() {
        let src = r#"
[[suppress]]
code = "E9082"
column = "weight"
reason = "real distinction, not typo"
"#;
        let p = parse_policy_toml(src).expect("policy parse");
        assert_eq!(p.suppressions.len(), 1);
        assert_eq!(p.suppressions[0].code, "E9082");
        assert_eq!(p.suppressions[0].column.as_deref(), Some("weight"));
        assert_eq!(p.suppressions[0].reason, "real distinction, not typo");
    }

    #[test]
    fn parse_policy_toml_full_three_kinds() {
        let src = r#"
[[suppress]]
code = "E9082"
column = "weight"
reason = "ack"

[[owner]]
team = "team-data-platform"
column = "patient_nbr"

[[require]]
code = "E9004"
operator = "==0"
threshold = 0
owner = "team-data-platform"
"#;
        let p = parse_policy_toml(src).expect("policy parse");
        assert_eq!(p.suppressions.len(), 1);
        assert_eq!(p.owners.len(), 1);
        assert_eq!(p.requirements.len(), 1);
        assert_eq!(p.owners[0].team, "team-data-platform");
        assert_eq!(p.requirements[0].code, "E9004");
        assert_eq!(
            p.requirements[0].operator,
            cjc_locke::RequirementOperator::EqZero
        );
        assert_eq!(p.requirements[0].owner.as_deref(), Some("team-data-platform"));
    }

    #[test]
    fn parse_policy_toml_missing_required_fields_errs() {
        // `reason` missing on suppress
        let src = r#"
[[suppress]]
code = "E9082"
"#;
        let err = parse_policy_toml(src).unwrap_err();
        assert!(err.contains("reason"), "got: {}", err);
    }

    #[test]
    fn parse_policy_toml_unknown_operator_errs() {
        let src = r#"
[[require]]
code = "E9004"
operator = "not-an-op"
threshold = 0
"#;
        let err = parse_policy_toml(src).unwrap_err();
        assert!(err.contains("unknown operator"), "got: {}", err);
    }

    #[test]
    fn parse_policy_toml_negative_threshold_errs() {
        let src = r#"
[[require]]
code = "E9004"
operator = "<="
threshold = -1
"#;
        let err = parse_policy_toml(src).unwrap_err();
        assert!(err.contains("non-negative"), "got: {}", err);
    }

    #[test]
    fn parse_policy_toml_empty_input_yields_empty_policy() {
        let p = parse_policy_toml("").expect("empty parses ok");
        assert!(p.suppressions.is_empty());
        assert!(p.owners.is_empty());
        assert!(p.requirements.is_empty());
    }

    #[test]
    fn parse_policy_toml_supports_alias_operators() {
        // "lte" should map to LessOrEqual same as "<=".
        let src = r#"
[[require]]
code = "E9001"
operator = "lte"
threshold = 5
"#;
        let p = parse_policy_toml(src).expect("parse ok");
        assert_eq!(
            p.requirements[0].operator,
            cjc_locke::RequirementOperator::LessOrEqual
        );
    }

    #[test]
    fn parse_policy_toml_preserves_declaration_order() {
        // First-match-wins matters: order in the TOML file equals order
        // in Policy.suppressions.
        let src = r#"
[[suppress]]
code = "E9001"
reason = "first"

[[suppress]]
code = "E9001"
reason = "second"
"#;
        let p = parse_policy_toml(src).expect("parse ok");
        assert_eq!(p.suppressions[0].reason, "first");
        assert_eq!(p.suppressions[1].reason, "second");
    }
}
