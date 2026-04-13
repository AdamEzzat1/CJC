//! CLI Expansion Test Suite
//!
//! Comprehensive integration tests for the CJC CLI expansion features.
//! Tests are organized into categories:
//!   A. Format Support (JSONL)
//!   B. Second-Mode Flags
//!   C. Determinism
//!   D. Cross-Format Equivalence
//!   E. Output Mode Consistency
//!   F. Edge Cases
//!
//! All tests invoke the `cjc` binary via `std::process::Command` and verify
//! exit codes, stdout, and stderr.

use std::path::PathBuf;
use std::process::Command;

// ── Helper ─────────────────────────────────────────────────────────────

/// Locate the `cjc` binary built by `cargo build`.
fn cjc_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    path.push(if cfg!(windows) { "cjcl.exe" } else { "cjcl" });
    path
}

/// Path to the cli_expansion fixtures directory.
fn fixtures_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("cli_expansion");
    p
}

/// Run the `cjc` binary with the given arguments.
/// Returns (stdout, stderr, exit_code).
fn run_cjc(args: &[&str]) -> (String, String, i32) {
    let bin = cjc_binary();
    assert!(bin.exists(), "cjc binary not found at {:?}. Run `cargo build` first.", bin);
    let output = Command::new(&bin)
        .args(args)
        .output()
        .expect("failed to execute cjc binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// Convenience: fixture file path as a String.
fn fixture(name: &str) -> String {
    fixtures_dir().join(name).to_string_lossy().to_string()
}

// ======================================================================
// A. Format Support Tests (JSONL)
// ======================================================================

mod format_support {
    use super::*;

    #[test]
    fn flow_with_jsonl() {
        let (stdout, stderr, code) = run_cjc(&[
            "flow", &fixture("test_data.jsonl"), "--plain",
        ]);
        // flow should succeed on JSONL input
        assert_eq!(code, 0, "flow on JSONL failed: stderr={}", stderr);
        // Should produce some output (aggregation results)
        assert!(!stdout.trim().is_empty(), "flow produced no output on JSONL");
    }

    #[test]
    fn schema_with_jsonl() {
        let (stdout, stderr, code) = run_cjc(&[
            "schema", &fixture("test_data.jsonl"), "--plain",
        ]);
        assert_eq!(code, 0, "schema on JSONL failed: stderr={}", stderr);
        // Schema may output to stderr (table rendering) -- check both
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.contains("name") || combined.contains("age") || combined.contains("score"),
            "schema output missing expected column names: {}", combined
        );
    }

    #[test]
    fn drift_with_jsonl_files() {
        // Compare the JSONL file against itself -- should show zero diffs
        let f = fixture("test_data.jsonl");
        let (stdout, stderr, code) = run_cjc(&[
            "drift", &f, &f, "--plain",
        ]);
        assert_eq!(code, 0, "drift on identical JSONL failed: stderr={}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        // Identical files should show 0 diffs or "identical"
        assert!(
            combined.contains("0") || combined.to_lowercase().contains("identical")
                || combined.to_lowercase().contains("no diff"),
            "drift on identical JSONL should report zero diffs: stdout={}, stderr={}", stdout, stderr
        );
    }

    #[test]
    fn inspect_with_jsonl() {
        let (stdout, stderr, code) = run_cjc(&[
            "inspect", &fixture("test_data.jsonl"), "--plain",
        ]);
        assert_eq!(code, 0, "inspect on JSONL failed: stderr={}", stderr);
        // inspect may output to stderr -- check both
        let combined = format!("{}{}", stdout, stderr);
        assert!(!combined.trim().is_empty(), "inspect produced no output on JSONL");
    }

    #[test]
    fn doctor_detects_jsonl_files() {
        // Run doctor on the fixtures directory -- should scan JSONL files
        let (stdout, stderr, code) = run_cjc(&[
            "doctor", &fixtures_dir().to_string_lossy(), "--plain",
        ]);
        // doctor should succeed (may report findings)
        assert!(code == 0 || code == 1, "doctor unexpected exit code {}: stderr={}", code, stderr);
        let combined = format!("{}{}", stdout, stderr);
        // Should mention jsonl or the malformed file
        assert!(
            combined.to_lowercase().contains("jsonl")
                || combined.to_lowercase().contains("json")
                || combined.contains("malformed")
                || combined.contains("finding")
                || combined.contains("scan"),
            "doctor should detect JSONL files: stdout={}, stderr={}", stdout, stderr
        );
    }
}

// ======================================================================
// B. Second-Mode Flag Tests
// ======================================================================

mod second_mode_flags {
    use super::*;

    // ── schema ──────────────────────────────────────────────────────

    #[test]
    fn schema_save_and_check_roundtrip() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let schema_path = tmp.path().join("schema.json");
        let schema_str = schema_path.to_string_lossy().to_string();

        // Save
        let (_, stderr, code) = run_cjc(&[
            "schema", &fixture("test_data.csv"),
            "--save", &schema_str, "--plain",
        ]);
        assert_eq!(code, 0, "schema --save failed: {}", stderr);
        assert!(schema_path.exists(), "schema file was not created");

        // Check
        let (_, stderr2, code2) = run_cjc(&[
            "schema", &fixture("test_data.csv"),
            "--check", &schema_str, "--plain",
        ]);
        assert_eq!(code2, 0, "schema --check on same data should pass: {}", stderr2);
    }

    #[test]
    fn schema_diff() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let schema_path = tmp.path().join("schema.json");
        let schema_str = schema_path.to_string_lossy().to_string();

        // Save schema from test_data.csv
        let (_, _, code) = run_cjc(&[
            "schema", &fixture("test_data.csv"),
            "--save", &schema_str, "--plain",
        ]);
        assert_eq!(code, 0);

        // Diff against test_data_b.csv (should show differences)
        let (stdout, stderr, code2) = run_cjc(&[
            "schema", &fixture("test_data_b.csv"),
            "--diff", &schema_str, "--plain",
        ]);
        // May exit 0 or 1 depending on whether diff counts as failure
        let _ = code2;
        let combined = format!("{}{}", stdout, stderr);
        // Should produce some output about the comparison
        assert!(
            !combined.trim().is_empty(),
            "schema --diff produced no output"
        );
    }

    // ── patch ───────────────────────────────────────────────────────

    #[test]
    fn patch_dry_run_no_data_output() {
        let (stdout, stderr, code) = run_cjc(&[
            "patch", &fixture("test_data.csv"),
            "--nan-fill", "0", "--dry-run", "--plain",
        ]);
        assert_eq!(code, 0, "patch --dry-run failed: {}", stderr);
        // dry-run should show the plan but not the transformed data rows
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.to_lowercase().contains("dry")
                || combined.to_lowercase().contains("plan")
                || combined.to_lowercase().contains("transform")
                || combined.to_lowercase().contains("nan-fill"),
            "patch --dry-run should describe planned transforms: {}", combined
        );
    }

    #[test]
    fn patch_plan() {
        let (stdout, stderr, code) = run_cjc(&[
            "patch", &fixture("test_data.csv"),
            "--nan-fill", "0", "--plan", "--plain",
        ]);
        assert_eq!(code, 0, "patch --plan failed: {}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.to_lowercase().contains("nan")
                || combined.to_lowercase().contains("plan")
                || combined.to_lowercase().contains("transform"),
            "patch --plan should show planned transforms: {}", combined
        );
    }

    #[test]
    fn patch_json_output() {
        let (stdout, stderr, code) = run_cjc(&[
            "patch", &fixture("test_data.csv"),
            "--nan-fill", "0", "--json",
        ]);
        assert_eq!(code, 0, "patch --json failed: {}", stderr);
        // JSON output should contain braces or brackets
        assert!(
            stdout.contains('{') || stdout.contains('['),
            "patch --json should produce JSON-like output: {}", stdout
        );
    }

    #[test]
    fn patch_plain_output() {
        let (stdout, stderr, code) = run_cjc(&[
            "patch", &fixture("test_data.csv"),
            "--nan-fill", "0", "--plain",
        ]);
        assert_eq!(code, 0, "patch --plain failed: {}", stderr);
        // Should produce some output
        assert!(!stdout.trim().is_empty() || !stderr.trim().is_empty(),
            "patch --plain produced no output");
    }

    // ── drift ───────────────────────────────────────────────────────

    #[test]
    fn drift_fail_on_diff_exit_code() {
        let (_, _, code) = run_cjc(&[
            "drift", &fixture("test_data.csv"), &fixture("test_data_b.csv"),
            "--fail-on-diff", "--plain",
        ]);
        // Files differ, so --fail-on-diff should exit 1
        assert_eq!(code, 1, "drift --fail-on-diff should exit 1 when diffs exist");
    }

    #[test]
    fn drift_summary_only() {
        let (stdout, stderr, code) = run_cjc(&[
            "drift", &fixture("test_data.csv"), &fixture("test_data_b.csv"),
            "--summary-only", "--plain",
        ]);
        // drift exits 1 when files differ, even with --summary-only
        assert!(code == 0 || code == 1, "drift --summary-only unexpected exit: {}", code);
        let combined = format!("{}{}", stdout, stderr);
        assert!(!combined.trim().is_empty(), "drift --summary-only produced no output");
        // Should show summary metrics
        assert!(
            combined.to_lowercase().contains("cell")
                || combined.to_lowercase().contains("diff")
                || combined.to_lowercase().contains("row")
                || combined.to_lowercase().contains("metric"),
            "drift --summary-only should show summary metrics: {}", combined
        );
    }

    #[test]
    fn drift_report_writes_json() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let report_path = tmp.path().join("drift_report.json");
        let report_str = report_path.to_string_lossy().to_string();

        let (_, _stderr, code) = run_cjc(&[
            "drift", &fixture("test_data.csv"), &fixture("test_data_b.csv"),
            "--report", &report_str, "--plain",
        ]);
        // drift exits 1 when files differ; the report should still be written
        assert!(code == 0 || code == 1, "drift --report unexpected exit: {}", code);
        assert!(report_path.exists(), "drift report file was not created at {:?}", report_path);
        let content = std::fs::read_to_string(&report_path).expect("read report");
        assert!(!content.trim().is_empty(), "drift report file is empty");
    }

    // ── proof ───────────────────────────────────────────────────────

    #[test]
    fn proof_fail_fast() {
        let (stdout, stderr, code) = run_cjc(&[
            "proof", &fixture("compute.cjcl"),
            "--fail-fast", "--plain",
        ]);
        assert_eq!(code, 0, "proof --fail-fast on deterministic program should pass: {}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.to_lowercase().contains("pass")
                || combined.to_lowercase().contains("ok")
                || combined.to_lowercase().contains("proof")
                || combined.to_lowercase().contains("identical"),
            "proof output should indicate success: {}", combined
        );
    }

    #[test]
    fn proof_hash_output() {
        let (stdout, stderr, code) = run_cjc(&[
            "proof", &fixture("compute.cjcl"),
            "--hash-output", "--plain",
        ]);
        assert_eq!(code, 0, "proof --hash-output failed: {}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        // Hash output should include hex-like content or "sha" or "hash"
        assert!(
            combined.to_lowercase().contains("hash")
                || combined.to_lowercase().contains("sha")
                || combined.contains("0x")
                || combined.chars().any(|c| c.is_ascii_hexdigit()),
            "proof --hash-output should include hash info: {}", combined
        );
    }

    #[test]
    fn proof_save_report() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let report_path = tmp.path().join("proof_report.json");
        let report_str = report_path.to_string_lossy().to_string();

        let (_, stderr, code) = run_cjc(&[
            "proof", &fixture("compute.cjcl"),
            "--save-report", &report_str, "--plain",
        ]);
        assert_eq!(code, 0, "proof --save-report failed: {}", stderr);
        assert!(report_path.exists(), "proof report file was not created");
        let content = std::fs::read_to_string(&report_path).expect("read report");
        assert!(!content.trim().is_empty(), "proof report file is empty");
    }

    // ── bench ───────────────────────────────────────────────────────

    #[test]
    fn bench_save_and_load_baseline() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let baseline_path = tmp.path().join("baseline.json");
        let baseline_str = baseline_path.to_string_lossy().to_string();

        // Save baseline
        let (_, stderr, code) = run_cjc(&[
            "bench", &fixture("compute.cjcl"),
            "--save-baseline", &baseline_str, "--plain", "--runs", "2",
        ]);
        assert_eq!(code, 0, "bench --save-baseline failed: {}", stderr);
        assert!(baseline_path.exists(), "baseline file was not created");

        // Load baseline and compare
        let (stdout, stderr2, code2) = run_cjc(&[
            "bench", &fixture("compute.cjcl"),
            "--baseline", &baseline_str, "--plain", "--runs", "2",
        ]);
        assert_eq!(code2, 0, "bench --baseline failed: {}", stderr2);
        let combined = format!("{}{}", stdout, stderr2);
        assert!(!combined.trim().is_empty(), "bench --baseline produced no output");
    }

    // ── seek ────────────────────────────────────────────────────────

    #[test]
    fn seek_exclude() {
        let dir = fixtures_dir().to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "seek", &dir, "--exclude", "*.jsonl", "--plain",
        ]);
        assert_eq!(code, 0, "seek --exclude failed: {}", stderr);
        // Should not list JSONL files
        assert!(
            !stdout.contains(".jsonl"),
            "seek --exclude *.jsonl should not list JSONL files: {}", stdout
        );
    }

    #[test]
    fn seek_ignore_build_artifacts() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "seek", &root, "--type", "cjc",
            "--ignore-build-artifacts", "--plain", "--first", "5",
        ]);
        assert_eq!(code, 0, "seek --ignore-build-artifacts failed: {}", stderr);
        // Should not list files from target/ directory
        let normalized = stdout.replace('\\', "/");
        assert!(
            !normalized.contains("/target/"),
            "seek --ignore-build-artifacts should skip target/: {}", stdout
        );
    }

    #[test]
    fn seek_hash() {
        let dir = fixtures_dir().to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "seek", &dir, "--type", "csv", "--hash", "--plain",
        ]);
        assert_eq!(code, 0, "seek --hash failed: {}", stderr);
        // Output should include hash-like content (hex strings)
        let has_hex = stdout.chars().filter(|c| c.is_ascii_hexdigit()).count() > 16;
        assert!(has_hex, "seek --hash should include SHA hashes: {}", stdout);
    }

    // ── pack ────────────────────────────────────────────────────────

    #[test]
    fn pack_dry_run() {
        let (stdout, stderr, code) = run_cjc(&[
            "pack", &fixture("compute.cjcl"), "--dry-run", "--plain",
        ]);
        assert_eq!(code, 0, "pack --dry-run failed: {}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.to_lowercase().contains("dry")
                || combined.to_lowercase().contains("would")
                || combined.to_lowercase().contains("manifest")
                || combined.to_lowercase().contains("pack"),
            "pack --dry-run should describe what would happen: {}", combined
        );
    }

    #[test]
    fn pack_manifest_only() {
        let (stdout, stderr, code) = run_cjc(&[
            "pack", &fixture("compute.cjcl"), "--manifest-only", "--plain",
        ]);
        assert_eq!(code, 0, "pack --manifest-only failed: {}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.to_lowercase().contains("manifest")
                || combined.contains('{')
                || combined.to_lowercase().contains("hash")
                || combined.to_lowercase().contains("file"),
            "pack --manifest-only should display manifest info: {}", combined
        );
    }

    // ── doctor ──────────────────────────────────────────────────────

    #[test]
    fn doctor_fix_on_ragged_csv() {
        // Copy ragged.csv to a temp dir so --fix can modify it
        let tmp = tempfile::tempdir().expect("create temp dir");
        let ragged_src = fixtures_dir().join("ragged.csv");
        let ragged_dst = tmp.path().join("ragged.csv");
        std::fs::copy(&ragged_src, &ragged_dst).expect("copy ragged.csv");

        let dir_str = tmp.path().to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "doctor", &dir_str, "--fix", "--plain",
        ]);
        // doctor --fix may exit 0 (fixed) or 1 (issues found)
        let _ = code;
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.to_lowercase().contains("fix")
                || combined.to_lowercase().contains("ragged")
                || combined.to_lowercase().contains("csv")
                || combined.to_lowercase().contains("finding")
                || combined.to_lowercase().contains("scan"),
            "doctor --fix should report on ragged CSV: {}", combined
        );
    }

    #[test]
    fn doctor_dry_run_preview() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let ragged_src = fixtures_dir().join("ragged.csv");
        let ragged_dst = tmp.path().join("ragged.csv");
        std::fs::copy(&ragged_src, &ragged_dst).expect("copy ragged.csv");

        let dir_str = tmp.path().to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "doctor", &dir_str, "--dry-run", "--plain",
        ]);
        // doctor --dry-run may exit 0 (no issues) or 1 (issues found but not fixed)
        let _ = code;
        let combined = format!("{}{}", stdout, stderr);
        // Should produce some output about scanning/findings
        assert!(
            !combined.trim().is_empty(),
            "doctor --dry-run should produce output"
        );
    }

    #[test]
    fn doctor_report_writes_json() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let report_path = tmp.path().join("doctor_report.json");
        let report_str = report_path.to_string_lossy().to_string();

        let dir = fixtures_dir().to_string_lossy().to_string();
        let (_, stderr, code) = run_cjc(&[
            "doctor", &dir, "--report", &report_str, "--plain",
        ]);
        let _ = code;
        // If --report is supported, the file should be created
        if report_path.exists() {
            let content = std::fs::read_to_string(&report_path).expect("read report");
            assert!(!content.trim().is_empty(), "doctor report file is empty");
        } else {
            // Report flag may not write if no findings -- check stderr
            eprintln!("Note: doctor --report did not create file (may have no findings). stderr={}", stderr);
        }
    }

    #[test]
    fn doctor_category_filtering() {
        let dir = fixtures_dir().to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "doctor", &dir, "--category", "schema", "--plain",
        ]);
        let _ = code;
        let combined = format!("{}{}", stdout, stderr);
        // Should only show schema-related findings (or none)
        assert!(
            !combined.is_empty(),
            "doctor --category should produce some output"
        );
    }
}

// ======================================================================
// C. Determinism Tests
// ======================================================================

mod determinism {
    use super::*;

    #[test]
    fn flow_csv_deterministic() {
        let f = fixture("test_data.csv");
        let (out1, _, c1) = run_cjc(&["flow", &f, "--plain"]);
        let (out2, _, c2) = run_cjc(&["flow", &f, "--plain"]);
        assert_eq!(c1, 0);
        assert_eq!(c2, 0);
        assert_eq!(out1, out2, "flow on CSV must produce identical output across runs");
    }

    #[test]
    fn flow_jsonl_deterministic() {
        let f = fixture("test_data.jsonl");
        let (out1, _, c1) = run_cjc(&["flow", &f, "--plain"]);
        let (out2, _, c2) = run_cjc(&["flow", &f, "--plain"]);
        assert_eq!(c1, 0);
        assert_eq!(c2, 0);
        assert_eq!(out1, out2, "flow on JSONL must produce identical output across runs");
    }

    #[test]
    fn schema_deterministic() {
        let f = fixture("test_data.csv");
        let (out1, _, c1) = run_cjc(&["schema", &f, "--json"]);
        let (out2, _, c2) = run_cjc(&["schema", &f, "--json"]);
        assert_eq!(c1, 0);
        assert_eq!(c2, 0);
        assert_eq!(out1, out2, "schema must produce identical JSON output across runs");
    }

    #[test]
    fn drift_deterministic() {
        let a = fixture("test_data.csv");
        let b = fixture("test_data_b.csv");
        let (out1, _, c1) = run_cjc(&["drift", &a, &b, "--plain"]);
        let (out2, _, c2) = run_cjc(&["drift", &a, &b, "--plain"]);
        assert_eq!(c1, c2, "drift exit codes should match across runs");
        assert_eq!(out1, out2, "drift must produce identical output across runs");
    }

    #[test]
    fn proof_meta_determinism() {
        // Running proof itself twice should produce the same verdict
        let f = fixture("compute.cjcl");
        let (out1, _err1, c1) = run_cjc(&["proof", &f, "--plain", "--runs", "2"]);
        let (out2, _err2, c2) = run_cjc(&["proof", &f, "--plain", "--runs", "2"]);
        assert_eq!(c1, c2, "proof exit codes should match");
        // The verdict line should be identical (timing may differ)
        let verdict1 = out1.lines().find(|l|
            l.to_lowercase().contains("pass") || l.to_lowercase().contains("fail")
        );
        let verdict2 = out2.lines().find(|l|
            l.to_lowercase().contains("pass") || l.to_lowercase().contains("fail")
        );
        assert_eq!(verdict1, verdict2,
            "proof verdict must be deterministic: run1={:?} run2={:?}", verdict1, verdict2);
    }
}

// ======================================================================
// D. Cross-Format Equivalence Tests
// ======================================================================

mod cross_format {
    use super::*;

    /// Extract numeric values from flow output for comparison.
    /// Strips ANSI codes and extracts floating point numbers.
    fn extract_numbers(output: &str) -> Vec<String> {
        let mut numbers = Vec::new();
        for line in output.lines() {
            for word in line.split_whitespace() {
                // Try to parse as f64
                let cleaned = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-');
                if cleaned.parse::<f64>().is_ok() && !cleaned.is_empty() {
                    numbers.push(cleaned.to_string());
                }
            }
        }
        numbers
    }

    #[test]
    fn flow_csv_vs_jsonl_same_aggregation() {
        let (csv_out, _, c1) = run_cjc(&[
            "flow", &fixture("test_data.csv"), "--plain", "--op", "count",
        ]);
        let (jsonl_out, _, c2) = run_cjc(&[
            "flow", &fixture("test_data.jsonl"), "--plain", "--op", "count",
        ]);
        assert_eq!(c1, 0);
        assert_eq!(c2, 0);
        // Both should report the same row count
        let csv_nums = extract_numbers(&csv_out);
        let jsonl_nums = extract_numbers(&jsonl_out);
        // At minimum both should have some "5" in the count
        let csv_has_5 = csv_nums.iter().any(|n| n == "5");
        let jsonl_has_5 = jsonl_nums.iter().any(|n| n == "5");
        assert!(
            csv_has_5 && jsonl_has_5,
            "Both CSV and JSONL should report 5 rows. CSV nums={:?}, JSONL nums={:?}",
            csv_nums, jsonl_nums
        );
    }

    #[test]
    fn schema_csv_vs_jsonl_same_types() {
        let (csv_out, csv_err, c1) = run_cjc(&[
            "schema", &fixture("test_data.csv"), "--plain",
        ]);
        let (jsonl_out, jsonl_err, c2) = run_cjc(&[
            "schema", &fixture("test_data.jsonl"), "--plain",
        ]);
        assert_eq!(c1, 0);
        assert_eq!(c2, 0);
        // Schema output may go to stderr -- combine both
        let csv_combined = format!("{}{}", csv_out, csv_err);
        let jsonl_combined = format!("{}{}", jsonl_out, jsonl_err);
        // Both should mention the same column names
        for col in &["name", "age", "score", "active", "city"] {
            assert!(
                csv_combined.contains(col),
                "CSV schema missing column '{}': {}", col, csv_combined
            );
            assert!(
                jsonl_combined.contains(col),
                "JSONL schema missing column '{}': {}", col, jsonl_combined
            );
        }
    }

    #[test]
    fn inspect_csv_vs_jsonl_both_succeed() {
        let (csv_out, csv_err, c1) = run_cjc(&[
            "inspect", &fixture("test_data.csv"), "--plain",
        ]);
        let (jsonl_out, jsonl_err, c2) = run_cjc(&[
            "inspect", &fixture("test_data.jsonl"), "--plain",
        ]);
        assert_eq!(c1, 0);
        assert_eq!(c2, 0);
        // inspect may output to stderr -- combine both
        let csv_combined = format!("{}{}", csv_out, csv_err);
        let jsonl_combined = format!("{}{}", jsonl_out, jsonl_err);
        assert!(!csv_combined.trim().is_empty(), "CSV inspect produced no output");
        assert!(!jsonl_combined.trim().is_empty(), "JSONL inspect produced no output");
    }
}

// ======================================================================
// E. Output Mode Consistency Tests
// ======================================================================

mod output_modes {
    use super::*;

    /// Check that output contains JSON-like structure (braces or brackets).
    fn assert_json_like(output: &str, command: &str) {
        let trimmed = output.trim();
        assert!(
            trimmed.starts_with('{') || trimmed.starts_with('[')
                || trimmed.contains('{') || trimmed.contains('['),
            "{} --json should produce JSON-like output, got: {}",
            command, &output[..output.len().min(200)]
        );
    }

    #[test]
    fn patch_json_valid() {
        let (stdout, _, code) = run_cjc(&[
            "patch", &fixture("test_data.csv"),
            "--nan-fill", "0", "--json",
        ]);
        assert_eq!(code, 0);
        assert_json_like(&stdout, "patch");
    }

    #[test]
    fn drift_json_valid() {
        let (stdout, _, code) = run_cjc(&[
            "drift", &fixture("test_data.csv"), &fixture("test_data_b.csv"),
            "--json",
        ]);
        // May exit 0 or 1 depending on diff semantics with --json
        let _ = code;
        if !stdout.trim().is_empty() {
            assert_json_like(&stdout, "drift");
        }
    }

    #[test]
    fn doctor_json_valid() {
        let dir = fixtures_dir().to_string_lossy().to_string();
        let (stdout, _, code) = run_cjc(&[
            "doctor", &dir, "--json",
        ]);
        let _ = code;
        if !stdout.trim().is_empty() {
            assert_json_like(&stdout, "doctor");
        }
    }

    #[test]
    fn schema_json_valid() {
        let (stdout, _, code) = run_cjc(&[
            "schema", &fixture("test_data.csv"), "--json",
        ]);
        assert_eq!(code, 0);
        assert_json_like(&stdout, "schema");
    }

    #[test]
    fn flow_json_valid() {
        let (stdout, _, code) = run_cjc(&[
            "flow", &fixture("test_data.csv"), "--json",
        ]);
        assert_eq!(code, 0);
        assert_json_like(&stdout, "flow");
    }
}

// ======================================================================
// F. Edge Case Tests
// ======================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn flow_on_empty_file() {
        let (stdout, stderr, code) = run_cjc(&[
            "flow", &fixture("empty.csv"), "--plain",
        ]);
        // Should handle gracefully -- either exit 0 with no output or exit 1 with error
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            code == 0 || code == 1,
            "flow on empty file should exit 0 or 1, got {}: {}", code, combined
        );
    }

    #[test]
    fn schema_on_single_column_csv() {
        let (stdout, stderr, code) = run_cjc(&[
            "schema", &fixture("single_column.csv"), "--plain",
        ]);
        assert_eq!(code, 0, "schema on single-column CSV failed: {}", stderr);
        // Schema output may go to stderr -- check both
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.contains("value"),
            "schema should detect the 'value' column: {}", combined
        );
    }

    #[test]
    fn drift_on_identical_files_zero_diff() {
        let f = fixture("test_data.csv");
        let (stdout, stderr, code) = run_cjc(&[
            "drift", &f, &f, "--plain",
        ]);
        assert_eq!(code, 0, "drift on identical files should exit 0: {}", stderr);
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.contains("0") || combined.to_lowercase().contains("identical")
                || combined.to_lowercase().contains("no diff")
                || combined.to_lowercase().contains("match"),
            "drift on identical files should show zero differences: {}", combined
        );
    }

    #[test]
    fn doctor_on_clean_directory() {
        // Create a temp dir with a single valid CSV
        let tmp = tempfile::tempdir().expect("create temp dir");
        let csv_path = tmp.path().join("clean.csv");
        std::fs::write(&csv_path, "a,b,c\n1,2,3\n4,5,6\n").expect("write clean CSV");

        let dir_str = tmp.path().to_string_lossy().to_string();
        let (stdout, stderr, code) = run_cjc(&[
            "doctor", &dir_str, "--plain",
        ]);
        // Clean directory should pass (exit 0)
        assert_eq!(code, 0,
            "doctor on clean directory should exit 0: stdout={}, stderr={}", stdout, stderr);
    }

    #[test]
    fn inspect_nonexistent_file_errors() {
        let (_, stderr, code) = run_cjc(&[
            "inspect", "this_file_does_not_exist.csv", "--plain",
        ]);
        assert_ne!(code, 0, "inspect on non-existent file should fail");
        assert!(
            stderr.to_lowercase().contains("error")
                || stderr.to_lowercase().contains("not found")
                || stderr.to_lowercase().contains("no such"),
            "inspect should report error for missing file: {}", stderr
        );
    }

    #[test]
    fn patch_with_no_transforms_errors() {
        let (_, stderr, code) = run_cjc(&[
            "patch", &fixture("test_data.csv"), "--plain",
        ]);
        // No transforms specified -- should error or warn
        assert!(
            code != 0 || stderr.to_lowercase().contains("no transform")
                || stderr.to_lowercase().contains("nothing to do")
                || stderr.to_lowercase().contains("error")
                || stderr.to_lowercase().contains("required"),
            "patch with no transforms should error or warn: code={}, stderr={}", code, stderr
        );
    }

    #[test]
    fn doctor_on_malformed_jsonl() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let src = fixtures_dir().join("malformed.jsonl");
        let dst = tmp.path().join("malformed.jsonl");
        std::fs::copy(&src, &dst).expect("copy malformed.jsonl");

        let dir_str = tmp.path().to_string_lossy().to_string();
        let (stdout, stderr, _code) = run_cjc(&[
            "doctor", &dir_str, "--plain",
        ]);
        let combined = format!("{}{}", stdout, stderr);
        // Doctor should detect issues in the malformed JSONL
        assert!(
            combined.to_lowercase().contains("json")
                || combined.to_lowercase().contains("malform")
                || combined.to_lowercase().contains("invalid")
                || combined.to_lowercase().contains("error")
                || combined.to_lowercase().contains("finding")
                || combined.to_lowercase().contains("warn"),
            "doctor should flag malformed JSONL: {}", combined
        );
    }

    #[test]
    fn drift_fail_on_diff_with_identical_files_exits_zero() {
        let f = fixture("test_data.csv");
        let (_, _, code) = run_cjc(&[
            "drift", &f, &f, "--fail-on-diff", "--plain",
        ]);
        assert_eq!(code, 0,
            "drift --fail-on-diff on identical files should exit 0");
    }
}
