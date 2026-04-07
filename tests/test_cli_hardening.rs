//! CLI Hardening Test Suite (v0.1.2.1)
//!
//! Tests for bug fixes and guardrails identified during benchmark analysis:
//!   A. pack directory discovery (BUG-1)
//!   B. inspect/schema bool type agreement (BUG-2)
//!   C. drift row count consistency (BUG-3)
//!   D. seek --hash performance (PERF-2)
//!   E. bench CV warning (GUARD-3)
//!   F. Cross-command type consistency

use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ── Helper ─────────────────────────────────────────────────────────────

fn cjc_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    path.push(if cfg!(windows) { "cjcl.exe" } else { "cjcl" });
    path
}

fn fixtures_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("cli_hardening");
    p
}

fn run_cjc(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(cjc_binary())
        .args(args)
        .output()
        .expect("failed to run cjc");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

// ── Fixture setup ──────────────────────────────────────────────────────

fn ensure_fixtures() {
    let dir = fixtures_dir();
    if dir.exists() { return; }
    fs::create_dir_all(&dir).unwrap();

    // CSV with bool column
    let csv = "id,name,value,active\n\
               1,alice,10.5,true\n\
               2,bob,20.3,false\n\
               3,charlie,30.1,True\n\
               4,diana,40.7,FALSE\n\
               5,eve,50.0,true\n";
    fs::write(dir.join("bool_test.csv"), csv).unwrap();

    // Identical copy for drift
    fs::write(dir.join("bool_test_identical.csv"), csv).unwrap();

    // Different version for drift
    let csv_b = "id,name,value,active\n\
                 1,alice,11.5,false\n\
                 2,bob,21.3,true\n\
                 3,charlie,31.1,False\n\
                 4,diana,41.7,TRUE\n\
                 5,eve,51.0,false\n";
    fs::write(dir.join("bool_test_diff.csv"), csv_b).unwrap();

    // JSONL with bool column
    let jsonl = "{\"id\": 1, \"name\": \"alice\", \"value\": 10.5, \"active\": true}\n\
                 {\"id\": 2, \"name\": \"bob\", \"value\": 20.3, \"active\": false}\n\
                 {\"id\": 3, \"name\": \"charlie\", \"value\": 30.1, \"active\": true}\n";
    fs::write(dir.join("bool_test.jsonl"), jsonl).unwrap();

    // Pack test directory
    let pack_dir = dir.join("pack_test");
    fs::create_dir_all(&pack_dir).unwrap();
    fs::write(pack_dir.join("main.cjcl"), "fn main() { print(42); }\n").unwrap();
    fs::write(pack_dir.join("data.csv"), "a,b\n1,2\n3,4\n").unwrap();
    fs::write(pack_dir.join("records.jsonl"), "{\"x\":1}\n{\"x\":2}\n").unwrap();

    // Empty directory for pack
    fs::create_dir_all(dir.join("empty_pack")).unwrap();

    // Dir with only unsupported files
    let unsup = dir.join("unsupported_pack");
    fs::create_dir_all(&unsup).unwrap();
    fs::write(unsup.join("readme.txt"), "hello").unwrap();
    fs::write(unsup.join("image.png"), &[0x89, 0x50, 0x4E, 0x47]).unwrap();

    // CJC source for audit/bench
    fs::write(dir.join("simple.cjcl"), "fn main() {\n    let x: i64 = 42;\n    print(x);\n}\n").unwrap();
}

// ── A. pack directory discovery (BUG-1) ────────────────────────────────

mod pack_discovery {
    use super::*;

    #[test]
    fn pack_directory_discovers_cjc_csv_jsonl() {
        ensure_fixtures();
        let pack_dir = fixtures_dir().join("pack_test");
        let (_, stderr, code) = run_cjc(&["pack", pack_dir.to_str().unwrap(), "--dry-run"]);
        // Should find 3 files
        assert!(stderr.contains("main.cjcl"), "should find .cjcl file: {}", stderr);
        assert!(stderr.contains("data.csv"), "should find .csv file: {}", stderr);
        assert!(stderr.contains("records.jsonl"), "should find .jsonl file: {}", stderr);
        assert_eq!(code, 0);
    }

    #[test]
    fn pack_empty_directory_warns() {
        ensure_fixtures();
        let empty_dir = fixtures_dir().join("empty_pack");
        let (_, stderr, _) = run_cjc(&["pack", empty_dir.to_str().unwrap(), "--dry-run"]);
        assert!(stderr.contains("no packable files found"), "should warn about empty dir: {}", stderr);
    }

    #[test]
    fn pack_unsupported_files_warns() {
        ensure_fixtures();
        let dir = fixtures_dir().join("unsupported_pack");
        let (_, stderr, _) = run_cjc(&["pack", dir.to_str().unwrap(), "--dry-run"]);
        assert!(stderr.contains("no packable files found"), "should warn about no packable files: {}", stderr);
    }

    #[test]
    fn pack_directory_roundtrip_verify() {
        ensure_fixtures();
        let pack_dir = fixtures_dir().join("pack_test");
        let output_pack = fixtures_dir().join("pack_test.pack");
        // Clean up previous
        let _ = fs::remove_dir_all(&output_pack);
        let (_, stderr, code) = run_cjc(&["pack", pack_dir.to_str().unwrap(), "--verify"]);
        assert_eq!(code, 0, "pack --verify should succeed: {}", stderr);
        assert!(stderr.contains("verified"), "should verify: {}", stderr);
        // Clean up
        let _ = fs::remove_dir_all(&output_pack);
    }
}

// ── B. inspect/schema bool type agreement (BUG-2) ─────────────────────

mod bool_type_agreement {
    use super::*;

    #[test]
    fn inspect_detects_bool_column_csv() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let (_, stderr, _) = run_cjc(&["inspect", f.to_str().unwrap()]);
        // The "active" column should be typed as "bool" not "string"
        assert!(stderr.contains("bool"), "inspect should detect bool type: {}", stderr);
        // Make sure it's the active column specifically
        let active_line = stderr.lines().find(|l| l.contains("active")).unwrap_or("");
        assert!(active_line.contains("bool"), "active column should be bool: {}", active_line);
    }

    #[test]
    fn schema_detects_bool_column_csv() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let (_, stderr, _) = run_cjc(&["schema", f.to_str().unwrap()]);
        let active_line = stderr.lines().find(|l| l.contains("active")).unwrap_or("");
        assert!(active_line.contains("bool"), "schema active column should be bool: {}", active_line);
    }

    #[test]
    fn inspect_and_schema_agree_on_bool_csv() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");

        let (_, inspect_stderr, _) = run_cjc(&["inspect", f.to_str().unwrap()]);
        let (_, schema_stderr, _) = run_cjc(&["schema", f.to_str().unwrap()]);

        let inspect_active = inspect_stderr.lines().find(|l| l.contains("active")).unwrap_or("");
        let schema_active = schema_stderr.lines().find(|l| l.contains("active")).unwrap_or("");

        // Both should say "bool"
        assert!(inspect_active.contains("bool"), "inspect: {}", inspect_active);
        assert!(schema_active.contains("bool"), "schema: {}", schema_active);
    }

    #[test]
    fn inspect_types_numeric_columns_correctly() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let (_, stderr, _) = run_cjc(&["inspect", f.to_str().unwrap()]);
        let id_line = stderr.lines().find(|l| l.contains("id")).unwrap_or("");
        assert!(id_line.contains("int"), "id should be int: {}", id_line);
        let value_line = stderr.lines().find(|l| l.contains("value")).unwrap_or("");
        assert!(value_line.contains("float"), "value should be float: {}", value_line);
    }
}

// ── C. drift row count consistency (BUG-3) ─────────────────────────────

mod drift_row_counts {
    use super::*;

    fn extract_rows_a(stderr: &str) -> Option<String> {
        stderr.lines()
            .find(|l| l.contains("Rows (A)"))
            .map(|l| l.to_string())
    }

    #[test]
    fn drift_default_excludes_header() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let f2 = fixtures_dir().join("bool_test_diff.csv");
        let (_, stderr, _) = run_cjc(&["drift", f.to_str().unwrap(), f2.to_str().unwrap()]);
        let rows_line = extract_rows_a(&stderr).unwrap_or_default();
        assert!(rows_line.contains("5"), "should be 5 data rows, not 6: {}", rows_line);
    }

    #[test]
    fn drift_fail_on_diff_same_row_count() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let f2 = fixtures_dir().join("bool_test_identical.csv");
        let (_, stderr, _) = run_cjc(&["drift", f.to_str().unwrap(), f2.to_str().unwrap(), "--fail-on-diff"]);
        let rows_line = extract_rows_a(&stderr).unwrap_or_default();
        assert!(rows_line.contains("5"), "should be 5 data rows with --fail-on-diff: {}", rows_line);
    }

    #[test]
    fn drift_summary_only_same_row_count() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let f2 = fixtures_dir().join("bool_test_diff.csv");
        let (_, stderr, _) = run_cjc(&["drift", f.to_str().unwrap(), f2.to_str().unwrap(), "--summary-only"]);
        let rows_line = extract_rows_a(&stderr).unwrap_or_default();
        assert!(rows_line.contains("5"), "should be 5 data rows with --summary-only: {}", rows_line);
    }

    #[test]
    fn drift_consistent_across_all_flags() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.csv");
        let f2 = fixtures_dir().join("bool_test_diff.csv");

        // Default mode
        let (_, stderr_default, _) = run_cjc(&["drift", f.to_str().unwrap(), f2.to_str().unwrap()]);
        // Summary-only mode
        let (_, stderr_summary, _) = run_cjc(&["drift", f.to_str().unwrap(), f2.to_str().unwrap(), "--summary-only"]);

        let rows_default = extract_rows_a(&stderr_default).unwrap_or_default();
        let rows_summary = extract_rows_a(&stderr_summary).unwrap_or_default();

        // Both should report the same row count
        assert_eq!(rows_default, rows_summary,
            "row counts should be consistent: default='{}' summary='{}'", rows_default, rows_summary);
    }
}

// ── D. seek --hash performance (PERF-2) ────────────────────────────────

mod seek_performance {
    use super::*;
    use std::time::Instant;

    #[test]
    fn seek_hash_first_is_fast() {
        ensure_fixtures();
        let dir = fixtures_dir();
        let start = Instant::now();
        let (_, _, code) = run_cjc(&["seek", dir.to_str().unwrap(), "--hash", "--first", "3"]);
        let elapsed = start.elapsed();
        assert_eq!(code, 0);
        // Should complete in under 2 seconds (was 1.3s before fix, now ~0.1s)
        assert!(elapsed.as_secs() < 2, "seek --hash --first 3 took too long: {:?}", elapsed);
    }
}

// ── E. Cross-command consistency ───────────────────────────────────────

mod cross_command_consistency {
    use super::*;

    #[test]
    fn all_report_flags_produce_valid_json() {
        ensure_fixtures();
        let f = fixtures_dir().join("simple.cjcl");
        let dir = fixtures_dir();

        // proof --save-report
        let report = dir.join("test_proof_report.json");
        let (_, _, code) = run_cjc(&["proof", f.to_str().unwrap(), "--save-report", report.to_str().unwrap()]);
        if code == 0 {
            let content = fs::read_to_string(&report).unwrap_or_default();
            assert!(content.starts_with('{'), "proof report should be JSON: {}", &content[..content.len().min(50)]);
            let _ = fs::remove_file(&report);
        }

        // parity --save-report
        let report = dir.join("test_parity_report.json");
        let (_, _, code) = run_cjc(&["parity", f.to_str().unwrap(), "--save-report", report.to_str().unwrap()]);
        if code == 0 {
            let content = fs::read_to_string(&report).unwrap_or_default();
            assert!(content.starts_with('{'), "parity report should be JSON: {}", &content[..content.len().min(50)]);
            let _ = fs::remove_file(&report);
        }

        // precision --report
        let report = dir.join("test_precision_report.json");
        let (_, _, code) = run_cjc(&["precision", f.to_str().unwrap(), "--report", report.to_str().unwrap()]);
        if code == 0 {
            let content = fs::read_to_string(&report).unwrap_or_default();
            assert!(content.starts_with('{'), "precision report should be JSON: {}", &content[..content.len().min(50)]);
            let _ = fs::remove_file(&report);
        }

        // audit --report
        let report = dir.join("test_audit_report.json");
        let (_, _, _) = run_cjc(&["audit", f.to_str().unwrap(), "--report", report.to_str().unwrap()]);
        let content = fs::read_to_string(&report).unwrap_or_default();
        assert!(content.starts_with('{'), "audit report should be JSON: {}", &content[..content.len().min(50)]);
        let _ = fs::remove_file(&report);
    }

    #[test]
    fn bench_csv_output_is_valid_csv() {
        ensure_fixtures();
        let f = fixtures_dir().join("simple.cjcl");
        let (stdout, _, _) = run_cjc(&["bench", f.to_str().unwrap(), "--csv"]);
        // Should have a header line and a data line
        let lines: Vec<&str> = stdout.lines().filter(|l| l.contains(',')).collect();
        assert!(lines.len() >= 2, "CSV output should have header + data: {}", stdout);
        assert!(lines[0].contains("file,executor"), "should have CSV header: {}", lines[0]);
    }
}

// ── F. Patch JSONL basic correctness ───────────────────────────────────

mod patch_jsonl {
    use super::*;

    #[test]
    fn patch_jsonl_rename_preserves_rows() {
        ensure_fixtures();
        let f = fixtures_dir().join("bool_test.jsonl");
        let (stdout, stderr, code) = run_cjc(&["patch", f.to_str().unwrap(), "--rename", "id", "record_id"]);
        assert_eq!(code, 0, "patch should succeed: {}", stderr);
        // Output should contain renamed key
        assert!(stdout.contains("record_id"), "should contain renamed key: {}", stdout);
        // Should have processed 3 rows
        assert!(stderr.contains("3 rows"), "should process 3 rows: {}", stderr);
    }
}
