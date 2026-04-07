//! CJC End-to-End Fixture Runner
//!
//! This test runner discovers `.cjcl` source files under `tests/fixtures/`,
//! compiles and executes each one via the MIR executor, and compares the
//! captured output against golden `.stdout` files.
//!
//! ## Directory layout
//!
//! ```text
//! tests/fixtures/
//!   <category>/
//!     <name>.cjcl       — CJC source file
//!     <name>.stdout     — expected stdout lines (one per line)
//!     <name>.stderr     — (optional) expected error substring
//!     <name>.exitcode   — (optional) expected exit code, default 0
//! ```
//!
//! ## Updating goldens
//!
//! Set the environment variable `CJC_FIXTURE_UPDATE=1` before running to
//! overwrite golden files with actual output.
//!
//! ## Running
//!
//! ```bash
//! cargo test --test fixtures
//! ```

use std::fs;
use std::path::{Path, PathBuf};

/// Collect all `.cjcl` files under `dir` recursively.
fn discover_fixtures(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if !dir.is_dir() {
        return result;
    }
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            result.extend(discover_fixtures(&path));
        } else if path.extension().map(|e| e == "cjcl").unwrap_or(false) {
            result.push(path);
        }
    }
    result.sort();
    result
}

/// Run a single fixture: parse + execute, return (stdout_lines, result_or_error).
///
/// When `expect_error` is true, we run the type-checked pipeline so that
/// compile-time errors are caught (the default pipeline skips type checking).
fn run_fixture(source: &str, expect_error: bool) -> (Vec<String>, Result<String, String>) {
    let (program, diags) = cjc_parser::parse_source(source);

    // If there are parse errors, return them as error output
    if diags.has_errors() {
        let err_msgs: Vec<String> = diags
            .diagnostics
            .iter()
            .filter(|d| d.severity == cjc_diag::Severity::Error)
            .map(|d| d.message.clone())
            .collect();
        return (vec![], Err(err_msgs.join("\n")));
    }

    if expect_error {
        // Use type-checked pipeline to detect compile-time errors
        match cjc_mir_exec::run_program_type_checked(&program, 42) {
            Ok(value) => (vec![], Ok(format!("{:?}", value))),
            Err(cjc_mir_exec::MirExecError::TypeErrors(errs)) => {
                (vec![], Err(errs.join("\n")))
            }
            Err(cjc_mir_exec::MirExecError::Runtime(msg)) => {
                (vec![], Err(format!("runtime error: {}", msg)))
            }
            Err(cjc_mir_exec::MirExecError::RuntimeError(e)) => {
                (vec![], Err(format!("runtime error: {}", e)))
            }
            Err(e) => {
                (vec![], Err(format!("{}", e)))
            }
        }
    } else {
        match cjc_mir_exec::run_program_with_executor(&program, 42) {
            Ok((value, executor)) => {
                let stdout = executor.output.clone();
                (stdout, Ok(format!("{:?}", value)))
            }
            Err(cjc_mir_exec::MirExecError::TypeErrors(errs)) => {
                (vec![], Err(errs.join("\n")))
            }
            Err(cjc_mir_exec::MirExecError::Runtime(msg)) => {
                (vec![], Err(format!("runtime error: {}", msg)))
            }
            Err(cjc_mir_exec::MirExecError::RuntimeError(e)) => {
                (vec![], Err(format!("runtime error: {}", e)))
            }
            Err(e) => {
                (vec![], Err(format!("{}", e)))
            }
        }
    }
}

/// Pretty diff between expected and actual lines.
fn diff_lines(expected: &str, actual: &str) -> String {
    let exp_lines: Vec<&str> = expected.lines().collect();
    let act_lines: Vec<&str> = actual.lines().collect();
    let mut diff = String::new();
    let max = exp_lines.len().max(act_lines.len());
    for i in 0..max {
        let e = exp_lines.get(i).unwrap_or(&"<missing>");
        let a = act_lines.get(i).unwrap_or(&"<missing>");
        if e != a {
            diff.push_str(&format!("  line {}: expected {:?}\n", i + 1, e));
            diff.push_str(&format!("  line {}:   actual {:?}\n", i + 1, a));
        }
    }
    if exp_lines.len() != act_lines.len() {
        diff.push_str(&format!(
            "  (expected {} lines, got {} lines)\n",
            exp_lines.len(),
            act_lines.len()
        ));
    }
    diff
}

#[test]
fn run_all_fixtures() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    let fixtures = discover_fixtures(&fixture_dir);

    if fixtures.is_empty() {
        panic!("No .cjcl fixture files found under {:?}", fixture_dir);
    }

    let update_mode = std::env::var("CJC_FIXTURE_UPDATE").unwrap_or_default() == "1";

    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();

    for fixture_path in &fixtures {
        let source = fs::read_to_string(fixture_path).unwrap();
        let stem = fixture_path.with_extension("");

        let stdout_golden = stem.with_extension("stdout");
        let stderr_golden = stem.with_extension("stderr");

        let expect_error = stderr_golden.exists();
        let (stdout_lines, result) = run_fixture(&source, expect_error);
        let actual_stdout = stdout_lines.join("\n");
        // Add trailing newline if there's content, to match golden file format
        let actual_stdout_with_nl = if actual_stdout.is_empty() {
            actual_stdout.clone()
        } else {
            format!("{}\n", actual_stdout)
        };

        // Check if this is an error fixture
        if stderr_golden.exists() {
            let expected_err = fs::read_to_string(&stderr_golden).unwrap();
            match result {
                Err(ref err_msg) => {
                    // Check that each expected error substring appears in the actual error
                    let mut ok = true;
                    for line in expected_err.lines() {
                        let line = line.trim();
                        if !line.is_empty() && !err_msg.contains(line) {
                            ok = false;
                            failures.push(format!(
                                "FAIL {}: expected error containing {:?}, got {:?}",
                                fixture_path.display(),
                                line,
                                err_msg
                            ));
                        }
                    }
                    if ok {
                        passed += 1;
                    } else {
                        failed += 1;
                    }
                }
                Ok(_) => {
                    failed += 1;
                    failures.push(format!(
                        "FAIL {}: expected error but got success",
                        fixture_path.display()
                    ));
                }
            }
            continue;
        }

        // Normal (success) fixture — compare stdout
        if stdout_golden.exists() {
            // Normalize CRLF → LF for cross-platform compatibility
            let expected = fs::read_to_string(&stdout_golden)
                .unwrap()
                .replace("\r\n", "\n");
            if actual_stdout_with_nl == expected {
                passed += 1;
            } else if update_mode {
                fs::write(&stdout_golden, &actual_stdout_with_nl).unwrap();
                eprintln!("UPDATED: {}", stdout_golden.display());
                passed += 1;
            } else {
                failed += 1;
                let d = diff_lines(&expected, &actual_stdout_with_nl);
                failures.push(format!(
                    "FAIL {}\n{}",
                    fixture_path.display(),
                    d
                ));
            }
        } else if update_mode {
            // Golden doesn't exist yet — create it
            fs::write(&stdout_golden, &actual_stdout_with_nl).unwrap();
            eprintln!("CREATED: {}", stdout_golden.display());
            passed += 1;
        } else {
            // No golden file and not in update mode: run and check at least no crash
            match result {
                Ok(_) => {
                    passed += 1;
                    eprintln!(
                        "WARN {}: no golden file, ran successfully (use CJC_FIXTURE_UPDATE=1 to create)",
                        fixture_path.display()
                    );
                }
                Err(e) => {
                    failed += 1;
                    failures.push(format!(
                        "FAIL {} (no golden, errored): {}",
                        fixture_path.display(),
                        e
                    ));
                }
            }
        }
    }

    eprintln!(
        "\n=== Fixture Summary: {} passed, {} failed, {} total ===",
        passed, failed, fixtures.len()
    );

    if !failures.is_empty() {
        panic!(
            "\n{} fixture(s) failed:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}
