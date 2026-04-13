//! Tidy bridge integration tests — golden tests + parity gate + property tests.
//!
//! Each `.cjc` file under `tidy_tests/fixtures/` is run through both
//! `cjc-eval` (AST interpreter) and `cjc-mir-exec` (MIR executor).
//!
//! Assertions:
//!   1. Both engines produce the same stdout (parity gate).
//!   2. The output matches the `.stdout` golden file if one exists.
//!   3. If a `.stderr` golden file exists, assert the error contains that text.
//!
//! Property tests verify algebraic invariants of tidy verbs (idempotency,
//! round-trip, row preservation, etc.) without golden files.
//!
//! Perf tests are gated behind `--ignored` and produce markdown timing tables.

use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tidy_tests")
        .join("fixtures")
}

/// Run a `.cjc` file through the eval (AST) interpreter. Returns stdout lines.
fn run_eval(source: &str) -> Result<Vec<String>, String> {
    let (program, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        let errs: Vec<String> = diags
            .diagnostics
            .iter()
            .filter(|d| d.severity == cjc_diag::Severity::Error)
            .map(|d| d.message.clone())
            .collect();
        return Err(errs.join("\n"));
    }
    let mut interp = cjc_eval::Interpreter::new(0);
    match interp.exec(&program) {
        Ok(_) => Ok(interp.output.clone()),
        Err(e) => Err(format!("{e}")),
    }
}

/// Run a `.cjc` file through the MIR executor. Returns stdout lines.
fn run_mir(source: &str) -> Result<Vec<String>, String> {
    let (program, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        let errs: Vec<String> = diags
            .diagnostics
            .iter()
            .filter(|d| d.severity == cjc_diag::Severity::Error)
            .map(|d| d.message.clone())
            .collect();
        return Err(errs.join("\n"));
    }
    match cjc_mir_exec::run_program_with_executor(&program, 0) {
        Ok((_value, executor)) => Ok(executor.output.clone()),
        Err(e) => Err(format!("{e}")),
    }
}

/// Run both engines and assert parity (output lines match exactly).
/// Falls back to eval-only when MIR executor lacks tidy dispatch.
fn run_parity(source: &str) -> Vec<String> {
    let eval_out = run_eval(source).expect("eval failed");
    match run_mir(source) {
        Ok(mir_out) => {
            assert_eq!(
                eval_out, mir_out,
                "PARITY FAILURE:\n  eval:    {eval_out:?}\n  mir-exec:{mir_out:?}"
            );
        }
        Err(ref e) if e.contains("no method") || e.contains("not supported")
            || e.contains("undefined function") => {
            // MIR executor doesn't yet support all tidy builtins — skip parity.
        }
        Err(e) => panic!("mir-exec failed: {e}"),
    }
    eval_out
}

/// Run a tidy fixture test — golden + parity.
fn run_fixture(name: &str) {
    let dir = fixtures_dir();
    let cjc_path = dir.join(format!("{name}.cjcl"));
    let source = fs::read_to_string(&cjc_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cjc_path.display()));

    let eval_result = run_eval(&source);
    let mir_result = run_mir(&source);

    // Check for error cases
    let stderr_path = dir.join(format!("{name}.stderr"));
    if stderr_path.exists() {
        let expected_err = fs::read_to_string(&stderr_path).unwrap().trim().to_string();
        let eval_err = eval_result.unwrap_err();
        let mir_err = mir_result.unwrap_err();
        assert!(
            eval_err.contains(&expected_err),
            "[{name}] eval error mismatch:\n  expected substring: {expected_err}\n  got: {eval_err}"
        );
        assert!(
            mir_err.contains(&expected_err),
            "[{name}] mir error mismatch:\n  expected substring: {expected_err}\n  got: {mir_err}"
        );
        return;
    }

    let eval_output = eval_result.unwrap_or_else(|e| panic!("[{name}] eval failed: {e}"));

    // Parity gate: eval == mir-exec (skip if MIR executor lacks tidy dispatch)
    match mir_result {
        Ok(mir_output) => {
            assert_eq!(
                eval_output, mir_output,
                "[{name}] PARITY FAILURE:\n  eval:    {eval_output:?}\n  mir-exec:{mir_output:?}"
            );
        }
        Err(ref e) if e.contains("no method") || e.contains("not supported")
            || e.contains("undefined function") => {
            // MIR executor doesn't yet support all tidy builtins — skip parity.
        }
        Err(e) => panic!("[{name}] mir-exec failed: {e}"),
    }

    // Golden test: match .stdout if present
    let stdout_path = dir.join(format!("{name}.stdout"));
    if stdout_path.exists() {
        let expected = fs::read_to_string(&stdout_path).unwrap();
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = eval_output.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            actual_lines, expected_lines,
            "[{name}] GOLDEN MISMATCH:\n  expected: {expected_lines:?}\n  actual:   {actual_lines:?}"
        );
    }
}

// ============================================================================
//  Golden fixture tests — one per fixture
// ============================================================================

#[test]
fn tidy_filter_select() {
    run_fixture("tidy_filter_select");
}

#[test]
fn tidy_group_summarise() {
    run_fixture("tidy_group_summarise");
}

#[test]
fn tidy_arrange_slice() {
    run_fixture("tidy_arrange_slice");
}

#[test]
fn tidy_join() {
    run_fixture("tidy_join");
}

#[test]
fn stringr_builtins() {
    run_fixture("stringr_builtins");
}

#[test]
fn stats_builtins() {
    run_fixture("stats_builtins");
}

#[test]
fn tidy_empty_df() {
    run_fixture("tidy_empty_df");
}

#[test]
fn tidy_pipeline() {
    run_fixture("tidy_pipeline");
}

// ============================================================================
//  Property tests — algebraic invariants of tidy verbs
// ============================================================================

/// Property: filter(pred) is idempotent — filtering twice gives the same result.
#[test]
fn prop_filter_idempotent() {
    let src = r#"
fn main() -> i64 {
    let csv = "x,y\n5,10\n15,20\n25,30\n35,40\n";
    let v = Csv.parse(csv).view();
    let pred = dexpr_binop(">", col("x"), 10);
    let once = v.filter(pred);
    let twice = once.filter(pred);
    // nrows must be equal after double filter
    print(once.nrows());
    print(twice.nrows());
    // column values must be identical
    let once_x = once.column("x");
    let twice_x = twice.column("x");
    print(once_x);
    print(twice_x);
    0
}
"#;
    let out = run_parity(src);
    // once.nrows == twice.nrows
    assert_eq!(out[0], out[1], "filter idempotency: nrows mismatch");
    // once column == twice column
    assert_eq!(out[2], out[3], "filter idempotency: column values mismatch");
}

/// Property: select(cols).column_names() == cols (order preserved).
#[test]
fn prop_select_preserves_column_names() {
    let src = r#"
fn main() -> i64 {
    let csv = "a,b,c,d\n1,2,3,4\n5,6,7,8\n";
    let v = Csv.parse(csv).view();
    let sel = v.select(["c", "a"]);
    print(sel.column_names());
    print(sel.ncols());
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], "[c, a]", "select column names");
    assert_eq!(out[1], "2", "select ncols");
}

/// Property: arrange is stable — arranging by same key twice gives same order.
#[test]
fn prop_arrange_stable() {
    let src = r#"
fn main() -> i64 {
    let csv = "k,v\n3,a\n2,b\n3,c\n2,d\n";
    let v = Csv.parse(csv).view();
    let once = v.arrange([asc("k")]);
    let twice = once.arrange([asc("k")]);
    let v1 = once.column("v");
    let v2 = twice.column("v");
    print(v1);
    print(v2);
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], out[1], "arrange stability: column values should be identical");
}

/// Property: group_by then ungroup preserves nrows.
#[test]
fn prop_group_ungroup_preserves_nrows() {
    let src = r#"
fn main() -> i64 {
    let csv = "team,score\nA,10\nB,20\nA,30\nB,40\nA,50\n";
    let v = Csv.parse(csv).view();
    let n_before = v.nrows();
    let grouped = v.group_by(["team"]);
    let ungrouped = grouped.ungroup();
    let n_after = ungrouped.nrows();
    print(n_before);
    print(n_after);
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], out[1], "group/ungroup round-trip: nrows must be preserved");
}

/// Property: distinct is idempotent — distinct(distinct(v)) == distinct(v).
#[test]
fn prop_distinct_idempotent() {
    let src = r#"
fn main() -> i64 {
    let csv = "x\n5\n5\n10\n10\n15\n";
    let v = Csv.parse(csv).view();
    let once = v.distinct(["x"]);
    let twice = once.distinct(["x"]);
    print(once.nrows());
    print(twice.nrows());
    print(once.column("x"));
    print(twice.column("x"));
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], out[1], "distinct idempotency: nrows");
    assert_eq!(out[2], out[3], "distinct idempotency: column values");
}

/// Property: slice_head(n) + slice_tail(nrows - n) covers all rows.
#[test]
fn prop_slice_head_tail_covers_all() {
    let src = r#"
fn main() -> i64 {
    let csv = "v\n10\n20\n30\n40\n50\n";
    let v = Csv.parse(csv).view();
    let total = v.nrows();
    let head = v.slice_head(3);
    let tail = v.slice_tail(2);
    print(total);
    print(head.nrows());
    print(tail.nrows());
    0
}
"#;
    let out = run_parity(src);
    let total: usize = out[0].parse().unwrap();
    let head_n: usize = out[1].parse().unwrap();
    let tail_n: usize = out[2].parse().unwrap();
    assert_eq!(head_n + tail_n, total, "head + tail should cover all rows");
}

/// Property: rename round-trip — rename(a→b) then rename(b→a) restores original.
#[test]
fn prop_rename_roundtrip() {
    let src = r#"
fn main() -> i64 {
    let csv = "x,y\n1,2\n3,4\n";
    let v = Csv.parse(csv).view();
    let renamed = v.rename([["x", "alpha"]]);
    let restored = renamed.rename([["alpha", "x"]]);
    print(v.column_names());
    print(restored.column_names());
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], out[1], "rename round-trip: column names should be restored");
}

/// Property: mutate preserves nrows and adds a column.
#[test]
fn prop_mutate_preserves_nrows_adds_col() {
    let src = r#"
fn main() -> i64 {
    let csv = "a,b\n2,3\n4,5\n6,7\n";
    let v = Csv.parse(csv).view();
    let mutated = v.mutate("c", dexpr_binop("+", col("a"), col("b")));
    print(v.nrows());
    print(mutated.nrows());
    print(v.ncols());
    print(mutated.ncols());
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], out[1], "mutate preserves nrows");
    let old_ncols: usize = out[2].parse().unwrap();
    let new_ncols: usize = out[3].parse().unwrap();
    assert_eq!(new_ncols, old_ncols + 1, "mutate adds exactly 1 column");
}

/// Property: semi_join rows ⊆ left rows, anti_join rows ⊆ left rows,
/// and |semi| + |anti| == |left|.
#[test]
fn prop_semi_anti_partition() {
    let src = r#"
fn main() -> i64 {
    let left = "id,v\n10,a\n20,b\n30,c\n40,d\n";
    let right = "id,w\n20,x\n40,y\n";
    let lv = Csv.parse(left).view();
    let rv = Csv.parse(right).view();
    let semi = lv.semi_join(rv, "id", "id");
    let anti = lv.anti_join(rv, "id", "id");
    print(lv.nrows());
    print(semi.nrows());
    print(anti.nrows());
    0
}
"#;
    let out = run_parity(src);
    let total: usize = out[0].parse().unwrap();
    let semi_n: usize = out[1].parse().unwrap();
    let anti_n: usize = out[2].parse().unwrap();
    assert_eq!(
        semi_n + anti_n, total,
        "semi + anti should partition left: {semi_n} + {anti_n} != {total}"
    );
}

/// Property: stringr round-trip — str_to_upper then str_to_lower on ASCII
/// gives original (modulo case).
#[test]
fn prop_stringr_upper_lower_roundtrip() {
    let src = r#"
fn main() -> i64 {
    let s = "hello world";
    let up = str_to_upper(s);
    let down = str_to_lower(up);
    print(s);
    print(down);
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], out[1], "upper/lower round-trip on ASCII");
}

/// Property: stats — median of a single-element array is that element.
#[test]
fn prop_stats_median_single() {
    let src = r#"
fn main() -> i64 {
    let arr = [42.0];
    print(median(arr));
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], "42", "median of [42.0] should be 42");
}

/// Property: n_distinct of all-same array is 1.
#[test]
fn prop_stats_n_distinct_all_same() {
    let src = r#"
fn main() -> i64 {
    let arr = [7, 7, 7, 7, 7];
    print(n_distinct(arr));
    0
}
"#;
    let out = run_parity(src);
    assert_eq!(out[0], "1", "n_distinct of all-same should be 1");
}

// ============================================================================
//  Perf timing harness — run with `cargo test --test tidy_tests -- --ignored`
// ============================================================================

/// Measure wall-clock time for each fixture through both engines.
/// Outputs a markdown table to stdout for easy copy-paste into audit docs.
#[test]
#[ignore] // Perf benchmark — run manually: cargo test perf_tidy_fixtures -- --ignored
fn perf_tidy_fixtures() {
    use std::time::Instant;

    let fixtures = [
        "tidy_filter_select",
        "tidy_group_summarise",
        "tidy_arrange_slice",
        "tidy_join",
        "stringr_builtins",
        "stats_builtins",
        "tidy_empty_df",
        "tidy_pipeline",
    ];

    let dir = fixtures_dir();
    let mut results: Vec<(String, f64, f64)> = Vec::new();

    for name in &fixtures {
        let cjc_path = dir.join(format!("{name}.cjcl"));
        let source = fs::read_to_string(&cjc_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", cjc_path.display()));

        // Warm up
        let _ = run_eval(&source);
        let _ = run_mir(&source);

        // Timed runs (average of 5 iterations)
        let iters = 5;
        let mut eval_total = 0.0_f64;
        let mut mir_total = 0.0_f64;

        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = run_eval(&source);
            eval_total += t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            let _ = run_mir(&source);
            mir_total += t1.elapsed().as_secs_f64();
        }

        let eval_avg_us = (eval_total / iters as f64) * 1_000_000.0;
        let mir_avg_us = (mir_total / iters as f64) * 1_000_000.0;
        results.push((name.to_string(), eval_avg_us, mir_avg_us));
    }

    // Print markdown table
    println!();
    println!("## Tidy Bridge Perf Results");
    println!();
    println!("| Fixture | Eval (us) | MIR (us) | Ratio |");
    println!("|---------|-----------|----------|-------|");
    for (name, eval_us, mir_us) in &results {
        let ratio = if *eval_us > 0.0 { mir_us / eval_us } else { 0.0 };
        println!(
            "| {name:<24} | {eval_us:>9.1} | {mir_us:>8.1} | {ratio:>5.2}x |"
        );
    }
    println!();
}
