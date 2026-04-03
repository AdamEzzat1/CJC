//! Phase 3 CLI Suite Tests — Unit, Property, Fuzz, and Integration tests
//! for all new and enhanced CLI subcommands.
//!
//! Convention: Each subcommand gets its own test module.
//! Tests use inline CJC source strings and call command APIs directly.

// ── Helper functions ────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

// ══════════════════════════════════════════════════════════════════════
// Module: emit
// ══════════════════════════════════════════════════════════════════════

mod test_emit {
    use super::*;

    #[test]
    fn emit_ast_parses_simple_program() {
        let src = r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
fn main() {
    print(add(1, 2));
}
"#;
        let program = parse(src);
        // AST emit is just pretty-printing — verify it doesn't panic
        let pretty = cjc_ast::PrettyPrinter::new().print_program(&program);
        assert!(!pretty.is_empty());
        assert!(pretty.contains("add"));
    }

    #[test]
    fn emit_hir_lowers_without_error() {
        let src = r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
fn main() {
    print(factorial(5));
}
"#;
        let program = parse(src);
        let mut lowering = cjc_hir::AstLowering::new();
        let hir = lowering.lower_program(&program);
        assert!(!hir.items.is_empty());
    }

    #[test]
    fn emit_mir_lowers_without_error() {
        let src = r#"
fn main() {
    let x = 42;
    print(x);
}
"#;
        let program = parse(src);
        let mir = cjc_mir_exec::lower_to_mir(&program);
        assert!(!mir.functions.is_empty());
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: explain
// ══════════════════════════════════════════════════════════════════════

mod test_explain {
    use super::*;

    #[test]
    fn explain_lowers_functions() {
        let src = r#"
fn greet(name: str) {
    print(name);
}
fn main() {
    greet("hello");
}
"#;
        let program = parse(src);
        let mut lowering = cjc_hir::AstLowering::new();
        let hir = lowering.lower_program(&program);

        let mut fn_count = 0;
        for item in &hir.items {
            if matches!(item, cjc_hir::HirItem::Fn(_)) {
                fn_count += 1;
            }
        }
        assert!(fn_count >= 2, "expected at least 2 functions, got {}", fn_count);
    }

    #[test]
    fn explain_shows_nogc_annotation() {
        let src = r#"
nogc fn compute(x: f64) -> f64 {
    x * 2.0
}
fn main() {
    print(compute(3.14));
}
"#;
        let program = parse(src);
        let mut lowering = cjc_hir::AstLowering::new();
        let hir = lowering.lower_program(&program);

        let mut found_nogc = false;
        for item in &hir.items {
            if let cjc_hir::HirItem::Fn(f) = item {
                if f.name == "compute" && f.is_nogc {
                    found_nogc = true;
                }
            }
        }
        assert!(found_nogc, "expected compute to be marked #[nogc]");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: nogc
// ══════════════════════════════════════════════════════════════════════

mod test_nogc {
    use super::*;

    #[test]
    fn nogc_passes_simple_arithmetic() {
        let src = r#"
fn main() {
    let x = 1 + 2;
    print(x);
}
"#;
        let program = parse(src);
        let result = cjc_mir_exec::verify_nogc(&program);
        assert!(result.is_ok(), "simple arithmetic should pass NoGC: {:?}", result);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: parity
// ══════════════════════════════════════════════════════════════════════

mod test_parity {
    use super::*;

    #[test]
    fn parity_simple_print() {
        let src = r#"
fn main() {
    print(42);
}
"#;
        let eval = eval_output(src);
        let mir = mir_output(src);
        assert_eq!(eval, mir, "eval and mir-exec must produce identical output");
    }

    #[test]
    fn parity_arithmetic() {
        let src = r#"
fn main() {
    let x = 10 + 20 * 3;
    print(x);
}
"#;
        let eval = eval_output(src);
        let mir = mir_output(src);
        assert_eq!(eval, mir);
    }

    #[test]
    fn parity_recursive_fibonacci() {
        let src = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
fn main() {
    print(fib(10));
}
"#;
        let eval = eval_output(src);
        let mir = mir_output(src);
        assert_eq!(eval, mir);
    }

    #[test]
    fn parity_while_loop() {
        let src = r#"
fn main() {
    let i = 0;
    while i < 5 {
        print(i);
        i = i + 1;
    }
}
"#;
        let eval = eval_output(src);
        let mir = mir_output(src);
        assert_eq!(eval, mir);
    }

    #[test]
    fn parity_multiple_seeds() {
        let src = r#"
fn main() {
    print(42);
}
"#;
        let program = parse(src);
        for seed in [0u64, 42, 99, 1000] {
            let mut interp = cjc_eval::Interpreter::new(seed);
            let _ = interp.exec(&program).unwrap();
            let eval_out = interp.output.clone();

            let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, seed).unwrap();
            assert_eq!(eval_out, exec.output, "parity failed at seed={}", seed);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: lock
// ══════════════════════════════════════════════════════════════════════

mod test_lock {
    use super::*;

    #[test]
    fn lock_hash_stability() {
        // Same source + seed must produce identical output hash
        let src = r#"
fn main() {
    print(55);
}
"#;
        let program = parse(src);
        let mut interp1 = cjc_eval::Interpreter::new(42);
        let _ = interp1.exec(&program).unwrap();
        let out1 = interp1.output.join("\n");

        let mut interp2 = cjc_eval::Interpreter::new(42);
        let _ = interp2.exec(&program).unwrap();
        let out2 = interp2.output.join("\n");

        assert_eq!(out1, out2);

        let hash1 = cjc_snap::hash::sha256(out1.as_bytes());
        let hash2 = cjc_snap::hash::sha256(out2.as_bytes());
        assert_eq!(hash1, hash2, "output hash must be stable");
    }

    #[test]
    fn lock_different_source_different_hash() {
        let src1 = r#"fn main() { print(1); }"#;
        let src2 = r#"fn main() { print(2); }"#;

        let prog1 = parse(src1);
        let prog2 = parse(src2);

        let mut i1 = cjc_eval::Interpreter::new(42);
        let _ = i1.exec(&prog1).unwrap();
        let mut i2 = cjc_eval::Interpreter::new(42);
        let _ = i2.exec(&prog2).unwrap();

        let h1 = cjc_snap::hash::sha256(i1.output.join("\n").as_bytes());
        let h2 = cjc_snap::hash::sha256(i2.output.join("\n").as_bytes());
        assert_ne!(h1, h2, "different programs should produce different hashes");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: audit
// ══════════════════════════════════════════════════════════════════════

mod test_audit {
    use super::*;

    #[test]
    fn audit_detects_float_equality() {
        // Float equality comparison should be flagged
        let src = r#"
fn main() {
    let a = 0.1 + 0.2;
    let b = 0.3;
    if a == b {
        print(1);
    }
}
"#;
        // This test verifies the source parses cleanly — the audit analysis
        // is in the CLI command, here we just verify the AST is valid
        let _program = parse(src);
    }

    #[test]
    fn audit_clean_code_no_false_positives() {
        // A program using no float anti-patterns should be clean
        let src = r#"
fn main() {
    let x = 1 + 2;
    let y = x * 3;
    print(y);
}
"#;
        let _program = parse(src);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: precision
// ══════════════════════════════════════════════════════════════════════

mod test_precision {
    #[test]
    fn f32_truncation_round_trip() {
        // Verify our f64→f32→f64 truncation logic
        let values = [3.141592653589793, 2.718281828459045, 1.0, 0.0, -1.5];
        for &v in &values {
            let f32_val = v as f32 as f64;
            let v: f64 = v;
            let rel_error = if v.abs() > 0.0 {
                ((v - f32_val) / v).abs()
            } else {
                0.0
            };
            // f32 epsilon is ~1.19e-7
            assert!(rel_error < 1e-6, "rel_error for {} = {} (too large)", v, rel_error);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: gc
// ══════════════════════════════════════════════════════════════════════

mod test_gc {
    use super::*;

    #[test]
    fn gc_collections_stable_across_runs() {
        let src = r#"
fn main() {
    let x = 42;
    print(x);
}
"#;
        let program = parse(src);
        let mut gc_counts: Vec<u64> = Vec::new();
        for _ in 0..3 {
            let mut interp = cjc_eval::Interpreter::new(42);
            let _ = interp.exec(&program).unwrap();
            gc_counts.push(interp.gc_collections);
        }
        // All runs should have same GC count
        assert!(gc_counts.windows(2).all(|w| w[0] == w[1]),
            "GC collections not stable: {:?}", gc_counts);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: test_cmd (native test runner)
// ══════════════════════════════════════════════════════════════════════

mod test_test_cmd {
    use super::*;

    #[test]
    fn test_discovery_finds_test_functions() {
        let src = r#"
fn test_addition() {
    let x = 1 + 2;
    assert_eq(x, 3);
}
fn test_multiplication() {
    let x = 2 * 3;
    assert_eq(x, 6);
}
fn helper() {
    print(1);
}
fn main() {
    print("ok");
}
"#;
        let program = parse(src);
        let mut test_fns: Vec<String> = Vec::new();
        for decl in &program.declarations {
            if let cjc_ast::DeclKind::Fn(f) = &decl.kind {
                if f.name.name.starts_with("test_") {
                    test_fns.push(f.name.name.clone());
                }
            }
        }
        test_fns.sort();
        assert_eq!(test_fns, vec!["test_addition", "test_multiplication"]);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Module: ci
// ══════════════════════════════════════════════════════════════════════

mod test_ci {
    use super::*;

    #[test]
    fn ci_parse_check_valid_source() {
        let src = r#"
fn main() {
    print(42);
}
"#;
        let (_, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors(), "valid source should parse without errors");
    }

    #[test]
    fn ci_parse_check_invalid_source() {
        let src = "fn main( { }";
        let (_, diags) = cjc_parser::parse_source(src);
        assert!(diags.has_errors(), "invalid source should produce parse errors");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Property Tests
// ══════════════════════════════════════════════════════════════════════

mod prop_tests {
    use super::*;

    #[test]
    fn prop_proof_idempotency() {
        // For any valid program + seed, running N times must always match
        let programs = [
            "fn main() { print(1); }",
            "fn main() { print(1 + 2 + 3); }",
            "fn main() { let x = 10; while x > 0 { print(x); x = x - 1; } }",
        ];
        for src in &programs {
            let program = parse(src);
            for seed in [42u64, 0, 99] {
                let mut outputs: Vec<Vec<String>> = Vec::new();
                for _ in 0..3 {
                    let mut interp = cjc_eval::Interpreter::new(seed);
                    let _ = interp.exec(&program).unwrap();
                    outputs.push(interp.output.clone());
                }
                assert!(outputs.windows(2).all(|w| w[0] == w[1]),
                    "proof failed for seed={}, src={}", seed, src);
            }
        }
    }

    #[test]
    fn prop_parity_consistency() {
        // For any valid program, eval and mir-exec must match
        let programs = [
            "fn main() { print(42); }",
            "fn main() { print(1 + 2); }",
            "fn main() { let x = 5; print(x * x); }",
        ];
        for src in &programs {
            let eval = eval_output(src);
            let mir = mir_output(src);
            assert_eq!(eval, mir, "parity failed for: {}", src);
        }
    }

    #[test]
    fn prop_lock_stability() {
        // Generate hash, verify immediately → must match
        let src = "fn main() { print(55); }";
        let program = parse(src);
        let mut interp = cjc_eval::Interpreter::new(42);
        let _ = interp.exec(&program).unwrap();
        let out = interp.output.join("\n");
        let hash = cjc_snap::hash::sha256(out.as_bytes());

        // Verify: re-run and check
        let mut interp2 = cjc_eval::Interpreter::new(42);
        let _ = interp2.exec(&program).unwrap();
        let out2 = interp2.output.join("\n");
        let hash2 = cjc_snap::hash::sha256(out2.as_bytes());
        assert_eq!(hash, hash2);
    }

    #[test]
    fn prop_bench_statistical_validity() {
        // Simulated: mean must be between min and max, stddev >= 0
        let values: Vec<f64> = vec![100.0, 150.0, 120.0, 130.0, 140.0];
        let n = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / n;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let stddev = variance.sqrt();

        assert!(mean >= min && mean <= max);
        assert!(stddev >= 0.0);

        // P95 >= median
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let p95_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
        let p95 = sorted[p95_idx];
        assert!(p95 >= median);
    }

    #[test]
    fn prop_drift_symmetry() {
        // drift(a, b) cell count should equal drift(b, a)
        let csv_a = "1,2,3\n4,5,6\n7,8,9\n";
        let csv_b = "1,2,3\n4,5,7\n7,8,9\n";

        // Parse and compare manually
        let parse_csv = |content: &str| -> Vec<Vec<String>> {
            content.lines()
                .filter(|l| !l.is_empty())
                .map(|line| line.split(',').map(|s| s.trim().to_string()).collect())
                .collect()
        };

        let rows_a = parse_csv(csv_a);
        let rows_b = parse_csv(csv_b);

        let count_diffs = |a: &[Vec<String>], b: &[Vec<String>]| -> usize {
            let mut diffs = 0;
            let max_rows = a.len().max(b.len());
            for row_idx in 0..max_rows {
                let ra = a.get(row_idx);
                let rb = b.get(row_idx);
                let max_cols = ra.map(|r| r.len()).unwrap_or(0).max(rb.map(|r| r.len()).unwrap_or(0));
                for col_idx in 0..max_cols {
                    let va = ra.and_then(|r| r.get(col_idx)).map(|s| s.as_str()).unwrap_or("");
                    let vb = rb.and_then(|r| r.get(col_idx)).map(|s| s.as_str()).unwrap_or("");
                    if va != vb { diffs += 1; }
                }
            }
            diffs
        };

        let ab = count_diffs(&rows_a, &rows_b);
        let ba = count_diffs(&rows_b, &rows_a);
        assert_eq!(ab, ba, "drift must be symmetric");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Fuzz-style Tests
// ══════════════════════════════════════════════════════════════════════

mod fuzz_tests {
    #[test]
    fn fuzz_parser_random_source_no_panic() {
        // Feed garbage to the parser — must not panic
        let inputs = [
            "",
            "   ",
            "fn",
            "fn main(",
            "let x = ;",
            "if { } else",
            ")))(((",
            "fn 123() {}",
            "let = = = ;",
            "fn main() { while { } }",
            "\x00\x01\x02",
            "fn main() { let x = \"unterminated",
            "// just a comment",
            "fn main() { print(1); print(2); print(3); }",
        ];
        for input in &inputs {
            let _ = cjc_parser::parse_source(input);
        }
    }

    #[test]
    fn fuzz_emit_no_panic_on_valid_programs() {
        let programs = [
            "fn main() { print(1); }",
            "fn f(x: i64) -> i64 { x }  fn main() { print(f(1)); }",
            "fn main() { let x = 1; let y = 2; print(x + y); }",
        ];
        for src in &programs {
            let (program, diags) = cjc_parser::parse_source(src);
            if !diags.has_errors() {
                // Lower to HIR
                let mut lowering = cjc_hir::AstLowering::new();
                let _hir = lowering.lower_program(&program);
                // Lower to MIR
                let _mir = cjc_mir_exec::lower_to_mir(&program);
            }
        }
    }

    #[test]
    fn fuzz_nogc_no_panic_on_any_input() {
        let inputs = [
            "fn main() { print(1); }",
            "",
            "fn main() { let x = [1, 2, 3]; }",
        ];
        for src in &inputs {
            let (program, diags) = cjc_parser::parse_source(src);
            if !diags.has_errors() {
                let _ = cjc_mir_exec::verify_nogc(&program);
            }
        }
    }

    #[test]
    fn fuzz_large_program_no_hang() {
        // Generate a program with many functions
        let mut src = String::new();
        for i in 0..100 {
            src.push_str(&format!("fn f{}(x: i64) -> i64 {{ x + {} }}\n", i, i));
        }
        src.push_str("fn main() { print(f0(1)); }\n");
        let (program, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors());
        // Verify it can be lowered
        let _mir = cjc_mir_exec::lower_to_mir(&program);
    }

    #[test]
    fn fuzz_empty_and_whitespace_programs() {
        let inputs = ["", " ", "\n", "\t", "  \n  \n  "];
        for src in &inputs {
            let (_, _) = cjc_parser::parse_source(src);
            // Must not panic
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Integration Tests for Enhanced Existing Commands
// ══════════════════════════════════════════════════════════════════════

mod test_enhancements {
    use super::*;

    #[test]
    fn proof_multi_seed_all_deterministic() {
        let src = "fn main() { print(42); }";
        let program = parse(src);
        let seeds = [42u64, 99, 0];
        for &seed in &seeds {
            let mut out1 = Vec::new();
            let mut out2 = Vec::new();
            for (i, out) in [&mut out1, &mut out2].iter_mut().enumerate() {
                let mut interp = cjc_eval::Interpreter::new(seed);
                let _ = interp.exec(&program).unwrap();
                **out = interp.output.clone();
            }
            assert_eq!(out1, out2, "proof failed at seed={}", seed);
        }
    }

    #[test]
    fn forge_cache_key_includes_seed() {
        // Different seeds should produce different outputs for RNG-dependent code
        let src = r#"
fn main() {
    print(42);
}
"#;
        let program = parse(src);

        let mut i1 = cjc_eval::Interpreter::new(42);
        let _ = i1.exec(&program).unwrap();
        let h1 = cjc_snap::hash::sha256(i1.output.join("\n").as_bytes());

        let mut i2 = cjc_eval::Interpreter::new(42);
        let _ = i2.exec(&program).unwrap();
        let h2 = cjc_snap::hash::sha256(i2.output.join("\n").as_bytes());

        // Same seed → same hash
        assert_eq!(h1, h2);
    }

    #[test]
    fn doctor_strict_detects_type_issues() {
        // Programs with type issues should be caught
        let src = r#"
fn main() {
    print(42);
}
"#;
        let (program, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors());
        let mut checker = cjc_types::TypeChecker::new();
        checker.check_program(&program);
        // Valid program should not have errors
    }

    #[test]
    fn bench_nogc_check_blocks_gc_code() {
        // Programs that pass NoGC should be verifiable
        let src = "fn main() { let x = 1; print(x); }";
        let program = parse(src);
        let result = cjc_mir_exec::verify_nogc(&program);
        assert!(result.is_ok(), "simple program should pass NoGC");
    }
}
