//! PINN v2 runner parity gate — `run_program_optimized_pinn_v2*` must
//! produce output byte-identical to the AST tree-walk interpreter.
//!
//! The trained thermal head can only change WHICH legal passes run; it
//! must never change observable program behavior (the legality gate +
//! semantics-preserving passes guarantee this — these tests lock the
//! guarantee for the new `--pinn-weights` entry points specifically).
//!
//! Weights come from the COMMITTED bundle at
//! `bench_results/cana_train_pinn/pinn_thermal_v2.cpb` (trained offline
//! by `cargo run --release -p cana-train-pinn -- train`), so the gate
//! exercises the exact artifact the CLI flag ships.

use std::path::Path;

use cjc_cana::pinn_thermal_v2::{PinnThermalV2, PINN_V2_MODEL_ID};
use cjc_cana_compress::pinn_bundle::read_bundle;

const BUNDLE_PATH: &str = "bench_results/cana_train_pinn/pinn_thermal_v2.cpb";
const SEED: u64 = 42;

fn trained_head() -> PinnThermalV2 {
    let bundle = read_bundle(Path::new(BUNDLE_PATH))
        .expect("committed CPB0 bundle must load — run cana-train-pinn -- train if missing");
    assert_eq!(bundle.model_id, PINN_V2_MODEL_ID);
    bundle.head
}

/// AST-eval vs PINN-v2-optimized MIR-exec output parity for one source.
fn assert_parity(src: &str) {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse: {:?}", diags.diagnostics);

    let mut interp = cjc_eval::Interpreter::new(SEED);
    interp.exec(&ast).expect("AST eval must succeed");

    let head = trained_head();
    let (_, exec) = cjc_mir_exec::run_program_optimized_pinn_v2_with_executor(&ast, SEED, &head)
        .expect("PINN v2 runner must succeed");

    assert_eq!(
        interp.output, exec.output,
        "PINN v2 plan changed observable behavior"
    );
}

#[test]
fn parity_integer_loop() {
    assert_parity(
        r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(sum_to(1000));
"#,
    );
}

#[test]
fn parity_float_polynomial() {
    // FP-dense function — exactly the shape where the trained head
    // diverges hardest from the v1 closed form (it predicts hot and
    // withholds passes). Output must be identical regardless.
    assert_parity(
        r#"
fn polynomial(x: f64) -> f64 {
    let a: f64 = 3.14;
    let b: f64 = 2.71;
    let c: f64 = 1.41;
    return a * x * x + b * x + c;
}
print(polynomial(1.5));
"#,
    );
}

#[test]
fn parity_fp_hot_nested_loops() {
    assert_parity(
        r#"
fn horner(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        let mut p: f64 = 1.0;
        while j < 16 {
            p = p * x + 0.5;
            acc = acc + p * 0.001;
            j = j + 1;
        }
        i = i + 1;
    }
    return acc;
}
print(horner(1.01, 50));
"#,
    );
}

#[test]
fn pinn_v2_runner_is_deterministic() {
    let src = r#"
fn work(n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i * 0.001;
        i = i + 1;
    }
    return acc;
}
print(work(100));
"#;
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let head = trained_head();
    let (_, first) = cjc_mir_exec::run_program_optimized_pinn_v2_with_executor(&ast, SEED, &head)
        .expect("run 1");
    for _ in 0..5 {
        let (_, again) =
            cjc_mir_exec::run_program_optimized_pinn_v2_with_executor(&ast, SEED, &head)
                .expect("run n");
        assert_eq!(first.output, again.output, "same seed must be bit-identical");
    }
}
