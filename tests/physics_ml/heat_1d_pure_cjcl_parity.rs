//! Phase 3c — parity gate for the pure-CJC-Lang heat-1D PINN demo.
//!
//! Loads `examples/physics_ml/pinn_heat_1d_pure.cjcl`, runs it through both
//! `cjc-eval` (AST tree-walk) and `cjc-mir-exec` (MIR register machine),
//! and asserts:
//!
//! 1. **Byte-equal printed output** — every line, every f64 → string
//!    rendering, identical across executors. This is the strongest possible
//!    parity contract: any divergence in the underlying tensor values would
//!    surface as a different rendered loss/error string.
//!
//! 2. **Loss is monotonically non-increasing across reported epochs** —
//!    catches a regression where the optimizer no longer descends.
//!
//! 3. **Final L2 / max-error are finite and within demo thresholds** —
//!    catches NaN/Inf creep in the FD residual or Adam moment buffers.
//!
//! The brief authorized RMSE<1e-6 as the parity tolerance (since the demo
//! takes the FD residual path, not native higher-order autodiff). In
//! practice both executors call into the same shared dispatch, so we get
//! exact byte-equality with no tolerance — and that's what we gate on.
//!
//! Wall clock note: the demo runs 50 epochs through the AST interpreter,
//! which under release on the bench machine clocks ~30s per executor. The
//! test is therefore release-only and `#[ignore]`-able if the local box is
//! slow; default behavior is to run it.

use std::path::PathBuf;

fn demo_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("examples/physics_ml/pinn_heat_1d_pure.cjcl")
}

fn load_demo_source() -> String {
    let p = demo_path();
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", p.display()))
}

fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in pinn_heat_1d_pure.cjcl:\n{:#?}",
        diags.diagnostics,
    );
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp.exec(&program)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in pinn_heat_1d_pure.cjcl:\n{:#?}",
        diags.diagnostics,
    );
    let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e:?}"));
    exec.output
}

/// Extract numeric loss values from `epoch=N  loss=M` lines.
fn parse_losses(out: &[String]) -> Vec<f64> {
    out.iter()
        .filter_map(|line| {
            let idx = line.find("loss=")?;
            line[idx + 5..].trim().parse::<f64>().ok()
        })
        .collect()
}

/// Extract the final L2 RMSE from the rendered output.
fn parse_l2(out: &[String]) -> Option<f64> {
    out.iter()
        .find(|line| line.contains("L2 RMSE"))
        .and_then(|line| {
            let idx = line.find(':')?;
            line[idx + 1..].trim().parse::<f64>().ok()
        })
}

/// Extract the final max abs error from the rendered output.
fn parse_max(out: &[String]) -> Option<f64> {
    out.iter()
        .find(|line| line.starts_with("  Max abs"))
        .and_then(|line| {
            let idx = line.find(':')?;
            line[idx + 1..].trim().parse::<f64>().ok()
        })
}

/// AST↔MIR byte-equal output. Strongest parity contract available.
#[test]
fn pure_cjcl_demo_eval_mir_byte_equal() {
    let src = load_demo_source();
    let eval_out = run_eval(&src, 42);
    let mir_out = run_mir(&src, 42);

    assert_eq!(
        eval_out.len(),
        mir_out.len(),
        "output line counts differ: eval={} mir={}",
        eval_out.len(),
        mir_out.len(),
    );

    for (i, (e, m)) in eval_out.iter().zip(mir_out.iter()).enumerate() {
        assert_eq!(
            e, m,
            "line {i} diverges:\n  eval: {e:?}\n  mir : {m:?}",
        );
    }
}

/// Loss is non-increasing across reported epochs (every 10th).
#[test]
fn pure_cjcl_demo_loss_decreases() {
    let src = load_demo_source();
    let out = run_eval(&src, 42);
    let losses = parse_losses(&out);
    assert!(
        losses.len() >= 5,
        "expected ≥5 reported losses (every 10 epochs over 50 epochs); got {}: {:?}",
        losses.len(),
        losses,
    );
    for w in losses.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-9,
            "loss increased between checkpoints: {} → {}",
            w[0],
            w[1],
        );
    }
}

/// Final metrics are finite and within the demo's loose thresholds.
/// This is the smoke gate; tighter gates require >50 epochs.
#[test]
fn pure_cjcl_demo_final_metrics_within_demo_thresholds() {
    let src = load_demo_source();
    let out = run_eval(&src, 42);
    let l2 = parse_l2(&out).expect("could not parse L2 RMSE from output");
    let max_e = parse_max(&out).expect("could not parse max abs from output");
    assert!(l2.is_finite(), "L2 RMSE not finite: {l2}");
    assert!(max_e.is_finite(), "max abs error not finite: {max_e}");
    // Demo budget (50 epochs) — calibrated 0.503/0.763 on the reference run.
    // Thresholds carry ~1.2× headroom over observed.
    assert!(l2 < 0.65, "L2 RMSE {l2} exceeds demo threshold 0.65");
    assert!(max_e < 1.0, "max abs {max_e} exceeds demo threshold 1.0");
}
