//! Phase 0.3c — AST eval ↔ MIR exec parity tests for the OOD /
//! calibration / drift builtins.

#![allow(clippy::needless_raw_string_hashes)]

use cjc_abng::dispatch::reset_arena;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    reset_arena();
    cjc_ad::dispatch::reset_ambient();
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed:\n{src}\nerror: {e:?}"));
            exec.output
        }
    }
}

fn assert_parity(label: &str, body: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
}

const SETUP: &str = r#"
let g = abng_new(7);
let hidden = Tensor.from_vec([], [0]);
abng_set_leaf_head(g, 2, hidden, 1, "none");
abng_set_blr_prior(g, 1.0, 1.5, 1.0);
abng_set_density_tracker(g);
abng_set_calibration(g, 15);
"#;

#[test]
fn parity_density_observe_and_score() {
    let body = format!(
        "{SETUP}
        let xs = Tensor.from_vec([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [3, 2]);
        abng_density_observe(g, 0, xs);
        print(abng_density_n_seen(g, 0));
        let phi = Tensor.from_vec([2.0, 2.0], [2]);
        print(abng_density_score(g, 0, phi));
        "
    );
    assert_parity("density observe + score", &body);
}

#[test]
fn parity_calibration_observe_and_ece() {
    let body = format!(
        "{SETUP}
        abng_calibration_observe(g, 0, 0.5, true);
        abng_calibration_observe(g, 0, 0.5, false);
        abng_calibration_observe(g, 0, 0.7, true);
        print(abng_calibration_n_seen(g, 0));
        print(abng_calibration_ece(g, 0));
        "
    );
    assert_parity("calibration observe + ece", &body);
}

#[test]
fn parity_drift_freeze_then_score() {
    let body = format!(
        "{SETUP}
        let xs = Tensor.from_vec([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [4, 2]);
        abng_density_observe(g, 0, xs);
        abng_freeze_drift_baseline(g, 0);
        print(abng_drift_score(g, 0));
        let xs2 = Tensor.from_vec([10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0], [4, 2]);
        abng_density_observe(g, 0, xs2);
        print(abng_drift_score(g, 0));
        "
    );
    assert_parity("drift baseline + score", &body);
}

#[test]
fn parity_ood_composite() {
    let body = format!(
        "{SETUP}
        let xs = Tensor.from_vec([0.0, 0.0, 1.0, 1.0], [2, 2]);
        abng_density_observe(g, 0, xs);
        let phi_in = Tensor.from_vec([0.5, 0.5], [2]);
        let phi_out = Tensor.from_vec([100.0, 100.0], [2]);
        print(abng_ood_score(g, 0, phi_in, 5, 5));
        print(abng_ood_score(g, 0, phi_out, 0, 5));
        "
    );
    assert_parity("OOD composite max", &body);
}

#[test]
fn parity_full_pipeline_chain_head() {
    let body = format!(
        "{SETUP}
        let xs = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [3, 2]);
        abng_density_observe(g, 0, xs);
        abng_calibration_observe(g, 0, 0.6, true);
        abng_calibration_observe(g, 0, 0.4, false);
        abng_freeze_drift_baseline(g, 0);
        print(abng_chain_head(g));
        "
    );
    assert_parity("full pipeline preserves chain head", &body);
}

#[test]
fn parity_double_run_chain_head() {
    let body = format!(
        "{SETUP}
        let xs = Tensor.from_vec([0.1, 0.2, 0.3, 0.4], [2, 2]);
        abng_density_observe(g, 0, xs);
        abng_calibration_observe(g, 0, 0.7, true);
        abng_freeze_drift_baseline(g, 0);
        print(abng_chain_head(g));
        "
    );
    let a = run(Backend::Eval, &body, 0);
    let b = run(Backend::Eval, &body, 0);
    assert_eq!(a, b, "eval double-run differs");
    let c = run(Backend::Mir, &body, 0);
    let d = run(Backend::Mir, &body, 0);
    assert_eq!(c, d, "MIR double-run differs");
    assert_eq!(a, c, "eval ↔ MIR differs");
}
