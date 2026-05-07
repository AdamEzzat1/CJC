//! Phase 0.3b — AST eval ↔ MIR exec parity tests for BLR builtins.

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

#[test]
fn parity_set_blr_prior() {
    assert_parity(
        "set_blr_prior + n_seen + state hash present",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        abng_set_blr_prior(g, 1.0, 1.5, 1.0);
        print(abng_blr_n_seen(g, 0));
        print(abng_blr_state_hash(g, 0));
        "#,
    );
}

#[test]
fn parity_blr_predict_initial() {
    // Before any update, prediction at uniform features should be
    // bit-identical across executors.
    assert_parity(
        "blr_predict at initial state is deterministic",
        r#"
        let g = abng_new(7);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        abng_set_blr_prior(g, 1.0, 2.0, 1.0);
        let phi = Tensor.from_vec([0.1, 0.2, 0.3, 0.4], [4]);
        print(abng_blr_predict(g, 0, phi));
        "#,
    );
}

#[test]
fn parity_blr_update_then_predict() {
    assert_parity(
        "blr_update + predict round-trip",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        abng_set_blr_prior(g, 1.0, 2.0, 1.0);
        let xs = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 4]);
        let ys = Tensor.from_vec([1.0, 2.0], [2]);
        abng_blr_update(g, 0, xs, ys);
        print(abng_blr_n_seen(g, 0));
        let phi = Tensor.from_vec([0.5, 0.5, 0.5, 0.5], [4]);
        print(abng_blr_predict(g, 0, phi));
        "#,
    );
}

#[test]
fn parity_blr_state_hash_changes_after_update() {
    assert_parity(
        "state hash changes after update",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        abng_set_blr_prior(g, 1.0, 1.5, 1.0);
        let h0 = abng_blr_state_hash(g, 0);
        let xs = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [1, 4]);
        let ys = Tensor.from_vec([5.0], [1]);
        abng_blr_update(g, 0, xs, ys);
        let h1 = abng_blr_state_hash(g, 0);
        print(h0 == h1);
        "#,
    );
}

#[test]
fn parity_full_uncertainty_pipeline() {
    // End-to-end: Xavier-init MLP → blr_features → train BLR → predict
    // with uncertainty. Chain head must agree across executors.
    assert_parity(
        "full uncertainty pipeline preserves chain head",
        r#"
        let g = abng_new(42);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        abng_set_blr_prior(g, 1.0, 2.0, 1.0);
        // Train BLR on a small batch (using synthetic features in the
        // 4-D space, since penultimate dim = 4).
        let xs = Tensor.from_vec(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            [3, 4]
        );
        let ys = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        abng_blr_update(g, 0, xs, ys);
        let phi = Tensor.from_vec([0.5, 0.5, 0.5, 0.5], [4]);
        print(abng_blr_predict(g, 0, phi));
        print(abng_chain_head(g));
        "#,
    );
}

#[test]
fn parity_double_run_blr_chain_head() {
    let body = r#"
        let g = abng_new(99);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        abng_set_blr_prior(g, 1.0, 2.0, 1.0);
        let xs = Tensor.from_vec([0.1, 0.2, 0.3, 0.4], [1, 4]);
        let ys = Tensor.from_vec([1.5], [1]);
        abng_blr_update(g, 0, xs, ys);
        print(abng_chain_head(g));
        print(abng_blr_state_hash(g, 0));
    "#;
    let a = run(Backend::Eval, body, 0);
    let b = run(Backend::Eval, body, 0);
    assert_eq!(a, b, "eval double-run differs");
    let c = run(Backend::Mir, body, 0);
    let d = run(Backend::Mir, body, 0);
    assert_eq!(c, d, "MIR double-run differs");
    assert_eq!(a, c, "eval ↔ MIR differs");
}
