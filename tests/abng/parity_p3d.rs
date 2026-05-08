//! Phase 0.3d-1 — AST eval ↔ MIR exec parity tests for the maturity
//! and signature builtins.

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

const SETUP_FULL: &str = r#"
let g = abng_new(7);
let hidden = Tensor.from_vec([], [0]);
abng_set_leaf_head(g, 2, hidden, 1, "none");
abng_set_blr_prior(g, 1.0, 1.5, 1.0);
abng_set_density_tracker(g);
abng_set_calibration(g, 15);
"#;

#[test]
fn parity_maturity_fresh_root() {
    let body = r#"
        let g = abng_new(0);
        let m = abng_node_maturity(g, 0);
        print(m);
    "#;
    assert_parity("maturity fresh root", body);
}

#[test]
fn parity_maturity_after_observations() {
    let body = r#"
        let g = abng_new(0);
        let i: i64 = 0;
        while i < 64 {
            abng_observe(g, 0, 1.0);
            i = i + 1;
        }
        print(abng_node_maturity(g, 0));
    "#;
    assert_parity("maturity after 64 observes", body);
}

#[test]
fn parity_maturity_full_stack() {
    let body = format!(
        "{SETUP_FULL}
        abng_observe(g, 0, 0.7);
        abng_observe(g, 0, 0.8);
        print(abng_node_maturity(g, 0));
        "
    );
    assert_parity("maturity full-stack post-observe", &body);
}

#[test]
fn parity_signature_fresh_root() {
    let body = r#"
        let g = abng_new(0);
        let s = abng_node_signature(g, 0);
        print(s);
    "#;
    assert_parity("signature fresh root", body);
}

#[test]
fn parity_signature_full_stack() {
    let body = format!(
        "{SETUP_FULL}
        abng_observe(g, 0, 1.5);
        print(abng_node_signature(g, 0));
        "
    );
    assert_parity("signature full-stack", &body);
}

#[test]
fn parity_signature_after_add_node() {
    // Exercises the routing-profile path: adding a child should
    // produce an AST-MIR-identical change to the parent signature.
    let body = r#"
        let g = abng_new(0);
        print(abng_node_signature(g, 0));
        abng_add_node(g, 0, 7);
        print(abng_node_signature(g, 0));
    "#;
    assert_parity("signature after add_node", body);
}

#[test]
fn parity_double_run_signature() {
    let body = format!(
        "{SETUP_FULL}
        abng_observe(g, 0, 0.5);
        print(abng_node_signature(g, 0));
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

// ─── Phase 0.3d-2 — expected_epistemic + calibrated OOD ──────────

#[test]
fn parity_expected_epistemic_uncaptured() {
    let body = format!(
        "{SETUP_FULL}
        print(abng_expected_epistemic(g, 0));
        "
    );
    assert_parity("expected_epistemic uncaptured", &body);
}

#[test]
fn parity_set_expected_epistemic_round_trip() {
    let body = format!(
        "{SETUP_FULL}
        abng_set_expected_epistemic(g, 0, 0.42);
        print(abng_expected_epistemic(g, 0));
        print(abng_chain_head(g));
        "
    );
    assert_parity("set + read expected_epistemic + chain_head", &body);
}

#[test]
fn parity_ood_score_calibrated_branch() {
    // Train BLR a bit, capture a small reference, then call ood_score
    // and expect the calibrated ratio formula to fire.
    let body = format!(
        "{SETUP_FULL}
        let xs = Tensor.from_vec([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [3, 2]);
        let ys = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        abng_blr_update(g, 0, xs, ys);
        abng_set_expected_epistemic(g, 0, 0.000001);
        let phi = Tensor.from_vec([5.0, 5.0], [2]);
        print(abng_ood_score(g, 0, phi, 0, 5));
        "
    );
    assert_parity("ood_score calibrated branch", &body);
}

#[test]
fn parity_double_run_expected_epistemic() {
    let body = format!(
        "{SETUP_FULL}
        abng_set_expected_epistemic(g, 0, 0.7);
        print(abng_chain_head(g));
        print(abng_expected_epistemic(g, 0));
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

// ─── Phase 0.3d-3 — DecisionPolicy + force-* + Dense ─────────────

const POLICY_INSTALL: &str = r#"
let thresholds = Tensor.from_vec([0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0, 1.0e9], [12]);
abng_set_decision_policy(g, thresholds);
"#;

#[test]
fn parity_set_decision_policy_then_hash() {
    let body = format!(
        r#"
        let g = abng_new(0);
        {POLICY_INSTALL}
        print(abng_decision_policy_hash(g));
        "#
    );
    assert_parity("set + read decision policy hash", &body);
}

#[test]
fn parity_force_grow_then_action_count() {
    let body = r#"
        let g = abng_new(0);
        let c = abng_force_grow(g, 0, 7);
        print(c);
        print(abng_action_count(g, 0));
        print(abng_chain_head(g));
    "#;
    assert_parity("force_grow + action_count", body);
}

#[test]
fn parity_force_split_returns_two_ids() {
    let body = r#"
        let g = abng_new(0);
        let pair = abng_force_split(g, 0);
        print(pair);
        print(abng_action_count(g, 1));
    "#;
    assert_parity("force_split returns Tensor[2]", body);
}

#[test]
fn parity_force_freeze_then_is_frozen() {
    let body = r#"
        let g = abng_new(0);
        print(abng_is_frozen(g, 0));
        abng_force_freeze(g, 0);
        print(abng_is_frozen(g, 0));
        print(abng_action_count(g, 5));
    "#;
    assert_parity("freeze + is_frozen", body);
}

#[test]
fn parity_force_prune_then_chain_head() {
    let body = r#"
        let g = abng_new(0);
        let c = abng_add_node(g, 0, 7);
        abng_force_prune(g, c);
        print(abng_action_count(g, 3));
        print(abng_chain_head(g));
    "#;
    assert_parity("force_prune + chain_head", body);
}

#[test]
fn parity_force_merge_chain_head() {
    let body = r#"
        let g = abng_new(0);
        let a = abng_add_node(g, 0, 1);
        let b = abng_add_node(g, 0, 2);
        abng_force_merge(g, a, b);
        print(abng_action_count(g, 2));
        print(abng_chain_head(g));
    "#;
    assert_parity("force_merge + chain_head", body);
}

#[test]
fn parity_force_compress_then_chain_head() {
    let body = r#"
        let g = abng_new(0);
        abng_add_node(g, 0, 1);
        abng_add_node(g, 0, 2);
        abng_force_compress(g, 0);
        print(abng_action_count(g, 4));
        print(abng_chain_head(g));
    "#;
    assert_parity("force_compress + chain_head", body);
}

#[test]
fn parity_double_run_full_p3d3_workflow() {
    let body = format!(
        r#"
        let g = abng_new(42);
        {POLICY_INSTALL}
        let c1 = abng_force_grow(g, 0, 1);
        abng_force_freeze(g, c1);
        print(abng_chain_head(g));
        print(abng_action_count(g, 0));
        print(abng_action_count(g, 5));
        "#
    );
    let a = run(Backend::Eval, &body, 0);
    let b = run(Backend::Eval, &body, 0);
    assert_eq!(a, b, "eval double-run differs");
    let c = run(Backend::Mir, &body, 0);
    let d = run(Backend::Mir, &body, 0);
    assert_eq!(c, d, "MIR double-run differs");
    assert_eq!(a, c, "eval ↔ MIR differs");
}

// ─── Phase 0.3d-4 — decide_step + unfreeze parity ────────────────

#[test]
fn parity_decide_step_no_policy_returns_zeros() {
    let body = r#"
        let g = abng_new(0);
        let counts = abng_decide_step(g);
        print(counts);
    "#;
    assert_parity("decide_step no-policy returns zeros", body);
}

#[test]
fn parity_decide_step_idle_advances_stability() {
    let body = format!(
        r#"
        let g = abng_new(0);
        {POLICY_INSTALL}
        let c1 = abng_decide_step(g);
        let c2 = abng_decide_step(g);
        let c3 = abng_decide_step(g);
        print(c1);
        print(c2);
        print(c3);
        print(abng_chain_head(g));
        "#
    );
    assert_parity("decide_step idle advances stability", &body);
}

#[test]
fn parity_decide_step_fires_grow() {
    let body = format!(
        r#"
        let g = abng_new(0);
        {POLICY_INSTALL}
        let i: i64 = 0;
        while i < 70 {{
            abng_observe(g, 0, 1.0);
            i = i + 1;
        }}
        let counts = abng_decide_step(g);
        print(counts);
        print(abng_node_count(g));
        "#
    );
    assert_parity("decide_step fires Grow after observations", &body);
}

#[test]
fn parity_unfreeze_round_trip() {
    let body = r#"
        let g = abng_new(0);
        abng_force_freeze(g, 0);
        print(abng_is_frozen(g, 0));
        abng_unfreeze(g, 0);
        print(abng_is_frozen(g, 0));
        print(abng_chain_head(g));
    "#;
    assert_parity("freeze + unfreeze round trip", body);
}

#[test]
fn parity_double_run_decide_step() {
    let body = format!(
        r#"
        let g = abng_new(42);
        {POLICY_INSTALL}
        let i: i64 = 0;
        while i < 70 {{
            abng_observe(g, 0, 1.0);
            i = i + 1;
        }}
        let c1 = abng_decide_step(g);
        let c2 = abng_decide_step(g);
        print(c1);
        print(c2);
        print(abng_chain_head(g));
        "#
    );
    let a = run(Backend::Eval, &body, 0);
    let b = run(Backend::Eval, &body, 0);
    assert_eq!(a, b, "eval double-run differs");
    let c = run(Backend::Mir, &body, 0);
    let d = run(Backend::Mir, &body, 0);
    assert_eq!(c, d, "MIR double-run differs");
    assert_eq!(a, c, "eval ↔ MIR differs");
}
