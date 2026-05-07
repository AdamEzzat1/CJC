//! Phase 0.3a — AST eval ↔ MIR exec parity tests for the leaf-head
//! builtins (`abng_set_leaf_head`, `abng_leaf_*`).

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

// ─── Configuration ────────────────────────────────────────────────

#[test]
fn parity_set_leaf_head_basic() {
    assert_parity(
        "set_leaf_head + leaf_param_count",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        print(abng_leaf_param_count(g, 0));
        "#,
    );
}

#[test]
fn parity_leaf_head_dims() {
    assert_parity(
        "leaf_head_dims encodes architecture",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([8.0, 4.0], [2]);
        abng_set_leaf_head(g, 3, hidden, 2, "relu");
        print(abng_leaf_head_dims(g));
        "#,
    );
}

// ─── Param read/write ─────────────────────────────────────────────

#[test]
fn parity_leaf_param_xavier_deterministic() {
    assert_parity(
        "Xavier init is deterministic across executors",
        r#"
        let g = abng_new(7);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        print(abng_leaf_param(g, 0, 0));
        print(abng_leaf_param(g, 0, 1));
        print(abng_leaf_param(g, 0, 2));
        print(abng_leaf_param(g, 0, 3));
        "#,
    );
}

#[test]
fn parity_leaf_set_param_writes_back() {
    assert_parity(
        "leaf_set_param round-trips a tensor",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        let new_w = Tensor.from_vec([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [4, 2]);
        abng_leaf_set_param(g, 0, 0, new_w);
        print(abng_leaf_param(g, 0, 0));
        "#,
    );
}

#[test]
fn parity_leaf_params_hash_changes_after_update() {
    assert_parity(
        "params hash changes on set_param",
        r#"
        let g = abng_new(0);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        let h0 = abng_leaf_params_hash(g, 0);
        let new_w = Tensor.from_vec([0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42], [4, 2]);
        abng_leaf_set_param(g, 0, 0, new_w);
        let h1 = abng_leaf_params_hash(g, 0);
        print(h0 == h1);
        "#,
    );
}

// ─── Forward into ambient GradGraph ───────────────────────────────

#[test]
fn parity_leaf_forward_output_value() {
    assert_parity(
        "leaf_forward output tensor matches across executors",
        r#"
        let g = abng_new(7);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        grad_graph_new();
        let x_idx = grad_graph_input(Tensor.from_vec([0.5, -0.5], [1, 2]));
        let result = abng_leaf_forward(g, 0, x_idx);
        // result[0] is y_idx; read its tensor.
        let y_idx = result[0];
        print(grad_graph_forward(y_idx));
        "#,
    );
}

#[test]
fn parity_full_train_step_chain_head() {
    // Full forward + backward + manual SGD + writeback. The chain head
    // must agree across executors.
    assert_parity(
        "full train step produces matching chain head",
        r#"
        let g = abng_new(42);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        grad_graph_new();
        let x_idx = grad_graph_input(Tensor.from_vec([0.5, 0.3], [1, 2]));
        let target_idx = grad_graph_input(Tensor.from_vec([1.0], [1, 1]));
        let result = abng_leaf_forward(g, 0, x_idx);
        let y_idx = result[0];
        let diff = grad_graph_sub(y_idx, target_idx);
        let sq = grad_graph_mul(diff, diff);
        let loss = grad_graph_sum(sq);
        // Build the params index array for backward_collect.
        let params = [result[1], result[2], result[3], result[4]];
        let grads = grad_graph_backward_collect(loss, params);
        // SGD step on each param: w' = w - 0.01 * grad.
        let lr = 0.01;
        let n = abng_leaf_param_count(g, 0);
        let mut k: i64 = 0;
        while k < n {
            let w = abng_leaf_param(g, 0, k);
            let grad = grads[k];
            let new_w = w - lr * grad;
            abng_leaf_set_param(g, 0, k, new_w);
            k = k + 1;
        }
        print(abng_chain_head(g));
        "#,
    );
}

#[test]
fn parity_double_run_train_step_chain_head() {
    let body = r#"
        let g = abng_new(99);
        let hidden = Tensor.from_vec([4.0], [1]);
        abng_set_leaf_head(g, 2, hidden, 1, "tanh");
        grad_graph_new();
        let x_idx = grad_graph_input(Tensor.from_vec([0.1, 0.9], [1, 2]));
        let target_idx = grad_graph_input(Tensor.from_vec([0.5], [1, 1]));
        let result = abng_leaf_forward(g, 0, x_idx);
        let y_idx = result[0];
        let diff = grad_graph_sub(y_idx, target_idx);
        let sq = grad_graph_mul(diff, diff);
        let loss = grad_graph_sum(sq);
        let params = [result[1], result[2], result[3], result[4]];
        let grads = grad_graph_backward_collect(loss, params);
        let n = abng_leaf_param_count(g, 0);
        let mut k: i64 = 0;
        while k < n {
            let w = abng_leaf_param(g, 0, k);
            let grad = grads[k];
            let new_w = w - 0.005 * grad;
            abng_leaf_set_param(g, 0, k, new_w);
            k = k + 1;
        }
        print(abng_chain_head(g));
    "#;
    let a = run(Backend::Eval, body, 0);
    let b = run(Backend::Eval, body, 0);
    assert_eq!(a, b, "eval double-run differs");
    let c = run(Backend::Mir, body, 0);
    let d = run(Backend::Mir, body, 0);
    assert_eq!(c, d, "MIR double-run differs");
    assert_eq!(a, c, "eval ↔ MIR differs");
}
