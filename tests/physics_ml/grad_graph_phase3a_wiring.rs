//! Phase 3a — wiring tests for the transformer-backbone `grad_graph_*`
//! builtins added on top of the Phase 3c surface.
//!
//! Each test runs a tiny `.cjcl` snippet through both `cjc-eval` (AST) and
//! `cjc-mir-exec` (MIR) and asserts byte-identical printed output. The
//! contract: every dispatch arm in `cjc_ad::dispatch_grad_graph` must
//! produce deterministic, executor-independent results.
//!
//! Phase 3a covers:
//!   - `grad_graph_softmax`        (Kahan-stable along last axis)
//!   - `grad_graph_cross_entropy`  (Kahan + log-sum-exp; scalar output)
//!   - `grad_graph_layer_norm`     (Kahan mean+var, eps=1e-5)
//!   - `grad_graph_gelu`           (pointwise activation)
//!   - `grad_graph_silu`           (pointwise activation)
//!   - `grad_graph_reshape`        (shape-only; element count must match)

#![allow(clippy::needless_raw_string_hashes)]

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
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed for snippet:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed for snippet:\n{src}\nerror: {e:?}"));
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

// ─── grad_graph_softmax ────────────────────────────────────────────

#[test]
fn wiring_softmax_outputs_probability_distribution() {
    assert_parity(
        "softmax probabilities sum to 1",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]));
        let s = grad_graph_softmax(x);
        let t = grad_graph_forward(s);
        print(t);
        "#,
    );
}

#[test]
fn wiring_softmax_chains_with_param() {
    assert_parity(
        "softmax over a param node",
        r#"
        grad_graph_new();
        let p = grad_graph_param(Tensor.from_vec([0.5, -0.5, 0.0], [3]));
        let s = grad_graph_softmax(p);
        print(grad_graph_forward(s));
        "#,
    );
}

// ─── grad_graph_cross_entropy ──────────────────────────────────────

#[test]
fn wiring_cross_entropy_returns_scalar() {
    assert_parity(
        "cross_entropy is scalar [1]",
        r#"
        grad_graph_new();
        let logits = grad_graph_input(Tensor.from_vec([1.0, 2.0, 0.5], [3]));
        let targets = grad_graph_input(Tensor.from_vec([0.0, 1.0, 0.0], [3]));
        let loss = grad_graph_cross_entropy(logits, targets);
        print(grad_graph_forward(loss));
        "#,
    );
}

#[test]
fn wiring_cross_entropy_backward_runs() {
    // CE → scalar → backward must succeed.
    assert_parity(
        "cross_entropy backward end-to-end",
        r#"
        grad_graph_new();
        let logits = grad_graph_param(Tensor.from_vec([0.5, 1.5, 0.0, -0.5], [4]));
        let targets = grad_graph_input(Tensor.from_vec([0.0, 1.0, 0.0, 0.0], [4]));
        let loss = grad_graph_cross_entropy(logits, targets);
        grad_graph_zero_grad();
        grad_graph_backward(loss);
        print(grad_graph_param_grad(logits));
        "#,
    );
}

// ─── grad_graph_layer_norm ─────────────────────────────────────────

#[test]
fn wiring_layer_norm_preserves_shape() {
    assert_parity(
        "layer_norm shape-preserving",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));
        let n = grad_graph_layer_norm(x);
        print(grad_graph_forward(n));
        "#,
    );
}

// ─── grad_graph_gelu ───────────────────────────────────────────────

#[test]
fn wiring_gelu_pointwise() {
    assert_parity(
        "gelu pointwise",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([-2.0, -1.0, 0.0, 1.0, 2.0], [5]));
        let g = grad_graph_gelu(x);
        print(grad_graph_forward(g));
        "#,
    );
}

#[test]
fn wiring_gelu_in_mlp_layer_substitute() {
    assert_parity(
        "gelu chained with matmul",
        r#"
        grad_graph_new();
        let w = grad_graph_param(Tensor.from_vec([0.5, -0.5, 1.0, 0.0], [2, 2]));
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0], [1, 2]));
        let z = grad_graph_matmul(x, w);
        let h = grad_graph_gelu(z);
        print(grad_graph_forward(h));
        "#,
    );
}

// ─── grad_graph_silu ───────────────────────────────────────────────

#[test]
fn wiring_silu_pointwise() {
    assert_parity(
        "silu pointwise",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([-2.0, -1.0, 0.0, 1.0, 2.0], [5]));
        let s = grad_graph_silu(x);
        print(grad_graph_forward(s));
        "#,
    );
}

// ─── grad_graph_reshape ────────────────────────────────────────────

#[test]
fn wiring_reshape_2x3_to_3x2() {
    assert_parity(
        "reshape 2x3 -> 3x2",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));
        let r = grad_graph_reshape(x, [3, 2]);
        print(grad_graph_forward(r));
        "#,
    );
}

#[test]
fn wiring_reshape_flatten() {
    assert_parity(
        "reshape 2x3 -> 6",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));
        let r = grad_graph_reshape(x, [6]);
        print(grad_graph_forward(r));
        "#,
    );
}

#[test]
fn wiring_reshape_chained_with_softmax() {
    assert_parity(
        "reshape -> softmax",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]));
        let r = grad_graph_reshape(x, [4]);
        let s = grad_graph_softmax(r);
        print(grad_graph_forward(s));
        "#,
    );
}

// ─── End-to-end transformer-block-shaped pipeline ──────────────────

#[test]
fn wiring_phase3a_softmax_layernorm_gelu_pipeline() {
    // A composite snippet exercising every Phase 3a builtin in series.
    // Forward only — we already cover backward above for cross_entropy.
    assert_parity(
        "softmax + layer_norm + gelu + reshape pipeline",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]));
        let n = grad_graph_layer_norm(x);
        let h = grad_graph_gelu(n);
        let s = grad_graph_silu(h);
        let r = grad_graph_reshape(s, [6]);
        let p = grad_graph_softmax(r);
        print(grad_graph_forward(p));
        "#,
    );
}
