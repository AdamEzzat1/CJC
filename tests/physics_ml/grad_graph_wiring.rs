//! Phase 3c — wiring tests for `grad_graph_*` builtins.
//!
//! Each test runs a tiny `.cjcl` snippet through both `cjc-eval` (AST) and
//! `cjc-mir-exec` (MIR), and asserts byte-identical printed output. This
//! validates that the satellite dispatch in `cjc-ad::dispatch_grad_graph`
//! is correctly routed from both executors and produces deterministic
//! results across the two pipelines.
//!
//! **Thread-local note:** the ambient `GradGraph` lives in a `thread_local!`
//! shared by both backends within a single test thread. Every snippet
//! therefore begins with `grad_graph_new()` to reset state — this is the
//! exact contract the brief asks user `.cjcl` code to honor.

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

// ─── Construction ──────────────────────────────────────────────────────

#[test]
fn wiring_grad_graph_new_resets() {
    assert_parity(
        "grad_graph_new resets",
        r#"
        grad_graph_new();
        grad_graph_param(Tensor.from_vec([1.0, 2.0], [2]));
        grad_graph_new();
        print(grad_graph_len());
        "#,
    );
}

#[test]
fn wiring_grad_graph_param_returns_index() {
    assert_parity(
        "param returns index",
        r#"
        grad_graph_new();
        let i = grad_graph_param(Tensor.from_vec([3.0, 4.0], [2]));
        print(i);
        print(grad_graph_len());
        "#,
    );
}

#[test]
fn wiring_grad_graph_input_returns_index() {
    assert_parity(
        "input returns index",
        r#"
        grad_graph_new();
        let i = grad_graph_input(Tensor.from_vec([0.5], [1]));
        print(i);
        "#,
    );
}

#[test]
fn wiring_grad_graph_const_returns_index() {
    assert_parity(
        "const returns index",
        r#"
        grad_graph_new();
        let i = grad_graph_const(Tensor.from_vec([1.5], [1]));
        print(i);
        "#,
    );
}

// ─── Pointwise ops (forward values) ────────────────────────────────────

#[test]
fn wiring_grad_graph_add() {
    assert_parity(
        "add forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let s = grad_graph_add(a, b);
        let t = grad_graph_forward(s);
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_sub() {
    assert_parity(
        "sub forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([5.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let t = grad_graph_forward(grad_graph_sub(a, b));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_mul() {
    assert_parity(
        "mul forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let t = grad_graph_forward(grad_graph_mul(a, b));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_div() {
    assert_parity(
        "div forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([8.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let t = grad_graph_forward(grad_graph_div(a, b));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_neg() {
    assert_parity(
        "neg forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.5], [1]));
        let t = grad_graph_forward(grad_graph_neg(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_scalar_mul() {
    assert_parity(
        "scalar_mul forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([2.0, 3.0], [2]));
        let s = grad_graph_scalar_mul(a, 2.5);
        let t = grad_graph_forward(s);
        print(t.get([0]));
        print(t.get([1]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_pow() {
    assert_parity(
        "pow forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let p = grad_graph_pow(a, 2.0);
        let t = grad_graph_forward(p);
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_exp() {
    assert_parity(
        "exp forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([0.0], [1]));
        let t = grad_graph_forward(grad_graph_exp(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_ln() {
    assert_parity(
        "ln forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([1.0], [1]));
        let t = grad_graph_forward(grad_graph_ln(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_sqrt() {
    assert_parity(
        "sqrt forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let t = grad_graph_forward(grad_graph_sqrt(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_sin() {
    assert_parity(
        "sin forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([0.0], [1]));
        let t = grad_graph_forward(grad_graph_sin(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_cos() {
    assert_parity(
        "cos forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([0.0], [1]));
        let t = grad_graph_forward(grad_graph_cos(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_tanh() {
    assert_parity(
        "tanh forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([0.0], [1]));
        let t = grad_graph_forward(grad_graph_tanh(a));
        print(t.get([0]));
        "#,
    );
}

// ─── Reductions / matmul ────────────────────────────────────────────

#[test]
fn wiring_grad_graph_sum() {
    assert_parity(
        "sum forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([1.0, 2.0, 3.0], [3]));
        let t = grad_graph_forward(grad_graph_sum(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_mean() {
    assert_parity(
        "mean forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([2.0, 4.0, 6.0], [3]));
        let t = grad_graph_forward(grad_graph_mean(a));
        print(t.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_matmul() {
    assert_parity(
        "matmul forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]));
        let b = grad_graph_param(Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [2, 2]));
        let m = grad_graph_matmul(a, b);
        let t = grad_graph_forward(m);
        print(t.get([0, 0]));
        print(t.get([0, 1]));
        print(t.get([1, 0]));
        print(t.get([1, 1]));
        "#,
    );
}

// ─── MLP fused layer ────────────────────────────────────────────────

#[test]
fn wiring_grad_graph_mlp_layer_tanh() {
    // Fused: tanh(x @ Wᵀ + b). x: [1,2], W: [3,2] (out=3, in=2), b: [3].
    assert_parity(
        "mlp_layer tanh forward",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([0.5, 0.5], [1, 2]));
        let w = grad_graph_param(Tensor.from_vec([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2]));
        let b = grad_graph_param(Tensor.from_vec([0.0, 0.0, 0.0], [3]));
        let y = grad_graph_mlp_layer(x, w, b, "tanh");
        let t = grad_graph_forward(y);
        print(t.get([0, 0]));
        print(t.get([0, 1]));
        print(t.get([0, 2]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_mlp_layer_none_passes_through() {
    // Identity activation: y = x @ Wᵀ + b. Same shape.
    assert_parity(
        "mlp_layer none forward",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0], [1, 2]));
        let w = grad_graph_param(Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]));
        let b = grad_graph_param(Tensor.from_vec([0.5, -0.5], [2]));
        let y = grad_graph_mlp_layer(x, w, b, "none");
        let t = grad_graph_forward(y);
        print(t.get([0, 0]));
        print(t.get([0, 1]));
        "#,
    );
}

// ─── Backward / state ───────────────────────────────────────────────

#[test]
fn wiring_grad_graph_backward_simple() {
    // L = a*b ⇒ dL/da = b, dL/db = a.
    assert_parity(
        "backward simple",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let l = grad_graph_mul(a, b);
        grad_graph_zero_grad();
        grad_graph_backward(l);
        let ga = grad_graph_param_grad(a);
        let gb = grad_graph_param_grad(b);
        print(ga.get([0]));
        print(gb.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_set_tensor_then_forward() {
    // Mutate parameter, re-read tensor at the parameter node.
    assert_parity(
        "set_tensor then forward",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([1.0, 2.0], [2]));
        grad_graph_set_tensor(a, Tensor.from_vec([10.0, 20.0], [2]));
        let t = grad_graph_forward(a);
        print(t.get([0]));
        print(t.get([1]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_zero_grad_clears_after_backward() {
    // Backward then zero_grad ⇒ grad reads back as zeros.
    assert_parity(
        "zero_grad clears",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let l = grad_graph_mul(a, b);
        grad_graph_zero_grad();
        grad_graph_backward(l);
        grad_graph_zero_grad();
        let ga = grad_graph_param_grad(a);
        let gb = grad_graph_param_grad(b);
        print(ga.get([0]));
        print(gb.get([0]));
        "#,
    );
}

#[test]
fn wiring_grad_graph_clip_grad_norm_returns_norm() {
    // Run backward producing known grads, then clip with very large max so
    // norm is unchanged but the function still returns a finite value.
    assert_parity(
        "clip_grad_norm returns",
        r#"
        grad_graph_new();
        let a = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let b = grad_graph_param(Tensor.from_vec([4.0], [1]));
        let l = grad_graph_mul(a, b);
        grad_graph_zero_grad();
        grad_graph_backward(l);
        let n = grad_graph_clip_grad_norm(1000.0);
        // Print only that the value is finite — bit-equality of a single
        // f64 print is what we're really gating on.
        print(n);
        "#,
    );
}

// ─── Sanity: the dispatch is actually registered ───────────────────

#[test]
fn wiring_dispatch_routes_through_both_executors() {
    // Tiny end-to-end: build, forward, backward, read grad — all in one
    // snippet using the full set of public verbs. Different shapes from
    // the unit tests so a stray hard-coded value would fail.
    assert_parity(
        "end-to-end mini",
        r#"
        grad_graph_new();
        let p = grad_graph_param(Tensor.from_vec([2.0, 3.0, 4.0], [3]));
        let s = grad_graph_sum(p);                    // forward = 9.0
        grad_graph_zero_grad();
        grad_graph_backward(s);
        let g = grad_graph_param_grad(p);             // [1, 1, 1]
        print(g.get([0]));
        print(g.get([1]));
        print(g.get([2]));
        let t = grad_graph_forward(s);
        print(t.get([0]));
        "#,
    );
}
