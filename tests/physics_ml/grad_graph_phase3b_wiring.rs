//! Phase 3b — wiring tests for the array-arg + state-recovery
//! `grad_graph_*` builtins added on top of Phase 3a's transformer surface.
//!
//! Each test runs a tiny `.cjcl` snippet through both `cjc-eval` (AST) and
//! `cjc-mir-exec` (MIR) and asserts byte-identical printed output.
//!
//! Phase 3b covers:
//!   - `grad_graph_batch_norm`        (Kahan mean+var across full tensor)
//!   - `grad_graph_gather`            (1-D pick + 2-D row-select)
//!   - `grad_graph_cat`               (1-D / 2-D concatenation)
//!   - `grad_graph_reforward`         (graph-reuse path: set_tensor → reforward)
//!   - `grad_graph_backward_collect`  (zero_grad + backward + grad gather, in one)

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

// ─── grad_graph_batch_norm ─────────────────────────────────────────

#[test]
fn wiring_batch_norm_preserves_shape() {
    assert_parity(
        "batch_norm shape-preserving",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));
        let n = grad_graph_batch_norm(x);
        print(grad_graph_forward(n));
        "#,
    );
}

#[test]
fn wiring_batch_norm_chains_with_layer_norm() {
    // Stress: norm → norm. Both ops use Kahan, so chaining them is a real
    // determinism test (any reordering of summation drifts).
    assert_parity(
        "batch_norm + layer_norm chain",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([0.5, -0.5, 1.5, -1.5], [4]));
        let bn = grad_graph_batch_norm(x);
        let ln = grad_graph_layer_norm(bn);
        print(grad_graph_forward(ln));
        "#,
    );
}

// ─── grad_graph_gather ─────────────────────────────────────────────

#[test]
fn wiring_gather_1d_picks_indices() {
    assert_parity(
        "gather 1-D pick",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec([10.0, 20.0, 30.0, 40.0, 50.0], [5]));
        let g = grad_graph_gather(x, [0, 2, 4], 0);
        print(grad_graph_forward(g));
        "#,
    );
}

#[test]
fn wiring_gather_2d_row_select() {
    assert_parity(
        "gather 2-D rows",
        r#"
        grad_graph_new();
        let x = grad_graph_input(Tensor.from_vec(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]));
        let g = grad_graph_gather(x, [0, 2], 0);
        print(grad_graph_forward(g));
        "#,
    );
}

// ─── grad_graph_cat ────────────────────────────────────────────────

#[test]
fn wiring_cat_two_1d_tensors() {
    assert_parity(
        "cat 1-D",
        r#"
        grad_graph_new();
        let a = grad_graph_input(Tensor.from_vec([1.0, 2.0], [2]));
        let b = grad_graph_input(Tensor.from_vec([3.0, 4.0, 5.0], [3]));
        let c = grad_graph_cat([a, b], 0);
        print(grad_graph_forward(c));
        "#,
    );
}

#[test]
fn wiring_cat_three_2d_axis0() {
    assert_parity(
        "cat three 2-D rows axis 0",
        r#"
        grad_graph_new();
        let a = grad_graph_input(Tensor.from_vec([1.0, 2.0], [1, 2]));
        let b = grad_graph_input(Tensor.from_vec([3.0, 4.0], [1, 2]));
        let c = grad_graph_input(Tensor.from_vec([5.0, 6.0], [1, 2]));
        let r = grad_graph_cat([a, b, c], 0);
        print(grad_graph_forward(r));
        "#,
    );
}

// ─── grad_graph_reforward ──────────────────────────────────────────

#[test]
fn wiring_reforward_after_set_tensor() {
    // Build x*y+z, set_tensor on x, reforward, observe new value.
    assert_parity(
        "reforward picks up set_tensor change",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([2.0], [1]));
        let y = grad_graph_param(Tensor.from_vec([3.0], [1]));
        let z = grad_graph_param(Tensor.from_vec([1.0], [1]));
        let xy = grad_graph_mul(x, y);
        let r = grad_graph_add(xy, z);
        print(grad_graph_forward(r));
        // Update x to 10, then reforward only the dependent ops (xy and r,
        // node indices 3..=4 — xy is op 3, r is op 4 since x/y/z are 0/1/2).
        grad_graph_set_tensor(x, Tensor.from_vec([10.0], [1]));
        grad_graph_reforward(3, 4);
        print(grad_graph_forward(r));
        "#,
    );
}

// ─── grad_graph_backward_collect ───────────────────────────────────

#[test]
fn wiring_backward_collect_returns_array_of_grads() {
    // Loss = sum(x*y), params = [x, y]. dL/dx = y, dL/dy = x.
    assert_parity(
        "backward_collect over two params",
        r#"
        grad_graph_new();
        let x = grad_graph_param(Tensor.from_vec([2.0, 3.0], [2]));
        let y = grad_graph_param(Tensor.from_vec([5.0, 7.0], [2]));
        let xy = grad_graph_mul(x, y);
        let loss = grad_graph_sum(xy);
        let grads = grad_graph_backward_collect(loss, [x, y]);
        print(grads);
        "#,
    );
}

// ─── End-to-end Phase 3b composite ─────────────────────────────────

#[test]
fn wiring_phase3b_full_pipeline() {
    // Two parameters, gather to select a subset, batch-norm,
    // concatenate with another input, sum-reduce, backward_collect.
    assert_parity(
        "gather + batch_norm + cat + backward_collect pipeline",
        r#"
        grad_graph_new();
        let p = grad_graph_param(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]));
        let q = grad_graph_input(Tensor.from_vec([0.5, 0.5], [2]));
        let g = grad_graph_gather(p, [0, 2, 4], 0);
        let n = grad_graph_batch_norm(g);
        let c = grad_graph_cat([n, q], 0);
        let loss = grad_graph_sum(c);
        let grads = grad_graph_backward_collect(loss, [p]);
        print(grads);
        "#,
    );
}
