//! Phase F wiring tests — creation-site allocation accounting.
//!
//! Mirrors the A1 precedent (`test_tensor_fp_accounting.rs`): the
//! recorded label must see Rc-buffer allocation at MODEL prices, the
//! prices must be exact for known programs, and instrumentation must
//! never perturb program output.

use cjc_mir_exec::run_program_instrumented;

const SEED: u64 = 42;

fn instrumented_alloc_total(src: &str) -> u64 {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "{:?}", diags.diagnostics);
    let (_val, _exec, events) = run_program_instrumented(&ast, SEED).expect("instrumented run");
    events.iter().map(|e| e.alloc_bytes_in_window).sum()
}

#[test]
fn churn_loop_prices_array_and_tuple_literals_exactly() {
    // The mem_grad shape: per iteration one 2-element array literal
    // (2 × 16 B) and one 2-element tuple literal (2 × 16 B) = 64 B.
    let src = r#"
fn churn(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let cell: Any = [i, i + 1];
        let pair: Any = (i, i * 2);
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(churn(10));
"#;
    let total = instrumented_alloc_total(src);
    assert_eq!(total, 10 * 64, "10 iterations × (32 array + 32 tuple)");
}

#[test]
fn alloc_volume_scales_with_iteration_count() {
    // The label-fix requirement: same per-iteration churn, more
    // iterations → proportionally more recorded allocation. This is
    // the gradient the mem_grad_a{1..5} family needs (the old label's
    // measured ceiling was 0.0078 BECAUSE it could not see this).
    let prog = |n: u64| {
        format!(
            r#"
fn churn(n: i64) -> i64 {{
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {{
        let cell: Any = [i, i + 1];
        total = total + i;
        i = i + 1;
    }}
    return total;
}}
print(churn({n}));
"#
        )
    };
    let small = instrumented_alloc_total(&prog(8));
    let large = instrumented_alloc_total(&prog(64));
    assert_eq!(small, 8 * 32);
    assert_eq!(large, 64 * 32);
    assert_eq!(large, small * 8, "volume must scale linearly");
}

#[test]
fn tensor_binops_price_result_buffers() {
    // u = t * 0.999 over a 4×4 tensor materializes a 16-element f64
    // result per iteration: 16 × 8 = 128 B. The builder loop's
    // array_push calls add 16 B each (16 pushes), and the final
    // from_vec / initial t * 1.0 also materialize tensors — assert the
    // EXACT total so price drift fails loudly.
    let src = r#"
fn build(n: i64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, 0.5 * float(i % 7));
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn work(t: Tensor, iters: i64) -> f64 {
    let mut u: Tensor = t * 1.0;
    let mut i: i64 = 0;
    while i < iters {
        u = u * 0.999;
        i = i + 1;
    }
    return u.sum();
}
let t: Tensor = build(4);
print(work(t, 5));
"#;
    let total = instrumented_alloc_total(src);
    // build(4): empty array literal (0 elems → 0) + 16 array_push × 16 B
    // + the `[n, n]` shape literal in the from_vec call (2 × 16 B).
    // `Tensor.from_vec` itself is deliberately unpriced (curated
    // under-count; static-method dispatch is not a tensor-receiver
    // method call).
    let build_bytes = 16 * 16 + 32;
    // work: t*1.0 (128) + 5 × u*0.999 (128 each) = 6 × 128; u.sum()
    // is a reduction → 0.
    let work_bytes = 6 * 128;
    assert_eq!(total, build_bytes + work_bytes);
}

#[test]
fn scalar_fp_programs_record_zero_alloc() {
    // Dense scalar FP allocates nothing the model prices — the alloc
    // label must read 0, keeping FP-hot programs distinguishable from
    // churn programs (the variance the memory head needs).
    let src = r#"
fn horner(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + x * 0.5;
        i = i + 1;
    }
    return acc;
}
print(horner(1.01, 100));
"#;
    assert_eq!(instrumented_alloc_total(src), 0);
}

#[test]
fn instrumentation_does_not_perturb_output() {
    // The transparency invariant, re-asserted for the new counter:
    // instrumented and uninstrumented runs produce byte-identical
    // output on an allocation-heavy program.
    let src = r#"
fn churn(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let cell: Any = [i, i + 1, i + 2];
        total = total + i % 5;
        i = i + 1;
    }
    return total;
}
print(churn(200));
"#;
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let (_val, exec, events) = run_program_instrumented(&ast, SEED).unwrap();
    let instrumented_output = exec.output.clone();
    assert!(events.iter().map(|e| e.alloc_bytes_in_window).sum::<u64>() > 0);

    let mut plain = cjc_mir_exec::MirExecutor::new(SEED);
    plain.scan_ast_imports(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mir = h2m.lower_program(&hir);
    plain.exec(&mir).unwrap();
    assert_eq!(plain.output, instrumented_output);
}
