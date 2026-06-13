//! Phase F1 wiring — the static creation-alloc feature must mirror the
//! Phase-F0 runtime label it exists to predict.
//!
//! The unit tests (`memory_proxy`, `pinn_memory_v1`) cover the pieces;
//! this proves the END-TO-END property: for the same programs, the
//! static `creation_alloc_bytes_estimate` (compile-time) moves in the
//! same direction as the runtime `alloc_bytes_in_window` total
//! (Phase F0). If they ever diverge in sign, the feature is lying to
//! the head and this gate fails.

use cjc_cana::analyze_program;
use cjc_cana::physical_cost::build_physical_query;
use cjc_mir_exec::run_program_instrumented;

const SEED: u64 = 42;

/// Sum of the static creation-alloc estimate across a program's
/// functions (neutral "dce" pass — the harness's row convention).
fn static_creation_estimate(src: &str) -> u64 {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "{:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = analyze_program(&mir).features;
    features
        .per_fn
        .iter()
        .map(|(name, ff)| build_physical_query(name, "dce", ff).creation_alloc_bytes_estimate)
        .fold(0u64, |a, b| a.saturating_add(b))
}

/// Sum of the runtime F0 allocation label across an instrumented run.
fn runtime_alloc_total(src: &str) -> u64 {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let (_v, _e, events) = run_program_instrumented(&ast, SEED).expect("instrumented run");
    events.iter().map(|e| e.alloc_bytes_in_window).sum()
}

const CHURN: &str = r#"
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
print(churn(200));
"#;

const SCALAR_FP: &str = r#"
fn horner(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + x * 0.5;
        i = i + 1;
    }
    return acc;
}
print(horner(1.01, 200));
"#;

#[test]
fn static_estimate_nonzero_exactly_when_runtime_label_is() {
    // Churn: both static and runtime see allocation.
    assert!(static_creation_estimate(CHURN) > 0, "churn must estimate >0");
    assert!(runtime_alloc_total(CHURN) > 0, "churn must record >0 at runtime");
    // Scalar FP: both see zero (no literals, no tensors — the feature
    // must NOT fire where the label doesn't, or it would mislead the
    // head on exactly the programs F0 distinguishes).
    assert_eq!(static_creation_estimate(SCALAR_FP), 0);
    assert_eq!(runtime_alloc_total(SCALAR_FP), 0);
}

#[test]
fn static_estimate_separates_literal_volume_at_one_site() {
    // Two churn shapes differing only in literal element count: the
    // static estimate must rank them the same way the runtime label
    // would (more slots → more bytes). This is the exact blindness the
    // pre-F1 `alloc_sites` count could not see.
    let small = r#"
fn f(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: Any = [i, i + 1];
        t = t + i;
        i = i + 1;
    }
    return t;
}
print(f(50));
"#;
    let large = r#"
fn f(n: i64) -> i64 {
    let mut t: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let a: Any = [i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7];
        t = t + i;
        i = i + 1;
    }
    return t;
}
print(f(50));
"#;
    let s_small = static_creation_estimate(small);
    let s_large = static_creation_estimate(large);
    assert!(s_large > s_small, "8-slot literal must estimate above 2-slot");
    // And the runtime label agrees (4× the slots → 4× the bytes,
    // amplification cancels in the ratio).
    let r_small = runtime_alloc_total(small);
    let r_large = runtime_alloc_total(large);
    assert_eq!(r_large, r_small * 4, "8 vs 2 slots = 4× runtime bytes");
}
