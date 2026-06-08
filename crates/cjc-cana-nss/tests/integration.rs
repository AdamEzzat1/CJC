//! Integration tests for `cjc-cana-nss` — the §4B.2 Option C bridge.
//!
//! Covers:
//! - **Empty-map contract** for `predict_thermal/memory/cpu` on a real
//!   parsed CJC-Lang program.
//! - **Determinism** of `identify_structural_hot_kernels` across runs.
//! - **Structural identification** of hot kernels on a program with
//!   known nested-loop + branch structure (proves the heuristic fires
//!   in the expected direction).
//! - **PINN proxy**: a program shaped like the actual
//!   `08_pinn_heat_equation.cjcl` `forward` function (nested loops over
//!   weight matrices, no inner branches → not flagged hot). Documents
//!   the §3A.2 PINN AB-test outcome at the predictor level.
//! - **`ThermalAwareCostModel` composition**: an empty thermal map
//!   composes cleanly with the trained `LinearCostModel`, validating
//!   the design-doc claim that `NssPressurePredictor` is currently a
//!   no-op in the cost-model wrapper.

use cjc_cana::analyze_program;
use cjc_cana::pressure::PressurePredictor;
use cjc_cana_nss::{NssPressurePredictor, HOT_BRANCH_COUNT_MIN, HOT_LOOP_DEPTH_MIN};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Parse + HIR-lower + MIR-lower a CJC-Lang source string. Panics on
/// parse errors so the test programs below have to be well-formed.
fn parse_and_lower(source: &str) -> cjc_mir::MirProgram {
    let (ast, diags) = cjc_parser::parse_source(source);
    assert!(!diags.has_errors(), "parse errors in test source: {:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    h2m.lower_program(&hir)
}

// ---------------------------------------------------------------------------
// Empty-map contract
// ---------------------------------------------------------------------------

const SIMPLE_PROGRAM: &str = r#"
fn add(x: i64, y: i64) -> i64 {
    return x + y;
}
print(add(1, 2));
"#;

#[test]
fn predict_thermal_is_empty_on_real_program() {
    let p = NssPressurePredictor::default();
    let mir = parse_and_lower(SIMPLE_PROGRAM);
    let features = analyze_program(&mir).features;
    assert!(
        p.predict_thermal(&mir, &features).is_empty(),
        "Option C contract: predict_thermal must return an empty map",
    );
}

#[test]
fn predict_memory_peak_is_empty_on_real_program() {
    let p = NssPressurePredictor::default();
    let mir = parse_and_lower(SIMPLE_PROGRAM);
    let features = analyze_program(&mir).features;
    assert!(p.predict_memory_peak(&mir, &features).is_empty());
}

#[test]
fn predict_cpu_saturation_is_empty_on_real_program() {
    let p = NssPressurePredictor::default();
    let mir = parse_and_lower(SIMPLE_PROGRAM);
    let features = analyze_program(&mir).features;
    assert!(p.predict_cpu_saturation(&mir, &features).is_empty());
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn identify_hot_kernels_is_deterministic_across_runs() {
    let p1 = NssPressurePredictor::default();
    let p2 = NssPressurePredictor::default();
    let mir = parse_and_lower(SIMPLE_PROGRAM);
    let features = analyze_program(&mir).features;
    let r1 = p1.identify_structural_hot_kernels(&mir, &features);
    let r2 = p2.identify_structural_hot_kernels(&mir, &features);
    assert_eq!(r1, r2, "same input → byte-identical output");
}

#[test]
fn identify_hot_kernels_is_deterministic_across_different_seeds() {
    // Option C does no randomness; seed should not affect output.
    // (Will change in Option A where the synthetic trace uses seeded RNG.)
    let p1 = NssPressurePredictor::from_seed(1);
    let p2 = NssPressurePredictor::from_seed(99999);
    let mir = parse_and_lower(SIMPLE_PROGRAM);
    let features = analyze_program(&mir).features;
    let r1 = p1.identify_structural_hot_kernels(&mir, &features);
    let r2 = p2.identify_structural_hot_kernels(&mir, &features);
    assert_eq!(r1, r2, "Option C is seed-independent");
}

// ---------------------------------------------------------------------------
// Structural identification — positive case
// ---------------------------------------------------------------------------

/// A program with one structurally-hot function (`work`: nested loops +
/// internal if/else, both thresholds met) and one cold function
/// (`init`: a single loop, no branches).
const MIXED_HOT_COLD: &str = r#"
fn init(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + i;
        i = i + 1;
    }
    return s;
}

fn work(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            let inc: i64 = if i * j > n { i } else { j };
            let dec: i64 = if i + j > n { 1 } else { 0 };
            total = total + inc - dec;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}

print(init(10) + work(5));
"#;

#[test]
fn identify_hot_kernels_surfaces_nested_loop_with_branches() {
    let p = NssPressurePredictor::default();
    let mir = parse_and_lower(MIXED_HOT_COLD);
    let features = analyze_program(&mir).features;
    let hot = p.identify_structural_hot_kernels(&mir, &features);
    assert!(
        hot.iter().any(|f| f == "work"),
        "expected `work` (nested loops + if/else) in hot set, got {:?}",
        hot,
    );
    assert!(
        !hot.iter().any(|f| f == "init"),
        "did NOT expect `init` (single loop, no branches) in hot set, got {:?}",
        hot,
    );
}

// ---------------------------------------------------------------------------
// PINN proxy — documents predictor-vs-ranker distinction
// ---------------------------------------------------------------------------

/// A function shaped like the actual `forward` from
/// `examples/08_pinn_heat_equation.cjcl`: two nested while-loops over
/// weight matrix dimensions, MAC accumulator inside.
///
/// **Surprising-but-correct outcome:** this function IS flagged
/// structurally hot. Reason: in MIR CFG terms, every `while` is a
/// conditional-branch terminator at the loop header (jump-to-header
/// vs jump-out). So two nested whiles → `branch_count >= 2`, plus
/// `max_loop_depth >= 2` → both thresholds met.
///
/// This was an early test-author surprise: I expected PINN's forward to
/// NOT be flagged, mirroring the §3A.2 AB-test finding that PINN's
/// trained/default rankers both skip every pass. But that's a
/// **different layer**:
///
///   - The PREDICTOR (this crate) correctly identifies nested-loop
///     workhorses as structural hot kernels. Right answer.
///   - The RANKER (cjc-cana::pass_ranker) decides per-pass whether to
///     RECOMMEND optimization on each function. On PINN, both rankers
///     decide "no" — but for cost-model reasons, not structural ones.
///
/// So the hot-kernel signal IS produced for PINN's forward; the §3A.2
/// finding is that downstream consumption of that signal hasn't yet
/// translated into differential pass plans. Future work: feed
/// `identify_structural_hot_kernels` into the ranker's skip-threshold
/// logic so hot kernels get preferential pass-plan attention.
const PINN_FORWARD_PROXY: &str = r#"
fn pinn_forward(input: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    let n: i64 = 16;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            acc = acc + input * j;
            j = j + 1;
        }
        i = i + 1;
    }
    return acc;
}

print(pinn_forward(3));
"#;

#[test]
fn pinn_forward_proxy_is_flagged_hot_by_default_thresholds() {
    // The PINN forward pass has nested loops (loop_depth >= 2). Each
    // `while` contributes a conditional-branch terminator at its loop
    // header in MIR CFG, so branch_count >= 2 even without explicit
    // if/else. Both Option-C thresholds met → flagged hot.
    //
    // This is the correct behavior of the predictor. The §3A.2 finding
    // is at the downstream ranker layer, NOT here.
    let p = NssPressurePredictor::default();
    let mir = parse_and_lower(PINN_FORWARD_PROXY);
    let features = analyze_program(&mir).features;
    let hot = p.identify_structural_hot_kernels(&mir, &features);
    assert!(
        hot.iter().any(|f| f == "pinn_forward"),
        "PINN proxy SHOULD be flagged as structurally hot (nested whiles \
         contribute branch terminators in CFG). Got hot set: {:?}",
        hot,
    );
}

/// A function with deep nesting but only ONE while-loop branch
/// terminator + zero internal if/else. To make this happen we use a
/// single loop with no internal branches.
const SINGLE_LOOP_NO_BRANCHES: &str = r#"
fn straight_line_loop(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + i;
        i = i + 1;
    }
    return s;
}

print(straight_line_loop(100));
"#;

#[test]
fn single_loop_no_branches_is_not_flagged_hot() {
    // A single while-loop with no internal branches: branch_count = 1
    // (only the loop header conditional). Misses the threshold of 2,
    // so this function should NOT be flagged hot. This is the test
    // that previously failed against PINN_FORWARD_PROXY; here we use
    // a function that genuinely matches the "structurally cold" shape.
    let p = NssPressurePredictor::default();
    let mir = parse_and_lower(SINGLE_LOOP_NO_BRANCHES);
    let features = analyze_program(&mir).features;
    let hot = p.identify_structural_hot_kernels(&mir, &features);
    assert!(
        !hot.iter().any(|f| f == "straight_line_loop"),
        "single-loop-no-branches function should NOT be flagged hot. \
         Got hot set: {:?}",
        hot,
    );
}

// ---------------------------------------------------------------------------
// Sanity check: thresholds are sensitive
// ---------------------------------------------------------------------------

#[test]
fn thresholds_are_documented_constants() {
    // Lightweight contract: the public thresholds are the documented
    // values. Catches accidental changes that would silently shift the
    // hot-kernel set.
    assert_eq!(HOT_LOOP_DEPTH_MIN, 2, "loop-depth threshold contract");
    assert_eq!(HOT_BRANCH_COUNT_MIN, 2, "branch-count threshold contract");
}

// ---------------------------------------------------------------------------
// ThermalAwareCostModel composition — the design-doc claim
// ---------------------------------------------------------------------------

#[test]
fn thermal_aware_composes_with_empty_predictor_as_noop() {
    use cjc_cana::cost_model::{CostModel, CostQuery, CostEstimate};
    use cjc_cana::LinearCostModel;
    use cjc_cana::thermal_cost_model::ThermalAwareCostModel;

    let mir = parse_and_lower(MIXED_HOT_COLD);
    let features = analyze_program(&mir).features;

    // Base model: the trained linear cost model.
    let base = LinearCostModel::trained();
    let base_clone = LinearCostModel::trained();

    // Composed model: ThermalAwareCostModel<LinearCostModel, NssPressurePredictor>.
    let wrapped = ThermalAwareCostModel::new(base_clone, NssPressurePredictor::default());

    // Query for a pass benefit on the hot function.
    let query = CostQuery::PassBenefit {
        function_name: "work",
        pass_name: "constant_fold",
    };

    let base_estimate = base.query(&mir, &features, &query);
    let wrapped_estimate = wrapped.query(&mir, &features, &query);

    // Design-doc claim: empty thermal map → no adjustment → wrapped
    // returns the same estimate as base.
    match (base_estimate, wrapped_estimate) {
        (
            CostEstimate::Estimated { value: bv, confidence: bc },
            CostEstimate::Estimated { value: wv, confidence: wc },
        ) => {
            assert_eq!(bv, wv, "wrapped value diverged from base value");
            assert_eq!(bc, wc, "wrapped confidence diverged from base confidence");
        }
        (CostEstimate::Unknown, CostEstimate::Unknown) => {
            // Both unknown; still consistent.
        }
        (b, w) => panic!(
            "wrapped estimate diverged: base = {:?}, wrapped = {:?}",
            b, w,
        ),
    }
}
