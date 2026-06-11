//! Phase C — energy-selector GATES (research doc §5: the four gates
//! that must exist before plans get aggressive) plus property/fuzz
//! layers per the test-discipline contract.
//!
//! Gates run against the COMMITTED CPB1 energy bundle — they prove the
//! shipped artifact + selector composition, not a synthetic head.

use std::path::Path;

use cjc_cana::legality::{
    LegalityGate, LegalityVerdict, PassSequence, PerPassLegalityGate, ProposedPass,
};
use cjc_cana::pinn_energy_v1::{EnergyQuery, PinnEnergyV1};
use cjc_cana::plan_selector::{select_argmin, PassPlanSelector};
use cjc_cana_compress::energy_bundle::read_energy_bundle;
use cjc_mir::optimize::{optimize_program_with_plan, PassPlan};
use proptest::prelude::*;

const BUNDLE_PATH: &str = "bench_results/cana_train_pinn/pinn_energy_v1.cpb";
const SEED: u64 = 42;

fn committed_head() -> PinnEnergyV1 {
    read_energy_bundle(Path::new(BUNDLE_PATH))
        .expect("committed CPB1 bundle must load")
        .head
}

fn lower(src: &str) -> (cjc_ast::Program, cjc_mir::MirProgram, cjc_cana::features::CanaFeatures) {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse: {:?}", diags.diagnostics);
    let mut al = cjc_hir::AstLowering::new();
    let hir = al.lower_program(&ast);
    let mut h2m = cjc_mir::HirToMir::new();
    let mir = h2m.lower_program(&hir);
    let features = cjc_cana::analyze_program(&mir).features;
    (ast, mir, features)
}

/// Representative program mix: int loop, scalar FP, tensor, multi-fn.
const PROGRAMS: &[(&str, &str)] = &[
    (
        "int_loop",
        r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(sum_to(1000));
"#,
    ),
    (
        "scalar_fp",
        r#"
fn horner(x: f64, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + x * 0.5;
        i = i + 1;
    }
    return acc;
}
print(horner(1.01, 500));
"#,
    ),
    (
        "tensor_mm",
        r#"
fn build(n: i64, scale: f64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n * n {
        buf = array_push(buf, scale * float(i % 7) + 0.25);
        i = i + 1;
    }
    return Tensor.from_vec(buf, [n, n]);
}
fn mm_hot(a: Tensor, b: Tensor, iters: i64) -> f64 {
    let mut s: f64 = 0.0;
    let mut i: i64 = 0;
    while i < iters {
        let c: Tensor = matmul(a, b);
        s = c.sum();
        i = i + 1;
    }
    return s;
}
let a: Tensor = build(8, 0.5);
let b: Tensor = build(8, 0.25);
print(mm_hot(a, b, 10));
"#,
    ),
    (
        "multi_fn",
        r#"
fn add1(x: i64) -> i64 { return x + 1; }
fn mul2(x: i64) -> i64 { return x * 2; }
fn driver(n: i64) -> i64 {
    let mut r: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        r = mul2(add1(r)) - r;
        i = i + 1;
    }
    return r;
}
print(driver(50));
"#,
    ),
];

// ---------------------------------------------------------------------------
// Gate 1 — legality of every selected pass (independent re-verification)
// ---------------------------------------------------------------------------

#[test]
fn gate1_every_selected_pass_is_individually_legal() {
    let selector = PassPlanSelector::new(committed_head()).unwrap();
    let gate = PerPassLegalityGate::new();
    for (name, src) in PROGRAMS {
        let (_ast, mir, features) = lower(src);
        let report = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        for (fn_name, passes) in &report.plan.per_function {
            for p in passes {
                let mut seq = PassSequence::default();
                seq.per_function
                    .insert(fn_name.clone(), vec![ProposedPass::Run(p.clone())]);
                assert!(
                    matches!(gate.verify(&mir, &seq, &features), LegalityVerdict::Approved),
                    "{name}/{fn_name}: selected pass {p} fails independent legality re-check"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gate 2 — selector determinism (double-run identity)
// ---------------------------------------------------------------------------

#[test]
fn gate2_selection_is_double_run_identical() {
    let selector = PassPlanSelector::new(committed_head()).unwrap();
    let gate = PerPassLegalityGate::new();
    for (name, src) in PROGRAMS {
        let (_ast, mir, features) = lower(src);
        let first = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        let again = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        assert_eq!(first, again, "{name}: selection must be deterministic");
    }
}

// ---------------------------------------------------------------------------
// Gate 3 — selector-on / selector-off output parity
// ---------------------------------------------------------------------------

#[test]
fn gate3_selected_plan_preserves_program_output() {
    let selector = PassPlanSelector::new(committed_head()).unwrap();
    let gate = PerPassLegalityGate::new();
    for (name, src) in PROGRAMS {
        let (ast, mir, features) = lower(src);

        // Reference: AST tree-walk.
        let mut interp = cjc_eval::Interpreter::new(SEED);
        interp.exec(&ast).expect("eval");

        // Selector-ON: MIR-exec over the selected plan.
        let report = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        let mut optimized = optimize_program_with_plan(&mir, &report.plan);
        cjc_mir::escape::annotate_program(&mut optimized);
        let mut exec = cjc_mir_exec::MirExecutor::new(SEED);
        exec.scan_ast_imports(&ast);
        exec.exec(&optimized).expect("mir-exec under selected plan");

        assert_eq!(
            interp.output, exec.output,
            "{name}: selected plan changed observable behavior"
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 4 — never worse than the ranked candidate (predicted criterion)
// ---------------------------------------------------------------------------

#[test]
fn gate4_chosen_never_worse_than_ranked_predicted() {
    let selector = PassPlanSelector::new(committed_head()).unwrap();
    let gate = PerPassLegalityGate::new();
    for (name, src) in PROGRAMS {
        let (_ast, mir, features) = lower(src);
        let report = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        for (fn_name, sel) in &report.per_function {
            assert!(
                sel.chosen.predicted_ln_score <= sel.ranked_predicted,
                "{name}/{fn_name}: argmin over a set containing the ranked plan regressed"
            );
            assert_eq!(sel.candidates_scored, 10, "{name}/{fn_name}: candidate set drifted");
        }
    }
}

// ---------------------------------------------------------------------------
// Trap test — explicit per-function entries (absence-means-default)
// ---------------------------------------------------------------------------

#[test]
fn selected_plan_has_explicit_entry_for_every_featurized_function() {
    let selector = PassPlanSelector::new(committed_head()).unwrap();
    let gate = PerPassLegalityGate::new();
    for (name, src) in PROGRAMS {
        let (_ast, mir, features) = lower(src);
        let report = selector.select(&mir, &features, &PassPlan::empty(), &gate);
        for fn_name in features.per_fn.keys() {
            assert!(
                report.plan.per_function.contains_key(fn_name),
                "{name}/{fn_name}: missing explicit entry — would silently run the FULL default sequence"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property layer — argmin total order
// ---------------------------------------------------------------------------

proptest! {
    /// Exactly one winner for any non-empty candidate list (NaN
    /// included — total_cmp is total), and the winner is invariant
    /// under permutation.
    #[test]
    fn argmin_unique_and_permutation_invariant(
        mut scored in proptest::collection::vec(any::<f64>(), 1..32)
    ) {
        let pairs: Vec<(f64, u32)> = scored.iter().copied().zip(0u32..).map(|(p, i)| (p, i)).collect();
        let winner = select_argmin(&pairs);
        prop_assert!(winner.is_some());
        let mut reversed = pairs.clone();
        reversed.reverse();
        prop_assert_eq!(select_argmin(&reversed), winner);
        // Stability under a rotation too.
        let mut rotated = pairs.clone();
        rotated.rotate_left(scored.len() / 2);
        prop_assert_eq!(select_argmin(&rotated), winner);
        scored.clear();
    }
}

// ---------------------------------------------------------------------------
// Fuzz layer — scoring path under adversarial inputs
// ---------------------------------------------------------------------------

/// The committed head's prediction must stay finite (or the neutral 0)
/// for ANY u64 workload extremes — the selector sweeps real candidate
/// plans over real programs, but the query fields are saturating
/// integers that can legitimately hit u64::MAX.
#[test]
fn fuzz_committed_head_is_total_over_query_space() {
    let head = committed_head();
    bolero::check!()
        .with_type::<(u64, u64, u64, u64, u32, u32)>()
        .for_each(|&(flops, bytes, float_ops, nodes, counts, depth)| {
            let q = EnergyQuery {
                flops_estimate: flops,
                bytes_read_estimate: bytes,
                bytes_written_estimate: bytes.wrapping_mul(2),
                allocation_bytes_estimate: nodes,
                working_set_bytes_estimate: nodes.wrapping_add(bytes),
                float_ops_estimate: float_ops,
                mir_nodes_before: nodes,
                recommended_count: counts,
                dropped_count: counts % 7,
                pass_counts: head.pass_counts(["dce", "licm"]),
                countable_loop_count: flops % 1024,
                max_loop_depth: depth % 64,
                mir_nodes_after: nodes / 2,
            };
            let p = head.predict_ln_score(&q);
            assert!(p.is_finite(), "non-finite prediction for fuzzed query");
        });
}
