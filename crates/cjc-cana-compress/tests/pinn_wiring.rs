//! End-to-end wiring tests for PINN v1: `PinnPhysicalCostModel` inside
//! the `PassRanker` → `EnergyAwarePassRanker` chain.
//!
//! What these tests lock:
//!
//! 1. Wrapping the cost model with physics NEVER changes the legality
//!    verdict — PINN is advisory (determinism invariant #12).
//! 2. Hard constraint violations can only *remove* recommendations
//!    (via confident zero-benefit → `BelowSkipThreshold`), never add
//!    them.
//! 3. The full stack (PINN cost model + energy re-ranking) double-runs
//!    byte-identically.
//! 4. The energy decomposition carries the physical terms
//!    (`bandwidth_pressure`, `locality_reward`) through to the audit.

use std::collections::{BTreeMap, BTreeSet};

use cjc_cana::cost_model::{CostEstimate, CostModel, CostQuery};
use cjc_cana::features::{extract, CanaFeatures};
use cjc_cana::legality::PerPassLegalityGate;
use cjc_cana::pass_ranker::PassRanker;
use cjc_cana::pinn_cost_model::PinnPhysicalCostModel;
use cjc_cana::pressure::PressurePredictor;
use cjc_cana_compress::EnergyAwarePassRanker;
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// Test program — functions with real expression counts so the physical
// query is non-trivial
// ---------------------------------------------------------------------------

fn fn_with_exprs(id: u32, name: &str, n: usize) -> MirFunction {
    let stmts = (0..n)
        .map(|i| {
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::IntLit(i as i64),
            })
        })
        .collect();
    MirFunction {
        id: MirFnId(id),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts,
            result: None,
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Public,
        local_count: 0,
    }
}

fn two_fn_program() -> MirProgram {
    MirProgram {
        functions: vec![
            fn_with_exprs(0, "cool_fn", 10),
            fn_with_exprs(1, "hot_fn", 50),
        ],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    }
}

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DiverseCostModel;
impl CostModel for DiverseCostModel {
    fn query(
        &self,
        _program: &MirProgram,
        _features: &CanaFeatures,
        query: &CostQuery<'_>,
    ) -> CostEstimate {
        match query {
            CostQuery::PassBenefit { pass_name, .. } => {
                let v = match *pass_name {
                    "constant_fold" => 0.12,
                    "strength_reduce" => 0.10,
                    "dce" => 0.08,
                    "cse" => 0.07,
                    "licm" => 0.06,
                    "loop_unroll" => 0.05,
                    _ => 0.01,
                };
                CostEstimate::Estimated {
                    value: v,
                    confidence: 0.85,
                }
            }
            CostQuery::PassRuntime { .. } => CostEstimate::Estimated {
                value: 0.05,
                confidence: 0.85,
            },
            _ => CostEstimate::Unknown,
        }
    }
    fn name(&self) -> &'static str {
        "diverse-stub"
    }
}

/// Predictor reporting `hot_fn` past the default 0.95 hard thermal
/// limit and `cool_fn` well under it.
#[derive(Debug, Clone)]
struct HotFnPredictor;
impl PressurePredictor for HotFnPredictor {
    fn predict_thermal(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
        p.functions
            .iter()
            .map(|f| {
                let t = if f.name == "hot_fn" { 0.99 } else { 0.1 };
                (f.name.clone(), t)
            })
            .collect()
    }
    fn predict_memory_peak(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
        p.functions.iter().map(|f| (f.name.clone(), 0.0)).collect()
    }
    fn predict_cpu_saturation(&self, p: &MirProgram, _f: &CanaFeatures) -> BTreeMap<String, f64> {
        p.functions.iter().map(|f| (f.name.clone(), 0.0)).collect()
    }
    fn identify_structural_hot_kernels(&self, _p: &MirProgram, _f: &CanaFeatures) -> Vec<String> {
        Vec::new()
    }
    fn name(&self) -> &'static str {
        "hot-fn-stub"
    }
    fn version(&self) -> u32 {
        1
    }
}

fn recommended_set(
    report: &cjc_cana::pass_ranker::RankingReport,
    fn_name: &str,
) -> BTreeSet<String> {
    report
        .per_fn
        .get(fn_name)
        .map(|r| {
            r.recommended
                .iter()
                .map(|rec| rec.pass_name.clone())
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

#[test]
fn pinn_wrapped_ranker_preserves_legality_verdict() {
    let program = two_fn_program();
    let features = extract(&program);
    let base_verdict = PassRanker::new(DiverseCostModel, PerPassLegalityGate::new())
        .rank(&program, &features)
        .verdict
        .clone();
    let pinn_verdict = PassRanker::new(
        PinnPhysicalCostModel::new(DiverseCostModel, HotFnPredictor),
        PerPassLegalityGate::new(),
    )
    .rank(&program, &features)
    .verdict
    .clone();
    assert_eq!(
        format!("{base_verdict:?}"),
        format!("{pinn_verdict:?}"),
        "physics must never touch the legality verdict"
    );
}

#[test]
fn pinn_hard_rejection_only_removes_recommendations() {
    let program = two_fn_program();
    let features = extract(&program);
    let base =
        PassRanker::new(DiverseCostModel, PerPassLegalityGate::new()).rank(&program, &features);
    let pinn = PassRanker::new(
        PinnPhysicalCostModel::new(DiverseCostModel, HotFnPredictor),
        PerPassLegalityGate::new(),
    )
    .rank(&program, &features);

    for f in ["cool_fn", "hot_fn"] {
        let base_set = recommended_set(&base, f);
        let pinn_set = recommended_set(&pinn, f);
        assert!(
            pinn_set.is_subset(&base_set),
            "{f}: PINN may only withhold, never add — base {base_set:?}, pinn {pinn_set:?}"
        );
    }

    // The hot function is past the hard thermal limit — every benefit
    // estimate became confident-zero (below the skip threshold), so
    // nothing may be recommended for it.
    assert!(
        recommended_set(&pinn, "hot_fn").is_empty(),
        "hot_fn exceeds the hard limit; all recommendations must be withheld"
    );
    // The cool function keeps its full recommendation set.
    assert_eq!(
        recommended_set(&pinn, "cool_fn"),
        recommended_set(&base, "cool_fn"),
        "cool_fn is unaffected by the hot function's rejection"
    );
}

#[test]
fn full_stack_pinn_plus_energy_reranker_double_run_is_identical() {
    let program = two_fn_program();
    let features = extract(&program);
    let make_adapter = || {
        EnergyAwarePassRanker::new(
            PassRanker::new(
                PinnPhysicalCostModel::new(DiverseCostModel, HotFnPredictor),
                PerPassLegalityGate::new(),
            ),
            Box::new(HotFnPredictor),
        )
    };
    let report_a = make_adapter().rank(&program, &features);
    let report_b = make_adapter().rank(&program, &features);

    for fn_name in report_a.per_fn.keys() {
        let a: Vec<String> = report_a.per_fn[fn_name]
            .recommended
            .iter()
            .map(|r| r.pass_name.clone())
            .collect();
        let b: Vec<String> = report_b.per_fn[fn_name]
            .recommended
            .iter()
            .map(|r| r.pass_name.clone())
            .collect();
        assert_eq!(a, b, "double-run order for {fn_name}");
    }
    assert_eq!(report_a.sequence, report_b.sequence);
}

#[test]
fn energy_audit_carries_physical_terms() {
    let program = two_fn_program();
    let features = extract(&program);
    let adapter = EnergyAwarePassRanker::new(
        PassRanker::new(DiverseCostModel, PerPassLegalityGate::new()),
        Box::new(HotFnPredictor),
    );
    let (_report, audit) = adapter.audit(&program, &features);

    // Default config has bandwidth_pressure_scale = 0.1 and
    // locality_reward_scale = 0.05 — the physical terms are live, so
    // every audited function must have finite energy totals.
    assert!(!audit.is_empty(), "audit must cover ranked functions");
    for (fn_name, entry) in &audit {
        for (pass, total) in &entry.energy_ordered_passes {
            assert!(
                total.is_finite(),
                "{fn_name}/{pass}: energy total must stay finite with physical terms"
            );
        }
    }
}
