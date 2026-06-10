//! Per-function feature aggregator and program-level extraction entry point.
//!
//! [`extract`] is the canonical way to build a [`CanaFeatures`] from a
//! [`MirProgram`]. It runs the existing `cjc-mir` analyses once and threads
//! the results into a per-function record.

use std::collections::BTreeMap;

use cjc_mir::cfg::CfgBuilder;
use cjc_mir::dominators::DominatorTree;
use cjc_mir::loop_analysis::{compute_loop_tree, LoopTree};
use cjc_mir::reduction::detect_reductions;
use cjc_mir::MirProgram;

use crate::cfg_metrics::CfgMetrics;
use crate::hash::{CanaHasher, FeatureHash, ProgramHash};
use crate::memory_proxy::MemoryProxy;
use crate::reduction_axes::ReductionAxes;
use crate::type_mix::TypeMix;

// ---------------------------------------------------------------------------
// FnFeatures — per-function record
// ---------------------------------------------------------------------------

/// All Phase-1 features for a single MIR function, bundled in a `Copy` struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FnFeatures {
    pub cfg: CfgMetrics,
    pub memory: MemoryProxy,
    pub reductions: ReductionAxes,
    /// Float/int operation mix (PINN v2): the static analog of the
    /// executor's runtime FP-binop counter. See [`crate::type_mix`].
    pub type_mix: TypeMix,
}

impl FnFeatures {
    pub(crate) fn feed(&self, hasher: &mut CanaHasher) {
        self.cfg.feed(hasher);
        self.memory.feed(hasher);
        self.reductions.feed(hasher);
        self.type_mix.feed(hasher);
    }
}

// ---------------------------------------------------------------------------
// CanaFeatures — program-level aggregate
// ---------------------------------------------------------------------------

/// Phase-1 features for an entire [`MirProgram`].
///
/// `per_fn` is a `BTreeMap<String, FnFeatures>` so iteration order is
/// guaranteed deterministic (sorted by function name).
#[derive(Debug, Clone)]
pub struct CanaFeatures {
    /// Per-function records, keyed by function name.
    pub per_fn: BTreeMap<String, FnFeatures>,
    /// Content-addressed fingerprint of the input program's *shape*
    /// (function names + signatures + body shape). Computed alongside
    /// extraction so callers don't have to walk the program twice.
    pub program_hash: ProgramHash,
    /// Content-addressed fingerprint of `per_fn`.
    pub feature_hash: FeatureHash,
}

impl CanaFeatures {
    /// Number of functions in the program.
    pub fn function_count(&self) -> usize {
        self.per_fn.len()
    }

    /// Total CFG block count across all functions.
    pub fn total_blocks(&self) -> u32 {
        self.per_fn
            .values()
            .map(|f| f.cfg.block_count)
            .fold(0u32, |a, b| a.saturating_add(b))
    }

    /// Total strict-reduction count across all functions — the single most
    /// important determinism legality signal.
    pub fn total_strict_reductions(&self) -> u32 {
        self.per_fn
            .values()
            .map(|f| f.reductions.strict_count())
            .fold(0u32, |a, b| a.saturating_add(b))
    }
}

// ---------------------------------------------------------------------------
// Entry point: extract()
// ---------------------------------------------------------------------------

/// Run the full Phase-1 featurization pipeline over a MIR program.
///
/// Internally:
/// 1. For each function: build CFG → dominator tree → loop tree.
/// 2. Run [`detect_reductions`] once over the whole program, passing the
///    loop trees collected in step 1.
/// 3. For each function: compute `CfgMetrics`, `MemoryProxy`, and
///    `ReductionAxes`, bundle into `FnFeatures`.
/// 4. Compute `ProgramHash` (over function names + param/return shape) and
///    `FeatureHash` (over the `per_fn` map in sorted key order).
///
/// Deterministic; same MIR → byte-identical hashes.
pub fn extract(program: &MirProgram) -> CanaFeatures {
    // Step 1: per-function CFG + dominator + loop tree, kept around so we
    // can pass them all to detect_reductions in one call.
    let mut cfgs: Vec<(String, cjc_mir::cfg::MirCfg)> = Vec::with_capacity(program.functions.len());
    let mut loop_trees: Vec<(String, LoopTree)> = Vec::with_capacity(program.functions.len());

    for func in &program.functions {
        let cfg = CfgBuilder::build(&func.body);
        let dom = DominatorTree::compute(&cfg);
        let loops = compute_loop_tree(&cfg, &dom);
        loop_trees.push((func.name.clone(), loops));
        cfgs.push((func.name.clone(), cfg));
    }

    // Step 2: program-wide reduction analysis.
    let reduction_report = detect_reductions(program, &loop_trees);

    // Step 3: aggregate per-function features.
    let mut per_fn: BTreeMap<String, FnFeatures> = BTreeMap::new();
    for (i, func) in program.functions.iter().enumerate() {
        let cfg = &cfgs[i].1;
        let loops = &loop_trees[i].1;
        let memory = MemoryProxy::from_function(func);
        let reductions = ReductionAxes::from_report_for_fn(&reduction_report, &func.name);
        let cfg_metrics = CfgMetrics::from_cfg(cfg, loops);
        let type_mix = TypeMix::from_function(func);
        per_fn.insert(
            func.name.clone(),
            FnFeatures {
                cfg: cfg_metrics,
                memory,
                reductions,
                type_mix,
            },
        );
    }

    // Step 4: hashes.
    let program_hash = compute_program_hash(program);
    let feature_hash = compute_feature_hash(&per_fn);

    CanaFeatures {
        per_fn,
        program_hash,
        feature_hash,
    }
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

const TAG_PROGRAM_HEADER: u8 = 0xF0;
const TAG_PROGRAM_FN: u8 = 0xF1;
const TAG_PROGRAM_PARAM: u8 = 0xF2;
const TAG_FEATURES_HEADER: u8 = 0xE0;
const TAG_FEATURES_FN: u8 = 0xE1;

fn compute_program_hash(program: &MirProgram) -> ProgramHash {
    let mut h = CanaHasher::new();
    h.write_tag(TAG_PROGRAM_HEADER);
    h.write_usize(program.functions.len());
    h.write_usize(program.struct_defs.len());
    h.write_usize(program.enum_defs.len());

    for func in &program.functions {
        h.write_tag(TAG_PROGRAM_FN);
        h.write_str(&func.name);
        h.write_u32(func.id.0);
        h.write_u8(func.is_nogc as u8);
        h.write_usize(func.params.len());
        for p in &func.params {
            h.write_tag(TAG_PROGRAM_PARAM);
            h.write_str(&p.name);
            h.write_str(&p.ty_name);
            h.write_u8(p.is_variadic as u8);
            h.write_u8(p.default.is_some() as u8);
        }
        if let Some(rt) = &func.return_type {
            h.write_u8(1);
            h.write_str(rt);
        } else {
            h.write_u8(0);
        }
        // Decorators in declaration order.
        h.write_usize(func.decorators.len());
        for d in &func.decorators {
            h.write_str(d);
        }
    }
    ProgramHash(h.finish())
}

fn compute_feature_hash(per_fn: &BTreeMap<String, FnFeatures>) -> FeatureHash {
    let mut h = CanaHasher::new();
    h.write_tag(TAG_FEATURES_HEADER);
    h.write_usize(per_fn.len());
    for (name, feats) in per_fn {
        h.write_tag(TAG_FEATURES_FN);
        h.write_str(name);
        feats.feed(&mut h);
    }
    FeatureHash(h.finish())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirParam, MirStmt};

    fn fn_with(name: &str, params: Vec<(&str, &str)>) -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: params
                .into_iter()
                .map(|(n, t)| MirParam {
                    name: n.to_string(),
                    ty_name: t.to_string(),
                    default: None,
                    is_variadic: false,
                })
                .collect(),
            return_type: None,
            body: MirBody {
                stmts: vec![],
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Public,
            local_count: 0,
        }
    }

    fn prog(fns: Vec<MirFunction>) -> MirProgram {
        let entry = fns.first().map(|f| f.id).unwrap_or(MirFnId(0));
        MirProgram {
            functions: fns,
            struct_defs: vec![],
            enum_defs: vec![],
            entry,
        }
    }

    #[test]
    fn extract_on_empty_program_succeeds() {
        let p = prog(vec![]);
        let f = extract(&p);
        assert_eq!(f.function_count(), 0);
        assert_eq!(f.total_blocks(), 0);
        // Even an empty program has a stable hash.
        let g = extract(&p);
        assert_eq!(f.program_hash, g.program_hash);
        assert_eq!(f.feature_hash, g.feature_hash);
    }

    #[test]
    fn extract_is_deterministic_across_repeated_calls() {
        let p = prog(vec![fn_with("a", vec![("x", "i64")]), fn_with("b", vec![])]);
        let mut prior: Option<(ProgramHash, FeatureHash)> = None;
        for _ in 0..50 {
            let f = extract(&p);
            let pair = (f.program_hash, f.feature_hash);
            if let Some(prev) = prior {
                assert_eq!(prev, pair);
            }
            prior = Some(pair);
        }
    }

    #[test]
    fn function_name_change_changes_program_hash() {
        let p1 = prog(vec![fn_with("foo", vec![])]);
        let p2 = prog(vec![fn_with("bar", vec![])]);
        let f1 = extract(&p1);
        let f2 = extract(&p2);
        assert_ne!(f1.program_hash, f2.program_hash);
    }

    #[test]
    fn param_change_changes_program_hash() {
        let p1 = prog(vec![fn_with("foo", vec![("x", "i64")])]);
        let p2 = prog(vec![fn_with("foo", vec![("x", "f64")])]);
        let f1 = extract(&p1);
        let f2 = extract(&p2);
        assert_ne!(f1.program_hash, f2.program_hash);
    }

    #[test]
    fn adding_a_function_changes_program_hash() {
        let p1 = prog(vec![fn_with("foo", vec![])]);
        let p2 = prog(vec![fn_with("foo", vec![]), fn_with("bar", vec![])]);
        let f1 = extract(&p1);
        let f2 = extract(&p2);
        assert_ne!(f1.program_hash, f2.program_hash);
        assert_eq!(f1.function_count(), 1);
        assert_eq!(f2.function_count(), 2);
    }

    #[test]
    fn per_fn_map_includes_every_function() {
        let p = prog(vec![
            fn_with("__main", vec![]),
            fn_with("solve", vec![("tol", "f64")]),
            fn_with("init", vec![]),
        ]);
        let f = extract(&p);
        assert_eq!(f.per_fn.len(), 3);
        assert!(f.per_fn.contains_key("__main"));
        assert!(f.per_fn.contains_key("solve"));
        assert!(f.per_fn.contains_key("init"));
    }

    #[test]
    fn feature_hash_changes_with_body_shape() {
        let mut p1 = fn_with("f", vec![]);
        p1.body.stmts.push(MirStmt::Expr(MirExpr {
            kind: MirExprKind::IntLit(0),
        }));
        let mut p2 = fn_with("f", vec![]);
        p2.body.stmts.push(MirStmt::Expr(MirExpr {
            kind: MirExprKind::IntLit(0),
        }));
        p2.body.stmts.push(MirStmt::Expr(MirExpr {
            kind: MirExprKind::IntLit(0),
        }));

        let r1 = extract(&prog(vec![p1]));
        let r2 = extract(&prog(vec![p2]));
        // expr_count differs; FeatureHash must differ.
        assert_ne!(r1.feature_hash, r2.feature_hash);
    }
}
