//! Reduction Analysis — Detect and classify accumulation patterns in loops
//!
//! CJC's #1 contract is determinism.  Floating-point reductions (sum, product,
//! dot-product, etc.) are the primary threat because reassociation changes
//! results.  This module makes reduction semantics **explicit** at the MIR
//! level so that:
//!
//! 1. The optimizer knows which variables are accumulators and cannot be
//!    freely reordered
//! 2. The verifier can check that reduction contracts are preserved across
//!    transformations
//! 3. Future parallel lowering can select the correct accumulator
//!    (Kahan vs BinnedAccumulator) based on the reduction kind
//!
//! ## Design decisions
//!
//! - **Vec + ID indexing** — `ReductionId(u32)` indexes into `Vec<ReductionInfo>`
//! - **Additive overlay** — does not modify MIR, CFG, or SSA
//! - **Conservative** — only classifies patterns we can prove; unknown patterns
//!   are tagged `Unknown` (never silently reordered)
//! - **Deterministic** — detection order follows block iteration order
//!
//! ## Detected patterns
//!
//! In tree-form MIR (inside `MirStmt::While` bodies):
//!
//! ```text
//! acc = acc + expr     → StrictFold (left fold, sequential)
//! acc = acc * expr     → StrictFold (product)
//! acc = acc - expr     → StrictFold (subtraction, non-commutative)
//! ```
//!
//! In function calls:
//!
//! ```text
//! sum(arr)             → BuiltinReduction("sum")
//! mean(arr)            → BuiltinReduction("mean")
//! dot(a, b)            → BuiltinReduction("dot")
//! ```
//!
//! ## Determinism guarantee
//!
//! Detection traverses MIR in statement order within each function.
//! Reductions are numbered in discovery order.  Same MIR → same results.

use cjc_ast::BinOp;

use crate::loop_analysis::{LoopId, LoopTree};
use crate::{MirBody, MirExpr, MirExprKind, MirFunction, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// Dense reduction identifier.  Index into `ReductionReport::reductions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReductionId(pub u32);

// ---------------------------------------------------------------------------
// Reduction classification
// ---------------------------------------------------------------------------

/// The semantic kind of a reduction.
///
/// This classification determines what transformations are legal:
///
/// | Kind | Reorder? | Parallel split? | Accumulator |
/// |------|----------|-----------------|-------------|
/// | StrictFold | NO | NO | Sequential only |
/// | KahanFold | NO | NO | KahanAccumulator |
/// | BinnedFold | YES (within bins) | YES | BinnedAccumulator |
/// | FixedTree | NO | YES (fixed tree) | Fixed reduction tree |
/// | BuiltinReduction | Depends on builtin | Depends | Runtime decides |
/// | Unknown | NO | NO | Conservative |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    /// Sequential left fold: `acc = acc ⊕ x[i]`.
    /// Order-dependent.  Cannot be parallelized without changing results.
    StrictFold,

    /// Kahan compensated summation.  Order-dependent but numerically stable.
    /// Used in `for`-loop reductions where order is fixed.
    KahanFold,

    /// Binned accumulation.  Order-independent (within bin capacity).
    /// Can be safely split across threads and merged.
    BinnedFold,

    /// Fixed-shape reduction tree.  Order is fixed by tree structure.
    /// Parallelizable along tree branches, but tree shape must be preserved.
    FixedTree,

    /// A call to a known builtin reduction function (e.g., `sum`, `mean`,
    /// `dot`).  The runtime selects the accumulator based on context.
    BuiltinReduction,

    /// Unclassified reduction pattern.  Conservative: no reordering allowed.
    Unknown,
}

impl ReductionKind {
    /// Can this reduction be safely reordered (elements processed in any order)?
    pub fn is_reorderable(&self) -> bool {
        matches!(self, ReductionKind::BinnedFold)
    }

    /// Can this reduction be split across parallel workers?
    pub fn is_parallelizable(&self) -> bool {
        matches!(
            self,
            ReductionKind::BinnedFold | ReductionKind::FixedTree
        )
    }

    /// Is this a strict sequential reduction that must not be reordered?
    pub fn is_strict(&self) -> bool {
        matches!(
            self,
            ReductionKind::StrictFold | ReductionKind::KahanFold | ReductionKind::Unknown
        )
    }
}

// ---------------------------------------------------------------------------
// Reduction operator
// ---------------------------------------------------------------------------

/// The mathematical operator used in the reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Addition (`acc + x`).
    Add,
    /// Multiplication (`acc * x`).
    Mul,
    /// Subtraction (`acc - x`).
    Sub,
    /// Minimum reduction.
    Min,
    /// Maximum reduction.
    Max,
    /// Bitwise OR reduction.
    BitwiseOr,
    /// Bitwise AND reduction.
    BitwiseAnd,
    /// A builtin function call (e.g., `sum`, `mean`).
    BuiltinCall,
}

// ---------------------------------------------------------------------------
// Reduction info
// ---------------------------------------------------------------------------

/// Specifies the required accumulator semantics for a reduction.
///
/// This metadata helps the verifier and future lowering passes select
/// the correct accumulator implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorSemantics {
    /// Plain sequential accumulation.  No special numerical handling.
    Plain,
    /// Kahan compensated summation required (carries a compensation term).
    Kahan,
    /// Binned accumulation required (reproducible across parallelism).
    Binned,
    /// Implementation-defined (for builtins — runtime chooses).
    RuntimeDefined,
}

impl std::fmt::Display for AccumulatorSemantics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccumulatorSemantics::Plain => write!(f, "plain"),
            AccumulatorSemantics::Kahan => write!(f, "kahan"),
            AccumulatorSemantics::Binned => write!(f, "binned"),
            AccumulatorSemantics::RuntimeDefined => write!(f, "runtime_defined"),
        }
    }
}

/// Information about a single detected reduction.
#[derive(Debug, Clone)]
pub struct ReductionInfo {
    /// Unique ID (= index in `ReductionReport::reductions`).
    pub id: ReductionId,
    /// The variable being accumulated into.
    pub accumulator_var: String,
    /// The reduction operator.
    pub op: ReductionOp,
    /// The semantic kind (determines legality of transformations).
    pub kind: ReductionKind,
    /// The loop containing this reduction, if detected inside a loop.
    pub loop_id: Option<LoopId>,
    /// The function containing this reduction.
    pub function_name: String,
    /// For builtin reductions, the function name (e.g., "sum", "mean").
    pub builtin_name: Option<String>,

    // ── Evolution v0.3: enriched reduction metadata ─────────────

    /// Whether reassociation of this reduction's operands is forbidden.
    /// `true` for all strict/Kahan folds and unknown reductions.
    /// `false` only for BinnedFold (within bin capacity).
    pub reassociation_forbidden: bool,
    /// Whether strict sequential execution order is required.
    /// `true` means no reordering of iteration steps is legal.
    pub strict_order_required: bool,
    /// The required accumulator semantics for this reduction.
    pub accumulator_semantics: AccumulatorSemantics,
}

// ---------------------------------------------------------------------------
// Reduction report
// ---------------------------------------------------------------------------

/// All reductions detected in a MIR program.
#[derive(Debug, Clone)]
pub struct ReductionReport {
    /// All detected reductions, indexed by `ReductionId`.
    pub reductions: Vec<ReductionInfo>,
}

impl ReductionReport {
    /// Return the number of detected reductions.
    pub fn len(&self) -> usize {
        self.reductions.len()
    }

    /// Return true if no reductions were detected.
    pub fn is_empty(&self) -> bool {
        self.reductions.is_empty()
    }

    /// Look up a reduction by its ID.
    pub fn get(&self, id: ReductionId) -> &ReductionInfo {
        &self.reductions[id.0 as usize]
    }

    /// Get all reductions in a specific loop.
    pub fn reductions_in_loop(&self, loop_id: LoopId) -> Vec<&ReductionInfo> {
        self.reductions
            .iter()
            .filter(|r| r.loop_id == Some(loop_id))
            .collect()
    }

    /// Get all reductions in a specific function.
    pub fn reductions_in_function(&self, fn_name: &str) -> Vec<&ReductionInfo> {
        self.reductions
            .iter()
            .filter(|r| r.function_name == fn_name)
            .collect()
    }

    /// Returns true if any reduction is strict (cannot be reordered).
    pub fn has_strict_reductions(&self) -> bool {
        self.reductions.iter().any(|r| r.kind.is_strict())
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Known builtin reduction functions.
const BUILTIN_REDUCTIONS: &[&str] = &[
    "sum",
    "mean",
    "dot",
    "prod",
    "variance",
    "sd",
    "norm",
    "min",
    "max",
    "median",
    "binned_sum",
    "kahan_sum",
    "trapz",
    "simps",
];

/// Detect all reductions in a MIR program.
///
/// This is a two-pass analysis:
/// 1. **Loop accumulation patterns** — detect `acc = acc ⊕ expr` inside while
///    loops using the loop tree
/// 2. **Builtin reduction calls** — detect calls to known reduction functions
///
/// ## Determinism
///
/// Functions are visited in program order.  Within each function, statements
/// are visited in tree-form order.  Reductions are numbered in discovery order.
pub fn detect_reductions(program: &MirProgram, loop_trees: &[(String, LoopTree)]) -> ReductionReport {
    let mut reductions = Vec::new();
    let mut next_id: u32 = 0;

    // Build a lookup from function name to loop tree.
    let loop_tree_map: Vec<(&str, &LoopTree)> = loop_trees
        .iter()
        .map(|(name, tree)| (name.as_str(), tree))
        .collect();

    for func in &program.functions {
        let tree = loop_tree_map
            .iter()
            .find(|(name, _)| *name == func.name)
            .map(|(_, t)| *t);

        // Pass 1: Detect accumulation patterns in while loops.
        detect_loop_reductions_body(
            &func.body,
            &func.name,
            tree,
            &mut reductions,
            &mut next_id,
        );

        // Pass 2: Detect builtin reduction calls anywhere in the function.
        detect_builtin_reductions_body(
            &func.body,
            &func.name,
            tree,
            &mut reductions,
            &mut next_id,
        );
    }

    ReductionReport { reductions }
}

/// Detect `acc = acc ⊕ expr` patterns inside while loop bodies.
fn detect_loop_reductions_body(
    body: &MirBody,
    fn_name: &str,
    loop_tree: Option<&LoopTree>,
    reductions: &mut Vec<ReductionInfo>,
    next_id: &mut u32,
) {
    for stmt in &body.stmts {
        match stmt {
            MirStmt::While { body: loop_body, .. } => {
                // Scan the loop body for accumulation patterns.
                scan_body_for_accumulations(
                    loop_body,
                    fn_name,
                    loop_tree,
                    reductions,
                    next_id,
                );
                // Recurse into nested structures.
                detect_loop_reductions_body(
                    loop_body,
                    fn_name,
                    loop_tree,
                    reductions,
                    next_id,
                );
            }
            MirStmt::If {
                then_body,
                else_body,
                ..
            } => {
                detect_loop_reductions_body(
                    then_body,
                    fn_name,
                    loop_tree,
                    reductions,
                    next_id,
                );
                if let Some(eb) = else_body {
                    detect_loop_reductions_body(
                        eb,
                        fn_name,
                        loop_tree,
                        reductions,
                        next_id,
                    );
                }
            }
            MirStmt::NoGcBlock(inner) => {
                detect_loop_reductions_body(
                    inner,
                    fn_name,
                    loop_tree,
                    reductions,
                    next_id,
                );
            }
            _ => {}
        }
    }
}

/// Scan a single loop body for accumulation patterns:
/// `acc = acc + expr`, `acc = acc * expr`, etc.
fn scan_body_for_accumulations(
    body: &MirBody,
    fn_name: &str,
    _loop_tree: Option<&LoopTree>,
    reductions: &mut Vec<ReductionInfo>,
    next_id: &mut u32,
) {
    for stmt in &body.stmts {
        if let MirStmt::Expr(expr) = stmt {
            if let Some((acc_name, op)) = match_accumulation_pattern(expr) {
                // Check we haven't already recorded this accumulator in this scan.
                let already = reductions.iter().any(|r| {
                    r.accumulator_var == acc_name
                        && r.function_name == fn_name
                        && r.kind == ReductionKind::StrictFold
                });
                if !already {
                    reductions.push(ReductionInfo {
                        id: ReductionId(*next_id),
                        accumulator_var: acc_name,
                        op,
                        kind: ReductionKind::StrictFold,
                        loop_id: None, // Could be refined with loop tree
                        function_name: fn_name.to_string(),
                        builtin_name: None,
                        reassociation_forbidden: true,
                        strict_order_required: true,
                        accumulator_semantics: AccumulatorSemantics::Plain,
                    });
                    *next_id += 1;
                }
            }
        }
    }
}

/// Match the pattern `acc = acc ⊕ expr` in an expression.
///
/// Returns `Some((accumulator_name, op))` if matched.
fn match_accumulation_pattern(expr: &MirExpr) -> Option<(String, ReductionOp)> {
    // Pattern: Assign { target: Var(acc), value: Binary { op, left: Var(acc), right: _ } }
    if let MirExprKind::Assign { target, value } = &expr.kind {
        if let MirExprKind::Var(acc_name) = &target.kind {
            if let MirExprKind::Binary { op, left, .. } = &value.kind {
                if let MirExprKind::Var(left_name) = &left.kind {
                    if left_name == acc_name {
                        let reduction_op = match op {
                            BinOp::Add => Some(ReductionOp::Add),
                            BinOp::Mul => Some(ReductionOp::Mul),
                            BinOp::Sub => Some(ReductionOp::Sub),
                            BinOp::BitOr => Some(ReductionOp::BitwiseOr),
                            BinOp::BitAnd => Some(ReductionOp::BitwiseAnd),
                            _ => None,
                        };
                        return reduction_op.map(|rop| (acc_name.clone(), rop));
                    }
                }
            }
        }
    }
    None
}

/// Detect calls to known builtin reduction functions.
fn detect_builtin_reductions_body(
    body: &MirBody,
    fn_name: &str,
    loop_tree: Option<&LoopTree>,
    reductions: &mut Vec<ReductionInfo>,
    next_id: &mut u32,
) {
    for stmt in &body.stmts {
        match stmt {
            MirStmt::Let { init, .. } => {
                detect_builtin_reductions_expr(init, fn_name, loop_tree, reductions, next_id);
            }
            MirStmt::Expr(expr) => {
                detect_builtin_reductions_expr(expr, fn_name, loop_tree, reductions, next_id);
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                detect_builtin_reductions_expr(cond, fn_name, loop_tree, reductions, next_id);
                detect_builtin_reductions_body(then_body, fn_name, loop_tree, reductions, next_id);
                if let Some(eb) = else_body {
                    detect_builtin_reductions_body(eb, fn_name, loop_tree, reductions, next_id);
                }
            }
            MirStmt::While { cond, body: wb } => {
                detect_builtin_reductions_expr(cond, fn_name, loop_tree, reductions, next_id);
                detect_builtin_reductions_body(wb, fn_name, loop_tree, reductions, next_id);
            }
            MirStmt::Return(Some(expr)) => {
                detect_builtin_reductions_expr(expr, fn_name, loop_tree, reductions, next_id);
            }
            MirStmt::NoGcBlock(inner) => {
                detect_builtin_reductions_body(inner, fn_name, loop_tree, reductions, next_id);
            }
            _ => {}
        }
    }
    if let Some(ref result) = body.result {
        detect_builtin_reductions_expr(result, fn_name, loop_tree, reductions, next_id);
    }
}

/// Detect builtin reduction calls in an expression tree.
fn detect_builtin_reductions_expr(
    expr: &MirExpr,
    fn_name: &str,
    loop_tree: Option<&LoopTree>,
    reductions: &mut Vec<ReductionInfo>,
    next_id: &mut u32,
) {
    match &expr.kind {
        MirExprKind::Call { callee, args } => {
            if let MirExprKind::Var(callee_name) = &callee.kind {
                if BUILTIN_REDUCTIONS.contains(&callee_name.as_str()) {
                    let (reassoc, strict, semantics) = classify_builtin_reduction(callee_name);
                    reductions.push(ReductionInfo {
                        id: ReductionId(*next_id),
                        accumulator_var: String::new(), // N/A for builtins
                        op: ReductionOp::BuiltinCall,
                        kind: ReductionKind::BuiltinReduction,
                        loop_id: None,
                        function_name: fn_name.to_string(),
                        builtin_name: Some(callee_name.clone()),
                        reassociation_forbidden: reassoc,
                        strict_order_required: strict,
                        accumulator_semantics: semantics,
                    });
                    *next_id += 1;
                }
            }
            // Recurse into arguments.
            for arg in args {
                detect_builtin_reductions_expr(arg, fn_name, loop_tree, reductions, next_id);
            }
        }
        MirExprKind::Binary { left, right, .. } => {
            detect_builtin_reductions_expr(left, fn_name, loop_tree, reductions, next_id);
            detect_builtin_reductions_expr(right, fn_name, loop_tree, reductions, next_id);
        }
        MirExprKind::Unary { operand, .. } => {
            detect_builtin_reductions_expr(operand, fn_name, loop_tree, reductions, next_id);
        }
        MirExprKind::Assign { target, value } => {
            detect_builtin_reductions_expr(target, fn_name, loop_tree, reductions, next_id);
            detect_builtin_reductions_expr(value, fn_name, loop_tree, reductions, next_id);
        }
        MirExprKind::Index { object, index } => {
            detect_builtin_reductions_expr(object, fn_name, loop_tree, reductions, next_id);
            detect_builtin_reductions_expr(index, fn_name, loop_tree, reductions, next_id);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Builtin reduction classification
// ---------------------------------------------------------------------------

/// Classify a builtin reduction function's reassociation and accumulator semantics.
///
/// Returns `(reassociation_forbidden, strict_order_required, accumulator_semantics)`.
fn classify_builtin_reduction(name: &str) -> (bool, bool, AccumulatorSemantics) {
    match name {
        // Kahan summation — order-dependent, Kahan accumulator
        "kahan_sum" => (true, true, AccumulatorSemantics::Kahan),
        // Binned summation — order-independent within bins
        "binned_sum" => (false, false, AccumulatorSemantics::Binned),
        // Integration rules — order-dependent (positional weights)
        "trapz" | "simps" => (true, true, AccumulatorSemantics::Plain),
        // All other builtins — runtime-defined accumulator, conservative
        _ => (true, true, AccumulatorSemantics::RuntimeDefined),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirBody, MirExpr, MirExprKind, MirFunction, MirFnId, MirParam, MirProgram, MirStmt};
    use cjc_ast::{BinOp, Visibility};

    fn int_expr(v: i64) -> MirExpr {
        MirExpr {
            kind: MirExprKind::IntLit(v),
        }
    }

    fn var_expr(name: &str) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Var(name.to_string()),
        }
    }

    fn bool_expr(b: bool) -> MirExpr {
        MirExpr {
            kind: MirExprKind::BoolLit(b),
        }
    }

    fn assign_acc_add(acc: &str, rhs: MirExpr) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Assign {
                target: Box::new(var_expr(acc)),
                value: Box::new(MirExpr {
                    kind: MirExprKind::Binary {
                        op: BinOp::Add,
                        left: Box::new(var_expr(acc)),
                        right: Box::new(rhs),
                    },
                }),
            },
        }
    }

    fn call_expr(name: &str, args: Vec<MirExpr>) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Call {
                callee: Box::new(var_expr(name)),
                args,
            },
        }
    }

    fn make_program(functions: Vec<MirFunction>) -> MirProgram {
        let entry = functions.last().map(|f| f.id).unwrap_or(MirFnId(0));
        MirProgram {
            functions,
            struct_defs: vec![],
            enum_defs: vec![],
            entry,
        }
    }

    fn make_fn(name: &str, body: MirBody) -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body,
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: Visibility::Private,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Detect accumulation pattern acc = acc + x
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_strict_fold_add() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "acc".into(),
                    mutable: true,
                    init: int_expr(0),
                    alloc_hint: None,
                },
                MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(assign_acc_add("acc", var_expr("x")))],
                        result: None,
                    },
                },
            ],
            result: None,
        };

        let program = make_program(vec![make_fn("test", body)]);
        let report = detect_reductions(&program, &[]);

        assert!(
            report.reductions.iter().any(|r| r.accumulator_var == "acc"
                && r.op == ReductionOp::Add
                && r.kind == ReductionKind::StrictFold),
            "should detect acc = acc + x as StrictFold"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Detect accumulation pattern acc = acc * x
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_strict_fold_mul() {
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: bool_expr(true),
                body: MirBody {
                    stmts: vec![MirStmt::Expr(MirExpr {
                        kind: MirExprKind::Assign {
                            target: Box::new(var_expr("prod")),
                            value: Box::new(MirExpr {
                                kind: MirExprKind::Binary {
                                    op: BinOp::Mul,
                                    left: Box::new(var_expr("prod")),
                                    right: Box::new(var_expr("x")),
                                },
                            }),
                        },
                    })],
                    result: None,
                },
            }],
            result: None,
        };

        let program = make_program(vec![make_fn("test", body)]);
        let report = detect_reductions(&program, &[]);

        assert!(
            report.reductions.iter().any(|r| r.accumulator_var == "prod"
                && r.op == ReductionOp::Mul
                && r.kind == ReductionKind::StrictFold),
            "should detect prod = prod * x as StrictFold"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Detect builtin reduction call sum(arr)
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_builtin_sum() {
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "total".into(),
                mutable: false,
                init: call_expr("sum", vec![var_expr("arr")]),
                alloc_hint: None,
            }],
            result: None,
        };

        let program = make_program(vec![make_fn("test", body)]);
        let report = detect_reductions(&program, &[]);

        assert!(
            report.reductions.iter().any(|r| r.kind == ReductionKind::BuiltinReduction
                && r.builtin_name.as_deref() == Some("sum")),
            "should detect sum() as BuiltinReduction"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Detect multiple builtin reductions
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_multiple_builtins() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "s".into(),
                    mutable: false,
                    init: call_expr("sum", vec![var_expr("a")]),
                    alloc_hint: None,
                },
                MirStmt::Let {
                    name: "m".into(),
                    mutable: false,
                    init: call_expr("mean", vec![var_expr("a")]),
                    alloc_hint: None,
                },
                MirStmt::Let {
                    name: "d".into(),
                    mutable: false,
                    init: call_expr("dot", vec![var_expr("a"), var_expr("b")]),
                    alloc_hint: None,
                },
            ],
            result: None,
        };

        let program = make_program(vec![make_fn("test", body)]);
        let report = detect_reductions(&program, &[]);

        assert_eq!(report.len(), 3, "should detect 3 builtin reductions");
        let names: Vec<&str> = report
            .reductions
            .iter()
            .filter_map(|r| r.builtin_name.as_deref())
            .collect();
        assert!(names.contains(&"sum"));
        assert!(names.contains(&"mean"));
        assert!(names.contains(&"dot"));
    }

    // -----------------------------------------------------------------------
    // Test 5: No false positives for non-reduction calls
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_false_positives() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "r".into(),
                    mutable: false,
                    init: call_expr("print", vec![var_expr("x")]),
                    alloc_hint: None,
                },
                MirStmt::Expr(MirExpr {
                    kind: MirExprKind::Assign {
                        target: Box::new(var_expr("y")),
                        // Not acc = acc + x — this is y = x + 1
                        value: Box::new(MirExpr {
                            kind: MirExprKind::Binary {
                                op: BinOp::Add,
                                left: Box::new(var_expr("x")),
                                right: Box::new(int_expr(1)),
                            },
                        }),
                    },
                }),
            ],
            result: None,
        };

        let program = make_program(vec![make_fn("test", body)]);
        let report = detect_reductions(&program, &[]);

        assert!(
            report.is_empty(),
            "should not detect reductions in non-reduction code"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: ReductionKind properties
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduction_kind_properties() {
        assert!(ReductionKind::StrictFold.is_strict());
        assert!(!ReductionKind::StrictFold.is_reorderable());
        assert!(!ReductionKind::StrictFold.is_parallelizable());

        assert!(ReductionKind::BinnedFold.is_reorderable());
        assert!(ReductionKind::BinnedFold.is_parallelizable());
        assert!(!ReductionKind::BinnedFold.is_strict());

        assert!(!ReductionKind::FixedTree.is_reorderable());
        assert!(ReductionKind::FixedTree.is_parallelizable());

        assert!(ReductionKind::Unknown.is_strict());
        assert!(!ReductionKind::Unknown.is_parallelizable());
    }

    // -----------------------------------------------------------------------
    // Test 7: reductions_in_function filter
    // -----------------------------------------------------------------------
    #[test]
    fn test_reductions_in_function() {
        let fn1 = make_fn(
            "compute",
            MirBody {
                stmts: vec![MirStmt::Let {
                    name: "s".into(),
                    mutable: false,
                    init: call_expr("sum", vec![var_expr("a")]),
                    alloc_hint: None,
                }],
                result: None,
            },
        );
        let fn2 = make_fn(
            "other",
            MirBody {
                stmts: vec![MirStmt::Let {
                    name: "m".into(),
                    mutable: false,
                    init: call_expr("mean", vec![var_expr("b")]),
                    alloc_hint: None,
                }],
                result: None,
            },
        );

        let program = make_program(vec![fn1, fn2]);
        let report = detect_reductions(&program, &[]);

        let compute_reds = report.reductions_in_function("compute");
        assert_eq!(compute_reds.len(), 1);
        assert_eq!(compute_reds[0].builtin_name.as_deref(), Some("sum"));

        let other_reds = report.reductions_in_function("other");
        assert_eq!(other_reds.len(), 1);
        assert_eq!(other_reds[0].builtin_name.as_deref(), Some("mean"));
    }

    // -----------------------------------------------------------------------
    // Test 8: Determinism — same input produces identical report
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduction_detection_determinism() {
        let body = MirBody {
            stmts: vec![
                MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(assign_acc_add("acc", var_expr("x")))],
                        result: None,
                    },
                },
                MirStmt::Let {
                    name: "s".into(),
                    mutable: false,
                    init: call_expr("sum", vec![var_expr("arr")]),
                    alloc_hint: None,
                },
                MirStmt::Let {
                    name: "d".into(),
                    mutable: false,
                    init: call_expr("dot", vec![var_expr("a"), var_expr("b")]),
                    alloc_hint: None,
                },
            ],
            result: None,
        };

        let program = make_program(vec![make_fn("test", body)]);

        let report1 = detect_reductions(&program, &[]);
        let report2 = detect_reductions(&program, &[]);

        assert_eq!(report1.len(), report2.len());
        for (a, b) in report1.reductions.iter().zip(report2.reductions.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.accumulator_var, b.accumulator_var);
            assert_eq!(a.op, b.op);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.function_name, b.function_name);
            assert_eq!(a.builtin_name, b.builtin_name);
        }
    }
}
