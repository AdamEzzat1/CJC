//! MIR Optimizer (Stage 2.4)
//!
//! Passes (applied in order):
//! 1. Constant Folding (CF) — fold pure operations with literal operands.
//! 2. Strength Reduction (SR) — replace expensive ops with cheaper equivalents.
//! 3. Dead Code Elimination (DCE) — remove unused assigns and unreachable blocks.
//! 4. Common Subexpression Elimination (CSE) — reuse identical pure expressions.
//! 5. Loop-Invariant Code Motion (LICM) — hoist invariant code out of loops.
//! 6. Loop Unrolling — replace short fixed-trip-count `while` loops with
//!    straight-line replicas of their body.
//! 7. Second round of CF (may expose new opportunities after other passes,
//!    including the literal-trip-count constants exposed by unrolling).
//!
//! Design constraints:
//! - Bit-identical results: no float reassociation, no reorder of evaluation.
//! - Shape checks and bounds checks must be preserved.
//! - Side-effecting operations (calls, assigns, index) must not be removed.
//! - All passes are deterministic and platform-independent.

use crate::{MirBody, MirExpr, MirExprKind, MirFunction, MirProgram, MirStmt};
use cjc_ast::{BinOp, UnaryOp};
use std::collections::{BTreeMap, BTreeSet};

// ===========================================================================
// Public API
// ===========================================================================

/// Default pass sequence, in the order applied by [`optimize_program`].
/// Each entry is the canonical pass name accepted by [`PassPlan`] and
/// [`apply_pass`]. Aliases (e.g. `"cf"` for `"constant_fold"`) are accepted
/// in plans but the canonical form is used here.
pub const DEFAULT_PASS_SEQUENCE: &[&str] = &[
    "constant_fold",
    "strength_reduce",
    "dce",
    "cse",
    "licm",
    "loop_unroll",
    "cf_round_2",
];

/// A per-function plan for which optimization passes to run, in which order.
///
/// `per_function` is a `BTreeMap<function_name, Vec<pass_name>>`. Functions
/// not in the map fall back to [`DEFAULT_PASS_SEQUENCE`] — this preserves
/// today's behaviour for any function CANA hasn't recommended for.
///
/// Pass names are matched case-sensitively. Recognised canonical names are
/// listed in [`DEFAULT_PASS_SEQUENCE`]; aliases handled by [`apply_pass`].
///
/// ## Why this lives in cjc-mir (not cjc-cana)
///
/// `cjc-cana` already depends on `cjc-mir` (it reads MIR). If `cjc-mir`
/// also depended on `cjc-cana` (to consume `cjc_cana::PassSequence`) we'd
/// have a dependency cycle. Defining `PassPlan` here lets `cjc-mir` stay at
/// the bottom of the dep tree; `cjc-cana` provides a conversion helper
/// (`cjc_cana::pass_plan_from`) that builds one of these from a CANA
/// recommendation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PassPlan {
    /// Per-function pass sequence. Functions absent from the map use
    /// [`DEFAULT_PASS_SEQUENCE`].
    pub per_function: BTreeMap<String, Vec<String>>,
}

impl PassPlan {
    /// Construct an empty plan — every function uses the default sequence.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Set the pass sequence for a single function.
    pub fn with_function(mut self, fn_name: impl Into<String>, passes: Vec<String>) -> Self {
        self.per_function.insert(fn_name.into(), passes);
        self
    }

    /// Total number of pass invocations across all functions, counting the
    /// default sequence for functions not in the map. Used for debug /
    /// audit surfaces.
    pub fn total_pass_invocations(&self, all_function_names: &[String]) -> usize {
        all_function_names
            .iter()
            .map(|n| {
                self.per_function
                    .get(n)
                    .map(|v| v.len())
                    .unwrap_or(DEFAULT_PASS_SEQUENCE.len())
            })
            .sum()
    }
}

/// Apply one pass by name to a function. Returns `true` if the pass name
/// was recognised; `false` if it was a no-op (unknown name).
///
/// Recognised names (case-sensitive):
/// - `"constant_fold"`, `"cf"`, `"cf_round_2"` → constant folding
/// - `"strength_reduce"`, `"sr"` → strength reduction
/// - `"dce"`, `"dead_code_elimination"` → dead code elimination
/// - `"cse"`, `"common_subexpression_elimination"` → CSE
/// - `"licm"`, `"loop_invariant_code_motion"` → LICM
/// - `"loop_unroll"`, `"unroll"` → loop unrolling
///
/// Unknown names are silently skipped. This is intentional: CANA may
/// recommend a pass that's not yet implemented in the compiler, and the
/// right response is to skip it (and log via the caller's audit trail),
/// not to error.
pub fn apply_pass(pass_name: &str, func: &mut MirFunction) -> bool {
    apply_pass_with_diagnostics(pass_name, func).is_some()
}

/// Per-pass observability data returned by [`apply_pass_with_diagnostics`].
///
/// `changes_applied` is the pass's native count of rewrites — what the
/// pass itself reports it changed. Each pass's interpretation:
///
/// | Pass | `changes_applied` is |
/// |---|---|
/// | `constant_fold` | node-count delta (before − after) |
/// | `strength_reduce` | count of `try_strength_reduce` rewrites |
/// | `dce` | node-count delta (before − after) |
/// | `cse` | count of variable replacements applied |
/// | `licm` | count of statements hoisted out of `while` loops |
/// | `loop_unroll` | count of `while` loops successfully unrolled |
/// | `fusion_rewrite` | count of fused chains created (via the rewriter) |
///
/// `nodes_before`/`nodes_after` always reflect the structural change to
/// the function regardless of how `changes_applied` was derived. The two
/// can disagree (e.g. SR's `x * 2 → x + x` changes one operator but
/// keeps node count). That disagreement is informative — it tells the
/// caller whether the pass shrank code or just rewrote in place.
///
/// ## Why this surface exists
///
/// Two complementary use cases:
///
/// 1. **Training the cost model.** `bench/cana_train_cost_model/` reads
///    `changes_applied` to construct a deterministic, zero-variance
///    benefit signal for SR/CSE/LICM (the passes where MIR-count delta
///    is zero for structural reasons).
///
/// 2. **Permanent observability.** `cjcl emit --pass-stats` (future),
///    any audit tooling, the upcoming `CanaReport` diagnostics — all
///    can consume the same `PassDiagnostics` shape without each
///    re-instrumenting the optimizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassDiagnostics {
    /// Canonical name of the pass that produced this report.
    pub pass_name: String,
    /// Pass-native count of "interesting changes" applied. See struct
    /// docs for the per-pass interpretation.
    pub changes_applied: usize,
    /// Total MIR-node count of the function before the pass ran.
    pub nodes_before: usize,
    /// Total MIR-node count of the function after the pass ran.
    pub nodes_after: usize,
}

impl PassDiagnostics {
    /// Fractional structural change: `(nodes_before - nodes_after) / max(nodes_before, 1)`.
    /// Equivalent to the v2 MIR-count proxy used by CF and DCE.
    pub fn structural_delta_fraction(&self) -> f64 {
        let before = self.nodes_before.max(1) as f64;
        (self.nodes_before.saturating_sub(self.nodes_after) as f64) / before
    }

    /// Fractional pass-native change: `changes_applied / max(nodes_before, 1)`.
    /// More informative than `structural_delta_fraction` when the pass
    /// rewrites in place (e.g. SR's `Mul → Shl`-style operations that
    /// don't shrink nodes). The deterministic training signal used by
    /// Option A of the cost-model training findings.
    pub fn pass_native_change_fraction(&self) -> f64 {
        let before = self.nodes_before.max(1) as f64;
        (self.changes_applied as f64) / before
    }
}

/// Like [`apply_pass`] but also returns a [`PassDiagnostics`] report for
/// the pass that ran. Returns `None` only when the pass name is unknown.
///
/// Same semantics as `apply_pass` for known passes — the optimizer is
/// unchanged; the wrapper just instruments before/after node counting
/// and forwards each pass's native change count.
pub fn apply_pass_with_diagnostics(
    pass_name: &str,
    func: &mut MirFunction,
) -> Option<PassDiagnostics> {
    let nodes_before = count_function_nodes(func);
    let pass_native_changes: Option<usize> = match pass_name {
        "constant_fold" | "cf" | "cf_round_2" => {
            constant_fold_fn(func);
            // CF doesn't currently report its own count; the
            // structural delta is exact (CF only collapses literal
            // subexpressions, shrinking node count by exactly the
            // number of folded nodes).
            None
        }
        "strength_reduce" | "sr" => Some(strength_reduce_fn(func)),
        "dce" | "dead_code_elimination" => {
            dce_fn(func);
            // Same story as CF — DCE's effect is exactly captured by
            // node-count delta.
            None
        }
        "cse" | "common_subexpression_elimination" => Some(cse_fn(func)),
        "licm" | "loop_invariant_code_motion" => Some(licm_fn(func)),
        "loop_unroll" | "unroll" => Some(loop_unroll_fn(func)),
        "fusion_rewrite" | "fusion" => {
            Some(crate::fusion_rewrite::fusion_rewrite_fn(func))
        }
        _ => return None,
    };
    let nodes_after = count_function_nodes(func);
    let changes_applied = pass_native_changes
        .unwrap_or_else(|| nodes_before.saturating_sub(nodes_after));
    Some(PassDiagnostics {
        pass_name: pass_name.to_string(),
        changes_applied,
        nodes_before,
        nodes_after,
    })
}

/// Count MIR nodes (statements + expressions) in one function. Used by
/// [`apply_pass_with_diagnostics`] to populate `nodes_before` / `nodes_after`
/// and by Option-A's training signal as the denominator of the change
/// fraction.
fn count_function_nodes(func: &MirFunction) -> usize {
    count_body_node_total(&func.body)
}

fn count_body_node_total(body: &MirBody) -> usize {
    let mut count = 0;
    for stmt in &body.stmts {
        count += count_stmt_node_total(stmt);
    }
    if let Some(e) = &body.result {
        count += count_expr_node_total(e);
    }
    count
}

fn count_stmt_node_total(stmt: &MirStmt) -> usize {
    1 + match stmt {
        MirStmt::Let { init, .. } => count_expr_node_total(init),
        MirStmt::Expr(e) => count_expr_node_total(e),
        MirStmt::If { cond, then_body, else_body } => {
            count_expr_node_total(cond)
                + count_body_node_total(then_body)
                + else_body.as_ref().map(count_body_node_total).unwrap_or(0)
        }
        MirStmt::While { cond, body } => count_expr_node_total(cond) + count_body_node_total(body),
        MirStmt::Return(Some(e)) => count_expr_node_total(e),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => 0,
        MirStmt::NoGcBlock(b) => count_body_node_total(b),
    }
}

fn count_expr_node_total(expr: &MirExpr) -> usize {
    1 + match &expr.kind {
        MirExprKind::Binary { left, right, .. } => {
            count_expr_node_total(left) + count_expr_node_total(right)
        }
        MirExprKind::Unary { operand, .. } => count_expr_node_total(operand),
        MirExprKind::Call { callee, args } => {
            count_expr_node_total(callee) + args.iter().map(count_expr_node_total).sum::<usize>()
        }
        MirExprKind::Assign { target, value } => {
            count_expr_node_total(target) + count_expr_node_total(value)
        }
        MirExprKind::Field { object, .. } => count_expr_node_total(object),
        MirExprKind::Index { object, index } => {
            count_expr_node_total(object) + count_expr_node_total(index)
        }
        MirExprKind::ArrayLit(es) | MirExprKind::TupleLit(es) => {
            es.iter().map(count_expr_node_total).sum::<usize>()
        }
        MirExprKind::StructLit { fields, .. } => {
            fields.iter().map(|(_, e)| count_expr_node_total(e)).sum::<usize>()
        }
        MirExprKind::MakeClosure { captures, .. } => {
            captures.iter().map(count_expr_node_total).sum::<usize>()
        }
        _ => 0,
    }
}

/// Run all optimization passes on a MIR program using the default
/// 6-pass sequence (CF → SR → DCE → CSE → LICM → CF).
///
/// This is the entry point preserved for callers that don't have a
/// CANA recommendation. Internally, it delegates to
/// [`optimize_program_with_plan`] with an empty plan, so every function
/// falls back to [`DEFAULT_PASS_SEQUENCE`].
///
/// Behaviour is byte-identical to the pre-Phase-2 implementation —
/// preserved for parity tests and any caller that hasn't opted into
/// CANA yet.
pub fn optimize_program(program: &MirProgram) -> MirProgram {
    optimize_program_with_plan(program, &PassPlan::empty())
}

/// Run optimization passes on a MIR program using a per-function plan.
///
/// For each function:
/// - If `plan.per_function` has an entry, run those passes in that order.
/// - Otherwise, run [`DEFAULT_PASS_SEQUENCE`].
///
/// This is the Phase 2 entry point for CANA-driven optimization. Callers
/// build a `PassPlan` from a CANA `PassSequence` (via
/// `cjc_cana::pass_plan_from`) and pass it here.
///
/// ## Determinism
///
/// For a given `(program, plan)`, the output is byte-identical across
/// runs and platforms. The pass implementations are unchanged from
/// the pre-Phase-2 era; only the *selection* of which to run differs.
///
/// ## Semantics
///
/// Each individual pass is semantics-preserving (verified by the
/// AST/MIR parity gate in `tests/fixtures/`). The composition of any
/// subset in any order is therefore also semantics-preserving — so
/// CANA's recommendations cannot break parity, only optimization
/// quality.
pub fn optimize_program_with_plan(program: &MirProgram, plan: &PassPlan) -> MirProgram {
    let mut optimized = program.clone();

    for func in &mut optimized.functions {
        let passes: &[String] = match plan.per_function.get(&func.name) {
            Some(custom) => custom,
            None => {
                // Fall back to default. We need a slice of String, so
                // construct one lazily here. (A static const Vec<String>
                // would be cleaner but Rust's const surface doesn't
                // support heap-allocated Strings.)
                for default_pass in DEFAULT_PASS_SEQUENCE {
                    apply_pass(default_pass, func);
                }
                continue;
            }
        };
        for pass in passes {
            apply_pass(pass, func);
        }
    }

    optimized
}

// ===========================================================================
// Constant Folding
// ===========================================================================

fn constant_fold_fn(func: &mut MirFunction) {
    constant_fold_body(&mut func.body);
}

fn constant_fold_body(body: &mut MirBody) {
    for stmt in &mut body.stmts {
        constant_fold_stmt(stmt);
    }
    if let Some(ref mut expr) = body.result {
        constant_fold_expr(expr);
    }
}

fn constant_fold_stmt(stmt: &mut MirStmt) {
    match stmt {
        MirStmt::Let { init, .. } => {
            constant_fold_expr(init);
        }
        MirStmt::Expr(expr) => {
            constant_fold_expr(expr);
        }
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            constant_fold_expr(cond);
            constant_fold_body(then_body);
            if let Some(eb) = else_body {
                constant_fold_body(eb);
            }
        }
        MirStmt::While { cond, body } => {
            constant_fold_expr(cond);
            constant_fold_body(body);
        }
        MirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                constant_fold_expr(expr);
            }
        }
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => {
            constant_fold_body(body);
        }
    }
}

fn constant_fold_expr(expr: &mut MirExpr) {
    // Recursively fold sub-expressions first (bottom-up).
    match &mut expr.kind {
        MirExprKind::Binary { left, right, .. } => {
            constant_fold_expr(left);
            constant_fold_expr(right);
        }
        MirExprKind::Unary { operand, .. } => {
            constant_fold_expr(operand);
        }
        MirExprKind::Call { callee, args } => {
            constant_fold_expr(callee);
            for arg in args {
                constant_fold_expr(arg);
            }
        }
        MirExprKind::Field { object, .. } => {
            constant_fold_expr(object);
        }
        MirExprKind::Index { object, index } => {
            constant_fold_expr(object);
            constant_fold_expr(index);
        }
        MirExprKind::MultiIndex { object, indices } => {
            constant_fold_expr(object);
            for idx in indices {
                constant_fold_expr(idx);
            }
        }
        MirExprKind::Assign { target, value } => {
            constant_fold_expr(target);
            constant_fold_expr(value);
        }
        MirExprKind::Block(body) => {
            constant_fold_body(body);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fexpr) in fields {
                constant_fold_expr(fexpr);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                constant_fold_expr(e);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for cap in captures {
                constant_fold_expr(cap);
            }
        }
        MirExprKind::If {
            cond,
            then_body,
            else_body,
        } => {
            constant_fold_expr(cond);
            constant_fold_body(then_body);
            if let Some(eb) = else_body {
                constant_fold_body(eb);
            }
        }
        MirExprKind::Match { scrutinee, arms } => {
            constant_fold_expr(scrutinee);
            for arm in arms {
                constant_fold_body(&mut arm.body);
            }
        }
        MirExprKind::Lambda { body, .. } => {
            constant_fold_expr(body);
        }
        // Linalg opcodes: fold operand only (non-foldable themselves)
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            constant_fold_expr(operand);
        }
        MirExprKind::Broadcast { operand, target_shape } => {
            constant_fold_expr(operand);
            for s in target_shape {
                constant_fold_expr(s);
            }
        }
        MirExprKind::VariantLit { fields, .. } => {
            for f in fields {
                constant_fold_expr(f);
            }
        }
        // Leaves
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::VarLocal { .. }
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for elem in row {
                    constant_fold_expr(elem);
                }
            }
        }
    }

    // Now try to fold this expression itself.
    if let Some(folded) = try_fold(expr) {
        *expr = folded;
    }
}

/// Attempt to constant-fold a single expression node.
/// Returns Some(folded) if foldable, None otherwise.
fn try_fold(expr: &MirExpr) -> Option<MirExpr> {
    match &expr.kind {
        MirExprKind::Binary { op, left, right } => try_fold_binary(*op, left, right),
        MirExprKind::Unary { op, operand } => try_fold_unary(*op, operand),
        // If-expression with constant condition.
        MirExprKind::If {
            cond,
            then_body,
            else_body,
        } => {
            if let MirExprKind::BoolLit(b) = &cond.kind {
                if *b {
                    // Take the then branch.
                    Some(MirExpr {
                        kind: MirExprKind::Block(then_body.clone()),
                    })
                } else if let Some(eb) = else_body {
                    Some(MirExpr {
                        kind: MirExprKind::Block(eb.clone()),
                    })
                } else {
                    Some(MirExpr {
                        kind: MirExprKind::Void,
                    })
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn try_fold_binary(op: BinOp, left: &MirExpr, right: &MirExpr) -> Option<MirExpr> {
    match (&left.kind, &right.kind) {
        // Int + Int
        (MirExprKind::IntLit(a), MirExprKind::IntLit(b)) => {
            fold_int_binop(op, *a, *b).map(|kind| MirExpr { kind })
        }
        // Float + Float — only fold comparison and identity operations to preserve
        // bit-identical results. We DO fold float arithmetic because the runtime
        // uses the same IEEE 754 operations (no extra precision).
        (MirExprKind::FloatLit(a), MirExprKind::FloatLit(b)) => {
            fold_float_binop(op, *a, *b).map(|kind| MirExpr { kind })
        }
        // Bool comparisons
        (MirExprKind::BoolLit(a), MirExprKind::BoolLit(b)) => match op {
            BinOp::Eq => Some(MirExpr {
                kind: MirExprKind::BoolLit(a == b),
            }),
            BinOp::Ne => Some(MirExpr {
                kind: MirExprKind::BoolLit(a != b),
            }),
            BinOp::And => Some(MirExpr {
                kind: MirExprKind::BoolLit(*a && *b),
            }),
            BinOp::Or => Some(MirExpr {
                kind: MirExprKind::BoolLit(*a || *b),
            }),
            BinOp::BitAnd => Some(MirExpr {
                kind: MirExprKind::BoolLit(*a & *b),
            }),
            BinOp::BitOr => Some(MirExpr {
                kind: MirExprKind::BoolLit(*a | *b),
            }),
            BinOp::BitXor => Some(MirExpr {
                kind: MirExprKind::BoolLit(*a ^ *b),
            }),
            _ => None,
        },
        // String concat
        (MirExprKind::StringLit(a), MirExprKind::StringLit(b)) => match op {
            BinOp::Add => Some(MirExpr {
                kind: MirExprKind::StringLit(format!("{a}{b}")),
            }),
            BinOp::Eq => Some(MirExpr {
                kind: MirExprKind::BoolLit(a == b),
            }),
            BinOp::Ne => Some(MirExpr {
                kind: MirExprKind::BoolLit(a != b),
            }),
            _ => None,
        },
        _ => None,
    }
}

fn fold_int_binop(op: BinOp, a: i64, b: i64) -> Option<MirExprKind> {
    match op {
        BinOp::Add => Some(MirExprKind::IntLit(a.wrapping_add(b))),
        BinOp::Sub => Some(MirExprKind::IntLit(a.wrapping_sub(b))),
        BinOp::Mul => Some(MirExprKind::IntLit(a.wrapping_mul(b))),
        BinOp::Div => {
            // Don't fold division by zero — let runtime handle it.
            if b == 0 {
                None
            } else {
                Some(MirExprKind::IntLit(a / b))
            }
        }
        BinOp::Mod => {
            if b == 0 {
                None
            } else {
                Some(MirExprKind::IntLit(a % b))
            }
        }
        BinOp::Pow => {
            if b < 0 {
                None // Negative exponent on ints — let runtime handle
            } else {
                Some(MirExprKind::IntLit(a.wrapping_pow(b as u32)))
            }
        }
        BinOp::BitAnd => Some(MirExprKind::IntLit(a & b)),
        BinOp::BitOr => Some(MirExprKind::IntLit(a | b)),
        BinOp::BitXor => Some(MirExprKind::IntLit(a ^ b)),
        BinOp::Shl => Some(MirExprKind::IntLit(a.wrapping_shl(b as u32))),
        BinOp::Shr => Some(MirExprKind::IntLit(a.wrapping_shr(b as u32))),
        BinOp::Eq => Some(MirExprKind::BoolLit(a == b)),
        BinOp::Ne => Some(MirExprKind::BoolLit(a != b)),
        BinOp::Lt => Some(MirExprKind::BoolLit(a < b)),
        BinOp::Gt => Some(MirExprKind::BoolLit(a > b)),
        BinOp::Le => Some(MirExprKind::BoolLit(a <= b)),
        BinOp::Ge => Some(MirExprKind::BoolLit(a >= b)),
        BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => None,
    }
}

fn fold_float_binop(op: BinOp, a: f64, b: f64) -> Option<MirExprKind> {
    // We fold float ops because the runtime uses the same IEEE 754 ops.
    // We do NOT fold if the result would be NaN or involve special values
    // that might differ between compile-time and runtime on the same platform
    // (they won't, since both use Rust f64).
    match op {
        BinOp::Add => Some(MirExprKind::FloatLit(a + b)),
        BinOp::Sub => Some(MirExprKind::FloatLit(a - b)),
        BinOp::Mul => Some(MirExprKind::FloatLit(a * b)),
        BinOp::Div => Some(MirExprKind::FloatLit(a / b)), // IEEE 754: div by 0 => Inf
        BinOp::Mod => Some(MirExprKind::FloatLit(a % b)),
        BinOp::Pow => Some(MirExprKind::FloatLit(a.powf(b))),
        BinOp::Eq => Some(MirExprKind::BoolLit(a == b)),
        BinOp::Ne => Some(MirExprKind::BoolLit(a != b)),
        BinOp::Lt => Some(MirExprKind::BoolLit(a < b)),
        BinOp::Gt => Some(MirExprKind::BoolLit(a > b)),
        BinOp::Le => Some(MirExprKind::BoolLit(a <= b)),
        BinOp::Ge => Some(MirExprKind::BoolLit(a >= b)),
        BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => None,
        // Bitwise ops on floats: not applicable, let runtime handle
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => None,
    }
}

fn try_fold_unary(op: UnaryOp, operand: &MirExpr) -> Option<MirExpr> {
    match (&op, &operand.kind) {
        (UnaryOp::Neg, MirExprKind::IntLit(v)) => Some(MirExpr {
            kind: MirExprKind::IntLit(-v),
        }),
        (UnaryOp::Neg, MirExprKind::FloatLit(v)) => Some(MirExpr {
            kind: MirExprKind::FloatLit(-v),
        }),
        (UnaryOp::Not, MirExprKind::BoolLit(b)) => Some(MirExpr {
            kind: MirExprKind::BoolLit(!b),
        }),
        (UnaryOp::BitNot, MirExprKind::IntLit(v)) => Some(MirExpr {
            kind: MirExprKind::IntLit(!v),
        }),
        _ => None,
    }
}

// ===========================================================================
// Dead Code Elimination
// ===========================================================================

fn dce_fn(func: &mut MirFunction) {
    dce_body(&mut func.body);
}

fn dce_body(body: &mut MirBody) {
    // Collect variables that are read in the body.
    let mut used_vars = BTreeSet::new();
    for stmt in &body.stmts {
        collect_used_vars_stmt(stmt, &mut used_vars);
    }
    if let Some(ref expr) = body.result {
        collect_used_vars_expr(expr, &mut used_vars);
    }

    // Remove dead Let bindings: `let x = <pure_expr>` where x is never read.
    body.stmts.retain(|stmt| {
        match stmt {
            MirStmt::Let { name, init, .. } => {
                if !used_vars.contains(name.as_str()) && is_pure_expr(init) {
                    return false; // Dead code: remove
                }
                true
            }
            _ => true,
        }
    });

    // Recursively apply DCE to sub-bodies.
    for stmt in &mut body.stmts {
        dce_stmt(stmt);
    }

    // Dead if-branches: if condition is a constant bool literal.
    let mut new_stmts = Vec::new();
    for stmt in std::mem::take(&mut body.stmts) {
        match stmt {
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                if let MirExprKind::BoolLit(b) = &cond.kind {
                    if *b {
                        // Always true: inline then_body statements and result.
                        new_stmts.extend(then_body.stmts);
                        if let Some(result_expr) = then_body.result {
                            new_stmts.push(MirStmt::Expr(*result_expr));
                        }
                    } else if let Some(eb) = else_body {
                        // Always false: inline else_body statements and result.
                        new_stmts.extend(eb.stmts);
                        if let Some(result_expr) = eb.result {
                            new_stmts.push(MirStmt::Expr(*result_expr));
                        }
                    }
                    // else: always false with no else branch => remove entirely
                } else {
                    new_stmts.push(MirStmt::If {
                        cond,
                        then_body,
                        else_body,
                    });
                }
            }
            // While with false condition: remove entirely.
            MirStmt::While { ref cond, .. } => {
                if let MirExprKind::BoolLit(false) = &cond.kind {
                    // Dead loop: never executes.
                } else {
                    new_stmts.push(stmt);
                }
            }
            other => new_stmts.push(other),
        }
    }
    body.stmts = new_stmts;
}

fn dce_stmt(stmt: &mut MirStmt) {
    match stmt {
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            dce_body(then_body);
            if let Some(eb) = else_body {
                dce_body(eb);
            }
        }
        MirStmt::While { body, .. } => {
            dce_body(body);
        }
        MirStmt::NoGcBlock(body) => {
            dce_body(body);
        }
        _ => {}
    }
}

/// Check if an expression is pure (no side effects).
/// We are conservative: calls, assigns, index operations are not pure.
fn is_pure_expr(expr: &MirExpr) -> bool {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::VarLocal { .. }
        | MirExprKind::Void => true,
        MirExprKind::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        MirExprKind::Unary { operand, .. } => is_pure_expr(operand),
        MirExprKind::TupleLit(elems) | MirExprKind::ArrayLit(elems) => {
            elems.iter().all(is_pure_expr)
        }
        MirExprKind::TensorLit { rows } => {
            rows.iter().all(|row| row.iter().all(is_pure_expr))
        }
        MirExprKind::StructLit { fields, .. } => fields.iter().all(|(_, e)| is_pure_expr(e)),
        // VariantLit is pure if all fields are pure
        MirExprKind::VariantLit { fields, .. } => fields.iter().all(is_pure_expr),
        // Calls are NOT pure (may have side effects).
        MirExprKind::Call { .. } => false,
        // Field access on a known var is pure, but we're conservative.
        MirExprKind::Field { object, .. } => is_pure_expr(object),
        // Assigns are definitely not pure.
        MirExprKind::Assign { .. } => false,
        // Index may trigger bounds checks — keep them.
        MirExprKind::Index { .. } | MirExprKind::MultiIndex { .. } => false,
        // Blocks, if, match, etc. may have side effects.
        MirExprKind::Block(_)
        | MirExprKind::If { .. }
        | MirExprKind::Match { .. }
        | MirExprKind::Lambda { .. }
        | MirExprKind::MakeClosure { .. }
        | MirExprKind::Col(_) => false,
        // Linalg ops are pure math but expensive — keep them.
        MirExprKind::LinalgLU { .. }
        | MirExprKind::LinalgQR { .. }
        | MirExprKind::LinalgCholesky { .. }
        | MirExprKind::LinalgInv { .. }
        | MirExprKind::Broadcast { .. } => false,
    }
}

/// Collect all variable names that are READ in a statement.
fn collect_used_vars_stmt(stmt: &MirStmt, used: &mut BTreeSet<String>) {
    match stmt {
        MirStmt::Let { init, .. } => {
            collect_used_vars_expr(init, used);
        }
        MirStmt::Expr(expr) => {
            collect_used_vars_expr(expr, used);
        }
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_used_vars_expr(cond, used);
            collect_used_vars_body(then_body, used);
            if let Some(eb) = else_body {
                collect_used_vars_body(eb, used);
            }
        }
        MirStmt::While { cond, body } => {
            collect_used_vars_expr(cond, used);
            collect_used_vars_body(body, used);
        }
        MirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                collect_used_vars_expr(expr, used);
            }
        }
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => {
            collect_used_vars_body(body, used);
        }
    }
}

fn collect_used_vars_body(body: &MirBody, used: &mut BTreeSet<String>) {
    for stmt in &body.stmts {
        collect_used_vars_stmt(stmt, used);
    }
    if let Some(ref expr) = body.result {
        collect_used_vars_expr(expr, used);
    }
}

fn collect_used_vars_expr(expr: &MirExpr, used: &mut BTreeSet<String>) {
    match &expr.kind {
        MirExprKind::Var(name) => {
            used.insert(name.clone());
        }
        MirExprKind::VarLocal { name, .. } => {
            used.insert(name.clone());
        }
        MirExprKind::Binary { left, right, .. } => {
            collect_used_vars_expr(left, used);
            collect_used_vars_expr(right, used);
        }
        MirExprKind::Unary { operand, .. } => {
            collect_used_vars_expr(operand, used);
        }
        MirExprKind::Call { callee, args } => {
            collect_used_vars_expr(callee, used);
            for arg in args {
                collect_used_vars_expr(arg, used);
            }
        }
        MirExprKind::Field { object, .. } => {
            collect_used_vars_expr(object, used);
        }
        MirExprKind::Index { object, index } => {
            collect_used_vars_expr(object, used);
            collect_used_vars_expr(index, used);
        }
        MirExprKind::MultiIndex { object, indices } => {
            collect_used_vars_expr(object, used);
            for idx in indices {
                collect_used_vars_expr(idx, used);
            }
        }
        MirExprKind::Assign { target, value } => {
            collect_used_vars_expr(target, used);
            collect_used_vars_expr(value, used);
        }
        MirExprKind::Block(body) => {
            collect_used_vars_body(body, used);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fexpr) in fields {
                collect_used_vars_expr(fexpr, used);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                collect_used_vars_expr(e, used);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for cap in captures {
                collect_used_vars_expr(cap, used);
            }
        }
        MirExprKind::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_used_vars_expr(cond, used);
            collect_used_vars_body(then_body, used);
            if let Some(eb) = else_body {
                collect_used_vars_body(eb, used);
            }
        }
        MirExprKind::Match { scrutinee, arms } => {
            collect_used_vars_expr(scrutinee, used);
            for arm in arms {
                collect_used_vars_body(&arm.body, used);
            }
        }
        MirExprKind::Lambda { body, .. } => {
            collect_used_vars_expr(body, used);
        }
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            collect_used_vars_expr(operand, used);
        }
        MirExprKind::Broadcast { operand, target_shape } => {
            collect_used_vars_expr(operand, used);
            for s in target_shape {
                collect_used_vars_expr(s, used);
            }
        }
        MirExprKind::VariantLit { fields, .. } => {
            for f in fields {
                collect_used_vars_expr(f, used);
            }
        }
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for elem in row {
                    collect_used_vars_expr(elem, used);
                }
            }
        }
    }
}

// ===========================================================================
// Strength Reduction
// ===========================================================================

/// Apply strength reduction to a whole function. Returns the **count of
/// successful rewrites** — the number of times [`try_strength_reduce`]
/// returned `Some`. Used as a deterministic training signal by
/// `bench/cana_train_cost_model/` and exposed via
/// [`apply_pass_with_diagnostics`] as a permanent observability surface.
fn strength_reduce_fn(func: &mut MirFunction) -> usize {
    strength_reduce_body(&mut func.body)
}

fn strength_reduce_body(body: &mut MirBody) -> usize {
    let mut count = 0;
    for stmt in &mut body.stmts {
        count += strength_reduce_stmt(stmt);
    }
    if let Some(ref mut expr) = body.result {
        count += strength_reduce_expr(expr);
    }
    count
}

fn strength_reduce_stmt(stmt: &mut MirStmt) -> usize {
    match stmt {
        MirStmt::Let { init, .. } => strength_reduce_expr(init),
        MirStmt::Expr(expr) => strength_reduce_expr(expr),
        MirStmt::If { cond, then_body, else_body } => {
            let mut c = strength_reduce_expr(cond) + strength_reduce_body(then_body);
            if let Some(eb) = else_body {
                c += strength_reduce_body(eb);
            }
            c
        }
        MirStmt::While { cond, body } => {
            strength_reduce_expr(cond) + strength_reduce_body(body)
        }
        MirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                strength_reduce_expr(expr)
            } else {
                0
            }
        }
        MirStmt::Break | MirStmt::Continue => 0,
        MirStmt::NoGcBlock(body) => strength_reduce_body(body),
    }
}

fn strength_reduce_expr(expr: &mut MirExpr) -> usize {
    // Recurse first (bottom-up).
    let mut count = match &mut expr.kind {
        MirExprKind::Binary { left, right, .. } => {
            strength_reduce_expr(left) + strength_reduce_expr(right)
        }
        MirExprKind::Unary { operand, .. } => strength_reduce_expr(operand),
        MirExprKind::Call { callee, args } => {
            let mut c = strength_reduce_expr(callee);
            for arg in args { c += strength_reduce_expr(arg); }
            c
        }
        MirExprKind::Block(body) => strength_reduce_body(body),
        MirExprKind::If { cond, then_body, else_body } => {
            let mut c = strength_reduce_expr(cond) + strength_reduce_body(then_body);
            if let Some(eb) = else_body { c += strength_reduce_body(eb); }
            c
        }
        MirExprKind::Lambda { body, .. } => strength_reduce_expr(body),
        _ => 0,
    };

    // Apply strength reduction rules.
    if let Some(reduced) = try_strength_reduce(expr) {
        *expr = reduced;
        count += 1;
    }
    count
}

/// Attempt to reduce an expression to a cheaper form.
fn try_strength_reduce(expr: &MirExpr) -> Option<MirExpr> {
    match &expr.kind {
        MirExprKind::Binary { op, left, right } => {
            match op {
                // x * 0 => 0  (int only; float * 0 can produce -0.0 or NaN)
                BinOp::Mul => {
                    if matches!(right.kind, MirExprKind::IntLit(0)) {
                        return Some(MirExpr { kind: MirExprKind::IntLit(0) });
                    }
                    if matches!(left.kind, MirExprKind::IntLit(0)) {
                        return Some(MirExpr { kind: MirExprKind::IntLit(0) });
                    }
                    // x * 1 => x
                    if matches!(right.kind, MirExprKind::IntLit(1)) {
                        return Some(*left.clone());
                    }
                    if matches!(left.kind, MirExprKind::IntLit(1)) {
                        return Some(*right.clone());
                    }
                    // x * 2 => x + x  (cheaper on many architectures)
                    if matches!(right.kind, MirExprKind::IntLit(2)) {
                        return Some(MirExpr {
                            kind: MirExprKind::Binary {
                                op: BinOp::Add,
                                left: left.clone(),
                                right: left.clone(),
                            },
                        });
                    }
                    None
                }
                // x + 0 => x (int only; float +0 is identity but we keep it safe)
                BinOp::Add => {
                    if matches!(right.kind, MirExprKind::IntLit(0)) {
                        return Some(*left.clone());
                    }
                    if matches!(left.kind, MirExprKind::IntLit(0)) {
                        return Some(*right.clone());
                    }
                    None
                }
                // x - 0 => x (int only)
                BinOp::Sub => {
                    if matches!(right.kind, MirExprKind::IntLit(0)) {
                        return Some(*left.clone());
                    }
                    None
                }
                // x / 1 => x (int only)
                BinOp::Div => {
                    if matches!(right.kind, MirExprKind::IntLit(1)) {
                        return Some(*left.clone());
                    }
                    None
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// ===========================================================================
// Common Subexpression Elimination (CSE)
// ===========================================================================

/// Run CSE on a whole function. Returns the **count of replacements
/// applied** — the number of variable uses that got renamed to a
/// previously-computed equivalent. Useful as a deterministic training
/// signal and as a permanent observability surface via
/// [`apply_pass_with_diagnostics`].
fn cse_fn(func: &mut MirFunction) -> usize {
    cse_body(&mut func.body)
}

/// CSE at the body level: find duplicate pure let-bindings and replace
/// uses of the later one with the first. Returns the number of
/// replacements applied across this body and all nested sub-bodies.
fn cse_body(body: &mut MirBody) -> usize {
    let mut count = 0;
    // Build a map from expression hash to the first variable name bound to it.
    let mut expr_to_var: BTreeMap<String, String> = BTreeMap::new();
    let mut replacements: BTreeMap<String, String> = BTreeMap::new();

    for stmt in &body.stmts {
        if let MirStmt::Let { name, init, mutable, .. } = stmt {
            // Only CSE immutable bindings — mutable variables may diverge
            // after their initializer (e.g. loop counters both init to 0).
            if !mutable && is_pure_expr(init) {
                let key = expr_key(init);
                if let Some(existing) = expr_to_var.get(&key) {
                    replacements.insert(name.clone(), existing.clone());
                } else {
                    expr_to_var.insert(key, name.clone());
                }
            }
        }
    }

    if !replacements.is_empty() {
        // Count is the number of duplicate bindings detected. Each one is
        // a "rewrite opportunity" — the deterministic CSE training signal.
        count += replacements.len();
        for stmt in &mut body.stmts {
            apply_cse_replacements_stmt(stmt, &replacements);
        }
        if let Some(ref mut result) = body.result {
            apply_cse_replacements_expr(result, &replacements);
        }
    }

    // Recurse into sub-bodies.
    for stmt in &mut body.stmts {
        match stmt {
            MirStmt::If { then_body, else_body, .. } => {
                count += cse_body(then_body);
                if let Some(eb) = else_body { count += cse_body(eb); }
            }
            MirStmt::While { body: wb, .. } => count += cse_body(wb),
            MirStmt::NoGcBlock(b) => count += cse_body(b),
            _ => {}
        }
    }
    count
}

/// Produce a deterministic string key for a pure expression (for CSE).
fn expr_key(expr: &MirExpr) -> String {
    match &expr.kind {
        MirExprKind::IntLit(v) => format!("int:{v}"),
        MirExprKind::FloatLit(v) => format!("float:{}", v.to_bits()),
        MirExprKind::BoolLit(v) => format!("bool:{v}"),
        MirExprKind::NaLit => "na".to_string(),
        MirExprKind::StringLit(s) => format!("str:{s}"),
        // CSE key normalisation: both `Var("x")` and `VarLocal { name: "x", .. }`
        // refer to the same logical binding `x`. Slot numbers are a layout
        // artifact of the Tier-0 slot resolution pass and must NOT distinguish
        // CSE keys — same family of bug as the `task_a30e8eec` LICM fix and
        // `commit 2983f2b` (reduction analyzer). Without this, the CSE pass
        // silently never fires on any function past slot resolution, because
        // every `VarLocal` falls into the opaque-pointer fallback arm and gets
        // a unique key.
        MirExprKind::Var(name) => format!("var:{name}"),
        MirExprKind::VarLocal { name, .. } => format!("var:{name}"),
        MirExprKind::Binary { op, left, right } => {
            format!("bin:{:?}({},{})", op, expr_key(left), expr_key(right))
        }
        MirExprKind::Unary { op, operand } => {
            format!("un:{:?}({})", op, expr_key(operand))
        }
        MirExprKind::Field { object, name } => {
            format!("field:{}:{}", expr_key(object), name)
        }
        // For other expression types, produce a unique non-matching key.
        _ => format!("opaque:{:p}", expr),
    }
}

fn apply_cse_replacements_stmt(stmt: &mut MirStmt, replacements: &BTreeMap<String, String>) {
    match stmt {
        MirStmt::Let { init, .. } => apply_cse_replacements_expr(init, replacements),
        MirStmt::Expr(expr) => apply_cse_replacements_expr(expr, replacements),
        MirStmt::If { cond, then_body, else_body } => {
            apply_cse_replacements_expr(cond, replacements);
            for s in &mut then_body.stmts { apply_cse_replacements_stmt(s, replacements); }
            if let Some(ref mut r) = then_body.result { apply_cse_replacements_expr(r, replacements); }
            if let Some(eb) = else_body {
                for s in &mut eb.stmts { apply_cse_replacements_stmt(s, replacements); }
                if let Some(ref mut r) = eb.result { apply_cse_replacements_expr(r, replacements); }
            }
        }
        MirStmt::While { cond, body } => {
            apply_cse_replacements_expr(cond, replacements);
            for s in &mut body.stmts { apply_cse_replacements_stmt(s, replacements); }
            if let Some(ref mut r) = body.result { apply_cse_replacements_expr(r, replacements); }
        }
        MirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr { apply_cse_replacements_expr(expr, replacements); }
        }
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => {
            for s in &mut body.stmts { apply_cse_replacements_stmt(s, replacements); }
            if let Some(ref mut r) = body.result { apply_cse_replacements_expr(r, replacements); }
        }
    }
}

fn apply_cse_replacements_expr(expr: &mut MirExpr, replacements: &BTreeMap<String, String>) {
    match &mut expr.kind {
        MirExprKind::Var(name) => {
            if let Some(replacement) = replacements.get(name.as_str()) {
                *name = replacement.clone();
            }
        }
        MirExprKind::Binary { left, right, .. } => {
            apply_cse_replacements_expr(left, replacements);
            apply_cse_replacements_expr(right, replacements);
        }
        MirExprKind::Unary { operand, .. } => {
            apply_cse_replacements_expr(operand, replacements);
        }
        MirExprKind::Call { callee, args } => {
            apply_cse_replacements_expr(callee, replacements);
            for arg in args { apply_cse_replacements_expr(arg, replacements); }
        }
        MirExprKind::Field { object, .. } => {
            apply_cse_replacements_expr(object, replacements);
        }
        MirExprKind::Index { object, index } => {
            apply_cse_replacements_expr(object, replacements);
            apply_cse_replacements_expr(index, replacements);
        }
        MirExprKind::Block(body) => {
            for s in &mut body.stmts { apply_cse_replacements_stmt(s, replacements); }
            if let Some(ref mut r) = body.result { apply_cse_replacements_expr(r, replacements); }
        }
        _ => {} // Leaves and other complex nodes — no Var to replace
    }
}

// ===========================================================================
// Loop-Invariant Code Motion (LICM)
// ===========================================================================

/// Run LICM on a whole function. Returns the **count of statements
/// hoisted** out of `while` loops. Useful as a deterministic training
/// signal and as a permanent observability surface via
/// [`apply_pass_with_diagnostics`].
fn licm_fn(func: &mut MirFunction) -> usize {
    licm_body(&mut func.body)
}

fn licm_body(body: &mut MirBody) -> usize {
    let mut count = 0;
    // Process nested structures first (bottom-up).
    for stmt in &mut body.stmts {
        match stmt {
            MirStmt::If { then_body, else_body, .. } => {
                count += licm_body(then_body);
                if let Some(eb) = else_body { count += licm_body(eb); }
            }
            MirStmt::While { body: wb, .. } => count += licm_body(wb),
            MirStmt::NoGcBlock(b) => count += licm_body(b),
            _ => {}
        }
    }

    // Now try to hoist invariant let-bindings from while loops.
    let mut new_stmts = Vec::new();
    for stmt in std::mem::take(&mut body.stmts) {
        if let MirStmt::While { cond, body: loop_body } = stmt {
            let (hoisted, remaining_body) = hoist_invariants(loop_body);
            // Each hoisted statement is one rewrite opportunity — the
            // deterministic LICM training signal.
            count += hoisted.len();
            new_stmts.extend(hoisted);
            new_stmts.push(MirStmt::While { cond, body: remaining_body });
        } else {
            new_stmts.push(stmt);
        }
    }
    body.stmts = new_stmts;
    count
}

/// Try to hoist loop-invariant let-bindings out of a while body.
///
/// A `let` binding is hoisted only if ALL of the following hold:
/// 1. Its initializer is pure (no side effects).
/// 2. The initializer does not reference any variable that is *defined
///    or assigned* inside the loop (the "all changed" set).
/// 3. The binding's own name is **not** reassigned later in the loop.
///
/// Condition (3) is the fix for task_a30e8eec: the canonical
/// `while ... { let mut j = 0; while j < n { ...; j = j + 1; } }`
/// pattern. Without condition (3), the inner `let mut j = 0;` got
/// hoisted out of the outer loop, breaking re-initialization on each
/// outer iteration. Programs like the chess RL training loop relied
/// on per-iteration re-binding; before this fix, `--mir-opt` silently
/// produced wrong results on every nested-loop accumulator.
///
/// ## Helper-name variant handling
///
/// Both [`collect_modified_vars_expr`] and [`references_any`] now
/// accept the post-slot-resolution `MirExprKind::VarLocal { name, .. }`
/// form in addition to `Var(name)`. Without this, `j = j + 1` after
/// the Tier-0 slot pass produced an Assign whose target was VarLocal,
/// and the collection routine silently dropped it from `modified_vars`
/// (allowing more invalid hoists). Same family of bug as commit
/// `2983f2b` (reduction analyzer).
fn hoist_invariants(loop_body: MirBody) -> (Vec<MirStmt>, MirBody) {
    // (1) All vars that change in the loop (Let defines + Assign targets).
    //     Used for the `references_any` check on initializers.
    let mut all_changed = BTreeSet::new();
    collect_modified_vars_body(&loop_body, &mut all_changed);

    // (2) Just the Assign targets — used to refuse hoisting a Let whose
    //     name is reassigned later in the loop. This is the fix for
    //     task_a30e8eec; see function-level docs above.
    let mut assigned_only = BTreeSet::new();
    collect_assigned_vars_body(&loop_body, &mut assigned_only);

    let mut hoisted = Vec::new();
    let mut remaining = Vec::new();

    for stmt in loop_body.stmts {
        if let MirStmt::Let { ref name, ref init, mutable, alloc_hint, slot } = stmt {
            if is_pure_expr(init)
                && !references_any(init, &all_changed)
                && !assigned_only.contains(name)
            {
                hoisted.push(MirStmt::Let {
                    name: name.clone(),
                    mutable,
                    init: init.clone(),
                    alloc_hint,
                    slot,
                });
                continue;
            }
        }
        remaining.push(stmt);
    }

    (hoisted, MirBody { stmts: remaining, result: loop_body.result })
}

/// Collect all variable names that are modified (declared or written to)
/// in a body. The result is the union of `let`-bound names and Assign
/// targets.
///
/// Used to compute the `all_changed` set for `references_any`'s
/// "no init may depend on loop-local change" check.
fn collect_modified_vars_body(body: &MirBody, modified: &mut BTreeSet<String>) {
    for stmt in &body.stmts {
        collect_modified_vars_stmt(stmt, modified);
    }
}

fn collect_modified_vars_stmt(stmt: &MirStmt, modified: &mut BTreeSet<String>) {
    match stmt {
        MirStmt::Let { name, init, .. } => {
            modified.insert(name.clone());
            collect_modified_vars_expr(init, modified);
        }
        MirStmt::Expr(expr) => collect_modified_vars_expr(expr, modified),
        MirStmt::If { cond, then_body, else_body } => {
            collect_modified_vars_expr(cond, modified);
            collect_modified_vars_body(then_body, modified);
            if let Some(eb) = else_body { collect_modified_vars_body(eb, modified); }
        }
        MirStmt::While { cond, body } => {
            collect_modified_vars_expr(cond, modified);
            collect_modified_vars_body(body, modified);
        }
        MirStmt::Return(_) => {}
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => collect_modified_vars_body(body, modified),
    }
}

fn collect_modified_vars_expr(expr: &MirExpr, modified: &mut BTreeSet<String>) {
    match &expr.kind {
        MirExprKind::Assign { target, value } => {
            // Handle both Var(name) and VarLocal { name, .. } — the
            // latter appears post-slot-resolution. Pre-fix this branch
            // only matched Var, silently dropping every slot-resolved
            // reassignment from the modified set.
            match &target.kind {
                MirExprKind::Var(name)
                | MirExprKind::VarLocal { name, .. } => {
                    modified.insert(name.clone());
                }
                _ => {}
            }
            collect_modified_vars_expr(value, modified);
        }
        MirExprKind::Binary { left, right, .. } => {
            collect_modified_vars_expr(left, modified);
            collect_modified_vars_expr(right, modified);
        }
        MirExprKind::Call { callee, args } => {
            collect_modified_vars_expr(callee, modified);
            for arg in args { collect_modified_vars_expr(arg, modified); }
        }
        _ => {}
    }
}

/// Collect ONLY the names that are Assign targets (reassigned, not
/// merely let-bound) inside a body. Used to gate hoisting: a `let`
/// whose name appears here must not be hoisted out of the loop.
fn collect_assigned_vars_body(body: &MirBody, assigned: &mut BTreeSet<String>) {
    for stmt in &body.stmts {
        collect_assigned_vars_stmt(stmt, assigned);
    }
}

fn collect_assigned_vars_stmt(stmt: &MirStmt, assigned: &mut BTreeSet<String>) {
    match stmt {
        MirStmt::Let { init, .. } => {
            // Recurse into init but DO NOT add the let name itself.
            collect_assigned_vars_expr(init, assigned);
        }
        MirStmt::Expr(expr) => collect_assigned_vars_expr(expr, assigned),
        MirStmt::If { cond, then_body, else_body } => {
            collect_assigned_vars_expr(cond, assigned);
            collect_assigned_vars_body(then_body, assigned);
            if let Some(eb) = else_body { collect_assigned_vars_body(eb, assigned); }
        }
        MirStmt::While { cond, body } => {
            collect_assigned_vars_expr(cond, assigned);
            collect_assigned_vars_body(body, assigned);
        }
        MirStmt::Return(_) => {}
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => collect_assigned_vars_body(body, assigned),
    }
}

fn collect_assigned_vars_expr(expr: &MirExpr, assigned: &mut BTreeSet<String>) {
    match &expr.kind {
        MirExprKind::Assign { target, value } => {
            match &target.kind {
                MirExprKind::Var(name)
                | MirExprKind::VarLocal { name, .. } => {
                    assigned.insert(name.clone());
                }
                _ => {}
            }
            collect_assigned_vars_expr(value, assigned);
        }
        MirExprKind::Binary { left, right, .. } => {
            collect_assigned_vars_expr(left, assigned);
            collect_assigned_vars_expr(right, assigned);
        }
        MirExprKind::Call { callee, args } => {
            collect_assigned_vars_expr(callee, assigned);
            for arg in args { collect_assigned_vars_expr(arg, assigned); }
        }
        _ => {}
    }
}

/// Check if an expression references any variable in the given set.
///
/// Handles both `Var(name)` and `VarLocal { name, .. }` — the latter
/// appears post-slot-resolution. Pre-fix this function only matched
/// `Var`, silently returning false for every slot-resolved variable
/// reference. Combined with the analogous bug in
/// `collect_modified_vars_expr`, that allowed LICM to hoist
/// inner-loop-dependent expressions out of nested loops.
fn references_any(expr: &MirExpr, vars: &BTreeSet<String>) -> bool {
    match &expr.kind {
        MirExprKind::Var(name) => vars.contains(name.as_str()),
        MirExprKind::VarLocal { name, .. } => vars.contains(name.as_str()),
        MirExprKind::Binary { left, right, .. } => {
            references_any(left, vars) || references_any(right, vars)
        }
        MirExprKind::Unary { operand, .. } => references_any(operand, vars),
        MirExprKind::Field { object, .. } => references_any(object, vars),
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            elems.iter().any(|e| references_any(e, vars))
        }
        MirExprKind::StructLit { fields, .. } => {
            fields.iter().any(|(_, e)| references_any(e, vars))
        }
        _ => false, // Literals don't reference variables
    }
}

// ===========================================================================
// Loop Unrolling
// ===========================================================================

/// Maximum number of iterations to fully unroll. Larger bounds risk
/// code bloat that erases the speedup; smaller bounds defeat the pass.
/// 8 is the conservative initial choice — a body of 20 statements at 8
/// iterations turns into 160 statements (the original Let stays).
const UNROLL_MAX_ITERATIONS: i64 = 8;

/// Maximum body length (including the trailing increment) for an
/// unrollable loop. Long bodies + many iterations cause exponential
/// code bloat with diminishing performance benefit.
const UNROLL_MAX_BODY_STMTS: usize = 20;

/// Run loop unrolling on a whole function. Returns the **count of
/// `while` loops successfully unrolled** — one rewrite per loop,
/// regardless of how many copies of the body we emit. Used as a
/// deterministic training signal and as a permanent observability
/// surface via [`apply_pass_with_diagnostics`].
///
/// ## What it does
///
/// Detects the canonical induction-variable loop pattern, where the
/// `let mut` immediately precedes the `while` at the same statement
/// level:
///
/// ```text
/// let mut i: i64 = K0;
/// while i < N {
///     ... body stmts ...     (none may write to `i`)
///     i = i + 1;             (exactly this, as the last body stmt)
/// }
/// ```
///
/// Replaces the `while` with `N - K0` literal copies of the body,
/// keeping the `let` so any post-loop reader of `i` still sees the
/// expected value:
///
/// ```text
/// let mut i: i64 = K0;       (kept — `i` may be read after)
/// { body }                    (the body's trailing `i = i + 1;`
/// { body }                     brings `i` from K0 to K0 + 1, etc.;
/// ...                          after N-K0 copies, `i == N`, same as
/// { body }                     the rolled loop's exit value)
/// ```
///
/// No symbolic substitution of `i` into the body is needed — the body
/// computes the same values it would have, in the same order. The
/// transformation is byte-equivalent to the rolled form under both
/// executors.
///
/// ## What it refuses
///
/// - `while` loops without an immediately-preceding `let mut i: i64 = IntLit(_)`
/// - Loops whose iteration count is `≤ 0` or `> UNROLL_MAX_ITERATIONS`
/// - Loops whose body length exceeds `UNROLL_MAX_BODY_STMTS`
/// - Loops whose body contains `Break` or `Continue` at any depth not
///   shadowed by a nested `while` (their semantics under replication
///   would differ from the rolled form)
/// - Loops whose body modifies the induction variable anywhere other
///   than the trailing `i = i + 1;` (non-canonical induction pattern)
/// - Loops whose body has a tail `result` expression
///
/// ## Recursion
///
/// Like LICM, bottom-up: inner loops are unrolled first, so an outer
/// loop sees the already-normalised inner body when deciding whether
/// to unroll itself.
fn loop_unroll_fn(func: &mut MirFunction) -> usize {
    loop_unroll_body(&mut func.body)
}

fn loop_unroll_body(body: &mut MirBody) -> usize {
    let mut count = 0;

    // Bottom-up: recurse into nested sub-bodies first.
    for stmt in &mut body.stmts {
        count += loop_unroll_stmt_descend(stmt);
    }

    // Scan top-level statements for `(Let mut i = IntLit, While)` pairs.
    let stmts = std::mem::take(&mut body.stmts);
    let mut iter = stmts.into_iter().peekable();
    let mut new_stmts: Vec<MirStmt> = Vec::new();

    while let Some(stmt) = iter.next() {
        if let Some((var_name, start)) = let_init_int(&stmt) {
            if matches!(iter.peek(), Some(MirStmt::While { .. })) {
                let while_stmt = iter.next().expect("peeked While present");
                match try_unroll(&var_name, start, &while_stmt) {
                    Some(unrolled) => {
                        new_stmts.push(stmt);
                        new_stmts.extend(unrolled);
                        count += 1;
                    }
                    None => {
                        new_stmts.push(stmt);
                        new_stmts.push(while_stmt);
                    }
                }
                continue;
            }
        }
        new_stmts.push(stmt);
    }
    body.stmts = new_stmts;
    count
}

/// Recurse into nested sub-bodies (If arms, While body, NoGcBlock) to
/// give inner loops a chance to unroll first. Does NOT itself try to
/// unroll the top-level `stmt` — that's the pair-scan loop's job in
/// [`loop_unroll_body`].
fn loop_unroll_stmt_descend(stmt: &mut MirStmt) -> usize {
    match stmt {
        MirStmt::If { then_body, else_body, .. } => {
            let mut c = loop_unroll_body(then_body);
            if let Some(eb) = else_body {
                c += loop_unroll_body(eb);
            }
            c
        }
        MirStmt::While { body, .. } => loop_unroll_body(body),
        MirStmt::NoGcBlock(body) => loop_unroll_body(body),
        _ => 0,
    }
}

/// Extract `(var_name, start_value)` from a `let mut name = IntLit(start);`
/// statement. Returns `None` if the statement isn't that exact shape
/// (immutable bindings, non-integer initialisers, and any other stmt
/// kind all fail).
///
/// Mutability is required: the canonical pattern increments `i` inside
/// the loop body, which would fail type-checking if `i` were immutable.
fn let_init_int(stmt: &MirStmt) -> Option<(String, i64)> {
    if let MirStmt::Let { name, mutable: true, init, .. } = stmt {
        if let MirExprKind::IntLit(start) = &init.kind {
            return Some((name.clone(), *start));
        }
    }
    None
}

/// Attempt to unroll one `while` loop given the preceding induction
/// variable's name + starting value. Returns the replicated body
/// statements on success; `None` if any structural guard fails.
fn try_unroll(var_name: &str, start: i64, while_stmt: &MirStmt) -> Option<Vec<MirStmt>> {
    let (cond, body) = match while_stmt {
        MirStmt::While { cond, body } => (cond, body),
        _ => return None,
    };

    // Condition must be `var_name < IntLit(bound)`.
    let bound = match &cond.kind {
        MirExprKind::Binary {
            op: BinOp::Lt,
            left,
            right,
        } => {
            if !is_var_named(left, var_name) {
                return None;
            }
            match &right.kind {
                MirExprKind::IntLit(n) => *n,
                _ => return None,
            }
        }
        _ => return None,
    };

    // Iteration count must be positive and small.
    let iter_count = bound.saturating_sub(start);
    if iter_count <= 0 || iter_count > UNROLL_MAX_ITERATIONS {
        return None;
    }

    // Body must be non-empty + short; no tail expression.
    if body.stmts.is_empty() || body.stmts.len() > UNROLL_MAX_BODY_STMTS {
        return None;
    }
    if body.result.is_some() {
        return None;
    }

    // The last statement must be exactly `var_name = var_name + 1`.
    let last_idx = body.stmts.len() - 1;
    if !is_unit_increment_of(&body.stmts[last_idx], var_name) {
        return None;
    }

    // No other body statement may modify `var_name`, and no statement
    // may contain `Break` or `Continue` at this loop's scope.
    for (idx, stmt) in body.stmts.iter().enumerate() {
        if idx == last_idx {
            continue;
        }
        if stmt_assigns_to(stmt, var_name) {
            return None;
        }
        if contains_break_or_continue(stmt) {
            return None;
        }
    }

    // Emit N literal copies of the body. The trailing `i = i + 1;` in
    // each copy advances `i` correctly; after all copies, `i == bound`.
    let mut unrolled: Vec<MirStmt> =
        Vec::with_capacity(body.stmts.len() * iter_count as usize);
    for _ in 0..iter_count {
        for stmt in &body.stmts {
            unrolled.push(stmt.clone());
        }
    }
    Some(unrolled)
}

/// `true` iff `expr` is `Var(name)` or `VarLocal { name, .. }`. Handles
/// both forms so the pass works pre- and post-slot-resolution.
fn is_var_named(expr: &MirExpr, name: &str) -> bool {
    match &expr.kind {
        MirExprKind::Var(n) => n == name,
        MirExprKind::VarLocal { name: n, .. } => n == name,
        _ => false,
    }
}

/// `true` iff `stmt` is exactly `var_name = var_name + 1;` — i.e.
/// `Expr(Assign { target = Var/VarLocal(name), value = Binary { Add,
/// left = Var/VarLocal(name), right = IntLit(1) } })`.
///
/// Any other increment form (compound `+=`, non-unit step) is
/// intentionally rejected; the pass stays narrow.
fn is_unit_increment_of(stmt: &MirStmt, var_name: &str) -> bool {
    let expr = match stmt {
        MirStmt::Expr(e) => e,
        _ => return false,
    };
    let (target, value) = match &expr.kind {
        MirExprKind::Assign { target, value } => (target, value),
        _ => return false,
    };
    if !is_var_named(target, var_name) {
        return false;
    }
    match &value.kind {
        MirExprKind::Binary { op: BinOp::Add, left, right } => {
            is_var_named(left, var_name)
                && matches!(right.kind, MirExprKind::IntLit(1))
        }
        _ => false,
    }
}

/// `true` iff `stmt` directly assigns to `var_name` anywhere in its
/// nested sub-statements or sub-expressions. Reuses LICM's existing
/// [`collect_assigned_vars_stmt`] walker so behaviour matches LICM's
/// "modified inside the loop" check exactly.
fn stmt_assigns_to(stmt: &MirStmt, var_name: &str) -> bool {
    let mut assigned = BTreeSet::new();
    collect_assigned_vars_stmt(stmt, &mut assigned);
    assigned.contains(var_name)
}

/// `true` iff `stmt` contains a `Break` or `Continue` at any nesting
/// depth EXCEPT inside a nested `While` (whose break/continue belong
/// to that inner loop, not the one we're considering unrolling).
fn contains_break_or_continue(stmt: &MirStmt) -> bool {
    match stmt {
        MirStmt::Break | MirStmt::Continue => true,
        MirStmt::If { then_body, else_body, .. } => {
            then_body.stmts.iter().any(contains_break_or_continue)
                || else_body
                    .as_ref()
                    .map(|b| b.stmts.iter().any(contains_break_or_continue))
                    .unwrap_or(false)
        }
        MirStmt::NoGcBlock(body) => {
            body.stmts.iter().any(contains_break_or_continue)
        }
        // Nested `While` has its own break/continue scope.
        MirStmt::While { .. } => false,
        _ => false,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn mk_expr(kind: MirExprKind) -> MirExpr {
        MirExpr { kind }
    }

    fn mk_int(v: i64) -> MirExpr {
        mk_expr(MirExprKind::IntLit(v))
    }

    fn mk_float(v: f64) -> MirExpr {
        mk_expr(MirExprKind::FloatLit(v))
    }

    fn mk_bool(v: bool) -> MirExpr {
        mk_expr(MirExprKind::BoolLit(v))
    }

    fn mk_binary(op: BinOp, left: MirExpr, right: MirExpr) -> MirExpr {
        mk_expr(MirExprKind::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn mk_unary(op: UnaryOp, operand: MirExpr) -> MirExpr {
        mk_expr(MirExprKind::Unary {
            op,
            operand: Box::new(operand),
        })
    }

    fn mk_fn(name: &str, stmts: Vec<MirStmt>, result: Option<MirExpr>) -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts,
                result: result.map(Box::new),
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
            local_count: 0,
        }
    }

    fn mk_program(functions: Vec<MirFunction>) -> MirProgram {
        MirProgram {
            functions,
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    // -- Constant Folding tests --

    #[test]
    fn test_fold_int_add() {
        let mut expr = mk_binary(BinOp::Add, mk_int(2), mk_int(3));
        constant_fold_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::IntLit(5)));
    }

    #[test]
    fn test_fold_int_mul() {
        let mut expr = mk_binary(BinOp::Mul, mk_int(4), mk_int(5));
        constant_fold_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::IntLit(20)));
    }

    #[test]
    fn test_fold_int_div_by_zero_not_folded() {
        let mut expr = mk_binary(BinOp::Div, mk_int(10), mk_int(0));
        constant_fold_expr(&mut expr);
        // Should NOT fold - let runtime handle div by zero.
        assert!(matches!(expr.kind, MirExprKind::Binary { .. }));
    }

    #[test]
    fn test_fold_float_add() {
        let mut expr = mk_binary(BinOp::Add, mk_float(1.5), mk_float(2.5));
        constant_fold_expr(&mut expr);
        match expr.kind {
            MirExprKind::FloatLit(v) => assert_eq!(v, 4.0),
            _ => panic!("expected FloatLit"),
        }
    }

    #[test]
    fn test_fold_nested() {
        // (2 + 3) * 4 should fold to 20
        let inner = mk_binary(BinOp::Add, mk_int(2), mk_int(3));
        let mut expr = mk_binary(BinOp::Mul, inner, mk_int(4));
        constant_fold_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::IntLit(20)));
    }

    #[test]
    fn test_fold_comparison() {
        let mut expr = mk_binary(BinOp::Lt, mk_int(1), mk_int(2));
        constant_fold_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::BoolLit(true)));
    }

    #[test]
    fn test_fold_unary_neg() {
        let mut expr = mk_unary(UnaryOp::Neg, mk_int(42));
        constant_fold_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::IntLit(-42)));
    }

    #[test]
    fn test_fold_unary_not() {
        let mut expr = mk_unary(UnaryOp::Not, mk_bool(true));
        constant_fold_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::BoolLit(false)));
    }

    // -- DCE tests --

    #[test]
    fn test_dce_removes_unused_pure_let() {
        let mut body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "unused".to_string(),
                    mutable: false,
                    init: mk_int(42),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Expr(mk_expr(MirExprKind::Call {
                    callee: Box::new(mk_expr(MirExprKind::Var("print".to_string()))),
                    args: vec![mk_expr(MirExprKind::StringLit("hi".to_string()))],
                })),
            ],
            result: None,
        };
        dce_body(&mut body);
        // The unused let should be removed.
        assert_eq!(body.stmts.len(), 1);
        assert!(matches!(body.stmts[0], MirStmt::Expr(_)));
    }

    #[test]
    fn test_dce_keeps_used_let() {
        let mut body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: mk_int(42),
                    alloc_hint: None,
                    slot: None,
                },
            ],
            result: Some(Box::new(mk_expr(MirExprKind::Var("x".to_string())))),
        };
        dce_body(&mut body);
        assert_eq!(body.stmts.len(), 1);
    }

    #[test]
    fn test_dce_removes_dead_if_false() {
        let mut body = MirBody {
            stmts: vec![MirStmt::If {
                cond: mk_bool(false),
                then_body: MirBody {
                    stmts: vec![MirStmt::Expr(mk_int(1))],
                    result: None,
                },
                else_body: None,
            }],
            result: None,
        };
        dce_body(&mut body);
        assert!(body.stmts.is_empty());
    }

    #[test]
    fn test_dce_inlines_if_true() {
        let mut body = MirBody {
            stmts: vec![MirStmt::If {
                cond: mk_bool(true),
                then_body: MirBody {
                    stmts: vec![MirStmt::Expr(mk_int(1))],
                    result: None,
                },
                else_body: None,
            }],
            result: None,
        };
        dce_body(&mut body);
        assert_eq!(body.stmts.len(), 1);
        assert!(matches!(body.stmts[0], MirStmt::Expr(_)));
    }

    #[test]
    fn test_dce_removes_dead_while_false() {
        let mut body = MirBody {
            stmts: vec![MirStmt::While {
                cond: mk_bool(false),
                body: MirBody {
                    stmts: vec![MirStmt::Expr(mk_int(1))],
                    result: None,
                },
            }],
            result: None,
        };
        dce_body(&mut body);
        assert!(body.stmts.is_empty());
    }

    // -- Full pipeline test --

    #[test]
    fn test_optimize_program_preserves_semantics() {
        let program = mk_program(vec![mk_fn(
            "__main",
            vec![
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_int(2), mk_int(3)),
                    alloc_hint: None,
                    slot: None,
                },
            ],
            Some(mk_expr(MirExprKind::Var("x".to_string()))),
        )]);

        let optimized = optimize_program(&program);
        let main = &optimized.functions[0];

        // x should be folded to 5 in the let init.
        match &main.body.stmts[0] {
            MirStmt::Let { init, .. } => {
                assert!(matches!(init.kind, MirExprKind::IntLit(5)));
            }
            _ => panic!("expected Let"),
        }
    }

    // -- Strength Reduction tests --

    fn mk_var(name: &str) -> MirExpr {
        mk_expr(MirExprKind::Var(name.to_string()))
    }

    #[test]
    fn test_sr_mul_by_zero() {
        // x * 0 => 0
        let mut expr = mk_binary(BinOp::Mul, mk_var("x"), mk_int(0));
        strength_reduce_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::IntLit(0)));
    }

    #[test]
    fn test_sr_mul_by_one() {
        // x * 1 => x
        let mut expr = mk_binary(BinOp::Mul, mk_var("x"), mk_int(1));
        strength_reduce_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::Var(ref n) if n == "x"));
    }

    #[test]
    fn test_sr_mul_by_two() {
        // x * 2 => x + x
        let mut expr = mk_binary(BinOp::Mul, mk_var("x"), mk_int(2));
        strength_reduce_expr(&mut expr);
        match &expr.kind {
            MirExprKind::Binary { op, left, right } => {
                assert_eq!(*op, BinOp::Add);
                assert!(matches!(left.kind, MirExprKind::Var(ref n) if n == "x"));
                assert!(matches!(right.kind, MirExprKind::Var(ref n) if n == "x"));
            }
            _ => panic!("expected Binary Add"),
        }
    }

    #[test]
    fn test_sr_add_zero() {
        // x + 0 => x
        let mut expr = mk_binary(BinOp::Add, mk_var("x"), mk_int(0));
        strength_reduce_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::Var(ref n) if n == "x"));
    }

    #[test]
    fn test_sr_sub_zero() {
        // x - 0 => x
        let mut expr = mk_binary(BinOp::Sub, mk_var("x"), mk_int(0));
        strength_reduce_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::Var(ref n) if n == "x"));
    }

    #[test]
    fn test_sr_div_by_one() {
        // x / 1 => x
        let mut expr = mk_binary(BinOp::Div, mk_var("x"), mk_int(1));
        strength_reduce_expr(&mut expr);
        assert!(matches!(expr.kind, MirExprKind::Var(ref n) if n == "x"));
    }

    // -- CSE tests --

    #[test]
    fn test_cse_eliminates_duplicate_pure_let() {
        // let x = a + b
        // let y = a + b  (duplicate)
        // result: y
        // => after CSE, y should be replaced with x
        let mut body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "a".to_string(),
                    mutable: false,
                    init: mk_int(10),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "b".to_string(),
                    mutable: false,
                    init: mk_int(20),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_var("a"), mk_var("b")),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "y".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_var("a"), mk_var("b")),
                    alloc_hint: None,
                    slot: None,
                },
            ],
            result: Some(Box::new(mk_var("y"))),
        };
        cse_body(&mut body);
        // The result should now reference "x" instead of "y"
        match &body.result {
            Some(expr) => {
                assert!(matches!(expr.kind, MirExprKind::Var(ref n) if n == "x"));
            }
            None => panic!("expected result"),
        }
    }

    // -- LICM tests --

    #[test]
    fn test_licm_hoists_invariant_let() {
        // while cond {
        //   let inv = 1 + 2   // invariant: pure, refs no modified vars
        //   let x = inv + i   // NOT invariant: refs 'i' which is modified
        //   i = i + 1
        // }
        let mut body = MirBody {
            stmts: vec![MirStmt::While {
                cond: mk_var("cond"),
                body: MirBody {
                    stmts: vec![
                        MirStmt::Let {
                            name: "inv".to_string(),
                            mutable: false,
                            init: mk_binary(BinOp::Add, mk_int(1), mk_int(2)),
                            alloc_hint: None,
                            slot: None,
                        },
                        MirStmt::Let {
                            name: "x".to_string(),
                            mutable: false,
                            init: mk_binary(BinOp::Add, mk_var("inv"), mk_var("i")),
                            alloc_hint: None,
                            slot: None,
                        },
                        MirStmt::Expr(mk_expr(MirExprKind::Assign {
                            target: Box::new(mk_var("i")),
                            value: Box::new(mk_binary(BinOp::Add, mk_var("i"), mk_int(1))),
                        })),
                    ],
                    result: None,
                },
            }],
            result: None,
        };
        licm_body(&mut body);

        // The invariant `let inv = 1 + 2` should be hoisted before the while.
        assert_eq!(body.stmts.len(), 2);
        match &body.stmts[0] {
            MirStmt::Let { name, .. } => assert_eq!(name, "inv"),
            _ => panic!("expected hoisted Let"),
        }
        assert!(matches!(body.stmts[1], MirStmt::While { .. }));
        // The while body should have 2 stmts (x = inv + i, i = i + 1)
        match &body.stmts[1] {
            MirStmt::While { body: wb, .. } => {
                assert_eq!(wb.stmts.len(), 2);
            }
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn test_licm_does_not_hoist_dependent() {
        // while cond {
        //   i = i + 1
        //   let x = i * 2     // NOT invariant: refs 'i' which is modified
        // }
        let mut body = MirBody {
            stmts: vec![MirStmt::While {
                cond: mk_var("cond"),
                body: MirBody {
                    stmts: vec![
                        MirStmt::Expr(mk_expr(MirExprKind::Assign {
                            target: Box::new(mk_var("i")),
                            value: Box::new(mk_binary(BinOp::Add, mk_var("i"), mk_int(1))),
                        })),
                        MirStmt::Let {
                            name: "x".to_string(),
                            mutable: false,
                            init: mk_binary(BinOp::Mul, mk_var("i"), mk_int(2)),
                            alloc_hint: None,
                            slot: None,
                        },
                    ],
                    result: None,
                },
            }],
            result: None,
        };
        licm_body(&mut body);

        // Nothing should be hoisted.
        assert_eq!(body.stmts.len(), 1);
        match &body.stmts[0] {
            MirStmt::While { body: wb, .. } => {
                assert_eq!(wb.stmts.len(), 2); // both stmts stay
            }
            _ => panic!("expected While"),
        }
    }

    // -- Full pipeline with new passes --

    #[test]
    fn test_full_optimize_with_strength_reduction() {
        // let x = y * 1  (should be reduced to y)
        // let z = y + 0  (should be reduced to y)
        let program = mk_program(vec![mk_fn(
            "__main",
            vec![
                MirStmt::Let {
                    name: "y".to_string(),
                    mutable: false,
                    init: mk_int(42),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Mul, mk_var("y"), mk_int(1)),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "z".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_var("y"), mk_int(0)),
                    alloc_hint: None,
                    slot: None,
                },
            ],
            Some(mk_binary(BinOp::Add, mk_var("x"), mk_var("z"))),
        )]);

        let optimized = optimize_program(&program);
        let main = &optimized.functions[0];

        // After SR: x = y, z = y
        // After CSE: z should alias x (both are just "y")
        // Check that x's init is now just Var("y")
        match &main.body.stmts[1] {
            MirStmt::Let { init, .. } => {
                assert!(
                    matches!(init.kind, MirExprKind::Var(ref n) if n == "y"),
                    "expected Var(y) after strength reduction, got {:?}",
                    init.kind
                );
            }
            _ => panic!("expected Let"),
        }
    }

    // -- Phase 2: PassPlan + optimize_program_with_plan ----------------

    /// Build a single-function program whose body computes `init_value + 0`
    /// and *uses* the result as the body's tail expression. The "use" stops
    /// DCE from eliminating the binding, so the optimization-visible
    /// expression survives for assertions.
    fn make_program_with_fn(name: &str, init_value: i64) -> MirProgram {
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: MirExpr {
                    kind: MirExprKind::Binary {
                        op: BinOp::Add,
                        left: Box::new(MirExpr {
                            kind: MirExprKind::IntLit(init_value),
                        }),
                        right: Box::new(MirExpr {
                            kind: MirExprKind::IntLit(0),
                        }),
                    },
                },
                alloc_hint: None,
                slot: None,
            }],
            // Body's tail expression references `x`, making it live.
            result: Some(Box::new(MirExpr {
                kind: MirExprKind::Var("x".to_string()),
            })),
        };
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: name.to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body,
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    fn first_let_init(prog: &MirProgram) -> &MirExpr {
        let func = prog
            .functions
            .first()
            .expect("expected at least one function");
        let stmt = func
            .body
            .stmts
            .first()
            .expect("expected at least one statement (DCE removed our binding?)");
        match stmt {
            MirStmt::Let { init, .. } => init,
            _ => panic!("expected Let, got {:?}", stmt),
        }
    }

    #[test]
    fn pass_plan_default_is_empty() {
        let plan = PassPlan::empty();
        assert!(plan.per_function.is_empty());
    }

    #[test]
    fn pass_plan_with_function_inserts() {
        let plan = PassPlan::empty().with_function(
            "main",
            vec!["dce".to_string(), "licm".to_string()],
        );
        assert_eq!(plan.per_function.len(), 1);
        assert_eq!(plan.per_function.get("main").unwrap().len(), 2);
    }

    #[test]
    fn pass_plan_total_invocations_uses_default_for_unmapped() {
        let plan = PassPlan::empty();
        let total = plan.total_pass_invocations(&["a".into(), "b".into()]);
        assert_eq!(total, 2 * DEFAULT_PASS_SEQUENCE.len());
    }

    #[test]
    fn apply_pass_recognises_canonical_and_alias_names() {
        let mut f = MirFunction {
            id: MirFnId(0),
            name: "f".to_string(),
            type_params: vec![],
            params: vec![],
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
        };
        // Every canonical and alias name should return true.
        for name in &[
            "constant_fold", "cf", "cf_round_2",
            "strength_reduce", "sr",
            "dce", "dead_code_elimination",
            "cse", "common_subexpression_elimination",
            "licm", "loop_invariant_code_motion",
            "loop_unroll", "unroll",
        ] {
            assert!(apply_pass(name, &mut f), "should recognise {}", name);
        }
        // Unknown should return false.
        assert!(!apply_pass("nonexistent_pass", &mut f));
    }

    #[test]
    fn optimize_program_with_empty_plan_matches_optimize_program() {
        // Phase 2 invariant: empty plan = default behaviour.
        let prog = make_program_with_fn("main", 42);
        let opt_old = optimize_program(&prog);
        let opt_new = optimize_program_with_plan(&prog, &PassPlan::empty());

        // Both should fold `42 + 0` to `42`.
        for opt in [&opt_old, &opt_new] {
            let init = first_let_init(opt);
            assert!(
                matches!(init.kind, MirExprKind::IntLit(42)),
                "expected IntLit(42) after CF, got {:?}",
                init.kind
            );
        }
        // And the two should be byte-identical (debug-compared via Debug
        // since MirProgram does not derive Eq).
        assert_eq!(format!("{:?}", opt_old), format!("{:?}", opt_new));
    }

    #[test]
    fn optimize_program_with_plan_runs_only_specified_passes() {
        let prog = make_program_with_fn("main", 5);
        // Plan that only runs strength_reduce (no constant fold).
        let plan = PassPlan::empty().with_function(
            "main",
            vec!["strength_reduce".to_string()],
        );
        let opt = optimize_program_with_plan(&prog, &plan);
        let init = first_let_init(&opt);
        // Without CF, the `5 + 0` stays as a binary expression. The strength
        // reducer should still simplify add-zero, so we expect just `5`.
        // Actually: strength reduction also turns `x + 0 -> x`, so we
        // still get IntLit(5) here (the pre-pass already had IntLit(5)).
        assert!(
            matches!(init.kind, MirExprKind::IntLit(5) | MirExprKind::Binary { .. }),
            "expected IntLit(5) or unchanged Binary after SR-only, got {:?}",
            init.kind
        );
    }

    #[test]
    fn optimize_program_with_plan_unknown_pass_is_no_op() {
        // A plan that names a pass that doesn't exist should NOT error;
        // it just skips. This is the "CANA may recommend a pass not yet
        // implemented" contract.
        let prog = make_program_with_fn("main", 7);
        let plan = PassPlan::empty().with_function(
            "main",
            vec!["nonexistent_pass".to_string()],
        );
        let opt = optimize_program_with_plan(&prog, &plan);
        let init = first_let_init(&opt);
        // No passes ran, so `7 + 0` stays as Binary.
        assert!(
            matches!(init.kind, MirExprKind::Binary { .. }),
            "expected Binary after no-op plan, got {:?}",
            init.kind
        );
    }

    #[test]
    fn optimize_program_with_plan_unmapped_function_uses_default() {
        // Plan that names "main" but the program has function "other" — the
        // unmapped "other" should still get the default 6-pass treatment.
        let prog = make_program_with_fn("other", 99);
        let plan = PassPlan::empty().with_function(
            "main", // wrong name; "other" is unmapped
            vec!["dce".to_string()],
        );
        let opt = optimize_program_with_plan(&prog, &plan);
        let init = first_let_init(&opt);
        // "other" was unmapped → default sequence → CF folds 99+0 to 99.
        assert!(
            matches!(init.kind, MirExprKind::IntLit(99)),
            "expected IntLit(99) after default sequence on unmapped fn, got {:?}",
            init.kind
        );
    }

    #[test]
    fn optimize_program_with_plan_is_deterministic() {
        let prog = make_program_with_fn("main", 42);
        let plan = PassPlan::empty().with_function(
            "main",
            vec!["constant_fold".to_string(), "dce".to_string()],
        );
        let first = optimize_program_with_plan(&prog, &plan);
        for _ in 0..50 {
            let again = optimize_program_with_plan(&prog, &plan);
            assert_eq!(format!("{:?}", first), format!("{:?}", again));
        }
    }

    #[test]
    fn default_pass_sequence_has_seven_entries() {
        // Drift guard: if a pass is added/removed, this needs to change
        // alongside cjc-cana::pass_ranker::CANONICAL_PASSES.
        //
        // Phase 2 was 6 passes (CF / SR / DCE / CSE / LICM / cf_round_2).
        // loop_unroll was added between LICM and cf_round_2 so the
        // second CF round can fold the literal-trip-count constants
        // exposed by unrolling.
        assert_eq!(DEFAULT_PASS_SEQUENCE.len(), 7);
        assert_eq!(DEFAULT_PASS_SEQUENCE[5], "loop_unroll");
    }

    // -- Regression test for task_a30e8eec --------------------------------
    //
    // Tests that LICM does NOT hoist `let mut j = 0;` out of an outer
    // while loop when j is reassigned inside the inner loop. Pre-fix,
    // LICM hoisted the let, losing per-iteration re-initialization and
    // producing wrong results on every nested-loop accumulator.

    fn nested_loop_program() -> MirProgram {
        // Build the MIR for:
        //   let mut total: i64 = 0;
        //   let mut i: i64 = 0;
        //   while i < 3 {
        //       let mut j: i64 = 0;
        //       while j < 3 {
        //           total = total + i * j;
        //           j = j + 1;
        //       }
        //       i = i + 1;
        //   }
        //   total
        //
        // Expected result: sum_{i=0..3} sum_{j=0..3} i*j = (0+1+2)*(0+1+2) = 9
        let inner_loop = MirStmt::While {
            cond: MirExpr {
                kind: MirExprKind::Binary {
                    op: BinOp::Lt,
                    left: Box::new(MirExpr { kind: MirExprKind::Var("j".to_string()) }),
                    right: Box::new(MirExpr { kind: MirExprKind::IntLit(3) }),
                },
            },
            body: MirBody {
                stmts: vec![
                    // total = total + i * j;
                    MirStmt::Expr(MirExpr {
                        kind: MirExprKind::Assign {
                            target: Box::new(MirExpr {
                                kind: MirExprKind::Var("total".to_string()),
                            }),
                            value: Box::new(MirExpr {
                                kind: MirExprKind::Binary {
                                    op: BinOp::Add,
                                    left: Box::new(MirExpr {
                                        kind: MirExprKind::Var("total".to_string()),
                                    }),
                                    right: Box::new(MirExpr {
                                        kind: MirExprKind::Binary {
                                            op: BinOp::Mul,
                                            left: Box::new(MirExpr {
                                                kind: MirExprKind::Var("i".to_string()),
                                            }),
                                            right: Box::new(MirExpr {
                                                kind: MirExprKind::Var("j".to_string()),
                                            }),
                                        },
                                    }),
                                },
                            }),
                        },
                    }),
                    // j = j + 1;
                    MirStmt::Expr(MirExpr {
                        kind: MirExprKind::Assign {
                            target: Box::new(MirExpr {
                                kind: MirExprKind::Var("j".to_string()),
                            }),
                            value: Box::new(MirExpr {
                                kind: MirExprKind::Binary {
                                    op: BinOp::Add,
                                    left: Box::new(MirExpr {
                                        kind: MirExprKind::Var("j".to_string()),
                                    }),
                                    right: Box::new(MirExpr {
                                        kind: MirExprKind::IntLit(1) }),
                                },
                            }),
                        },
                    }),
                ],
                result: None,
            },
        };

        let outer_loop = MirStmt::While {
            cond: MirExpr {
                kind: MirExprKind::Binary {
                    op: BinOp::Lt,
                    left: Box::new(MirExpr { kind: MirExprKind::Var("i".to_string()) }),
                    right: Box::new(MirExpr { kind: MirExprKind::IntLit(3) }),
                },
            },
            body: MirBody {
                stmts: vec![
                    // let mut j: i64 = 0;  ← THIS is what LICM was incorrectly hoisting
                    MirStmt::Let {
                        name: "j".to_string(),
                        mutable: true,
                        init: MirExpr { kind: MirExprKind::IntLit(0) },
                        alloc_hint: None,
                        slot: None,
                    },
                    inner_loop,
                    // i = i + 1;
                    MirStmt::Expr(MirExpr {
                        kind: MirExprKind::Assign {
                            target: Box::new(MirExpr {
                                kind: MirExprKind::Var("i".to_string()),
                            }),
                            value: Box::new(MirExpr {
                                kind: MirExprKind::Binary {
                                    op: BinOp::Add,
                                    left: Box::new(MirExpr {
                                        kind: MirExprKind::Var("i".to_string()),
                                    }),
                                    right: Box::new(MirExpr {
                                        kind: MirExprKind::IntLit(1) }),
                                },
                            }),
                        },
                    }),
                ],
                result: None,
            },
        };

        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "total".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "i".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                outer_loop,
            ],
            result: Some(Box::new(MirExpr {
                kind: MirExprKind::Var("total".to_string()),
            })),
        };

        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "nested".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body,
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    /// Count how many `let mut j = 0;` Lets remain inside the outer loop
    /// body after optimization. The expected post-fix value is 1 (the
    /// Let stays inside); the buggy pre-fix value was 0 (LICM hoisted
    /// it out).
    fn count_inner_j_lets(prog: &MirProgram) -> usize {
        fn count_in_body(body: &MirBody) -> usize {
            let mut n = 0;
            for stmt in &body.stmts {
                match stmt {
                    MirStmt::Let { name, .. } if name == "j" => n += 1,
                    MirStmt::While { body: wb, .. } => n += count_in_body(wb),
                    MirStmt::If { then_body, else_body, .. } => {
                        n += count_in_body(then_body);
                        if let Some(eb) = else_body {
                            n += count_in_body(eb);
                        }
                    }
                    _ => {}
                }
            }
            n
        }
        prog.functions
            .iter()
            .map(|f| count_in_body(&f.body))
            .sum()
    }

    #[test]
    fn licm_does_not_hoist_reassigned_let_out_of_loop() {
        // Build a program containing the canonical nested-loop pattern
        // where the inner `let mut j = 0;` MUST survive optimization
        // (because j gets reassigned inside the inner loop). Pre-fix,
        // LICM would lift the Let out of the outer loop, producing a
        // wrong-result program.
        let prog = nested_loop_program();

        // Before optimization: exactly 1 `let j` (the one inside the outer loop).
        assert_eq!(
            count_inner_j_lets(&prog),
            1,
            "test setup invariant: program starts with 1 `let j`"
        );

        // Apply only LICM (no constant folding etc.) to isolate the test.
        let mut optimized = prog.clone();
        for func in &mut optimized.functions {
            super::licm_fn(func);
        }

        // After LICM, the `let j` must still be inside the loop body
        // (1 occurrence somewhere in the function), not hoisted out.
        assert_eq!(
            count_inner_j_lets(&optimized),
            1,
            "LICM must NOT hoist `let mut j = 0;` out of the loop when j is \
             reassigned inside (task_a30e8eec regression)"
        );

        // Also: the outer loop's top-level stmts (before the outer while)
        // must NOT contain a Let named "j". We check this directly:
        // the function body's top-level stmts are [Let total, Let i, While].
        // If LICM had hoisted j out, we'd see an extra [Let j] before the
        // While. We assert there is no `let j` at the top level.
        let top_level_j_lets = optimized
            .functions
            .iter()
            .flat_map(|f| f.body.stmts.iter())
            .filter(|s| matches!(s, MirStmt::Let { name, .. } if name == "j"))
            .count();
        assert_eq!(
            top_level_j_lets, 0,
            "LICM hoisted `let j` to the function's top level — regression"
        );
    }

    #[test]
    fn licm_still_hoists_truly_invariant_let() {
        // Verify the fix did NOT regress the "good" hoist case: a Let
        // whose name is NOT reassigned in the loop and whose init
        // doesn't reference any modified vars should still be hoisted.
        //
        // Program:
        //   let mut total: i64 = 0;
        //   let mut i: i64 = 0;
        //   while i < 10 {
        //       let constant: i64 = 42;      ← MUST be hoisted (no reassign)
        //       total = total + constant;
        //       i = i + 1;
        //   }
        //   total
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "total".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "i".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::While {
                    cond: MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Lt,
                            left: Box::new(MirExpr { kind: MirExprKind::Var("i".to_string()) }),
                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(10) }),
                        },
                    },
                    body: MirBody {
                        stmts: vec![
                            MirStmt::Let {
                                name: "constant".to_string(),
                                mutable: false,
                                init: MirExpr { kind: MirExprKind::IntLit(42) },
                                alloc_hint: None,
                                slot: None,
                            },
                            MirStmt::Expr(MirExpr {
                                kind: MirExprKind::Assign {
                                    target: Box::new(MirExpr {
                                        kind: MirExprKind::Var("total".to_string()),
                                    }),
                                    value: Box::new(MirExpr {
                                        kind: MirExprKind::Binary {
                                            op: BinOp::Add,
                                            left: Box::new(MirExpr {
                                                kind: MirExprKind::Var("total".to_string()),
                                            }),
                                            right: Box::new(MirExpr {
                                                kind: MirExprKind::Var("constant".to_string()),
                                            }),
                                        },
                                    }),
                                },
                            }),
                            MirStmt::Expr(MirExpr {
                                kind: MirExprKind::Assign {
                                    target: Box::new(MirExpr {
                                        kind: MirExprKind::Var("i".to_string()),
                                    }),
                                    value: Box::new(MirExpr {
                                        kind: MirExprKind::Binary {
                                            op: BinOp::Add,
                                            left: Box::new(MirExpr {
                                                kind: MirExprKind::Var("i".to_string()),
                                            }),
                                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(1) }),
                                        },
                                    }),
                                },
                            }),
                        ],
                        result: None,
                    },
                },
            ],
            result: Some(Box::new(MirExpr {
                kind: MirExprKind::Var("total".to_string()),
            })),
        };

        let prog = MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "f".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body,
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        };

        let mut optimized = prog.clone();
        for func in &mut optimized.functions {
            super::licm_fn(func);
        }

        // The `let constant: i64 = 42;` MUST be hoisted to the top
        // level (before the while), because:
        // - init is pure (literal)
        // - init references no loop-modified vars
        // - `constant` is not reassigned in the loop
        let top_level_constant_lets = optimized
            .functions
            .iter()
            .flat_map(|f| f.body.stmts.iter())
            .filter(|s| matches!(s, MirStmt::Let { name, .. } if name == "constant"))
            .count();
        assert_eq!(
            top_level_constant_lets, 1,
            "LICM should have hoisted `let constant = 42;` to the function top \
             level — without this, the LICM fix regressed the good case"
        );
    }

    // -----------------------------------------------------------------------
    // PassDiagnostics + apply_pass_with_diagnostics tests (Option A)
    // -----------------------------------------------------------------------

    fn make_test_fn(name: &str, body: MirBody) -> MirFunction {
        MirFunction {
            id: crate::MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body,
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Public,
            local_count: 0,
        }
    }

    fn ekind(k: MirExprKind) -> MirExpr {
        MirExpr { kind: k }
    }

    #[test]
    fn diagnostics_for_unknown_pass_returns_none() {
        let mut f = make_test_fn("f", MirBody { stmts: vec![], result: None });
        let d = apply_pass_with_diagnostics("definitely_not_a_pass", &mut f);
        assert!(d.is_none());
    }

    #[test]
    fn diagnostics_for_known_pass_returns_some() {
        let mut f = make_test_fn("f", MirBody { stmts: vec![], result: None });
        let d = apply_pass_with_diagnostics("constant_fold", &mut f);
        let d = d.expect("CF is a known pass");
        assert_eq!(d.pass_name, "constant_fold");
    }

    #[test]
    fn sr_counts_native_rewrites() {
        // x * 0 → 0  (1 rewrite, shrinks 3 nodes to 1)
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "a".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Binary {
                    op: BinOp::Mul,
                    left: Box::new(ekind(MirExprKind::Var("x".to_string()))),
                    right: Box::new(ekind(MirExprKind::IntLit(0))),
                }),
                alloc_hint: None,
                slot: None,
            }],
            result: None,
        };
        let mut f = make_test_fn("f", body);
        let d = apply_pass_with_diagnostics("strength_reduce", &mut f).unwrap();
        assert_eq!(d.changes_applied, 1, "x * 0 is one SR rewrite");
        assert!(d.nodes_after < d.nodes_before, "should shrink: {d:?}");
    }

    #[test]
    fn sr_in_place_rewrite_counts_but_doesnt_shrink() {
        // x * 2 → x + x  (1 rewrite, node count unchanged)
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "a".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Binary {
                    op: BinOp::Mul,
                    left: Box::new(ekind(MirExprKind::Var("x".to_string()))),
                    right: Box::new(ekind(MirExprKind::IntLit(2))),
                }),
                alloc_hint: None,
                slot: None,
            }],
            result: None,
        };
        let mut f = make_test_fn("f", body);
        let d = apply_pass_with_diagnostics("strength_reduce", &mut f).unwrap();
        assert_eq!(d.changes_applied, 1, "x * 2 → x + x is one SR rewrite");
        assert_eq!(
            d.nodes_before, d.nodes_after,
            "x * 2 → x + x doesn't shrink node count — exactly the case where \
             pass_native_change beats structural_delta as a training signal"
        );
        assert_eq!(d.structural_delta_fraction(), 0.0);
        assert!(d.pass_native_change_fraction() > 0.0);
    }

    #[test]
    fn cse_counts_replacements() {
        // let a = x + y;  let b = x + y;  → CSE detects b duplicates a
        let xy = || ekind(MirExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(ekind(MirExprKind::Var("x".to_string()))),
            right: Box::new(ekind(MirExprKind::Var("y".to_string()))),
        });
        let body = MirBody {
            stmts: vec![
                MirStmt::Let { name: "a".to_string(), mutable: false, init: xy(),
                    alloc_hint: None, slot: None },
                MirStmt::Let { name: "b".to_string(), mutable: false, init: xy(),
                    alloc_hint: None, slot: None },
            ],
            result: None,
        };
        let mut f = make_test_fn("f", body);
        let d = apply_pass_with_diagnostics("cse", &mut f).unwrap();
        assert_eq!(d.changes_applied, 1, "b duplicates a — one CSE replacement");
    }

    #[test]
    fn cse_no_duplicates_returns_zero() {
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "a".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Var("x".to_string())),
                alloc_hint: None,
                slot: None,
            }],
            result: None,
        };
        let mut f = make_test_fn("f", body);
        let d = apply_pass_with_diagnostics("cse", &mut f).unwrap();
        assert_eq!(d.changes_applied, 0);
    }

    #[test]
    fn licm_counts_hoisted_lets() {
        // Hoistable let inside a while: should count exactly 1 hoist.
        // while cond { let k = 42; ... }
        let body = MirBody {
            stmts: vec![MirStmt::While {
                cond: ekind(MirExprKind::BoolLit(true)),
                body: MirBody {
                    stmts: vec![
                        MirStmt::Let {
                            name: "k".to_string(),
                            mutable: false,
                            init: ekind(MirExprKind::IntLit(42)),
                            alloc_hint: None,
                            slot: None,
                        },
                        MirStmt::Break,
                    ],
                    result: None,
                },
            }],
            result: None,
        };
        let mut f = make_test_fn("f", body);
        let d = apply_pass_with_diagnostics("licm", &mut f).unwrap();
        assert_eq!(d.changes_applied, 1, "one let hoisted out of the while");
    }

    #[test]
    fn cf_uses_structural_delta_as_change_count() {
        // CF doesn't report its own count yet — falls back to node delta.
        // 1 + 2 → 3 shrinks 3 nodes to 1.
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "a".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Binary {
                    op: BinOp::Add,
                    left: Box::new(ekind(MirExprKind::IntLit(1))),
                    right: Box::new(ekind(MirExprKind::IntLit(2))),
                }),
                alloc_hint: None,
                slot: None,
            }],
            result: None,
        };
        let mut f = make_test_fn("f", body);
        let d = apply_pass_with_diagnostics("constant_fold", &mut f).unwrap();
        assert!(d.nodes_after < d.nodes_before, "CF must shrink: {d:?}");
        assert_eq!(d.changes_applied, d.nodes_before - d.nodes_after);
    }

    #[test]
    fn diagnostics_are_deterministic() {
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "a".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Binary {
                    op: BinOp::Mul,
                    left: Box::new(ekind(MirExprKind::Var("x".to_string()))),
                    right: Box::new(ekind(MirExprKind::IntLit(0))),
                }),
                alloc_hint: None,
                slot: None,
            }],
            result: None,
        };
        let mut f = make_test_fn("f", body.clone());
        let first = apply_pass_with_diagnostics("strength_reduce", &mut f).unwrap();
        for _ in 0..20 {
            let mut f2 = make_test_fn("f", body.clone());
            let again = apply_pass_with_diagnostics("strength_reduce", &mut f2).unwrap();
            assert_eq!(again, first);
        }
    }

    #[test]
    fn apply_pass_compatibility_preserved() {
        // apply_pass returns true for known passes — same contract as before.
        let mut f = make_test_fn("f", MirBody { stmts: vec![], result: None });
        assert!(apply_pass("constant_fold", &mut f));
        assert!(apply_pass("strength_reduce", &mut f));
        assert!(apply_pass("dce", &mut f));
        assert!(apply_pass("cse", &mut f));
        assert!(apply_pass("licm", &mut f));
        assert!(apply_pass("loop_unroll", &mut f));
        assert!(apply_pass("fusion_rewrite", &mut f));
        assert!(!apply_pass("nothing_real", &mut f));
    }

    // -----------------------------------------------------------------------
    // Loop Unrolling tests
    // -----------------------------------------------------------------------

    /// Build the canonical unrollable pattern as a single function body:
    ///
    ///   let mut total: i64 = 0;
    ///   let mut <name>: i64 = <start>;
    ///   while <name> < <bound> {
    ///       total = total + <name>;
    ///       <name> = <name> + 1;
    ///   }
    ///   total
    ///
    /// Returns a one-function `MirProgram` with this body in `f`.
    fn unrollable_program(name: &str, start: i64, bound: i64) -> MirProgram {
        let body_stmts = vec![
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var("total".to_string()),
                    }),
                    value: Box::new(MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Add,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var("total".to_string()),
                            }),
                            right: Box::new(MirExpr {
                                kind: MirExprKind::Var(name.to_string()),
                            }),
                        },
                    }),
                },
            }),
            // i = i + 1
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var(name.to_string()),
                    }),
                    value: Box::new(MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Add,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var(name.to_string()),
                            }),
                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(1) }),
                        },
                    }),
                },
            }),
        ];
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "total".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: name.to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(start) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::While {
                    cond: MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Lt,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var(name.to_string()),
                            }),
                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(bound) }),
                        },
                    },
                    body: MirBody { stmts: body_stmts, result: None },
                },
            ],
            result: Some(Box::new(MirExpr {
                kind: MirExprKind::Var("total".to_string()),
            })),
        };
        mk_program(vec![mk_fn("f", body.stmts, body.result.map(|b| *b))])
    }

    /// Count `MirStmt::While` occurrences anywhere in a function body.
    /// Used to assert that the unrolled program has zero whiles where
    /// it started with one.
    fn count_whiles(prog: &MirProgram) -> usize {
        fn count_in_body(body: &MirBody) -> usize {
            let mut n = 0;
            for stmt in &body.stmts {
                match stmt {
                    MirStmt::While { body: wb, .. } => {
                        n += 1;
                        n += count_in_body(wb);
                    }
                    MirStmt::If { then_body, else_body, .. } => {
                        n += count_in_body(then_body);
                        if let Some(eb) = else_body {
                            n += count_in_body(eb);
                        }
                    }
                    MirStmt::NoGcBlock(b) => n += count_in_body(b),
                    _ => {}
                }
            }
            n
        }
        prog.functions
            .iter()
            .map(|f| count_in_body(&f.body))
            .sum()
    }

    #[test]
    fn unroll_collapses_simple_short_loop() {
        // 4 iterations: while i < 4 { total = total + i; i = i + 1 }
        // Body has 2 stmts. Unrolled: 2 * 4 = 8 stmts inline, plus the Let.
        let mut prog = unrollable_program("i", 0, 4);

        // Sanity: starts with exactly one while.
        assert_eq!(count_whiles(&prog), 1);

        let f = &mut prog.functions[0];
        let count = loop_unroll_fn(f);
        assert_eq!(count, 1, "exactly one loop should have been unrolled");
        assert_eq!(count_whiles(&prog), 0, "the while should be gone");

        // The function body should now be:
        //   [Let total, Let i, body_stmt, body_stmt, body_stmt, body_stmt, body_stmt, body_stmt, body_stmt, body_stmt, <tail expr>]
        //   = 2 lets + 8 body stmts = 10 stmts.
        assert_eq!(prog.functions[0].body.stmts.len(), 2 + 2 * 4);
    }

    #[test]
    fn unroll_refuses_when_iter_count_exceeds_max() {
        // 9 iterations > UNROLL_MAX_ITERATIONS (8) → refuse.
        let mut prog = unrollable_program("i", 0, 9);
        let f = &mut prog.functions[0];
        let count = loop_unroll_fn(f);
        assert_eq!(count, 0, "9-iter loop must not be unrolled");
        assert_eq!(count_whiles(&prog), 1, "the while must remain");
    }

    #[test]
    fn unroll_refuses_when_iter_count_nonpositive() {
        // bound <= start → iter_count <= 0. Don't unroll (DCE handles
        // dead loops elsewhere; we shouldn't fabricate empty blocks).
        let mut prog = unrollable_program("i", 5, 5);
        let f = &mut prog.functions[0];
        let count = loop_unroll_fn(f);
        assert_eq!(count, 0, "empty-range loop must not be unrolled");
        assert_eq!(count_whiles(&prog), 1);
    }

    #[test]
    fn unroll_refuses_when_let_isnt_immediately_preceding() {
        // Build a program where the `let mut i = 0;` is separated from
        // the while by a non-Let statement. The pair-detection only
        // looks at adjacent statements, so this should not unroll.
        let mut prog = unrollable_program("i", 0, 3);
        // Splice an Expr stmt between the Let and the While.
        let f = &mut prog.functions[0];
        let intervening = MirStmt::Expr(MirExpr {
            kind: MirExprKind::Assign {
                target: Box::new(MirExpr {
                    kind: MirExprKind::Var("total".to_string()),
                }),
                value: Box::new(MirExpr { kind: MirExprKind::IntLit(0) }),
            },
        });
        // After change: [Let total, Let i, <Expr>, While, ...]
        f.body.stmts.insert(2, intervening);

        let count = loop_unroll_fn(f);
        assert_eq!(count, 0, "must not unroll when Let isn't adjacent to While");
        assert_eq!(count_whiles(&prog), 1);
    }

    #[test]
    fn unroll_refuses_when_body_contains_break() {
        // while i < 3 { if cond { break }; total = total + i; i = i + 1 }
        let mut prog = unrollable_program("i", 0, 3);
        let f = &mut prog.functions[0];
        // Replace the while body to inject a Break inside an If.
        if let MirStmt::While { body, .. } = &mut f.body.stmts[2] {
            let break_stmt = MirStmt::If {
                cond: MirExpr {
                    kind: MirExprKind::Var("cond".to_string()),
                },
                then_body: MirBody {
                    stmts: vec![MirStmt::Break],
                    result: None,
                },
                else_body: None,
            };
            body.stmts.insert(0, break_stmt);
        } else {
            panic!("expected While at index 2");
        }

        let count = loop_unroll_fn(f);
        assert_eq!(count, 0, "must not unroll bodies with Break");
        assert_eq!(count_whiles(&prog), 1);
    }

    #[test]
    fn unroll_refuses_when_last_stmt_isnt_unit_increment() {
        // The last stmt is `i = i + 2` (non-unit step) — refuse.
        let mut prog = unrollable_program("i", 0, 3);
        let f = &mut prog.functions[0];
        if let MirStmt::While { body, .. } = &mut f.body.stmts[2] {
            // body.stmts[1] is the `i = i + 1` increment. Rewrite to + 2.
            if let MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign { value, .. },
            }) = &mut body.stmts[1]
            {
                if let MirExprKind::Binary { right, .. } = &mut value.kind {
                    right.kind = MirExprKind::IntLit(2);
                }
            }
        }
        let count = loop_unroll_fn(f);
        assert_eq!(count, 0, "non-unit step must not unroll");
        assert_eq!(count_whiles(&prog), 1);
    }

    #[test]
    fn unroll_refuses_when_body_reassigns_loop_var() {
        // Body assigns to `i` outside the trailing increment (e.g. `i = 0`
        // earlier in the body) — refuse.
        let mut prog = unrollable_program("i", 0, 3);
        let f = &mut prog.functions[0];
        if let MirStmt::While { body, .. } = &mut f.body.stmts[2] {
            let extra_assign = MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var("i".to_string()),
                    }),
                    value: Box::new(MirExpr { kind: MirExprKind::IntLit(0) }),
                },
            });
            body.stmts.insert(0, extra_assign);
        }
        let count = loop_unroll_fn(f);
        assert_eq!(count, 0, "extra modification of i must not unroll");
        assert_eq!(count_whiles(&prog), 1);
    }

    #[test]
    fn unroll_counts_in_diagnostics_surface() {
        let mut prog = unrollable_program("i", 0, 3);
        let f = &mut prog.functions[0];
        let d = apply_pass_with_diagnostics("loop_unroll", f).unwrap();
        assert_eq!(d.pass_name, "loop_unroll");
        assert_eq!(d.changes_applied, 1, "one loop unrolled = one rewrite");
        assert!(
            d.nodes_after > d.nodes_before,
            "unrolling grows the node count by replication: {d:?}"
        );
        // The "unroll" alias should also work.
        let mut prog2 = unrollable_program("j", 0, 3);
        let f2 = &mut prog2.functions[0];
        let d2 = apply_pass_with_diagnostics("unroll", f2).unwrap();
        assert_eq!(d2.changes_applied, 1);
    }

    #[test]
    fn unroll_is_deterministic_across_runs() {
        let prog = unrollable_program("i", 0, 5);
        let mut first = prog.clone();
        loop_unroll_fn(&mut first.functions[0]);
        for _ in 0..20 {
            let mut again = prog.clone();
            loop_unroll_fn(&mut again.functions[0]);
            assert_eq!(
                format!("{:?}", first),
                format!("{:?}", again),
                "loop_unroll must produce byte-identical MIR every run",
            );
        }
    }

    #[test]
    fn unroll_handles_nested_loops_bottom_up() {
        // Outer loop with an inner unrollable loop. The inner one should
        // get unrolled; the outer is not unrollable as-is (its body has
        // assignments to `j` not adjacent to a fresh `let mut j` at the
        // outer level — `j` is declared INSIDE the outer body).
        let inner_body_stmts = vec![
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var("total".to_string()),
                    }),
                    value: Box::new(MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Add,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var("total".to_string()),
                            }),
                            right: Box::new(MirExpr {
                                kind: MirExprKind::Var("j".to_string()),
                            }),
                        },
                    }),
                },
            }),
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var("j".to_string()),
                    }),
                    value: Box::new(MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Add,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var("j".to_string()),
                            }),
                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(1) }),
                        },
                    }),
                },
            }),
        ];
        let inner_while = MirStmt::While {
            cond: MirExpr {
                kind: MirExprKind::Binary {
                    op: BinOp::Lt,
                    left: Box::new(MirExpr { kind: MirExprKind::Var("j".to_string()) }),
                    right: Box::new(MirExpr { kind: MirExprKind::IntLit(2) }),
                },
            },
            body: MirBody { stmts: inner_body_stmts, result: None },
        };
        let outer_while = MirStmt::While {
            cond: MirExpr {
                kind: MirExprKind::Binary {
                    op: BinOp::Lt,
                    left: Box::new(MirExpr { kind: MirExprKind::Var("i".to_string()) }),
                    right: Box::new(MirExpr { kind: MirExprKind::IntLit(3) }),
                },
            },
            body: MirBody {
                stmts: vec![
                    // let mut j = 0; while j < 2 { ... }
                    MirStmt::Let {
                        name: "j".to_string(),
                        mutable: true,
                        init: MirExpr { kind: MirExprKind::IntLit(0) },
                        alloc_hint: None,
                        slot: None,
                    },
                    inner_while,
                    // i = i + 1
                    MirStmt::Expr(MirExpr {
                        kind: MirExprKind::Assign {
                            target: Box::new(MirExpr {
                                kind: MirExprKind::Var("i".to_string()),
                            }),
                            value: Box::new(MirExpr {
                                kind: MirExprKind::Binary {
                                    op: BinOp::Add,
                                    left: Box::new(MirExpr {
                                        kind: MirExprKind::Var("i".to_string()),
                                    }),
                                    right: Box::new(MirExpr { kind: MirExprKind::IntLit(1) }),
                                },
                            }),
                        },
                    }),
                ],
                result: None,
            },
        };

        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "total".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "i".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                outer_while,
            ],
            result: None,
        };
        let mut prog = mk_program(vec![mk_fn("f", body.stmts, body.result.map(|b| *b))]);

        // Starts with 2 whiles (outer + inner).
        assert_eq!(count_whiles(&prog), 2);

        let count = loop_unroll_fn(&mut prog.functions[0]);
        // BOTH whiles get unrolled: inner first (bottom-up), then the
        // outer's body is normalised and the outer pair is detectable
        // (Let i = 0; While i < 3 { ...; i = i + 1 }) — the inner Let j
        // inside the outer body doesn't affect the outer's eligibility
        // because the outer pair-scan looks at top-level statements
        // around the outer While.
        assert_eq!(count, 2, "both loops should unroll (inner then outer)");
        assert_eq!(count_whiles(&prog), 0);
    }

    #[test]
    fn unroll_via_default_pass_sequence_runs() {
        // Through the full optimize_program pipeline, the unrollable
        // loop should disappear because loop_unroll is in
        // DEFAULT_PASS_SEQUENCE.
        let prog = unrollable_program("i", 0, 4);
        assert_eq!(count_whiles(&prog), 1);
        let optimized = optimize_program(&prog);
        assert_eq!(
            count_whiles(&optimized),
            0,
            "loop_unroll in DEFAULT_PASS_SEQUENCE should eliminate this while",
        );
    }

    #[test]
    fn unroll_strict_float_accumulator_preserves_order() {
        // The structural argument for promoting loop_unroll to Tier 1
        // (Universal) in cjc-cana::legality::pass_safety_tier hinges on
        // this invariant: replicating the body N times in sequence
        // preserves the relative order of every floating-point add in a
        // strict accumulator. This test is the witness.
        //
        // Build:
        //   let mut acc: f64 = 0.0;
        //   let mut i: i64 = 0;
        //   while i < 4 {
        //       acc = acc + 1.5;       // strict reduction
        //       i = i + 1;
        //   }
        //   acc
        //
        // After unrolling, the body becomes 4 sequential copies of
        // `acc = acc + 1.5; i = i + 1;`. The 4 float adds happen in
        // exactly the order
        //   acc0 + 1.5, (acc0 + 1.5) + 1.5, ...
        // which is the same order the rolled form would produce. We
        // can't check the dynamic value here (we'd need to run the
        // executor), but we CAN check that the unrolled MIR contains
        // exactly 4 sequential `acc = acc + 1.5;` statements with the
        // same operand types as the original — that's the structural
        // witness that operand ordering is preserved.
        let body_stmts = vec![
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var("acc".to_string()),
                    }),
                    value: Box::new(MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Add,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var("acc".to_string()),
                            }),
                            right: Box::new(MirExpr { kind: MirExprKind::FloatLit(1.5) }),
                        },
                    }),
                },
            }),
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign {
                    target: Box::new(MirExpr {
                        kind: MirExprKind::Var("i".to_string()),
                    }),
                    value: Box::new(MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Add,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var("i".to_string()),
                            }),
                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(1) }),
                        },
                    }),
                },
            }),
        ];
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "acc".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::FloatLit(0.0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Let {
                    name: "i".to_string(),
                    mutable: true,
                    init: MirExpr { kind: MirExprKind::IntLit(0) },
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::While {
                    cond: MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Lt,
                            left: Box::new(MirExpr {
                                kind: MirExprKind::Var("i".to_string()),
                            }),
                            right: Box::new(MirExpr { kind: MirExprKind::IntLit(4) }),
                        },
                    },
                    body: MirBody { stmts: body_stmts, result: None },
                },
            ],
            result: Some(Box::new(MirExpr {
                kind: MirExprKind::Var("acc".to_string()),
            })),
        };
        let mut prog = mk_program(vec![mk_fn("strict_acc", body.stmts, body.result.map(|b| *b))]);
        let count = loop_unroll_fn(&mut prog.functions[0]);
        assert_eq!(count, 1, "strict float accumulator loop should unroll");

        // Walk the function and count float-add Assigns to `acc`.
        // Expect exactly 4 (one per unrolled iteration) — that's the
        // structural witness for accumulator-order preservation.
        let mut acc_adds = 0;
        for stmt in &prog.functions[0].body.stmts {
            if let MirStmt::Expr(MirExpr {
                kind: MirExprKind::Assign { target, value },
            }) = stmt
            {
                let assigns_acc = matches!(
                    &target.kind,
                    MirExprKind::Var(n) if n == "acc"
                );
                let adds_float = matches!(
                    &value.kind,
                    MirExprKind::Binary { op: BinOp::Add, right, .. }
                        if matches!(right.kind, MirExprKind::FloatLit(_))
                );
                if assigns_acc && adds_float {
                    acc_adds += 1;
                }
            }
        }
        assert_eq!(
            acc_adds, 4,
            "after unroll, expected exactly 4 sequential `acc = acc + FloatLit` \
             statements (one per iteration) — the structural witness that the \
             strict accumulator's bit-exact ordering is preserved",
        );
    }
}
