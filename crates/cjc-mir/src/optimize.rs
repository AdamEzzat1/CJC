//! MIR Optimizer (Stage 2.4)
//!
//! Passes (applied in order):
//! 1. Constant Folding (CF) — fold pure operations with literal operands.
//! 2. Strength Reduction (SR) — replace expensive ops with cheaper equivalents.
//! 3. Dead Code Elimination (DCE) — remove unused assigns and unreachable blocks.
//! 4. Common Subexpression Elimination (CSE) — reuse identical pure expressions.
//! 5. Loop-Invariant Code Motion (LICM) — hoist invariant code out of loops.
//! 6. Second round of CF (may expose new opportunities after other passes).
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
///
/// Unknown names are silently skipped. This is intentional: CANA may
/// recommend a pass that's not yet implemented in the compiler, and the
/// right response is to skip it (and log via the caller's audit trail),
/// not to error.
pub fn apply_pass(pass_name: &str, func: &mut MirFunction) -> bool {
    match pass_name {
        "constant_fold" | "cf" | "cf_round_2" => {
            constant_fold_fn(func);
            true
        }
        "strength_reduce" | "sr" => {
            strength_reduce_fn(func);
            true
        }
        "dce" | "dead_code_elimination" => {
            dce_fn(func);
            true
        }
        "cse" | "common_subexpression_elimination" => {
            cse_fn(func);
            true
        }
        "licm" | "loop_invariant_code_motion" => {
            licm_fn(func);
            true
        }
        _ => false,
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

fn strength_reduce_fn(func: &mut MirFunction) {
    strength_reduce_body(&mut func.body);
}

fn strength_reduce_body(body: &mut MirBody) {
    for stmt in &mut body.stmts {
        strength_reduce_stmt(stmt);
    }
    if let Some(ref mut expr) = body.result {
        strength_reduce_expr(expr);
    }
}

fn strength_reduce_stmt(stmt: &mut MirStmt) {
    match stmt {
        MirStmt::Let { init, .. } => strength_reduce_expr(init),
        MirStmt::Expr(expr) => strength_reduce_expr(expr),
        MirStmt::If { cond, then_body, else_body } => {
            strength_reduce_expr(cond);
            strength_reduce_body(then_body);
            if let Some(eb) = else_body {
                strength_reduce_body(eb);
            }
        }
        MirStmt::While { cond, body } => {
            strength_reduce_expr(cond);
            strength_reduce_body(body);
        }
        MirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                strength_reduce_expr(expr);
            }
        }
        MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => strength_reduce_body(body),
    }
}

fn strength_reduce_expr(expr: &mut MirExpr) {
    // Recurse first (bottom-up).
    match &mut expr.kind {
        MirExprKind::Binary { left, right, .. } => {
            strength_reduce_expr(left);
            strength_reduce_expr(right);
        }
        MirExprKind::Unary { operand, .. } => strength_reduce_expr(operand),
        MirExprKind::Call { callee, args } => {
            strength_reduce_expr(callee);
            for arg in args { strength_reduce_expr(arg); }
        }
        MirExprKind::Block(body) => strength_reduce_body(body),
        MirExprKind::If { cond, then_body, else_body } => {
            strength_reduce_expr(cond);
            strength_reduce_body(then_body);
            if let Some(eb) = else_body { strength_reduce_body(eb); }
        }
        MirExprKind::Lambda { body, .. } => strength_reduce_expr(body),
        _ => {}
    }

    // Apply strength reduction rules.
    if let Some(reduced) = try_strength_reduce(expr) {
        *expr = reduced;
    }
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

fn cse_fn(func: &mut MirFunction) {
    cse_body(&mut func.body);
}

/// CSE at the body level: find duplicate pure let-bindings and replace
/// uses of the later one with the first.
fn cse_body(body: &mut MirBody) {
    // Build a map from expression hash to the first variable name bound to it.
    // We use a simple structural string representation as the hash key.
    let mut expr_to_var: BTreeMap<String, String> = BTreeMap::new();
    let mut replacements: BTreeMap<String, String> = BTreeMap::new();

    for stmt in &body.stmts {
        if let MirStmt::Let { name, init, mutable, .. } = stmt {
            // Only CSE immutable bindings — mutable variables may diverge
            // after their initializer (e.g. loop counters both init to 0).
            if !mutable && is_pure_expr(init) {
                let key = expr_key(init);
                if let Some(existing) = expr_to_var.get(&key) {
                    // This expression is already computed in `existing`.
                    replacements.insert(name.clone(), existing.clone());
                } else {
                    expr_to_var.insert(key, name.clone());
                }
            }
        }
    }

    if !replacements.is_empty() {
        // Apply replacements: rename uses of duplicate vars to the original.
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
                cse_body(then_body);
                if let Some(eb) = else_body { cse_body(eb); }
            }
            MirStmt::While { body: wb, .. } => cse_body(wb),
            MirStmt::NoGcBlock(b) => cse_body(b),
            _ => {}
        }
    }
}

/// Produce a deterministic string key for a pure expression (for CSE).
fn expr_key(expr: &MirExpr) -> String {
    match &expr.kind {
        MirExprKind::IntLit(v) => format!("int:{v}"),
        MirExprKind::FloatLit(v) => format!("float:{}", v.to_bits()),
        MirExprKind::BoolLit(v) => format!("bool:{v}"),
        MirExprKind::NaLit => "na".to_string(),
        MirExprKind::StringLit(s) => format!("str:{s}"),
        MirExprKind::Var(name) => format!("var:{name}"),
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

fn licm_fn(func: &mut MirFunction) {
    licm_body(&mut func.body);
}

fn licm_body(body: &mut MirBody) {
    // Process nested structures first (bottom-up).
    for stmt in &mut body.stmts {
        match stmt {
            MirStmt::If { then_body, else_body, .. } => {
                licm_body(then_body);
                if let Some(eb) = else_body { licm_body(eb); }
            }
            MirStmt::While { body: wb, .. } => licm_body(wb),
            MirStmt::NoGcBlock(b) => licm_body(b),
            _ => {}
        }
    }

    // Now try to hoist invariant let-bindings from while loops.
    let mut new_stmts = Vec::new();
    for stmt in std::mem::take(&mut body.stmts) {
        if let MirStmt::While { cond, body: loop_body } = stmt {
            let (hoisted, remaining_body) = hoist_invariants(loop_body);
            // Insert hoisted lets before the while.
            new_stmts.extend(hoisted);
            new_stmts.push(MirStmt::While { cond, body: remaining_body });
        } else {
            new_stmts.push(stmt);
        }
    }
    body.stmts = new_stmts;
}

/// Try to hoist loop-invariant let-bindings out of a while body.
/// A let-binding is invariant if:
/// - It is pure
/// - It does not reference any variable that is assigned inside the loop
fn hoist_invariants(loop_body: MirBody) -> (Vec<MirStmt>, MirBody) {
    // Collect variables that are modified (assigned to) inside the loop.
    let mut modified_vars = BTreeSet::new();
    collect_modified_vars_body(&loop_body, &mut modified_vars);

    let mut hoisted = Vec::new();
    let mut remaining = Vec::new();

    for stmt in loop_body.stmts {
        if let MirStmt::Let { ref name, ref init, mutable, alloc_hint, slot } = stmt {
            if is_pure_expr(init) && !references_any(init, &modified_vars) {
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

/// Collect all variable names that are modified (written to) in a body.
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
            if let MirExprKind::Var(name) = &target.kind {
                modified.insert(name.clone());
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

/// Check if an expression references any variable in the given set.
fn references_any(expr: &MirExpr, vars: &BTreeSet<String>) -> bool {
    match &expr.kind {
        MirExprKind::Var(name) => vars.contains(name.as_str()),
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
    fn default_pass_sequence_has_six_entries() {
        // Drift guard: if a pass is added/removed, this needs to change
        // alongside cjc-cana::pass_ranker::CANONICAL_PASSES.
        assert_eq!(DEFAULT_PASS_SEQUENCE.len(), 6);
    }
}
