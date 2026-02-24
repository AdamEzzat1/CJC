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
use std::collections::{HashMap, HashSet};

// ===========================================================================
// Public API
// ===========================================================================

/// Run all optimization passes on a MIR program.
/// Returns a new program with optimizations applied.
pub fn optimize_program(program: &MirProgram) -> MirProgram {
    let mut optimized = program.clone();

    // Pass 1: Constant Folding
    for func in &mut optimized.functions {
        constant_fold_fn(func);
    }

    // Pass 2: Strength Reduction
    for func in &mut optimized.functions {
        strength_reduce_fn(func);
    }

    // Pass 3: Dead Code Elimination
    for func in &mut optimized.functions {
        dce_fn(func);
    }

    // Pass 4: Common Subexpression Elimination
    for func in &mut optimized.functions {
        cse_fn(func);
    }

    // Pass 5: Loop-Invariant Code Motion
    for func in &mut optimized.functions {
        licm_fn(func);
    }

    // Pass 6: Second round of CF (may expose new opportunities after other passes)
    for func in &mut optimized.functions {
        constant_fold_fn(func);
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
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
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
        BinOp::Eq => Some(MirExprKind::BoolLit(a == b)),
        BinOp::Ne => Some(MirExprKind::BoolLit(a != b)),
        BinOp::Lt => Some(MirExprKind::BoolLit(a < b)),
        BinOp::Gt => Some(MirExprKind::BoolLit(a > b)),
        BinOp::Le => Some(MirExprKind::BoolLit(a <= b)),
        BinOp::Ge => Some(MirExprKind::BoolLit(a >= b)),
        BinOp::And | BinOp::Or | BinOp::Match | BinOp::NotMatch => None,
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
    let mut used_vars = HashSet::new();
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
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
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
fn collect_used_vars_stmt(stmt: &MirStmt, used: &mut HashSet<String>) {
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
        MirStmt::NoGcBlock(body) => {
            collect_used_vars_body(body, used);
        }
    }
}

fn collect_used_vars_body(body: &MirBody, used: &mut HashSet<String>) {
    for stmt in &body.stmts {
        collect_used_vars_stmt(stmt, used);
    }
    if let Some(ref expr) = body.result {
        collect_used_vars_expr(expr, used);
    }
}

fn collect_used_vars_expr(expr: &MirExpr, used: &mut HashSet<String>) {
    match &expr.kind {
        MirExprKind::Var(name) => {
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
    let mut expr_to_var: HashMap<String, String> = HashMap::new();
    let mut replacements: HashMap<String, String> = HashMap::new();

    for stmt in &body.stmts {
        if let MirStmt::Let { name, init, .. } = stmt {
            if is_pure_expr(init) {
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

fn apply_cse_replacements_stmt(stmt: &mut MirStmt, replacements: &HashMap<String, String>) {
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
        MirStmt::NoGcBlock(body) => {
            for s in &mut body.stmts { apply_cse_replacements_stmt(s, replacements); }
            if let Some(ref mut r) = body.result { apply_cse_replacements_expr(r, replacements); }
        }
    }
}

fn apply_cse_replacements_expr(expr: &mut MirExpr, replacements: &HashMap<String, String>) {
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
    let mut modified_vars = HashSet::new();
    collect_modified_vars_body(&loop_body, &mut modified_vars);

    let mut hoisted = Vec::new();
    let mut remaining = Vec::new();

    for stmt in loop_body.stmts {
        if let MirStmt::Let { ref name, ref init, mutable } = stmt {
            if is_pure_expr(init) && !references_any(init, &modified_vars) {
                hoisted.push(MirStmt::Let {
                    name: name.clone(),
                    mutable,
                    init: init.clone(),
                });
                continue;
            }
        }
        remaining.push(stmt);
    }

    (hoisted, MirBody { stmts: remaining, result: loop_body.result })
}

/// Collect all variable names that are modified (written to) in a body.
fn collect_modified_vars_body(body: &MirBody, modified: &mut HashSet<String>) {
    for stmt in &body.stmts {
        collect_modified_vars_stmt(stmt, modified);
    }
}

fn collect_modified_vars_stmt(stmt: &MirStmt, modified: &mut HashSet<String>) {
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
        MirStmt::NoGcBlock(body) => collect_modified_vars_body(body, modified),
    }
}

fn collect_modified_vars_expr(expr: &MirExpr, modified: &mut HashSet<String>) {
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
fn references_any(expr: &MirExpr, vars: &HashSet<String>) -> bool {
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
                },
                MirStmt::Let {
                    name: "b".to_string(),
                    mutable: false,
                    init: mk_int(20),
                },
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_var("a"), mk_var("b")),
                },
                MirStmt::Let {
                    name: "y".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_var("a"), mk_var("b")),
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
                        },
                        MirStmt::Let {
                            name: "x".to_string(),
                            mutable: false,
                            init: mk_binary(BinOp::Add, mk_var("inv"), mk_var("i")),
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
                },
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Mul, mk_var("y"), mk_int(1)),
                },
                MirStmt::Let {
                    name: "z".to_string(),
                    mutable: false,
                    init: mk_binary(BinOp::Add, mk_var("y"), mk_int(0)),
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
}
