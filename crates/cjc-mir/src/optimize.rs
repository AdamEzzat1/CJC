//! MIR Optimizer (Stage 2.4)
//!
//! Two passes:
//! 1. Constant Folding (CF) — fold pure operations with literal operands.
//! 2. Dead Code Elimination (DCE) — remove unused assigns and unreachable blocks.
//!
//! Design constraints:
//! - Bit-identical results: no float reassociation, no reorder of evaluation.
//! - Shape checks and bounds checks must be preserved.
//! - Side-effecting operations (calls, assigns, index) must not be removed.

use crate::{MirBody, MirExpr, MirExprKind, MirFunction, MirProgram, MirStmt};
use cjc_ast::{BinOp, UnaryOp};
use std::collections::HashSet;

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

    // Pass 2: Dead Code Elimination
    for func in &mut optimized.functions {
        dce_fn(func);
    }

    // Pass 3: Second round of CF (may expose new opportunities after DCE)
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
}
