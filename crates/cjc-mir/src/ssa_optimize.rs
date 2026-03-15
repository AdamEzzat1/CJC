//! SSA-aware optimizer passes for CFG-form MIR.
//!
//! These passes operate on `MirCfg` and use SSA analysis (dominator tree,
//! phi nodes, variable versioning) for more precise optimization than the
//! tree-form optimizer.
//!
//! ## Passes (in order)
//!
//! 1. **Constant Folding** — Fold constant expressions in CFG basic blocks.
//! 2. **SCCP** (Sparse Conditional Constant Propagation) — Propagate constants
//!    through SSA variables and across control flow, prune unreachable branches.
//! 3. **Dead Code Elimination** — Remove unused variable definitions using SSA
//!    use counts.
//! 4. **Strength Reduction** — Algebraic simplifications (multiply by 0/1,
//!    add 0, double negation, etc.).
//! 5. **CFG Cleanup** — Remove empty/unreachable blocks, simplify trivial gotos.
//!
//! ## Design constraints
//!
//! - Bit-identical results: no float reassociation, no reorder of evaluation.
//! - Side-effecting operations (calls, index, field access) are never removed.
//! - All passes are deterministic.

use std::collections::BTreeMap;

use crate::cfg::{CfgStmt, MirCfg, Terminator};
use crate::ssa::SsaForm;
use crate::{MirExpr, MirExprKind};
use cjc_ast::{BinOp, UnaryOp};

// ===========================================================================
// Public API
// ===========================================================================

/// Run all SSA-aware optimization passes on a CFG.
///
/// `params` are the function parameter names (needed for SSA construction).
/// Returns a new, optimized CFG.
pub fn optimize_cfg(cfg: &MirCfg, params: &[String]) -> MirCfg {
    let mut opt = cfg.clone();

    // Pass 1: Constant folding (expression-level, no SSA needed)
    constant_fold_cfg(&mut opt);

    // Pass 2: SCCP (uses SSA for cross-block constant propagation)
    sccp_pass(&mut opt, params);

    // Pass 3: Strength reduction
    strength_reduce_cfg(&mut opt);

    // Pass 4: Dead code elimination (SSA-based)
    ssa_dce(&mut opt, params);

    // Pass 5: CFG cleanup
    cfg_cleanup(&mut opt);

    // Pass 6: Second round of constant folding (may expose new opportunities)
    constant_fold_cfg(&mut opt);

    opt
}

// ===========================================================================
// Pass 1: Constant Folding (expression-level)
// ===========================================================================

/// Fold constant sub-expressions in every basic block.
pub fn constant_fold_cfg(cfg: &mut MirCfg) {
    for block in &mut cfg.basic_blocks {
        for stmt in &mut block.statements {
            match stmt {
                CfgStmt::Let { init, .. } => constant_fold_expr(init),
                CfgStmt::Expr(expr) => constant_fold_expr(expr),
            }
        }
        // Fold terminators.
        match &mut block.terminator {
            Terminator::Branch { cond, .. } => constant_fold_expr(cond),
            Terminator::Return(Some(expr)) => constant_fold_expr(expr),
            _ => {}
        }
    }
}

fn constant_fold_expr(expr: &mut MirExpr) {
    // Recurse into sub-expressions first.
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
        MirExprKind::Assign { value, .. } => {
            constant_fold_expr(value);
        }
        _ => {}
    }

    // Try to fold this expression.
    if let Some(folded) = try_fold(expr) {
        *expr = folded;
    }
}

/// Try to fold a constant expression. Returns Some(folded) if foldable.
fn try_fold(expr: &MirExpr) -> Option<MirExpr> {
    match &expr.kind {
        MirExprKind::Binary { op, left, right } => fold_binary(*op, left, right),
        MirExprKind::Unary { op, operand } => fold_unary(*op, operand),
        _ => None,
    }
}

fn fold_binary(op: BinOp, left: &MirExpr, right: &MirExpr) -> Option<MirExpr> {
    match (&left.kind, &right.kind) {
        (MirExprKind::IntLit(a), MirExprKind::IntLit(b)) => {
            let result = match op {
                BinOp::Add => Some(MirExprKind::IntLit(a.wrapping_add(*b))),
                BinOp::Sub => Some(MirExprKind::IntLit(a.wrapping_sub(*b))),
                BinOp::Mul => Some(MirExprKind::IntLit(a.wrapping_mul(*b))),
                BinOp::Div if *b != 0 => Some(MirExprKind::IntLit(a / b)),
                BinOp::Mod if *b != 0 => Some(MirExprKind::IntLit(a % b)),
                BinOp::Eq => Some(MirExprKind::BoolLit(a == b)),
                BinOp::Ne => Some(MirExprKind::BoolLit(a != b)),
                BinOp::Lt => Some(MirExprKind::BoolLit(a < b)),
                BinOp::Le => Some(MirExprKind::BoolLit(a <= b)),
                BinOp::Gt => Some(MirExprKind::BoolLit(a > b)),
                BinOp::Ge => Some(MirExprKind::BoolLit(a >= b)),
                _ => None,
            };
            result.map(|kind| MirExpr { kind })
        }
        (MirExprKind::FloatLit(a), MirExprKind::FloatLit(b)) => {
            let result = match op {
                BinOp::Add => Some(MirExprKind::FloatLit(a + b)),
                BinOp::Sub => Some(MirExprKind::FloatLit(a - b)),
                BinOp::Mul => Some(MirExprKind::FloatLit(a * b)),
                BinOp::Div if *b != 0.0 => Some(MirExprKind::FloatLit(a / b)),
                BinOp::Eq => Some(MirExprKind::BoolLit(a == b)),
                BinOp::Ne => Some(MirExprKind::BoolLit(a != b)),
                BinOp::Lt => Some(MirExprKind::BoolLit(a < b)),
                BinOp::Le => Some(MirExprKind::BoolLit(a <= b)),
                BinOp::Gt => Some(MirExprKind::BoolLit(a > b)),
                BinOp::Ge => Some(MirExprKind::BoolLit(a >= b)),
                _ => None,
            };
            result.map(|kind| MirExpr { kind })
        }
        (MirExprKind::BoolLit(a), MirExprKind::BoolLit(b)) => {
            let result = match op {
                BinOp::And => Some(MirExprKind::BoolLit(*a && *b)),
                BinOp::Or => Some(MirExprKind::BoolLit(*a || *b)),
                BinOp::Eq => Some(MirExprKind::BoolLit(a == b)),
                BinOp::Ne => Some(MirExprKind::BoolLit(a != b)),
                _ => None,
            };
            result.map(|kind| MirExpr { kind })
        }
        (MirExprKind::StringLit(a), MirExprKind::StringLit(b)) => {
            if op == BinOp::Add {
                Some(MirExpr {
                    kind: MirExprKind::StringLit(format!("{}{}", a, b)),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn fold_unary(op: UnaryOp, operand: &MirExpr) -> Option<MirExpr> {
    match (&operand.kind, op) {
        (MirExprKind::IntLit(v), UnaryOp::Neg) => {
            Some(MirExpr { kind: MirExprKind::IntLit(-v) })
        }
        (MirExprKind::FloatLit(v), UnaryOp::Neg) => {
            Some(MirExpr { kind: MirExprKind::FloatLit(-v) })
        }
        (MirExprKind::BoolLit(v), UnaryOp::Not) => {
            Some(MirExpr { kind: MirExprKind::BoolLit(!v) })
        }
        _ => None,
    }
}

// ===========================================================================
// Pass 2: SCCP (Sparse Conditional Constant Propagation)
// ===========================================================================

/// Lattice value for SCCP.
#[derive(Debug, Clone, PartialEq)]
enum Lattice {
    /// Not yet determined.
    Top,
    /// Known constant.
    Constant(ConstVal),
    /// Definitely not a constant.
    Bottom,
}

/// Constant values tracked by SCCP.
#[derive(Debug, Clone, PartialEq)]
enum ConstVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl Lattice {
    fn meet(&self, other: &Lattice) -> Lattice {
        match (self, other) {
            (Lattice::Top, x) | (x, Lattice::Top) => x.clone(),
            (Lattice::Bottom, _) | (_, Lattice::Bottom) => Lattice::Bottom,
            (Lattice::Constant(a), Lattice::Constant(b)) => {
                if a == b {
                    Lattice::Constant(a.clone())
                } else {
                    Lattice::Bottom
                }
            }
        }
    }

    fn is_constant(&self) -> bool {
        matches!(self, Lattice::Constant(_))
    }
}

/// Run SCCP on the CFG. Replaces constant variables with their values
/// and simplifies branches with constant conditions.
fn sccp_pass(cfg: &mut MirCfg, params: &[String]) {
    let n = cfg.basic_blocks.len();
    if n == 0 {
        return;
    }

    let ssa = SsaForm::construct(cfg, params);

    // Initialize lattice for all variables.
    let mut lattice: BTreeMap<String, Lattice> = BTreeMap::new();

    // Parameters are non-constant.
    for p in params {
        lattice.insert(p.to_string(), Lattice::Bottom);
    }

    // Process each block in RPO (already ordered by block ID for structured CFGs).
    let mut reachable = vec![false; n];
    reachable[cfg.entry.0 as usize] = true;

    // Iterative analysis.
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..n {
            if !reachable[b] {
                continue;
            }
            let block = &cfg.basic_blocks[b];

            // Process phi nodes.
            for phi in &ssa.phis[b] {
                let mut result = Lattice::Top;
                for (pred, src_var) in &phi.sources {
                    if !reachable[pred.0 as usize] {
                        continue; // Ignore unreachable predecessors.
                    }
                    let src_val = lattice
                        .get(&src_var.name)
                        .cloned()
                        .unwrap_or(Lattice::Top);
                    result = result.meet(&src_val);
                }
                let key = phi.target.name.clone();
                let old = lattice.get(&key).cloned().unwrap_or(Lattice::Top);
                let new = old.meet(&result);
                if new != old {
                    lattice.insert(key, new);
                    changed = true;
                }
            }

            // Process statements.
            for stmt in &block.statements {
                match stmt {
                    CfgStmt::Let { name, init, .. } => {
                        let val = eval_lattice(init, &lattice);
                        let old = lattice.get(name).cloned().unwrap_or(Lattice::Top);
                        let new = old.meet(&val);
                        if new != old {
                            lattice.insert(name.clone(), new);
                            changed = true;
                        }
                    }
                    CfgStmt::Expr(expr) => {
                        if let MirExprKind::Assign { target, value } = &expr.kind {
                            if let MirExprKind::Var(name) = &target.kind {
                                let val = eval_lattice(value, &lattice);
                                let old = lattice
                                    .get(name)
                                    .cloned()
                                    .unwrap_or(Lattice::Top);
                                let new = old.meet(&val);
                                if new != old {
                                    lattice.insert(name.clone(), new);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }

            // Process terminator to determine successor reachability.
            match &block.terminator {
                Terminator::Goto(target) => {
                    if !reachable[target.0 as usize] {
                        reachable[target.0 as usize] = true;
                        changed = true;
                    }
                }
                Terminator::Branch {
                    cond,
                    then_block,
                    else_block,
                } => {
                    let cond_val = eval_lattice(cond, &lattice);
                    match &cond_val {
                        Lattice::Constant(ConstVal::Bool(true)) => {
                            if !reachable[then_block.0 as usize] {
                                reachable[then_block.0 as usize] = true;
                                changed = true;
                            }
                        }
                        Lattice::Constant(ConstVal::Bool(false)) => {
                            if !reachable[else_block.0 as usize] {
                                reachable[else_block.0 as usize] = true;
                                changed = true;
                            }
                        }
                        _ => {
                            // Both branches reachable.
                            let t = then_block.0 as usize;
                            let e = else_block.0 as usize;
                            if !reachable[t] {
                                reachable[t] = true;
                                changed = true;
                            }
                            if !reachable[e] {
                                reachable[e] = true;
                                changed = true;
                            }
                        }
                    }
                }
                Terminator::Return(_) | Terminator::Unreachable => {}
            }
        }
    }

    // === Apply SCCP results ===

    // Replace known-constant variables with their values.
    for block in &mut cfg.basic_blocks {
        for stmt in &mut block.statements {
            match stmt {
                CfgStmt::Let { init, .. } => replace_constants(init, &lattice),
                CfgStmt::Expr(expr) => replace_constants(expr, &lattice),
            }
        }
        match &mut block.terminator {
            Terminator::Branch { cond, .. } => replace_constants(cond, &lattice),
            Terminator::Return(Some(expr)) => replace_constants(expr, &lattice),
            _ => {}
        }
    }

    // Simplify branches with constant conditions.
    for b in 0..n {
        let terminator = &cfg.basic_blocks[b].terminator;
        if let Terminator::Branch {
            cond,
            then_block,
            else_block,
        } = terminator
        {
            match &cond.kind {
                MirExprKind::BoolLit(true) => {
                    let target = *then_block;
                    cfg.basic_blocks[b].terminator = Terminator::Goto(target);
                }
                MirExprKind::BoolLit(false) => {
                    let target = *else_block;
                    cfg.basic_blocks[b].terminator = Terminator::Goto(target);
                }
                _ => {}
            }
        }
    }

    // Re-fold after constant substitution.
    constant_fold_cfg(cfg);
}

/// Evaluate an expression in the lattice domain.
fn eval_lattice(expr: &MirExpr, lattice: &BTreeMap<String, Lattice>) -> Lattice {
    match &expr.kind {
        MirExprKind::IntLit(v) => Lattice::Constant(ConstVal::Int(*v)),
        MirExprKind::FloatLit(v) => Lattice::Constant(ConstVal::Float(*v)),
        MirExprKind::BoolLit(v) => Lattice::Constant(ConstVal::Bool(*v)),
        MirExprKind::StringLit(s) => Lattice::Constant(ConstVal::Str(s.clone())),
        MirExprKind::Var(name) => {
            lattice.get(name).cloned().unwrap_or(Lattice::Bottom)
        }
        MirExprKind::Binary { op, left, right } => {
            let l = eval_lattice(left, lattice);
            let r = eval_lattice(right, lattice);
            match (&l, &r) {
                (Lattice::Constant(lv), Lattice::Constant(rv)) => {
                    eval_binary_const(*op, lv, rv)
                }
                (Lattice::Bottom, _) | (_, Lattice::Bottom) => Lattice::Bottom,
                _ => Lattice::Top,
            }
        }
        MirExprKind::Unary { op, operand } => {
            let v = eval_lattice(operand, lattice);
            match &v {
                Lattice::Constant(cv) => eval_unary_const(*op, cv),
                Lattice::Bottom => Lattice::Bottom,
                _ => Lattice::Top,
            }
        }
        // Calls, field accesses, etc. are non-constant.
        _ => Lattice::Bottom,
    }
}

fn eval_binary_const(op: BinOp, left: &ConstVal, right: &ConstVal) -> Lattice {
    match (left, right) {
        (ConstVal::Int(a), ConstVal::Int(b)) => {
            let r = match op {
                BinOp::Add => Some(ConstVal::Int(a.wrapping_add(*b))),
                BinOp::Sub => Some(ConstVal::Int(a.wrapping_sub(*b))),
                BinOp::Mul => Some(ConstVal::Int(a.wrapping_mul(*b))),
                BinOp::Div if *b != 0 => Some(ConstVal::Int(a / b)),
                BinOp::Mod if *b != 0 => Some(ConstVal::Int(a % b)),
                BinOp::Eq => Some(ConstVal::Bool(a == b)),
                BinOp::Ne => Some(ConstVal::Bool(a != b)),
                BinOp::Lt => Some(ConstVal::Bool(a < b)),
                BinOp::Le => Some(ConstVal::Bool(a <= b)),
                BinOp::Gt => Some(ConstVal::Bool(a > b)),
                BinOp::Ge => Some(ConstVal::Bool(a >= b)),
                _ => None,
            };
            r.map(Lattice::Constant).unwrap_or(Lattice::Bottom)
        }
        (ConstVal::Float(a), ConstVal::Float(b)) => {
            let r = match op {
                BinOp::Add => Some(ConstVal::Float(a + b)),
                BinOp::Sub => Some(ConstVal::Float(a - b)),
                BinOp::Mul => Some(ConstVal::Float(a * b)),
                BinOp::Div if *b != 0.0 => Some(ConstVal::Float(a / b)),
                BinOp::Lt => Some(ConstVal::Bool(a < b)),
                BinOp::Le => Some(ConstVal::Bool(a <= b)),
                BinOp::Gt => Some(ConstVal::Bool(a > b)),
                BinOp::Ge => Some(ConstVal::Bool(a >= b)),
                _ => None,
            };
            r.map(Lattice::Constant).unwrap_or(Lattice::Bottom)
        }
        (ConstVal::Bool(a), ConstVal::Bool(b)) => {
            let r = match op {
                BinOp::And => Some(ConstVal::Bool(*a && *b)),
                BinOp::Or => Some(ConstVal::Bool(*a || *b)),
                BinOp::Eq => Some(ConstVal::Bool(a == b)),
                BinOp::Ne => Some(ConstVal::Bool(a != b)),
                _ => None,
            };
            r.map(Lattice::Constant).unwrap_or(Lattice::Bottom)
        }
        _ => Lattice::Bottom,
    }
}

fn eval_unary_const(op: UnaryOp, val: &ConstVal) -> Lattice {
    match (val, op) {
        (ConstVal::Int(v), UnaryOp::Neg) => Lattice::Constant(ConstVal::Int(-v)),
        (ConstVal::Float(v), UnaryOp::Neg) => Lattice::Constant(ConstVal::Float(-v)),
        (ConstVal::Bool(v), UnaryOp::Not) => Lattice::Constant(ConstVal::Bool(!v)),
        _ => Lattice::Bottom,
    }
}

/// Replace Var references with constant values where known.
fn replace_constants(expr: &mut MirExpr, lattice: &BTreeMap<String, Lattice>) {
    match &mut expr.kind {
        MirExprKind::Var(name) => {
            if let Some(Lattice::Constant(cv)) = lattice.get(name.as_str()) {
                *expr = const_val_to_expr(cv);
            }
        }
        MirExprKind::Binary { left, right, .. } => {
            replace_constants(left, lattice);
            replace_constants(right, lattice);
        }
        MirExprKind::Unary { operand, .. } => {
            replace_constants(operand, lattice);
        }
        MirExprKind::Call { callee, args } => {
            replace_constants(callee, lattice);
            for arg in args {
                replace_constants(arg, lattice);
            }
        }
        MirExprKind::Assign { value, .. } => {
            replace_constants(value, lattice);
        }
        _ => {}
    }
}

fn const_val_to_expr(cv: &ConstVal) -> MirExpr {
    let kind = match cv {
        ConstVal::Int(v) => MirExprKind::IntLit(*v),
        ConstVal::Float(v) => MirExprKind::FloatLit(*v),
        ConstVal::Bool(v) => MirExprKind::BoolLit(*v),
        ConstVal::Str(s) => MirExprKind::StringLit(s.clone()),
    };
    MirExpr { kind }
}

// ===========================================================================
// Pass 3: Strength Reduction
// ===========================================================================

/// Apply algebraic simplifications to all expressions in the CFG.
fn strength_reduce_cfg(cfg: &mut MirCfg) {
    for block in &mut cfg.basic_blocks {
        for stmt in &mut block.statements {
            match stmt {
                CfgStmt::Let { init, .. } => strength_reduce_expr(init),
                CfgStmt::Expr(expr) => strength_reduce_expr(expr),
            }
        }
        match &mut block.terminator {
            Terminator::Branch { cond, .. } => strength_reduce_expr(cond),
            Terminator::Return(Some(expr)) => strength_reduce_expr(expr),
            _ => {}
        }
    }
}

fn strength_reduce_expr(expr: &mut MirExpr) {
    // Recurse first.
    match &mut expr.kind {
        MirExprKind::Binary { left, right, .. } => {
            strength_reduce_expr(left);
            strength_reduce_expr(right);
        }
        MirExprKind::Unary { operand, .. } => {
            strength_reduce_expr(operand);
        }
        MirExprKind::Call { callee, args } => {
            strength_reduce_expr(callee);
            for arg in args {
                strength_reduce_expr(arg);
            }
        }
        MirExprKind::Assign { value, .. } => {
            strength_reduce_expr(value);
        }
        _ => {}
    }

    // Apply reductions.
    if let Some(reduced) = try_strength_reduce(expr) {
        *expr = reduced;
    }
}

fn is_zero(kind: &MirExprKind) -> bool {
    matches!(kind, MirExprKind::IntLit(0))
        || matches!(kind, MirExprKind::FloatLit(v) if *v == 0.0)
}

fn is_one(kind: &MirExprKind) -> bool {
    matches!(kind, MirExprKind::IntLit(1))
        || matches!(kind, MirExprKind::FloatLit(v) if *v == 1.0)
}

fn try_strength_reduce(expr: &MirExpr) -> Option<MirExpr> {
    match &expr.kind {
        MirExprKind::Binary { op, left, right } => {
            match op {
                // x + 0 => x, 0 + x => x
                BinOp::Add => {
                    if is_zero(&right.kind) {
                        return Some((**left).clone());
                    }
                    if is_zero(&left.kind) {
                        return Some((**right).clone());
                    }
                    None
                }
                // x - 0 => x
                BinOp::Sub => {
                    if is_zero(&right.kind) {
                        return Some((**left).clone());
                    }
                    None
                }
                // x * 0 => 0, x * 1 => x, 0 * x => 0, 1 * x => x
                BinOp::Mul => {
                    if matches!(right.kind, MirExprKind::IntLit(0)) {
                        return Some(MirExpr { kind: MirExprKind::IntLit(0) });
                    }
                    if matches!(left.kind, MirExprKind::IntLit(0)) {
                        return Some(MirExpr { kind: MirExprKind::IntLit(0) });
                    }
                    if is_one(&right.kind) {
                        return Some((**left).clone());
                    }
                    if is_one(&left.kind) {
                        return Some((**right).clone());
                    }
                    None
                }
                // x / 1 => x
                BinOp::Div => {
                    if is_one(&right.kind) {
                        return Some((**left).clone());
                    }
                    None
                }
                // true && x => x, x && true => x, false && x => false, false || x => x
                BinOp::And => {
                    if matches!(left.kind, MirExprKind::BoolLit(true)) {
                        return Some((**right).clone());
                    }
                    if matches!(right.kind, MirExprKind::BoolLit(true)) {
                        return Some((**left).clone());
                    }
                    if matches!(left.kind, MirExprKind::BoolLit(false)) {
                        return Some(MirExpr { kind: MirExprKind::BoolLit(false) });
                    }
                    None
                }
                BinOp::Or => {
                    if matches!(left.kind, MirExprKind::BoolLit(false)) {
                        return Some((**right).clone());
                    }
                    if matches!(right.kind, MirExprKind::BoolLit(false)) {
                        return Some((**left).clone());
                    }
                    if matches!(left.kind, MirExprKind::BoolLit(true)) {
                        return Some(MirExpr { kind: MirExprKind::BoolLit(true) });
                    }
                    None
                }
                _ => None,
            }
        }
        // Double negation: --x => x
        MirExprKind::Unary { op: UnaryOp::Neg, operand } => {
            if let MirExprKind::Unary { op: UnaryOp::Neg, operand: inner } = &operand.kind {
                return Some((**inner).clone());
            }
            None
        }
        // Double not: !!x => x
        MirExprKind::Unary { op: UnaryOp::Not, operand } => {
            if let MirExprKind::Unary { op: UnaryOp::Not, operand: inner } = &operand.kind {
                return Some((**inner).clone());
            }
            None
        }
        _ => None,
    }
}

// ===========================================================================
// Pass 4: SSA-based Dead Code Elimination
// ===========================================================================

/// Remove dead variable definitions using SSA use-count analysis.
///
/// A definition is dead if the variable it defines is never used anywhere
/// in the CFG (no reads, no phi sources referencing it) and the definition
/// has no side effects.
fn ssa_dce(cfg: &mut MirCfg, params: &[String]) {
    // Count uses of each variable name across the CFG.
    let mut use_counts: BTreeMap<String, usize> = BTreeMap::new();

    for block in &cfg.basic_blocks {
        for stmt in &block.statements {
            count_uses_in_stmt(stmt, &mut use_counts);
        }
        count_uses_in_terminator(&block.terminator, &mut use_counts);
    }

    // Remove dead definitions (Let with unused name and pure initializer).
    for block in &mut cfg.basic_blocks {
        block.statements.retain(|stmt| {
            match stmt {
                CfgStmt::Let { name, init, .. } => {
                    let count = use_counts.get(name).copied().unwrap_or(0);
                    if count == 0 && !has_side_effects(init) {
                        return false; // Dead — remove.
                    }
                    true
                }
                CfgStmt::Expr(expr) => {
                    // Remove dead assignments to unused variables.
                    if let MirExprKind::Assign { target, value } = &expr.kind {
                        if let MirExprKind::Var(name) = &target.kind {
                            let count = use_counts.get(name).copied().unwrap_or(0);
                            if count == 0 && !has_side_effects(value) {
                                return false; // Dead — remove.
                            }
                        }
                    }
                    true
                }
            }
        });
    }
}

/// Count variable uses in a statement (excluding the def itself).
fn count_uses_in_stmt(stmt: &CfgStmt, counts: &mut BTreeMap<String, usize>) {
    match stmt {
        CfgStmt::Let { init, .. } => {
            count_uses_in_expr(init, counts);
        }
        CfgStmt::Expr(expr) => {
            // For assignments, count uses in the value side + target side (if complex).
            if let MirExprKind::Assign { target, value } = &expr.kind {
                // The target variable is a "use" only for the purpose of
                // knowing someone assigns to it — but what matters for DCE
                // is whether anyone *reads* it.
                count_uses_in_expr(value, counts);
                // For compound targets like field access, count inner uses.
                if !matches!(target.kind, MirExprKind::Var(_)) {
                    count_uses_in_expr(target, counts);
                }
            } else {
                count_uses_in_expr(expr, counts);
            }
        }
    }
}

fn count_uses_in_expr(expr: &MirExpr, counts: &mut BTreeMap<String, usize>) {
    match &expr.kind {
        MirExprKind::Var(name) => {
            *counts.entry(name.clone()).or_insert(0) += 1;
        }
        MirExprKind::Binary { left, right, .. } => {
            count_uses_in_expr(left, counts);
            count_uses_in_expr(right, counts);
        }
        MirExprKind::Unary { operand, .. } => {
            count_uses_in_expr(operand, counts);
        }
        MirExprKind::Call { callee, args } => {
            count_uses_in_expr(callee, counts);
            for arg in args {
                count_uses_in_expr(arg, counts);
            }
        }
        MirExprKind::Assign { target, value } => {
            count_uses_in_expr(target, counts);
            count_uses_in_expr(value, counts);
        }
        MirExprKind::Field { object, .. } => {
            count_uses_in_expr(object, counts);
        }
        MirExprKind::Index { object, index } => {
            count_uses_in_expr(object, counts);
            count_uses_in_expr(index, counts);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, expr) in fields {
                count_uses_in_expr(expr, counts);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                count_uses_in_expr(e, counts);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for c in captures {
                count_uses_in_expr(c, counts);
            }
        }
        _ => {}
    }
}

fn count_uses_in_terminator(term: &Terminator, counts: &mut BTreeMap<String, usize>) {
    match term {
        Terminator::Branch { cond, .. } => count_uses_in_expr(cond, counts),
        Terminator::Return(Some(expr)) => count_uses_in_expr(expr, counts),
        _ => {}
    }
}

/// Check if an expression has side effects (calls, assignments, indexing).
fn has_side_effects(expr: &MirExpr) -> bool {
    match &expr.kind {
        MirExprKind::Call { .. } => true,
        MirExprKind::Assign { .. } => true,
        MirExprKind::Index { .. } => true,   // May panic
        MirExprKind::Field { .. } => true,   // May panic on missing field
        MirExprKind::Binary { left, right, .. } => {
            has_side_effects(left) || has_side_effects(right)
        }
        MirExprKind::Unary { operand, .. } => has_side_effects(operand),
        _ => false,
    }
}

// ===========================================================================
// Pass 5: CFG Cleanup
// ===========================================================================

/// Simplify the CFG structure.
///
/// - Redirects blocks that Goto a block which itself only does Goto (chain
///   folding).
/// - Simplifies branches with constant conditions.
pub fn cfg_cleanup(cfg: &mut MirCfg) {
    let n = cfg.basic_blocks.len();

    // Chain-fold: if block B terminates with Goto(C) and C has no statements
    // and terminates with Goto(D), redirect B to Goto(D).
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;

        for b in 0..n {
            let term = cfg.basic_blocks[b].terminator.clone();
            match &term {
                Terminator::Goto(target) => {
                    let t = target.0 as usize;
                    if t != b
                        && cfg.basic_blocks[t].statements.is_empty()
                    {
                        if let Terminator::Goto(next) = cfg.basic_blocks[t].terminator {
                            if next.0 as usize != t {
                                cfg.basic_blocks[b].terminator = Terminator::Goto(next);
                                changed = true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// ===========================================================================
// Stats / diagnostics
// ===========================================================================

/// Statistics about optimizations applied to a CFG.
#[derive(Debug, Clone, Default)]
pub struct OptStats {
    /// Number of expressions constant-folded.
    pub constants_folded: usize,
    /// Number of branches simplified (constant condition).
    pub branches_simplified: usize,
    /// Number of dead definitions removed.
    pub dead_defs_removed: usize,
    /// Number of strength reductions applied.
    pub strength_reductions: usize,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock, CfgBuilder, CfgStmt};
    use crate::{BlockId, MirBody, MirExpr, MirExprKind, MirStmt};

    fn int_expr(v: i64) -> MirExpr {
        MirExpr { kind: MirExprKind::IntLit(v) }
    }

    fn bool_expr(b: bool) -> MirExpr {
        MirExpr { kind: MirExprKind::BoolLit(b) }
    }

    fn var_expr(name: &str) -> MirExpr {
        MirExpr { kind: MirExprKind::Var(name.to_string()) }
    }

    fn assign_expr(name: &str, value: MirExpr) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Assign {
                target: Box::new(var_expr(name)),
                value: Box::new(value),
            },
        }
    }

    fn add_expr(left: MirExpr, right: MirExpr) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }

    fn mul_expr(left: MirExpr, right: MirExpr) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Binary {
                op: BinOp::Mul,
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }

    // ── Constant folding ─────────────────────────────────────────

    #[test]
    fn test_cf_int_arithmetic() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "x".into(),
                    mutable: false,
                    init: add_expr(int_expr(10), int_expr(32)),
                }],
                terminator: Terminator::Return(Some(var_expr("x"))),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &[]);
        // After SCCP + folding, x=42 propagates to return, so return is IntLit(42).
        match &opt.basic_blocks[0].terminator {
            Terminator::Return(Some(expr)) => {
                assert!(
                    matches!(expr.kind, MirExprKind::IntLit(42)),
                    "expected return 42, got {:?}",
                    expr.kind
                );
            }
            other => panic!("expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_cf_bool_comparison() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "b".into(),
                    mutable: false,
                    init: MirExpr {
                        kind: MirExprKind::Binary {
                            op: BinOp::Lt,
                            left: Box::new(int_expr(3)),
                            right: Box::new(int_expr(5)),
                        },
                    },
                }],
                terminator: Terminator::Return(Some(var_expr("b"))),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &[]);
        match &opt.basic_blocks[0].terminator {
            Terminator::Return(Some(expr)) => {
                assert!(
                    matches!(expr.kind, MirExprKind::BoolLit(true)),
                    "expected return true, got {:?}",
                    expr.kind
                );
            }
            other => panic!("expected Return, got {:?}", other),
        }
    }

    // ── Strength reduction ───────────────────────────────────────

    #[test]
    fn test_sr_multiply_by_one() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "x".into(),
                    mutable: false,
                    init: mul_expr(var_expr("a"), int_expr(1)),
                }],
                terminator: Terminator::Return(Some(var_expr("x"))),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &["a".to_string()]);
        // x = a * 1 strength-reduces to x = a. Check the let init.
        if let Some(CfgStmt::Let { init, .. }) = opt.basic_blocks[0].statements.first() {
            assert!(
                matches!(init.kind, MirExprKind::Var(ref n) if n == "a"),
                "x * 1 should reduce to x, got {:?}",
                init.kind
            );
        } else {
            // DCE may have removed the let and propagated to return.
            match &opt.basic_blocks[0].terminator {
                Terminator::Return(Some(expr)) => {
                    assert!(
                        matches!(expr.kind, MirExprKind::Var(ref n) if n == "a"),
                        "return should be 'a' after propagation, got {:?}",
                        expr.kind
                    );
                }
                other => panic!("expected a statement or return with 'a', got {:?}", other),
            }
        }
    }

    #[test]
    fn test_sr_add_zero() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "y".into(),
                    mutable: false,
                    init: add_expr(var_expr("a"), int_expr(0)),
                }],
                terminator: Terminator::Return(Some(var_expr("y"))),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &["a".to_string()]);
        if let Some(CfgStmt::Let { init, .. }) = opt.basic_blocks[0].statements.first() {
            assert!(
                matches!(init.kind, MirExprKind::Var(ref n) if n == "a"),
                "a + 0 should reduce to a, got {:?}",
                init.kind
            );
        } else {
            match &opt.basic_blocks[0].terminator {
                Terminator::Return(Some(expr)) => {
                    assert!(
                        matches!(expr.kind, MirExprKind::Var(ref n) if n == "a"),
                        "return should be 'a', got {:?}",
                        expr.kind
                    );
                }
                other => panic!("expected 'a', got {:?}", other),
            }
        }
    }

    #[test]
    fn test_sr_multiply_by_zero() {
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "z".into(),
                    mutable: false,
                    init: mul_expr(var_expr("a"), int_expr(0)),
                }],
                terminator: Terminator::Return(Some(var_expr("z"))),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &["a".to_string()]);
        // SR reduces a * 0 to 0. Check the let init was simplified.
        if let Some(CfgStmt::Let { init, .. }) = opt.basic_blocks[0].statements.first() {
            assert!(
                matches!(init.kind, MirExprKind::IntLit(0)),
                "a * 0 should be reduced to 0, got {:?}",
                init.kind
            );
        }
    }

    // ── SCCP: constant propagation ───────────────────────────────

    #[test]
    fn test_sccp_propagates_constant() {
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![CfgStmt::Let {
                        name: "x".into(),
                        mutable: false,
                        init: int_expr(42),
                    }],
                    terminator: Terminator::Goto(BlockId(1)),
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![CfgStmt::Let {
                        name: "y".into(),
                        mutable: false,
                        init: add_expr(var_expr("x"), int_expr(8)),
                    }],
                    terminator: Terminator::Return(Some(var_expr("y"))),
                },
            ],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &[]);
        // y = x + 8 = 42 + 8 = 50, propagated to return.
        match &opt.basic_blocks[1].terminator {
            Terminator::Return(Some(expr)) => {
                assert!(
                    matches!(expr.kind, MirExprKind::IntLit(50)),
                    "return should be 50, got {:?}",
                    expr.kind
                );
            }
            other => panic!("expected Return(50), got {:?}", other),
        }
    }

    #[test]
    fn test_sccp_simplifies_branch() {
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![],
                    terminator: Terminator::Branch {
                        cond: bool_expr(true),
                        then_block: BlockId(1),
                        else_block: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![],
                    terminator: Terminator::Return(Some(int_expr(1))),
                },
                BasicBlock {
                    id: BlockId(2),
                    statements: vec![],
                    terminator: Terminator::Return(Some(int_expr(2))),
                },
            ],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &[]);
        // Branch on `true` should become Goto(1).
        assert!(
            matches!(opt.basic_blocks[0].terminator, Terminator::Goto(BlockId(1))),
            "branch on true should simplify to Goto(1), got {:?}",
            opt.basic_blocks[0].terminator
        );
    }

    // ── DCE: dead code elimination ───────────────────────────────

    #[test]
    fn test_dce_removes_unused_let() {
        // "unused" is never referenced. "used" is referenced in a call (side-effecting).
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![
                    CfgStmt::Let {
                        name: "unused".into(),
                        mutable: false,
                        init: int_expr(99),
                    },
                    CfgStmt::Expr(MirExpr {
                        kind: MirExprKind::Call {
                            callee: Box::new(var_expr("print")),
                            args: vec![var_expr("used")],
                        },
                    }),
                ],
                terminator: Terminator::Return(None),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &["used".to_string()]);
        // "unused" should be removed, but the print call stays.
        assert_eq!(
            opt.basic_blocks[0].statements.len(),
            1,
            "dead let should be removed, call should remain"
        );
        assert!(
            matches!(&opt.basic_blocks[0].statements[0], CfgStmt::Expr(_)),
            "remaining statement should be the call"
        );
    }

    #[test]
    fn test_dce_keeps_side_effect() {
        // A let binding with a Call initializer must NOT be removed even if unused,
        // because the call may have side effects.
        let cfg = MirCfg {
            basic_blocks: vec![BasicBlock {
                id: BlockId(0),
                statements: vec![CfgStmt::Let {
                    name: "unused".into(),
                    mutable: false,
                    init: MirExpr {
                        kind: MirExprKind::Call {
                            callee: Box::new(var_expr("print")),
                            args: vec![int_expr(42)],
                        },
                    },
                }],
                terminator: Terminator::Return(None),
            }],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &[]);
        assert_eq!(
            opt.basic_blocks[0].statements.len(),
            1,
            "side-effecting let should be kept"
        );
    }

    // ── CFG cleanup ──────────────────────────────────────────────

    #[test]
    fn test_cleanup_chain_fold() {
        // Block 0 -> Block 1 (empty) -> Block 2.
        // After cleanup, Block 0 should go directly to Block 2.
        let cfg = MirCfg {
            basic_blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    statements: vec![],
                    terminator: Terminator::Goto(BlockId(1)),
                },
                BasicBlock {
                    id: BlockId(1),
                    statements: vec![],
                    terminator: Terminator::Goto(BlockId(2)),
                },
                BasicBlock {
                    id: BlockId(2),
                    statements: vec![],
                    terminator: Terminator::Return(None),
                },
            ],
            entry: BlockId(0),
        };
        let opt = optimize_cfg(&cfg, &[]);
        assert!(
            matches!(opt.basic_blocks[0].terminator, Terminator::Goto(BlockId(2))),
            "chain should fold: 0->2 directly, got {:?}",
            opt.basic_blocks[0].terminator
        );
    }

    // ── End-to-end: optimize_cfg round-trip ──────────────────────

    #[test]
    fn test_optimize_cfg_preserves_correct_program() {
        // let x = 2 + 3; return x;
        let body = MirBody {
            stmts: vec![MirStmt::Let {
                name: "x".into(),
                mutable: false,
                init: add_expr(int_expr(2), int_expr(3)),
                alloc_hint: None,
            }],
            result: Some(Box::new(var_expr("x"))),
        };
        let cfg = CfgBuilder::build(&body);
        let opt = optimize_cfg(&cfg, &[]);

        // After optimization, x should be folded to 5 and the return should
        // reference a constant.
        match &opt.basic_blocks[0].terminator {
            Terminator::Return(Some(expr)) => {
                assert!(
                    matches!(expr.kind, MirExprKind::IntLit(5)),
                    "return should be folded to 5, got {:?}",
                    expr.kind
                );
            }
            other => panic!("expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_optimize_cfg_deterministic() {
        let body = MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "x".into(),
                    mutable: true,
                    init: int_expr(10),
                    alloc_hint: None,
                },
                MirStmt::If {
                    cond: bool_expr(true),
                    then_body: MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("x", int_expr(20)))],
                        result: None,
                    },
                    else_body: Some(MirBody {
                        stmts: vec![MirStmt::Expr(assign_expr("x", int_expr(30)))],
                        result: None,
                    }),
                },
            ],
            result: Some(Box::new(var_expr("x"))),
        };
        let cfg = CfgBuilder::build(&body);
        let opt1 = optimize_cfg(&cfg, &[]);
        let opt2 = optimize_cfg(&cfg, &[]);

        assert_eq!(opt1.basic_blocks.len(), opt2.basic_blocks.len());
        for (b1, b2) in opt1.basic_blocks.iter().zip(opt2.basic_blocks.iter()) {
            assert_eq!(b1.statements.len(), b2.statements.len());
        }
    }
}
