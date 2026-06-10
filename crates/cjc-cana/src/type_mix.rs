//! Per-function float/int operation mix — the static analog of the
//! executor's runtime FP-binop counter.
//!
//! ## Why this exists (PINN v2 §2.1 data-sanity finding)
//!
//! Recorded thermal pressure is FP-op *density* (`thermal_intensity =
//! fp_ops / instructions` per trace event), but every pre-v2 static
//! feature was type-blind: `estimated_flops` counted ALL expressions.
//! No cost model — linear or neural — can predict FP density from
//! inputs that carry no FP signal. This module closes that information
//! gap with a conservative intra-function float-propagation walk.
//!
//! ## Semantics contract
//!
//! The runtime counter (`cjc-mir-exec/src/lib.rs`, Option-B trace site
//! 4) increments for ANY binary operation where at least one operand
//! evaluates to `Value::Float` — comparisons included, since the
//! increment happens before dispatch on the operator. The static
//! [`TypeMix::float_binop_count`] mirrors that definition: any
//! `Binary` node whose conservative operand typing marks either side
//! float, regardless of operator.
//!
//! Propagation is deliberately simple and deterministic:
//!
//! - Seed: parameters annotated `f64`.
//! - `FloatLit` is float. `Var`/`VarLocal` are float iff bound float.
//! - Arithmetic (`+ - * / % **`) produces float iff either operand is
//!   float; comparisons/logic produce bool.
//! - `let`/assignment of a float-valued expression marks the binding
//!   float — once float, always float (monotone two-level lattice, so
//!   the fixpoint loop converges in ≤ depth-of-chained-copies rounds;
//!   a hard cap keeps it total).
//! - Calls are conservatively non-float (under-count; cross-function
//!   return-type propagation is future work).
//!
//! Under-counting is the safe direction for a *density* signal: it can
//! only make hot-FP functions look cooler, never invent heat that
//! would withhold passes from cold code.

use std::collections::BTreeSet;

use cjc_ast::BinOp;
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFunction, MirStmt};

use crate::hash::CanaHasher;

/// Fixpoint-round cap for the float-binding propagation. The lattice
/// is monotone (bindings only ever become float), so each round either
/// grows the set or terminates; the cap is a totality guard for
/// pathological copy chains, not a tuning knob.
const PROPAGATION_ROUND_CAP: usize = 8;

/// Float/int operation mix for one MIR function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TypeMix {
    /// `Binary` nodes (any operator) where the conservative propagation
    /// marks at least one operand float. Static analog of the runtime
    /// FP-binop trace counter.
    pub float_binop_count: u32,
    /// All `Binary` expression nodes walked.
    pub binop_count: u32,
    /// Parameters annotated `f64`.
    pub float_param_count: u32,
}

impl TypeMix {
    /// Analyze one function: propagate float bindings to fixpoint, then
    /// count binary ops by operand type.
    pub fn from_function(func: &MirFunction) -> Self {
        let mut float_vars: BTreeSet<String> = BTreeSet::new();
        let mut float_param_count: u32 = 0;
        for p in &func.params {
            if p.ty_name == "f64" {
                float_vars.insert(p.name.clone());
                float_param_count = float_param_count.saturating_add(1);
            }
        }

        // Fixpoint: each round walks the whole body, marking bindings
        // whose initializer/assigned value is float under the current
        // set. Monotone — stop as soon as a round adds nothing.
        for _ in 0..PROPAGATION_ROUND_CAP {
            let before = float_vars.len();
            propagate_body(&func.body, &mut float_vars);
            if float_vars.len() == before {
                break;
            }
        }

        let mut mix = TypeMix {
            float_binop_count: 0,
            binop_count: 0,
            float_param_count,
        };
        count_body(&func.body, &float_vars, &mut mix);
        mix
    }

    /// Feed into a streaming hasher (joins `FnFeatures::feed`).
    pub(crate) fn feed(&self, hasher: &mut CanaHasher) {
        hasher.write_tag(TAG_TYPE_MIX);
        hasher.write_u32(self.float_binop_count);
        hasher.write_u32(self.binop_count);
        hasher.write_u32(self.float_param_count);
    }

    /// FP-op density in `[0, 1]`: float binops over all binops. The
    /// per-function static proxy for the recorded `thermal_intensity`.
    pub fn float_density(&self) -> f64 {
        if self.binop_count == 0 {
            return 0.0;
        }
        self.float_binop_count as f64 / self.binop_count as f64
    }
}

/// Discriminator tag for the type-mix feed stream (see
/// `memory_proxy.rs` for the tag-space convention: 0xA0 memory, 0xB0
/// reductions, 0xE0/0xF0 program/feature headers).
const TAG_TYPE_MIX: u8 = 0xC0;

// ---------------------------------------------------------------------------
// Pass 1 — float-binding propagation
// ---------------------------------------------------------------------------

fn propagate_body(body: &MirBody, floats: &mut BTreeSet<String>) {
    for stmt in &body.stmts {
        propagate_stmt(stmt, floats);
    }
    if let Some(result) = &body.result {
        propagate_expr(result, floats);
    }
}

fn propagate_stmt(stmt: &MirStmt, floats: &mut BTreeSet<String>) {
    match stmt {
        MirStmt::Let { name, init, .. } => {
            propagate_expr(init, floats);
            if expr_is_float(init, floats) {
                floats.insert(name.clone());
            }
        }
        MirStmt::Expr(e) => propagate_expr(e, floats),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            propagate_expr(cond, floats);
            propagate_body(then_body, floats);
            if let Some(eb) = else_body {
                propagate_body(eb, floats);
            }
        }
        MirStmt::While { cond, body } => {
            propagate_expr(cond, floats);
            propagate_body(body, floats);
        }
        MirStmt::Return(Some(e)) => propagate_expr(e, floats),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(b) => propagate_body(b, floats),
    }
}

fn propagate_expr(expr: &MirExpr, floats: &mut BTreeSet<String>) {
    if let MirExprKind::Assign { target, value } = &expr.kind {
        propagate_expr(value, floats);
        if expr_is_float(value, floats) {
            match &target.kind {
                MirExprKind::Var(name) | MirExprKind::VarLocal { name, .. } => {
                    floats.insert(name.clone());
                }
                _ => {}
            }
        }
        return;
    }
    for child in expr_children(expr) {
        propagate_expr(child, floats);
    }
    for body in expr_bodies(expr) {
        propagate_body(body, floats);
    }
}

// ---------------------------------------------------------------------------
// Pass 2 — binop counting under the converged float set
// ---------------------------------------------------------------------------

fn count_body(body: &MirBody, floats: &BTreeSet<String>, mix: &mut TypeMix) {
    for stmt in &body.stmts {
        count_stmt(stmt, floats, mix);
    }
    if let Some(result) = &body.result {
        count_expr(result, floats, mix);
    }
}

fn count_stmt(stmt: &MirStmt, floats: &BTreeSet<String>, mix: &mut TypeMix) {
    match stmt {
        MirStmt::Let { init, .. } => count_expr(init, floats, mix),
        MirStmt::Expr(e) => count_expr(e, floats, mix),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            count_expr(cond, floats, mix);
            count_body(then_body, floats, mix);
            if let Some(eb) = else_body {
                count_body(eb, floats, mix);
            }
        }
        MirStmt::While { cond, body } => {
            count_expr(cond, floats, mix);
            count_body(body, floats, mix);
        }
        MirStmt::Return(Some(e)) => count_expr(e, floats, mix),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(b) => count_body(b, floats, mix),
    }
}

fn count_expr(expr: &MirExpr, floats: &BTreeSet<String>, mix: &mut TypeMix) {
    if let MirExprKind::Binary { left, right, .. } = &expr.kind {
        mix.binop_count = mix.binop_count.saturating_add(1);
        // Runtime-counter semantics: ANY operator, either operand float.
        if expr_is_float(left, floats) || expr_is_float(right, floats) {
            mix.float_binop_count = mix.float_binop_count.saturating_add(1);
        }
    }
    for child in expr_children(expr) {
        count_expr(child, floats, mix);
    }
    for body in expr_bodies(expr) {
        count_body(body, floats, mix);
    }
}

// ---------------------------------------------------------------------------
// Conservative value typing
// ---------------------------------------------------------------------------

/// Is the VALUE of this expression float under the current binding set?
fn expr_is_float(expr: &MirExpr, floats: &BTreeSet<String>) -> bool {
    match &expr.kind {
        MirExprKind::FloatLit(_) => true,
        MirExprKind::Var(name) | MirExprKind::VarLocal { name, .. } => floats.contains(name),
        MirExprKind::Binary { op, left, right } => {
            arith_op(op) && (expr_is_float(left, floats) || expr_is_float(right, floats))
        }
        MirExprKind::Unary { operand, .. } => expr_is_float(operand, floats),
        MirExprKind::If {
            then_body,
            else_body,
            ..
        } => {
            body_result_is_float(then_body, floats)
                || else_body
                    .as_ref()
                    .map(|b| body_result_is_float(b, floats))
                    .unwrap_or(false)
        }
        MirExprKind::Block(body) => body_result_is_float(body, floats),
        MirExprKind::Assign { value, .. } => expr_is_float(value, floats),
        // Calls, indexing, fields, literals of other types: conservatively
        // non-float (see module docs — under-counting is the safe
        // direction for a density signal).
        _ => false,
    }
}

fn body_result_is_float(body: &MirBody, floats: &BTreeSet<String>) -> bool {
    body.result
        .as_ref()
        .map(|e| expr_is_float(e, floats))
        .unwrap_or(false)
}

/// Float-producing operators. Comparisons/logic produce `Bool` and are
/// excluded HERE — but they still count as FP ops in `count_expr` when
/// an operand is float, matching the runtime counter.
fn arith_op(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow
    )
}

// ---------------------------------------------------------------------------
// Child enumeration (kept exhaustive-by-construction: any new
// MirExprKind with children must be added here; the catch-all returns
// no children, which under-counts rather than panics)
// ---------------------------------------------------------------------------

fn expr_children(expr: &MirExpr) -> Vec<&MirExpr> {
    match &expr.kind {
        MirExprKind::Binary { left, right, .. } => vec![left, right],
        MirExprKind::Unary { operand, .. } => vec![operand],
        MirExprKind::Call { callee, args } => {
            let mut v: Vec<&MirExpr> = vec![callee];
            v.extend(args.iter());
            v
        }
        MirExprKind::Field { object, .. } => vec![object],
        MirExprKind::Index { object, index } => vec![object, index],
        MirExprKind::MultiIndex { object, indices } => {
            let mut v: Vec<&MirExpr> = vec![object];
            v.extend(indices.iter());
            v
        }
        MirExprKind::Assign { target, value } => vec![target, value],
        MirExprKind::TensorLit { rows } => rows.iter().flatten().collect(),
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => elems.iter().collect(),
        MirExprKind::StructLit { fields, .. } => fields.iter().map(|(_, e)| e).collect(),
        MirExprKind::VariantLit { fields, .. } => fields.iter().collect(),
        MirExprKind::MakeClosure { captures, .. } => captures.iter().collect(),
        MirExprKind::Lambda { body, .. } => vec![body],
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => vec![operand],
        MirExprKind::Broadcast {
            operand,
            target_shape,
        } => {
            let mut v: Vec<&MirExpr> = vec![operand];
            v.extend(target_shape.iter());
            v
        }
        MirExprKind::Match { scrutinee, .. } => vec![scrutinee],
        // If-expressions: the CONDITION is an expr child; the branch
        // bodies are enumerated by `expr_bodies`. Forgetting the cond
        // here silently skips every `x > 0.5`-style comparison (caught
        // by `float_comparison_counts_as_fp_op`).
        MirExprKind::If { cond, .. } => vec![cond],
        _ => vec![],
    }
}

fn expr_bodies(expr: &MirExpr) -> Vec<&MirBody> {
    match &expr.kind {
        MirExprKind::Block(body) => vec![body],
        MirExprKind::If {
            then_body,
            else_body,
            ..
        } => {
            let mut v = vec![then_body];
            if let Some(eb) = else_body {
                v.push(eb);
            }
            v
        }
        MirExprKind::Match { arms, .. } => arms.iter().map(|a| &a.body).collect(),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Lower source → MIR and analyze one function by name.
    fn mix_of(src: &str, fn_name: &str) -> TypeMix {
        let (ast, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors(), "parse: {:?}", diags.diagnostics);
        let mut al = cjc_hir::AstLowering::new();
        let hir = al.lower_program(&ast);
        let mut h2m = cjc_mir::HirToMir::new();
        let mir = h2m.lower_program(&hir);
        let func = mir
            .functions
            .iter()
            .find(|f| f.name == fn_name)
            .unwrap_or_else(|| panic!("function {fn_name} not found"));
        TypeMix::from_function(func)
    }

    #[test]
    fn int_only_function_has_zero_float_ops() {
        let mix = mix_of(
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
print(sum_to(10));
"#,
            "sum_to",
        );
        assert_eq!(mix.float_binop_count, 0);
        assert!(mix.binop_count > 0);
        assert_eq!(mix.float_param_count, 0);
        assert_eq!(mix.float_density(), 0.0);
    }

    #[test]
    fn float_param_seeds_propagation() {
        let mix = mix_of(
            r#"
fn poly(x: f64) -> f64 {
    let a: f64 = 3.0;
    return a * x * x + a;
}
print(poly(1.5));
"#,
            "poly",
        );
        // a*x, (a*x)*x, +a — all float.
        assert_eq!(mix.float_param_count, 1);
        assert_eq!(mix.binop_count, 3);
        assert_eq!(mix.float_binop_count, 3);
        assert_eq!(mix.float_density(), 1.0);
    }

    #[test]
    fn loop_carried_float_converges_via_fixpoint() {
        // facc is float only because its initializer is FloatLit; the
        // accumulation inside the loop must be counted float.
        let mix = mix_of(
            r#"
fn work(n: i64) -> i64 {
    let mut facc: f64 = 0.0;
    let mut iacc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        facc = facc + 0.5;
        iacc = iacc + i * 3;
        i = i + 1;
    }
    print(facc);
    return iacc;
}
print(work(4));
"#,
            "work",
        );
        // Float: facc + 0.5 (1). Int: i < n, iacc + i*3 (2), i + 1.
        assert_eq!(mix.float_binop_count, 1);
        assert_eq!(mix.binop_count, 5);
    }

    #[test]
    fn float_comparison_counts_as_fp_op() {
        // Runtime counter increments on (Float, Float) comparisons too.
        let mix = mix_of(
            r#"
fn check(x: f64) -> i64 {
    let r: i64 = if x > 0.5 { 1 } else { 0 };
    return r;
}
print(check(0.7));
"#,
            "check",
        );
        assert_eq!(mix.float_binop_count, 1);
        assert_eq!(mix.binop_count, 1);
    }

    #[test]
    fn mixed_int_float_binop_is_float() {
        // i * 0.001 — int×float promotes; the runtime counts it.
        let mix = mix_of(
            r#"
fn scale(n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i * 0.001;
        i = i + 1;
    }
    return acc;
}
print(scale(3));
"#,
            "scale",
        );
        // Float: i * 0.001, acc + (...). Int: i < n, i + 1.
        assert_eq!(mix.float_binop_count, 2);
        assert_eq!(mix.binop_count, 4);
    }

    #[test]
    fn density_is_ratio_of_counts() {
        let mix = TypeMix {
            float_binop_count: 3,
            binop_count: 12,
            float_param_count: 0,
        };
        assert!((mix.float_density() - 0.25).abs() < 1e-15);
        assert_eq!(TypeMix::default().float_density(), 0.0);
    }
}
