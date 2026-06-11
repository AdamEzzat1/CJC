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
//!
//! ## Tensor tracking (Phase A1 fix)
//!
//! The A1 probe (`bench/cana_tensor_probe`) showed both instruments
//! were tensor-blind: a `Tensor + Tensor` binop is statically just a
//! `Binary` node and dynamically one statement, yet it executes
//! `len()` hardware FP operations. The runtime counter now prices
//! tensor ops by element count; this module mirrors it with a second
//! propagated set:
//!
//! - Seed: parameters annotated `Tensor`.
//! - `TensorLit`, `Broadcast`, `LinalgInv` are tensors. Calls to
//!   tensor-returning builtins (`matmul`, `transpose`, `zeros`, ...),
//!   `Tensor.<constructor>(...)` static methods, and
//!   tensor-to-tensor methods (`t.add(u)`, `t.reshape(...)`, ...)
//!   are tensors.
//! - Arithmetic with a tensor operand produces a tensor (broadcast
//!   promotion), and counts as [`TypeMix::tensor_binop_count`] — NOT
//!   as a scalar float binop, mirroring the runtime dispatch arms
//!   which skip the scalar counter for tensor pairs.
//!
//! The element-count scale lives in `physical_cost.rs`
//! (`TENSOR_FP_PER_OP`), not here: TypeMix stays a pure op-count
//! mirror of the runtime sites.

use std::collections::BTreeSet;

use cjc_ast::BinOp;
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFunction, MirStmt};

use crate::hash::CanaHasher;

/// Fixpoint-round cap for the float-binding propagation. The lattice
/// is monotone (bindings only ever become float), so each round either
/// grows the set or terminates; the cap is a totality guard for
/// pathological copy chains, not a tuning knob.
const PROPAGATION_ROUND_CAP: usize = 8;

/// Float/int/tensor operation mix for one MIR function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TypeMix {
    /// `Binary` nodes (any operator) where the conservative propagation
    /// marks at least one operand float AND no operand tensor. Static
    /// analog of the runtime SCALAR FP-binop trace counter (the tensor
    /// arms skip that counter, so tensor-involving binops are excluded
    /// here too).
    pub float_binop_count: u32,
    /// All `Binary` expression nodes walked.
    pub binop_count: u32,
    /// Parameters annotated `f64`.
    pub float_param_count: u32,
    /// `Binary` nodes where at least one operand is a tensor
    /// (element-wise arithmetic / scalar broadcast). Static analog of
    /// the runtime tensor-binop accounting (Phase A1 fix); priced at
    /// `TENSOR_FP_PER_OP` elements each by `build_physical_query`.
    pub tensor_binop_count: u32,
}

/// The two monotone binding sets the propagation converges.
#[derive(Debug, Default)]
struct Bindings {
    floats: BTreeSet<String>,
    tensors: BTreeSet<String>,
}

impl Bindings {
    fn len(&self) -> usize {
        self.floats.len() + self.tensors.len()
    }
}

impl TypeMix {
    /// Analyze one function: propagate float + tensor bindings to
    /// fixpoint, then count binary ops by operand type.
    pub fn from_function(func: &MirFunction) -> Self {
        let mut bindings = Bindings::default();
        let mut float_param_count: u32 = 0;
        for p in &func.params {
            if p.ty_name == "f64" {
                bindings.floats.insert(p.name.clone());
                float_param_count = float_param_count.saturating_add(1);
            }
            if p.ty_name == "Tensor" {
                bindings.tensors.insert(p.name.clone());
            }
        }

        // Fixpoint: each round walks the whole body, marking bindings
        // whose initializer/assigned value is float/tensor under the
        // current sets. Monotone — stop as soon as a round adds nothing.
        for _ in 0..PROPAGATION_ROUND_CAP {
            let before = bindings.len();
            propagate_body(&func.body, &mut bindings);
            if bindings.len() == before {
                break;
            }
        }

        let mut mix = TypeMix {
            float_binop_count: 0,
            binop_count: 0,
            float_param_count,
            tensor_binop_count: 0,
        };
        count_body(&func.body, &bindings, &mut mix);
        mix
    }

    /// Feed into a streaming hasher (joins `FnFeatures::feed`).
    pub(crate) fn feed(&self, hasher: &mut CanaHasher) {
        hasher.write_tag(TAG_TYPE_MIX);
        hasher.write_u32(self.float_binop_count);
        hasher.write_u32(self.binop_count);
        hasher.write_u32(self.float_param_count);
        hasher.write_u32(self.tensor_binop_count);
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
// Pass 1 — float/tensor-binding propagation
// ---------------------------------------------------------------------------

fn propagate_body(body: &MirBody, b: &mut Bindings) {
    for stmt in &body.stmts {
        propagate_stmt(stmt, b);
    }
    if let Some(result) = &body.result {
        propagate_expr(result, b);
    }
}

fn mark_binding(name: &str, value: &MirExpr, b: &mut Bindings) {
    // Tensor takes precedence: `2.0 * t` is a tensor, not a float.
    if expr_is_tensor(value, b) {
        b.tensors.insert(name.to_string());
    } else if expr_is_float(value, b) {
        b.floats.insert(name.to_string());
    }
}

fn propagate_stmt(stmt: &MirStmt, b: &mut Bindings) {
    match stmt {
        MirStmt::Let { name, init, .. } => {
            propagate_expr(init, b);
            mark_binding(name, init, b);
        }
        MirStmt::Expr(e) => propagate_expr(e, b),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            propagate_expr(cond, b);
            propagate_body(then_body, b);
            if let Some(eb) = else_body {
                propagate_body(eb, b);
            }
        }
        MirStmt::While { cond, body } => {
            propagate_expr(cond, b);
            propagate_body(body, b);
        }
        MirStmt::Return(Some(e)) => propagate_expr(e, b),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => propagate_body(body, b),
    }
}

fn propagate_expr(expr: &MirExpr, b: &mut Bindings) {
    if let MirExprKind::Assign { target, value } = &expr.kind {
        propagate_expr(value, b);
        match &target.kind {
            MirExprKind::Var(name) | MirExprKind::VarLocal { name, .. } => {
                mark_binding(name, value, b);
            }
            _ => {}
        }
        return;
    }
    for child in expr_children(expr) {
        propagate_expr(child, b);
    }
    for body in expr_bodies(expr) {
        propagate_body(body, b);
    }
}

// ---------------------------------------------------------------------------
// Pass 2 — binop counting under the converged binding sets
// ---------------------------------------------------------------------------

fn count_body(body: &MirBody, b: &Bindings, mix: &mut TypeMix) {
    for stmt in &body.stmts {
        count_stmt(stmt, b, mix);
    }
    if let Some(result) = &body.result {
        count_expr(result, b, mix);
    }
}

fn count_stmt(stmt: &MirStmt, b: &Bindings, mix: &mut TypeMix) {
    match stmt {
        MirStmt::Let { init, .. } => count_expr(init, b, mix),
        MirStmt::Expr(e) => count_expr(e, b, mix),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            count_expr(cond, b, mix);
            count_body(then_body, b, mix);
            if let Some(eb) = else_body {
                count_body(eb, b, mix);
            }
        }
        MirStmt::While { cond, body } => {
            count_expr(cond, b, mix);
            count_body(body, b, mix);
        }
        MirStmt::Return(Some(e)) => count_expr(e, b, mix),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(body) => count_body(body, b, mix),
    }
}

fn count_expr(expr: &MirExpr, b: &Bindings, mix: &mut TypeMix) {
    if let MirExprKind::Binary { left, right, .. } = &expr.kind {
        mix.binop_count = mix.binop_count.saturating_add(1);
        // Runtime-counter semantics: tensor arms skip the scalar
        // counter, so the tensor check takes precedence; otherwise ANY
        // operator with either operand float counts as scalar FP.
        if expr_is_tensor(left, b) || expr_is_tensor(right, b) {
            mix.tensor_binop_count = mix.tensor_binop_count.saturating_add(1);
        } else if expr_is_float(left, b) || expr_is_float(right, b) {
            mix.float_binop_count = mix.float_binop_count.saturating_add(1);
        }
    }
    for child in expr_children(expr) {
        count_expr(child, b, mix);
    }
    for body in expr_bodies(expr) {
        count_body(body, b, mix);
    }
}

// ---------------------------------------------------------------------------
// Conservative value typing
// ---------------------------------------------------------------------------

/// Free-call builtins that return tensors. Mirrors the runtime's
/// tensor-returning dispatch surface (curated; missing one under-counts).
const TENSOR_RESULT_BUILTINS: &[&str] = &[
    "matmul",
    "transpose",
    "tensor_new",
    "tensor_zeros",
    "tensor_ones",
    "zeros",
    "ones",
    "tensor_concat_1d",
];

/// `Tensor.<name>(...)` static constructors.
const TENSOR_STATIC_CONSTRUCTORS: &[&str] = &["from_vec", "zeros", "ones", "randn", "eye", "new"];

/// Methods on a tensor receiver that return tensors (`t.add(u)`,
/// `t.reshape(...)`, ...). Reductions (`sum`, `mean`) return floats and
/// are deliberately absent — their result would need cross-call
/// propagation to reach the float set (conservative under-count today).
const TENSOR_TO_TENSOR_METHODS: &[&str] = &[
    "add",
    "sub",
    "transpose",
    "reshape",
    "abs",
    "relu",
    "gelu",
    "softmax",
];

/// Is the VALUE of this expression float under the current binding sets?
/// Tensor-ness dominates: `2.0 * t` broadcasts to a tensor, so it is
/// NOT a scalar float (mirrors the runtime dispatch arm precedence).
fn expr_is_float(expr: &MirExpr, b: &Bindings) -> bool {
    match &expr.kind {
        MirExprKind::FloatLit(_) => true,
        MirExprKind::Var(name) | MirExprKind::VarLocal { name, .. } => {
            b.floats.contains(name) && !b.tensors.contains(name)
        }
        MirExprKind::Binary { op, left, right } => {
            arith_op(op)
                && (expr_is_float(left, b) || expr_is_float(right, b))
                && !(expr_is_tensor(left, b) || expr_is_tensor(right, b))
        }
        MirExprKind::Unary { operand, .. } => {
            expr_is_float(operand, b) && !expr_is_tensor(operand, b)
        }
        MirExprKind::If {
            then_body,
            else_body,
            ..
        } => {
            body_result_is(then_body, b, expr_is_float)
                || else_body
                    .as_ref()
                    .map(|eb| body_result_is(eb, b, expr_is_float))
                    .unwrap_or(false)
        }
        MirExprKind::Block(body) => body_result_is(body, b, expr_is_float),
        MirExprKind::Assign { value, .. } => expr_is_float(value, b),
        // Calls, indexing, fields, literals of other types: conservatively
        // non-float (see module docs — under-counting is the safe
        // direction for a density signal).
        _ => false,
    }
}

/// Is the VALUE of this expression a tensor under the current binding
/// sets? (Phase A1 fix — see module docs "Tensor tracking".)
fn expr_is_tensor(expr: &MirExpr, b: &Bindings) -> bool {
    match &expr.kind {
        MirExprKind::TensorLit { .. } => true,
        MirExprKind::Broadcast { .. } => true,
        MirExprKind::LinalgInv { .. } => true,
        // LU/QR/Cholesky return tuples of tensors — tracking those
        // requires tuple-element typing; conservatively non-tensor.
        MirExprKind::Var(name) | MirExprKind::VarLocal { name, .. } => b.tensors.contains(name),
        MirExprKind::Binary { op, left, right } => {
            arith_op(op) && (expr_is_tensor(left, b) || expr_is_tensor(right, b))
        }
        MirExprKind::Unary { operand, .. } => expr_is_tensor(operand, b),
        MirExprKind::If {
            then_body,
            else_body,
            ..
        } => {
            body_result_is(then_body, b, expr_is_tensor)
                || else_body
                    .as_ref()
                    .map(|eb| body_result_is(eb, b, expr_is_tensor))
                    .unwrap_or(false)
        }
        MirExprKind::Block(body) => body_result_is(body, b, expr_is_tensor),
        MirExprKind::Assign { value, .. } => expr_is_tensor(value, b),
        MirExprKind::Call { callee, .. } => match &callee.kind {
            MirExprKind::Var(name) => TENSOR_RESULT_BUILTINS.contains(&name.as_str()),
            MirExprKind::Field { object, name } => {
                // `Tensor.from_vec(...)` static constructor…
                if let MirExprKind::Var(obj) = &object.kind {
                    if obj == "Tensor" && TENSOR_STATIC_CONSTRUCTORS.contains(&name.as_str()) {
                        return true;
                    }
                }
                // …or a tensor-to-tensor method on a tensor receiver.
                expr_is_tensor(object, b) && TENSOR_TO_TENSOR_METHODS.contains(&name.as_str())
            }
            _ => false,
        },
        _ => false,
    }
}

fn body_result_is(body: &MirBody, b: &Bindings, pred: fn(&MirExpr, &Bindings) -> bool) -> bool {
    body.result.as_ref().map(|e| pred(e, b)).unwrap_or(false)
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
            tensor_binop_count: 0,
        };
        assert!((mix.float_density() - 0.25).abs() < 1e-15);
        assert_eq!(TypeMix::default().float_density(), 0.0);
    }

    // -- Tensor tracking (Phase A1 fix) -----------------------------------

    #[test]
    fn tensor_param_seeds_tensor_binops() {
        // a + b and a * b on Tensor params are tensor binops, NOT
        // scalar float binops (mirrors the runtime arm precedence).
        let mix = mix_of(
            r#"
fn ew(a: Tensor, b: Tensor) -> Tensor {
    let c: Tensor = a + b;
    let d: Tensor = a * b;
    return c + d;
}
let t: Tensor = Tensor.from_vec([1.0, 2.0], [2]);
let u: Tensor = Tensor.from_vec([3.0, 4.0], [2]);
print(1);
"#,
            "ew",
        );
        assert_eq!(mix.tensor_binop_count, 3);
        assert_eq!(mix.float_binop_count, 0);
        assert_eq!(mix.binop_count, 3);
    }

    #[test]
    fn scalar_broadcast_is_tensor_not_float() {
        // 2.0 * t broadcasts: the float literal must NOT make this a
        // scalar float binop.
        let mix = mix_of(
            r#"
fn scale(t: Tensor) -> Tensor {
    return 2.0 * t + 1.0;
}
print(1);
"#,
            "scale",
        );
        // (2.0 * t) is tensor; (... + 1.0) is tensor + float → tensor.
        assert_eq!(mix.tensor_binop_count, 2);
        assert_eq!(mix.float_binop_count, 0);
    }

    #[test]
    fn tensor_constructor_call_propagates() {
        // Tensor.from_vec result assigned to a local; subsequent binop
        // on the local is a tensor binop only via propagation.
        let mix = mix_of(
            r#"
fn build(n: i64) -> Tensor {
    let mut buf: Any = [];
    let mut i: i64 = 0;
    while i < n {
        buf = array_push(buf, 0.5);
        i = i + 1;
    }
    let t: Tensor = Tensor.from_vec(buf, [n]);
    let u: Tensor = t + t;
    return u;
}
print(1);
"#,
            "build",
        );
        assert_eq!(mix.tensor_binop_count, 1);
        // i < n, i + 1 are int; no scalar float binops.
        assert_eq!(mix.float_binop_count, 0);
    }

    #[test]
    fn matmul_free_call_result_is_tensor() {
        let mix = mix_of(
            r#"
fn mm(a: Tensor, b: Tensor) -> Tensor {
    let c: Tensor = matmul(a, b);
    return c + a;
}
print(1);
"#,
            "mm",
        );
        assert_eq!(mix.tensor_binop_count, 1);
        assert_eq!(mix.float_binop_count, 0);
    }

    #[test]
    fn scalar_float_counting_unchanged_by_tensor_tracking() {
        // Regression: a function with BOTH scalar FP and tensor work
        // keeps the scalar count the runtime scalar counter would see.
        let mix = mix_of(
            r#"
fn mixed(t: Tensor, n: i64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i64 = 0;
    while i < n {
        let u: Tensor = t * 2.0;
        acc = acc + 0.001;
        i = i + 1;
    }
    return acc;
}
print(1);
"#,
            "mixed",
        );
        assert_eq!(mix.tensor_binop_count, 1); // t * 2.0
        assert_eq!(mix.float_binop_count, 1); // acc + 0.001
    }
}
