//! Per-function memory-pressure *proxies* derived from MIR shape.
//!
//! These are NOT measurements — Phase 1 cannot run the program. They are
//! deterministic *counts of sites likely to cause* allocation, COW writes, or
//! tensor materialization. Phase 5 (profile-guided runtime feedback) is what
//! replaces proxies with measured byte/allocation counts.
//!
//! ## Three honest signals at MIR level
//!
//! | Signal           | What we count                                       |
//! |------------------|-----------------------------------------------------|
//! | `alloc_sites`    | Literals + alloc-like builtin calls                 |
//! | `cow_write_sites`| Calls that use `Rc::make_mut` semantics             |
//! | `tensor_heavy_ops`| Linalg opcodes + tensor matmul/reduction builtins  |
//!
//! Phase 4 (NSS integration) will project these onto NSS's
//! `PressureKind::Memory`, `Cpu`, and `Throughput` fields respectively.

use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFunction, MirStmt};

use crate::hash::CanaHasher;

/// Memory-pressure proxy counts for one MIR function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryProxy {
    /// Literal and constructor sites: `TensorLit`, `ArrayLit`, `StructLit`,
    /// `TupleLit`, `VariantLit`, `MakeClosure`, `StringLit` (with non-trivial
    /// content), plus calls to alloc-like builtins.
    pub alloc_sites: u32,
    /// Calls that write to a COW-shared buffer and therefore trigger
    /// `Rc::make_mut` semantics in the runtime: `array_push`, `array_pop`,
    /// `array_reverse`, `arr_set`, etc.
    pub cow_write_sites: u32,
    /// Tensor- or linalg-heavy operations: `LinalgLU`, `LinalgQR`,
    /// `LinalgCholesky`, `LinalgInv`, `Broadcast`, plus calls to matmul,
    /// transpose, dot, sum, mean, etc.
    pub tensor_heavy_ops: u32,
    /// Total expression-node count walked. Useful as a normalization
    /// denominator and as a sanity check that the walker visited everything.
    pub expr_count: u32,
    /// Phase F1: total ELEMENT SLOTS across array/tuple literals — the
    /// static mirror of the runtime's creation-site allocation prices.
    /// `alloc_sites` counts a 2-element and a 774-element literal the
    /// same (1 site each); the recorded Phase-F label prices them 16 B
    /// per ELEMENT, so the per-site count carries no volume signal —
    /// the measured cause of the memory head's R²(test) 0.048
    /// generalization failure. Literal element counts are statically
    /// EXACT; array/tuple only, mirroring exactly which runtime hooks
    /// price per-element (TensorLit/StructLit price 0 at runtime too).
    pub lit_elem_slots: u32,
}

impl MemoryProxy {
    /// Walk a function's body and accumulate proxy counts.
    pub fn from_function(func: &MirFunction) -> Self {
        let mut p = Self {
            alloc_sites: 0,
            cow_write_sites: 0,
            tensor_heavy_ops: 0,
            expr_count: 0,
            lit_elem_slots: 0,
        };
        p.walk_body(&func.body);
        p
    }

    /// Feed into a streaming hasher.
    pub(crate) fn feed(&self, hasher: &mut CanaHasher) {
        hasher.write_tag(TAG_MEMORY_PROXY);
        hasher.write_u32(self.alloc_sites);
        hasher.write_u32(self.cow_write_sites);
        hasher.write_u32(self.tensor_heavy_ops);
        hasher.write_u32(self.expr_count);
        // Phase F1 — a content-addressed fingerprint doing its job:
        // the features genuinely changed, so every FeatureHash changes.
        hasher.write_u32(self.lit_elem_slots);
    }

    fn walk_body(&mut self, body: &MirBody) {
        for stmt in &body.stmts {
            self.walk_stmt(stmt);
        }
        if let Some(result) = &body.result {
            self.walk_expr(result);
        }
    }

    fn walk_stmt(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Let { init, .. } => self.walk_expr(init),
            MirStmt::Expr(e) => self.walk_expr(e),
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.walk_expr(cond);
                self.walk_body(then_body);
                if let Some(eb) = else_body {
                    self.walk_body(eb);
                }
            }
            MirStmt::While { cond, body } => {
                self.walk_expr(cond);
                self.walk_body(body);
            }
            MirStmt::Return(opt) => {
                if let Some(e) = opt {
                    self.walk_expr(e);
                }
            }
            MirStmt::Break | MirStmt::Continue => {}
            MirStmt::NoGcBlock(b) => self.walk_body(b),
        }
    }

    fn walk_expr(&mut self, expr: &MirExpr) {
        self.expr_count = self.expr_count.saturating_add(1);
        match &expr.kind {
            // ----- Literals & constructors that allocate -----
            MirExprKind::TensorLit { rows } => {
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                for row in rows {
                    for cell in row {
                        self.walk_expr(cell);
                    }
                }
            }
            MirExprKind::ArrayLit(elems) => {
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                self.lit_elem_slots = self.lit_elem_slots.saturating_add(elems.len() as u32);
                for e in elems {
                    self.walk_expr(e);
                }
            }
            MirExprKind::StructLit { fields, .. } => {
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                for (_, e) in fields {
                    self.walk_expr(e);
                }
            }
            MirExprKind::TupleLit(elems) => {
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                self.lit_elem_slots = self.lit_elem_slots.saturating_add(elems.len() as u32);
                for e in elems {
                    self.walk_expr(e);
                }
            }
            MirExprKind::VariantLit { fields, .. } => {
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                for e in fields {
                    self.walk_expr(e);
                }
            }
            MirExprKind::MakeClosure { captures, .. } => {
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                for e in captures {
                    self.walk_expr(e);
                }
            }
            MirExprKind::StringLit(s) | MirExprKind::RawStringLit(s) => {
                if !s.is_empty() {
                    self.alloc_sites = self.alloc_sites.saturating_add(1);
                }
            }
            MirExprKind::ByteStringLit(b) | MirExprKind::RawByteStringLit(b) => {
                if !b.is_empty() {
                    self.alloc_sites = self.alloc_sites.saturating_add(1);
                }
            }

            // ----- Linalg opcodes — tensor-heavy, often allocate -----
            MirExprKind::LinalgLU { operand }
            | MirExprKind::LinalgQR { operand }
            | MirExprKind::LinalgCholesky { operand }
            | MirExprKind::LinalgInv { operand } => {
                self.tensor_heavy_ops = self.tensor_heavy_ops.saturating_add(1);
                self.alloc_sites = self.alloc_sites.saturating_add(1);
                self.walk_expr(operand);
            }
            MirExprKind::Broadcast {
                operand,
                target_shape,
            } => {
                self.tensor_heavy_ops = self.tensor_heavy_ops.saturating_add(1);
                self.walk_expr(operand);
                for s in target_shape {
                    self.walk_expr(s);
                }
            }

            // ----- Calls — classify by callee name (free call) or by
            // method name (Field callee). The method form was invisible
            // pre-Phase-A1: `t.sum()` performed the same tensor work as
            // `sum(t)` but counted zero tensor_heavy_ops (found by the
            // tensor-blindness probe).
            MirExprKind::Call { callee, args } => {
                match &callee.kind {
                    MirExprKind::Var(name) => classify_builtin_call(name, self),
                    MirExprKind::Field { object, name } => {
                        classify_method_call(object, name, self)
                    }
                    _ => {}
                }
                self.walk_expr(callee);
                for a in args {
                    self.walk_expr(a);
                }
            }

            // ----- Pure structural recursion (no allocation flags) -----
            MirExprKind::Binary { left, right, .. } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            MirExprKind::Unary { operand, .. } => self.walk_expr(operand),
            MirExprKind::Field { object, .. } => self.walk_expr(object),
            MirExprKind::Index { object, index } => {
                self.walk_expr(object);
                self.walk_expr(index);
            }
            MirExprKind::MultiIndex { object, indices } => {
                self.walk_expr(object);
                for i in indices {
                    self.walk_expr(i);
                }
            }
            MirExprKind::Assign { target, value } => {
                self.walk_expr(target);
                self.walk_expr(value);
            }
            MirExprKind::Block(body) => self.walk_body(body),
            MirExprKind::If {
                cond,
                then_body,
                else_body,
            } => {
                self.walk_expr(cond);
                self.walk_body(then_body);
                if let Some(eb) = else_body {
                    self.walk_body(eb);
                }
            }
            MirExprKind::Match { scrutinee, arms } => {
                self.walk_expr(scrutinee);
                for arm in arms {
                    self.walk_body(&arm.body);
                }
            }
            MirExprKind::Lambda { body, .. } => {
                // Lambda expression: walking the body is informative for
                // shape but the *site* itself isn't an allocation — the
                // MakeClosure that wraps it is.
                self.walk_expr(body);
            }

            // ----- Leaves with no children, no allocation -----
            MirExprKind::IntLit(_)
            | MirExprKind::FloatLit(_)
            | MirExprKind::BoolLit(_)
            | MirExprKind::ByteCharLit(_)
            | MirExprKind::NaLit
            | MirExprKind::Var(_)
            | MirExprKind::VarLocal { .. }
            | MirExprKind::Col(_)
            | MirExprKind::Void => {}
            // Empty-string literals fall into this category — already
            // skipped in their arms above.
            MirExprKind::RegexLit { .. } => {
                // Compiled regex objects allocate; count one site.
                self.alloc_sites = self.alloc_sites.saturating_add(1);
            }
        }
    }
}

/// Discriminator tag for the memory-proxy feed stream. Distinct tags per
/// component prevent collisions when the per-component feeds are concatenated
/// into the per-function feature hash.
const TAG_MEMORY_PROXY: u8 = 0xA0;

// ---------------------------------------------------------------------------
// Builtin classification — the wiring-pattern surface
// ---------------------------------------------------------------------------

/// COW-write builtins: see [project memory v2.4 + v2.1] —
/// `array_push`, `array_pop`, `array_reverse`, `arr_set` use
/// `Rc::make_mut`. Adding new ones is a write-only operation on the
/// match arm; no other code path needs updating.
const COW_WRITE_BUILTINS: &[&str] = &[
    "array_push",
    "array_pop",
    "array_reverse",
    "arr_set",
    "file_append",
];

/// Tensor-heavy builtins: matmul, transpose, sum, mean, dot, fused MLP,
/// and the chess-RL native kernels (encode_state_fast, score_moves_batch).
/// Conservative — counting too many tensor-heavy ops is safe; missing a
/// real one would underweight memory pressure.
const TENSOR_HEAVY_BUILTINS: &[&str] = &[
    "matmul",
    "transpose",
    "dot",
    "sum",
    "mean",
    "tensor_concat_1d",
    "mlp_forward",
    "mlp_layer",
    "encode_state_fast",
    "score_moves_batch",
    "adam_step",
];

/// Alloc-like builtins: producing a new buffer / tensor / collection.
const ALLOC_BUILTINS: &[&str] = &[
    "array_new",
    "tensor_new",
    "tensor_zeros",
    "tensor_ones",
    "zeros",
    "ones",
];

fn classify_builtin_call(name: &str, p: &mut MemoryProxy) {
    if COW_WRITE_BUILTINS.contains(&name) {
        p.cow_write_sites = p.cow_write_sites.saturating_add(1);
    }
    if TENSOR_HEAVY_BUILTINS.contains(&name) {
        p.tensor_heavy_ops = p.tensor_heavy_ops.saturating_add(1);
    }
    if ALLOC_BUILTINS.contains(&name) {
        p.alloc_sites = p.alloc_sites.saturating_add(1);
    }
}

/// Tensor-heavy METHOD names (`t.sum()`, `t.matmul(u)`, ...). Same
/// surface as [`TENSOR_HEAVY_BUILTINS`] plus the tensor reductions and
/// activations only reachable as methods. Classification is by name
/// alone (MIR has no receiver types); a `.sum()` on a DataFrame
/// over-counts tensor_heavy_ops, which the module docs already declare
/// the safe direction.
const TENSOR_HEAVY_METHODS: &[&str] = &[
    "matmul",
    "dot",
    "sum",
    "mean",
    "binned_sum",
    "transpose",
    "reshape",
    "add",
    "sub",
    "abs",
    "relu",
    "gelu",
    "softmax",
    "layer_norm",
    "var",
    "std",
];

/// `Tensor.<name>(...)` static constructors — allocation sites in
/// method form (`Tensor.from_vec` materializes a fresh buffer exactly
/// like the free-call `tensor_new`).
const TENSOR_CONSTRUCTOR_METHODS: &[&str] = &["from_vec", "zeros", "ones", "randn", "eye", "new"];

fn classify_method_call(object: &MirExpr, name: &str, p: &mut MemoryProxy) {
    if TENSOR_HEAVY_METHODS.contains(&name) {
        p.tensor_heavy_ops = p.tensor_heavy_ops.saturating_add(1);
    }
    if let MirExprKind::Var(obj) = &object.kind {
        if obj == "Tensor" && TENSOR_CONSTRUCTOR_METHODS.contains(&name) {
            p.alloc_sites = p.alloc_sites.saturating_add(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirStmt};

    fn empty_main() -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: "__main".to_string(),
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
        }
    }

    fn ekind(k: MirExprKind) -> MirExpr {
        MirExpr { kind: k }
    }

    #[test]
    fn empty_function_has_zero_proxies() {
        let p = MemoryProxy::from_function(&empty_main());
        assert_eq!(p.alloc_sites, 0);
        assert_eq!(p.cow_write_sites, 0);
        assert_eq!(p.tensor_heavy_ops, 0);
        assert_eq!(p.expr_count, 0);
    }

    #[test]
    fn array_lit_counts_one_alloc() {
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::ArrayLit(vec![
            ekind(MirExprKind::IntLit(1)),
            ekind(MirExprKind::IntLit(2)),
        ]))));
        let p = MemoryProxy::from_function(&f);
        assert_eq!(p.alloc_sites, 1);
        // expr_count: ArrayLit + 2 IntLits = 3
        assert_eq!(p.expr_count, 3);
        // Phase F1: 2 element slots — sites are volume-blind, slots are not.
        assert_eq!(p.lit_elem_slots, 2);
    }

    #[test]
    fn lit_elem_slots_distinguish_literal_sizes() {
        // Two functions with ONE alloc site each but different element
        // counts must produce identical alloc_sites and different
        // lit_elem_slots — the exact blindness Phase F1 fixes.
        let mk = |n: usize| {
            let mut f = empty_main();
            f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::ArrayLit(
                (0..n).map(|i| ekind(MirExprKind::IntLit(i as i64))).collect(),
            ))));
            MemoryProxy::from_function(&f)
        };
        let small = mk(2);
        let large = mk(40);
        assert_eq!(small.alloc_sites, large.alloc_sites);
        assert_eq!(small.lit_elem_slots, 2);
        assert_eq!(large.lit_elem_slots, 40);
        // Tuples count slots too; struct/variant/tensor literals do NOT
        // (mirroring which runtime hooks price per element).
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::TupleLit(vec![
            ekind(MirExprKind::IntLit(1)),
            ekind(MirExprKind::IntLit(2)),
            ekind(MirExprKind::IntLit(3)),
        ]))));
        assert_eq!(MemoryProxy::from_function(&f).lit_elem_slots, 3);
    }

    #[test]
    fn array_push_counts_cow_write() {
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::Call {
            callee: Box::new(ekind(MirExprKind::Var("array_push".to_string()))),
            args: vec![
                ekind(MirExprKind::Var("arr".to_string())),
                ekind(MirExprKind::IntLit(1)),
            ],
        })));
        let p = MemoryProxy::from_function(&f);
        assert_eq!(p.cow_write_sites, 1);
        assert_eq!(p.alloc_sites, 0);
    }

    #[test]
    fn linalg_lu_counts_tensor_heavy_and_alloc() {
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::LinalgLU {
            operand: Box::new(ekind(MirExprKind::Var("a".to_string()))),
        })));
        let p = MemoryProxy::from_function(&f);
        assert_eq!(p.tensor_heavy_ops, 1);
        assert_eq!(p.alloc_sites, 1);
    }

    #[test]
    fn method_call_sum_counts_tensor_heavy() {
        // `t.sum()` — Field callee. Pre-Phase-A1 this counted zero.
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::Call {
            callee: Box::new(ekind(MirExprKind::Field {
                object: Box::new(ekind(MirExprKind::Var("t".to_string()))),
                name: "sum".to_string(),
            })),
            args: vec![],
        })));
        let p = MemoryProxy::from_function(&f);
        assert_eq!(p.tensor_heavy_ops, 1);
        assert_eq!(p.alloc_sites, 0);
    }

    #[test]
    fn tensor_from_vec_counts_alloc_site() {
        // `Tensor.from_vec(buf, shape)` — static constructor method.
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::Call {
            callee: Box::new(ekind(MirExprKind::Field {
                object: Box::new(ekind(MirExprKind::Var("Tensor".to_string()))),
                name: "from_vec".to_string(),
            })),
            args: vec![ekind(MirExprKind::Var("buf".to_string()))],
        })));
        let p = MemoryProxy::from_function(&f);
        assert_eq!(p.alloc_sites, 1);
        // from_vec is a constructor, not an FP-heavy op.
        assert_eq!(p.tensor_heavy_ops, 0);
    }

    #[test]
    fn non_tensor_method_counts_nothing() {
        let mut f = empty_main();
        f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::Call {
            callee: Box::new(ekind(MirExprKind::Field {
                object: Box::new(ekind(MirExprKind::Var("s".to_string()))),
                name: "to_upper".to_string(),
            })),
            args: vec![],
        })));
        let p = MemoryProxy::from_function(&f);
        assert_eq!(p.tensor_heavy_ops, 0);
        assert_eq!(p.alloc_sites, 0);
    }

    #[test]
    fn nested_walk_visits_all_subexpressions() {
        // (1 + 2) + (3 + 4)
        let mut f = empty_main();
        let inner_a = ekind(MirExprKind::Binary {
            op: cjc_ast::BinOp::Add,
            left: Box::new(ekind(MirExprKind::IntLit(1))),
            right: Box::new(ekind(MirExprKind::IntLit(2))),
        });
        let inner_b = ekind(MirExprKind::Binary {
            op: cjc_ast::BinOp::Add,
            left: Box::new(ekind(MirExprKind::IntLit(3))),
            right: Box::new(ekind(MirExprKind::IntLit(4))),
        });
        let outer = ekind(MirExprKind::Binary {
            op: cjc_ast::BinOp::Add,
            left: Box::new(inner_a),
            right: Box::new(inner_b),
        });
        f.body.stmts.push(MirStmt::Expr(outer));
        let p = MemoryProxy::from_function(&f);
        // 3 Binary + 4 IntLit = 7 expressions visited.
        assert_eq!(p.expr_count, 7);
    }
}
