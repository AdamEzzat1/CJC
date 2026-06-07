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
}

impl MemoryProxy {
    /// Walk a function's body and accumulate proxy counts.
    pub fn from_function(func: &MirFunction) -> Self {
        let mut p = Self {
            alloc_sites: 0,
            cow_write_sites: 0,
            tensor_heavy_ops: 0,
            expr_count: 0,
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

            // ----- Calls — classify by callee name when it's a Var -----
            MirExprKind::Call { callee, args } => {
                if let MirExprKind::Var(name) = &callee.kind {
                    classify_builtin_call(name, self);
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
