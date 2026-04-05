//! AST Metrics — Structural statistics computed via visitor traversal
//!
//! Computes node counts, depth measurements, operator frequencies, and
//! feature-presence flags from an AST `Program`.
//!
//! ## Design decisions
//!
//! - **Uses `AstVisitor`** — built on the visitor from `visit.rs`
//! - **Deterministic** — `BTreeMap` for operator counts, sorted iteration
//! - **Read-only** — does not modify the AST
//! - **Zero dependencies** — leaf crate contract preserved

use std::collections::BTreeMap;

use crate::visit::{self, AstVisitor};
use crate::{
    BinOp, Block, DeclKind, Expr, ExprKind, FnDecl, Param, Pattern, Program, Stmt, StmtKind,
};

// ---------------------------------------------------------------------------
// Metrics struct
// ---------------------------------------------------------------------------

/// Structural metrics for a CJC AST program.
///
/// All counts are deterministic — same AST always produces identical metrics.
#[derive(Debug, Clone)]
pub struct AstMetrics {
    /// Total AST nodes (expr + stmt + decl + pattern).
    pub total_nodes: u32,
    /// Number of expression nodes.
    pub expr_count: u32,
    /// Number of statement nodes.
    pub stmt_count: u32,
    /// Number of declaration nodes.
    pub decl_count: u32,
    /// Number of pattern nodes.
    pub pattern_count: u32,
    /// Maximum expression nesting depth.
    pub max_expr_depth: u32,
    /// Maximum statement nesting depth (if/while/for).
    pub max_stmt_depth: u32,
    /// Number of function declarations.
    pub function_count: u32,
    /// Number of lambda (closure) expressions.
    pub closure_count: u32,
    /// Number of loop constructs (while + for).
    pub loop_count: u32,
    /// Number of match expressions.
    pub match_count: u32,
    /// Per-operator usage frequency (e.g., "+" → 5).
    pub binary_op_counts: BTreeMap<String, u32>,
    /// Whether any `@nogc` function exists.
    pub has_nogc: bool,
    /// Whether any decorated function exists.
    pub has_decorators: bool,
    /// Whether any variadic parameter exists.
    pub has_variadics: bool,
}

// ---------------------------------------------------------------------------
// Computation
// ---------------------------------------------------------------------------

/// Compute structural metrics for a program.
///
/// Uses the `AstVisitor` trait for a single traversal pass.
pub fn compute_metrics(program: &Program) -> AstMetrics {
    let mut collector = MetricsCollector::new();
    collector.visit_program(program);

    AstMetrics {
        total_nodes: collector.expr_count
            + collector.stmt_count
            + collector.decl_count
            + collector.pattern_count,
        expr_count: collector.expr_count,
        stmt_count: collector.stmt_count,
        decl_count: collector.decl_count,
        pattern_count: collector.pattern_count,
        max_expr_depth: collector.max_expr_depth,
        max_stmt_depth: collector.max_stmt_depth,
        function_count: collector.function_count,
        closure_count: collector.closure_count,
        loop_count: collector.loop_count,
        match_count: collector.match_count,
        binary_op_counts: collector.binary_op_counts,
        has_nogc: collector.has_nogc,
        has_decorators: collector.has_decorators,
        has_variadics: collector.has_variadics,
    }
}

/// Internal visitor that accumulates metrics during a single AST traversal.
///
/// Tracks running counts, current/max depths, and feature-presence flags.
/// After the traversal completes, its fields are copied into an [`AstMetrics`].
struct MetricsCollector {
    expr_count: u32,
    stmt_count: u32,
    decl_count: u32,
    pattern_count: u32,
    current_expr_depth: u32,
    max_expr_depth: u32,
    current_stmt_depth: u32,
    max_stmt_depth: u32,
    function_count: u32,
    closure_count: u32,
    loop_count: u32,
    match_count: u32,
    binary_op_counts: BTreeMap<String, u32>,
    has_nogc: bool,
    has_decorators: bool,
    has_variadics: bool,
}

impl MetricsCollector {
    /// Create a new collector with all counters zeroed.
    fn new() -> Self {
        Self {
            expr_count: 0,
            stmt_count: 0,
            decl_count: 0,
            pattern_count: 0,
            current_expr_depth: 0,
            max_expr_depth: 0,
            current_stmt_depth: 0,
            max_stmt_depth: 0,
            function_count: 0,
            closure_count: 0,
            loop_count: 0,
            match_count: 0,
            binary_op_counts: BTreeMap::new(),
            has_nogc: false,
            has_decorators: false,
            has_variadics: false,
        }
    }
}

impl AstVisitor for MetricsCollector {
    fn visit_expr(&mut self, expr: &Expr) {
        self.expr_count += 1;
        self.current_expr_depth += 1;
        if self.current_expr_depth > self.max_expr_depth {
            self.max_expr_depth = self.current_expr_depth;
        }

        match &expr.kind {
            ExprKind::Binary { op, .. } => {
                *self
                    .binary_op_counts
                    .entry(format!("{}", op))
                    .or_insert(0) += 1;
            }
            ExprKind::Lambda { .. } => {
                self.closure_count += 1;
            }
            ExprKind::Match { .. } => {
                self.match_count += 1;
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
        self.current_expr_depth -= 1;
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        self.stmt_count += 1;

        let nests = matches!(
            &stmt.kind,
            StmtKind::If(_) | StmtKind::While(_) | StmtKind::For(_) | StmtKind::NoGcBlock(_)
        );

        if nests {
            self.current_stmt_depth += 1;
            if self.current_stmt_depth > self.max_stmt_depth {
                self.max_stmt_depth = self.current_stmt_depth;
            }
        }

        match &stmt.kind {
            StmtKind::While(_) | StmtKind::For(_) => {
                self.loop_count += 1;
            }
            _ => {}
        }

        visit::walk_stmt(self, stmt);

        if nests {
            self.current_stmt_depth -= 1;
        }
    }

    fn visit_decl(&mut self, decl: &crate::Decl) {
        self.decl_count += 1;
        visit::walk_decl(self, decl);
    }

    fn visit_fn_decl(&mut self, f: &FnDecl) {
        self.function_count += 1;
        if f.is_nogc {
            self.has_nogc = true;
        }
        if !f.decorators.is_empty() {
            self.has_decorators = true;
        }
        visit::walk_fn_decl(self, f);
    }

    fn visit_param(&mut self, param: &Param) {
        if param.is_variadic {
            self.has_variadics = true;
        }
        visit::walk_param(self, param);
    }

    fn visit_pattern(&mut self, pattern: &Pattern) {
        self.pattern_count += 1;
        visit::walk_pattern(self, pattern);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn dummy_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span::dummy(),
        }
    }

    #[test]
    fn test_empty_program() {
        let program = Program {
            declarations: Vec::new(),
        };
        let m = compute_metrics(&program);
        assert_eq!(m.total_nodes, 0);
        assert_eq!(m.expr_count, 0);
        assert_eq!(m.function_count, 0);
    }

    #[test]
    fn test_binary_ops_counted() {
        let expr = dummy_expr(ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(dummy_expr(ExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(dummy_expr(ExprKind::IntLit(1))),
                right: Box::new(dummy_expr(ExprKind::IntLit(2))),
            })),
            right: Box::new(dummy_expr(ExprKind::IntLit(3))),
        });
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Stmt(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: Span::dummy(),
                }),
                span: Span::dummy(),
            }],
        };
        let m = compute_metrics(&program);
        assert_eq!(m.binary_op_counts.get("+"), Some(&2));
        assert_eq!(m.expr_count, 5); // 2 binary + 3 int literals
    }

    #[test]
    fn test_expr_depth() {
        // depth 3: Binary(Binary(IntLit, IntLit), IntLit)
        let expr = dummy_expr(ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(dummy_expr(ExprKind::Binary {
                op: BinOp::Mul,
                left: Box::new(dummy_expr(ExprKind::IntLit(1))),
                right: Box::new(dummy_expr(ExprKind::IntLit(2))),
            })),
            right: Box::new(dummy_expr(ExprKind::IntLit(3))),
        });
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Stmt(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: Span::dummy(),
                }),
                span: Span::dummy(),
            }],
        };
        let m = compute_metrics(&program);
        assert_eq!(m.max_expr_depth, 3);
    }

    #[test]
    fn test_metrics_determinism() {
        let expr = dummy_expr(ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(dummy_expr(ExprKind::IntLit(1))),
            right: Box::new(dummy_expr(ExprKind::IntLit(2))),
        });
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Stmt(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: Span::dummy(),
                }),
                span: Span::dummy(),
            }],
        };
        let m1 = compute_metrics(&program);
        let m2 = compute_metrics(&program);
        assert_eq!(m1.total_nodes, m2.total_nodes);
        assert_eq!(m1.expr_count, m2.expr_count);
        assert_eq!(m1.max_expr_depth, m2.max_expr_depth);
    }
}
