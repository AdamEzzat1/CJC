//! AST Node Utilities — Pure query methods on existing types
//!
//! Adds convenience methods to `Expr`, `Block`, and `Program` via new `impl`
//! blocks.  These are read-only, side-effect-free query methods.
//!
//! ## Design decisions
//!
//! - **New `impl` blocks** — safe in Rust; does not modify existing impls
//! - **No mutation** — all methods return computed values
//! - **No dependencies** — uses only types from this crate

use crate::{Block, DeclKind, Expr, ExprKind, Program};

// ---------------------------------------------------------------------------
// Expr utilities
// ---------------------------------------------------------------------------

impl Expr {
    /// Number of direct child expressions.
    pub fn child_count(&self) -> usize {
        match &self.kind {
            ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::ByteStringLit(_)
            | ExprKind::ByteCharLit(_)
            | ExprKind::RawStringLit(_)
            | ExprKind::RawByteStringLit(_)
            | ExprKind::RegexLit { .. }
            | ExprKind::BoolLit(_)
            | ExprKind::NaLit
            | ExprKind::Ident(_)
            | ExprKind::Col(_) => 0,

            ExprKind::FStringLit(segs) => segs.iter().filter(|(_, e)| e.is_some()).count(),
            ExprKind::TensorLit { rows } => rows.iter().map(|r| r.len()).sum(),
            ExprKind::Unary { .. } | ExprKind::Try(_) => 1,
            ExprKind::Binary { .. }
            | ExprKind::Assign { .. }
            | ExprKind::CompoundAssign { .. }
            | ExprKind::Pipe { .. }
            | ExprKind::Index { .. } => 2,
            ExprKind::Field { .. } => 1,
            ExprKind::MultiIndex { object: _, indices } => 1 + indices.len(),
            ExprKind::Call { args, .. } => 1 + args.len(), // callee + args
            ExprKind::IfExpr { .. } => 1,                  // condition
            ExprKind::Block(_) => 0,
            ExprKind::StructLit { fields, .. } => fields.len(),
            ExprKind::ArrayLit(elems) => elems.len(),
            ExprKind::TupleLit(elems) => elems.len(),
            ExprKind::Lambda { .. } => 1,                  // body
            ExprKind::Match { arms, .. } => 1 + arms.len(), // scrutinee + arms
            ExprKind::VariantLit { fields, .. } => fields.len(),
            ExprKind::Cast { .. } => 1, // the inner expression
        }
    }

    /// Returns true if this expression is a literal value.
    pub fn is_literal(&self) -> bool {
        matches!(
            &self.kind,
            ExprKind::IntLit(_)
                | ExprKind::FloatLit(_)
                | ExprKind::StringLit(_)
                | ExprKind::ByteStringLit(_)
                | ExprKind::ByteCharLit(_)
                | ExprKind::RawStringLit(_)
                | ExprKind::RawByteStringLit(_)
                | ExprKind::BoolLit(_)
                | ExprKind::NaLit
                | ExprKind::RegexLit { .. }
        )
    }

    /// Returns true if this expression is a valid assignment target (place expression).
    pub fn is_place(&self) -> bool {
        matches!(
            &self.kind,
            ExprKind::Ident(_) | ExprKind::Field { .. } | ExprKind::Index { .. }
        )
    }

    /// Returns true if this expression is a compound (non-leaf) expression.
    pub fn is_compound(&self) -> bool {
        matches!(
            &self.kind,
            ExprKind::Binary { .. }
                | ExprKind::Unary { .. }
                | ExprKind::Call { .. }
                | ExprKind::Match { .. }
                | ExprKind::IfExpr { .. }
                | ExprKind::Block(_)
                | ExprKind::Pipe { .. }
                | ExprKind::Lambda { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Block utilities
// ---------------------------------------------------------------------------

impl Block {
    /// Returns true if the block has no statements and no trailing expression.
    pub fn is_empty(&self) -> bool {
        self.stmts.is_empty() && self.expr.is_none()
    }

    /// Number of statements in this block.
    pub fn stmt_count(&self) -> usize {
        self.stmts.len()
    }

    /// Returns true if the block has a trailing expression.
    pub fn has_trailing_expr(&self) -> bool {
        self.expr.is_some()
    }
}

// ---------------------------------------------------------------------------
// Program utilities
// ---------------------------------------------------------------------------

impl Program {
    /// Number of function declarations (including nested in impls).
    pub fn function_count(&self) -> usize {
        let mut count = 0;
        for decl in &self.declarations {
            match &decl.kind {
                DeclKind::Fn(_) => count += 1,
                DeclKind::Impl(i) => count += i.methods.len(),
                _ => {}
            }
        }
        count
    }

    /// Number of struct declarations.
    pub fn struct_count(&self) -> usize {
        self.declarations
            .iter()
            .filter(|d| matches!(&d.kind, DeclKind::Struct(_)))
            .count()
    }

    /// Returns true if there is a function named "main".
    pub fn has_main_function(&self) -> bool {
        self.declarations.iter().any(|d| {
            if let DeclKind::Fn(f) = &d.kind {
                f.name.name == "main"
            } else {
                false
            }
        })
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
    fn test_expr_child_count() {
        assert_eq!(dummy_expr(ExprKind::IntLit(1)).child_count(), 0);
        assert_eq!(
            dummy_expr(ExprKind::Binary {
                op: BinOp::Add,
                left: Box::new(dummy_expr(ExprKind::IntLit(1))),
                right: Box::new(dummy_expr(ExprKind::IntLit(2))),
            })
            .child_count(),
            2
        );
    }

    #[test]
    fn test_expr_is_literal() {
        assert!(dummy_expr(ExprKind::IntLit(1)).is_literal());
        assert!(dummy_expr(ExprKind::FloatLit(1.0)).is_literal());
        assert!(dummy_expr(ExprKind::BoolLit(true)).is_literal());
        assert!(!dummy_expr(ExprKind::Ident(Ident::dummy("x"))).is_literal());
    }

    #[test]
    fn test_expr_is_place() {
        assert!(dummy_expr(ExprKind::Ident(Ident::dummy("x"))).is_place());
        assert!(!dummy_expr(ExprKind::IntLit(1)).is_place());
    }

    #[test]
    fn test_expr_is_compound() {
        assert!(dummy_expr(ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(dummy_expr(ExprKind::IntLit(1))),
            right: Box::new(dummy_expr(ExprKind::IntLit(2))),
        })
        .is_compound());
        assert!(!dummy_expr(ExprKind::IntLit(1)).is_compound());
    }

    #[test]
    fn test_block_utils() {
        let empty = Block {
            stmts: vec![],
            expr: None,
            span: Span::dummy(),
        };
        assert!(empty.is_empty());
        assert_eq!(empty.stmt_count(), 0);
        assert!(!empty.has_trailing_expr());

        let with_expr = Block {
            stmts: vec![Stmt {
                kind: StmtKind::Expr(dummy_expr(ExprKind::IntLit(1))),
                span: Span::dummy(),
            }],
            expr: Some(Box::new(dummy_expr(ExprKind::IntLit(2)))),
            span: Span::dummy(),
        };
        assert!(!with_expr.is_empty());
        assert_eq!(with_expr.stmt_count(), 1);
        assert!(with_expr.has_trailing_expr());
    }

    #[test]
    fn test_program_utils() {
        let program = Program {
            declarations: vec![
                Decl {
                    kind: DeclKind::Fn(FnDecl {
                        name: Ident::dummy("main"),
                        type_params: vec![],
                        params: vec![],
                        return_type: None,
                        body: Block {
                            stmts: vec![],
                            expr: None,
                            span: Span::dummy(),
                        },
                        is_nogc: false,
                        effect_annotation: None,
                        decorators: vec![],
                        vis: Visibility::Private,
                    }),
                    span: Span::dummy(),
                },
                Decl {
                    kind: DeclKind::Struct(StructDecl {
                        name: Ident::dummy("Point"),
                        type_params: vec![],
                        fields: vec![],
                        vis: Visibility::Private,
                    }),
                    span: Span::dummy(),
                },
            ],
        };
        assert_eq!(program.function_count(), 1);
        assert_eq!(program.struct_count(), 1);
        assert!(program.has_main_function());
    }
}
