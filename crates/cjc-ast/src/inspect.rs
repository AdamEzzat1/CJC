//! AST Inspect/Diagnostics — Deterministic text dumps for debugging and tests
//!
//! Provides human-readable, deterministic text output for AST programs,
//! metrics, validation reports, and expression trees.
//!
//! ## Design decisions
//!
//! - **Read-only** — never modifies any AST structure
//! - **Deterministic** — sorted iteration, no HashMap, identical output
//! - **Plain text** — no ANSI colors, easy to diff

use crate::metrics::AstMetrics;
use crate::validate::ValidationReport;
use crate::{DeclKind, ExprKind, Program, StmtKind, Expr};

// ---------------------------------------------------------------------------
// Program summary
// ---------------------------------------------------------------------------

/// One-line-per-declaration overview of a program.
///
/// Example:
/// ```text
/// AstSummary (3 declarations):
///   [0] fn count(n: i64) -> i64
///   [1] fn main()
///   [2] stmt: expr
/// ```
pub fn dump_ast_summary(program: &Program) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "AstSummary ({} declarations):\n",
        program.declarations.len()
    ));

    for (i, decl) in program.declarations.iter().enumerate() {
        let desc = match &decl.kind {
            DeclKind::Fn(f) => {
                let params: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name.name, type_expr_to_str(&p.ty)))
                    .collect();
                let ret = f
                    .return_type
                    .as_ref()
                    .map(|t| format!(" -> {}", type_expr_to_str(t)))
                    .unwrap_or_default();
                format!("fn {}({}){}", f.name.name, params.join(", "), ret)
            }
            DeclKind::Struct(s) => format!("struct {} ({} fields)", s.name.name, s.fields.len()),
            DeclKind::Class(c) => format!("class {} ({} fields)", c.name.name, c.fields.len()),
            DeclKind::Record(r) => format!("record {} ({} fields)", r.name.name, r.fields.len()),
            DeclKind::Enum(e) => {
                format!("enum {} ({} variants)", e.name.name, e.variants.len())
            }
            DeclKind::Trait(t) => {
                format!("trait {} ({} methods)", t.name.name, t.methods.len())
            }
            DeclKind::Impl(i) => format!("impl {}", type_expr_to_str(&i.target)),
            DeclKind::Let(l) => format!("let {}", l.name.name),
            DeclKind::Const(c) => format!("const {}", c.name.name),
            DeclKind::Import(i) => {
                let path: Vec<&str> = i.path.iter().map(|id| id.name.as_str()).collect();
                format!("import {}", path.join("::"))
            }
            DeclKind::Stmt(s) => match &s.kind {
                StmtKind::Expr(_) => "stmt: expr".to_string(),
                StmtKind::If(_) => "stmt: if".to_string(),
                StmtKind::While(_) => "stmt: while".to_string(),
                StmtKind::For(_) => "stmt: for".to_string(),
                _ => "stmt: other".to_string(),
            },
        };
        out.push_str(&format!("  [{}] {}\n", i, desc));
    }

    out
}

fn type_expr_to_str(ty: &crate::TypeExpr) -> String {
    match &ty.kind {
        crate::TypeExprKind::Named { name, args } => {
            if args.is_empty() {
                name.name.clone()
            } else {
                format!("{}<...>", name.name)
            }
        }
        crate::TypeExprKind::Array { .. } => "[T; N]".to_string(),
        crate::TypeExprKind::Tuple(tys) => {
            let inner: Vec<String> = tys.iter().map(type_expr_to_str).collect();
            format!("({})", inner.join(", "))
        }
        crate::TypeExprKind::Fn { .. } => "fn(...)".to_string(),
        crate::TypeExprKind::ShapeLit(_) => "[shape]".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Metrics dump
// ---------------------------------------------------------------------------

/// Formatted metrics dump.
pub fn dump_ast_metrics(metrics: &AstMetrics) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "AstMetrics (total_nodes={}):\n",
        metrics.total_nodes
    ));
    out.push_str(&format!("  exprs={}, stmts={}, decls={}, patterns={}\n",
        metrics.expr_count, metrics.stmt_count, metrics.decl_count, metrics.pattern_count));
    out.push_str(&format!(
        "  max_expr_depth={}, max_stmt_depth={}\n",
        metrics.max_expr_depth, metrics.max_stmt_depth
    ));
    out.push_str(&format!(
        "  functions={}, closures={}, loops={}, matches={}\n",
        metrics.function_count, metrics.closure_count, metrics.loop_count, metrics.match_count
    ));
    if !metrics.binary_op_counts.is_empty() {
        let ops: Vec<String> = metrics
            .binary_op_counts
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        out.push_str(&format!("  binary_ops: {}\n", ops.join(", ")));
    }
    out.push_str(&format!(
        "  nogc={}, decorators={}, variadics={}\n",
        metrics.has_nogc, metrics.has_decorators, metrics.has_variadics
    ));
    out
}

// ---------------------------------------------------------------------------
// Validation report dump
// ---------------------------------------------------------------------------

/// Formatted validation report.
pub fn dump_validation_report(report: &ValidationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "ValidationReport: {}/{} checks passed",
        report.checks_passed, report.checks_run
    ));
    if report.is_ok() && report.findings.is_empty() {
        out.push_str(" (clean)\n");
    } else {
        out.push_str(&format!(" ({} findings)\n", report.findings.len()));
        for f in &report.findings {
            out.push_str(&format!("  {}\n", f));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Expression tree dump
// ---------------------------------------------------------------------------

/// Indented tree view of an expression.
///
/// Example:
/// ```text
/// Binary(+)
///   IntLit(1)
///   IntLit(2)
/// ```
pub fn dump_expr_tree(expr: &Expr) -> String {
    let mut out = String::new();
    dump_expr_recursive(expr, 0, &mut out);
    out
}

fn dump_expr_recursive(expr: &Expr, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    match &expr.kind {
        ExprKind::IntLit(v) => out.push_str(&format!("{}IntLit({})\n", pad, v)),
        ExprKind::FloatLit(v) => out.push_str(&format!("{}FloatLit({})\n", pad, v)),
        ExprKind::StringLit(s) => out.push_str(&format!("{}StringLit(\"{}\")\n", pad, s)),
        ExprKind::BoolLit(b) => out.push_str(&format!("{}BoolLit({})\n", pad, b)),
        ExprKind::NaLit => out.push_str(&format!("{}NaLit\n", pad)),
        ExprKind::Ident(id) => out.push_str(&format!("{}Ident({})\n", pad, id.name)),
        ExprKind::Binary { op, left, right } => {
            out.push_str(&format!("{}Binary({})\n", pad, op));
            dump_expr_recursive(left, indent + 1, out);
            dump_expr_recursive(right, indent + 1, out);
        }
        ExprKind::Unary { op, operand } => {
            out.push_str(&format!("{}Unary({})\n", pad, op));
            dump_expr_recursive(operand, indent + 1, out);
        }
        ExprKind::Call { callee, args } => {
            out.push_str(&format!("{}Call\n", pad));
            dump_expr_recursive(callee, indent + 1, out);
            for arg in args {
                dump_expr_recursive(&arg.value, indent + 1, out);
            }
        }
        ExprKind::Assign { target, value } => {
            out.push_str(&format!("{}Assign\n", pad));
            dump_expr_recursive(target, indent + 1, out);
            dump_expr_recursive(value, indent + 1, out);
        }
        ExprKind::Index { object, index } => {
            out.push_str(&format!("{}Index\n", pad));
            dump_expr_recursive(object, indent + 1, out);
            dump_expr_recursive(index, indent + 1, out);
        }
        ExprKind::Field { object, name } => {
            out.push_str(&format!("{}Field(.{})\n", pad, name.name));
            dump_expr_recursive(object, indent + 1, out);
        }
        ExprKind::Lambda { params, body } => {
            let pnames: Vec<&str> = params.iter().map(|p| p.name.name.as_str()).collect();
            out.push_str(&format!("{}Lambda(|{}|)\n", pad, pnames.join(", ")));
            dump_expr_recursive(body, indent + 1, out);
        }
        ExprKind::Match { scrutinee, arms } => {
            out.push_str(&format!("{}Match ({} arms)\n", pad, arms.len()));
            dump_expr_recursive(scrutinee, indent + 1, out);
        }
        ExprKind::IfExpr { condition, .. } => {
            out.push_str(&format!("{}IfExpr\n", pad));
            dump_expr_recursive(condition, indent + 1, out);
        }
        ExprKind::ArrayLit(elems) => {
            out.push_str(&format!("{}ArrayLit({} elems)\n", pad, elems.len()));
        }
        ExprKind::TupleLit(elems) => {
            out.push_str(&format!("{}TupleLit({} elems)\n", pad, elems.len()));
        }
        ExprKind::Pipe { left, right } => {
            out.push_str(&format!("{}Pipe\n", pad));
            dump_expr_recursive(left, indent + 1, out);
            dump_expr_recursive(right, indent + 1, out);
        }
        _ => {
            out.push_str(&format!("{}Expr({:?})\n", pad, std::mem::discriminant(&expr.kind)));
        }
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
    fn test_dump_ast_summary_empty() {
        let program = Program {
            declarations: Vec::new(),
        };
        let text = dump_ast_summary(&program);
        assert!(text.contains("0 declarations"));
    }

    #[test]
    fn test_dump_ast_summary_fn() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(FnDecl {
                    name: Ident::dummy("compute"),
                    type_params: vec![],
                    params: vec![Param {
                        name: Ident::dummy("x"),
                        ty: TypeExpr {
                            kind: TypeExprKind::Named {
                                name: Ident::dummy("i64"),
                                args: vec![],
                            },
                            span: Span::dummy(),
                        },
                        default: None,
                        is_variadic: false,
                        span: Span::dummy(),
                    }],
                    return_type: Some(TypeExpr {
                        kind: TypeExprKind::Named {
                            name: Ident::dummy("i64"),
                            args: vec![],
                        },
                        span: Span::dummy(),
                    }),
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
            }],
        };
        let text = dump_ast_summary(&program);
        assert!(text.contains("fn compute(x: i64) -> i64"));
    }

    #[test]
    fn test_dump_expr_tree() {
        let expr = dummy_expr(ExprKind::Binary {
            op: BinOp::Add,
            left: Box::new(dummy_expr(ExprKind::IntLit(1))),
            right: Box::new(dummy_expr(ExprKind::IntLit(2))),
        });
        let text = dump_expr_tree(&expr);
        assert!(text.contains("Binary(+)"));
        assert!(text.contains("IntLit(1)"));
        assert!(text.contains("IntLit(2)"));
    }

    #[test]
    fn test_dump_determinism() {
        let expr = dummy_expr(ExprKind::Binary {
            op: BinOp::Mul,
            left: Box::new(dummy_expr(ExprKind::Ident(Ident::dummy("x")))),
            right: Box::new(dummy_expr(ExprKind::FloatLit(3.14))),
        });
        let t1 = dump_expr_tree(&expr);
        let t2 = dump_expr_tree(&expr);
        assert_eq!(t1, t2, "expr tree dump must be deterministic");
    }
}
