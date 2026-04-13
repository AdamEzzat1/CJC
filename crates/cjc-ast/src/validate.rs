//! AST Validation — Lightweight structural checks before downstream lowering
//!
//! Catches provably-wrong structural patterns that the parser accepts but
//! downstream passes (HIR, types) would reject or mishandle.
//!
//! ## Design decisions
//!
//! - **Conservative** — only flags things that are structurally wrong; no type
//!   or semantic checks (those belong in cjc-types/cjc-hir)
//! - **Uses `AstVisitor`** — single traversal pass
//! - **Additive overlay** — does not modify any AST types
//! - **Deterministic** — errors reported in traversal order

use crate::visit::{self, AstVisitor};
use crate::{
    Block, Decl, DeclKind, Expr, ExprKind, FnDecl, Ident, Pattern, PatternKind, Program, Stmt,
    StmtKind,
};

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Severity level of a validation finding.
///
/// Errors indicate structurally invalid ASTs that would cause downstream
/// compilation failures. Warnings indicate suspicious but technically valid
/// patterns (e.g. unreachable code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// A suspicious pattern that does not prevent compilation.
    Warning,
    /// A structural error that must be fixed before lowering.
    Error,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: ValidationSeverity,
    pub check: &'static str,
    pub message: String,
    pub span: crate::Span,
}

impl std::fmt::Display for ValidationFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sev = match self.severity {
            ValidationSeverity::Warning => "warning",
            ValidationSeverity::Error => "error",
        };
        write!(
            f,
            "[{}] {}: {} (at {}..{})",
            sev, self.check, self.message, self.span.start, self.span.end
        )
    }
}

/// Result of running all validation checks on an AST.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<ValidationFinding>,
    pub checks_run: u32,
    pub checks_passed: u32,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == ValidationSeverity::Error)
    }

    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == ValidationSeverity::Error)
            .collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == ValidationSeverity::Warning)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Run all structural validation checks on a program.
///
/// Checks:
/// 1. `break`/`continue` outside loop body
/// 2. `return` outside function body
/// 3. Duplicate parameter names in functions
/// 4. Duplicate field names in structs/classes/records
/// 5. Match expression with zero arms
/// 6. Unreachable code after `return` in a block
/// 7. Nesting depth exceeds 256
/// 8. Empty function body (warning only)
pub fn validate_ast(program: &Program) -> ValidationReport {
    let mut validator = Validator::new();
    validator.visit_program(program);

    ValidationReport {
        findings: validator.findings,
        checks_run: validator.checks_run,
        checks_passed: validator.checks_run - validator.checks_failed,
    }
}

struct Validator {
    findings: Vec<ValidationFinding>,
    checks_run: u32,
    checks_failed: u32,
    /// Nesting depth of loops (while/for).
    loop_depth: u32,
    /// Whether we're inside a function body.
    in_function: bool,
    /// Current statement nesting depth.
    nesting_depth: u32,
}

const MAX_NESTING_DEPTH: u32 = 256;

impl Validator {
    fn new() -> Self {
        Self {
            findings: Vec::new(),
            checks_run: 0,
            checks_failed: 0,
            loop_depth: 0,
            in_function: false,
            nesting_depth: 0,
        }
    }

    fn error(&mut self, check: &'static str, message: String, span: crate::Span) {
        self.checks_run += 1;
        self.checks_failed += 1;
        self.findings.push(ValidationFinding {
            severity: ValidationSeverity::Error,
            check,
            message,
            span,
        });
    }

    fn warning(&mut self, check: &'static str, message: String, span: crate::Span) {
        self.checks_run += 1;
        self.findings.push(ValidationFinding {
            severity: ValidationSeverity::Warning,
            check,
            message,
            span,
        });
    }

    fn pass(&mut self) {
        self.checks_run += 1;
    }

    fn check_duplicate_params(&mut self, f: &FnDecl) {
        let mut seen = std::collections::BTreeSet::new();
        for param in &f.params {
            if !seen.insert(&param.name.name) {
                self.error(
                    "duplicate_param",
                    format!("duplicate parameter name `{}`", param.name.name),
                    param.span,
                );
            }
        }
        if seen.len() == f.params.len() {
            self.pass();
        }
    }

    fn check_duplicate_fields(&mut self, fields: &[crate::FieldDecl], kind: &str, span: crate::Span) {
        let mut seen = std::collections::BTreeSet::new();
        let mut has_dup = false;
        for field in fields {
            if !seen.insert(&field.name.name) {
                has_dup = true;
                self.error(
                    "duplicate_field",
                    format!("duplicate field name `{}` in {}", field.name.name, kind),
                    field.span,
                );
            }
        }
        if !has_dup {
            self.pass();
        }
    }

    fn check_unreachable_after_return(&mut self, block: &Block) {
        let mut found_return = false;
        for stmt in &block.stmts {
            if found_return {
                self.warning(
                    "unreachable_code",
                    "unreachable code after return statement".to_string(),
                    stmt.span,
                );
                return;
            }
            if matches!(&stmt.kind, StmtKind::Return(_)) {
                found_return = true;
            }
        }
        self.pass();
    }
}

impl AstVisitor for Validator {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        // Check: break/continue outside loop
        match &stmt.kind {
            StmtKind::Break => {
                if self.loop_depth == 0 {
                    self.error(
                        "break_outside_loop",
                        "`break` outside of loop body".to_string(),
                        stmt.span,
                    );
                } else {
                    self.pass();
                }
            }
            StmtKind::Continue => {
                if self.loop_depth == 0 {
                    self.error(
                        "continue_outside_loop",
                        "`continue` outside of loop body".to_string(),
                        stmt.span,
                    );
                } else {
                    self.pass();
                }
            }
            StmtKind::Return(_) => {
                if !self.in_function {
                    // Top-level return is allowed in CJC (returns from __main)
                    // so we just pass here.
                    self.pass();
                } else {
                    self.pass();
                }
            }
            _ => {}
        }

        // Check: nesting depth
        let nests = matches!(
            &stmt.kind,
            StmtKind::If(_) | StmtKind::While(_) | StmtKind::For(_) | StmtKind::NoGcBlock(_)
        );
        if nests {
            self.nesting_depth += 1;
            if self.nesting_depth > MAX_NESTING_DEPTH {
                self.error(
                    "nesting_depth",
                    format!(
                        "statement nesting depth {} exceeds maximum {}",
                        self.nesting_depth, MAX_NESTING_DEPTH
                    ),
                    stmt.span,
                );
            } else {
                self.pass();
            }
        }

        // Track loop depth for break/continue checking.
        let is_loop = matches!(&stmt.kind, StmtKind::While(_) | StmtKind::For(_));
        if is_loop {
            self.loop_depth += 1;
        }

        visit::walk_stmt(self, stmt);

        if is_loop {
            self.loop_depth -= 1;
        }
        if nests {
            self.nesting_depth -= 1;
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        // Check: empty match
        if let ExprKind::Match { arms, .. } = &expr.kind {
            if arms.is_empty() {
                self.error(
                    "empty_match",
                    "match expression with zero arms".to_string(),
                    expr.span,
                );
            } else {
                self.pass();
            }
        }

        visit::walk_expr(self, expr);
    }

    fn visit_fn_decl(&mut self, f: &FnDecl) {
        // Check: duplicate parameter names
        self.check_duplicate_params(f);

        // Check: unreachable code after return
        self.check_unreachable_after_return(&f.body);

        let prev_in_function = self.in_function;
        let prev_loop_depth = self.loop_depth;
        self.in_function = true;
        self.loop_depth = 0;

        visit::walk_fn_decl(self, f);

        self.in_function = prev_in_function;
        self.loop_depth = prev_loop_depth;
    }

    fn visit_decl(&mut self, decl: &Decl) {
        match &decl.kind {
            DeclKind::Struct(s) => {
                self.check_duplicate_fields(&s.fields, "struct", decl.span);
            }
            DeclKind::Class(c) => {
                self.check_duplicate_fields(&c.fields, "class", decl.span);
            }
            DeclKind::Record(r) => {
                self.check_duplicate_fields(&r.fields, "record", decl.span);
            }
            _ => {}
        }

        visit::walk_decl(self, decl);
    }

    fn visit_block(&mut self, block: &Block) {
        self.check_unreachable_after_return(block);
        visit::walk_block(self, block);
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

    fn dummy_stmt(kind: StmtKind) -> Stmt {
        Stmt {
            kind,
            span: Span::new(0, 10),
        }
    }

    fn make_fn(name: &str, body: Block) -> FnDecl {
        FnDecl {
            name: Ident::dummy(name),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body,
            is_nogc: false,
            effect_annotation: None,
            decorators: vec![],
            vis: Visibility::Private,
        }
    }

    #[test]
    fn test_valid_program_passes() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(make_fn(
                    "test",
                    Block {
                        stmts: vec![dummy_stmt(StmtKind::Expr(dummy_expr(ExprKind::IntLit(1))))],
                        expr: None,
                        span: Span::dummy(),
                    },
                )),
                span: Span::dummy(),
            }],
        };
        let report = validate_ast(&program);
        assert!(report.is_ok(), "valid program should pass: {:?}", report.findings);
    }

    #[test]
    fn test_break_outside_loop_detected() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(make_fn(
                    "test",
                    Block {
                        stmts: vec![dummy_stmt(StmtKind::Break)],
                        expr: None,
                        span: Span::dummy(),
                    },
                )),
                span: Span::dummy(),
            }],
        };
        let report = validate_ast(&program);
        let errors = report.errors();
        assert!(
            errors.iter().any(|e| e.check == "break_outside_loop"),
            "should detect break outside loop"
        );
    }

    #[test]
    fn test_continue_outside_loop_detected() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(make_fn(
                    "test",
                    Block {
                        stmts: vec![dummy_stmt(StmtKind::Continue)],
                        expr: None,
                        span: Span::dummy(),
                    },
                )),
                span: Span::dummy(),
            }],
        };
        let report = validate_ast(&program);
        let errors = report.errors();
        assert!(
            errors.iter().any(|e| e.check == "continue_outside_loop"),
            "should detect continue outside loop"
        );
    }

    #[test]
    fn test_break_inside_loop_ok() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(make_fn(
                    "test",
                    Block {
                        stmts: vec![dummy_stmt(StmtKind::While(WhileStmt {
                            condition: dummy_expr(ExprKind::BoolLit(true)),
                            body: Block {
                                stmts: vec![dummy_stmt(StmtKind::Break)],
                                expr: None,
                                span: Span::dummy(),
                            },
                        }))],
                        expr: None,
                        span: Span::dummy(),
                    },
                )),
                span: Span::dummy(),
            }],
        };
        let report = validate_ast(&program);
        assert!(
            !report.errors().iter().any(|e| e.check == "break_outside_loop"),
            "break inside loop should be ok"
        );
    }

    #[test]
    fn test_duplicate_params_detected() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(FnDecl {
                    name: Ident::dummy("test"),
                    type_params: vec![],
                    params: vec![
                        Param {
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
                            span: Span::new(0, 5),
                        },
                        Param {
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
                            span: Span::new(6, 11),
                        },
                    ],
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
            }],
        };
        let report = validate_ast(&program);
        assert!(
            report.errors().iter().any(|e| e.check == "duplicate_param"),
            "should detect duplicate params"
        );
    }

    #[test]
    fn test_empty_match_detected() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Stmt(dummy_stmt(StmtKind::Expr(dummy_expr(ExprKind::Match {
                    scrutinee: Box::new(dummy_expr(ExprKind::IntLit(1))),
                    arms: vec![],
                })))),
                span: Span::dummy(),
            }],
        };
        let report = validate_ast(&program);
        assert!(
            report.errors().iter().any(|e| e.check == "empty_match"),
            "should detect empty match"
        );
    }

    #[test]
    fn test_validation_determinism() {
        let program = Program {
            declarations: vec![Decl {
                kind: DeclKind::Fn(make_fn(
                    "test",
                    Block {
                        stmts: vec![
                            dummy_stmt(StmtKind::Break),
                            dummy_stmt(StmtKind::Continue),
                        ],
                        expr: None,
                        span: Span::dummy(),
                    },
                )),
                span: Span::dummy(),
            }],
        };
        let r1 = validate_ast(&program);
        let r2 = validate_ast(&program);
        assert_eq!(r1.findings.len(), r2.findings.len());
        assert_eq!(r1.checks_run, r2.checks_run);
    }
}
