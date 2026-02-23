//! Integration tests for CJC Stage 2.2: Match + Patterns
//!
//! Tests cover:
//!   - Lexer: `match` keyword and `_` wildcard token
//!   - Parser: match expression parsing, all pattern kinds
//!   - AST tree-walk evaluator: match execution
//!   - Full pipeline (AST → HIR → MIR → MIR-exec): match parity
//!   - Structural destructuring: Tuple and Struct patterns
//!   - Decision tree compilation: pattern ordering and fallthrough

use cjc_ast::*;
use cjc_eval::Interpreter;
use cjc_lexer::Lexer;
use cjc_lexer::TokenKind;
use cjc_parser::parse_source;
use cjc_runtime::Value;

// ═══════════════════════════════════════════════════════════════════════
// Helper constructors
// ═══════════════════════════════════════════════════════════════════════

fn span() -> Span {
    Span::dummy()
}

fn ident(name: &str) -> Ident {
    Ident::dummy(name)
}

fn int_expr(v: i64) -> Expr {
    Expr { kind: ExprKind::IntLit(v), span: span() }
}

fn bool_expr(v: bool) -> Expr {
    Expr { kind: ExprKind::BoolLit(v), span: span() }
}

fn ident_expr(name: &str) -> Expr {
    Expr { kind: ExprKind::Ident(ident(name)), span: span() }
}

fn string_expr(s: &str) -> Expr {
    Expr { kind: ExprKind::StringLit(s.to_string()), span: span() }
}

fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr {
        kind: ExprKind::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
        span: span(),
    }
}

fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    let call_args: Vec<CallArg> = args
        .into_iter()
        .map(|value| CallArg { name: None, value, span: span() })
        .collect();
    Expr {
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args: call_args,
        },
        span: span(),
    }
}

fn let_stmt(name: &str, init: Expr) -> Stmt {
    Stmt {
        kind: StmtKind::Let(LetStmt {
            name: ident(name),
            mutable: false,
            ty: None,
            init: Box::new(init),
        }),
        span: span(),
    }
}

fn let_mut_stmt(name: &str, init: Expr) -> Stmt {
    Stmt {
        kind: StmtKind::Let(LetStmt {
            name: ident(name),
            mutable: true,
            ty: None,
            init: Box::new(init),
        }),
        span: span(),
    }
}

fn expr_stmt(expr: Expr) -> Stmt {
    Stmt { kind: StmtKind::Expr(expr), span: span() }
}

fn assign_expr(target: Expr, value: Expr) -> Expr {
    Expr {
        kind: ExprKind::Assign {
            target: Box::new(target),
            value: Box::new(value),
        },
        span: span(),
    }
}

fn dummy_type_expr() -> TypeExpr {
    TypeExpr {
        kind: TypeExprKind::Named { name: ident("i64"), args: vec![] },
        span: span(),
    }
}

fn make_param(name: &str) -> Param {
    Param { name: ident(name), ty: dummy_type_expr(), span: span() }
}

fn make_fn_decl(name: &str, params: Vec<&str>, body: Block) -> Decl {
    Decl {
        kind: DeclKind::Fn(FnDecl {
            name: ident(name),
            type_params: vec![],
            params: params.into_iter().map(make_param).collect(),
            return_type: None,
            body,
            is_nogc: false,
        }),
        span: span(),
    }
}

fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block { stmts, expr: expr.map(Box::new), span: span() }
}

fn make_struct_decl(name: &str, fields: Vec<(&str, &str)>) -> Decl {
    Decl {
        kind: DeclKind::Struct(StructDecl {
            name: ident(name),
            type_params: vec![],
            fields: fields
                .into_iter()
                .map(|(n, t)| FieldDecl {
                    name: ident(n),
                    ty: TypeExpr {
                        kind: TypeExprKind::Named { name: ident(t), args: vec![] },
                        span: span(),
                    },
                    default: None,
                    span: span(),
                })
                .collect(),
        }),
        span: span(),
    }
}

/// Build a match expression from a scrutinee and arms
fn match_expr(scrutinee: Expr, arms: Vec<(Pattern, Expr)>) -> Expr {
    Expr {
        kind: ExprKind::Match {
            scrutinee: Box::new(scrutinee),
            arms: arms
                .into_iter()
                .map(|(pattern, body)| MatchArm {
                    pattern,
                    body,
                    span: span(),
                })
                .collect(),
        },
        span: span(),
    }
}

fn wildcard_pat() -> Pattern {
    Pattern { kind: PatternKind::Wildcard, span: span() }
}

fn binding_pat(name: &str) -> Pattern {
    Pattern { kind: PatternKind::Binding(ident(name)), span: span() }
}

fn int_pat(v: i64) -> Pattern {
    Pattern { kind: PatternKind::LitInt(v), span: span() }
}

fn bool_pat(v: bool) -> Pattern {
    Pattern { kind: PatternKind::LitBool(v), span: span() }
}

fn string_pat(s: &str) -> Pattern {
    Pattern { kind: PatternKind::LitString(s.to_string()), span: span() }
}

fn tuple_pat(pats: Vec<Pattern>) -> Pattern {
    Pattern { kind: PatternKind::Tuple(pats), span: span() }
}

fn struct_pat(name: &str, fields: Vec<(&str, Option<Pattern>)>) -> Pattern {
    Pattern {
        kind: PatternKind::Struct {
            name: ident(name),
            fields: fields
                .into_iter()
                .map(|(n, p)| PatternField {
                    name: ident(n),
                    pattern: p,
                    span: span(),
                })
                .collect(),
        },
        span: span(),
    }
}

fn tuple_expr(elems: Vec<Expr>) -> Expr {
    Expr { kind: ExprKind::TupleLit(elems), span: span() }
}

fn struct_lit(name: &str, fields: Vec<(&str, Expr)>) -> Expr {
    Expr {
        kind: ExprKind::StructLit {
            name: ident(name),
            fields: fields
                .into_iter()
                .map(|(n, v)| FieldInit {
                    name: ident(n),
                    value: v,
                    span: span(),
                })
                .collect(),
        },
        span: span(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Evaluation helpers
// ═══════════════════════════════════════════════════════════════════════

fn eval_program(program: &Program) -> (Value, Vec<String>) {
    let mut interp = Interpreter::new(0);
    let result = interp.exec(program).expect("eval failed");
    (result, interp.output.clone())
}

fn mir_exec_program(program: &Program) -> (Value, Vec<String>) {
    let (result, executor) =
        cjc_mir_exec::run_program_with_executor(program, 0).expect("MIR exec failed");
    (result, executor.output)
}

fn parse_ok(src: &str) -> Program {
    let (program, diags) = parse_source(src);
    if diags.has_errors() {
        let rendered = diags.render_all(src, "<test>");
        panic!("unexpected parse errors:\n{}", rendered);
    }
    program
}

fn run_source(src: &str) -> (Value, Vec<String>) {
    mir_exec_program(&parse_ok(src))
}

fn run_source_eval(src: &str) -> (Value, Vec<String>) {
    eval_program(&parse_ok(src))
}

// ═══════════════════════════════════════════════════════════════════════
// 1. LEXER TESTS
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_lexer_match_keyword() {
    let (tokens, _) = Lexer::new("match").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::Match);
}

#[test]
fn test_lexer_underscore_token() {
    let (tokens, _) = Lexer::new("_").tokenize();
    assert_eq!(tokens[0].kind, TokenKind::Underscore);
}

#[test]
fn test_lexer_match_expression_tokens() {
    let (tokens, _) = Lexer::new("match x { 42 => true, _ => false }").tokenize();
    let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
    assert_eq!(kinds[0], TokenKind::Match);
    assert_eq!(kinds[1], TokenKind::Ident);      // x
    assert_eq!(kinds[2], TokenKind::LBrace);
    assert_eq!(kinds[3], TokenKind::IntLit);      // 42
    assert_eq!(kinds[4], TokenKind::FatArrow);
    assert_eq!(kinds[5], TokenKind::True);
    assert_eq!(kinds[6], TokenKind::Comma);
    assert_eq!(kinds[7], TokenKind::Underscore);  // _
    assert_eq!(kinds[8], TokenKind::FatArrow);
    assert_eq!(kinds[9], TokenKind::False);
    assert_eq!(kinds[10], TokenKind::RBrace);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. PARSER TESTS
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_parse_match_with_wildcard() {
    let prog = parse_ok("fn main() -> i64 { match x { _ => 42 } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { scrutinee, arms } => {
                    assert!(matches!(scrutinee.kind, ExprKind::Ident(_)));
                    assert_eq!(arms.len(), 1);
                    assert!(matches!(arms[0].pattern.kind, PatternKind::Wildcard));
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_literals() {
    let prog = parse_ok("fn main() -> i64 { match x { 1 => 10, 2 => 20, _ => 0 } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 3);
                    assert!(matches!(arms[0].pattern.kind, PatternKind::LitInt(1)));
                    assert!(matches!(arms[1].pattern.kind, PatternKind::LitInt(2)));
                    assert!(matches!(arms[2].pattern.kind, PatternKind::Wildcard));
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_binding() {
    let prog = parse_ok("fn main() -> i64 { match x { n => n } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 1);
                    match &arms[0].pattern.kind {
                        PatternKind::Binding(id) => assert_eq!(id.name, "n"),
                        _ => panic!("expected Binding"),
                    }
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_tuple_pattern() {
    let prog = parse_ok("fn main() -> i64 { match pair { (a, b) => a } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    match &arms[0].pattern.kind {
                        PatternKind::Tuple(pats) => {
                            assert_eq!(pats.len(), 2);
                            match &pats[0].kind {
                                PatternKind::Binding(id) => assert_eq!(id.name, "a"),
                                _ => panic!("expected Binding"),
                            }
                        }
                        _ => panic!("expected Tuple"),
                    }
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_struct_pattern() {
    let prog = parse_ok("fn main() -> i64 { match p { Point { x, y } => x } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    match &arms[0].pattern.kind {
                        PatternKind::Struct { name, fields } => {
                            assert_eq!(name.name, "Point");
                            assert_eq!(fields.len(), 2);
                            assert_eq!(fields[0].name.name, "x");
                            assert!(fields[0].pattern.is_none());
                            assert_eq!(fields[1].name.name, "y");
                        }
                        _ => panic!("expected Struct pattern"),
                    }
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_struct_explicit_bindings() {
    let prog = parse_ok("fn main() -> i64 { match p { Point { x: px, y: py } => px } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    match &arms[0].pattern.kind {
                        PatternKind::Struct { fields, .. } => {
                            assert_eq!(fields[0].name.name, "x");
                            match fields[0].pattern.as_ref().unwrap().kind {
                                PatternKind::Binding(ref id) => assert_eq!(id.name, "px"),
                                _ => panic!("expected Binding"),
                            }
                        }
                        _ => panic!("expected Struct pattern"),
                    }
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_bool_patterns() {
    let prog = parse_ok("fn main() -> i64 { match flag { true => 1, false => 0 } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 2);
                    assert!(matches!(arms[0].pattern.kind, PatternKind::LitBool(true)));
                    assert!(matches!(arms[1].pattern.kind, PatternKind::LitBool(false)));
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_match_with_negative_literal() {
    let prog = parse_ok("fn main() -> i64 { match x { -1 => 1, _ => 0 } }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let block_expr = f.body.expr.as_ref().unwrap();
            match &block_expr.kind {
                ExprKind::Match { arms, .. } => {
                    assert!(matches!(arms[0].pattern.kind, PatternKind::LitInt(-1)));
                }
                _ => panic!("expected Match"),
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_tuple_literal() {
    let prog = parse_ok("fn main() -> i64 { let t = (1, 2, 3); 0 }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            if let StmtKind::Let(l) = &f.body.stmts[0].kind {
                match &l.init.kind {
                    ExprKind::TupleLit(elems) => assert_eq!(elems.len(), 3),
                    _ => panic!("expected TupleLit, got {:?}", l.init.kind),
                }
            }
        }
        _ => panic!("expected Fn"),
    }
}

#[test]
fn test_parse_single_paren_not_tuple() {
    let prog = parse_ok("fn main() -> i64 { (42) }");
    match &prog.declarations[0].kind {
        DeclKind::Fn(f) => {
            let e = f.body.expr.as_ref().unwrap();
            assert!(matches!(e.kind, ExprKind::IntLit(42)));
        }
        _ => panic!("expected Fn"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. AST TREE-WALK EVALUATOR TESTS (cjc-eval)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_eval_match_wildcard() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(int_expr(42), vec![(wildcard_pat(), int_expr(100))]))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(100)));
}

#[test]
fn test_eval_match_literal_int() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(2),
                vec![
                    (int_pat(1), int_expr(10)),
                    (int_pat(2), int_expr(20)),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(20)));
}

#[test]
fn test_eval_match_literal_fallthrough() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(99),
                vec![
                    (int_pat(1), int_expr(10)),
                    (int_pat(2), int_expr(20)),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(0)));
}

#[test]
fn test_eval_match_binding() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(42),
                vec![(
                    binding_pat("n"),
                    binary(BinOp::Add, ident_expr("n"), int_expr(1)),
                )],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(43)));
}

#[test]
fn test_eval_match_bool() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                bool_expr(true),
                vec![
                    (bool_pat(true), int_expr(1)),
                    (bool_pat(false), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_eval_match_string() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                string_expr("hello"),
                vec![
                    (string_pat("hello"), int_expr(1)),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_eval_match_tuple_destructure() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                tuple_expr(vec![int_expr(1), int_expr(2)]),
                vec![(
                    tuple_pat(vec![binding_pat("a"), binding_pat("b")]),
                    binary(BinOp::Add, ident_expr("a"), ident_expr("b")),
                )],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn test_eval_match_nested_tuple() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                tuple_expr(vec![int_expr(1), tuple_expr(vec![int_expr(2), int_expr(3)])]),
                vec![(
                    tuple_pat(vec![
                        binding_pat("a"),
                        tuple_pat(vec![binding_pat("b"), binding_pat("c")]),
                    ]),
                    binary(BinOp::Add, binary(BinOp::Add, ident_expr("a"), ident_expr("b")), ident_expr("c")),
                )],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(6)));
}

#[test]
fn test_eval_match_struct_destructure() {
    let program = Program {
        declarations: vec![
            make_struct_decl("Point", vec![("x", "i64"), ("y", "i64")]),
            make_fn_decl(
                "main", vec![],
                make_block(
                    vec![let_stmt("p", struct_lit("Point", vec![("x", int_expr(3)), ("y", int_expr(4))]))],
                    Some(match_expr(
                        ident_expr("p"),
                        vec![(
                            struct_pat("Point", vec![("x", None), ("y", None)]),
                            binary(BinOp::Add, ident_expr("x"), ident_expr("y")),
                        )],
                    )),
                ),
            ),
        ],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(7)));
}

#[test]
fn test_eval_match_struct_explicit_binding() {
    let program = Program {
        declarations: vec![
            make_struct_decl("Point", vec![("x", "i64"), ("y", "i64")]),
            make_fn_decl(
                "main", vec![],
                make_block(
                    vec![let_stmt("p", struct_lit("Point", vec![("x", int_expr(10)), ("y", int_expr(20))]))],
                    Some(match_expr(
                        ident_expr("p"),
                        vec![(
                            struct_pat("Point", vec![("x", Some(binding_pat("a"))), ("y", Some(binding_pat("b")))]),
                            binary(BinOp::Add, ident_expr("a"), ident_expr("b")),
                        )],
                    )),
                ),
            ),
        ],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(30)));
}

#[test]
fn test_eval_match_first_arm_wins() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(1),
                vec![
                    (binding_pat("n"), binary(BinOp::Mul, ident_expr("n"), int_expr(10))),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(10)));
}

#[test]
fn test_eval_match_tuple_with_literal() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(
                vec![let_stmt("pair", tuple_expr(vec![int_expr(1), int_expr(2)]))],
                Some(match_expr(
                    ident_expr("pair"),
                    vec![
                        (tuple_pat(vec![int_pat(0), binding_pat("b")]), ident_expr("b")),
                        (tuple_pat(vec![int_pat(1), binding_pat("b")]), binary(BinOp::Mul, ident_expr("b"), int_expr(10))),
                        (wildcard_pat(), int_expr(0)),
                    ],
                )),
            ),
        )],
    };
    let (result, _) = eval_program(&program);
    assert!(matches!(result, Value::Int(20)));
}

// ═══════════════════════════════════════════════════════════════════════
// 4. FULL PIPELINE TESTS (AST → HIR → MIR → MIR-exec)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_mir_match_wildcard() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(int_expr(42), vec![(wildcard_pat(), int_expr(100))]))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(100)));
}

#[test]
fn test_mir_match_literal_int() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(2),
                vec![
                    (int_pat(1), int_expr(10)),
                    (int_pat(2), int_expr(20)),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(20)));
}

#[test]
fn test_mir_match_binding() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(42),
                vec![(binding_pat("n"), binary(BinOp::Add, ident_expr("n"), int_expr(1)))],
            ))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(43)));
}

#[test]
fn test_mir_match_tuple() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                tuple_expr(vec![int_expr(1), int_expr(2)]),
                vec![(
                    tuple_pat(vec![binding_pat("a"), binding_pat("b")]),
                    binary(BinOp::Add, ident_expr("a"), ident_expr("b")),
                )],
            ))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn test_mir_match_struct() {
    let program = Program {
        declarations: vec![
            make_struct_decl("Point", vec![("x", "i64"), ("y", "i64")]),
            make_fn_decl(
                "main", vec![],
                make_block(
                    vec![let_stmt("p", struct_lit("Point", vec![("x", int_expr(5)), ("y", int_expr(7))]))],
                    Some(match_expr(
                        ident_expr("p"),
                        vec![(
                            struct_pat("Point", vec![("x", None), ("y", None)]),
                            binary(BinOp::Add, ident_expr("x"), ident_expr("y")),
                        )],
                    )),
                ),
            ),
        ],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(12)));
}

#[test]
fn test_mir_match_nested_tuple() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                tuple_expr(vec![int_expr(10), tuple_expr(vec![int_expr(20), int_expr(30)])]),
                vec![(
                    tuple_pat(vec![
                        binding_pat("a"),
                        tuple_pat(vec![binding_pat("b"), binding_pat("c")]),
                    ]),
                    binary(BinOp::Add, binary(BinOp::Add, ident_expr("a"), ident_expr("b")), ident_expr("c")),
                )],
            ))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(60)));
}

#[test]
fn test_mir_match_fallthrough() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(99),
                vec![
                    (int_pat(1), int_expr(10)),
                    (int_pat(2), int_expr(20)),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(0)));
}

#[test]
fn test_mir_match_bool() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                bool_expr(false),
                vec![
                    (bool_pat(true), int_expr(1)),
                    (bool_pat(false), int_expr(0)),
                ],
            ))),
        )],
    };
    let (result, _) = mir_exec_program(&program);
    assert!(matches!(result, Value::Int(0)));
}

// ═══════════════════════════════════════════════════════════════════════
// 5. PARITY TESTS: AST-eval and MIR-exec produce identical results
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_parity_match_int_literals() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(vec![], Some(match_expr(
                int_expr(3),
                vec![
                    (int_pat(1), int_expr(100)),
                    (int_pat(2), int_expr(200)),
                    (int_pat(3), int_expr(300)),
                    (wildcard_pat(), int_expr(0)),
                ],
            ))),
        )],
    };
    let (eval_r, _) = eval_program(&program);
    let (mir_r, _) = mir_exec_program(&program);
    assert!(matches!(eval_r, Value::Int(300)));
    assert!(matches!(mir_r, Value::Int(300)));
}

#[test]
fn test_parity_match_tuple_destructure() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(
                vec![let_stmt("t", tuple_expr(vec![int_expr(10), int_expr(20)]))],
                Some(match_expr(
                    ident_expr("t"),
                    vec![(
                        tuple_pat(vec![binding_pat("a"), binding_pat("b")]),
                        binary(BinOp::Mul, ident_expr("a"), ident_expr("b")),
                    )],
                )),
            ),
        )],
    };
    let (eval_r, _) = eval_program(&program);
    let (mir_r, _) = mir_exec_program(&program);
    assert!(matches!(eval_r, Value::Int(200)));
    assert!(matches!(mir_r, Value::Int(200)));
}

#[test]
fn test_parity_match_struct_destructure() {
    let program = Program {
        declarations: vec![
            make_struct_decl("Vec2", vec![("x", "i64"), ("y", "i64")]),
            make_fn_decl(
                "main", vec![],
                make_block(
                    vec![let_stmt("v", struct_lit("Vec2", vec![("x", int_expr(3)), ("y", int_expr(4))]))],
                    Some(match_expr(
                        ident_expr("v"),
                        vec![(
                            struct_pat("Vec2", vec![("x", None), ("y", None)]),
                            binary(BinOp::Add, ident_expr("x"), ident_expr("y")),
                        )],
                    )),
                ),
            ),
        ],
    };
    let (eval_r, _) = eval_program(&program);
    let (mir_r, _) = mir_exec_program(&program);
    assert!(matches!(eval_r, Value::Int(7)));
    assert!(matches!(mir_r, Value::Int(7)));
}

// ═══════════════════════════════════════════════════════════════════════
// 6. MATCH COMBINED WITH OTHER FEATURES
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_match_in_function_body() {
    let program = Program {
        declarations: vec![
            make_fn_decl(
                "classify", vec!["x"],
                make_block(vec![], Some(match_expr(
                    ident_expr("x"),
                    vec![
                        (int_pat(0), int_expr(0)),
                        (int_pat(1), int_expr(1)),
                        (wildcard_pat(), int_expr(2)),
                    ],
                ))),
            ),
            make_fn_decl(
                "main", vec![],
                make_block(vec![], Some(binary(
                    BinOp::Add,
                    binary(
                        BinOp::Add,
                        call(ident_expr("classify"), vec![int_expr(0)]),
                        call(ident_expr("classify"), vec![int_expr(1)]),
                    ),
                    call(ident_expr("classify"), vec![int_expr(99)]),
                ))),
            ),
        ],
    };
    let (eval_r, _) = eval_program(&program);
    let (mir_r, _) = mir_exec_program(&program);
    // 0 + 1 + 2 = 3
    assert!(matches!(eval_r, Value::Int(3)));
    assert!(matches!(mir_r, Value::Int(3)));
}

#[test]
fn test_match_with_print_output() {
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(
                vec![
                    let_stmt("result", match_expr(
                        int_expr(42),
                        vec![
                            (int_pat(42), string_expr("found it")),
                            (wildcard_pat(), string_expr("nope")),
                        ],
                    )),
                    expr_stmt(call(ident_expr("print"), vec![ident_expr("result")])),
                ],
                None,
            ),
        )],
    };
    let (_, eval_out) = eval_program(&program);
    let (_, mir_out) = mir_exec_program(&program);
    assert_eq!(eval_out, vec!["found it"]);
    assert_eq!(mir_out, vec!["found it"]);
}

#[test]
fn test_match_inside_while_loop() {
    let match_e = match_expr(
        ident_expr("i"),
        vec![
            (int_pat(0), int_expr(10)),
            (int_pat(1), int_expr(20)),
            (wildcard_pat(), int_expr(1)),
        ],
    );
    let program = Program {
        declarations: vec![make_fn_decl(
            "main", vec![],
            make_block(
                vec![
                    let_mut_stmt("i", int_expr(0)),
                    let_mut_stmt("sum", int_expr(0)),
                    Stmt {
                        kind: StmtKind::While(WhileStmt {
                            condition: binary(BinOp::Lt, ident_expr("i"), int_expr(5)),
                            body: make_block(
                                vec![
                                    expr_stmt(assign_expr(ident_expr("sum"), binary(BinOp::Add, ident_expr("sum"), match_e.clone()))),
                                    expr_stmt(assign_expr(ident_expr("i"), binary(BinOp::Add, ident_expr("i"), int_expr(1)))),
                                ],
                                None,
                            ),
                        }),
                        span: span(),
                    },
                ],
                Some(ident_expr("sum")),
            ),
        )],
    };
    let (eval_r, _) = eval_program(&program);
    let (mir_r, _) = mir_exec_program(&program);
    // sum = 10 + 20 + 1 + 1 + 1 = 33
    assert!(matches!(eval_r, Value::Int(33)));
    assert!(matches!(mir_r, Value::Int(33)));
}

// ═══════════════════════════════════════════════════════════════════════
// 7. END-TO-END TESTS (from source text through full pipeline)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_e2e_match_basic() {
    let (result, _) = run_source(r#"
        fn main() -> i64 {
            match 42 {
                1 => 10,
                42 => 420,
                _ => 0
            }
        }
    "#);
    assert!(matches!(result, Value::Int(420)));
}

#[test]
fn test_e2e_match_binding() {
    let (result, _) = run_source(r#"
        fn main() -> i64 {
            match 7 {
                n => n * n
            }
        }
    "#);
    assert!(matches!(result, Value::Int(49)));
}

#[test]
fn test_e2e_match_tuple() {
    let (result, _) = run_source(r#"
        fn main() -> i64 {
            let pair = (3, 4);
            match pair {
                (a, b) => a + b
            }
        }
    "#);
    assert!(matches!(result, Value::Int(7)));
}

#[test]
fn test_e2e_match_struct() {
    let (result, _) = run_source(r#"
        struct Point { x: i64, y: i64 }
        fn main() -> i64 {
            let p = Point { x: 10, y: 20 };
            match p {
                Point { x, y } => x + y
            }
        }
    "#);
    assert!(matches!(result, Value::Int(30)));
}

#[test]
fn test_e2e_match_in_function() {
    let (result, _) = run_source(r#"
        fn abs_val(x: i64) -> i64 {
            match x {
                0 => 0,
                n => n
            }
        }
        fn main() -> i64 {
            abs_val(0) + abs_val(5)
        }
    "#);
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_e2e_match_fallthrough() {
    let (result, _) = run_source(r#"
        fn main() -> i64 {
            let x = 5;
            match x {
                1 => 100,
                2 => 200,
                3 => 300,
                _ => 999
            }
        }
    "#);
    assert!(matches!(result, Value::Int(999)));
}

#[test]
fn test_e2e_match_bool() {
    let (result, _) = run_source(r#"
        fn main() -> i64 {
            let flag = false;
            match flag {
                true => 1,
                false => 0
            }
        }
    "#);
    assert!(matches!(result, Value::Int(0)));
}

#[test]
fn test_e2e_parity() {
    let src = r#"
        struct Pair { a: i64, b: i64 }
        fn main() -> i64 {
            let p = Pair { a: 7, b: 3 };
            match p {
                Pair { a, b } => a * b
            }
        }
    "#;
    let program = parse_ok(src);
    let (eval_r, _) = eval_program(&program);
    let (mir_r, _) = mir_exec_program(&program);
    assert!(matches!(eval_r, Value::Int(21)));
    assert!(matches!(mir_r, Value::Int(21)));
}

// ═══════════════════════════════════════════════════════════════════════
// 8. HIR LOWERING TESTS
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_hir_lower_match() {
    let program = parse_ok(r#"
        fn main() -> i64 {
            match 42 {
                1 => 10,
                _ => 0
            }
        }
    "#);
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);
    let fn_item = hir.items.iter().find(|item| matches!(item, cjc_hir::HirItem::Fn(f) if f.name == "main"));
    assert!(fn_item.is_some());
}

#[test]
fn test_hir_lower_tuple() {
    let program = parse_ok(r#"
        fn main() -> i64 {
            let t = (1, 2, 3);
            match t {
                (a, b, c) => a + b + c
            }
        }
    "#);
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);
    assert!(!hir.items.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// 9. MIR LOWERING TESTS
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_mir_lower_match() {
    let program = parse_ok(r#"
        fn main() -> i64 {
            match 42 {
                1 => 10,
                _ => 0
            }
        }
    "#);
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(&program);
    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);
    assert!(mir.functions.len() >= 2);
}

#[test]
fn test_mir_lower_tuple_and_struct_patterns() {
    let program = parse_ok(r#"
        struct Point { x: i64, y: i64 }
        fn main() -> i64 {
            let t = (1, 2);
            let p = Point { x: 3, y: 4 };
            let a = match t { (x, y) => x + y };
            let b = match p { Point { x, y } => x + y };
            a + b
        }
    "#);
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(&program);
    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);
    assert!(!mir.functions.is_empty());
    assert_eq!(mir.struct_defs.len(), 1);
    assert_eq!(mir.struct_defs[0].name, "Point");
}
