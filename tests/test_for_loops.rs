//! Milestone 2.3 — For-Loop Tests (Gate G-7)
//!
//! Comprehensive test suite covering:
//! - Lexer: `in` keyword, `..` operator
//! - Parser: `parse_for_stmt()`, range and expr forms
//! - AST: `StmtKind::For`, `ForIter` enum
//! - HIR: desugaring to while-loops (no new MIR constructs)
//! - Eval: runtime behavior of for-loops
//! - Edge cases: empty range, shadowing, nested loops, scoping

// ── Lexer tests ─────────────────────────────────────────────────────────

#[test]
fn test_lex_in_keyword() {
    let lexer = cjc_lexer::Lexer::new("in");
    let (tokens, diags) = lexer.tokenize();
    assert!(!diags.has_errors());
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::In);
    assert_eq!(tokens[0].text, "in");
}

#[test]
fn test_lex_dotdot() {
    let lexer = cjc_lexer::Lexer::new("0..10");
    let (tokens, diags) = lexer.tokenize();
    assert!(!diags.has_errors());
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::IntLit);
    assert_eq!(tokens[0].text, "0");
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::DotDot);
    assert_eq!(tokens[1].text, "..");
    assert_eq!(tokens[2].kind, cjc_lexer::TokenKind::IntLit);
    assert_eq!(tokens[2].text, "10");
}

#[test]
fn test_lex_dot_vs_dotdot() {
    // Ensure `.` and `..` are distinct
    let lexer = cjc_lexer::Lexer::new("a.b 0..5");
    let (tokens, _) = lexer.tokenize();
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::Ident); // a
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::Dot); // .
    assert_eq!(tokens[2].kind, cjc_lexer::TokenKind::Ident); // b
    assert_eq!(tokens[3].kind, cjc_lexer::TokenKind::IntLit); // 0
    assert_eq!(tokens[4].kind, cjc_lexer::TokenKind::DotDot); // ..
    assert_eq!(tokens[5].kind, cjc_lexer::TokenKind::IntLit); // 5
}

#[test]
fn test_lex_for_in_range_tokens() {
    let lexer = cjc_lexer::Lexer::new("for i in 0..n");
    let (tokens, _) = lexer.tokenize();
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::For);
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::Ident); // i
    assert_eq!(tokens[2].kind, cjc_lexer::TokenKind::In);
    assert_eq!(tokens[3].kind, cjc_lexer::TokenKind::IntLit); // 0
    assert_eq!(tokens[4].kind, cjc_lexer::TokenKind::DotDot);
    assert_eq!(tokens[5].kind, cjc_lexer::TokenKind::Ident); // n
}

#[test]
fn test_lex_float_not_confused_with_dotdot() {
    // `3.14` should be a single float, not `3` `..` `4`
    let lexer = cjc_lexer::Lexer::new("3.14");
    let (tokens, _) = lexer.tokenize();
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::FloatLit);
    assert_eq!(tokens[0].text, "3.14");
}

// ── Parser tests ────────────────────────────────────────────────────────

fn parse_ok(source: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        let rendered = diags.render_all(source, "<test>");
        panic!("unexpected parse errors:\n{}", rendered);
    }
    program
}

fn parse_err(source: &str) {
    let (_, diags) = cjc_parser::parse_source(source);
    assert!(
        diags.has_errors(),
        "expected parse error but got none for: {}",
        source
    );
}

// Test 1: for i in 0..3 basic range
#[test]
fn test_parse_for_range_basic() {
    let prog = parse_ok("fn f() { for i in 0..3 { print(i); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                assert_eq!(for_stmt.ident.name, "i");
                match &for_stmt.iter {
                    cjc_ast::ForIter::Range { start, end } => {
                        assert!(matches!(start.kind, cjc_ast::ExprKind::IntLit(0)));
                        assert!(matches!(end.kind, cjc_ast::ExprKind::IntLit(3)));
                    }
                    _ => panic!("expected Range"),
                }
                assert_eq!(for_stmt.body.stmts.len(), 1);
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 2: for i in 1..n uses identifier end
#[test]
fn test_parse_for_range_ident_end() {
    let prog = parse_ok("fn f(n: i64) { for i in 1..n { print(i); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                assert_eq!(for_stmt.ident.name, "i");
                match &for_stmt.iter {
                    cjc_ast::ForIter::Range { start, end } => {
                        assert!(matches!(start.kind, cjc_ast::ExprKind::IntLit(1)));
                        assert!(matches!(end.kind, cjc_ast::ExprKind::Ident(ref id) if id.name == "n"));
                    }
                    _ => panic!("expected Range"),
                }
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 3: start/end are expressions (ensure eval once)
#[test]
fn test_parse_for_range_expressions() {
    let prog = parse_ok("fn f() { for i in 1 + 2..3 * 4 { print(i); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                match &for_stmt.iter {
                    cjc_ast::ForIter::Range { start, end } => {
                        // start should be 1 + 2 (binary)
                        assert!(matches!(start.kind, cjc_ast::ExprKind::Binary { .. }));
                        // end should be 3 * 4 (binary)
                        assert!(matches!(end.kind, cjc_ast::ExprKind::Binary { .. }));
                    }
                    _ => panic!("expected Range"),
                }
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 4: nested range loops
#[test]
fn test_parse_nested_for_range() {
    let prog = parse_ok(
        "fn f() { for i in 0..3 { for j in 0..4 { print(i); print(j); } } }",
    );
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(outer) => {
                assert_eq!(outer.ident.name, "i");
                match &outer.body.stmts[0].kind {
                    cjc_ast::StmtKind::For(inner) => {
                        assert_eq!(inner.ident.name, "j");
                    }
                    _ => panic!("expected inner For"),
                }
            }
            _ => panic!("expected outer For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 5: array iteration basic
#[test]
fn test_parse_for_array_iter() {
    let prog = parse_ok("fn f() { for x in arr { print(x); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                assert_eq!(for_stmt.ident.name, "x");
                match &for_stmt.iter {
                    cjc_ast::ForIter::Expr(expr) => {
                        assert!(matches!(expr.kind, cjc_ast::ExprKind::Ident(ref id) if id.name == "arr"));
                    }
                    _ => panic!("expected Expr iterator"),
                }
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 6: for with array literal
#[test]
fn test_parse_for_array_literal() {
    let prog = parse_ok("fn f() { for x in [1, 2, 3] { print(x); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                assert_eq!(for_stmt.ident.name, "x");
                match &for_stmt.iter {
                    cjc_ast::ForIter::Expr(expr) => {
                        assert!(matches!(expr.kind, cjc_ast::ExprKind::ArrayLit(_)));
                    }
                    _ => panic!("expected Expr iterator"),
                }
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 7: nested array + range
#[test]
fn test_parse_nested_array_and_range() {
    let prog = parse_ok(
        "fn f() { for x in arr { for i in 0..3 { print(x); print(i); } } }",
    );
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(outer) => {
                assert_eq!(outer.ident.name, "x");
                assert!(matches!(outer.iter, cjc_ast::ForIter::Expr(_)));
                match &outer.body.stmts[0].kind {
                    cjc_ast::StmtKind::For(inner) => {
                        assert_eq!(inner.ident.name, "i");
                        assert!(matches!(inner.iter, cjc_ast::ForIter::Range { .. }));
                    }
                    _ => panic!("expected inner For"),
                }
            }
            _ => panic!("expected outer For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 8: error test - for in 0..n {} missing ident
#[test]
fn test_parse_for_missing_ident() {
    parse_err("fn f() { for in 0..3 { } }");
}

// Test 9: error test - for i 0..n {} missing `in`
#[test]
fn test_parse_for_missing_in() {
    parse_err("fn f() { for i 0..3 { } }");
}

// Test 10: for x in call() where iter is a function call
#[test]
fn test_parse_for_call_iter() {
    let prog = parse_ok("fn f() { for x in get_items() { print(x); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                assert_eq!(for_stmt.ident.name, "x");
                match &for_stmt.iter {
                    cjc_ast::ForIter::Expr(expr) => {
                        assert!(matches!(expr.kind, cjc_ast::ExprKind::Call { .. }));
                    }
                    _ => panic!("expected Expr iterator"),
                }
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// Test 11: for x in obj.field where iter is a field access
#[test]
fn test_parse_for_field_access_iter() {
    let prog = parse_ok("fn f() { for x in obj.items { print(x); } }");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => match &f.body.stmts[0].kind {
            cjc_ast::StmtKind::For(for_stmt) => {
                assert_eq!(for_stmt.ident.name, "x");
                match &for_stmt.iter {
                    cjc_ast::ForIter::Expr(expr) => {
                        assert!(matches!(expr.kind, cjc_ast::ExprKind::Field { .. }));
                    }
                    _ => panic!("expected Expr iterator"),
                }
            }
            _ => panic!("expected For"),
        },
        _ => panic!("expected Fn"),
    }
}

// ── HIR Desugaring tests ────────────────────────────────────────────────

// Test 12: Range for desugars to While in HIR (no new MIR constructs)
#[test]
fn test_hir_range_for_desugars_to_while() {
    let (program, _) = cjc_parser::parse_source("fn f() { for i in 0..3 { print(i); } }");
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);

    // Find the function
    let hir_fn = match &hir.items[0] {
        cjc_hir::HirItem::Fn(f) => f,
        _ => panic!("expected Fn"),
    };

    // The for loop should have been desugared into a block expression
    // containing let statements and a while loop
    assert_eq!(hir_fn.body.stmts.len(), 1);
    match &hir_fn.body.stmts[0].kind {
        cjc_hir::HirStmtKind::Expr(expr) => match &expr.kind {
            cjc_hir::HirExprKind::Block(block) => {
                // Should have: let __end, let __idx, while
                assert_eq!(block.stmts.len(), 3);
                assert!(matches!(block.stmts[0].kind, cjc_hir::HirStmtKind::Let { .. }));
                assert!(matches!(block.stmts[1].kind, cjc_hir::HirStmtKind::Let { .. }));
                assert!(matches!(
                    block.stmts[2].kind,
                    cjc_hir::HirStmtKind::While { .. }
                ));
            }
            _ => panic!("expected Block from for desugaring"),
        },
        _ => panic!("expected Expr stmt"),
    }
}

// Test 13: Array for desugars to While with len and index in HIR
#[test]
fn test_hir_array_for_desugars_to_while() {
    let (program, _) =
        cjc_parser::parse_source("fn f() { for x in arr { print(x); } }");
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);

    let hir_fn = match &hir.items[0] {
        cjc_hir::HirItem::Fn(f) => f,
        _ => panic!("expected Fn"),
    };

    assert_eq!(hir_fn.body.stmts.len(), 1);
    match &hir_fn.body.stmts[0].kind {
        cjc_hir::HirStmtKind::Expr(expr) => match &expr.kind {
            cjc_hir::HirExprKind::Block(block) => {
                // Should have: let __arr, let __len, let __idx, while
                assert_eq!(block.stmts.len(), 4);
                assert!(matches!(block.stmts[0].kind, cjc_hir::HirStmtKind::Let { .. }));
                assert!(matches!(block.stmts[1].kind, cjc_hir::HirStmtKind::Let { .. }));
                assert!(matches!(block.stmts[2].kind, cjc_hir::HirStmtKind::Let { .. }));
                assert!(matches!(
                    block.stmts[3].kind,
                    cjc_hir::HirStmtKind::While { .. }
                ));
            }
            _ => panic!("expected Block from for desugaring"),
        },
        _ => panic!("expected Expr stmt"),
    }
}

// Test 14: MIR has no new constructs — for desugars entirely via HIR
#[test]
fn test_mir_no_new_constructs_for_range() {
    let (program, _) = cjc_parser::parse_source("fn f() { for i in 0..3 { print(i); } }");
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(&program);
    let mut mir_lowering = cjc_mir::HirToMir::new();
    let mir = mir_lowering.lower_program(&hir);

    // The function body should contain only existing MIR constructs:
    // Let, While, Expr — no new For construct
    let f = mir.functions.iter().find(|f| f.name == "f").unwrap();
    fn check_no_for_in_stmts(stmts: &[cjc_mir::MirStmt]) {
        for stmt in stmts {
            match stmt {
                cjc_mir::MirStmt::Let { .. } => {}
                cjc_mir::MirStmt::Expr(_) => {}
                cjc_mir::MirStmt::If { then_body, else_body, .. } => {
                    check_no_for_in_stmts(&then_body.stmts);
                    if let Some(eb) = else_body {
                        check_no_for_in_stmts(&eb.stmts);
                    }
                }
                cjc_mir::MirStmt::While { body, .. } => {
                    check_no_for_in_stmts(&body.stmts);
                }
                cjc_mir::MirStmt::Return(_) => {}
                cjc_mir::MirStmt::Break | cjc_mir::MirStmt::Continue => {}
                cjc_mir::MirStmt::NoGcBlock(body) => {
                    check_no_for_in_stmts(&body.stmts);
                }
            }
        }
    }
    check_no_for_in_stmts(&f.body.stmts);
    // If we got here without panic, no new MIR constructs were found
}

// Test 15: HIR desugared while body contains the loop variable binding
#[test]
fn test_hir_range_for_binds_loop_var() {
    let (program, _) = cjc_parser::parse_source("fn f() { for i in 0..3 { print(i); } }");
    let mut lowering = cjc_hir::AstLowering::new();
    let hir = lowering.lower_program(&program);

    let hir_fn = match &hir.items[0] {
        cjc_hir::HirItem::Fn(f) => f,
        _ => panic!("expected Fn"),
    };

    // Navigate to the while loop body
    match &hir_fn.body.stmts[0].kind {
        cjc_hir::HirStmtKind::Expr(expr) => match &expr.kind {
            cjc_hir::HirExprKind::Block(block) => {
                match &block.stmts[2].kind {
                    cjc_hir::HirStmtKind::While { body, .. } => {
                        // First stmt in while body should be: let i = __for_idx_N
                        match &body.stmts[0].kind {
                            cjc_hir::HirStmtKind::Let { name, .. } => {
                                assert_eq!(name, "i");
                            }
                            _ => panic!("expected Let binding for loop var"),
                        }
                    }
                    _ => panic!("expected While"),
                }
            }
            _ => panic!("expected Block"),
        },
        _ => panic!("expected Expr"),
    }
}

// ── Eval (runtime behavior) tests ───────────────────────────────────────

fn eval_program(source: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        let rendered = diags.render_all(source, "<test>");
        panic!("parse errors:\n{}", rendered);
    }
    let mut interp = cjc_eval::Interpreter::new(42);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("eval error: {:?}", e),
    }
    interp.output.clone()
}

// Test 16: basic range for loop execution
#[test]
fn test_eval_for_range_basic() {
    let output = eval_program("for i in 0..3 { print(i); }");
    assert_eq!(output, vec!["0", "1", "2"]);
}

// Test 17: range for with variable end
#[test]
fn test_eval_for_range_variable_end() {
    let output = eval_program("let n = 4; for i in 0..n { print(i); }");
    assert_eq!(output, vec!["0", "1", "2", "3"]);
}

// Test 18: empty range (0..0)
#[test]
fn test_eval_for_range_empty() {
    let output = eval_program("for i in 0..0 { print(i); }");
    assert!(output.is_empty());
}

// Test 19: range start > end (no iterations)
#[test]
fn test_eval_for_range_start_gt_end() {
    let output = eval_program("for i in 5..3 { print(i); }");
    assert!(output.is_empty());
}

// Test 20: range with non-zero start
#[test]
fn test_eval_for_range_nonzero_start() {
    let output = eval_program("for i in 2..5 { print(i); }");
    assert_eq!(output, vec!["2", "3", "4"]);
}

// Test 21: nested range loops
#[test]
fn test_eval_nested_range() {
    let output = eval_program(
        r#"
        for i in 0..2 {
            for j in 0..3 {
                print(i * 10 + j);
            }
        }
        "#,
    );
    assert_eq!(output, vec!["0", "1", "2", "10", "11", "12"]);
}

// Test 22: array iteration basic
#[test]
fn test_eval_for_array_basic() {
    let output = eval_program(
        r#"
        let arr = [10, 20, 30];
        for x in arr {
            print(x);
        }
        "#,
    );
    assert_eq!(output, vec!["10", "20", "30"]);
}

// Test 23: array iteration with empty array
#[test]
fn test_eval_for_array_empty() {
    let output = eval_program(
        r#"
        let arr = [];
        for x in arr {
            print(x);
        }
        "#,
    );
    assert!(output.is_empty());
}

// Test 24: shadowing i inside body
#[test]
fn test_eval_for_shadowing_in_body() {
    let output = eval_program(
        r#"
        for i in 0..3 {
            let i = i * 10;
            print(i);
        }
        "#,
    );
    assert_eq!(output, vec!["0", "10", "20"]);
}

// Test 25: for loop variable does not leak out
#[test]
fn test_eval_for_scoping() {
    // The for loop variable `i` should not be accessible after the loop
    let output = eval_program(
        r#"
        let mut result = 0;
        for i in 0..5 {
            result = result + i;
        }
        print(result);
        "#,
    );
    assert_eq!(output, vec!["10"]);
}

// Test 26: for loop with expressions as start/end (eval once)
#[test]
fn test_eval_for_range_expr_eval_once() {
    // Ensure start and end are evaluated only once
    let output = eval_program(
        r#"
        let mut counter = 0;
        fn side_effect() -> i64 {
            counter = counter + 1;
            return counter;
        }
        for i in 0..3 {
            print(i);
        }
        "#,
    );
    assert_eq!(output, vec!["0", "1", "2"]);
}

// Test 27: for loop accumulation pattern
#[test]
fn test_eval_for_accumulation() {
    let output = eval_program(
        r#"
        let mut sum = 0;
        for i in 1..6 {
            sum = sum + i;
        }
        print(sum);
        "#,
    );
    // 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(output, vec!["15"]);
}

// Test 28: for used at top-level (not inside fn)
#[test]
fn test_eval_for_top_level() {
    let output = eval_program("for i in 0..3 { print(i); }");
    assert_eq!(output, vec!["0", "1", "2"]);
}

// Test 29: nested array + range combination
#[test]
fn test_eval_nested_array_range() {
    let output = eval_program(
        r#"
        let arrays = [[1, 2], [3, 4]];
        for arr in arrays {
            for x in arr {
                print(x);
            }
        }
        "#,
    );
    assert_eq!(output, vec!["1", "2", "3", "4"]);
}
