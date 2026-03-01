//! H-7: Break/Continue — Full Pipeline Tests
//!
//! Verifies break/continue from lexer through executor:
//! - Lexer: `break` and `continue` keywords
//! - Parser: parse_block Break/Continue, loop_depth validation
//! - HIR: lowering to HirStmtKind::Break/Continue
//! - MIR: lowering to MirStmt::Break/Continue
//! - Eval: runtime behavior in while/for loops
//! - MIR Executor: runtime behavior in while/for loops

// ── Lexer tests ──────────────────────────────────────────────────────────

#[test]
fn lex_break_keyword() {
    let lexer = cjc_lexer::Lexer::new("break");
    let (tokens, diags) = lexer.tokenize();
    assert!(!diags.has_errors());
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::Break);
    assert_eq!(tokens[0].text, "break");
}

#[test]
fn lex_continue_keyword() {
    let lexer = cjc_lexer::Lexer::new("continue");
    let (tokens, diags) = lexer.tokenize();
    assert!(!diags.has_errors());
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::Continue);
    assert_eq!(tokens[0].text, "continue");
}

#[test]
fn lex_break_is_keyword_not_ident() {
    let lexer = cjc_lexer::Lexer::new("break");
    let (tokens, _) = lexer.tokenize();
    assert!(tokens[0].kind.is_keyword());
    assert_ne!(tokens[0].kind, cjc_lexer::TokenKind::Ident);
}

// ── Parser tests ─────────────────────────────────────────────────────────

#[test]
fn parse_break_in_while() {
    let src = "fn main() { while true { break; } }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "should parse break inside while");
    assert_eq!(program.declarations.len(), 1);
}

#[test]
fn parse_continue_in_while() {
    let src = "fn main() { while true { continue; } }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "should parse continue inside while");
    assert_eq!(program.declarations.len(), 1);
}

#[test]
fn parse_break_in_for() {
    let src = "fn main() { for i in 0..10 { break; } }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "should parse break inside for");
    assert_eq!(program.declarations.len(), 1);
}

#[test]
fn parse_break_outside_loop_emits_error() {
    let src = "fn main() { break; }";
    let (_program, diags) = cjc_parser::parse_source(src);
    assert!(diags.has_errors(), "break outside loop should be an error");
}

#[test]
fn parse_continue_outside_loop_emits_error() {
    let src = "fn main() { continue; }";
    let (_program, diags) = cjc_parser::parse_source(src);
    assert!(diags.has_errors(), "continue outside loop should be an error");
}

#[test]
fn parse_break_in_nested_loops() {
    let src = "fn main() { while true { for i in 0..10 { break; } } }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    assert_eq!(program.declarations.len(), 1);
}

// ── Eval (AST interpreter) tests ─────────────────────────────────────────

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

#[test]
fn eval_break_in_while() {
    let output = eval_program(r#"
        let mut i = 0;
        while true {
            if i == 3 { break; }
            print(i);
            i = i + 1;
        }
    "#);
    assert_eq!(output, vec!["0", "1", "2"]);
}

#[test]
fn eval_continue_in_while() {
    let output = eval_program(r#"
        let mut i = 0;
        while i < 5 {
            i = i + 1;
            if i == 3 { continue; }
            print(i);
        }
    "#);
    // Skips printing 3
    assert_eq!(output, vec!["1", "2", "4", "5"]);
}

#[test]
fn eval_break_in_for_range() {
    let output = eval_program(r#"
        for i in 0..10 {
            if i == 4 { break; }
            print(i);
        }
    "#);
    assert_eq!(output, vec!["0", "1", "2", "3"]);
}

#[test]
fn eval_continue_in_for_range() {
    let output = eval_program(r#"
        for i in 0..6 {
            if i == 2 { continue; }
            if i == 4 { continue; }
            print(i);
        }
    "#);
    assert_eq!(output, vec!["0", "1", "3", "5"]);
}

#[test]
fn eval_break_only_exits_inner_loop() {
    let output = eval_program(r#"
        for i in 0..3 {
            for j in 0..10 {
                if j == 2 { break; }
                print(j);
            }
            print(i);
        }
    "#);
    // Inner loop prints 0,1 then breaks; outer prints i
    assert_eq!(output, vec!["0", "1", "0", "0", "1", "1", "0", "1", "2"]);
}

#[test]
fn eval_continue_only_affects_inner_loop() {
    let output = eval_program(r#"
        for i in 0..2 {
            for j in 0..4 {
                if j == 1 { continue; }
                if j == 3 { continue; }
                print(j);
            }
        }
    "#);
    // Each outer iteration: inner prints 0, 2 (skips 1, 3)
    assert_eq!(output, vec!["0", "2", "0", "2"]);
}

#[test]
fn eval_break_in_for_expr() {
    let output = eval_program(r#"
        let arr = [10, 20, 30, 40, 50];
        for x in arr {
            if x == 30 { break; }
            print(x);
        }
    "#);
    assert_eq!(output, vec!["10", "20"]);
}

// ── MIR executor tests ──────────────────────────────────────────────────

fn mir_exec(source: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(source);
    if diags.has_errors() {
        let rendered = diags.render_all(source, "<test>");
        panic!("parse errors:\n{}", rendered);
    }
    match cjc_mir_exec::run_program_with_executor(&program, 42) {
        Ok((_value, executor)) => executor.output.clone(),
        Err(e) => panic!("MIR exec error: {e}"),
    }
}

#[test]
fn mir_break_in_while() {
    let output = mir_exec(r#"
        fn main() {
            let mut i = 0;
            while true {
                if i == 3 { break; }
                print(i);
                i = i + 1;
            }
        }
    "#);
    assert_eq!(output, vec!["0", "1", "2"]);
}

#[test]
fn mir_continue_in_while() {
    let output = mir_exec(r#"
        fn main() {
            let mut i = 0;
            while i < 5 {
                i = i + 1;
                if i == 3 { continue; }
                print(i);
            }
        }
    "#);
    assert_eq!(output, vec!["1", "2", "4", "5"]);
}

#[test]
fn mir_break_in_for_range() {
    let output = mir_exec(r#"
        fn main() {
            for i in 0..10 {
                if i == 4 { break; }
                print(i);
            }
        }
    "#);
    assert_eq!(output, vec!["0", "1", "2", "3"]);
}

#[test]
fn mir_continue_in_for_range() {
    let output = mir_exec(r#"
        fn main() {
            for i in 0..6 {
                if i == 2 { continue; }
                if i == 4 { continue; }
                print(i);
            }
        }
    "#);
    assert_eq!(output, vec!["0", "1", "3", "5"]);
}

#[test]
fn mir_break_nested_loops() {
    let output = mir_exec(r#"
        fn main() {
            for i in 0..3 {
                let mut j = 0;
                while true {
                    if j == 2 { break; }
                    print(j);
                    j = j + 1;
                }
                print(i);
            }
        }
    "#);
    assert_eq!(output, vec!["0", "1", "0", "0", "1", "1", "0", "1", "2"]);
}

#[test]
fn mir_continue_nested_loops() {
    let output = mir_exec(r#"
        fn main() {
            for i in 0..2 {
                for j in 0..4 {
                    if j == 1 { continue; }
                    if j == 3 { continue; }
                    print(j);
                }
            }
        }
    "#);
    assert_eq!(output, vec!["0", "2", "0", "2"]);
}
