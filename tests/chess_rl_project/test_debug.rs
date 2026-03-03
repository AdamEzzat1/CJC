//! Temporary debug test for CJC language features

pub fn run_mir_debug(src: &str) -> Vec<String> {
    let (tokens, _diag) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _diag) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

pub fn run_eval_debug(src: &str) -> Vec<String> {
    let (tokens, _diag) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _diag) = cjc_parser::Parser::new(tokens).parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap_or_else(|e| panic!("Eval failed: {e}"));
    interp.output.clone()
}

pub fn run_source_debug(src: &str) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .unwrap_or_else(|e| panic!("MIR-exec (parse_source) failed: {e}"));
    executor.output
}

#[test]
fn debug_array_push_returns_new() {
    // array_push RETURNS a new array, doesn't mutate
    let out = run_mir_debug(r#"
let arr = [];
arr = array_push(arr, 1);
arr = array_push(arr, 2);
arr = array_push(arr, 3);
print(len(arr));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn debug_array_push_in_loop() {
    let out = run_mir_debug(r#"
let arr = [];
let i = 0;
while i < 3 {
    arr = array_push(arr, i);
    i = i + 1;
};
print(len(arr));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn debug_function_typed_params() {
    // Use type annotations on function parameters (required by parser)
    let out = run_source_debug(r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
print(add(3, 4));
"#);
    assert_eq!(out, vec!["7"]);
}

#[test]
#[ignore] // Known limitation: }; after while inside fn body causes parse error
fn debug_function_no_params_no_return() {
    // No return type annotation
    let out = run_source_debug(r#"
fn build() {
    let arr = [];
    let i = 0;
    while i < 3 {
        arr = array_push(arr, i * 10);
        i = i + 1;
    };
    arr
}
let x = build();
print(len(x));
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn debug_function_no_params_simple() {
    let out = run_source_debug(r#"
fn five() -> i64 {
    5
}
print(five());
"#);
    assert_eq!(out, vec!["5"]);
}

#[test]
fn debug_function_no_params_decl_count() {
    let src = r#"
fn five() -> i64 { 5 }
print(five());
"#;
    let (prog, diag) = cjc_parser::parse_source(src);
    eprintln!("Decl count for no-params fn: {}", prog.declarations.len());
    eprintln!("Has errors: {}", diag.has_errors());
    for (i, decl) in prog.declarations.iter().enumerate() {
        eprintln!("  decl[{}]: {:?}", i, std::mem::discriminant(&decl.kind));
    }
}

#[test]
fn debug_fn_body_let() {
    let out = run_source_debug(r#"
fn f(x: i64) -> i64 {
    let y = x + 1;
    y
}
print(f(5));
"#);
    assert_eq!(out, vec!["6"]);
}

#[test]
fn debug_fn_body_two_lets() {
    let out = run_source_debug(r#"
fn f(x: i64) -> i64 {
    let a = x + 1;
    let b = a + 1;
    b
}
print(f(5));
"#);
    assert_eq!(out, vec!["7"]);
}

#[test]
fn debug_fn_body_if() {
    let out = run_source_debug(r#"
fn f(x: i64) -> i64 {
    if x > 0 { x } else { 0 - x }
}
print(f(5));
print(f(0 - 3));
"#);
    assert_eq!(out, vec!["5", "3"]);
}

#[test]
fn debug_fn_body_while_no_semi() {
    // Try without semicolon after while block
    let out = run_source_debug(r#"
fn f(n: i64) -> i64 {
    let i = 0;
    let sum = 0;
    while i < n {
        sum = sum + i;
        i = i + 1;
    }
    sum
}
print(f(4));
"#);
    assert_eq!(out, vec!["6"]);
}

#[test]
#[ignore] // Known limitation: }; after while inside fn body causes parse error
fn debug_fn_body_while_with_semi() {
    // With semicolon after while block (like top-level)
    let out = run_source_debug(r#"
fn f(n: i64) -> i64 {
    let i = 0;
    let sum = 0;
    while i < n {
        sum = sum + i;
        i = i + 1;
    };
    sum
}
print(f(4));
"#);
    assert_eq!(out, vec!["6"]);
}

#[test]
fn debug_fn_body_array() {
    let out = run_source_debug(r#"
fn f(x: i64) -> i64 {
    let arr = [];
    arr = array_push(arr, x);
    len(arr)
}
print(f(42));
"#);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn debug_function_typed_eval_path() {
    let (program, _diag) = cjc_parser::parse_source(r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
print(add(3, 4));
"#);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap_or_else(|e| panic!("Eval failed: {e}"));
    assert_eq!(interp.output, vec!["7"]);
}

#[test]
fn debug_function_typed_decl_count() {
    let src = r#"
fn double(x: i64) -> i64 { x * 2 }
print(double(5));
"#;
    let (prog, _) = cjc_parser::parse_source(src);
    let decl_count = prog.declarations.len();
    eprintln!("Typed decl count: {decl_count}");
    assert_eq!(decl_count, 2, "Should have fn decl + stmt decl");
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&prog, 42).unwrap();
    assert_eq!(exec.output, vec!["10"]);
}

#[test]
fn debug_function_typed_mir_path() {
    // Test via run_mir_debug (Parser::new path) with typed params
    let out = run_mir_debug(r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
print(add(3, 4));
"#);
    assert_eq!(out, vec!["7"]);
}

#[test]
fn debug_function_any_type() {
    // Test with 'Any' type for dynamic dispatch
    let out = run_source_debug(r#"
fn identity(x: Any) -> Any {
    x
}
print(identity(42));
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn debug_chess_env_parse() {
    use super::cjc_source::CHESS_ENV;
    let src = format!("{CHESS_ENV}\nprint(len(init_board()));");
    let (prog, diag) = cjc_parser::parse_source(&src);
    eprintln!("CHESS_ENV parse errors: {}", diag.has_errors());
    // Print diagnostics
    for d in diag.diagnostics.iter() {
        eprintln!("  DIAG: {:?}", d);
    }
    eprintln!("CHESS_ENV decl count: {}", prog.declarations.len());
    for (i, d) in prog.declarations.iter().enumerate() {
        match &d.kind {
            cjc_ast::DeclKind::Fn(f) => eprintln!("  decl[{i}] Fn: {}", f.name.name),
            cjc_ast::DeclKind::Stmt(_) => eprintln!("  decl[{i}] Stmt"),
            cjc_ast::DeclKind::Let(_) => eprintln!("  decl[{i}] Let"),
            _ => eprintln!("  decl[{i}] Other"),
        }
    }
}

#[test]
fn debug_init_board_only() {
    let src = r#"
fn init_board() -> Any {
    let b = [
        4, 2, 3, 5, 6, 3, 2, 4,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
       -1,-1,-1,-1,-1,-1,-1,-1,
       -4,-2,-3,-5,-6,-3,-2,-4
    ];
    b
}
print(len(init_board()));
"#;
    let out = run_source_debug(src);
    assert_eq!(out, vec!["64"]);
}

#[test]
#[ignore] // Known limitation: }; after while inside fn body causes parse error
fn debug_fn_while_eval_path() {
    // Test while-loop function via eval (not MIR-exec)
    let (program, _diag) = cjc_parser::parse_source(r#"
fn f(n: i64) -> i64 {
    let i = 0;
    let sum = 0;
    while i < n {
        sum = sum + i;
        i = i + 1;
    };
    sum
}
print(f(4));
"#);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap_or_else(|e| panic!("Eval failed: {e}"));
    assert_eq!(interp.output, vec!["6"]);
}

#[test]
fn debug_fn_while_mir_diagnostic() {
    // Diagnose: check HIR and MIR output for while-loop function
    let src = r#"
fn f(n: i64) -> i64 {
    let i = 0;
    let sum = 0;
    while i < n {
        sum = sum + i;
        i = i + 1;
    };
    sum
}
print(f(4));
"#;
    let (program, diag) = cjc_parser::parse_source(src);
    eprintln!("Parse errors: {}", diag.has_errors());
    eprintln!("Decl count: {}", program.declarations.len());
    for (i, d) in program.declarations.iter().enumerate() {
        eprintln!("  decl[{}] kind: {:?}", i, std::mem::discriminant(&d.kind));
        if let cjc_ast::DeclKind::Fn(f) = &d.kind {
            eprintln!("    fn name: {}, params: {}, body stmts: {}", f.name.name, f.params.len(), f.body.stmts.len());
        }
    }

    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(&program);
    eprintln!("HIR items: {}", hir.items.len());
    for (i, item) in hir.items.iter().enumerate() {
        match item {
            cjc_hir::HirItem::Fn(f) => eprintln!("  hir[{}] Fn: {}, params: {}", i, f.name, f.params.len()),
            cjc_hir::HirItem::Stmt(_) => eprintln!("  hir[{}] Stmt", i),
            cjc_hir::HirItem::Let(l) => eprintln!("  hir[{}] Let: {}", i, l.name),
            _ => eprintln!("  hir[{}] other", i),
        }
    }

    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mir = hir_to_mir.lower_program(&hir);
    eprintln!("MIR functions: {}", mir.functions.len());
    for f in &mir.functions {
        eprintln!("  mir fn: {}, params: {}, body stmts: {}", f.name, f.params.len(), f.body.stmts.len());
    }
}
