// CJC Test Suite — Module System
// Tests: mod keyword parsing, import parsing, multi-file resolution.

// ---------------------------------------------------------------------------
// Lexer: mod keyword
// ---------------------------------------------------------------------------

#[test]
fn lex_mod_keyword() {
    let (tokens, diag) = cjc_lexer::Lexer::new("mod math").tokenize();
    assert_eq!(diag.count(), 0);
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::Mod);
    assert_eq!(tokens[0].text, "mod");
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::Ident);
    assert_eq!(tokens[1].text, "math");
}

// ---------------------------------------------------------------------------
// Parser: mod declaration
// ---------------------------------------------------------------------------

#[test]
fn parse_mod_decl() {
    let src = "mod math";
    let (prog, diag) = cjc_parser::parse_source(src);
    assert_eq!(diag.count(), 0);
    assert_eq!(prog.declarations.len(), 1);
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Import(imp) => {
            assert_eq!(imp.path.len(), 1);
            assert_eq!(imp.path[0].name, "math");
            assert!(imp.alias.is_none());
        }
        _ => panic!("expected Import (from mod desugaring)"),
    }
}

#[test]
fn parse_import_dotted() {
    let src = "import stats.linear";
    let (prog, diag) = cjc_parser::parse_source(src);
    assert_eq!(diag.count(), 0);
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Import(imp) => {
            assert_eq!(imp.path.len(), 2);
            assert_eq!(imp.path[0].name, "stats");
            assert_eq!(imp.path[1].name, "linear");
        }
        _ => panic!("expected ImportDecl"),
    }
}

#[test]
fn parse_import_with_alias() {
    let src = "import math.linalg as la";
    let (prog, diag) = cjc_parser::parse_source(src);
    assert_eq!(diag.count(), 0);
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Import(imp) => {
            assert_eq!(imp.path.len(), 2);
            assert_eq!(imp.alias.as_ref().unwrap().name, "la");
        }
        _ => panic!("expected ImportDecl"),
    }
}

// ---------------------------------------------------------------------------
// Module graph: construction & cycle detection
// ---------------------------------------------------------------------------

#[test]
fn module_graph_single_file() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let entry = dir.path().join("main.cjcl");
    std::fs::write(&entry, "let x = 42;").unwrap();

    let graph = cjc_module::build_module_graph(&entry).unwrap();
    assert_eq!(graph.module_count(), 1);
}

#[test]
fn module_graph_with_import() {
    let dir = tempfile::tempdir().unwrap();

    // math.cjcl
    std::fs::write(dir.path().join("math.cjcl"), r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
"#).unwrap();

    // main.cjcl imports math
    std::fs::write(dir.path().join("main.cjcl"), r#"
import math
"#).unwrap();

    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    assert_eq!(graph.module_count(), 2);
}

#[test]
fn module_graph_cyclic_dependency_detected() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(dir.path().join("a.cjcl"), "import b").unwrap();
    std::fs::write(dir.path().join("b.cjcl"), "import a").unwrap();

    let entry = dir.path().join("a.cjcl");
    let result = cjc_module::build_module_graph(&entry);
    assert!(result.is_err(), "should detect cycle");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("cycl") || err_msg.contains("Cycl"),
            "error should mention cycle: {}", err_msg);
}

#[test]
fn module_graph_deterministic_order() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(dir.path().join("alpha.cjcl"), "fn alpha_fn() -> i64 { 1 }").unwrap();
    std::fs::write(dir.path().join("beta.cjcl"), "fn beta_fn() -> i64 { 2 }").unwrap();
    std::fs::write(dir.path().join("main.cjcl"), "import alpha\nimport beta").unwrap();

    let entry = dir.path().join("main.cjcl");

    // Build graph twice — must produce identical topological order
    let g1 = cjc_module::build_module_graph(&entry).unwrap();
    let g2 = cjc_module::build_module_graph(&entry).unwrap();

    let order1 = g1.topological_order().unwrap();
    let order2 = g2.topological_order().unwrap();

    assert_eq!(order1, order2, "topological order must be deterministic");
}

// ---------------------------------------------------------------------------
// Module execution: run_program_with_modules
// ---------------------------------------------------------------------------

#[test]
fn module_exec_single_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.cjcl"), r#"
fn square(x: i64) -> i64 {
    x * x
}
print(square(7));
"#).unwrap();

    let entry = dir.path().join("main.cjcl");
    let (val, exec) = cjc_mir_exec::run_program_with_modules_executor(&entry, 42).unwrap();
    assert!(exec.output.iter().any(|s| s.contains("49")),
            "expected 49 in output, got {:?}", exec.output);
}

#[test]
fn module_exec_two_files() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(dir.path().join("mathlib.cjcl"), r#"
fn double(x: i64) -> i64 {
    x * 2
}
"#).unwrap();

    std::fs::write(dir.path().join("main.cjcl"), r#"
import mathlib
print(double(21));
"#).unwrap();

    let entry = dir.path().join("main.cjcl");
    let (_val, exec) = cjc_mir_exec::run_program_with_modules_executor(&entry, 42).unwrap();
    assert!(exec.output.iter().any(|s| s.contains("42")),
            "expected 42 in output, got {:?}", exec.output);
}
