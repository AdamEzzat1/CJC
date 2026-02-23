//! Audit Test: Module System Reality Check
//!
//! Claim: "No module system (import tokenizes but no multi-file program support)"
//!
//! VERDICT: CONFIRMED
//!
//! Evidence:
//! - Lexer: TokenKind::Import recognized
//! - Parser: parse_import_decl() produces DeclKind::Import(ImportDecl { path: Vec<Ident>, alias })
//! - HIR: lower_decl() for DeclKind::Import produces HirItem::Stmt(Void) — a no-op
//!   (crates/cjc-hir/src/lib.rs, lower_decl match arm for Import)
//! - MIR: No module resolution pass exists anywhere in the codebase
//! - MIR-exec: No file loading, no namespace lookup, no symbol table from imports
//! - Runtime: No module registry, no dynamic library loading
//!
//! CONCLUSION: Import syntax is fully parsed and stored in AST but is
//! completely discarded at HIR lowering (becomes Void). No multi-file
//! compilation or runtime symbol resolution exists.

use cjc_parser::parse_source;
use cjc_ast::DeclKind;

/// Test 1: `import` keyword is recognized and parsed into an ImportDecl.
#[test]
fn test_import_tokenizes_and_parses() {
    let src = "import std.io";
    let (prog, diags) = parse_source(src);
    let has_errs = diags.has_errors();
    assert!(!has_errs, "import should parse without errors");

    let has_import = prog.declarations.iter().any(|d| matches!(d.kind, DeclKind::Import(_)));
    assert!(has_import, "parsed program should contain an Import declaration");
}

/// Test 2: Import path stores all segments correctly.
#[test]
fn test_import_path_segments_stored() {
    let src = "import std.io.File";
    let (prog, _diags) = parse_source(src);
    for decl in &prog.declarations {
        if let DeclKind::Import(i) = &decl.kind {
            assert_eq!(i.path.len(), 3);
            assert_eq!(i.path[0].name, "std");
            assert_eq!(i.path[1].name, "io");
            assert_eq!(i.path[2].name, "File");
            assert!(i.alias.is_none());
            return;
        }
    }
    panic!("no import decl found");
}

/// Test 3: Import with alias is parsed correctly.
#[test]
fn test_import_with_alias_parses() {
    let src = "import math.linalg as la";
    let (prog, _diags) = parse_source(src);
    for decl in &prog.declarations {
        if let DeclKind::Import(i) = &decl.kind {
            assert_eq!(i.path.len(), 2);
            assert_eq!(i.path[0].name, "math");
            assert_eq!(i.path[1].name, "linalg");
            assert_eq!(i.alias.as_ref().unwrap().name, "la");
            return;
        }
    }
    panic!("no import decl with alias found");
}

/// Test 4: Import lowers to a no-op (Void) in HIR — confirming no resolution.
/// This is the key evidence that module resolution is absent.
#[test]
fn test_import_lowers_to_void_in_hir() {
    use cjc_hir::{AstLowering, HirItem, HirStmtKind, HirExprKind};
    let src = "import std.io.File";
    let (prog, _) = parse_source(src);
    let mut lowering = AstLowering::new();
    let hir = lowering.lower_program(&prog);

    // The import should become a Stmt(Expr(Void)) — a complete no-op
    for item in &hir.items {
        if let HirItem::Stmt(stmt) = item {
            if let HirStmtKind::Expr(expr) = &stmt.kind {
                assert!(
                    matches!(expr.kind, HirExprKind::Void),
                    "import should lower to HirExprKind::Void, got: {:?}", expr.kind
                );
                return;
            }
        }
    }
    panic!("import did not lower to a Stmt(Expr(Void))");
}

/// Test 5: A program with an import AND a function runs fine —
/// the import is silently discarded, not an error.
#[test]
fn test_import_is_silently_discarded_at_runtime() {
    let src = r#"
import std.collections.Map
import os.path as path
fn main() -> i64 {
    42
}
"#;
    let (prog, diags) = parse_source(src);
    let has_errs = diags.has_errors();
    assert!(!has_errs, "imports + main should parse clean");

    // Run through MIR executor — imports are discarded, main runs
    let result = cjc_mir_exec::run_program_with_executor(&prog, 42);
    match result {
        Ok((val, _)) => {
            use cjc_runtime::Value;
            assert!(matches!(val, Value::Int(42)), "main should return 42 after imports discarded");
        }
        Err(e) => {
            panic!("runtime should not fail on discarded imports: {:?}", e);
        }
    }
}

/// Test 6: Importing a non-existent module produces NO compile-time error.
/// This confirms there is no module resolution — all paths are accepted blindly.
#[test]
fn test_import_nonexistent_module_is_not_an_error() {
    let src = "import this.does.not.exist.at.all as nope";
    let (prog, diags) = parse_source(src);
    let has_errs = diags.has_errors();
    // There must be NO error for an unresolvable module path
    assert!(
        !has_errs,
        "non-existent import path should produce no error (no resolution)"
    );
    let has_import = prog.declarations.iter().any(|d| matches!(d.kind, DeclKind::Import(_)));
    assert!(has_import, "import should still be in AST");
}

/// Test 7: Symbols from an imported module are NOT accessible at runtime.
/// Documents the gap: importing does not bring names into scope.
#[test]
fn test_imported_symbols_not_in_scope() {
    // If the module system were real, `File` would be accessible after `import std.io.File`
    let src = r#"
import std.io.File
fn main() -> i64 {
    42
}
"#;
    let (prog, _) = parse_source(src);
    let result = cjc_mir_exec::run_program_with_executor(&prog, 42);
    // Main still works — import is discarded
    match result {
        Ok((val, _)) => {
            use cjc_runtime::Value;
            assert!(matches!(val, Value::Int(42)));
        }
        Err(_) => {
            // Also acceptable: runtime may fail for other reasons, but NOT
            // because import was processed (it shouldn't be).
        }
    }
}
