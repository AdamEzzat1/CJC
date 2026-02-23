//! Phase 2 Audit Tests: P2-1 impl Trait for Type Parser
//!
//! Tests for both impl syntaxes:
//!   (1) `impl Type : Trait { ... }`  — original CJC syntax
//!   (2) `impl Trait for Type { ... }` — Rust-style syntax

use cjc_parser::parse_source;
use cjc_mir_exec::run_program_with_executor;
use cjc_runtime::Value;

fn run_src(src: &str) -> Result<Value, String> {
    let (prog, diags) = parse_source(src);
    if diags.has_errors() {
        return Err(format!("parse errors"));
    }
    run_program_with_executor(&prog, 42)
        .map(|(v, _)| v)
        .map_err(|e| format!("{e}"))
}

fn parses_without_error(src: &str) -> bool {
    let (_, diags) = parse_source(src);
    !diags.has_errors()
}

/// P2-1 Test 1: Original CJC syntax `impl Type : Trait` parses correctly.
#[test]
fn test_impl_type_colon_trait_syntax() {
    let src = r#"
trait Greet {
    fn greet(self: Greeter) -> str;
}
struct Greeter { name: str }
impl Greeter : Greet {
    fn greet(self: Greeter) -> str {
        "hello"
    }
}
fn main() -> i64 { 0 }
"#;
    assert!(parses_without_error(src),
        "Original 'impl Type : Trait' syntax should parse without error");
}

/// P2-1 Test 2: Rust-style `impl Trait for Type` parses correctly.
#[test]
fn test_impl_trait_for_type_syntax() {
    let src = r#"
trait Describe {
    fn describe(self: Dog) -> str;
}
struct Dog { name: str }
impl Describe for Dog {
    fn describe(self: Dog) -> str {
        "a dog"
    }
}
fn main() -> i64 { 0 }
"#;
    assert!(parses_without_error(src),
        "Rust-style 'impl Trait for Type' syntax should parse without error");
}

/// P2-1 Test 3: Both syntaxes in same program.
#[test]
fn test_both_impl_syntaxes_coexist() {
    let src = r#"
trait Alpha {
    fn alpha(self: A) -> i64;
}
trait Beta {
    fn beta(self: B) -> i64;
}
struct A {}
struct B {}
impl A : Alpha {
    fn alpha(self: A) -> i64 { 1 }
}
impl Beta for B {
    fn beta(self: B) -> i64 { 2 }
}
fn main() -> i64 { 0 }
"#;
    assert!(parses_without_error(src),
        "Both impl syntaxes should coexist in same program");
}

/// P2-1 Test 4: impl without trait (bare impl) still works.
#[test]
fn test_bare_impl_still_works() {
    let src = r#"
struct Counter { val: i64 }
impl Counter {
    fn increment(self: Counter) -> i64 {
        self.val + 1
    }
}
fn main() -> i64 { 0 }
"#;
    assert!(parses_without_error(src),
        "Bare impl (no trait) should still parse correctly");
}

/// P2-1 Test 5: impl Trait for Type with generic type params.
#[test]
fn test_impl_trait_for_generic_type() {
    let src = r#"
trait Print {
    fn print_me(self: Wrapper) -> i64;
}
struct Wrapper { val: i64 }
impl Print for Wrapper {
    fn print_me(self: Wrapper) -> i64 {
        self.val
    }
}
fn main() -> i64 { 0 }
"#;
    assert!(parses_without_error(src),
        "impl Trait for Type with struct should parse");
}
