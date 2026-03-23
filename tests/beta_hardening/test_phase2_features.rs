//! Phase 2 feature tests — I/O builtins, map operations, array_slice parity.

/// Run CJC source through eval, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        let rendered = diag.render_all(src, "<test>");
        panic!("Parse errors:\n{rendered}");
    }
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp.exec(&program).unwrap_or_else(|e| panic!("Eval failed for source:\n{src}\nError: {e}"));
    interp.output
}

/// Run CJC source through MIR-exec, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        let rendered = diag.render_all(src, "<test>");
        panic!("Parse errors:\n{rendered}");
    }
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

// ──────────────────────────────────────────────────────────────────
// 2.1  args() returns an array in both executors
// ──────────────────────────────────────────────────────────────────

#[test]
fn args_returns_array_eval() {
    let src = r#"
let a = args();
print(len(a) >= 0);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn args_returns_array_mir() {
    let src = r#"
let a = args();
print(len(a) >= 0);
"#;
    let out = run_mir(src, 42);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn args_parity() {
    let src = r#"
let a = args();
print(len(a) >= 0);
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "args() parity failed");
}

// ──────────────────────────────────────────────────────────────────
// 2.2  getenv() returns a string in both executors
// ──────────────────────────────────────────────────────────────────

#[test]
fn getenv_returns_string_eval() {
    // PATH is always set on all platforms
    let src = r#"
let p = getenv("PATH");
print(len(p) > 0);
"#;
    let out = run_eval(src, 42);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn getenv_returns_string_mir() {
    let src = r#"
let p = getenv("PATH");
print(len(p) > 0);
"#;
    let out = run_mir(src, 42);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn getenv_missing_returns_empty() {
    let src = r#"
let x = getenv("CJC_NONEXISTENT_VAR_12345");
print(x == "");
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, vec!["true"]);
    assert_eq!(mir_out, vec!["true"]);
}

// ──────────────────────────────────────────────────────────────────
// 2.3  Functional map builtins — parity tests
// ──────────────────────────────────────────────────────────────────

#[test]
fn map_new_and_set_eval() {
    let src = r#"
let m = map_new();
let m2 = map_set(m, "a", 1);
let m3 = map_set(m2, "b", 2);
print(map_get(m3, "a"));
print(map_get(m3, "b"));
print(map_contains(m3, "a"));
print(map_contains(m3, "c"));
"#;
    let out = run_eval(src, 42);
    assert_eq!(out, vec!["1", "2", "true", "false"]);
}

#[test]
fn map_new_and_set_mir() {
    let src = r#"
let m = map_new();
let m2 = map_set(m, "a", 1);
let m3 = map_set(m2, "b", 2);
print(map_get(m3, "a"));
print(map_get(m3, "b"));
print(map_contains(m3, "a"));
print(map_contains(m3, "c"));
"#;
    let out = run_mir(src, 42);
    assert_eq!(out, vec!["1", "2", "true", "false"]);
}

#[test]
fn map_parity() {
    let src = r#"
let m = map_new();
let m = map_set(m, "x", 10);
let m = map_set(m, "y", 20);
print(map_get(m, "x"));
print(map_get(m, "y"));
print(map_contains(m, "x"));
print(map_contains(m, "z"));
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "map builtins parity failed");
}

#[test]
fn map_keys_and_values_eval() {
    let src = r#"
let m = map_new();
let m = map_set(m, "a", 1);
let m = map_set(m, "b", 2);
let ks = map_keys(m);
let vs = map_values(m);
print(len(ks));
print(len(vs));
"#;
    let out = run_eval(src, 42);
    assert_eq!(out, vec!["2", "2"]);
}

#[test]
fn map_keys_and_values_mir() {
    let src = r#"
let m = map_new();
let m = map_set(m, "a", 1);
let m = map_set(m, "b", 2);
let ks = map_keys(m);
let vs = map_values(m);
print(len(ks));
print(len(vs));
"#;
    let out = run_mir(src, 42);
    assert_eq!(out, vec!["2", "2"]);
}

#[test]
fn map_cow_semantics() {
    // map_set returns a NEW map, original is unchanged
    let src = r#"
let m = map_new();
let m2 = map_set(m, "a", 1);
print(map_contains(m, "a"));
print(map_contains(m2, "a"));
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, vec!["false", "true"]);
    assert_eq!(mir_out, vec!["false", "true"]);
}

#[test]
fn map_get_missing_returns_void() {
    let src = r#"
let m = map_new();
let v = map_get(m, "nope");
print(v);
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out);
}

// ──────────────────────────────────────────────────────────────────
// 2.4  array_slice parity
// ──────────────────────────────────────────────────────────────────

#[test]
fn array_slice_parity() {
    let src = r#"
let arr = [10, 20, 30, 40, 50];
let s = array_slice(arr, 1, 4);
print(len(s));
print(s[0]);
print(s[1]);
print(s[2]);
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, vec!["3", "20", "30", "40"]);
    assert_eq!(mir_out, vec!["3", "20", "30", "40"]);
}

#[test]
fn array_slice_empty() {
    let src = r#"
let arr = [1, 2, 3];
let s = array_slice(arr, 1, 1);
print(len(s));
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, vec!["0"]);
    assert_eq!(mir_out, vec!["0"]);
}
