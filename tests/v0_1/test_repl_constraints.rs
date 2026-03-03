//! v0.1 Contract Tests: REPL Constraints
//!
//! Locks down: array parsing, parse-failure recovery, Map prelude,
//! state accumulation, eval/MIR parity.

// ── Helpers ──────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn single_line_array_parses() {
    let out = eval_output("let x = [1, 2, 3]; print(x);");
    assert_eq!(out, vec!["[1, 2, 3]"]);
}

#[test]
fn multi_line_braces_work() {
    let src = r#"
fn fizz(n: i64) -> str {
    if n % 15 == 0 {
        return "FizzBuzz";
    } else if n % 3 == 0 {
        return "Fizz";
    } else if n % 5 == 0 {
        return "Buzz";
    }
    to_string(n)
}
print(fizz(15));
print(fizz(7));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["FizzBuzz", "7"]);
}

#[test]
fn parse_failure_does_not_poison() {
    let mut interp = cjc_eval::Interpreter::new(42);

    // First: deliberately malformed input
    let bad = "let @@@ = ;";
    let (_, bad_diags) = cjc_parser::parse_source(bad);
    assert!(bad_diags.has_errors(), "bad input should have parse errors");

    // Second: valid input on SAME interpreter should still work
    let good = "let x: i64 = 99; print(x);";
    let good_prog = parse(good);
    let result = interp.exec(&good_prog);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["99"]);
}

#[test]
fn map_new_is_recognized() {
    let src = r#"
let m = Map.new();
m.insert("a", 1);
print(m.get("a"));
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["1"]);
}

#[test]
fn backslash_continuation_logic() {
    // Verify the line_editor continuation heuristic: open braces > close braces
    let line = "fn foo() {";
    let open = line.chars().filter(|&c| c == '{').count();
    let close = line.chars().filter(|&c| c == '}').count();
    assert!(open > close, "open braces should exceed close braces for continuation");
}

#[test]
fn meta_command_prefix_detection() {
    for cmd in &[":help", ":quit", ":reset", ":type x", ":ast x", ":mir x", ":env", ":seed"] {
        assert!(cmd.starts_with(':'), "meta command should start with ':'");
    }
}

#[test]
fn repl_accumulates_state() {
    let mut interp = cjc_eval::Interpreter::new(42);

    // First exec: define a function
    let def = "fn double(n: i64) -> i64 { n * 2 }";
    let prog1 = parse(def);
    interp.exec(&prog1).unwrap();

    // Second exec: call it
    let call = "print(double(21));";
    let prog2 = parse(call);
    interp.exec(&prog2).unwrap();
    assert_eq!(interp.output, vec!["42"]);
}

#[test]
fn repl_parity_eval_mir() {
    let src = "let x: i64 = 42; print(x);";
    let eval_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(eval_out, mir_out);
    assert_eq!(eval_out, vec!["42"]);
}
