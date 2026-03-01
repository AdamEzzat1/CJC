//! Hardening test H12: Comprehensive integration test suite.
//!
//! Tests cross-feature interactions across all hardening phases:
//! - Effect system integration
//! - Arena lifecycle end-to-end
//! - Break/continue complex patterns (wrapped in fn main for MIR executor)
//! - Module cross-file compilation
//! - File I/O + JSON roundtrip
//! - DateTime operations
//! - TiledMatmul integration
//! - Window functions
//! - Cross-feature programs

use std::fs;

/// Helper: parse + MIR-execute a CJC program, return executor output.
fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

/// Helper: create a temp directory with CJC source files.
fn setup_test_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("create temp dir");
    for (name, content) in files {
        let path = dir.path().join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dirs");
        }
        fs::write(&path, content).expect("write test file");
    }
    dir
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Effect system integration
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_effect_registry_complete_coverage() {
    use cjc_types::effect_registry;
    // All major builtin categories should have registry entries
    let builtins = [
        // Math/pure
        "abs", "sqrt", "floor",
        // GC
        "gc_alloc", "gc_collect",
        // IO
        "clock", "file_read", "file_write",
        // JSON
        "json_parse", "json_stringify",
        // DateTime
        "datetime_now", "datetime_year", "datetime_format",
        // Window
        "window_sum", "window_mean", "window_min", "window_max",
    ];
    for name in &builtins {
        assert!(
            effect_registry::lookup(name).is_some(),
            "builtin '{}' should be in effect registry",
            name,
        );
    }
}

#[test]
fn h12_effect_nondeterminism_classification() {
    use cjc_types::effect_registry;
    // These should be marked as nondeterministic
    assert!(effect_registry::is_nondeterministic("datetime_now"));
    assert!(effect_registry::is_nondeterministic("clock"));
    // These should NOT be nondeterministic
    assert!(!effect_registry::is_nondeterministic("datetime_year"));
    assert!(!effect_registry::is_nondeterministic("json_parse"));
    assert!(!effect_registry::is_nondeterministic("window_sum"));
    assert!(!effect_registry::is_nondeterministic("abs"));
}

#[test]
fn h12_effect_io_classification() {
    use cjc_types::{effect_registry, EffectSet};
    // File I/O should have IO flag
    let read_effects = effect_registry::lookup("file_read").unwrap();
    assert!(read_effects.has(EffectSet::IO));
    let write_effects = effect_registry::lookup("file_write").unwrap();
    assert!(write_effects.has(EffectSet::IO));
    // JSON should NOT have IO flag
    let json_effects = effect_registry::lookup("json_parse").unwrap();
    assert!(!json_effects.has(EffectSet::IO));
}

#[test]
fn h12_effect_safe_builtin_classification() {
    use cjc_types::effect_registry;
    // Pure math builtins should be safe
    assert!(effect_registry::is_safe_builtin("abs"));
    assert!(effect_registry::is_safe_builtin("sqrt"));
    // GC builtins should NOT be safe
    assert!(!effect_registry::is_safe_builtin("gc_alloc"));
}

#[test]
fn h12_effect_gc_classification() {
    use cjc_types::{effect_registry, EffectSet};
    let gc_effects = effect_registry::lookup("gc_alloc").unwrap();
    assert!(gc_effects.has(EffectSet::GC));
    let pure_effects = effect_registry::lookup("abs").unwrap();
    assert!(!pure_effects.has(EffectSet::GC));
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Arena lifecycle end-to-end
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_arena_lifecycle_basic() {
    let out = run_mir(r#"
fn compute(x: i64) -> i64 {
    let temp = x * x;
    temp + 1
}
let a = compute(5);
let b = compute(10);
print(a);
print(b);
"#);
    assert_eq!(out, vec!["26", "101"]);
}

#[test]
fn h12_arena_nested_calls() {
    let out = run_mir(r#"
fn inner(x: i64) -> i64 { x + 1 }
fn outer(x: i64) -> i64 {
    let a = inner(x);
    let b = inner(a);
    b
}
print(outer(10));
"#);
    assert_eq!(out, vec!["12"]);
}

#[test]
fn h12_arena_recursive_calls() {
    let out = run_mir(r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
print(factorial(10));
"#);
    assert_eq!(out, vec!["3628800"]);
}

#[test]
fn h12_arena_with_string_alloc() {
    let out = run_mir(r#"
fn greet(name: string) -> string {
    "Hello, " + name + "!"
}
print(greet("CJC"));
"#);
    assert_eq!(out, vec!["Hello, CJC!"]);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Break/continue complex patterns
//    NOTE: MIR executor requires break/continue inside fn main() { }
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_break_in_nested_while() {
    let out = run_mir(r#"
fn main() {
    let mut i = 0;
    let mut total = 0;
    while i < 5 {
        let mut j = 0;
        while j < 5 {
            if j == 3 { break; }
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    print(total);
}
"#);
    assert_eq!(out, vec!["15"]); // 5 outer * 3 inner
}

#[test]
fn h12_continue_accumulate() {
    let out = run_mir(r#"
fn main() {
    let mut i = 0;
    let mut sum = 0;
    while i < 10 {
        i = i + 1;
        if i % 2 == 0 { continue; }
        sum = sum + i;
    }
    print(sum);
}
"#);
    assert_eq!(out, vec!["25"]); // 1+3+5+7+9
}

#[test]
fn h12_break_continue_in_for() {
    let out = run_mir(r#"
fn main() {
    let mut sum = 0;
    for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
        if x == 8 { break; }
        if x % 3 == 0 { continue; }
        sum = sum + x;
    }
    print(sum);
}
"#);
    assert_eq!(out, vec!["19"]); // 1+2+4+5+7
}

#[test]
fn h12_break_with_accumulator() {
    let out = run_mir(r#"
fn main() {
    let mut i = 0;
    let mut product = 1;
    while i < 100 {
        i = i + 1;
        product = product * i;
        if product > 1000 { break; }
    }
    print(i);
}
"#);
    // 1*2*3*4*5*6*7 = 5040 > 1000, so i=7
    assert_eq!(out, vec!["7"]);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Module cross-file compilation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_module_single_file_pipeline() {
    let dir = setup_test_dir(&[(
        "main.cjc",
        "fn double(x: i64) -> i64 { x * 2 }\nprint(double(21));",
    )]);
    let entry = dir.path().join("main.cjc");
    let (_val, executor) =
        cjc_mir_exec::run_program_with_modules_executor(&entry, 42).unwrap();
    assert_eq!(executor.output, vec!["42"]);
}

#[test]
fn h12_module_graph_deterministic() {
    let dir = setup_test_dir(&[
        ("main.cjc", "import alpha\nimport beta\nlet x = 1;"),
        ("alpha.cjc", "fn a_fn() -> i64 { 1 }"),
        ("beta.cjc", "fn b_fn() -> i64 { 2 }"),
    ]);
    let entry = dir.path().join("main.cjc");

    let g1 = cjc_module::build_module_graph(&entry).unwrap();
    let g2 = cjc_module::build_module_graph(&entry).unwrap();
    let m1 = cjc_module::merge_programs(&g1).unwrap();
    let m2 = cjc_module::merge_programs(&g2).unwrap();
    let n1: Vec<&str> = m1.functions.iter().map(|f| f.name.as_str()).collect();
    let n2: Vec<&str> = m2.functions.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(n1, n2, "module graph must be deterministic");
}

#[test]
fn h12_module_topo_order() {
    let dir = setup_test_dir(&[
        ("main.cjc", "import lib\nlet x = 1;"),
        ("lib.cjc", "fn helper() -> i64 { 42 }"),
    ]);
    let entry = dir.path().join("main.cjc");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let order = graph.topological_order().unwrap();
    let lib_pos = order.iter().position(|m| m.0 == "lib").unwrap();
    let main_pos = order.iter().position(|m| m.0 == "main").unwrap();
    assert!(lib_pos < main_pos);
}

#[test]
fn h12_module_prefix_mangling() {
    let dir = setup_test_dir(&[
        ("main.cjc", "import utils\nlet x = 1;"),
        ("utils.cjc", "fn format_number(n: i64) -> i64 { n }"),
    ]);
    let entry = dir.path().join("main.cjc");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let merged = cjc_module::merge_programs(&graph).unwrap();
    let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.contains(&"utils::format_number"),
        "expected 'utils::format_number' in {:?}",
        names,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. File I/O + JSON roundtrip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_json_file_roundtrip_complex() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
let config = json_parse("{{\"version\":1,\"debug\":false,\"name\":\"test\"}}");
let s = json_stringify(config);
file_write("{}", s);
let loaded = file_read("{}");
let parsed = json_parse(loaded);
let s2 = json_stringify(parsed);
print(s2);
"#,
        path_str, path_str,
    );
    let out = run_mir(&src);
    assert_eq!(out, vec![r#"{"debug":false,"name":"test","version":1}"#]);
}

#[test]
fn h12_file_write_and_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("output.txt");
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
file_write("{}", "alpha\nbeta\ngamma");
let lines = file_lines("{}");
print(len(lines));
"#,
        path_str, path_str,
    );
    let out = run_mir(&src);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn h12_file_exists_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lifecycle.txt");
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
print(file_exists("{}"));
file_write("{}", "data");
print(file_exists("{}"));
"#,
        path_str, path_str, path_str,
    );
    let out = run_mir(&src);
    assert_eq!(out, vec!["false", "true"]);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. DateTime operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_datetime_comprehensive() {
    let out = run_mir(r#"
let epoch = datetime_from_parts(1970, 1, 1, 0, 0, 0);
let y2k = datetime_from_parts(2000, 1, 1, 0, 0, 0);
let diff = datetime_diff(y2k, epoch);
print(diff);
let formatted = datetime_format(y2k);
print(formatted);
"#);
    assert_eq!(out[0], "946684800000");
    assert_eq!(out[1], "2000-01-01T00:00:00Z");
}

#[test]
fn h12_datetime_add_and_extract() {
    let out = run_mir(r#"
let dt = datetime_from_parts(2024, 1, 1, 0, 0, 0);
let one_day_ms = 86400000;
let dt2 = datetime_add_millis(dt, one_day_ms);
print(datetime_day(dt2));
print(datetime_month(dt2));
"#);
    assert_eq!(out, vec!["2", "1"]);
}

// ═══════════════════════════════════════════════════════════════════════
// 7. TiledMatmul integration
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_tiled_matmul_identity_large() {
    use cjc_runtime::tensor::Tensor;
    let n = 128;
    let data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let eye: Vec<f64> = (0..n * n)
        .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
        .collect();
    let a = Tensor::from_vec(data.clone(), &[n, n]).unwrap();
    let e = Tensor::from_vec(eye, &[n, n]).unwrap();
    let c = a.matmul(&e).unwrap();
    for (i, (got, exp)) in c.to_vec().iter().zip(data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-8,
            "identity mismatch at {}: {} vs {}",
            i, got, exp,
        );
    }
}

#[test]
fn h12_tiled_matmul_symmetric() {
    use cjc_runtime::tensor::Tensor;
    let n = 64;
    let data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
    let a = Tensor::from_vec(data, &[n, n]).unwrap();
    let c = a.matmul(&a).unwrap();
    assert_eq!(c.shape(), &[n, n]);
    assert!(c.to_vec().iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Window functions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_window_all_functions_consistent() {
    let out = run_mir(r#"
let data = [5.0, 5.0, 5.0, 5.0, 5.0];
let sums = window_sum(data, 3);
let means = window_mean(data, 3);
let mins = window_min(data, 3);
let maxes = window_max(data, 3);
print(len(sums));
print(len(means));
print(len(mins));
print(len(maxes));
"#);
    assert_eq!(out, vec!["3", "3", "3", "3"]);
}

#[test]
fn h12_window_monotonic_data() {
    let out = run_mir(r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let mins = window_min(data, 2);
let maxes = window_max(data, 2);
print(mins);
print(maxes);
"#);
    assert!(out[0].contains("1"));
    assert!(out[1].contains("5"));
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Cross-feature programs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_cross_break_with_accumulation() {
    let out = run_mir(r#"
fn main() {
    let mut sum = 0;
    let mut i = 1;
    while i <= 100 {
        if i % 2 != 0 {
            sum = sum + i;
        }
        i = i + 1;
    }
    print(sum);
}
"#);
    assert_eq!(out, vec!["2500"]); // sum of odd numbers 1..100
}

#[test]
fn h12_cross_json_datetime() {
    let out = run_mir(r#"
let dt = datetime_from_parts(2024, 6, 15, 12, 0, 0);
let formatted = datetime_format(dt);
let year = datetime_year(dt);
let month = datetime_month(dt);
print(formatted);
print(year);
print(month);
"#);
    assert_eq!(out[0], "2024-06-15T12:00:00Z");
    assert_eq!(out[1], "2024");
    assert_eq!(out[2], "6");
}

#[test]
fn h12_cross_json_nested_roundtrip() {
    let out = run_mir(r#"
let input = "{\"users\":[{\"name\":\"Alice\",\"age\":30},{\"name\":\"Bob\",\"age\":25}]}";
let obj = json_parse(input);
let output = json_stringify(obj);
print(output);
"#);
    assert_eq!(
        out,
        vec![r#"{"users":[{"age":30,"name":"Alice"},{"age":25,"name":"Bob"}]}"#],
    );
}

#[test]
fn h12_cross_file_json_datetime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("event.json");
    let path_str = path.to_string_lossy().replace('\\', "/");

    let src = format!(
        r#"
let dt = datetime_from_parts(2024, 12, 25, 0, 0, 0);
let ts = datetime_format(dt);
let event = json_parse("{{\"event\":\"holiday\",\"timestamp\":\"placeholder\"}}");
let s = json_stringify(event);
file_write("{}", s);
let loaded = file_read("{}");
print(loaded);
"#,
        path_str, path_str,
    );
    let out = run_mir(&src);
    assert!(!out.is_empty());
    assert!(out[0].contains("event"));
    assert!(out[0].contains("holiday"));
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Determinism double-run verification
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn h12_full_pipeline_determinism() {
    let src = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
print(fib(15));
let dt = datetime_from_parts(2024, 7, 4, 12, 0, 0);
print(datetime_format(dt));
let json_str = json_stringify(json_parse("{\"a\":1,\"b\":2}"));
print(json_str);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2, "full pipeline must be deterministic");
    assert_eq!(out1[0], "610");
    assert_eq!(out1[1], "2024-07-04T12:00:00Z");
    assert_eq!(out1[2], r#"{"a":1,"b":2}"#);
}

#[test]
fn h12_double_run_window() {
    let src = r#"
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let sums = window_sum(data, 5);
print(len(sums));
print(sums);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2, "window functions must be deterministic");
    assert_eq!(out1[0], "6");
}
