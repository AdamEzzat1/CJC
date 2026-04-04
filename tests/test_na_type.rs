// CJC Test Suite — NA Type, Array HOFs, Categorical, DataFrame Inspection
// Tests for Features 1-4 of the data science gap analysis implementation.

// ── Helpers ──────────────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("parse error:\n{}", diags.render_all(src, "<test>"));
    }
    program
}

fn run_eval(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn run_mir(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

/// Run both executors and assert they produce identical output.
fn run_parity(src: &str) -> Vec<String> {
    let eval_out = run_eval(src);
    let mir_out = run_mir(src);
    assert_eq!(eval_out, mir_out, "PARITY FAILURE:\n  eval: {:?}\n  mir:  {:?}", eval_out, mir_out);
    eval_out
}

// ════════════════════════════════════════════════════════════════════════
//  Feature 1: NA Type
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_na_literal() {
    let out = run_parity("fn main() { print(NA); }");
    assert_eq!(out, vec!["NA"]);
}

#[test]
fn test_na_equality() {
    // NA == NA → false (SQL semantics)
    let out = run_parity("fn main() { print(NA == NA); }");
    assert_eq!(out, vec!["false"]);
}

#[test]
fn test_na_inequality() {
    // NA != NA → true
    let out = run_parity("fn main() { print(NA != NA); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_na_arithmetic_propagation() {
    let out = run_parity("fn main() { print(NA + 1); print(NA * 3.14); print(NA - 10); }");
    assert_eq!(out, vec!["NA", "NA", "NA"]);
}

#[test]
fn test_na_comparison_propagation() {
    let out = run_parity("fn main() { print(NA < 5); print(NA > 0); print(NA >= 1); }");
    assert_eq!(out, vec!["NA", "NA", "NA"]);
}

#[test]
fn test_na_logical_propagation() {
    let out = run_parity("fn main() { print(NA && true); print(NA || false); }");
    assert_eq!(out, vec!["NA", "NA"]);
}

#[test]
fn test_is_na_true() {
    let out = run_parity("fn main() { print(is_na(NA)); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_is_na_false() {
    let out = run_parity("fn main() { print(is_na(42)); print(is_na(3.14)); print(is_na(\"hello\")); }");
    assert_eq!(out, vec!["false", "false", "false"]);
}

#[test]
fn test_drop_na() {
    let out = run_parity(r#"
fn main() {
    let arr = [1, NA, 2, NA, 3];
    let clean = drop_na(arr);
    print(clean);
}
"#);
    assert_eq!(out, vec!["[1, 2, 3]"]);
}

#[test]
fn test_na_in_array() {
    let out = run_parity(r#"
fn main() {
    let arr = [1, NA, 3];
    print(arr);
}
"#);
    assert_eq!(out, vec!["[1, NA, 3]"]);
}

#[test]
fn test_na_coalesce() {
    let out = run_parity(r#"
fn main() {
    print(coalesce(NA, 42));
    print(coalesce(7, 42));
    print(coalesce(NA, NA, 99));
}
"#);
    assert_eq!(out, vec!["42", "7", "99"]);
}

#[test]
fn test_na_is_not_null() {
    let out = run_parity("fn main() { print(is_not_null(NA)); print(is_not_null(1)); }");
    assert_eq!(out, vec!["false", "true"]);
}

// ── NA Determinism ──────────────────────────────────────────────────────

#[test]
fn test_na_determinism() {
    let src = r#"
fn main() {
    let arr = [1, NA, 2, NA, 3];
    let clean = drop_na(arr);
    print(is_na(NA));
    print(NA == NA);
    print(NA + 1);
    print(clean);
}
"#;
    // Run twice — must produce identical output
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "NA determinism failure");
}

// ════════════════════════════════════════════════════════════════════════
//  Feature 4: Array Higher-Order Functions
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_range_one_arg() {
    let out = run_parity("fn main() { print(range(5)); }");
    assert_eq!(out, vec!["[0, 1, 2, 3, 4]"]);
}

#[test]
fn test_range_two_args() {
    let out = run_parity("fn main() { print(range(2, 6)); }");
    assert_eq!(out, vec!["[2, 3, 4, 5]"]);
}

#[test]
fn test_range_three_args() {
    let out = run_parity("fn main() { print(range(0, 10, 3)); }");
    assert_eq!(out, vec!["[0, 3, 6, 9]"]);
}

#[test]
fn test_range_negative_step() {
    let out = run_parity("fn main() { print(range(10, 0, -3)); }");
    assert_eq!(out, vec!["[10, 7, 4, 1]"]);
}

#[test]
fn test_array_map_named_fn() {
    let out = run_parity(r#"
fn double(x: i64) -> i64 { return x * 2; }
fn main() {
    let arr = [1, 2, 3, 4];
    let result = array_map(arr, double);
    print(result);
}
"#);
    assert_eq!(out, vec!["[2, 4, 6, 8]"]);
}

#[test]
fn test_array_map_closure() {
    let out = run_parity(r#"
fn main() {
    let arr = [1, 2, 3];
    let factor = 10;
    let mul = |x: i64| x * factor;
    let result = array_map(arr, mul);
    print(result);
}
"#);
    assert_eq!(out, vec!["[10, 20, 30]"]);
}

#[test]
fn test_array_filter() {
    let out = run_parity(r#"
fn is_even(x: i64) -> bool { return x % 2 == 0; }
fn main() {
    let arr = [1, 2, 3, 4, 5, 6];
    let result = array_filter(arr, is_even);
    print(result);
}
"#);
    assert_eq!(out, vec!["[2, 4, 6]"]);
}

#[test]
fn test_array_reduce() {
    let out = run_parity(r#"
fn add(a: i64, b: i64) -> i64 { return a + b; }
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let total = array_reduce(arr, 0, add);
    print(total);
}
"#);
    assert_eq!(out, vec!["15"]);
}

#[test]
fn test_array_any() {
    let out = run_parity(r#"
fn gt3(x: i64) -> bool { return x > 3; }
fn main() {
    print(array_any([1, 2, 3], gt3));
    print(array_any([1, 2, 5], gt3));
}
"#);
    assert_eq!(out, vec!["false", "true"]);
}

#[test]
fn test_array_all() {
    let out = run_parity(r#"
fn positive(x: i64) -> bool { return x > 0; }
fn main() {
    print(array_all([1, 2, 3], positive));
    print(array_all([1, -1, 3], positive));
}
"#);
    assert_eq!(out, vec!["true", "false"]);
}

#[test]
fn test_array_find() {
    let out = run_parity(r#"
fn gt3(x: i64) -> bool { return x > 3; }
fn main() {
    print(array_find([1, 2, 5, 8], gt3));
    print(array_find([1, 2, 3], gt3));
    print(is_na(array_find([1, 2, 3], gt3)));
}
"#);
    assert_eq!(out, vec!["5", "NA", "true"]);
}

#[test]
fn test_array_enumerate() {
    let out = run_parity(r#"
fn main() {
    let result = array_enumerate(["a", "b", "c"]);
    print(result);
}
"#);
    assert_eq!(out, vec!["[(0, a), (1, b), (2, c)]"]);
}

#[test]
fn test_array_zip() {
    let out = run_parity(r#"
fn main() {
    let result = array_zip([1, 2, 3], ["a", "b", "c"]);
    print(result);
}
"#);
    assert_eq!(out, vec!["[(1, a), (2, b), (3, c)]"]);
}

#[test]
fn test_array_sort_by() {
    let out = run_parity(r#"
fn neg(x: i64) -> i64 { return 0 - x; }
fn main() {
    let arr = [3, 1, 4, 1, 5];
    let result = array_sort_by(arr, neg);
    print(result);
}
"#);
    assert_eq!(out, vec!["[5, 4, 3, 1, 1]"]);
}

#[test]
fn test_array_unique() {
    let out = run_parity(r#"
fn main() {
    let arr = [1, 2, 3, 2, 1, 4];
    let result = array_unique(arr);
    print(result);
}
"#);
    assert_eq!(out, vec!["[1, 2, 3, 4]"]);
}

#[test]
fn test_array_filter_with_closure() {
    let out = run_parity(r#"
fn main() {
    let threshold = 3;
    let arr = [1, 5, 2, 7, 3, 9];
    let gt_thresh = |x: i64| x > threshold;
    let result = array_filter(arr, gt_thresh);
    print(result);
}
"#);
    assert_eq!(out, vec!["[5, 7, 9]"]);
}

#[test]
fn test_array_hof_chaining() {
    // Filter, then map — functional pipeline
    let out = run_parity(r#"
fn is_even(x: i64) -> bool { return x % 2 == 0; }
fn double(x: i64) -> i64 { return x * 2; }
fn main() {
    let arr = [1, 2, 3, 4, 5, 6];
    let evens = array_filter(arr, is_even);
    let doubled = array_map(evens, double);
    print(doubled);
}
"#);
    assert_eq!(out, vec!["[4, 8, 12]"]);
}

// ── Array HOF Determinism ───────────────────────────────────────────────

#[test]
fn test_array_hof_determinism() {
    let src = r#"
fn neg(x: i64) -> i64 { return 0 - x; }
fn is_pos(x: i64) -> bool { return x > 0; }
fn main() {
    let arr = [5, -3, 2, -1, 4];
    print(array_sort_by(arr, neg));
    print(array_filter(arr, is_pos));
    print(array_unique([1, 1, 2, 2, 3]));
    print(range(1, 6, 2));
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "Array HOF determinism failure");
}

// ════════════════════════════════════════════════════════════════════════
//  Feature 3: Categorical / Factor Builtins
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_as_factor_basic() {
    let out = run_parity(r#"
fn main() {
    let f = as_factor(["red", "blue", "red", "green", "blue"]);
    print(factor_levels(f));
}
"#);
    assert_eq!(out, vec!["[red, blue, green]"]);
}

#[test]
fn test_factor_codes() {
    let out = run_parity(r#"
fn main() {
    let f = as_factor(["a", "b", "a", "c"]);
    print(factor_codes(f));
}
"#);
    assert_eq!(out, vec!["[0, 1, 0, 2]"]);
}

#[test]
fn test_fct_relevel() {
    let out = run_parity(r#"
fn main() {
    let f = as_factor(["low", "med", "high", "low"]);
    let reordered = fct_relevel(f, ["high", "med", "low"]);
    print(factor_levels(reordered));
    print(factor_codes(reordered));
}
"#);
    // After releveling: levels = [high, med, low]
    // Original codes: low=0, med=1, high=2
    // New mapping: high→0, med→1, low→2
    assert_eq!(out[0], "[high, med, low]");
    assert_eq!(out[1], "[2, 1, 0, 2]"); // low→2, med→1, high→0, low→2
}

#[test]
fn test_fct_lump() {
    let out = run_parity(r#"
fn main() {
    let f = as_factor(["a", "a", "a", "b", "b", "c", "d", "e"]);
    let lumped = fct_lump(f, 2);
    print(factor_levels(lumped));
}
"#);
    // Top 2 most frequent: "a" (3), "b" (2). Rest lumped to "Other".
    assert_eq!(out, vec!["[a, b, Other]"]);
}

#[test]
fn test_fct_count() {
    let out = run_parity(r#"
fn main() {
    let f = as_factor(["x", "y", "x", "x", "y", "z"]);
    let counts = fct_count(f);
    print(counts);
}
"#);
    assert_eq!(out, vec!["[(x, 3), (y, 2), (z, 1)]"]);
}

#[test]
fn test_factor_determinism() {
    let src = r#"
fn main() {
    let f = as_factor(["b", "a", "c", "a", "b", "c", "c"]);
    print(factor_levels(f));
    print(factor_codes(f));
    let lumped = fct_lump(f, 1);
    print(factor_levels(lumped));
}
"#;
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    assert_eq!(out1, out2, "Factor determinism failure");
}

// ════════════════════════════════════════════════════════════════════════
//  Feature 5: End-to-End Pipeline Tests (mini versions)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_e2e_data_cleaning_pipeline() {
    // Simulate: create data → filter NAs → transform → reduce
    let out = run_parity(r#"
fn double(x: i64) -> i64 { return x * 2; }
fn add(a: i64, b: i64) -> i64 { return a + b; }
fn main() {
    let raw = [1, NA, 3, NA, 5];
    let clean = drop_na(raw);
    let transformed = array_map(clean, double);
    let total = array_reduce(transformed, 0, add);
    print(total);
}
"#);
    // (1+3+5) * 2 = 18
    assert_eq!(out, vec!["18"]);
}

#[test]
fn test_e2e_categorical_analysis() {
    let out = run_parity(r#"
fn main() {
    let data = ["cat", "dog", "cat", "bird", "dog", "cat", "fish", "bird"];
    let f = as_factor(data);
    print(factor_levels(f));
    let counts = fct_count(f);
    print(counts);
    let lumped = fct_lump(f, 2);
    print(factor_levels(lumped));
}
"#);
    assert_eq!(out[0], "[cat, dog, bird, fish]");
    assert_eq!(out[1], "[(cat, 3), (dog, 2), (bird, 2), (fish, 1)]");
    // Top 2: cat (3), dog (2) — bird ties with dog but comes after in BTreeMap ordering
    // Actually the sort is by frequency desc then code asc, so dog(code=1, freq=2) and bird(code=2, freq=2)
    // dog appears first since code 1 < code 2
    assert_eq!(out[2], "[cat, dog, Other]");
}

#[test]
fn test_e2e_functional_statistics() {
    // Compute sum of squares using array HOFs
    let out = run_parity(r#"
fn square(x: i64) -> i64 { return x * x; }
fn add(a: i64, b: i64) -> i64 { return a + b; }
fn main() {
    let data = range(1, 6);
    let squares = array_map(data, square);
    let sum_sq = array_reduce(squares, 0, add);
    print(sum_sq);
}
"#);
    // 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
    assert_eq!(out, vec!["55"]);
}

#[test]
fn test_e2e_filter_sort_pipeline() {
    let out = run_parity(r#"
fn abs_val(x: i64) -> i64 {
    if x < 0 { return 0 - x; }
    return x;
}
fn positive(x: i64) -> bool { return x > 0; }
fn main() {
    let data = [3, -1, 4, -1, 5, -9, 2, 6];
    let positives = array_filter(data, positive);
    let sorted = array_sort_by(positives, abs_val);
    print(sorted);
    let gt5 = |x: i64| x > 5;
    print(array_any(data, gt5));
    print(array_all(positives, positive));
}
"#);
    assert_eq!(out[0], "[2, 3, 4, 5, 6]");
    assert_eq!(out[1], "true");
    assert_eq!(out[2], "true");
}

#[test]
fn test_e2e_zip_enumerate() {
    let out = run_parity(r#"
fn main() {
    let names = ["alice", "bob", "carol"];
    let scores = [95, 87, 92];
    let paired = array_zip(names, scores);
    print(paired);
    let indexed = array_enumerate(scores);
    print(indexed);
}
"#);
    assert_eq!(out[0], "[(alice, 95), (bob, 87), (carol, 92)]");
    assert_eq!(out[1], "[(0, 95), (1, 87), (2, 92)]");
}

#[test]
fn test_e2e_na_aware_analysis() {
    // Full pipeline with NA handling
    let out = run_parity(r#"
fn not_na(x: Any) -> bool { return is_na(x) == false; }
fn main() {
    let raw = [10, NA, 20, NA, 30, NA, 40];
    let clean = drop_na(raw);
    let n = len(clean);
    print(n);
    print(array_find(raw, not_na));
    print(is_na(NA + 1));
}
"#);
    assert_eq!(out[0], "4");  // 4 non-NA values
    assert_eq!(out[1], "10"); // first non-NA
    assert_eq!(out[2], "true");
}
