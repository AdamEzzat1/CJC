// CJC Test Suite — End-to-End Pipeline Tests
// 10 realistic data science/analytics workflows exercising multiple features.

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

fn run_parity(src: &str) -> Vec<String> {
    let eval_out = run_eval(src);
    let mir_out = run_mir(src);
    assert_eq!(eval_out, mir_out, "PARITY FAILURE:\n  eval: {:?}\n  mir:  {:?}", eval_out, mir_out);
    eval_out
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 1: Survey data cleaning + factor analysis
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_survey_cleaning() {
    let out = run_parity(r#"
fn main() {
    // Raw survey responses with missing values
    let responses = ["good", NA, "bad", "good", NA, "neutral", "bad", "good", NA, "neutral"];
    let clean = drop_na(responses);
    let f = as_factor(clean);
    let counts = fct_count(f);
    print(factor_levels(f));
    print(counts);
    print(len(clean));
}
"#);
    assert_eq!(out[0], "[good, bad, neutral]");
    assert_eq!(out[1], "[(good, 3), (bad, 2), (neutral, 2)]");
    assert_eq!(out[2], "7");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 2: Numerical feature engineering
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_feature_engineering() {
    let out = run_parity(r#"
fn square(x: i64) -> i64 { return x * x; }
fn add(a: i64, b: i64) -> i64 { return a + b; }
fn positive(x: i64) -> bool { return x > 0; }
fn main() {
    let raw = range(1, 11);
    let squared = array_map(raw, square);
    let total = array_reduce(squared, 0, add);
    let n = len(raw);
    print(total);
    print(n);
    // All values should be positive
    print(array_all(raw, positive));
    // Find first value > 7
    let big = |x: i64| x > 7;
    print(array_find(raw, big));
}
"#);
    // sum of squares 1..10 = 385
    assert_eq!(out[0], "385");
    assert_eq!(out[1], "10");
    assert_eq!(out[2], "true");
    assert_eq!(out[3], "8");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 3: Category lumping + encoding
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_category_encoding() {
    let out = run_parity(r#"
fn main() {
    let colors = ["red", "red", "red", "blue", "blue", "green", "yellow", "purple", "orange", "red"];
    let f = as_factor(colors);
    // Keep top 2, lump rest into "Other"
    let simplified = fct_lump(f, 2);
    print(factor_levels(simplified));
    print(factor_codes(simplified));

    // Reorder: put blue before red
    let reordered = fct_relevel(simplified, ["blue", "red", "Other"]);
    print(factor_levels(reordered));
}
"#);
    assert_eq!(out[0], "[red, blue, Other]");
    // red=0(x4), blue=1(x2), green→Other=2, yellow→Other=2, purple→Other=2, orange→Other=2
    assert_eq!(out[1], "[0, 0, 0, 1, 1, 2, 2, 2, 2, 0]");
    assert_eq!(out[2], "[blue, red, Other]");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 4: NA-aware aggregation
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_na_aggregation() {
    let out = run_parity(r#"
fn add_f(a: f64, b: f64) -> f64 { return a + b; }
fn main() {
    let measurements = [1.5, NA, 2.3, NA, 4.1, NA, 3.7];
    let clean = drop_na(measurements);
    let n = len(clean);
    let total = array_reduce(clean, 0.0, add_f);
    print(n);
    print(total);
    // Verify NA count
    let na_count = len(measurements) - n;
    print(na_count);
}
"#);
    assert_eq!(out[0], "4");
    assert_eq!(out[1], "11.6");
    assert_eq!(out[2], "3");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 5: Paired data analysis
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_paired_data() {
    let out = run_parity(r#"
fn main() {
    let students = ["alice", "bob", "carol", "dave"];
    let scores_a = [85, 92, 78, 88];
    let scores_b = [90, 88, 82, 91];

    // Pair them
    let paired = array_zip(scores_a, scores_b);
    print(paired);

    // Index them
    let indexed = array_enumerate(students);
    print(indexed);

    // Unique students
    let all_names = ["alice", "bob", "alice", "carol", "bob", "dave"];
    print(array_unique(all_names));
}
"#);
    assert_eq!(out[0], "[(85, 90), (92, 88), (78, 82), (88, 91)]");
    assert_eq!(out[1], "[(0, alice), (1, bob), (2, carol), (3, dave)]");
    assert_eq!(out[2], "[alice, bob, carol, dave]");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 6: Sorting and ranking
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_sorting_ranking() {
    let out = run_parity(r#"
fn neg(x: i64) -> i64 { return 0 - x; }
fn identity(x: i64) -> i64 { return x; }
fn main() {
    let data = [42, 17, 93, 8, 55, 31];

    // Sort ascending
    let asc = array_sort_by(data, identity);
    print(asc);

    // Sort descending
    let desc = array_sort_by(data, neg);
    print(desc);

    // Top 3 (sort desc, then take first 3 manually)
    print(len(desc));
}
"#);
    assert_eq!(out[0], "[8, 17, 31, 42, 55, 93]");
    assert_eq!(out[1], "[93, 55, 42, 31, 17, 8]");
    assert_eq!(out[2], "6");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 7: Functional composition
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_functional_composition() {
    let out = run_parity(r#"
fn is_even(x: i64) -> bool { return x % 2 == 0; }
fn triple(x: i64) -> i64 { return x * 3; }
fn add(a: i64, b: i64) -> i64 { return a + b; }
fn main() {
    // Generate → Filter → Map → Reduce
    let data = range(1, 21);
    let evens = array_filter(data, is_even);
    let tripled = array_map(evens, triple);
    let total = array_reduce(tripled, 0, add);
    print(evens);
    print(tripled);
    print(total);
}
"#);
    // Evens: 2,4,6,8,10,12,14,16,18,20
    assert_eq!(out[0], "[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]");
    // Tripled: 6,12,18,24,30,36,42,48,54,60
    assert_eq!(out[1], "[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]");
    // Sum = 330
    assert_eq!(out[2], "330");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 8: Mixed types with NA handling
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_mixed_types_na() {
    let out = run_parity(r#"
fn main() {
    // Coalesce: fill in defaults for missing values
    print(coalesce(NA, 42));
    print(coalesce(7, 100));
    print(coalesce(NA, NA, "default"));

    // NA propagation in arithmetic
    let x = NA + 10;
    print(is_na(x));

    // fillna on array
    let arr = [1.0, NA, 3.0, NA, 5.0];
    let filled = fillna(arr, 0.0);
    print(filled);
}
"#);
    assert_eq!(out[0], "42");
    assert_eq!(out[1], "7");
    assert_eq!(out[2], "default");
    assert_eq!(out[3], "true");
    assert_eq!(out[4], "[1, 0, 3, 0, 5]");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 9: Data validation
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_data_validation() {
    let out = run_parity(r#"
fn in_range(x: i64) -> bool { return x >= 0 && x <= 100; }
fn has_na(x: Any) -> bool { return is_na(x); }
fn main() {
    let scores = [85, 92, -1, 101, 50, 73, 88];

    // Validate all in range
    let valid = array_all(scores, in_range);
    print(valid);

    // Find invalid values
    let invalid = |x: i64| x < 0 || x > 100;
    let any_invalid = array_any(scores, invalid);
    print(any_invalid);

    // Filter to valid only
    let clean = array_filter(scores, in_range);
    print(clean);
    print(len(clean));
}
"#);
    assert_eq!(out[0], "false"); // not all valid
    assert_eq!(out[1], "true");  // some invalid
    assert_eq!(out[2], "[85, 92, 50, 73, 88]");
    assert_eq!(out[3], "5");
}

// ════════════════════════════════════════════════════════════════════════
//  Pipeline 10: Full determinism stress test
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_determinism_stress() {
    let src = r#"
fn square(x: i64) -> i64 { return x * x; }
fn add(a: i64, b: i64) -> i64 { return a + b; }
fn is_odd(x: i64) -> bool { return x % 2 != 0; }
fn neg(x: i64) -> i64 { return 0 - x; }
fn main() {
    let data = range(1, 51);
    let odds = array_filter(data, is_odd);
    let sq = array_map(odds, square);
    let total = array_reduce(sq, 0, add);
    print(total);

    let sorted_desc = array_sort_by(data, neg);
    print(sorted_desc[0]);
    print(sorted_desc[49]);

    let f = as_factor(["a", "b", "c", "a", "b", "a"]);
    print(fct_count(f));

    print(is_na(NA));
    print(is_na(42));
    print(coalesce(NA, NA, 99));

    let uniq = array_unique([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    print(uniq);
}
"#;
    // Run 3 times — must produce identical output each time
    let out1 = run_parity(src);
    let out2 = run_parity(src);
    let out3 = run_parity(src);
    assert_eq!(out1, out2, "determinism failure (run 1 vs 2)");
    assert_eq!(out2, out3, "determinism failure (run 2 vs 3)");

    // Sum of squares of odd numbers 1..50: 1+9+25+49+...+2401 = 20825
    // Actually: 1^2 + 3^2 + 5^2 + ... + 49^2 = sum_{k=0}^{24} (2k+1)^2
    // = sum of first 25 odd squares = 25*(2*25-1)*(2*25+1)/3 = 25*49*51/3 = 20825
    assert_eq!(out1[0], "20825");
    assert_eq!(out1[1], "50");  // sorted desc, first element
    assert_eq!(out1[2], "1");   // sorted desc, last element
    assert_eq!(out1[3], "[(a, 3), (b, 2), (c, 1)]");
    assert_eq!(out1[4], "true");
    assert_eq!(out1[5], "false");
    assert_eq!(out1[6], "99");
    assert_eq!(out1[7], "[1, 2, 3, 4, 5]");
}
