//! Module visibility enforcement tests.

use std::path::PathBuf;
use std::fs;

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

/// A call to an imported `pub fn` is resolved to its qualified name; no
/// unprefixed alias is created (ADR-0047).
#[test]
fn test_pub_fn_call_resolves_to_qualified_name() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import utils\nlet x = double(21);"),
        ("utils.cjcl", "pub fn double(n: i64) -> i64 { n * 2 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let merged = cjc_module::merge_programs(&graph).unwrap();
    let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"utils::double"), "{:?}", names);
    assert!(!names.contains(&"double"), "no unprefixed alias: {:?}", names);
}

#[test]
fn test_private_fn_not_aliased() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import utils\nlet x = 1;"),
        ("utils.cjcl", "pub fn public_fn() -> i64 { 1 }\nfn private_fn() -> i64 { 2 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let merged = cjc_module::merge_programs(&graph).unwrap();
    let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(!names.contains(&"private_fn"), "private fn should not be aliased");
    assert!(names.contains(&"utils::private_fn"), "private fn still exists with prefix");
}

#[test]
fn test_visibility_violation_detected() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import math.Matrix\nlet x = 1;"),
        ("math.cjcl", "struct Matrix { x: f64 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let violations = cjc_module::check_visibility(&graph);
    assert!(!violations.is_empty(), "should detect visibility violation");
    assert_eq!(violations[0].kind, "struct");
}

#[test]
fn test_pub_struct_no_violation() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import math.Matrix\nlet x = 1;"),
        ("math.cjcl", "pub struct Matrix { x: f64 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let violations = cjc_module::check_visibility(&graph);
    assert!(violations.is_empty(), "pub struct should not violate: {:?}", violations.iter().map(|v| v.to_string()).collect::<Vec<_>>());
}

#[test]
fn test_multi_module_topological_order() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import alpha\nimport beta\nlet x = 1;"),
        ("alpha.cjcl", "pub fn a() -> i64 { 1 }"),
        ("beta.cjcl", "import alpha\npub fn b() -> i64 { 2 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let order = graph.topological_order().unwrap();
    // alpha must come before beta (beta depends on alpha)
    let alpha_pos = order.iter().position(|m| m.0 == "alpha").unwrap();
    let beta_pos = order.iter().position(|m| m.0 == "beta").unwrap();
    assert!(alpha_pos < beta_pos, "alpha must be before beta in topo order");
}

#[test]
fn test_cyclic_dependency_detected() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import a\nlet x = 1;"),
        ("a.cjcl", "import b\npub fn fa() -> i64 { 1 }"),
        ("b.cjcl", "import a\npub fn fb() -> i64 { 2 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let result = cjc_module::build_module_graph(&entry);
    assert!(result.is_err(), "cyclic dependency should be detected");
}

#[test]
fn test_module_merge_deterministic() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import utils\nlet x = 1;"),
        ("utils.cjcl", "pub fn helper() -> i64 { 42 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    // Build twice and compare
    let graph1 = cjc_module::build_module_graph(&entry).unwrap();
    let merged1 = cjc_module::merge_programs(&graph1).unwrap();
    let graph2 = cjc_module::build_module_graph(&entry).unwrap();
    let merged2 = cjc_module::merge_programs(&graph2).unwrap();

    let names1: Vec<&str> = merged1.functions.iter().map(|f| f.name.as_str()).collect();
    let names2: Vec<&str> = merged2.functions.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names1, names2, "merge must be deterministic");
}

// ---------------------------------------------------------------------------
// Module-scoped name resolution (ADR-0047). `cjc_module::resolve_modules`
// qualifies every function reference before either executor runs, so both
// executors see the same names. A private function is not in scope outside
// its module; a failed call to one gets a note saying why.
// ---------------------------------------------------------------------------

/// Runs the program in both executors; returns (eval, mir) results with
/// values rendered via Display and errors via Debug.
fn run_both(files: &[(&str, &str)]) -> (Result<String, String>, Result<String, String>) {
    let dir = setup_test_dir(files);
    let entry: PathBuf = dir.path().join("main.cjcl");
    let eval = cjc_eval::run_program_with_modules_eval(&entry, 42)
        .map(|v| format!("{}", v))
        .map_err(|e| format!("{:?}", e));
    let mir = cjc_mir_exec::run_program_with_modules(&entry, 42)
        .map(|v| format!("{}", v))
        .map_err(|e| format!("{:?}", e));
    (eval, mir)
}

/// Both executors return `Ok(expected)`.
fn assert_both_ok(files: &[(&str, &str)], expected: &str) {
    let (eval, mir) = run_both(files);
    assert_eq!(eval, Ok(expected.to_string()), "AST-eval");
    assert_eq!(mir, Ok(expected.to_string()), "MIR-exec");
}

/// Both executors fail, and both errors contain every fragment.
fn assert_both_err(files: &[(&str, &str)], fragments: &[&str]) {
    let (eval, mir) = run_both(files);
    for (who, r) in [("AST-eval", eval), ("MIR-exec", mir)] {
        let e = r.expect_err(who);
        for frag in fragments {
            assert!(e.contains(frag), "{who} error lacks {frag:?}: {e}");
        }
    }
}

const UTILS: &str = "pub fn quad(x: i64) -> i64 { double(double(x)) }\n\
                     fn double(x: i64) -> i64 { x * 2 }";

const PRIVATE_NOTE: &str = "note: `double` is a private function of module `utils`; \
                            mark it `pub` to use it from other modules";

#[test]
fn pub_fn_calling_private_helper_runs_in_both_executors() {
    assert_both_ok(
        &[("main.cjcl", "import utils\nfn main() -> i64 { quad(3) }"), ("utils.cjcl", UTILS)],
        "12",
    );
}

#[test]
fn private_fn_is_not_in_scope_for_importer() {
    assert_both_err(
        &[("main.cjcl", "import utils\nfn main() -> i64 { double(3) }"), ("utils.cjcl", UTILS)],
        &["undefined function `double`", PRIVATE_NOTE],
    );
}

#[test]
fn private_fn_used_as_value_is_not_in_scope() {
    assert_both_err(
        &[
            ("main.cjcl", "import utils\nfn main() -> i64 { let f = double; f(3) }"),
            ("utils.cjcl", UTILS),
        ],
        &[PRIVATE_NOTE],
    );
}

#[test]
fn dependency_module_cannot_use_another_modules_private_fn() {
    assert_both_err(
        &[
            ("main.cjcl", "import alpha\nimport beta\nfn main() -> i64 { b(1) }"),
            ("alpha.cjcl", "fn secret(x: i64) -> i64 { x }"),
            ("beta.cjcl", "import alpha\npub fn b(x: i64) -> i64 { secret(x) }"),
        ],
        &["undefined function `secret`", "`secret` is a private function of module `alpha`"],
    );
}

#[test]
fn same_named_private_helpers_do_not_collide() {
    // Each module's `pub fn` must call its own `helper`. Before module-scoped
    // resolution AST-eval returned 202 and MIR-exec 101.
    assert_both_ok(
        &[
            ("main.cjcl", "import alpha\nimport beta\nfn main() -> i64 { fa() * 100 + fb() }"),
            ("alpha.cjcl", "pub fn fa() -> i64 { helper() }\nfn helper() -> i64 { 1 }"),
            ("beta.cjcl", "pub fn fb() -> i64 { helper() }\nfn helper() -> i64 { 2 }"),
        ],
        "102",
    );
}

#[test]
fn nested_import_resolves_in_both_executors() {
    // MIR-exec used to fail with "undefined function `fb`": only the entry
    // module's imports were aliased.
    assert_both_ok(
        &[
            ("main.cjcl", "import alpha\nfn main() -> i64 { fa() }"),
            ("alpha.cjcl", "import beta\npub fn fa() -> i64 { fb() + 1 }"),
            ("beta.cjcl", "pub fn fb() -> i64 { 10 }"),
        ],
        "11",
    );
}

#[test]
fn private_fn_does_not_shadow_builtin_in_importer() {
    // main's `abs` is the builtin (2.0); alpha's `fa` calls alpha's own
    // private `abs` (100.0).
    assert_both_ok(
        &[
            ("main.cjcl", "import alpha\nfn main() -> f64 { abs(-2.0) + fa() }"),
            ("alpha.cjcl", "pub fn fa() -> f64 { abs(1.0) }\nfn abs(x: f64) -> f64 { 100.0 }"),
        ],
        "102",
    );
}

#[test]
fn local_binding_shadows_imported_fn() {
    // `double` in main is the local variable, not utils' function.
    assert_both_ok(
        &[
            ("main.cjcl", "import utils\nfn main() -> i64 { let double = 5; double + quad(1) }"),
            ("utils.cjcl", UTILS),
        ],
        "9",
    );
}

#[test]
fn own_fn_takes_precedence_over_imported_one() {
    assert_both_ok(
        &[
            ("main.cjcl", "import alpha\nfn f(x: i64) -> i64 { x + 1 }\nfn main() -> i64 { f(1) }"),
            ("alpha.cjcl", "pub fn f(x: i64) -> i64 { x * 100 }"),
        ],
        "2",
    );
}

#[test]
fn private_fn_elsewhere_does_not_hide_pub_fn_of_same_name() {
    assert_both_ok(
        &[
            ("main.cjcl", "import alpha\nimport beta\nfn main() -> i64 { shared(1) }"),
            ("alpha.cjcl", "pub fn shared(x: i64) -> i64 { x }"),
            ("beta.cjcl", "fn shared(x: i64) -> i64 { x + 1 }"),
        ],
        "1",
    );
}

#[test]
fn symbol_import_with_alias_resolves() {
    assert_both_ok(
        &[("main.cjcl", "import utils.quad as q\nfn main() -> i64 { q(2) }"), ("utils.cjcl", UTILS)],
        "8",
    );
}

#[test]
fn recursion_and_lambdas_inside_a_module_resolve() {
    assert_both_ok(
        &[
            ("main.cjcl", "import m\nfn main() -> i64 { run(5) }"),
            (
                "m.cjcl",
                "fn fact(n: i64) -> i64 { if n <= 1 { 1 } else { n * fact(n - 1) } }\n\
                 pub fn run(n: i64) -> i64 { let g = |x: i64| fact(x); g(n) }",
            ),
        ],
        "120",
    );
}

#[test]
fn resolution_is_deterministic() {
    let files = [
        ("main.cjcl", "import alpha\nimport beta\nfn main() -> i64 { a1(1) + b1(2) }"),
        ("alpha.cjcl", "pub fn a1(x: i64) -> i64 { h(x) }\nfn h(x: i64) -> i64 { x }"),
        (
            "beta.cjcl",
            "import alpha\npub fn b1(x: i64) -> i64 { a1(x) + h(x) }\nfn h(x: i64) -> i64 { x * 3 }",
        ),
    ];
    let dir = setup_test_dir(&files);
    let graph = cjc_module::build_module_graph(&dir.path().join("main.cjcl")).unwrap();
    let render = |r: &cjc_module::ResolvedModules| format!("{:?}", r);
    let first = render(&cjc_module::resolve_modules(&graph).unwrap());
    for _ in 0..5 {
        assert_eq!(render(&cjc_module::resolve_modules(&graph).unwrap()), first);
    }
    // a1(1) = 1; b1(2) = a1(2) + beta's h(2) = 2 + 6.
    assert_both_ok(&files, "9");
}
