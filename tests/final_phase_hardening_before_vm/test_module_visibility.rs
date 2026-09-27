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

#[test]
fn test_pub_fn_aliased_in_merged_program() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import utils\nlet x = double(21);"),
        ("utils.cjcl", "pub fn double(n: i64) -> i64 { n * 2 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let merged = cjc_module::merge_programs(&graph).unwrap();
    let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"double"), "pub fn should be aliased: {:?}", names);
}

/// Private functions of an imported module are aliased too. Merging renames
/// functions to `utils::name` but leaves call sites in bodies unprefixed, so
/// a `pub fn` that calls a private helper resolves the helper only through
/// its unprefixed alias. Matches `test_visibility_pub_functions_aliased` in
/// `cjc-module`. Other modules still cannot use the alias: `check_visibility`
/// rejects any reference to it (see the `private_fn_*` tests below).
#[test]
fn test_private_fn_aliased_for_intra_module_calls() {
    let dir = setup_test_dir(&[
        ("main.cjcl", "import utils\nlet x = 1;"),
        ("utils.cjcl", "pub fn public_fn() -> i64 { 1 }\nfn private_fn() -> i64 { 2 }"),
    ]);
    let entry = dir.path().join("main.cjcl");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let merged = cjc_module::merge_programs(&graph).unwrap();
    let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"public_fn"), "pub fn should be aliased");
    assert!(names.contains(&"private_fn"), "private fn is aliased for intra-module calls");
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
// Private functions are not usable from other modules. `check_visibility`
// is a static check on the ASTs, and both executors' multi-file entry points
// run it first, so they accept and reject exactly the same programs.
// ---------------------------------------------------------------------------

fn violation_strings(files: &[(&str, &str)]) -> Vec<String> {
    let dir = setup_test_dir(files);
    let graph = cjc_module::build_module_graph(&dir.path().join("main.cjcl")).unwrap();
    cjc_module::check_visibility(&graph)
        .iter()
        .map(|v| v.to_string())
        .collect()
}

/// Runs the program in both executors; returns (eval, mir) results with
/// values and errors rendered as strings.
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

const UTILS: &str = "pub fn quad(x: i64) -> i64 { double(double(x)) }\n\
                     fn double(x: i64) -> i64 { x * 2 }";

#[test]
fn private_fn_call_from_importer_is_rejected() {
    let files = [
        ("main.cjcl", "import utils\nfn main() -> i64 { double(3) }"),
        ("utils.cjcl", UTILS),
    ];
    let v = violation_strings(&files);
    assert_eq!(
        v,
        vec!["function `double` is private to module `utils` and cannot be used \
              from module `main` (mark it `pub` to export it)"
            .to_string()]
    );

    let (eval, mir) = run_both(&files);
    let (eval_err, mir_err) = (eval.unwrap_err(), mir.unwrap_err());
    assert!(eval_err.contains("visibility error: function `double` is private"), "{eval_err}");
    assert!(mir_err.contains("visibility error: function `double` is private"), "{mir_err}");
}

#[test]
fn pub_fn_calling_private_helper_runs_in_both_executors() {
    let files = [
        ("main.cjcl", "import utils\nfn main() -> i64 { quad(3) }"),
        ("utils.cjcl", UTILS),
    ];
    assert!(violation_strings(&files).is_empty());
    let (eval, mir) = run_both(&files);
    assert_eq!(eval, Ok("12".to_string()));
    assert_eq!(mir, Ok("12".to_string()));
}

#[test]
fn private_fn_used_as_value_is_rejected() {
    let v = violation_strings(&[
        ("main.cjcl", "import utils\nlet f = double;"),
        ("utils.cjcl", UTILS),
    ]);
    assert_eq!(v.len(), 1, "{v:?}");
    assert!(v[0].contains("`double` is private to module `utils`"), "{v:?}");
}

#[test]
fn local_binding_with_private_fn_name_is_not_a_violation() {
    // `double` here is the importer's own variable, not utils' function.
    let files = [
        ("main.cjcl", "import utils\nfn main() -> i64 { let double = 5; double + quad(1) }"),
        ("utils.cjcl", UTILS),
    ];
    assert!(violation_strings(&files).is_empty());
    let (eval, mir) = run_both(&files);
    assert_eq!(eval, Ok("9".to_string()));
    assert_eq!(mir, eval);
}

#[test]
fn own_private_fn_of_same_name_is_not_a_violation() {
    let v = violation_strings(&[
        ("main.cjcl", "import utils\nfn double(x: i64) -> i64 { x + x }\nlet y = double(2);"),
        ("utils.cjcl", UTILS),
    ]);
    assert!(v.is_empty(), "{v:?}");
}

#[test]
fn name_that_is_pub_in_some_module_is_not_a_violation() {
    let v = violation_strings(&[
        ("main.cjcl", "import alpha\nimport beta\nlet y = shared(1);"),
        ("alpha.cjcl", "pub fn shared(x: i64) -> i64 { x }"),
        ("beta.cjcl", "fn shared(x: i64) -> i64 { x + 1 }"),
    ]);
    assert!(v.is_empty(), "{v:?}");
}

#[test]
fn dependency_module_using_another_modules_private_fn_is_rejected() {
    let v = violation_strings(&[
        ("main.cjcl", "import alpha\nimport beta\nlet y = b(1);"),
        ("alpha.cjcl", "fn secret(x: i64) -> i64 { x }"),
        ("beta.cjcl", "import alpha\npub fn b(x: i64) -> i64 { secret(x) }"),
    ]);
    assert_eq!(
        v,
        vec!["function `secret` is private to module `alpha` and cannot be used \
              from module `beta` (mark it `pub` to export it)"
            .to_string()]
    );
}

#[test]
fn visibility_violations_are_deterministic() {
    let files = [
        ("main.cjcl", "import alpha\nimport beta\nlet y = s1(1) + s2(2);"),
        ("alpha.cjcl", "fn s2(x: i64) -> i64 { x }\nfn s1(x: i64) -> i64 { x }"),
        ("beta.cjcl", "import alpha\nlet z = s1(3);"),
    ];
    let first = violation_strings(&files);
    assert_eq!(first.len(), 3, "{first:?}");
    for _ in 0..5 {
        assert_eq!(violation_strings(&files), first);
    }
}
