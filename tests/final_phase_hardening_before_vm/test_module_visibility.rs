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
    assert!(names.contains(&"public_fn"), "pub fn should be aliased");
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
