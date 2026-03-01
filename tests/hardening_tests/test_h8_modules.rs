//! Hardening test H8: Module system integration tests.
//!
//! Tests the multi-file module pipeline: source files → module graph →
//! merged MIR program → execution.
//!
//! Current model: `import foo` brings all of `foo.cjc`'s declarations into
//! the merged program with `foo::` prefix. Calls use the mangled name directly.
//! A future step will add source-level qualified call rewriting.

use std::fs;

/// Helper: create a temp directory with CJC source files and return it.
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

// ── Module graph construction tests ─────────────────────────────────

#[test]
fn h8_build_graph_with_import() {
    let dir = setup_test_dir(&[
        ("main.cjc", "import math\nlet x = 1;"),
        ("math.cjc", "fn add(a: f64, b: f64) -> f64 { a + b }"),
    ]);
    let entry = dir.path().join("main.cjc");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    assert_eq!(graph.module_count(), 2);
}

#[test]
fn h8_topo_order_deps_first() {
    let dir = setup_test_dir(&[
        ("main.cjc", "import alpha\nlet x = 1;"),
        ("alpha.cjc", "fn a_fn() -> i64 { 1 }"),
    ]);
    let entry = dir.path().join("main.cjc");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let order = graph.topological_order().unwrap();

    let alpha_pos = order.iter().position(|m| m.0 == "alpha").unwrap();
    let main_pos = order.iter().position(|m| m.0 == "main").unwrap();
    assert!(alpha_pos < main_pos, "dependencies must come before dependents");
}

// ── Merge tests ─────────────────────────────────────────────────────

#[test]
fn h8_merge_prefixes_non_entry_functions() {
    let dir = setup_test_dir(&[
        ("main.cjc", "import math\nlet x = 1;"),
        ("math.cjc", "fn add(a: f64, b: f64) -> f64 { a + b }"),
    ]);
    let entry = dir.path().join("main.cjc");
    let graph = cjc_module::build_module_graph(&entry).unwrap();
    let merged = cjc_module::merge_programs(&graph).unwrap();

    let names: Vec<&str> = merged.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.contains(&"math::add"),
        "expected 'math::add' in {:?}",
        names
    );
}

// ── Execution tests ─────────────────────────────────────────────────

#[test]
fn h8_single_file_module_pipeline() {
    // Even a single file works through the module pipeline
    let dir = setup_test_dir(&[(
        "main.cjc",
        "fn square(x: i64) -> i64 { x * x }\nprint(square(7));",
    )]);
    let entry = dir.path().join("main.cjc");
    let (_val, executor) =
        cjc_mir_exec::run_program_with_modules_executor(&entry, 42).unwrap();
    assert_eq!(executor.output, vec!["49"]);
}

#[test]
fn h8_deterministic_merge_order() {
    // Build module graph multiple times — merged output must be identical
    let dir = setup_test_dir(&[
        ("main.cjc", "import alpha\nimport beta\nlet x = 1;"),
        ("alpha.cjc", "fn a_fn() -> i64 { 1 }"),
        ("beta.cjc", "fn b_fn() -> i64 { 2 }"),
    ]);
    let entry = dir.path().join("main.cjc");

    let graph1 = cjc_module::build_module_graph(&entry).unwrap();
    let merged1 = cjc_module::merge_programs(&graph1).unwrap();
    let names1: Vec<&str> = merged1.functions.iter().map(|f| f.name.as_str()).collect();

    let graph2 = cjc_module::build_module_graph(&entry).unwrap();
    let merged2 = cjc_module::merge_programs(&graph2).unwrap();
    let names2: Vec<&str> = merged2.functions.iter().map(|f| f.name.as_str()).collect();

    assert_eq!(names1, names2, "merge order must be deterministic");
}
