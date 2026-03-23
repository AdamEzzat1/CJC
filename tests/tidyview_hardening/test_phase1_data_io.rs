//! TidyView Phase 1: Data I/O builtins — read_csv, write_csv, dir_list, path_join.
//! Also verifies snap_save / snap_load (already wired) via CJC source.

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

/// Helper to create a unique temp directory for each test.
fn temp_dir(test_name: &str) -> String {
    let dir = format!("{}/__tidyview_test_{}", env!("CARGO_MANIFEST_DIR"), test_name);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn cleanup(dir: &str) {
    let _ = std::fs::remove_dir_all(dir);
}

// ──────────────────────────────────────────────────────────────────
// 1. read_csv / write_csv roundtrip
// ──────────────────────────────────────────────────────────────────

#[test]
fn read_csv_eval() {
    let dir = temp_dir("read_csv_eval");
    let csv_path = format!("{}/data.csv", dir);
    std::fs::write(&csv_path, "name,age,score\nAlice,30,9.5\nBob,25,8.1\n").unwrap();

    let src = format!(r#"
let df = read_csv("{csv_path}");
print(df.__nrows);
print(len(df.name));
print(df.name[0]);
print(df.age[1]);
print(df.score[0]);
"#, csv_path = csv_path.replace('\\', "\\\\"));

    let out = run_eval(&src, 42);
    assert_eq!(out[0], "2");
    assert_eq!(out[1], "2");
    assert_eq!(out[2], "Alice");
    assert_eq!(out[3], "25");
    assert_eq!(out[4], "9.5");
    cleanup(&dir);
}

#[test]
fn read_csv_mir() {
    let dir = temp_dir("read_csv_mir");
    let csv_path = format!("{}/data.csv", dir);
    std::fs::write(&csv_path, "name,age,score\nAlice,30,9.5\nBob,25,8.1\n").unwrap();

    let src = format!(r#"
let df = read_csv("{csv_path}");
print(df.__nrows);
print(len(df.name));
print(df.name[0]);
print(df.age[1]);
print(df.score[0]);
"#, csv_path = csv_path.replace('\\', "\\\\"));

    let out = run_mir(&src, 42);
    assert_eq!(out[0], "2");
    assert_eq!(out[1], "2");
    assert_eq!(out[2], "Alice");
    assert_eq!(out[3], "25");
    assert_eq!(out[4], "9.5");
    cleanup(&dir);
}

#[test]
fn read_csv_parity() {
    let dir = temp_dir("read_csv_parity");
    let csv_path = format!("{}/data.csv", dir);
    std::fs::write(&csv_path, "x,y\n1.0,2.0\n3.0,4.0\n").unwrap();

    let src = format!(r#"
let df = read_csv("{csv_path}");
print(df.__nrows);
print(df.x[0]);
print(df.y[1]);
"#, csv_path = csv_path.replace('\\', "\\\\"));

    let eval_out = run_eval(&src, 42);
    let mir_out = run_mir(&src, 42);
    assert_eq!(eval_out, mir_out, "read_csv parity failed");
    cleanup(&dir);
}

#[test]
fn write_csv_roundtrip_eval() {
    let dir = temp_dir("write_csv_rt_eval");
    let csv_in = format!("{}/in.csv", dir);
    let csv_out = format!("{}/out.csv", dir);
    // Use values that won't be inferred as Bool (avoid 0/1)
    std::fs::write(&csv_in, "a,b\n10,hello\n20,world\n").unwrap();

    let src = format!(r#"
let df = read_csv("{csv_in}");
let ok = write_csv("{csv_out}", df);
print(ok);
let df2 = read_csv("{csv_out}");
print(df2.__nrows);
print(df2.a[0]);
print(df2.b[1]);
"#,
        csv_in = csv_in.replace('\\', "\\\\"),
        csv_out = csv_out.replace('\\', "\\\\"),
    );

    let out = run_eval(&src, 42);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "2");
    assert_eq!(out[2], "10");
    assert_eq!(out[3], "world");
    cleanup(&dir);
}

#[test]
fn write_csv_roundtrip_mir() {
    let dir = temp_dir("write_csv_rt_mir");
    let csv_in = format!("{}/in.csv", dir);
    let csv_out = format!("{}/out.csv", dir);
    std::fs::write(&csv_in, "a,b\n10,hello\n20,world\n").unwrap();

    let src = format!(r#"
let df = read_csv("{csv_in}");
let ok = write_csv("{csv_out}", df);
print(ok);
let df2 = read_csv("{csv_out}");
print(df2.__nrows);
print(df2.a[0]);
print(df2.b[1]);
"#,
        csv_in = csv_in.replace('\\', "\\\\"),
        csv_out = csv_out.replace('\\', "\\\\"),
    );

    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "2");
    assert_eq!(out[2], "10");
    assert_eq!(out[3], "world");
    cleanup(&dir);
}

// ──────────────────────────────────────────────────────────────────
// 2. snap_save / snap_load roundtrip
// ──────────────────────────────────────────────────────────────────

#[test]
fn snap_roundtrip_eval() {
    let dir = temp_dir("snap_rt_eval");
    let snap_path = format!("{}/value.snap", dir);

    let src = format!(r#"
snap_save(42, "{snap_path}");
let v = snap_load("{snap_path}");
print(v);
"#, snap_path = snap_path.replace('\\', "\\\\"));

    let out = run_eval(&src, 42);
    assert_eq!(out, vec!["42"]);
    cleanup(&dir);
}

#[test]
fn snap_roundtrip_mir() {
    let dir = temp_dir("snap_rt_mir");
    let snap_path = format!("{}/value.snap", dir);

    let src = format!(r#"
snap_save(42, "{snap_path}");
let v = snap_load("{snap_path}");
print(v);
"#, snap_path = snap_path.replace('\\', "\\\\"));

    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["42"]);
    cleanup(&dir);
}

#[test]
fn snap_roundtrip_parity() {
    let dir = temp_dir("snap_rt_parity");
    let snap_eval = format!("{}/eval.snap", dir);
    let snap_mir = format!("{}/mir.snap", dir);

    let src_eval = format!(r#"
snap_save(3.14, "{path}");
let v = snap_load("{path}");
print(v);
"#, path = snap_eval.replace('\\', "\\\\"));

    let src_mir = format!(r#"
snap_save(3.14, "{path}");
let v = snap_load("{path}");
print(v);
"#, path = snap_mir.replace('\\', "\\\\"));

    let eval_out = run_eval(&src_eval, 42);
    let mir_out = run_mir(&src_mir, 42);
    assert_eq!(eval_out, mir_out, "snap_save/snap_load parity failed");
    cleanup(&dir);
}

// ──────────────────────────────────────────────────────────────────
// 3. dir_list returns sorted results
// ──────────────────────────────────────────────────────────────────

#[test]
fn dir_list_sorted_eval() {
    let dir = temp_dir("dir_list_eval");
    // Create files in non-alphabetical order
    std::fs::write(format!("{}/charlie.txt", dir), "c").unwrap();
    std::fs::write(format!("{}/alpha.txt", dir), "a").unwrap();
    std::fs::write(format!("{}/bravo.txt", dir), "b").unwrap();

    let src = format!(r#"
let entries = dir_list("{dir}");
print(len(entries));
print(entries[0]);
print(entries[1]);
print(entries[2]);
"#, dir = dir.replace('\\', "\\\\"));

    let out = run_eval(&src, 42);
    assert_eq!(out[0], "3");
    assert_eq!(out[1], "alpha.txt");
    assert_eq!(out[2], "bravo.txt");
    assert_eq!(out[3], "charlie.txt");
    cleanup(&dir);
}

#[test]
fn dir_list_sorted_mir() {
    let dir = temp_dir("dir_list_mir");
    std::fs::write(format!("{}/charlie.txt", dir), "c").unwrap();
    std::fs::write(format!("{}/alpha.txt", dir), "a").unwrap();
    std::fs::write(format!("{}/bravo.txt", dir), "b").unwrap();

    let src = format!(r#"
let entries = dir_list("{dir}");
print(len(entries));
print(entries[0]);
print(entries[1]);
print(entries[2]);
"#, dir = dir.replace('\\', "\\\\"));

    let out = run_mir(&src, 42);
    assert_eq!(out[0], "3");
    assert_eq!(out[1], "alpha.txt");
    assert_eq!(out[2], "bravo.txt");
    assert_eq!(out[3], "charlie.txt");
    cleanup(&dir);
}

#[test]
fn dir_list_parity() {
    let dir = temp_dir("dir_list_parity");
    std::fs::write(format!("{}/z.txt", dir), "z").unwrap();
    std::fs::write(format!("{}/a.txt", dir), "a").unwrap();

    let src = format!(r#"
let entries = dir_list("{dir}");
print(len(entries));
print(entries[0]);
print(entries[1]);
"#, dir = dir.replace('\\', "\\\\"));

    let eval_out = run_eval(&src, 42);
    let mir_out = run_mir(&src, 42);
    assert_eq!(eval_out, mir_out, "dir_list parity failed");
    cleanup(&dir);
}

// ──────────────────────────────────────────────────────────────────
// 4. path_join
// ──────────────────────────────────────────────────────────────────

#[test]
fn path_join_eval() {
    let src = r#"
let p = path_join("/tmp", "file.txt");
print(p);
"#;
    let out = run_eval(src, 42);
    // On Windows this may produce /tmp\file.txt or /tmp/file.txt
    assert!(out[0].contains("file.txt"), "Expected path containing file.txt, got: {}", out[0]);
    assert!(out[0].starts_with("/tmp"), "Expected path starting with /tmp, got: {}", out[0]);
}

#[test]
fn path_join_mir() {
    let src = r#"
let p = path_join("/tmp", "file.txt");
print(p);
"#;
    let out = run_mir(src, 42);
    assert!(out[0].contains("file.txt"), "Expected path containing file.txt, got: {}", out[0]);
    assert!(out[0].starts_with("/tmp"), "Expected path starting with /tmp, got: {}", out[0]);
}

#[test]
fn path_join_parity() {
    let src = r#"
let p = path_join("base", "sub");
print(p);
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "path_join parity failed");
}

// ──────────────────────────────────────────────────────────────────
// 5. Error handling
// ──────────────────────────────────────────────────────────────────

#[test]
fn read_csv_missing_file_eval() {
    let src = r#"
let df = read_csv("/nonexistent/path/data.csv");
"#;
    let (program, diag) = cjc_parser::parse_source(src);
    assert!(!diag.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "Expected error for missing CSV file");
}

#[test]
fn read_csv_missing_file_mir() {
    let src = r#"
let df = read_csv("/nonexistent/path/data.csv");
"#;
    let (program, diag) = cjc_parser::parse_source(src);
    assert!(!diag.has_errors());
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_err(), "Expected error for missing CSV file");
}

#[test]
fn dir_list_missing_dir_eval() {
    let src = r#"
let entries = dir_list("/nonexistent/dir/xyz");
"#;
    let (program, diag) = cjc_parser::parse_source(src);
    assert!(!diag.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "Expected error for missing directory");
}

#[test]
fn dir_list_missing_dir_mir() {
    let src = r#"
let entries = dir_list("/nonexistent/dir/xyz");
"#;
    let (program, diag) = cjc_parser::parse_source(src);
    assert!(!diag.has_errors());
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_err(), "Expected error for missing directory");
}

// ──────────────────────────────────────────────────────────────────
// 6. Determinism: repeated reads produce identical results
// ──────────────────────────────────────────────────────────────────

#[test]
fn read_csv_determinism() {
    let dir = temp_dir("read_csv_determinism");
    let csv_path = format!("{}/data.csv", dir);
    std::fs::write(&csv_path, "x,y,z\n1,2,3\n4,5,6\n7,8,9\n").unwrap();

    let src = format!(r#"
let df = read_csv("{csv_path}");
print(df.x[0]);
print(df.y[1]);
print(df.z[2]);
"#, csv_path = csv_path.replace('\\', "\\\\"));

    let out1 = run_eval(&src, 42);
    let out2 = run_eval(&src, 42);
    assert_eq!(out1, out2, "read_csv must be deterministic across runs");
    cleanup(&dir);
}
