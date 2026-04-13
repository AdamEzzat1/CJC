//! Tests for the `file_append` builtin (Phase C2 of Chess RL v2.1).
//!
//! This is a simple file-I/O builtin but it crosses the shared dispatch
//! boundary so we want explicit parity + non-panic coverage.

use bolero::check;
use std::fs;

fn run_eval(src: &str) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(1);
    interp
        .exec(&prog)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

fn run_mir(src: &str) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_val, executor) =
        cjc_mir_exec::run_program_with_executor(&prog, 1).unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

fn tmp_path(tag: &str) -> String {
    let pid = std::process::id();
    let thread_id = format!("{:?}", std::thread::current().id());
    let safe: String = thread_id.chars().filter(|c| c.is_alphanumeric()).collect();
    let raw = std::env::temp_dir()
        .join(format!("{tag}_{pid}_{safe}.txt"))
        .to_string_lossy()
        .into_owned();
    raw.replace('\\', "/")
}

/// file_append appends bytes to a new file.
#[test]
fn file_append_creates_new() {
    let path = tmp_path("file_append_new");
    let src = format!(
        r#"
        file_append("{path}", "line1\n");
        file_append("{path}", "line2\n");
        let content = file_read("{path}");
        print(content);
    "#
    );
    let out = run_eval(&src);
    let joined = out.join("\n");
    assert!(joined.contains("line1"));
    assert!(joined.contains("line2"));
    let _ = fs::remove_file(&path);
}

/// file_append vs file_write: append preserves prior content.
#[test]
fn file_append_preserves() {
    let path = tmp_path("file_append_preserve");
    let src = format!(
        r#"
        file_write("{path}", "first\n");
        file_append("{path}", "second\n");
        let content = file_read("{path}");
        print(content);
    "#
    );
    let out = run_eval(&src);
    let joined = out.join("\n");
    assert!(joined.contains("first"));
    assert!(joined.contains("second"));
    let _ = fs::remove_file(&path);
}

/// Cross-executor parity: same program writes the same bytes.
#[test]
fn file_append_parity() {
    let path_e = tmp_path("file_append_parity_e");
    let path_m = tmp_path("file_append_parity_m");
    let src_e = format!(
        r#"
        file_append("{path_e}", "a\n");
        file_append("{path_e}", "b\n");
        let content = file_read("{path_e}");
        print(content);
    "#
    );
    let src_m = format!(
        r#"
        file_append("{path_m}", "a\n");
        file_append("{path_m}", "b\n");
        let content = file_read("{path_m}");
        print(content);
    "#
    );
    let e = run_eval(&src_e);
    let m = run_mir(&src_m);
    assert_eq!(
        e, m,
        "file_append parity violation\neval: {e:?}\nmir: {m:?}"
    );
    let _ = fs::remove_file(&path_e);
    let _ = fs::remove_file(&path_m);
}

/// Fuzz: random byte strings appended to a file never panic the builtin.
#[test]
fn fuzz_file_append_no_panic() {
    check!().with_type::<Vec<u8>>().for_each(|bytes| {
        let path = tmp_path("file_append_fuzz");
        // Trim to ASCII-printable so we don't have to deal with invalid UTF-8
        // in the CJC-Lang string literal path.
        let safe: String = bytes
            .iter()
            .filter(|b| **b >= 0x20 && **b <= 0x7E && **b != b'"' && **b != b'\\')
            .take(64)
            .map(|b| *b as char)
            .collect();
        let src = format!(
            r#"
            file_append("{path}", "{safe}");
        "#
        );
        // Must not panic; parse/exec errors are fine.
        let (prog, _diags) = cjc_parser::parse_source(&src);
        let mut interp = cjc_eval::Interpreter::new(1);
        let _ = interp.exec(&prog);
        let _ = fs::remove_file(&path);
    });
}
