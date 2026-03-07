// CJC Test Suite — Vizor parity (AST-eval vs MIR-exec)
// Ensures both execution backends produce identical Vizor output.

use cjc_eval::Interpreter;
use cjc_mir_exec::run_program_with_executor;
use cjc_parser::parse_source;

fn assert_output_parity(src: &str) {
    let (program, diags) = parse_source(src);
    assert!(
        !diags.has_errors(),
        "Parse errors: {}",
        diags.render_all(src, "<test>")
    );

    // AST-eval
    let mut interp = Interpreter::new(42);
    interp.exec(&program).expect("AST eval failed");
    let ast_out = interp.output.clone();

    // MIR-exec
    let (_, exec) = run_program_with_executor(&program, 42).expect("MIR exec failed");
    let mir_out = exec.output;

    assert_eq!(
        ast_out, mir_out,
        "Output mismatch between AST-eval and MIR-exec"
    );
}

#[test]
fn parity_basic_scatter_print() {
    assert_output_parity("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_point();\nprint(p);");
}

#[test]
fn parity_line_plot_print() {
    assert_output_parity("import vizor\nlet p = vizor_plot([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0]);\nlet p = p.geom_line();\nlet p = p.title(\"Quadratic\");\nprint(p);");
}

#[test]
fn parity_bar_chart_print() {
    assert_output_parity("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [25.0, 50.0, 75.0]);\nlet p = p.geom_bar();\nlet p = p.title(\"Bars\");\nlet p = p.xlab(\"Category\");\nlet p = p.ylab(\"Count\");\nprint(p);");
}

#[test]
fn parity_scatter_svg_via_save() {
    // Both backends save to files and we compare the SVG output
    let tmpdir = std::env::temp_dir();
    let path_ast = tmpdir.join(format!("parity_ast_{}.svg", std::process::id()));
    let path_mir = tmpdir.join(format!("parity_mir_{}.svg", std::process::id()));

    let base = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_point();\nlet p = p.title(\"Parity\");";

    // AST-eval
    let src_ast = format!("{}\np.save(\"{}\");", base, path_ast.to_string_lossy().replace('\\', "/"));
    let (prog_ast, diags) = parse_source(&src_ast);
    assert!(!diags.has_errors());
    let mut interp = Interpreter::new(42);
    interp.exec(&prog_ast).expect("AST eval failed");

    // MIR-exec
    let src_mir = format!("{}\np.save(\"{}\");", base, path_mir.to_string_lossy().replace('\\', "/"));
    let (prog_mir, diags) = parse_source(&src_mir);
    assert!(!diags.has_errors());
    run_program_with_executor(&prog_mir, 42).expect("MIR exec failed");

    let ast_svg = std::fs::read_to_string(&path_ast).expect("read AST SVG");
    let mir_svg = std::fs::read_to_string(&path_mir).expect("read MIR SVG");

    assert_eq!(ast_svg, mir_svg, "SVG output should match between AST-eval and MIR-exec");

    std::fs::remove_file(&path_ast).ok();
    std::fs::remove_file(&path_mir).ok();
}

#[test]
fn parity_bmp_via_save() {
    let tmpdir = std::env::temp_dir();
    let path_ast = tmpdir.join(format!("parity_ast_{}.bmp", std::process::id()));
    let path_mir = tmpdir.join(format!("parity_mir_{}.bmp", std::process::id()));

    let base = "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();";

    let src_ast = format!("{}\np.save(\"{}\");", base, path_ast.to_string_lossy().replace('\\', "/"));
    let (prog_ast, diags) = parse_source(&src_ast);
    assert!(!diags.has_errors());
    let mut interp = Interpreter::new(42);
    interp.exec(&prog_ast).expect("AST eval failed");

    let src_mir = format!("{}\np.save(\"{}\");", base, path_mir.to_string_lossy().replace('\\', "/"));
    let (prog_mir, diags) = parse_source(&src_mir);
    assert!(!diags.has_errors());
    run_program_with_executor(&prog_mir, 42).expect("MIR exec failed");

    let ast_bytes = std::fs::read(&path_ast).expect("read AST BMP");
    let mir_bytes = std::fs::read(&path_mir).expect("read MIR BMP");

    assert_eq!(ast_bytes, mir_bytes, "BMP bytes should match between AST-eval and MIR-exec");

    std::fs::remove_file(&path_ast).ok();
    std::fs::remove_file(&path_mir).ok();
}
