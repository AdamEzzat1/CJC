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

// ========================================================================
// Parity for Phase 2B: Distribution geoms
// ========================================================================

#[test]
fn parity_density_print() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_density();\nprint(p);"
    );
}

#[test]
fn parity_ecdf_print() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([3.0, 1.0, 4.0, 1.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_ecdf();\nprint(p);"
    );
}

#[test]
fn parity_area_print() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 15.0]);\nlet p = p.geom_area();\nprint(p);"
    );
}

#[test]
fn parity_rug_print() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_rug();\nprint(p);"
    );
}

// ========================================================================
// Parity for Phase 2B: Categorical geoms
// ========================================================================

#[test]
fn parity_box_print() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_box();\nprint(p);"
    );
}

#[test]
fn parity_violin_print() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_violin();\nprint(p);"
    );
}

#[test]
fn parity_strip_print() {
    assert_output_parity(
        "import vizor\nlet cats = [\"X\", \"X\", \"Y\", \"Y\"];\nlet vals = [1.0, 2.0, 3.0, 4.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_strip();\nprint(p);"
    );
}

#[test]
fn parity_boxen_print() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_boxen();\nprint(p);"
    );
}

// ========================================================================
// Parity for new Phase 2B constructors
// ========================================================================

#[test]
fn parity_cat_plot_print() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [10.0, 20.0, 30.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_bar();\nprint(p);"
    );
}

#[test]
fn parity_matrix_plot_print() {
    assert_output_parity(
        "import vizor\nlet mat = [[1.0, 2.0], [3.0, 4.0]];\nlet rows = [\"r0\", \"r1\"];\nlet cols = [\"c0\", \"c1\"];\nlet p = vizor_plot_matrix(mat, rows, cols);\nprint(p);"
    );
}

#[test]
fn parity_cat_plot_svg() {
    let tmpdir = std::env::temp_dir();
    let path_ast = tmpdir.join(format!("parity_cat_ast_{}.svg", std::process::id()));
    let path_mir = tmpdir.join(format!("parity_cat_mir_{}.svg", std::process::id()));

    let base = "import vizor\nlet cats = [\"X\", \"Y\", \"Z\"];\nlet vals = [5.0, 10.0, 15.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_bar();\nlet p = p.title(\"Cat Parity\");";

    let src_ast = format!("{}\np.save(\"{}\");", base, path_ast.to_string_lossy().replace('\\', "/"));
    let (prog_ast, diags) = parse_source(&src_ast);
    assert!(!diags.has_errors());
    let mut interp = Interpreter::new(42);
    interp.exec(&prog_ast).expect("AST eval failed");

    let src_mir = format!("{}\np.save(\"{}\");", base, path_mir.to_string_lossy().replace('\\', "/"));
    let (prog_mir, diags) = parse_source(&src_mir);
    assert!(!diags.has_errors());
    run_program_with_executor(&prog_mir, 42).expect("MIR exec failed");

    let ast_svg = std::fs::read_to_string(&path_ast).expect("read AST SVG");
    let mir_svg = std::fs::read_to_string(&path_mir).expect("read MIR SVG");

    assert_eq!(ast_svg, mir_svg, "SVG output should match between AST-eval and MIR-exec for cat plot");

    std::fs::remove_file(&path_ast).ok();
    std::fs::remove_file(&path_mir).ok();
}

// ── Phase 2B: Regression parity ──

#[test]
fn parity_regression_line() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_regression();\nlet p = p.title(\"Regression\");\nprint(p);",
    );
}

#[test]
fn parity_residual_plot() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.5, 4.0, 6.5, 7.0, 10.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_residplot();\nlet p = p.title(\"Residuals\");\nprint(p);",
    );
}

#[test]
fn parity_theme_publication() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.theme_publication();\nprint(p);",
    );
}

#[test]
fn parity_theme_dark() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.theme_dark();\nprint(p);",
    );
}

#[test]
fn parity_clustermap() {
    assert_output_parity(
        "import vizor\nlet c1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];\nlet labels = [\"a\", \"b\", \"c\"];\nlet p = vizor_clustermap(c1, labels);\nprint(p);",
    );
}

// ========================================================================
// Phase 3: Polar geom parity tests
// ========================================================================

#[test]
fn parity_geom_pie() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_pie();\nprint(p);",
    );
}

#[test]
fn parity_geom_donut() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_donut(0.4);\nprint(p);",
    );
}

#[test]
fn parity_geom_rose() {
    assert_output_parity(
        "import vizor\nlet cats = [\"N\", \"E\", \"S\", \"W\"];\nlet vals = [10.0, 20.0, 15.0, 25.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_rose();\nprint(p);",
    );
}

#[test]
fn parity_geom_radar() {
    assert_output_parity(
        "import vizor\nlet cats = [\"Speed\", \"Str\", \"Def\"];\nlet vals = [8.0, 6.0, 7.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_radar();\nprint(p);",
    );
}

#[test]
fn parity_coord_polar() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.coord_polar();\nprint(p);",
    );
}

#[test]
fn parity_geom_density2d() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_density2d();\nprint(p);",
    );
}

#[test]
fn parity_geom_contour() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_contour();\nprint(p);",
    );
}

// ── Phase 3.2: Error bars, step, legend, subtitle, log scales ──

#[test]
fn parity_geom_errorbar() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_errorbar();\nprint(p);",
    );
}

#[test]
fn parity_geom_step() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]);\nlet p = p.geom_step();\nprint(p);",
    );
}

#[test]
fn parity_subtitle() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point().title(\"T\").subtitle(\"S\");\nprint(p);",
    );
}

#[test]
fn parity_no_legend() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point().geom_line().no_legend();\nprint(p);",
    );
}

#[test]
fn parity_scale_x_log() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 10.0, 100.0], [1.0, 2.0, 3.0]);\nlet p = p.geom_point().scale_x_log(10.0);\nprint(p);",
    );
}

#[test]
fn parity_scale_y_log() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [1.0, 10.0, 100.0]);\nlet p = p.geom_point().scale_y_log(10.0);\nprint(p);",
    );
}
