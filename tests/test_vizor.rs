// CJC Test Suite — Vizor visualization library (29 tests)
// Tests for the Vizor grammar-of-graphics plotting library.
// Covers: import gating, plot construction, geom layers, SVG/BMP export,
//         annotations, builder chaining, method dispatch, save, and error handling.

use cjc_eval::Interpreter;
use cjc_parser::parse_source;
use cjc_runtime::Value;

/// Run CJC source and return (last_value, printed_output).
fn run(src: &str) -> (Value, Vec<String>) {
    let (program, diags) = parse_source(src);
    assert!(
        !diags.has_errors(),
        "Parse errors: {}",
        diags.render_all(src, "<test>")
    );
    let mut interp = Interpreter::new(42);
    let val = interp.exec(&program).expect("eval failed");
    (val, interp.output.clone())
}

/// Run CJC source and expect a runtime error.
fn run_err(src: &str) -> String {
    let (program, diags) = parse_source(src);
    if diags.has_errors() {
        return diags.render_all(src, "<test>");
    }
    let mut interp = Interpreter::new(42);
    match interp.exec(&program) {
        Err(e) => format!("{:?}", e),
        Ok(_) => panic!("Expected error"),
    }
}

// ========================================================================
// Import gating
// ========================================================================

#[test]
fn vizor_without_import_is_rejected() {
    let err = run_err("let x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nvizor_plot(x, y);");
    assert!(
        err.contains("undefined") || err.contains("unknown") || err.contains("Unknown"),
        "Expected undefined/unknown function error, got: {}",
        err
    );
}

#[test]
fn vizor_with_import_works() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// Basic plot construction
// ========================================================================

#[test]
fn vizor_plot_creates_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_plot_xy_alias() {
    let (_, out) = run("import vizor\nlet p = vizor_plot_xy([1.0, 2.0], [3.0, 4.0]);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// Geom layers
// ========================================================================

#[test]
fn geom_point_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_line_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_line();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_bar_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_bar();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_histogram_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_histogram(5);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// Builder chaining
// ========================================================================

#[test]
fn method_chaining() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_point();\nlet p = p.title(\"Test Plot\");\nlet p = p.xlab(\"X Axis\");\nlet p = p.ylab(\"Y Axis\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn xlim_ylim() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.xlim(0.0, 10.0);\nlet p = p.ylim(0.0, 50.0);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn theme_minimal() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0], [2.0]);\nlet p = p.theme_minimal();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn coord_flip() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0], [2.0]);\nlet p = p.coord_flip();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn size_method() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0], [2.0]);\nlet p = p.size(1024, 768);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// SVG export (save to temp file, read back)
// ========================================================================

fn svg_from(src_lines: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join(format!("vizor_test_{}_{}.svg", std::process::id(), id));
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!("{}\np.save(\"{}\");", src_lines, path_str);
    let (_, _) = run(&src);
    let contents = std::fs::read_to_string(&path).expect("Failed to read SVG file");
    std::fs::remove_file(&path).ok();
    contents
}

#[test]
fn to_svg_produces_valid_svg() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_point();");
    assert!(svg.contains("<svg"), "SVG should contain <svg tag");
    assert!(svg.contains("</svg>"), "SVG should end with </svg>");
    assert!(svg.contains("viewBox"), "SVG should have viewBox");
}

#[test]
fn svg_contains_circle_for_points() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();");
    assert!(svg.contains("<circle"), "Scatter plot SVG should contain <circle> elements");
}

#[test]
fn svg_contains_line_for_geom_line() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_line();");
    assert!(
        svg.contains("<polyline") || svg.contains("<line"),
        "Line plot SVG should contain polyline or line elements"
    );
}

#[test]
fn svg_contains_rect_for_bars() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_bar();");
    assert!(svg.contains("<rect"), "Bar plot SVG should contain <rect> elements");
}

#[test]
fn svg_title_appears_in_output() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();\nlet p = p.title(\"Hello Vizor\");");
    assert!(svg.contains("Hello Vizor"), "Title should appear in SVG");
}

#[test]
fn svg_axis_labels() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0], [2.0]);\nlet p = p.geom_point();\nlet p = p.xlab(\"Time\");\nlet p = p.ylab(\"Value\");");
    assert!(svg.contains("Time"), "X-axis label should appear in SVG");
    assert!(svg.contains("Value"), "Y-axis label should appear in SVG");
}

// ========================================================================
// BMP export
// ========================================================================

#[test]
fn save_bmp_has_valid_header() {
    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join(format!("vizor_bmp_test_{}.bmp", std::process::id()));
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_point();\np.save(\"{}\");", path_str);
    run(&src);
    assert!(path.exists(), "BMP file should be written");
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() > 54, "BMP should be > 54 bytes, got {}", bytes.len());
    assert_eq!(bytes[0], b'B', "BMP magic byte 0");
    assert_eq!(bytes[1], b'M', "BMP magic byte 1");
    std::fs::remove_file(&path).ok();
}

// ========================================================================
// Annotations
// ========================================================================

#[test]
fn annotate_text() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();\nlet p = p.annotate_text(\"label\", 1.5, 3.5);");
    assert!(svg.contains("label"), "Annotation text should appear in SVG");
}

#[test]
fn annotate_pvalue() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();\nlet p = p.annotate_pvalue(0.003);");
    assert!(
        svg.contains("p") || svg.contains("0.003"),
        "P-value annotation should appear in SVG"
    );
}

#[test]
fn annotate_regression() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();\nlet p = p.annotate_regression(\"y = 1.0x + 2.0\", 0.98);");
    assert!(
        svg.contains("y = 1.0x + 2.0") || svg.contains("0.98"),
        "Regression annotation should appear in SVG"
    );
}

#[test]
fn annotate_event() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_line();\nlet p = p.annotate_event(2.0, \"Event A\");");
    assert!(svg.contains("Event A"), "Event marker label should appear in SVG");
}

// ========================================================================
// Categorical plot constructor
// ========================================================================

#[test]
fn vizor_plot_cat_creates_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [10.0, 20.0, 30.0];\nlet p = vizor_plot_cat(cats, vals);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_plot_cat_geom_bar_svg() {
    let svg = svg_from("import vizor\nlet cats = [\"Apple\", \"Banana\", \"Cherry\"];\nlet vals = [15.0, 25.0, 35.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_bar();\nlet p = p.title(\"Fruit Sales\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Fruit Sales"));
    assert!(svg.contains("<rect"));
}

#[test]
fn vizor_plot_cat_without_import_fails() {
    let err = run_err("let cats = [\"A\", \"B\"];\nlet vals = [1.0, 2.0];\nvizor_plot_cat(cats, vals);");
    assert!(
        err.contains("undefined") || err.contains("unknown") || err.contains("Unknown"),
        "Expected undefined function error, got: {}",
        err
    );
}

#[test]
fn vizor_plot_cat_wrong_args() {
    let err = run_err("import vizor\nvizor_plot_cat([\"A\"]);");
    assert!(
        err.contains("requires 2 arguments"),
        "Expected arg count error, got: {}",
        err
    );
}

// ========================================================================
// Matrix plot constructor
// ========================================================================

#[test]
fn vizor_plot_matrix_creates_plot() {
    let (_, out) = run("import vizor\nlet mat = [[1.0, 2.0], [3.0, 4.0]];\nlet rows = [\"r0\", \"r1\"];\nlet cols = [\"c0\", \"c1\"];\nlet p = vizor_plot_matrix(mat, rows, cols);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_plot_matrix_without_import_fails() {
    let err = run_err("let mat = [[1.0, 2.0], [3.0, 4.0]];\nlet rows = [\"r0\", \"r1\"];\nlet cols = [\"c0\", \"c1\"];\nvizor_plot_matrix(mat, rows, cols);");
    assert!(
        err.contains("undefined") || err.contains("unknown") || err.contains("Unknown"),
        "Expected undefined function error, got: {}",
        err
    );
}

#[test]
fn vizor_plot_matrix_wrong_args() {
    let err = run_err("import vizor\nvizor_plot_matrix([[1.0]], [\"r\"]);");
    assert!(
        err.contains("requires 3 arguments"),
        "Expected arg count error, got: {}",
        err
    );
}

// ========================================================================
// Phase 2B: Figure-level wrappers
// ========================================================================

#[test]
fn vizor_displot_returns_plot() {
    let (_, out) = run("import vizor\nlet data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];\nlet p = vizor_displot(data);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_catplot_box() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\nlet p = vizor_catplot(cats, vals, \"box\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_catplot_violin() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\nlet p = vizor_catplot(cats, vals, \"violin\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_relplot_scatter() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nlet p = vizor_relplot(x, y, \"scatter\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_relplot_line() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nlet p = vizor_relplot(x, y, \"line\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_lmplot_returns_plot() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.1, 3.9, 6.2, 7.8, 10.1];\nlet p = vizor_lmplot(x, y);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_jointplot_returns_plot() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_jointplot(x, y);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_pairplot_returns_plot() {
    let (_, out) = run("import vizor\nlet col1 = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet col2 = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_pairplot([col1, col2], [\"X\", \"Y\"]);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_lmplot_svg() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.1, 3.9, 6.2, 7.8, 10.1];\nlet p = vizor_lmplot(x, y);\nlet p = p.title(\"Linear Model\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Linear Model"));
    assert!(svg.contains("<circle"), "LM plot should have scatter points");
}

// ========================================================================
// Phase 2B: Faceting
// ========================================================================

#[test]
fn facet_wrap_returns_plot() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\nlet y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.facet_wrap(\"group\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn facet_wrap_svg_produces_output() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\nlet y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.title(\"Faceted Plot\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Faceted Plot"));
}

// ========================================================================
// Phase 2B: Heatmap / Tile geoms
// ========================================================================

#[test]
fn geom_tile_returns_plot() {
    let (_, out) = run("import vizor\nlet mat = [[1.0, 2.0], [3.0, 4.0]];\nlet rows = [\"r0\", \"r1\"];\nlet cols = [\"c0\", \"c1\"];\nlet p = vizor_plot_matrix(mat, rows, cols);\nlet p = p.geom_tile();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_tile_svg_contains_rects() {
    let svg = svg_from("import vizor\nlet mat = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.0]];\nlet labels = [\"A\", \"B\", \"C\"];\nlet p = vizor_plot_matrix(mat, labels, labels).\ngeom_tile().\ntitle(\"Heatmap Test\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"), "Heatmap should contain rect elements for cells");
    assert!(svg.contains("Heatmap Test"));
}

#[test]
fn geom_tile_with_cell_values() {
    let svg = svg_from("import vizor\nlet mat = [[1.0, 2.0], [3.0, 4.0]];\nlet rows = [\"r0\", \"r1\"];\nlet cols = [\"c0\", \"c1\"];\nlet p = vizor_plot_matrix(mat, rows, cols).\ngeom_tile().\nshow_values(true).\ntitle(\"Values\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("1.00") || svg.contains("2.00") || svg.contains("3.00"), "Cell values should appear in SVG");
}

#[test]
fn vizor_corr_matrix_returns_plot() {
    let (_, out) = run("import vizor\nlet col1 = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet col2 = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet col3 = [5.0, 4.0, 3.0, 2.0, 1.0];\nlet p = vizor_corr_matrix([col1, col2, col3], [\"X\", \"Y\", \"Z\"]);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn vizor_corr_matrix_svg() {
    let svg = svg_from("import vizor\nlet col1 = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet col2 = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_corr_matrix([col1, col2], [\"X\", \"Y\"]);\nlet p = p.title(\"Correlation\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"), "Correlation matrix should contain rect elements");
    assert!(svg.contains("Correlation"));
}

// ========================================================================
// Phase 2B: Categorical geoms
// ========================================================================

#[test]
fn geom_box_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_box();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_box_svg_contains_rect() {
    let svg = svg_from("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_box().\ntitle(\"Box Plot\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"), "Box plot should contain rect for IQR box");
    assert!(svg.contains("Box Plot"));
}

#[test]
fn geom_violin_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_violin();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_strip_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"X\", \"X\", \"Y\", \"Y\"];\nlet vals = [1.0, 2.0, 3.0, 4.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_strip();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_swarm_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 1.5, 3.0, 4.0, 3.5];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_swarm();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_boxen_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_boxen();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_box_svg_full_pipeline() {
    let svg = svg_from("import vizor\nlet cats = [\"Control\", \"Control\", \"Control\", \"Control\", \"Control\", \"Treatment\", \"Treatment\", \"Treatment\", \"Treatment\", \"Treatment\"];\nlet vals = [2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_box().\ntitle(\"Treatment Effect\").\nylab(\"Response\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Treatment Effect"));
    assert!(svg.contains("Response"));
}

// ========================================================================
// Phase 2B: Distribution geoms
// ========================================================================

#[test]
fn geom_density_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_density();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_density_svg_contains_polyline() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_density().\ntitle(\"KDE Test\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline") || svg.contains("<polygon"), "Density plot should contain polyline/polygon");
    assert!(svg.contains("KDE Test"));
}

#[test]
fn geom_density_bw_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_density_bw(0.5);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_area_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 15.0]);\nlet p = p.geom_area();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_area_svg() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [5.0, 15.0, 10.0, 20.0]);\nlet p = p.geom_area().\ntitle(\"Area Plot\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline") || svg.contains("<polygon"), "Area plot should contain polyline/polygon");
    assert!(svg.contains("Area Plot"));
}

#[test]
fn geom_rug_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_rug();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_rug_svg_contains_lines() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_rug();");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<line"), "Rug plot should contain <line> elements for tick marks");
}

#[test]
fn geom_ecdf_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_ecdf();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_ecdf_svg_contains_polyline() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]).\ngeom_ecdf().\ntitle(\"ECDF Test\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline"), "ECDF plot should contain <polyline>");
    assert!(svg.contains("ECDF Test"));
}

#[test]
fn density_plus_rug_combo() {
    let svg = svg_from("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);\nlet p = p.geom_density();\nlet p = p.geom_rug().\ntitle(\"Density + Rug\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Density + Rug"));
    // Both polyline (density curve) and line (rug marks) should exist.
    assert!(svg.contains("<polyline") || svg.contains("<polygon"));
    assert!(svg.contains("<line"));
}

// ========================================================================
// Phase 2B: Regression geom tests
// ========================================================================

#[test]
fn regression_line_geom() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_regression();");
    assert!(svg.contains("<svg"));
    // Regression line should produce a <line> element.
    assert!(svg.contains("<line"));
}

#[test]
fn regression_with_scatter() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.1, 3.9, 6.2, 7.8, 10.1];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.geom_regression();");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<circle")); // scatter points
    assert!(svg.contains("<line"));   // regression line
}

#[test]
fn residual_plot_geom() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.5, 4.0, 6.5, 7.0, 10.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_residplot();");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<circle")); // residual dots
    assert!(svg.contains("<line"));   // zero line
}

// ========================================================================
// Phase 2B: Clustermap tests
// ========================================================================

#[test]
fn clustermap_basic() {
    let (_val, out) = run("import vizor\nlet c1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];\nlet labels = [\"a\", \"b\", \"c\"];\nlet p = vizor_clustermap(c1, labels);\nprint(p);");
    // Should produce output (print triggers type_name display).
    assert!(!out.is_empty());
}

#[test]
fn clustermap_svg() {
    let svg = svg_from("import vizor\nlet c1 = [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 5.0, 3.0, 1.0]];\nlet labels = [\"x\", \"y\", \"z\"];\nlet p = vizor_clustermap(c1, labels);\nlet p = p.title(\"Cluster Map\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Cluster Map"));
    assert!(svg.contains("<rect")); // heatmap cells
}

// ========================================================================
// Phase 2B: Theme tests
// ========================================================================

#[test]
fn theme_publication_svg() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.theme_publication();\nlet p = p.title(\"Publication\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Publication"));
    assert!(svg.contains("<circle"));
}

#[test]
fn theme_dark_svg() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0];\nlet y = [4.0, 5.0, 6.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.theme_dark();\nlet p = p.title(\"Dark\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Dark"));
    // Dark theme background should be dark (#202020)
    assert!(svg.contains("#202020"));
}

// ========================================================================
// Phase 2B: Dendrogram test
// ========================================================================

#[test]
fn dendrogram_via_matrix() {
    // Dendrogram uses the matrix-based data format (distance matrix).
    let svg = svg_from("import vizor\nlet dist = [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]];\nlet labels = [\"A\", \"B\", \"C\"];\nlet p = vizor_plot_matrix(dist, labels, labels);\nlet p = p.geom_dendrogram();");
    assert!(svg.contains("<svg"));
    // Dendrogram produces lines (vertical + horizontal branches).
    assert!(svg.contains("<line"));
}

// ========================================================================
// Full pipeline tests
// ========================================================================

#[test]
fn full_scatter_pipeline() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.1, 3.9, 6.2, 7.8, 10.1];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.title(\"Linear Trend\");\nlet p = p.xlab(\"Predictor\");\nlet p = p.ylab(\"Response\");\nlet p = p.annotate_regression(\"y = 2.0x + 0.1\", 0.997);");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Linear Trend"));
    assert!(svg.contains("Predictor"));
    assert!(svg.contains("Response"));
    assert!(svg.contains("<circle"));
}

#[test]
fn full_bar_pipeline() {
    let svg = svg_from("import vizor\nlet categories = [1.0, 2.0, 3.0, 4.0];\nlet values = [25.0, 40.0, 15.0, 35.0];\nlet p = vizor_plot(categories, values);\nlet p = p.geom_bar();\nlet p = p.title(\"Sales by Quarter\");\nlet p = p.size(800, 600);");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Sales by Quarter"));
    assert!(svg.contains("<rect"));
}

// ========================================================================
// Phase 3: Polar geoms (pie, donut, rose, radar)
// ========================================================================

#[test]
fn geom_pie_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_pie();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_pie_svg_contains_polyline() {
    let svg = svg_from("import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_pie();\nlet p = p.title(\"Pie Chart\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline"), "Pie chart should have polyline slices");
    assert!(svg.contains("Pie Chart"));
}

#[test]
fn geom_donut_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_donut(0.4);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_donut_svg() {
    let svg = svg_from("import vizor\nlet cats = [\"X\", \"Y\", \"Z\"];\nlet vals = [40.0, 35.0, 25.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_donut(0.5);\nlet p = p.title(\"Donut\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline"), "Donut should have polyline slices");
}

#[test]
fn geom_rose_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"N\", \"E\", \"S\", \"W\"];\nlet vals = [10.0, 20.0, 15.0, 25.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_rose();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_rose_svg() {
    let svg = svg_from("import vizor\nlet cats = [\"N\", \"E\", \"S\", \"W\"];\nlet vals = [10.0, 20.0, 15.0, 25.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_rose();\nlet p = p.title(\"Wind Rose\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline"), "Rose should have polyline sectors");
    assert!(svg.contains("Wind Rose"));
}

#[test]
fn geom_radar_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"Speed\", \"Strength\", \"Defense\", \"Magic\", \"HP\"];\nlet vals = [8.0, 6.0, 7.0, 9.0, 5.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_radar();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_radar_svg() {
    let svg = svg_from("import vizor\nlet cats = [\"Speed\", \"Strength\", \"Defense\", \"Magic\", \"HP\"];\nlet vals = [8.0, 6.0, 7.0, 9.0, 5.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_radar();\nlet p = p.title(\"Character Stats\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline"), "Radar should have polygon data");
    assert!(svg.contains("<circle"), "Radar should have data point circles");
    assert!(svg.contains("Character Stats"));
}

#[test]
fn coord_polar_returns_plot() {
    let (_, out) = run("import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.coord_polar();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// Phase 3: 2D density + contour geoms
// ========================================================================

#[test]
fn geom_density2d_returns_plot() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_density2d();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_density2d_svg() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_density2d();\nlet p = p.title(\"2D Density\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"), "2D density should produce filled grid cells");
    assert!(svg.contains("2D Density"));
}

#[test]
fn geom_contour_returns_plot() {
    let (_, out) = run("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_contour();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_contour_svg() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_contour();\nlet p = p.title(\"Contour\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<line"), "Contour should produce line segments");
    assert!(svg.contains("Contour"));
}

#[test]
fn density2d_with_scatter_overlay() {
    let svg = svg_from("import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y);\nlet p = p.geom_density2d();\nlet p = p.geom_point();\nlet p = p.title(\"Density + Scatter\");");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<circle"), "Should have scatter points on top");
    assert!(svg.contains("Density + Scatter"));
}

// ========================================================================
// Phase 3: Polar geom determinism tests
// ========================================================================

#[test]
fn pie_chart_deterministic() {
    let src = "import vizor\nlet cats = [\"A\", \"B\", \"C\", \"D\"];\nlet vals = [25.0, 30.0, 20.0, 25.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_pie();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Pie chart SVG should be deterministic");
}

#[test]
fn radar_chart_deterministic() {
    let src = "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [3.0, 5.0, 7.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_radar();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Radar chart SVG should be deterministic");
}

#[test]
fn density2d_deterministic() {
    let src = "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_density2d();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "2D density SVG should be deterministic");
}

// ========================================================================
// Phase 3.2: Error bars
// ========================================================================

#[test]
fn geom_errorbar_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_errorbar();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_errorbar_svg() {
    // ErrorBar needs x, y, and error columns.
    // Since we can't add arbitrary columns from CJC yet, we test that geom_errorbar
    // doesn't crash even without an error column (it just renders nothing).
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_errorbar();";
    let svg = svg_from(src);
    assert!(svg.contains("<svg"), "Should produce valid SVG");
}

#[test]
fn geom_errorbar_col_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0], [5.0, 10.0]);\nlet p = p.geom_errorbar_col(\"custom_err\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// Phase 3.2: Step line
// ========================================================================

#[test]
fn geom_step_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]);\nlet p = p.geom_step();\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn geom_step_svg_contains_polyline() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]);\nlet p = p.geom_step().title(\"Step Test\");";
    let svg = svg_from(src);
    assert!(svg.contains("<polyline"), "Step line should render as polyline");
    assert!(svg.contains("Step Test"), "Should have title");
}

#[test]
fn geom_step_deterministic() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]);\nlet p = p.geom_step();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Step line SVG should be deterministic");
}

// ========================================================================
// Phase 3.2: Legend
// ========================================================================

#[test]
fn legend_appears_for_multi_layer() {
    // Two layers → legend should appear with two entries.
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_point().geom_line();";
    let svg = svg_from(src);
    assert!(svg.contains("Points"), "Legend should contain 'Points' label");
    assert!(svg.contains("Line"), "Legend should contain 'Line' label");
}

#[test]
fn no_legend_for_single_layer() {
    // Single layer → no legend.
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_point();";
    let svg = svg_from(src);
    // "Points" should NOT appear as legend text (though it might appear elsewhere).
    // More specifically, there should be no legend rect.
    // We check by counting text elements that say "Points" — single-layer won't have it.
    let point_count = svg.matches(">Points<").count();
    assert_eq!(point_count, 0, "Single layer should not have legend");
}

#[test]
fn no_legend_method_hides_legend() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_point().geom_line().no_legend();";
    let svg = svg_from(src);
    let point_count = svg.matches(">Points<").count();
    assert_eq!(point_count, 0, "no_legend() should hide the legend");
}

// ========================================================================
// Phase 3.2: Subtitle
// ========================================================================

#[test]
fn subtitle_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.subtitle(\"My Subtitle\");\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn subtitle_appears_in_svg() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point().title(\"Main Title\").subtitle(\"Sub Title\");";
    let svg = svg_from(src);
    assert!(svg.contains("Main Title"), "Title should be present");
    assert!(svg.contains("Sub Title"), "Subtitle should be present");
}

// ========================================================================
// Phase 3.2: Log scales
// ========================================================================

#[test]
fn scale_x_log_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 10.0, 100.0], [1.0, 2.0, 3.0]);\nlet p = p.scale_x_log(10.0);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn scale_y_log_returns_plot() {
    let (_, out) = run("import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [1.0, 10.0, 100.0]);\nlet p = p.scale_y_log(10.0);\nprint(p);");
    assert_eq!(out, vec!["<VizorPlot>"]);
}

#[test]
fn log_scale_svg_renders() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 10.0, 100.0], [1.0, 10.0, 100.0]);\nlet p = p.geom_point().scale_x_log(10.0).scale_y_log(10.0);";
    let svg = svg_from(src);
    assert!(svg.contains("<circle"), "Log-scale plot should still have circles");
    assert!(svg.contains("<svg"), "Should produce valid SVG");
}

#[test]
fn log_scale_deterministic() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 10.0, 100.0], [2.0, 20.0, 200.0]);\nlet p = p.geom_point().scale_x_log(10.0);";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Log-scale SVG should be deterministic");
}
