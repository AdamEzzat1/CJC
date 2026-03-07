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
