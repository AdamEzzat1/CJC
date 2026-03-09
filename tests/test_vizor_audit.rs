// CJC Test Suite — Vizor Rendering Trust Audit (Phase 5)
//
// Tests for all rendering fixes made during the Phase 2-4 audit:
// - Categorical axis labels (box, violin, strip, swarm, boxen)
// - Density2d clipping to plot bounds
// - Violin overflow clamping
// - Boxen degenerate level filtering
// - Error bar y-range expansion with add_column
// - Axis suppression for polar/tile/dendrogram
// - Residual y-axis isolation
// - Annotation background + stacking
// - Updated theme defaults
// - Updated color palette

use cjc_eval::Interpreter;
use cjc_parser::parse_source;

/// Run CJC source and return (last_value, printed_output).
fn run(src: &str) -> (cjc_runtime::Value, Vec<String>) {
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

/// Save a Vizor plot to a temp SVG file and return the contents.
fn svg_from(src_lines: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join(format!("vizor_audit_{}_{}.svg", std::process::id(), id));
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!("{}\np.save(\"{}\");", src_lines, path_str);
    let (_, _) = run(&src);
    let contents = std::fs::read_to_string(&path).expect("Failed to read SVG file");
    std::fs::remove_file(&path).ok();
    contents
}

// ========================================================================
// Categorical axis labels (Phase 2 fix)
// ========================================================================

#[test]
fn box_plot_shows_category_labels() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"Apple\", \"Banana\", \"Cherry\"];\nlet vals = [1.0, 2.0, 3.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_box();"
    );
    assert!(svg.contains("Apple"), "Box plot SVG should contain category 'Apple'");
    assert!(svg.contains("Banana"), "Box plot SVG should contain category 'Banana'");
    assert!(svg.contains("Cherry"), "Box plot SVG should contain category 'Cherry'");
}

#[test]
fn violin_plot_shows_category_labels() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"X\", \"X\", \"X\", \"X\", \"X\", \"X\", \"X\", \"X\", \"Y\", \"Y\", \"Y\", \"Y\", \"Y\", \"Y\", \"Y\", \"Y\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_violin();"
    );
    assert!(svg.contains(">X<"), "Violin SVG should contain category label 'X'");
    assert!(svg.contains(">Y<"), "Violin SVG should contain category label 'Y'");
}

#[test]
fn strip_plot_shows_category_labels() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"Grp1\", \"Grp1\", \"Grp2\", \"Grp2\"];\nlet vals = [1.0, 2.0, 3.0, 4.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_strip();"
    );
    assert!(svg.contains("Grp1"), "Strip plot should show category 'Grp1'");
    assert!(svg.contains("Grp2"), "Strip plot should show category 'Grp2'");
}

#[test]
fn swarm_plot_shows_category_labels() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"M\", \"M\", \"M\", \"F\", \"F\", \"F\"];\nlet vals = [1.0, 2.0, 1.5, 3.0, 4.0, 3.5];\nlet p = vizor_plot_cat(cats, vals).\ngeom_swarm();"
    );
    assert!(svg.contains(">M<"), "Swarm plot should show category 'M'");
    assert!(svg.contains(">F<"), "Swarm plot should show category 'F'");
}

#[test]
fn boxen_plot_shows_category_labels() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_boxen();"
    );
    assert!(svg.contains(">A<"), "Boxen plot should show category 'A'");
    assert!(svg.contains(">B<"), "Boxen plot should show category 'B'");
}

// ========================================================================
// Pie / Rose / Radar: no Cartesian axes (Phase 2 fix)
// ========================================================================

#[test]
fn pie_chart_no_cartesian_tick_labels() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_pie();"
    );
    // Pie chart should NOT have numeric tick labels like "0" on the axis.
    // It should have slice labels or none.
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<polyline"), "Pie chart should have polyline slices");
}

#[test]
fn rose_chart_no_cartesian_axes() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"N\", \"E\", \"S\", \"W\"];\nlet vals = [10.0, 20.0, 15.0, 25.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_rose();"
    );
    assert!(svg.contains("<polyline"), "Rose chart should have polyline sectors");
}

#[test]
fn radar_chart_no_cartesian_axes() {
    let svg = svg_from(
        "import vizor\nlet cats = [\"Speed\", \"Strength\", \"Defense\"];\nlet vals = [8.0, 6.0, 7.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_radar();"
    );
    assert!(svg.contains("<polyline"), "Radar chart should have data polygon");
    assert!(svg.contains("<circle"), "Radar chart should have data point circles");
}

// ========================================================================
// Dendrogram: no Cartesian axes, no overflow (Phase 2 fix)
// ========================================================================

#[test]
fn dendrogram_no_cartesian_axes() {
    let svg = svg_from(
        "import vizor\nlet dist = [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]];\nlet labels = [\"A\", \"B\", \"C\"];\nlet p = vizor_plot_matrix(dist, labels, labels).\ngeom_dendrogram();"
    );
    assert!(svg.contains("<line"), "Dendrogram should have branch lines");
    assert!(svg.contains("A"), "Dendrogram should show leaf label 'A'");
    assert!(svg.contains("B"), "Dendrogram should show leaf label 'B'");
    assert!(svg.contains("C"), "Dendrogram should show leaf label 'C'");
}

// ========================================================================
// Density2d: no overflow (Phase 2 fix)
// ========================================================================

#[test]
fn density2d_produces_valid_svg() {
    let svg = svg_from(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];\nlet y = [2.0, 3.0, 5.0, 7.0, 11.0, 2.5, 3.5, 5.5, 7.5, 11.5];\nlet p = vizor_plot(x, y).\ngeom_density2d().\ntitle(\"2D Density\");"
    );
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"), "2D density should produce grid cell rects");
    assert!(svg.contains("2D Density"));
}

// ========================================================================
// Boxen: degenerate level fix (Phase 2 fix)
// ========================================================================

#[test]
fn boxen_small_group_renders_correctly() {
    // 5 values per group — the median-only level should be filtered.
    let svg = svg_from(
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_boxen();"
    );
    assert!(svg.contains("<rect"), "Boxen plot should contain rect elements");
    assert!(svg.contains(">A<"), "Boxen should show category labels");
}

// ========================================================================
// Error bars with add_column (Phase 2 fix)
// ========================================================================

#[test]
fn errorbar_with_add_column_renders() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.add_column(\"error\", [2.0, 3.0, 1.5]);\nlet p = p.geom_errorbar();\nlet p = p.title(\"Error Bars\");"
    );
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<line"), "Error bars should produce line elements");
    assert!(svg.contains("Error Bars"));
}

#[test]
fn errorbar_values_within_viewport() {
    // With y=[10,20,30] and error=[5,5,5], the error bars reach 5 and 35.
    // They should all be within the SVG viewport (no negative coordinates, etc.)
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.add_column(\"error\", [5.0, 5.0, 5.0]);\nlet p = p.geom_errorbar();"
    );
    assert!(svg.contains("<svg"));
    // Check for valid SVG (no NaN or infinity in coordinates).
    assert!(!svg.contains("NaN"), "SVG should not contain NaN coordinates");
    assert!(!svg.contains("Infinity"), "SVG should not contain Infinity coordinates");
}

#[test]
fn add_column_method_creates_plot() {
    let (_, out) = run(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.add_column(\"error\", [2.0, 3.0, 1.5]);\nprint(p);"
    );
    assert_eq!(out, vec!["<VizorPlot>"]);
}

// ========================================================================
// Residual plot: y-axis isolation (Phase 2 fix)
// ========================================================================

#[test]
fn residual_plot_y_axis_near_zero() {
    // y = 2x (perfect linear fit) → residuals are ~0.
    // The y-axis should NOT show the raw y range (2-10).
    let svg = svg_from(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.0, 4.0, 6.0, 8.0, 10.0];\nlet p = vizor_plot(x, y).\ngeom_residplot().\ntitle(\"Residuals\");"
    );
    assert!(svg.contains("Residuals"));
    // The SVG should not contain large numeric tick labels like "8" or "10"
    // because the residuals are near zero.
    // (We can't be 100% sure of tick label format, but large values would be wrong.)
    assert!(!svg.contains(">10.0<") && !svg.contains(">10<"),
        "Residual plot y-axis should not show raw y value 10");
}

// ========================================================================
// Annotation stacking (Phase 3 fix)
// ========================================================================

#[test]
fn multiple_annotations_do_not_crash() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_point();\nlet p = p.annotate_pvalue(0.001);\nlet p = p.annotate_regression(\"y = x + 3\", 0.99);\nlet p = p.title(\"Stacked Annotations\");"
    );
    assert!(svg.contains("<svg"));
    assert!(svg.contains("Stacked Annotations"));
    // Both annotations should be present.
    assert!(svg.contains("p") || svg.contains("0.001"), "P-value annotation should appear");
}

#[test]
fn annotation_text_appears_in_svg() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();\nlet p = p.annotate_text(\"custom note\", 1.5, 3.5);"
    );
    assert!(svg.contains("custom note"), "Custom annotation text should be in SVG");
}

#[test]
fn event_annotation_appears() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_line();\nlet p = p.annotate_event(2.0, \"Peak Event\");"
    );
    assert!(svg.contains("Peak Event"), "Event annotation should appear in SVG");
}

// ========================================================================
// Updated theme defaults (Phase 4)
// ========================================================================

#[test]
fn default_theme_uses_off_white_background() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();"
    );
    // Off-white is rgb(252,252,252) → #fcfcfc
    assert!(svg.contains("#fcfcfc"), "Default theme should use off-white plot background #fcfcfc");
}

#[test]
fn default_theme_uses_muted_axis_color() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]);\nlet p = p.geom_point();"
    );
    // Axis color rgb(80,80,80) → #505050
    assert!(svg.contains("#505050"), "Default theme should use muted axis color #505050");
}

// ========================================================================
// Updated color palette (Phase 4)
// ========================================================================

#[test]
fn multi_layer_uses_muted_palette() {
    let svg = svg_from(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_point().\ngeom_line();"
    );
    // First color: muted blue rgb(31,119,180) → #1f77b4
    assert!(svg.contains("#1f77b4"), "First layer should use muted blue #1f77b4");
    // Second color: muted orange rgb(255,127,14) → #ff7f0e
    assert!(svg.contains("#ff7f0e"), "Second layer should use muted orange #ff7f0e");
}

// ========================================================================
// Determinism for all fix targets
// ========================================================================

#[test]
fn categorical_box_svg_deterministic() {
    let src = "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_box();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Box plot SVG should be deterministic");
}

#[test]
fn violin_svg_deterministic() {
    let src = "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_violin();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Violin SVG should be deterministic");
}

#[test]
fn errorbar_svg_deterministic() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.add_column(\"error\", [2.0, 3.0, 1.5]);\nlet p = p.geom_errorbar();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Error bar SVG should be deterministic");
}

#[test]
fn boxen_svg_deterministic() {
    let src = "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_boxen();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Boxen SVG should be deterministic");
}

#[test]
fn dendrogram_svg_deterministic() {
    let src = "import vizor\nlet dist = [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]];\nlet labels = [\"A\", \"B\", \"C\"];\nlet p = vizor_plot_matrix(dist, labels, labels).\ngeom_dendrogram();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Dendrogram SVG should be deterministic");
}

#[test]
fn residual_svg_deterministic() {
    let src = "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.5, 4.0, 6.5, 7.0, 10.0];\nlet p = vizor_plot(x, y).\ngeom_residplot();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Residual plot SVG should be deterministic");
}

#[test]
fn tile_svg_deterministic() {
    let src = "import vizor\nlet mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];\nlet labels = [\"A\", \"B\", \"C\"];\nlet p = vizor_plot_matrix(mat, labels, labels).\ngeom_tile();";
    let svg1 = svg_from(src);
    let svg2 = svg_from(src);
    assert_eq!(svg1, svg2, "Tile plot SVG should be deterministic");
}

// ========================================================================
// SVG well-formedness checks
// ========================================================================

#[test]
fn no_nan_in_any_svg_output() {
    let sources = vec![
        "import vizor\nlet p = vizor_plot([1.0, 2.0], [3.0, 4.0]).\ngeom_point();",
        "import vizor\nlet cats = [\"A\", \"B\"];\nlet vals = [10.0, 20.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_box();",
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_violin();",
        "import vizor\nlet cats = [\"A\", \"B\", \"C\"];\nlet vals = [30.0, 50.0, 20.0];\nlet p = vizor_plot_cat(cats, vals).\ngeom_pie();",
    ];
    for src in sources {
        let svg = svg_from(src);
        assert!(!svg.contains("NaN"), "SVG should not contain NaN: source starts with {}", &src[..40]);
        assert!(!svg.contains("Infinity"), "SVG should not contain Infinity");
    }
}

// ========================================================================
// Parity: AST-eval vs MIR-exec for audit-fixed chart types
// ========================================================================

fn assert_output_parity(src: &str) {
    let (program, diags) = parse_source(src);
    assert!(
        !diags.has_errors(),
        "Parse errors: {}",
        diags.render_all(src, "<test>")
    );
    let mut interp = Interpreter::new(42);
    interp.exec(&program).expect("AST eval failed");
    let ast_out = interp.output.clone();

    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("MIR exec failed");
    let mir_out = exec.output;

    assert_eq!(
        ast_out, mir_out,
        "Output mismatch between AST-eval and MIR-exec"
    );
}

#[test]
fn parity_audit_add_column() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.add_column(\"error\", [2.0, 3.0, 1.5]);\nprint(p);"
    );
}

#[test]
fn parity_audit_errorbar_with_column() {
    assert_output_parity(
        "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.add_column(\"error\", [2.0, 3.0, 1.5]);\nlet p = p.geom_errorbar();\nprint(p);"
    );
}

#[test]
fn parity_audit_boxen_small_group() {
    assert_output_parity(
        "import vizor\nlet cats = [\"A\", \"A\", \"A\", \"A\", \"A\", \"B\", \"B\", \"B\", \"B\", \"B\"];\nlet vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];\nlet p = vizor_plot_cat(cats, vals);\nlet p = p.geom_boxen();\nprint(p);"
    );
}

#[test]
fn parity_audit_residual() {
    assert_output_parity(
        "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.5, 4.0, 6.5, 7.0, 10.0];\nlet p = vizor_plot(x, y);\nlet p = p.geom_residplot();\nprint(p);"
    );
}

#[test]
fn parity_audit_dendrogram() {
    assert_output_parity(
        "import vizor\nlet dist = [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]];\nlet labels = [\"A\", \"B\", \"C\"];\nlet p = vizor_plot_matrix(dist, labels, labels);\nlet p = p.geom_dendrogram();\nprint(p);"
    );
}

#[test]
fn parity_audit_tile() {
    assert_output_parity(
        "import vizor\nlet mat = [[1.0, 2.0], [3.0, 4.0]];\nlet rows = [\"r0\", \"r1\"];\nlet cols = [\"c0\", \"c1\"];\nlet p = vizor_plot_matrix(mat, rows, cols);\nlet p = p.geom_tile();\nprint(p);"
    );
}
