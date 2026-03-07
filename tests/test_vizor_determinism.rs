// CJC Test Suite — Vizor determinism
// Ensures that identical inputs always produce byte-identical outputs.
// This is critical for CJC's reproducibility guarantee.

use cjc_eval::Interpreter;
use cjc_parser::parse_source;

/// Save a plot to a temp SVG file and return the contents.
fn eval_to_svg(src: &str, suffix: &str) -> String {
    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join(format!("vizor_det_{}_{}.svg", suffix, std::process::id()));
    let path_str = path.to_string_lossy().replace('\\', "/");
    let full_src = format!("{}\np.save(\"{}\");", src, path_str);

    let (program, diags) = parse_source(&full_src);
    assert!(!diags.has_errors(), "Parse errors: {}", diags.render_all(&full_src, "<test>"));
    let mut interp = Interpreter::new(42);
    interp.exec(&program).expect("eval failed");

    let contents = std::fs::read_to_string(&path).expect("read SVG");
    std::fs::remove_file(&path).ok();
    contents
}

/// Save a plot to a temp BMP file and return the bytes.
fn eval_to_bmp(src: &str, suffix: &str) -> Vec<u8> {
    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join(format!("vizor_det_{}_{}.bmp", suffix, std::process::id()));
    let path_str = path.to_string_lossy().replace('\\', "/");
    let full_src = format!("{}\np.save(\"{}\");", src, path_str);

    let (program, diags) = parse_source(&full_src);
    assert!(!diags.has_errors());
    let mut interp = Interpreter::new(42);
    interp.exec(&program).expect("eval failed");

    let bytes = std::fs::read(&path).expect("read BMP");
    std::fs::remove_file(&path).ok();
    bytes
}

// ========================================================================
// SVG determinism (repeated runs must produce identical output)
// ========================================================================

#[test]
fn svg_deterministic_across_runs() {
    let src = "import vizor\nlet x = [1.0, 2.0, 3.0, 4.0, 5.0];\nlet y = [2.1, 3.9, 6.2, 7.8, 10.1];\nlet p = vizor_plot(x, y);\nlet p = p.geom_point();\nlet p = p.title(\"Determinism Test\");\nlet p = p.xlab(\"X\");\nlet p = p.ylab(\"Y\");";

    let svg1 = eval_to_svg(src, "run1");
    let svg2 = eval_to_svg(src, "run2");
    let svg3 = eval_to_svg(src, "run3");

    assert_eq!(svg1, svg2, "Run 1 vs Run 2 SVG mismatch");
    assert_eq!(svg2, svg3, "Run 2 vs Run 3 SVG mismatch");
}

#[test]
fn svg_deterministic_line_plot() {
    let src = "import vizor\nlet p = vizor_plot([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.25, 1.0, 2.25, 4.0]);\nlet p = p.geom_line();\nlet p = p.title(\"Quadratic\");";

    let a = eval_to_svg(src, "lineA");
    let b = eval_to_svg(src, "lineB");
    assert_eq!(a, b, "Line plot SVG should be deterministic");
}

#[test]
fn svg_deterministic_bar_chart() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0, 4.0], [25.0, 50.0, 75.0, 100.0]);\nlet p = p.geom_bar();\nlet p = p.title(\"Bar Determinism\");";

    let a = eval_to_svg(src, "barA");
    let b = eval_to_svg(src, "barB");
    assert_eq!(a, b, "Bar chart SVG should be deterministic");
}

#[test]
fn svg_deterministic_with_annotations() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [3.0, 6.0, 9.0]);\nlet p = p.geom_point();\nlet p = p.annotate_text(\"peak\", 3.0, 9.0);\nlet p = p.annotate_pvalue(0.042);\nlet p = p.annotate_regression(\"y = 3x\", 1.0);";

    let a = eval_to_svg(src, "annA");
    let b = eval_to_svg(src, "annB");
    assert_eq!(a, b, "Annotated SVG should be deterministic");
}

// ========================================================================
// BMP determinism
// ========================================================================

#[test]
fn bmp_deterministic_across_runs() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);\nlet p = p.geom_point();\nlet p = p.title(\"BMP Determinism\");";

    let bmp1 = eval_to_bmp(src, "bmpA");
    let bmp2 = eval_to_bmp(src, "bmpB");
    assert_eq!(bmp1, bmp2, "BMP bytes should be deterministic");
    assert!(bmp1.len() > 54, "BMP should be > 54 bytes");
}

// ========================================================================
// Float precision determinism
// ========================================================================

#[test]
fn svg_float_precision_is_fixed() {
    let src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [1.333333, 2.666666, 3.999999]);\nlet p = p.geom_point();";

    let svg = eval_to_svg(src, "float");
    assert!(
        !svg.contains("1.333333"),
        "SVG should use truncated float precision, not raw 1.333333"
    );
}

// ========================================================================
// Different seeds still produce same Vizor output
// ========================================================================

#[test]
fn vizor_output_independent_of_seed() {
    let base_src = "import vizor\nlet p = vizor_plot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);\nlet p = p.geom_point();\nlet p = p.title(\"Seed Independence\");";

    let tmpdir = std::env::temp_dir();
    let path1 = tmpdir.join(format!("vizor_seed1_{}.svg", std::process::id()));
    let path2 = tmpdir.join(format!("vizor_seed2_{}.svg", std::process::id()));

    // Seed 1
    let src1 = format!("{}\np.save(\"{}\");", base_src, path1.to_string_lossy().replace('\\', "/"));
    let (prog1, diags) = parse_source(&src1);
    assert!(!diags.has_errors());
    let mut interp1 = Interpreter::new(1);
    interp1.exec(&prog1).expect("seed 1 failed");

    // Seed 9999
    let src2 = format!("{}\np.save(\"{}\");", base_src, path2.to_string_lossy().replace('\\', "/"));
    let (prog2, diags) = parse_source(&src2);
    assert!(!diags.has_errors());
    let mut interp2 = Interpreter::new(9999);
    interp2.exec(&prog2).expect("seed 9999 failed");

    let svg1 = std::fs::read_to_string(&path1).expect("read seed1 SVG");
    let svg2 = std::fs::read_to_string(&path2).expect("read seed2 SVG");

    assert_eq!(svg1, svg2, "Vizor output should not depend on interpreter seed");

    std::fs::remove_file(&path1).ok();
    std::fs::remove_file(&path2).ok();
}
