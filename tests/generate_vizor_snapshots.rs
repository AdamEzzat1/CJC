// tests/generate_vizor_snapshots.rs
// ──────────────────────────────────────────────────────────────────────
// Snapshot generator: produces reference SVG and BMP files in
// `artifacts/vizor_snapshots/`.  Run with:
//     cargo test --test generate_vizor_snapshots -- --ignored
//
// Each snapshot is generated via the CJC interpreter so the output
// tracks the rendering pipeline exactly.
// ──────────────────────────────────────────────────────────────────────

use cjc_eval::Interpreter;
use cjc_parser::parse_source;

/// Root directory for snapshot artifacts (relative to workspace root).
fn snap_dir() -> std::path::PathBuf {
    let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    ws.join("artifacts").join("vizor_snapshots")
}

/// Run CJC source and return interpreter output lines.
fn run(src: &str) -> Vec<String> {
    let (program, diags) = parse_source(src);
    assert!(
        diags.diagnostics.iter().all(|d| {
            d.severity != cjc_diag::Severity::Error
        }),
        "Parse errors: {:?}",
        diags.diagnostics
    );
    let mut interp = Interpreter::new(42);
    let _val = interp.exec(&program);
    interp.output
}

// ─── Scatter plot ────────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_scatter_svg() {
    let path = snap_dir().join("scatter.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [2.1, 3.9, 6.2, 7.8, 10.1];
let p = vizor_plot_xy(x, y);
let p = p.geom_point();
let p = p.title("Scatter Plot");
let p = p.xlab("X Axis");
let p = p.ylab("Y Axis");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<svg"));
}

// ─── Line plot ───────────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_line_svg() {
    let path = snap_dir().join("line.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0];
let p = vizor_plot_xy(x, y);
let p = p.geom_line();
let p = p.title("Quadratic Curve");
let p = p.xlab("x");
let p = p.ylab("x^2");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains("<polyline"));
}

// ─── Bar chart ───────────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_bar_svg() {
    let path = snap_dir().join("bar.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [12.0, 25.0, 18.0, 30.0, 22.0];
let p = vizor_plot_xy(x, y);
let p = p.geom_bar();
let p = p.title("Bar Chart");
let p = p.xlab("Category");
let p = p.ylab("Count");
let p = p.theme_minimal();
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains("<rect"));
}

// ─── Histogram ───────────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_histogram_svg() {
    let path = snap_dir().join("histogram.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.2, 2.3, 2.8, 3.1, 3.5, 3.7, 4.0, 4.2, 4.8, 5.1, 5.5, 6.0, 6.3, 7.1, 8.0];
let y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
let p = vizor_plot_xy(x, y);
let p = p.geom_histogram();
let p = p.title("Distribution");
let p = p.xlab("Value");
let p = p.ylab("Frequency");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
}

// ─── Annotated scatter ───────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_annotated_svg() {
    let path = snap_dir().join("annotated.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let y = [2.2, 4.1, 5.8, 8.1, 9.9, 12.2];
let p = vizor_plot_xy(x, y);
let p = p.geom_point();
let p = p.annotate_regression("y = 2.0x + 0.1", 0.99);
let p = p.annotate_text("Peak region", 5.0, 10.0);
let p = p.title("Annotated Plot");
let p = p.xlab("X");
let p = p.ylab("Y");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains("Peak region"));
}

// ─── Scatter + line overlay ──────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_scatter_line_svg() {
    let path = snap_dir().join("scatter_line.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [1.5, 3.2, 4.8, 6.5, 8.1];
let p = vizor_plot_xy(x, y);
let p = p.geom_point();
let p = p.geom_line();
let p = p.title("Scatter + Line");
let p = p.xlab("x");
let p = p.ylab("y");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains("<circle"));
    assert!(contents.contains("<polyline"));
}

// ─── Styled wide plot ────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_wide_svg() {
    let path = snap_dir().join("wide.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [10.0, 20.0, 15.0, 25.0, 18.0];
let p = vizor_plot_xy(x, y);
let p = p.geom_bar();
let p = p.size(900.0, 400.0);
let p = p.title("Wide Bar Chart");
let p = p.theme_minimal();
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains("width=\"900\"") || contents.contains("900"));
}

// ─── Coord-flip bar ──────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_flipped_svg() {
    let path = snap_dir().join("flipped.svg");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0];
let y = [8.0, 15.0, 12.0, 20.0];
let p = vizor_plot_xy(x, y);
let p = p.geom_bar();
let p = p.coord_flip();
let p = p.title("Horizontal Bars");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "SVG not written: {}", path.display());
}

// ─── BMP scatter ─────────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_scatter_bmp() {
    let path = snap_dir().join("scatter.bmp");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [2.1, 3.9, 6.2, 7.8, 10.1];
let p = vizor_plot_xy(x, y);
let p = p.geom_point();
let p = p.title("Scatter BMP");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "BMP not written: {}", path.display());
    let data = std::fs::read(&path).unwrap();
    assert_eq!(&data[0..2], b"BM", "Not a valid BMP header");
}

// ─── BMP bar ─────────────────────────────────────────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_bar_bmp() {
    let path = snap_dir().join("bar.bmp");
    let path_str = path.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [10.0, 25.0, 18.0, 30.0, 22.0];
let p = vizor_plot_xy(x, y);
let p = p.geom_bar();
let p = p.title("Bar BMP");
p.save("{}");"#,
        path_str
    );
    run(&src);
    assert!(path.exists(), "BMP not written: {}", path.display());
    let data = std::fs::read(&path).unwrap();
    assert_eq!(&data[0..2], b"BM");
}

// ─── Determinism check: re-generate and compare ─────────────────────

#[test]
#[ignore] // Snapshot generator — run manually: cargo test --test generate_vizor_snapshots -- --ignored
fn snapshot_determinism_scatter_svg() {
    let path1 = snap_dir().join("det_scatter_1.svg");
    let path2 = snap_dir().join("det_scatter_2.svg");
    let make_src = |p: &std::path::Path| {
        let ps = p.to_string_lossy().replace('\\', "/");
        format!(
            r#"import vizor

let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [2.1, 3.9, 6.2, 7.8, 10.1];
let p = vizor_plot_xy(x, y);
let p = p.geom_point();
let p = p.title("Determinism Test");
p.save("{}");"#,
            ps
        )
    };
    run(&make_src(&path1));
    run(&make_src(&path2));
    let a = std::fs::read_to_string(&path1).unwrap();
    let b = std::fs::read_to_string(&path2).unwrap();
    assert_eq!(a, b, "SVG outputs differ — determinism violated");
    // Clean up the second; keep the first as artifact.
    std::fs::remove_file(&path2).ok();
    std::fs::rename(&path1, snap_dir().join("det_scatter.svg")).ok();
}
