//! Builtin documentation metadata for IDE/LSP integration.
//!
//! Static documentation strings for each Vizor builtin,
//! consumable by cjc-analyzer for hover info and completion.

/// A documentation entry for a Vizor builtin or method.
#[derive(Debug, Clone)]
pub struct DocEntry {
    pub name: &'static str,
    pub signature: &'static str,
    pub description: &'static str,
    pub kind: DocKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind {
    Function,
    Method,
}

/// All Vizor builtin documentation entries.
pub fn vizor_docs() -> Vec<DocEntry> {
    vec![
        // ── Free functions ──
        DocEntry {
            name: "vizor_plot",
            signature: "vizor_plot(x: Array<f64>, y: Array<f64>) -> VizorPlot",
            description: "Create a new plot specification from x and y data arrays.",
            kind: DocKind::Function,
        },
        DocEntry {
            name: "vizor_plot_xy",
            signature: "vizor_plot_xy(x: Array<f64>, y: Array<f64>) -> VizorPlot",
            description: "Alias for vizor_plot. Create a plot from x and y arrays.",
            kind: DocKind::Function,
        },
        DocEntry {
            name: "vizor_plot_cat",
            signature: "vizor_plot_cat(categories: Array<String>, values: Array<f64>) -> VizorPlot",
            description: "Create a plot with categorical (string) x-axis and numeric y-axis.",
            kind: DocKind::Function,
        },
        DocEntry {
            name: "vizor_plot_matrix",
            signature: "vizor_plot_matrix(matrix: Array<Array<f64>>, row_labels: Array<String>, col_labels: Array<String>) -> VizorPlot",
            description: "Create a plot from a 2D matrix with row and column labels (for heatmaps).",
            kind: DocKind::Function,
        },
        DocEntry {
            name: "vizor_corr_matrix",
            signature: "vizor_corr_matrix(columns: Array<Array<f64>>, labels: Array<String>) -> VizorPlot",
            description: "Compute a correlation matrix from data columns and create a heatmap plot.",
            kind: DocKind::Function,
        },
        DocEntry {
            name: "vizor_clustermap",
            signature: "vizor_clustermap(columns: Array<Array<f64>>, labels: Array<String>) -> VizorPlot",
            description: "Compute a clustered heatmap from data columns with hierarchical clustering reordering.",
            kind: DocKind::Function,
        },
        // ── Methods on VizorPlot ──
        DocEntry {
            name: "geom_point",
            signature: ".geom_point() -> VizorPlot",
            description: "Add a point (scatter) geometry layer to the plot.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_line",
            signature: ".geom_line() -> VizorPlot",
            description: "Add a line geometry layer connecting data points in order.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_bar",
            signature: ".geom_bar() -> VizorPlot",
            description: "Add a bar geometry layer.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_histogram",
            signature: ".geom_histogram(bins: i64) -> VizorPlot",
            description: "Add a histogram geometry with the specified number of bins.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_density",
            signature: ".geom_density() -> VizorPlot",
            description: "Add a density (Gaussian KDE) geometry with Silverman bandwidth.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_density_bw",
            signature: ".geom_density_bw(bandwidth: f64) -> VizorPlot",
            description: "Add a density geometry with explicit bandwidth.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_area",
            signature: ".geom_area() -> VizorPlot",
            description: "Add a filled area geometry under the data curve.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_rug",
            signature: ".geom_rug() -> VizorPlot",
            description: "Add rug marks (tick marks along axis edge) for each data point.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_ecdf",
            signature: ".geom_ecdf() -> VizorPlot",
            description: "Add an empirical CDF step-function geometry.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_box",
            signature: ".geom_box() -> VizorPlot",
            description: "Add a box-and-whisker plot (Tukey fences, 1.5×IQR whiskers, outlier dots).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_violin",
            signature: ".geom_violin() -> VizorPlot",
            description: "Add a violin plot (mirrored KDE for each category).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_strip",
            signature: ".geom_strip() -> VizorPlot",
            description: "Add a strip plot (jittered points for each category).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_swarm",
            signature: ".geom_swarm() -> VizorPlot",
            description: "Add a swarm plot (packed non-overlapping points for each category).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_boxen",
            signature: ".geom_boxen() -> VizorPlot",
            description: "Add a boxen (letter-value) plot with nested quantile boxes.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_regression",
            signature: ".geom_regression() -> VizorPlot",
            description: "Add a fitted regression line across the data range.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_residplot",
            signature: ".geom_residplot() -> VizorPlot",
            description: "Add a residual plot (points at (x, residual) from linear fit).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_dendrogram",
            signature: ".geom_dendrogram() -> VizorPlot",
            description: "Add a dendrogram (hierarchical clustering tree) geometry.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "title",
            signature: ".title(text: String) -> VizorPlot",
            description: "Set the plot title.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "xlab",
            signature: ".xlab(text: String) -> VizorPlot",
            description: "Set the x-axis label.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "ylab",
            signature: ".ylab(text: String) -> VizorPlot",
            description: "Set the y-axis label.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "xlim",
            signature: ".xlim(min: f64, max: f64) -> VizorPlot",
            description: "Set x-axis limits.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "ylim",
            signature: ".ylim(min: f64, max: f64) -> VizorPlot",
            description: "Set y-axis limits.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "theme_minimal",
            signature: ".theme_minimal() -> VizorPlot",
            description: "Apply the minimal theme (lighter grid, more whitespace).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "theme_publication",
            signature: ".theme_publication() -> VizorPlot",
            description: "Apply a publication-ready theme with clean lines and no gridlines.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "theme_dark",
            signature: ".theme_dark() -> VizorPlot",
            description: "Apply a dark theme with light text on dark background.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "coord_flip",
            signature: ".coord_flip() -> VizorPlot",
            description: "Flip x and y coordinates.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "size",
            signature: ".size(width: i64, height: i64) -> VizorPlot",
            description: "Set the plot dimensions in pixels.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "to_svg",
            signature: ".to_svg() -> String",
            description: "Render the plot to an SVG string.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "to_bmp",
            signature: ".to_bmp() -> Bytes",
            description: "Render the plot to a BMP byte buffer.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "save",
            signature: ".save(path: String) -> Void",
            description: "Save the plot to a file. Format is inferred from extension (.svg or .bmp).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_text",
            signature: ".annotate_text(text: String, x: f64, y: f64) -> VizorPlot",
            description: "Add a free-form text annotation at data coordinates.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_regression",
            signature: ".annotate_regression(equation: String, r_squared: f64) -> VizorPlot",
            description: "Add a regression summary annotation (equation + R²).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_ci",
            signature: ".annotate_ci(level: f64, lower: f64, upper: f64) -> VizorPlot",
            description: "Add a confidence interval annotation.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_pvalue",
            signature: ".annotate_pvalue(value: f64) -> VizorPlot",
            description: "Add a p-value annotation.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_event",
            signature: ".annotate_event(x: f64, label: String) -> VizorPlot",
            description: "Add a vertical event marker at an x-value.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_note",
            signature: ".annotate_note(text: String) -> VizorPlot",
            description: "Add a small note annotation.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_data_note",
            signature: ".annotate_data_note(text: String) -> VizorPlot",
            description: "Add a data provenance / source note.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "annotate_inline_label",
            signature: ".annotate_inline_label(text: String, x: f64, y: f64) -> VizorPlot",
            description: "Add an inline label near a data point.",
            kind: DocKind::Method,
        },
        // ── Phase 3: Polar geoms ──
        DocEntry {
            name: "geom_pie",
            signature: ".geom_pie() -> VizorPlot",
            description: "Add a pie chart geometry (requires categorical x + numeric y data).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_donut",
            signature: ".geom_donut(inner_radius: f64) -> VizorPlot",
            description: "Add a donut chart (pie with inner hole). inner_radius 0.0–1.0.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_rose",
            signature: ".geom_rose() -> VizorPlot",
            description: "Add a rose (Nightingale) chart geometry with bars in polar coordinates.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_radar",
            signature: ".geom_radar() -> VizorPlot",
            description: "Add a radar (spider) chart geometry with polygon on radial axes.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "coord_polar",
            signature: ".coord_polar() -> VizorPlot",
            description: "Switch to polar coordinate system (x → angle, y → radius).",
            kind: DocKind::Method,
        },
        // ── Phase 3: 2D density + contour ──
        DocEntry {
            name: "geom_density2d",
            signature: ".geom_density2d() -> VizorPlot",
            description: "Add a 2D kernel density estimation (filled contour bands).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_contour",
            signature: ".geom_contour() -> VizorPlot",
            description: "Add contour lines at density levels.",
            kind: DocKind::Method,
        },
        // ── Phase 3.2: Error bars, step line, legend, scales ──
        DocEntry {
            name: "geom_errorbar",
            signature: ".geom_errorbar() -> VizorPlot",
            description: "Add error bars (vertical bars at each data point, requires 'error' column).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_errorbar_col",
            signature: ".geom_errorbar_col(column: String) -> VizorPlot",
            description: "Add error bars using a custom error column name.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "geom_step",
            signature: ".geom_step() -> VizorPlot",
            description: "Add a step line (staircase) geometry.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "no_legend",
            signature: ".no_legend() -> VizorPlot",
            description: "Disable the legend for this plot.",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "subtitle",
            signature: ".subtitle(text: String) -> VizorPlot",
            description: "Set the plot subtitle (displayed below the title).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "scale_x_log",
            signature: ".scale_x_log(base: f64) -> VizorPlot",
            description: "Set x-axis to logarithmic scale with the given base (default 10).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "scale_y_log",
            signature: ".scale_y_log(base: f64) -> VizorPlot",
            description: "Set y-axis to logarithmic scale with the given base (default 10).",
            kind: DocKind::Method,
        },
        DocEntry {
            name: "add_column",
            signature: ".add_column(name: String, values: Array<f64>) -> VizorPlot",
            description: "Add or replace a named data column (e.g., error bars).",
            kind: DocKind::Method,
        },
    ]
}

/// List of all Vizor builtin function names (for import gating).
pub const VIZOR_BUILTIN_NAMES: &[&str] = &[
    "vizor_plot",
    "vizor_plot_xy",
    "vizor_plot_cat",
    "vizor_plot_matrix",
    "vizor_corr_matrix",
    "vizor_displot",
    "vizor_catplot",
    "vizor_relplot",
    "vizor_lmplot",
    "vizor_jointplot",
    "vizor_pairplot",
    "vizor_clustermap",
];

/// List of all Vizor method names.
pub const VIZOR_METHOD_NAMES: &[&str] = &[
    "geom_point", "geom_line", "geom_bar", "geom_histogram",
    "geom_density", "geom_density_bw", "geom_area", "geom_rug", "geom_ecdf",
    "geom_box", "geom_violin", "geom_strip", "geom_swarm", "geom_boxen",
    "geom_tile", "geom_regression", "geom_residplot", "geom_dendrogram",
    // Phase 3: Polar + 2D density
    "geom_pie", "geom_donut", "geom_rose", "geom_radar",
    "coord_polar",
    "geom_density2d", "geom_contour",
    // Phase 3.2: Error bars, step, legend, scales
    "geom_errorbar", "geom_errorbar_col", "geom_step",
    "no_legend", "subtitle",
    "scale_x_log", "scale_y_log",
    "scale_color_diverging", "show_values",
    "add_column",
    "facet_wrap", "facet_wrap_ncol", "facet_grid",
    "title", "xlab", "ylab", "xlim", "ylim",
    "theme_minimal", "theme_publication", "theme_dark",
    "coord_flip", "size",
    "to_svg", "to_bmp", "to_png", "save",
    "annotate_text", "annotate_regression", "annotate_ci",
    "annotate_pvalue", "annotate_event", "annotate_note",
    "annotate_data_note", "annotate_inline_label",
];
