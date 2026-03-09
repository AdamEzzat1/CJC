//! Plot specification types — the declarative description of a plot.
//!
//! A `PlotSpec` is immutable once constructed. Every builder method
//! returns a new `PlotSpec` with the modification applied.

use crate::annotation::Annotation;
use crate::color::Color;
use crate::theme::Theme;

/// The top-level plot specification.
#[derive(Debug, Clone)]
pub struct PlotSpec {
    pub data: PlotData,
    pub layers: Vec<Layer>,
    pub scales: ScaleSet,
    pub labels: Labels,
    pub theme: Theme,
    pub coord: CoordSystem,
    pub annotations: Vec<Annotation>,
    pub facet: crate::facet::FacetSpec,
    pub width: u32,
    pub height: u32,
    /// Whether to show the legend (auto-enabled for 2+ layers).
    pub show_legend: bool,
}

impl PlotSpec {
    /// Create a new empty plot spec with given data.
    pub fn new(data: PlotData) -> Self {
        PlotSpec {
            data,
            layers: Vec::new(),
            scales: ScaleSet::default(),
            labels: Labels::default(),
            theme: Theme::default(),
            coord: CoordSystem::Cartesian,
            annotations: Vec::new(),
            facet: crate::facet::FacetSpec::None,
            width: 800,
            height: 600,
            show_legend: true,
        }
    }

    /// Create a plot from x and y float arrays.
    pub fn from_xy(x: Vec<f64>, y: Vec<f64>) -> Self {
        let data = PlotData {
            columns: vec![
                ("x".to_string(), DataColumn::Float(x)),
                ("y".to_string(), DataColumn::Float(y)),
            ],
        };
        PlotSpec::new(data)
    }

    /// Create a plot from categorical (string) x-axis and float y-axis.
    pub fn from_cat(categories: Vec<String>, values: Vec<f64>) -> Self {
        let data = PlotData {
            columns: vec![
                ("x".to_string(), DataColumn::Str(categories)),
                ("y".to_string(), DataColumn::Float(values)),
            ],
        };
        PlotSpec::new(data)
    }

    /// Create a plot from arbitrary named columns.
    pub fn from_columns(columns: Vec<(String, DataColumn)>) -> Self {
        PlotSpec::new(PlotData { columns })
    }

    /// Create a plot from a matrix (for heatmaps).
    ///
    /// The matrix is stored as a flat Float column `"__values"`, with
    /// metadata columns for dimensions and labels.
    pub fn from_matrix(
        matrix: Vec<Vec<f64>>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
    ) -> Self {
        let nrows = matrix.len();
        let ncols = if nrows > 0 { matrix[0].len() } else { 0 };
        let flat: Vec<f64> = matrix.into_iter().flat_map(|row| row.into_iter()).collect();
        let data = PlotData {
            columns: vec![
                ("__values".to_string(), DataColumn::Float(flat)),
                ("__nrows".to_string(), DataColumn::Int(vec![nrows as i64])),
                ("__ncols".to_string(), DataColumn::Int(vec![ncols as i64])),
                ("__row_labels".to_string(), DataColumn::Str(row_labels)),
                ("__col_labels".to_string(), DataColumn::Str(col_labels)),
            ],
        };
        PlotSpec::new(data)
    }

    /// Add a layer.
    pub fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Add a point geometry with default aesthetics.
    pub fn geom_point(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Point,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a line geometry with default aesthetics.
    pub fn geom_line(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Line,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a bar geometry with default aesthetics.
    pub fn geom_bar(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Bar,
            aes: AesMapping::xy(),
            params: GeomParams {
                bar_width: Some(0.8),
                ..GeomParams::default()
            },
        })
    }

    /// Add a histogram geometry.
    pub fn geom_histogram(self, bins: usize) -> Self {
        self.add_layer(Layer {
            geom: Geom::Histogram,
            aes: AesMapping {
                x: Some("x".to_string()),
                ..AesMapping::default()
            },
            params: GeomParams {
                bins: Some(bins),
                bar_width: Some(1.0),
                ..GeomParams::default()
            },
        })
    }

    // ── Phase 2B: Distribution geoms ──────────────────────────────────

    /// Add a density (KDE) geometry with Silverman bandwidth.
    pub fn geom_density(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Density,
            aes: AesMapping {
                x: Some("x".to_string()),
                ..AesMapping::default()
            },
            params: GeomParams::default(),
        })
    }

    /// Add a density geometry with explicit bandwidth.
    pub fn geom_density_bw(self, bw: f64) -> Self {
        self.add_layer(Layer {
            geom: Geom::Density,
            aes: AesMapping {
                x: Some("x".to_string()),
                ..AesMapping::default()
            },
            params: GeomParams {
                bandwidth: Some(bw),
                ..GeomParams::default()
            },
        })
    }

    /// Add a filled area geometry.
    pub fn geom_area(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Area,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add rug marks along the bottom edge.
    pub fn geom_rug(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Rug,
            aes: AesMapping {
                x: Some("x".to_string()),
                ..AesMapping::default()
            },
            params: GeomParams::default(),
        })
    }

    /// Add rug marks along a specified side.
    pub fn geom_rug_side(self, side: RugSide) -> Self {
        self.add_layer(Layer {
            geom: Geom::Rug,
            aes: AesMapping {
                x: Some("x".to_string()),
                ..AesMapping::default()
            },
            params: GeomParams {
                rug_side: side,
                ..GeomParams::default()
            },
        })
    }

    /// Add an empirical CDF geometry.
    pub fn geom_ecdf(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Ecdf,
            aes: AesMapping {
                x: Some("x".to_string()),
                ..AesMapping::default()
            },
            params: GeomParams::default(),
        })
    }

    // ── Phase 2B: Categorical geoms ────────────────────────────────────

    /// Add a box plot geometry (Tukey box-and-whisker).
    pub fn geom_box(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Box,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a violin plot geometry (mirrored KDE).
    pub fn geom_violin(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Violin,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a strip (jittered point) geometry.
    pub fn geom_strip(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Strip,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a swarm (packed point) geometry.
    pub fn geom_swarm(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Swarm,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a boxen (letter-value) geometry.
    pub fn geom_boxen(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Boxen,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    // ── Phase 2B: Regression geoms ─────────────────────────────────

    /// Add a regression line fitted to the x/y data.
    pub fn geom_regression(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::RegressionLine,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a residual plot (points at (x, residual)).
    pub fn geom_residplot(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Residual,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    // ── Phase 2B: Dendrogram ─────────────────────────────────────────

    /// Add a dendrogram (hierarchical clustering tree) geometry.
    pub fn geom_dendrogram(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Dendrogram,
            aes: AesMapping::default(),
            params: GeomParams::default(),
        })
    }

    // ── Phase 2B: Heatmap / Tile geoms ──────────────────────────────

    /// Add a tile (heatmap) geometry.
    pub fn geom_tile(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Tile,
            aes: AesMapping::default(),
            params: GeomParams::default(),
        })
    }

    /// Set the color scale to a two-color gradient.
    pub fn scale_color_gradient(mut self, low: Color, high: Color) -> Self {
        self.scales.color = ColorScale::Gradient { low, high };
        self
    }

    /// Set the color scale to a diverging palette (blue → white → red).
    pub fn scale_color_diverging(mut self) -> Self {
        self.scales.color = ColorScale::Diverging;
        self
    }

    /// Enable or disable showing cell values on heatmap tiles.
    pub fn show_values(mut self, show: bool) -> Self {
        // Update the most recently added Tile layer's params
        if let Some(layer) = self.layers.last_mut() {
            if layer.geom == Geom::Tile {
                layer.params.show_cell_values = show;
            }
        }
        self
    }

    // ── Phase 2B: Faceting ────────────────────────────────────────────

    /// Facet by a column, wrapping panels into rows (default 2 columns).
    pub fn facet_wrap(mut self, column: &str) -> Self {
        self.facet = crate::facet::FacetSpec::Wrap {
            column: column.to_string(),
            ncol: 2,
        };
        self
    }

    /// Facet by a column with explicit number of columns.
    pub fn facet_wrap_ncol(mut self, column: &str, ncol: usize) -> Self {
        self.facet = crate::facet::FacetSpec::Wrap {
            column: column.to_string(),
            ncol,
        };
        self
    }

    /// Facet into a row × column grid.
    pub fn facet_grid(mut self, row: &str, col: &str) -> Self {
        self.facet = crate::facet::FacetSpec::Grid {
            row: row.to_string(),
            col: col.to_string(),
        };
        self
    }

    /// Set the plot title.
    pub fn title(mut self, title: &str) -> Self {
        self.labels.title = Some(title.to_string());
        self
    }

    /// Set the plot subtitle.
    pub fn subtitle(mut self, sub: &str) -> Self {
        self.labels.subtitle = Some(sub.to_string());
        self
    }

    /// Set the x-axis label.
    pub fn xlab(mut self, label: &str) -> Self {
        self.labels.x = Some(label.to_string());
        self
    }

    /// Set the y-axis label.
    pub fn ylab(mut self, label: &str) -> Self {
        self.labels.y = Some(label.to_string());
        self
    }

    /// Set x-axis limits.
    pub fn xlim(mut self, min: f64, max: f64) -> Self {
        self.scales.x = Scale::Linear {
            min: Some(min),
            max: Some(max),
        };
        self
    }

    /// Set y-axis limits.
    pub fn ylim(mut self, min: f64, max: f64) -> Self {
        self.scales.y = Scale::Linear {
            min: Some(min),
            max: Some(max),
        };
        self
    }

    /// Use the minimal theme.
    pub fn theme_minimal(mut self) -> Self {
        self.theme = Theme::minimal();
        self
    }

    /// Use the publication-ready theme.
    pub fn theme_publication(mut self) -> Self {
        self.theme = Theme::publication();
        self
    }

    /// Use the dark theme.
    pub fn theme_dark(mut self) -> Self {
        self.theme = Theme::dark();
        self
    }

    /// Flip x and y coordinates.
    pub fn coord_flip(mut self) -> Self {
        self.coord = CoordSystem::FlipXY;
        self
    }

    /// Use polar coordinates (x → angle, y → radius).
    /// Start at 12 o'clock (π/2), go clockwise.
    pub fn coord_polar(mut self) -> Self {
        self.coord = CoordSystem::Polar {
            start_angle: std::f64::consts::FRAC_PI_2,
            direction: PolarDirection::CW,
        };
        self
    }

    // ── Phase 3: Polar geoms ─────────────────────────────────────────

    /// Add a pie chart geometry (requires categorical x + numeric y data).
    pub fn geom_pie(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Pie,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a pie chart with donut hole (inner_radius as fraction of outer: 0.0–1.0).
    pub fn geom_donut(self, inner_frac: f64) -> Self {
        self.add_layer(Layer {
            geom: Geom::Pie,
            aes: AesMapping::xy(),
            params: GeomParams {
                inner_radius: inner_frac.clamp(0.0, 0.95),
                ..GeomParams::default()
            },
        })
    }

    /// Add a rose (Nightingale) chart geometry.
    pub fn geom_rose(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Rose,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a radar (spider) chart geometry.
    pub fn geom_radar(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Radar,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    // ── Phase 3: 2D density + contour geoms ─────────────────────────

    /// Add a 2D density estimation geom (filled contour bands).
    pub fn geom_density2d(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Density2d,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    /// Add a 2D density with custom number of levels and grid size.
    pub fn geom_density2d_opts(self, n_levels: usize, grid_size: usize) -> Self {
        self.add_layer(Layer {
            geom: Geom::Density2d,
            aes: AesMapping::xy(),
            params: GeomParams {
                n_levels,
                grid_size,
                ..GeomParams::default()
            },
        })
    }

    /// Add contour lines.
    pub fn geom_contour(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Contour,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    // ── Phase 3.2: Error bars + Step line ───────────────────────────

    /// Add error bars geom. Expects "x", "y", and "error" columns in data.
    pub fn geom_errorbar(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::ErrorBar,
            aes: AesMapping::xy(),
            params: GeomParams {
                error_column: Some("error".to_string()),
                ..GeomParams::default()
            },
        })
    }

    /// Add error bars with a custom error column name.
    pub fn geom_errorbar_col(self, col: &str) -> Self {
        self.add_layer(Layer {
            geom: Geom::ErrorBar,
            aes: AesMapping::xy(),
            params: GeomParams {
                error_column: Some(col.to_string()),
                ..GeomParams::default()
            },
        })
    }

    /// Add a step line geom (horizontal-then-vertical segments).
    pub fn geom_step(self) -> Self {
        self.add_layer(Layer {
            geom: Geom::Step,
            aes: AesMapping::xy(),
            params: GeomParams::default(),
        })
    }

    // ── Phase 3.2: Legend + Log scale ─────────────────────────────────

    /// Disable the legend.
    pub fn no_legend(mut self) -> Self {
        self.show_legend = false;
        self
    }

    /// Set x-axis to log scale.
    pub fn scale_x_log(mut self, base: f64) -> Self {
        self.scales.x = Scale::Log { base };
        self
    }

    /// Set y-axis to log scale.
    pub fn scale_y_log(mut self, base: f64) -> Self {
        self.scales.y = Scale::Log { base };
        self
    }

    /// Set the plot dimensions in pixels.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Add (or replace) a named data column.
    ///
    /// This is useful for attaching additional data (like error magnitudes)
    /// after initial construction:
    /// ```text
    /// plot.add_column("error", DataColumn::Float(vec![0.1, 0.2, 0.3]))
    /// ```
    pub fn add_column(mut self, name: &str, col: DataColumn) -> Self {
        // Replace existing column with the same name, or append.
        if let Some(pos) = self.data.columns.iter().position(|(n, _)| n == name) {
            self.data.columns[pos] = (name.to_string(), col);
        } else {
            self.data.columns.push((name.to_string(), col));
        }
        self
    }

    /// Add an annotation.
    pub fn annotate(mut self, ann: Annotation) -> Self {
        self.annotations.push(ann);
        self
    }
}

/// Column-oriented data for a plot.
#[derive(Debug, Clone)]
pub struct PlotData {
    pub columns: Vec<(String, DataColumn)>,
}

impl PlotData {
    /// Get a column by name.
    pub fn get(&self, name: &str) -> Option<&DataColumn> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// Number of rows (length of first column, or 0).
    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }
}

/// A data column: typed array of values.
#[derive(Debug, Clone)]
pub enum DataColumn {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Str(Vec<String>),
}

impl DataColumn {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            DataColumn::Float(v) => v.len(),
            DataColumn::Int(v) => v.len(),
            DataColumn::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to f64 values (Int → cast, Str → index, Float → identity).
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            DataColumn::Float(v) => v.clone(),
            DataColumn::Int(v) => v.iter().map(|&x| x as f64).collect(),
            DataColumn::Str(v) => (0..v.len()).map(|i| i as f64).collect(),
        }
    }

    /// Get string labels (Str → identity, others → formatted).
    pub fn labels(&self) -> Vec<String> {
        match self {
            DataColumn::Str(v) => v.clone(),
            DataColumn::Float(v) => v.iter().map(|x| crate::text::format_tick(*x)).collect(),
            DataColumn::Int(v) => v.iter().map(|x| x.to_string()).collect(),
        }
    }

    /// Returns true if this is a discrete (string) column.
    pub fn is_discrete(&self) -> bool {
        matches!(self, DataColumn::Str(_))
    }
}

/// A visual layer: one geometry + aesthetics + parameters.
#[derive(Debug, Clone)]
pub struct Layer {
    pub geom: Geom,
    pub aes: AesMapping,
    pub params: GeomParams,
}

/// Geometry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geom {
    Point,
    Line,
    Bar,
    Histogram,
    // Phase 2B: Distribution geoms
    Density,
    Area,
    Rug,
    Ecdf,
    // Phase 2B: Categorical geoms
    Box,
    Violin,
    Strip,
    Swarm,
    Boxen,
    // Phase 2B: Matrix/Heatmap
    Tile,
    // Phase 2B: Regression
    RegressionLine,
    Residual,
    // Phase 2B: Dendrogram
    Dendrogram,
    // Phase 3: Polar geoms
    /// Pie chart: slices from category/value data.
    Pie,
    /// Rose (Nightingale) chart: bars in polar coordinates.
    Rose,
    /// Radar (spider) chart: polygon on radial axes.
    Radar,
    // Phase 3: 2D density + contour
    /// 2D kernel density estimation rendered as filled contour bands.
    Density2d,
    /// Contour lines at specified density/value levels.
    Contour,
    // Phase 3.2: Error bars + step line
    /// Error bars: vertical bars at each data point (x, y ± error).
    ErrorBar,
    /// Step line: horizontal-then-vertical segments (for step functions).
    Step,
}

/// Aesthetic mappings: column names → visual channels.
#[derive(Debug, Clone, Default)]
pub struct AesMapping {
    pub x: Option<String>,
    pub y: Option<String>,
    pub color: Option<String>,
    pub size: Option<String>,
    pub label: Option<String>,
    pub fill: Option<String>,
    pub group: Option<String>,
}

impl AesMapping {
    /// Default x/y mapping.
    pub fn xy() -> Self {
        AesMapping {
            x: Some("x".to_string()),
            y: Some("y".to_string()),
            ..Default::default()
        }
    }
}

/// Which side to place rug marks on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RugSide {
    Bottom,
    Left,
    Top,
    Right,
}

impl Default for RugSide {
    fn default() -> Self {
        RugSide::Bottom
    }
}

/// Parameters for geometry rendering.
#[derive(Debug, Clone)]
pub struct GeomParams {
    pub point_size: f64,
    pub line_width: f64,
    pub bar_width: Option<f64>,
    pub bins: Option<usize>,
    pub alpha: f64,
    pub color_override: Option<Color>,
    // Phase 2B: Distribution params
    pub bandwidth: Option<f64>,
    pub n_grid_points: Option<usize>,
    pub rug_side: RugSide,
    pub rug_length: f64,
    // Phase 2B: Categorical params
    pub jitter_width: f64,
    pub violin_bw: Option<f64>,
    pub show_outliers: bool,
    pub category_width: f64,
    // Phase 2B: Heatmap params
    pub show_cell_values: bool,
    pub cell_format: Option<String>,
    // Phase 3: Polar params
    /// Inner radius for donut-style pie charts (0.0 = full pie, >0 = donut).
    pub inner_radius: f64,
    /// Whether to show percentage labels on pie slices.
    pub show_labels: bool,
    // Phase 3: 2D density / contour params
    /// Number of contour levels for density2d / contour geoms.
    pub n_levels: usize,
    /// Grid resolution for 2D density estimation.
    pub grid_size: usize,
    // Phase 3.2: Error bar params
    /// Column name for error magnitudes (for ErrorBar geom).
    pub error_column: Option<String>,
    /// Cap width for error bar whiskers (in data units fraction, default 0.3).
    pub cap_width: f64,
    // Phase 3.2: Step direction
    /// Step direction: true = horizontal-first (step-post), false = vertical-first.
    pub step_post: bool,
    // Phase 3.2: Legend
    /// Whether to show the legend. Auto-enables for 2+ layer plots.
    pub show_legend: bool,
}

impl Default for GeomParams {
    fn default() -> Self {
        GeomParams {
            point_size: 4.0,
            line_width: 2.0,
            bar_width: None,
            bins: None,
            alpha: 1.0,
            color_override: None,
            bandwidth: None,
            n_grid_points: None,
            rug_side: RugSide::Bottom,
            rug_length: 8.0,
            jitter_width: 0.3,
            violin_bw: None,
            show_outliers: true,
            category_width: 0.7,
            show_cell_values: false,
            cell_format: None,
            inner_radius: 0.0,
            show_labels: true,
            n_levels: 7,
            grid_size: 50,
            error_column: None,
            cap_width: 0.3,
            step_post: true,
            show_legend: true,
        }
    }
}

/// Scale configuration.
#[derive(Debug, Clone)]
pub struct ScaleSet {
    pub x: Scale,
    pub y: Scale,
    pub color: ColorScale,
}

impl Default for ScaleSet {
    fn default() -> Self {
        ScaleSet {
            x: Scale::Linear { min: None, max: None },
            y: Scale::Linear { min: None, max: None },
            color: ColorScale::Default,
        }
    }
}

/// A single axis scale.
#[derive(Debug, Clone)]
pub enum Scale {
    Linear { min: Option<f64>, max: Option<f64> },
    Log { base: f64 },
    Discrete { levels: Vec<String> },
}

/// Color scale.
#[derive(Debug, Clone)]
pub enum ColorScale {
    Default,
    Manual(Vec<Color>),
    Gradient { low: Color, high: Color },
    /// Sequential: white → blue.
    Sequential,
    /// Diverging: blue → white → red.
    Diverging,
}

/// Axis labels and title.
#[derive(Debug, Clone, Default)]
pub struct Labels {
    pub title: Option<String>,
    pub x: Option<String>,
    pub y: Option<String>,
    pub subtitle: Option<String>,
}

/// Coordinate system.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordSystem {
    Cartesian,
    FlipXY,
    /// Polar coordinate system: x → angle (theta), y → radius.
    /// `start_angle` is the angle (in radians) for the first data point.
    /// `direction` is CW (clockwise, like a clock) or CCW.
    Polar {
        start_angle: f64,
        direction: PolarDirection,
    },
}

/// Direction for polar coordinate traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarDirection {
    /// Clockwise (like a clock).
    CW,
    /// Counter-clockwise (mathematical convention).
    CCW,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_spec_builder() {
        let p = PlotSpec::from_xy(vec![1.0, 2.0], vec![3.0, 4.0])
            .geom_point()
            .title("Test")
            .xlab("X")
            .ylab("Y");
        assert_eq!(p.layers.len(), 1);
        assert_eq!(p.labels.title, Some("Test".to_string()));
    }

    #[test]
    fn test_data_column_len() {
        let c = DataColumn::Float(vec![1.0, 2.0, 3.0]);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn test_plot_data_get() {
        let d = PlotData {
            columns: vec![
                ("x".to_string(), DataColumn::Float(vec![1.0])),
                ("y".to_string(), DataColumn::Float(vec![2.0])),
            ],
        };
        assert!(d.get("x").is_some());
        assert!(d.get("z").is_none());
    }

    // ── Phase 5 (Audit): add_column method ──

    #[test]
    fn test_add_column_new() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0], vec![3.0, 4.0])
            .add_column("error", DataColumn::Float(vec![0.5, 1.0]));
        assert!(spec.data.get("error").is_some());
        let col = spec.data.get("error").unwrap();
        assert_eq!(col.len(), 2);
    }

    #[test]
    fn test_add_column_replaces_existing() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0], vec![3.0, 4.0])
            .add_column("y", DataColumn::Float(vec![10.0, 20.0]));
        // y column should be replaced, not duplicated.
        let y_count = spec.data.columns.iter().filter(|(n, _)| n == "y").count();
        assert_eq!(y_count, 1, "Should have exactly one 'y' column after replace");
        let col = spec.data.get("y").unwrap();
        let vals = col.to_f64();
        assert_eq!(vals, vec![10.0, 20.0]);
    }

    #[test]
    fn test_add_column_preserves_other_columns() {
        let spec = PlotSpec::from_xy(vec![1.0, 2.0], vec![3.0, 4.0])
            .add_column("extra", DataColumn::Float(vec![5.0, 6.0]));
        // x and y should still exist.
        assert!(spec.data.get("x").is_some());
        assert!(spec.data.get("y").is_some());
        assert!(spec.data.get("extra").is_some());
        assert_eq!(spec.data.columns.len(), 3);
    }

    #[test]
    fn test_add_column_chain() {
        let spec = PlotSpec::from_xy(vec![1.0], vec![2.0])
            .add_column("a", DataColumn::Float(vec![3.0]))
            .add_column("b", DataColumn::Float(vec![4.0]));
        assert!(spec.data.get("a").is_some());
        assert!(spec.data.get("b").is_some());
        assert_eq!(spec.data.columns.len(), 4);
    }
}
