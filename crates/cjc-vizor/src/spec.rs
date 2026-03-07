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
    pub width: u32,
    pub height: u32,
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
            width: 800,
            height: 600,
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

    /// Set the plot title.
    pub fn title(mut self, title: &str) -> Self {
        self.labels.title = Some(title.to_string());
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

    /// Flip x and y coordinates.
    pub fn coord_flip(mut self) -> Self {
        self.coord = CoordSystem::FlipXY;
        self
    }

    /// Set the plot dimensions in pixels.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
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

/// Parameters for geometry rendering.
#[derive(Debug, Clone)]
pub struct GeomParams {
    pub point_size: f64,
    pub line_width: f64,
    pub bar_width: Option<f64>,
    pub bins: Option<usize>,
    pub alpha: f64,
    pub color_override: Option<Color>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordSystem {
    Cartesian,
    FlipXY,
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
}
