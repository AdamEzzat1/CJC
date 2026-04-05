//! Semantic annotation types for data visualization.
//!
//! Annotations are typed objects with stable rendering and placement rules.
//! They carry semantic intent, not just raw text.

/// Position hint for annotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// Absolute pixel position.
    Absolute { x: f64, y: f64 },
    /// Data coordinates (transformed by scales).
    Data { x: f64, y: f64 },
    /// Relative to plot area (0.0–1.0).
    Relative { x: f64, y: f64 },
    /// Top-right corner of plot area.
    TopRight,
    /// Top-left corner of plot area.
    TopLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Bottom-left corner.
    BottomLeft,
}

impl Default for Position {
    fn default() -> Self {
        Position::TopRight
    }
}

/// A semantic annotation on a plot.
#[derive(Debug, Clone)]
pub enum Annotation {
    /// Free-form text at a position.
    Text {
        text: String,
        position: Position,
        font_size: Option<f64>,
    },

    /// A note (small caption-like text).
    Note {
        text: String,
        position: Position,
    },

    /// A callout with an arrow pointing to target data coords.
    Callout {
        text: String,
        target: Position,
        label_offset: (f64, f64),
    },

    /// Regression summary (equation + R²).
    RegressionSummary {
        equation: String,
        r_squared: f64,
        position: Position,
    },

    /// Confidence interval label.
    ConfidenceInterval {
        level: f64,
        lower: f64,
        upper: f64,
        position: Position,
    },

    /// P-value annotation.
    PValue {
        value: f64,
        significance_level: f64,
        position: Position,
    },

    /// Model metrics box (key-value pairs).
    ModelMetrics {
        metrics: Vec<(String, f64)>,
        position: Position,
    },

    /// Vertical event marker at an x-value.
    EventMarker {
        x: f64,
        label: String,
    },

    /// Data provenance / source note.
    DataNote {
        text: String,
        position: Position,
    },

    /// Inline label near a data point.
    InlineLabel {
        text: String,
        x: f64,
        y: f64,
    },
}

impl Annotation {
    // ── Constructors ─────────────────────────────────────────────

    /// Create a text annotation at the given data coordinates.
    pub fn text(text: &str, x: f64, y: f64) -> Self {
        Annotation::Text {
            text: text.to_string(),
            position: Position::Data { x, y },
            font_size: None,
        }
    }

    /// Create a small note annotation at the given position.
    pub fn note(text: &str, position: Position) -> Self {
        Annotation::Note {
            text: text.to_string(),
            position,
        }
    }

    /// Create a regression summary annotation (equation + R-squared).
    pub fn regression(equation: &str, r_squared: f64) -> Self {
        Annotation::RegressionSummary {
            equation: equation.to_string(),
            r_squared,
            position: Position::TopRight,
        }
    }

    /// Create a confidence interval annotation at the default position.
    pub fn ci(level: f64, lower: f64, upper: f64) -> Self {
        Annotation::ConfidenceInterval {
            level,
            lower,
            upper,
            position: Position::TopRight,
        }
    }

    /// Create a p-value annotation with default significance level (0.05).
    pub fn pvalue(value: f64) -> Self {
        Annotation::PValue {
            value,
            significance_level: 0.05,
            position: Position::TopRight,
        }
    }

    /// Create a model metrics box with key-value pairs.
    pub fn model_metrics(metrics: Vec<(String, f64)>) -> Self {
        Annotation::ModelMetrics {
            metrics,
            position: Position::TopRight,
        }
    }

    /// Create a vertical event marker at the given x-value.
    pub fn event_marker(x: f64, label: &str) -> Self {
        Annotation::EventMarker {
            x,
            label: label.to_string(),
        }
    }

    /// Create a data provenance / source note at bottom-left.
    pub fn data_note(text: &str) -> Self {
        Annotation::DataNote {
            text: text.to_string(),
            position: Position::BottomLeft,
        }
    }

    /// Create an inline label positioned near a specific data point.
    pub fn inline_label(text: &str, x: f64, y: f64) -> Self {
        Annotation::InlineLabel {
            text: text.to_string(),
            x,
            y,
        }
    }
}

/// Format a p-value for display.
pub fn format_pvalue(p: f64) -> String {
    if p < 0.001 {
        "p < 0.001".to_string()
    } else if p < 0.01 {
        format!("p = {:.3}", p)
    } else if p < 0.05 {
        format!("p = {:.3}", p)
    } else {
        format!("p = {:.2}", p)
    }
}

/// Format R² for display.
pub fn format_r_squared(r2: f64) -> String {
    format!("R\u{00b2} = {:.4}", r2)
}

/// Format a confidence interval.
pub fn format_ci(level: f64, lower: f64, upper: f64) -> String {
    format!("{:.0}% CI: [{:.3}, {:.3}]", level * 100.0, lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_pvalue() {
        assert_eq!(format_pvalue(0.0001), "p < 0.001");
        assert_eq!(format_pvalue(0.03), "p = 0.030");
        assert_eq!(format_pvalue(0.5), "p = 0.50");
    }

    #[test]
    fn test_format_r_squared() {
        let s = format_r_squared(0.95);
        assert!(s.contains("0.9500"));
    }

    #[test]
    fn test_format_ci() {
        let s = format_ci(0.95, 1.2, 3.4);
        assert!(s.contains("95%"));
        assert!(s.contains("1.200"));
    }
}
