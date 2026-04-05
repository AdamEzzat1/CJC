//! Theme definitions — margins, colors, font sizes, spacing.

use crate::color::Color;

/// Visual theme for plot rendering.
///
/// Controls margins, colors, font sizes, line widths, and other
/// visual properties that apply globally to a plot.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Top margin in pixels.
    pub margin_top: f64,
    /// Right margin in pixels.
    pub margin_right: f64,
    /// Bottom margin in pixels.
    pub margin_bottom: f64,
    /// Left margin in pixels.
    pub margin_left: f64,
    /// Outer background color (behind the plot area).
    pub background: Color,
    /// Inner plot area background color.
    pub plot_background: Color,
    /// Color for axis lines and tick marks.
    pub axis_color: Color,
    /// Color for grid lines.
    pub grid_color: Color,
    /// Color for title, axis labels, and tick labels.
    pub text_color: Color,
    /// Font size for the plot title (pixels).
    pub title_font_size: f64,
    /// Font size for axis labels (pixels).
    pub axis_label_font_size: f64,
    /// Font size for tick labels (pixels).
    pub tick_label_font_size: f64,
    /// Fixed-width text measurement ratio: char_width = font_size * ratio.
    pub font_width_ratio: f64,
    /// Default radius for scatter plot points (pixels).
    pub point_size: f64,
    /// Default stroke width for line geoms (pixels).
    pub line_width: f64,
    /// Stroke width for grid lines (pixels).
    pub grid_line_width: f64,
    /// Stroke width for axis lines (pixels).
    pub axis_line_width: f64,
    /// Length of axis tick marks (pixels).
    pub tick_length: f64,
}

impl Default for Theme {
    fn default() -> Self {
        Theme {
            margin_top: 45.0,
            margin_right: 25.0,
            margin_bottom: 60.0,
            margin_left: 70.0,
            background: Color::WHITE,
            plot_background: Color::rgb(252, 252, 252), // Subtle off-white plot bg
            axis_color: Color::rgb(80, 80, 80),
            grid_color: Color::rgb(230, 230, 230),      // Lighter grid for clean look
            text_color: Color::rgb(50, 50, 50),
            title_font_size: 16.0,
            axis_label_font_size: 12.0,
            tick_label_font_size: 10.0,
            font_width_ratio: 0.6,
            point_size: 4.0,
            line_width: 2.0,
            grid_line_width: 0.4,
            axis_line_width: 1.0,
            tick_length: 5.0,
        }
    }
}

impl Theme {
    /// A minimal theme with lighter grid and more whitespace.
    pub fn minimal() -> Self {
        Theme {
            margin_top: 30.0,
            margin_right: 15.0,
            margin_bottom: 50.0,
            margin_left: 60.0,
            background: Color::WHITE,
            plot_background: Color::WHITE,
            axis_color: Color::GRAY,
            grid_color: Color::rgb(240, 240, 240),
            text_color: Color::GRAY,
            title_font_size: 14.0,
            axis_label_font_size: 11.0,
            tick_label_font_size: 9.0,
            font_width_ratio: 0.6,
            point_size: 3.5,
            line_width: 1.5,
            grid_line_width: 0.3,
            axis_line_width: 0.5,
            tick_length: 4.0,
        }
    }

    /// A publication-ready theme with clean lines and no gridlines.
    pub fn publication() -> Self {
        Theme {
            margin_top: 35.0,
            margin_right: 20.0,
            margin_bottom: 55.0,
            margin_left: 65.0,
            background: Color::WHITE,
            plot_background: Color::WHITE,
            axis_color: Color::BLACK,
            grid_color: Color::rgba(0, 0, 0, 0), // no gridlines
            text_color: Color::BLACK,
            title_font_size: 14.0,
            axis_label_font_size: 12.0,
            tick_label_font_size: 10.0,
            font_width_ratio: 0.6,
            point_size: 3.5,
            line_width: 1.5,
            grid_line_width: 0.0,
            axis_line_width: 1.2,
            tick_length: 5.0,
        }
    }

    /// A dark theme with light text on dark background.
    pub fn dark() -> Self {
        Theme {
            margin_top: 40.0,
            margin_right: 20.0,
            margin_bottom: 60.0,
            margin_left: 70.0,
            background: Color::rgb(32, 32, 32),
            plot_background: Color::rgb(40, 40, 40),
            axis_color: Color::rgb(180, 180, 180),
            grid_color: Color::rgb(60, 60, 60),
            text_color: Color::rgb(200, 200, 200),
            title_font_size: 16.0,
            axis_label_font_size: 12.0,
            tick_label_font_size: 10.0,
            font_width_ratio: 0.6,
            point_size: 4.0,
            line_width: 2.0,
            grid_line_width: 0.5,
            axis_line_width: 1.0,
            tick_length: 5.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_theme() {
        let t = Theme::default();
        assert!(t.margin_left > 0.0);
        assert!(t.title_font_size > 0.0);
    }

    #[test]
    fn test_minimal_theme() {
        let t = Theme::minimal();
        assert!(t.grid_line_width < Theme::default().grid_line_width);
    }

    // ── Phase 5 (Audit): verify updated default theme values ──

    #[test]
    fn test_default_theme_margins() {
        let t = Theme::default();
        assert_eq!(t.margin_top, 45.0, "margin_top should be 45");
        assert_eq!(t.margin_right, 25.0, "margin_right should be 25");
        assert_eq!(t.margin_bottom, 60.0);
        assert_eq!(t.margin_left, 70.0);
    }

    #[test]
    fn test_default_theme_colors() {
        let t = Theme::default();
        // Off-white plot background.
        assert_eq!(t.plot_background, Color::rgb(252, 252, 252));
        // Muted axis color.
        assert_eq!(t.axis_color, Color::rgb(80, 80, 80));
        // Light grid color.
        assert_eq!(t.grid_color, Color::rgb(230, 230, 230));
        // Dark text color.
        assert_eq!(t.text_color, Color::rgb(50, 50, 50));
    }

    #[test]
    fn test_default_theme_grid_line_width() {
        let t = Theme::default();
        assert!((t.grid_line_width - 0.4).abs() < 1e-10, "grid_line_width should be 0.4");
    }

    #[test]
    fn test_publication_theme_has_no_gridlines() {
        let t = Theme::publication();
        // Publication theme should have thinner or zero grid lines.
        assert!(t.grid_line_width <= 0.5);
    }

    #[test]
    fn test_dark_theme_dark_background() {
        let t = Theme::dark();
        assert_eq!(t.background, Color::rgb(32, 32, 32));
        assert_eq!(t.plot_background, Color::rgb(40, 40, 40));
        // Text should be light.
        assert!(t.text_color.r > 150);
    }
}
