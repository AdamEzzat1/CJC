//! Theme definitions — margins, colors, font sizes, spacing.

use crate::color::Color;

/// Visual theme for plot rendering.
#[derive(Debug, Clone)]
pub struct Theme {
    pub margin_top: f64,
    pub margin_right: f64,
    pub margin_bottom: f64,
    pub margin_left: f64,
    pub background: Color,
    pub plot_background: Color,
    pub axis_color: Color,
    pub grid_color: Color,
    pub text_color: Color,
    pub title_font_size: f64,
    pub axis_label_font_size: f64,
    pub tick_label_font_size: f64,
    /// Fixed-width text measurement ratio: char_width = font_size * ratio.
    pub font_width_ratio: f64,
    pub point_size: f64,
    pub line_width: f64,
    pub grid_line_width: f64,
    pub axis_line_width: f64,
    pub tick_length: f64,
}

impl Default for Theme {
    fn default() -> Self {
        Theme {
            margin_top: 40.0,
            margin_right: 20.0,
            margin_bottom: 60.0,
            margin_left: 70.0,
            background: Color::WHITE,
            plot_background: Color::WHITE,
            axis_color: Color::DARK_GRAY,
            grid_color: Color::LIGHT_GRAY,
            text_color: Color::DARK_GRAY,
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
}
