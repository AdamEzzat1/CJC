//! Fixed-width text measurement policy.
//!
//! Without font libraries, we approximate text width using a fixed ratio.
//! SVG output defers to the rendering engine; BMP uses a built-in bitmap font.

use crate::theme::Theme;

/// Measure text dimensions using fixed-width approximation.
/// Returns (width, height) in pixels.
pub fn measure_text(text: &str, font_size: f64, theme: &Theme) -> (f64, f64) {
    let char_width = font_size * theme.font_width_ratio;
    let width = text.len() as f64 * char_width;
    let height = font_size;
    (width, height)
}

/// Format a float for display on axes/annotations.
/// Uses deterministic fixed-precision formatting.
pub fn format_tick(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    if abs >= 1e6 || (abs < 0.01 && abs > 0.0) {
        // Scientific notation for very large/small numbers.
        format!("{:.2e}", value)
    } else if (value - value.round()).abs() < 1e-9 {
        // Integer-like values.
        format!("{:.0}", value)
    } else if abs >= 100.0 {
        format!("{:.1}", value)
    } else if abs >= 1.0 {
        format!("{:.2}", value)
    } else {
        format!("{:.3}", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_text() {
        let theme = Theme::default();
        let (w, h) = measure_text("Hello", 10.0, &theme);
        assert_eq!(w, 5.0 * 10.0 * 0.6); // 30.0
        assert_eq!(h, 10.0);
    }

    #[test]
    fn test_format_tick_integer() {
        assert_eq!(format_tick(42.0), "42");
        assert_eq!(format_tick(0.0), "0");
        assert_eq!(format_tick(-100.0), "-100");
    }

    #[test]
    fn test_format_tick_decimal() {
        assert_eq!(format_tick(3.14), "3.14");
        assert_eq!(format_tick(0.5), "0.500");
    }

    #[test]
    fn test_format_tick_scientific() {
        let s = format_tick(1e7);
        assert!(s.contains('e'));
    }

    #[test]
    fn test_format_tick_deterministic() {
        let a = format_tick(1.23456);
        let b = format_tick(1.23456);
        assert_eq!(a, b);
    }
}
