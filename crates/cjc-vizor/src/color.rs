//! Color types, hex parsing, and default categorical palette.

/// An RGBA color with u8 components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create an opaque RGB color.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b, a: 255 }
    }

    /// Create an RGBA color.
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    /// Parse a hex color string: "#RRGGBB" or "#RGB".
    pub fn hex(s: &str) -> Option<Self> {
        let s = s.strip_prefix('#').unwrap_or(s);
        match s.len() {
            6 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                Some(Color::rgb(r, g, b))
            }
            3 => {
                let r = u8::from_str_radix(&s[0..1], 16).ok()?;
                let g = u8::from_str_radix(&s[1..2], 16).ok()?;
                let b = u8::from_str_radix(&s[2..3], 16).ok()?;
                Some(Color::rgb(r * 17, g * 17, b * 17))
            }
            _ => None,
        }
    }

    /// Format as SVG-compatible color string.
    pub fn to_svg(&self) -> String {
        if self.a == 255 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!(
                "rgba({},{},{},{:.2})",
                self.r,
                self.g,
                self.b,
                self.a as f64 / 255.0
            )
        }
    }

    // Named colors.
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    pub const TRANSPARENT: Color = Color::rgba(0, 0, 0, 0);
    pub const LIGHT_GRAY: Color = Color::rgb(229, 229, 229);
    pub const GRAY: Color = Color::rgb(128, 128, 128);
    pub const DARK_GRAY: Color = Color::rgb(64, 64, 64);
}

/// Default categorical palette (8 colors, ggplot2-inspired).
pub const PALETTE: [Color; 8] = [
    Color::rgb(228, 26, 28),   // red
    Color::rgb(55, 126, 184),  // blue
    Color::rgb(77, 175, 74),   // green
    Color::rgb(152, 78, 163),  // purple
    Color::rgb(255, 127, 0),   // orange
    Color::rgb(255, 255, 51),  // yellow
    Color::rgb(166, 86, 40),   // brown
    Color::rgb(247, 129, 191), // pink
];

/// Get a color from the default palette (wraps around).
pub fn palette_color(index: usize) -> Color {
    PALETTE[index % PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_parsing_6() {
        assert_eq!(Color::hex("#ff0000"), Some(Color::rgb(255, 0, 0)));
        assert_eq!(Color::hex("00ff00"), Some(Color::rgb(0, 255, 0)));
        assert_eq!(Color::hex("#0000ff"), Some(Color::rgb(0, 0, 255)));
    }

    #[test]
    fn test_hex_parsing_3() {
        assert_eq!(Color::hex("#f00"), Some(Color::rgb(255, 0, 0)));
        assert_eq!(Color::hex("#fff"), Some(Color::rgb(255, 255, 255)));
    }

    #[test]
    fn test_hex_invalid() {
        assert_eq!(Color::hex("#xyz"), None);
        assert_eq!(Color::hex(""), None);
        assert_eq!(Color::hex("#12345"), None);
    }

    #[test]
    fn test_to_svg_opaque() {
        let c = Color::rgb(255, 0, 128);
        assert_eq!(c.to_svg(), "#ff0080");
    }

    #[test]
    fn test_to_svg_alpha() {
        let c = Color::rgba(255, 0, 0, 128);
        assert!(c.to_svg().starts_with("rgba("));
    }

    #[test]
    fn test_palette_wrap() {
        assert_eq!(palette_color(0), palette_color(8));
        assert_eq!(palette_color(1), palette_color(9));
    }
}
