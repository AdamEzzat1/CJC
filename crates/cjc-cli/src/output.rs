//! Shared output formatting for CJC CLI commands.
//!
//! Provides unified output modes (plain, color, JSON, table) and
//! deterministic formatting utilities used by all subcommands.

use crate::table::Table;

// ── ANSI color constants ─────────────────────────────────────────────

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const BLUE: &str = "\x1b[34m";
pub const MAGENTA: &str = "\x1b[35m";
pub const CYAN: &str = "\x1b[36m";
pub const BOLD_RED: &str = "\x1b[1;31m";
pub const BOLD_GREEN: &str = "\x1b[1;32m";
pub const BOLD_YELLOW: &str = "\x1b[1;33m";
pub const BOLD_BLUE: &str = "\x1b[1;34m";
pub const BOLD_CYAN: &str = "\x1b[1;36m";

/// Output mode for CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Human-readable with ANSI colors.
    Color,
    /// Plain text, no colors.
    Plain,
    /// Machine-readable JSON.
    Json,
    /// ASCII/Unicode table format.
    Table,
}

impl OutputMode {
    /// Parse from --output flag value.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "color" => Ok(OutputMode::Color),
            "plain" => Ok(OutputMode::Plain),
            "json" => Ok(OutputMode::Json),
            "table" => Ok(OutputMode::Table),
            other => Err(format!("unknown output mode `{}` (expected: color, plain, json, table)", other)),
        }
    }

    pub fn use_color(&self) -> bool {
        *self == OutputMode::Color
    }
}

/// Colorize text if mode supports it.
pub fn colorize(mode: OutputMode, color: &str, text: &str) -> String {
    if mode.use_color() {
        format!("{}{}{}", color, text, RESET)
    } else {
        text.to_string()
    }
}

/// Format a hash as a short hex prefix (first 8 chars) or full hex.
pub fn format_hash_short(hash: &[u8]) -> String {
    hash.iter().take(4).map(|b| format!("{:02x}", b)).collect()
}

pub fn format_hash_full(hash: &[u8]) -> String {
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Deterministic float formatting: fixed precision, canonical NaN/Inf handling.
pub fn format_f64(v: f64, precision: usize) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Inf".to_string() } else { "-Inf".to_string() }
    } else {
        format!("{:.prec$}", v, prec = precision)
    }
}

/// Format a byte count as a human-readable size.
pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Emit a simple JSON object from key-value string pairs.
/// Values are emitted as JSON strings unless they start with `[`, `{`, or are numeric/bool.
pub fn json_object(pairs: &[(&str, &str)]) -> String {
    let mut out = String::from("{\n");
    for (i, (key, val)) in pairs.iter().enumerate() {
        let json_val = if val.starts_with('[') || val.starts_with('{')
            || *val == "true" || *val == "false" || *val == "null"
            || val.parse::<f64>().is_ok()
        {
            val.to_string()
        } else {
            format!("\"{}\"", val.replace('\\', "\\\\").replace('"', "\\\""))
        };
        out.push_str(&format!("  \"{}\": {}", key, json_val));
        if i + 1 < pairs.len() { out.push(','); }
        out.push('\n');
    }
    out.push('}');
    out
}

/// Emit a JSON array of objects.
pub fn json_array(objects: &[Vec<(&str, &str)>]) -> String {
    let mut out = String::from("[\n");
    for (i, obj) in objects.iter().enumerate() {
        let inner = json_object(obj);
        // Indent each line
        for (j, line) in inner.lines().enumerate() {
            out.push_str("  ");
            out.push_str(line);
            if j < inner.lines().count() - 1 {
                out.push('\n');
            }
        }
        if i + 1 < objects.len() { out.push(','); }
        out.push('\n');
    }
    out.push(']');
    out
}

/// Create a Table from headers and rows (convenience wrapper).
pub fn make_table(headers: Vec<&str>, rows: Vec<Vec<String>>) -> String {
    let mut t = Table::new(headers);
    for row in rows {
        t.add_row_owned(row);
    }
    t.render()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_f64_normal() {
        assert_eq!(format_f64(3.14159, 4), "3.1416");
        assert_eq!(format_f64(0.0, 2), "0.00");
    }

    #[test]
    fn test_format_f64_special() {
        assert_eq!(format_f64(f64::NAN, 4), "NaN");
        assert_eq!(format_f64(f64::INFINITY, 4), "Inf");
        assert_eq!(format_f64(f64::NEG_INFINITY, 4), "-Inf");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(2048), "2.0 KiB");
        assert_eq!(format_size(1048576), "1.0 MiB");
    }

    #[test]
    fn test_format_hash_short() {
        let hash = [0xab, 0xcd, 0xef, 0x01, 0x23];
        assert_eq!(format_hash_short(&hash), "abcdef01");
    }

    #[test]
    fn test_json_object() {
        let obj = json_object(&[("name", "test"), ("count", "42")]);
        assert!(obj.contains("\"name\": \"test\""));
        assert!(obj.contains("\"count\": 42"));
    }

    #[test]
    fn test_output_mode_parse() {
        assert_eq!(OutputMode::from_str("json").unwrap(), OutputMode::Json);
        assert_eq!(OutputMode::from_str("plain").unwrap(), OutputMode::Plain);
        assert!(OutputMode::from_str("bogus").is_err());
    }

    #[test]
    fn test_colorize_plain() {
        assert_eq!(colorize(OutputMode::Plain, RED, "error"), "error");
    }

    #[test]
    fn test_colorize_color() {
        let result = colorize(OutputMode::Color, RED, "error");
        assert!(result.contains("\x1b[31m"));
        assert!(result.contains("error"));
    }
}
