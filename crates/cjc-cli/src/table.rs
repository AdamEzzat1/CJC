//! Zero-dependency table formatter for the CJC REPL and debugging output.
//!
//! Produces aligned ASCII/Unicode box-drawing tables with auto-sized columns.
//! Designed for displaying symbol tables, environment bindings, and debug info.

/// A simple table with headers and rows, formatted with box-drawing characters.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    /// Create a new table with the given column headers.
    pub fn new(headers: Vec<&str>) -> Self {
        Self {
            headers: headers.into_iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    /// Add a row of values. Must have the same number of columns as headers.
    pub fn add_row(&mut self, cells: Vec<&str>) {
        self.rows.push(cells.into_iter().map(|c| c.to_string()).collect());
    }

    /// Add a row from owned Strings.
    pub fn add_row_owned(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    /// Render the table to a string using Unicode box-drawing characters.
    ///
    /// Example output:
    /// ```text
    /// ┌──────┬──────┬────────┐
    /// │ Name │ Type │ Value  │
    /// ├──────┼──────┼────────┤
    /// │ x    │ i64  │ 42     │
    /// │ name │ str  │ "hello"│
    /// └──────┴──────┴────────┘
    /// ```
    pub fn render(&self) -> String {
        if self.headers.is_empty() {
            return String::new();
        }

        let col_count = self.headers.len();
        let widths = self.compute_widths();

        let mut out = String::new();

        // Top border
        out.push_str(&self.border_line(&widths, '┌', '┬', '┐'));

        // Header row
        out.push_str(&self.data_line(&self.headers, &widths));

        // Header separator
        out.push_str(&self.border_line(&widths, '├', '┼', '┤'));

        // Data rows
        if self.rows.is_empty() {
            // Show "(empty)" spanning all columns
            let total_inner: usize = widths.iter().sum::<usize>() + (col_count - 1) * 3;
            out.push_str(&format!("│ {:<width$} │\n", "(empty)", width = total_inner));
        } else {
            for row in &self.rows {
                out.push_str(&self.data_line(row, &widths));
            }
        }

        // Bottom border
        out.push_str(&self.border_line(&widths, '└', '┴', '┘'));

        out
    }

    /// Render a compact table without box characters (for piping/grep).
    pub fn render_plain(&self) -> String {
        if self.headers.is_empty() {
            return String::new();
        }

        let widths = self.compute_widths();
        let mut out = String::new();

        // Header
        out.push_str(&self.plain_line(&self.headers, &widths));

        // Separator
        for (i, w) in widths.iter().enumerate() {
            if i > 0 { out.push_str("  "); }
            for _ in 0..*w { out.push('─'); }
        }
        out.push('\n');

        // Rows
        for row in &self.rows {
            out.push_str(&self.plain_line(row, &widths));
        }

        out
    }

    /// Compute the max width for each column.
    fn compute_widths(&self) -> Vec<usize> {
        let col_count = self.headers.len();
        let mut widths = vec![0usize; col_count];

        for (i, h) in self.headers.iter().enumerate() {
            widths[i] = widths[i].max(display_width(h));
        }

        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < col_count {
                    widths[i] = widths[i].max(display_width(cell));
                }
            }
        }

        widths
    }

    /// Render a border line: ┌──┬──┐ or ├──┼──┤ or └──┴──┘
    fn border_line(&self, widths: &[usize], left: char, mid: char, right: char) -> String {
        let mut out = String::new();
        out.push(left);
        for (i, w) in widths.iter().enumerate() {
            if i > 0 { out.push(mid); }
            for _ in 0..(*w + 2) { out.push('─'); }
        }
        out.push(right);
        out.push('\n');
        out
    }

    /// Render a data line: │ val │ val │
    fn data_line(&self, cells: &[String], widths: &[usize]) -> String {
        let mut out = String::new();
        out.push_str("│ ");
        for (i, w) in widths.iter().enumerate() {
            if i > 0 { out.push_str(" │ "); }
            let cell = cells.get(i).map(|s| s.as_str()).unwrap_or("");
            let padding = w.saturating_sub(display_width(cell));
            out.push_str(cell);
            for _ in 0..padding { out.push(' '); }
        }
        out.push_str(" │\n");
        out
    }

    /// Render a plain data line (no box chars).
    fn plain_line(&self, cells: &[String], widths: &[usize]) -> String {
        let mut out = String::new();
        for (i, w) in widths.iter().enumerate() {
            if i > 0 { out.push_str("  "); }
            let cell = cells.get(i).map(|s| s.as_str()).unwrap_or("");
            let padding = w.saturating_sub(display_width(cell));
            out.push_str(cell);
            for _ in 0..padding { out.push(' '); }
        }
        out.push('\n');
        out
    }
}

/// Compute display width of a string (strips ANSI codes for accurate measurement).
fn display_width(s: &str) -> usize {
    if !s.contains('\x1b') {
        return s.chars().count();
    }
    // Strip ANSI escape sequences
    let mut width = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if in_escape {
            if c == 'm' { in_escape = false; }
        } else if c == '\x1b' {
            in_escape = true;
        } else {
            width += 1;
        }
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_table() {
        let t = Table::new(vec![]);
        assert_eq!(t.render(), "");
    }

    #[test]
    fn test_single_column() {
        let mut t = Table::new(vec!["Name"]);
        t.add_row(vec!["foo"]);
        t.add_row(vec!["bar"]);
        let output = t.render();
        assert!(output.contains("Name"));
        assert!(output.contains("foo"));
        assert!(output.contains("bar"));
        assert!(output.contains('┌'));
        assert!(output.contains('└'));
    }

    #[test]
    fn test_multi_column_alignment() {
        let mut t = Table::new(vec!["Name", "Type", "Value"]);
        t.add_row(vec!["x", "i64", "42"]);
        t.add_row(vec!["long_name", "str", "\"hello world\""]);
        let output = t.render();
        // Verify alignment: the Name column should be padded to "long_name" width
        assert!(output.contains("long_name"));
        // Each row should have 3 │ delimiters (left edge, between cols, right edge)
        for line in output.lines() {
            if line.contains("Name") || line.contains("x") || line.contains("long_name") {
                let pipe_count = line.chars().filter(|&c| c == '│').count();
                assert_eq!(pipe_count, 4, "line: {}", line); // │ val │ val │ val │
            }
        }
    }

    #[test]
    fn test_empty_rows() {
        let t = Table::new(vec!["Name", "Value"]);
        let output = t.render();
        assert!(output.contains("(empty)"));
    }

    #[test]
    fn test_plain_render() {
        let mut t = Table::new(vec!["A", "B"]);
        t.add_row(vec!["1", "2"]);
        let output = t.render_plain();
        assert!(!output.contains('│'));
        assert!(!output.contains('┌'));
        assert!(output.contains("A"));
        assert!(output.contains("1"));
    }

    #[test]
    fn test_add_row_owned() {
        let mut t = Table::new(vec!["X"]);
        t.add_row_owned(vec!["hello".to_string()]);
        let output = t.render();
        assert!(output.contains("hello"));
    }

    #[test]
    fn test_display_width_plain() {
        assert_eq!(display_width("hello"), 5);
        assert_eq!(display_width(""), 0);
    }

    #[test]
    fn test_display_width_ansi() {
        // ANSI-colored text should measure only the visible chars
        let colored = "\x1b[31mhello\x1b[0m";
        assert_eq!(display_width(colored), 5);
    }
}
