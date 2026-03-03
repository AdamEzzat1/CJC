/// Source span: byte offset range in source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Hint,
}

#[derive(Debug, Clone)]
pub struct Label {
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub span: Span,
    pub labels: Vec<Label>,
    pub hints: Vec<String>,
}

impl Diagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            span,
            labels: Vec::new(),
            hints: Vec::new(),
        }
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            span,
            labels: Vec::new(),
            hints: Vec::new(),
        }
    }

    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
        });
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }
}

// ANSI color code constants.
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const BOLD_RED: &str = "\x1b[1;31m";
const BOLD_YELLOW: &str = "\x1b[1;33m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const BOLD_BLUE: &str = "\x1b[1;34m";
const BOLD_GREEN: &str = "\x1b[1;32m";

/// Renders diagnostics to a human-readable string with source context.
pub struct DiagnosticRenderer<'a> {
    source: &'a str,
    filename: &'a str,
    use_color: bool,
}

impl<'a> DiagnosticRenderer<'a> {
    /// Create a new renderer with color disabled (backward compatible).
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self { source, filename, use_color: false }
    }

    /// Create a new renderer with explicit color control.
    pub fn new_with_color(source: &'a str, filename: &'a str, use_color: bool) -> Self {
        Self { source, filename, use_color }
    }

    /// Wrap `text` with ANSI color codes if color is enabled.
    fn colorize(&self, code: &str, text: &str) -> String {
        if self.use_color {
            format!("{}{}{}", code, text, RESET)
        } else {
            text.to_string()
        }
    }

    /// Convert a byte offset to (line, column), both 1-based.
    fn offset_to_line_col(&self, offset: usize) -> (usize, usize) {
        let mut line = 1;
        let mut col = 1;
        for (i, ch) in self.source.char_indices() {
            if i >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    /// Get the source line (1-based).
    fn get_line(&self, line_num: usize) -> &str {
        self.source.lines().nth(line_num - 1).unwrap_or("")
    }

    pub fn render(&self, diag: &Diagnostic) -> String {
        let mut out = String::new();
        let (line, col) = self.offset_to_line_col(diag.span.start);

        let severity_str = match diag.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Hint => "hint",
        };

        let severity_color = match diag.severity {
            Severity::Error => BOLD_RED,
            Severity::Warning => BOLD_YELLOW,
            Severity::Hint => BOLD_CYAN,
        };

        // Header: error[E0001]: message
        out.push_str(&format!(
            "{}[{}]: {}\n",
            self.colorize(severity_color, severity_str),
            self.colorize(BOLD, &diag.code),
            diag.message
        ));

        // Location: --> file:line:col
        out.push_str(&format!(
            "  {} {}:{}:{}\n",
            self.colorize(BOLD_BLUE, "-->"),
            self.filename, line, col
        ));

        // Source line with underline
        let source_line = self.get_line(line);
        let line_num_width = format!("{}", line).len();
        let padding = " ".repeat(line_num_width);

        out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
        out.push_str(&format!(
            "{} {} {}\n",
            self.colorize(BOLD_BLUE, &format!("{}", line)),
            self.colorize(BOLD_BLUE, "|"),
            source_line
        ));

        // Underline
        let underline_start = col - 1;
        let underline_len = (diag.span.end - diag.span.start).max(1);
        let carets = "^".repeat(underline_len);
        out.push_str(&format!(
            "{} {} {}{}",
            padding,
            self.colorize(BOLD_BLUE, "|"),
            " ".repeat(underline_start),
            self.colorize(BOLD_RED, &carets)
        ));

        // Primary label
        if !diag.labels.is_empty() {
            out.push_str(&format!(" {}", self.colorize(BOLD_RED, &diag.labels[0].message)));
        }
        out.push('\n');

        // Additional labels
        for label in diag.labels.iter().skip(1) {
            let (l_line, l_col) = self.offset_to_line_col(label.span.start);
            let l_source = self.get_line(l_line);
            let l_len = (label.span.end - label.span.start).max(1);
            let l_carets = "^".repeat(l_len);
            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
            out.push_str(&format!(
                "{} {} {}\n",
                self.colorize(BOLD_BLUE, &format!("{}", l_line)),
                self.colorize(BOLD_BLUE, "|"),
                l_source
            ));
            out.push_str(&format!(
                "{} {} {}{} {}\n",
                padding,
                self.colorize(BOLD_BLUE, "|"),
                " ".repeat(l_col - 1),
                self.colorize(BOLD_GREEN, &l_carets),
                self.colorize(BOLD_GREEN, &label.message)
            ));
        }

        // Hints
        for hint in &diag.hints {
            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
            out.push_str(&format!(
                "{} = {}: {}\n",
                padding,
                self.colorize(BOLD_CYAN, "hint"),
                self.colorize(BOLD_CYAN, hint)
            ));
        }

        out
    }
}

/// Collects diagnostics during compilation.
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBag {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    pub fn emit(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    pub fn count(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn truncate(&mut self, len: usize) {
        self.diagnostics.truncate(len);
    }

    pub fn render_all(&self, source: &str, filename: &str) -> String {
        self.render_all_color(source, filename, false)
    }

    pub fn render_all_color(&self, source: &str, filename: &str, use_color: bool) -> String {
        let renderer = DiagnosticRenderer::new_with_color(source, filename, use_color);
        let mut out = String::new();
        for diag in &self.diagnostics {
            out.push_str(&renderer.render(diag));
            out.push('\n');
        }
        if self.has_errors() {
            let prefix = if use_color {
                format!("{}{}{}", BOLD_RED, "error", RESET)
            } else {
                "error".to_string()
            };
            out.push_str(&format!(
                "{}: aborting due to {} previous error{}\n",
                prefix,
                self.error_count(),
                if self.error_count() == 1 { "" } else { "s" }
            ));
        }
        out
    }
}

impl Default for DiagnosticBag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_merge() {
        let a = Span::new(5, 10);
        let b = Span::new(8, 15);
        let merged = a.merge(b);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 15);
    }

    #[test]
    fn test_diagnostic_render() {
        let source = "let x = 42 +;\n";
        let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14))
            .with_label(Span::new(13, 14), "expected expression")
            .with_hint("remove the trailing `+` or add an expression after it");

        let renderer = DiagnosticRenderer::new(source, "test.cjc");
        let output = renderer.render(&diag);

        assert!(output.contains("error[E0001]"));
        assert!(output.contains("unexpected token"));
        assert!(output.contains("test.cjc:1:14"));
        assert!(output.contains("expected expression"));
        assert!(output.contains("hint:"));
    }

    #[test]
    fn test_diagnostic_bag() {
        let mut bag = DiagnosticBag::new();
        assert!(!bag.has_errors());

        bag.emit(Diagnostic::error("E0001", "test error", Span::new(0, 1)));
        assert!(bag.has_errors());
        assert_eq!(bag.error_count(), 1);

        bag.emit(Diagnostic::warning("W0001", "test warning", Span::new(0, 1)));
        assert_eq!(bag.error_count(), 1);
    }

    #[test]
    fn test_diagnostic_render_color() {
        let source = "let x = 42 +;\n";
        let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14))
            .with_label(Span::new(13, 14), "expected expression")
            .with_hint("remove the trailing `+` or add an expression after it");

        let renderer = DiagnosticRenderer::new_with_color(source, "test.cjc", true);
        let output = renderer.render(&diag);

        // Should contain ANSI escape codes
        assert!(output.contains("\x1b["));
        // Should still contain the essential text
        assert!(output.contains("E0001"));
        assert!(output.contains("unexpected token"));
        assert!(output.contains("test.cjc:1:14"));
        assert!(output.contains("expected expression"));
        assert!(output.contains("hint"));
    }

    #[test]
    fn test_render_all_color() {
        let mut bag = DiagnosticBag::new();
        bag.emit(Diagnostic::error("E0001", "test error", Span::new(0, 1)));

        let source = "x";
        let plain = bag.render_all(source, "test.cjc");
        let colored = bag.render_all_color(source, "test.cjc", true);

        // Plain should not contain ANSI escapes
        assert!(!plain.contains("\x1b["));
        // Colored should contain ANSI escapes
        assert!(colored.contains("\x1b["));
    }
}
