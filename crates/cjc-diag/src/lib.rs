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

/// Renders diagnostics to a human-readable string with source context.
pub struct DiagnosticRenderer<'a> {
    source: &'a str,
    filename: &'a str,
}

impl<'a> DiagnosticRenderer<'a> {
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self { source, filename }
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

        // Header: error[E0001]: message
        out.push_str(&format!(
            "{}[{}]: {}\n",
            severity_str, diag.code, diag.message
        ));

        // Location: --> file:line:col
        out.push_str(&format!(
            "  --> {}:{}:{}\n",
            self.filename, line, col
        ));

        // Source line with underline
        let source_line = self.get_line(line);
        let line_num_width = format!("{}", line).len();
        let padding = " ".repeat(line_num_width);

        out.push_str(&format!("{} |\n", padding));
        out.push_str(&format!("{} | {}\n", line, source_line));

        // Underline
        let underline_start = col - 1;
        let underline_len = (diag.span.end - diag.span.start).max(1);
        out.push_str(&format!(
            "{} | {}{}",
            padding,
            " ".repeat(underline_start),
            "^".repeat(underline_len)
        ));

        // Primary label
        if !diag.labels.is_empty() {
            out.push_str(&format!(" {}", diag.labels[0].message));
        }
        out.push('\n');

        // Additional labels
        for label in diag.labels.iter().skip(1) {
            let (l_line, l_col) = self.offset_to_line_col(label.span.start);
            let l_source = self.get_line(l_line);
            let l_len = (label.span.end - label.span.start).max(1);
            out.push_str(&format!("{} |\n", padding));
            out.push_str(&format!("{} | {}\n", l_line, l_source));
            out.push_str(&format!(
                "{} | {}{} {}\n",
                padding,
                " ".repeat(l_col - 1),
                "^".repeat(l_len),
                label.message
            ));
        }

        // Hints
        for hint in &diag.hints {
            out.push_str(&format!("{} |\n", padding));
            out.push_str(&format!("{} = hint: {}\n", padding, hint));
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
        let renderer = DiagnosticRenderer::new(source, filename);
        let mut out = String::new();
        for diag in &self.diagnostics {
            out.push_str(&renderer.render(diag));
            out.push('\n');
        }
        if self.has_errors() {
            out.push_str(&format!(
                "error: aborting due to {} previous error{}\n",
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
}
