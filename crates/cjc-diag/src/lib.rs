pub mod error_codes;

pub use error_codes::ErrorCode;

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

/// A suggested fix: replace the span's content with `replacement`.
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    pub span: Span,
    pub replacement: String,
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
    pub fix_suggestions: Vec<FixSuggestion>,
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
            fix_suggestions: Vec::new(),
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
            fix_suggestions: Vec::new(),
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

    pub fn with_fix(mut self, span: Span, replacement: impl Into<String>, message: impl Into<String>) -> Self {
        self.fix_suggestions.push(FixSuggestion {
            span,
            replacement: replacement.into(),
            message: message.into(),
        });
        self
    }
}

// ── DiagnosticBuilder (fluent API using typed ErrorCode) ─────────────

/// Fluent builder for constructing diagnostics from typed error codes.
pub struct DiagnosticBuilder {
    code: ErrorCode,
    span: Span,
    message: Option<String>,
    labels: Vec<Label>,
    hints: Vec<String>,
    fix_suggestions: Vec<FixSuggestion>,
}

impl DiagnosticBuilder {
    /// Create a new builder from a typed error code and span.
    pub fn new(code: ErrorCode, span: Span) -> Self {
        Self {
            code,
            span,
            message: None,
            labels: Vec::new(),
            hints: Vec::new(),
            fix_suggestions: Vec::new(),
        }
    }

    /// Override the default message template.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Add a label at a specific span.
    pub fn label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: msg.into(),
        });
        self
    }

    /// Add a hint.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    /// Add a fix suggestion.
    pub fn fix(mut self, span: Span, replacement: impl Into<String>, msg: impl Into<String>) -> Self {
        self.fix_suggestions.push(FixSuggestion {
            span,
            replacement: replacement.into(),
            message: msg.into(),
        });
        self
    }

    /// Build the final Diagnostic.
    pub fn build(self) -> Diagnostic {
        let message = self.message.unwrap_or_else(|| self.code.message_template().to_string());
        Diagnostic {
            severity: self.code.severity(),
            code: self.code.code_str().to_string(),
            message,
            span: self.span,
            labels: self.labels,
            hints: self.hints,
            fix_suggestions: self.fix_suggestions,
        }
    }
}

// ── ANSI color code constants ────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const BOLD_RED: &str = "\x1b[1;31m";
const BOLD_YELLOW: &str = "\x1b[1;33m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const BOLD_BLUE: &str = "\x1b[1;34m";
const BOLD_GREEN: &str = "\x1b[1;32m";
const BOLD_MAGENTA: &str = "\x1b[1;35m";

/// Renders diagnostics to a human-readable string with source context.
/// Supports Rust+Elm hybrid style: multi-line spans, fix suggestions,
/// and rich secondary labels.
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
        let (end_line, _end_col) = self.offset_to_line_col(diag.span.end);

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

        // Determine max line number width for alignment
        let max_line = end_line.max(line);
        let line_num_width = format!("{}", max_line).len();
        let padding = " ".repeat(line_num_width);

        // Multi-line span rendering
        if end_line > line {
            // Multi-line: show start line, ellipsis, end line
            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));

            // Start line
            let start_source = self.get_line(line);
            out.push_str(&format!(
                "{} {} {}\n",
                self.colorize(BOLD_BLUE, &format!("{:>width$}", line, width = line_num_width)),
                self.colorize(BOLD_BLUE, "|"),
                start_source
            ));
            let underline_start = col - 1;
            let first_line_len = start_source.len().saturating_sub(underline_start);
            out.push_str(&format!(
                "{} {} {}{}\n",
                padding,
                self.colorize(BOLD_BLUE, "|"),
                " ".repeat(underline_start),
                self.colorize(BOLD_RED, &"^".repeat(first_line_len.max(1)))
            ));

            // Middle lines (show up to 3, then ellipsis)
            let middle_lines = end_line - line - 1;
            if middle_lines > 0 {
                let show = middle_lines.min(3);
                for i in 0..show {
                    let ml = line + 1 + i;
                    let ml_source = self.get_line(ml);
                    out.push_str(&format!(
                        "{} {} {}\n",
                        self.colorize(BOLD_BLUE, &format!("{:>width$}", ml, width = line_num_width)),
                        self.colorize(BOLD_BLUE, "|"),
                        ml_source
                    ));
                }
                if middle_lines > 3 {
                    out.push_str(&format!(
                        "{} {} ...\n",
                        padding,
                        self.colorize(BOLD_BLUE, "|")
                    ));
                }
            }

            // End line
            if end_line != line {
                let end_source = self.get_line(end_line);
                out.push_str(&format!(
                    "{} {} {}\n",
                    self.colorize(BOLD_BLUE, &format!("{:>width$}", end_line, width = line_num_width)),
                    self.colorize(BOLD_BLUE, "|"),
                    end_source
                ));
            }
        } else {
            // Single-line span rendering (original logic)
            let source_line = self.get_line(line);

            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
            out.push_str(&format!(
                "{} {} {}\n",
                self.colorize(BOLD_BLUE, &format!("{:>width$}", line, width = line_num_width)),
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
        }

        // Additional labels
        for label in diag.labels.iter().skip(if end_line > line { 0 } else { 1 }) {
            let (l_line, l_col) = self.offset_to_line_col(label.span.start);
            let l_source = self.get_line(l_line);
            let l_len = (label.span.end - label.span.start).max(1);
            let l_carets = "^".repeat(l_len);
            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
            out.push_str(&format!(
                "{} {} {}\n",
                self.colorize(BOLD_BLUE, &format!("{:>width$}", l_line, width = line_num_width)),
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

        // Fix suggestions (Elm-style)
        for fix in &diag.fix_suggestions {
            let (f_line, f_col) = self.offset_to_line_col(fix.span.start);
            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
            out.push_str(&format!(
                "{} = {}: {}\n",
                padding,
                self.colorize(BOLD_MAGENTA, "fix"),
                self.colorize(BOLD_MAGENTA, &fix.message)
            ));
            let fix_source = self.get_line(f_line);
            // Show the original line
            let fix_start = f_col - 1;
            let fix_end = fix_start + (fix.span.end - fix.span.start);
            // Build the suggested replacement line
            let mut suggested = String::new();
            suggested.push_str(&fix_source[..fix_start.min(fix_source.len())]);
            suggested.push_str(&fix.replacement);
            if fix_end < fix_source.len() {
                suggested.push_str(&fix_source[fix_end..]);
            }
            out.push_str(&format!(
                "{} {} {}\n",
                self.colorize(BOLD_BLUE, &format!("{:>width$}", f_line, width = line_num_width)),
                self.colorize(BOLD_BLUE, "|"),
                self.colorize(BOLD_GREEN, &suggested)
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

    /// Emit a diagnostic built from a typed ErrorCode.
    pub fn emit_coded(&mut self, builder: DiagnosticBuilder) {
        self.diagnostics.push(builder.build());
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

    #[test]
    fn test_diagnostic_builder() {
        let diag = DiagnosticBuilder::new(ErrorCode::E1000, Span::new(5, 10))
            .message("unexpected `}` here")
            .label(Span::new(5, 6), "this `}`")
            .hint("did you forget to close a previous block?")
            .build();

        assert_eq!(diag.code, "E1000");
        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.message, "unexpected `}` here");
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.hints.len(), 1);
    }

    #[test]
    fn test_diagnostic_builder_default_message() {
        let diag = DiagnosticBuilder::new(ErrorCode::E2001, Span::new(0, 1)).build();
        assert_eq!(diag.message, "type mismatch");
        assert_eq!(diag.code, "E2001");
    }

    #[test]
    fn test_fix_suggestion_render() {
        let source = "let x: i32 = \"hello\";\n";
        let diag = Diagnostic::error("E2001", "type mismatch", Span::new(14, 21))
            .with_label(Span::new(14, 21), "expected `i32`, found `str`")
            .with_fix(Span::new(7, 10), "str", "change type annotation to `str`");

        let renderer = DiagnosticRenderer::new(source, "test.cjc");
        let output = renderer.render(&diag);

        assert!(output.contains("fix:"));
        assert!(output.contains("change type annotation to `str`"));
    }

    #[test]
    fn test_warning_builder() {
        let diag = DiagnosticBuilder::new(ErrorCode::W0001, Span::new(4, 5)).build();
        assert_eq!(diag.severity, Severity::Warning);
        assert_eq!(diag.code, "W0001");
        assert_eq!(diag.message, "unused variable");
    }

    #[test]
    fn test_emit_coded() {
        let mut bag = DiagnosticBag::new();
        bag.emit_coded(
            DiagnosticBuilder::new(ErrorCode::E4001, Span::new(0, 5))
                .label(Span::new(0, 5), "this calls gc_alloc")
        );
        assert!(bag.has_errors());
        assert_eq!(bag.diagnostics[0].code, "E4001");
    }
}
