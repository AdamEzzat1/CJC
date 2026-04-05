//! Diagnostic infrastructure for the CJC compiler.
//!
//! This crate provides the core building blocks for reporting errors, warnings,
//! and hints throughout the CJC compilation pipeline (lexer through MIR execution).
//!
//! # Key types
//!
//! - [`Span`] — byte-offset range identifying a region in source code.
//! - [`SourceMap`] — precomputed line-start table for O(log n) byte-offset to
//!   (line, column) lookups.
//! - [`Diagnostic`] — a single error, warning, or hint with labels, hints, and
//!   optional fix suggestions.
//! - [`DiagnosticBag`] — growable collection of diagnostics with configurable
//!   error-count limits and batch rendering.
//! - [`DiagnosticBuilder`] — fluent builder for constructing diagnostics from
//!   typed [`ErrorCode`] values.
//! - [`DiagnosticRenderer`] — renders diagnostics to human-readable strings
//!   with source context, underlines, and optional ANSI color.
//! - [`ErrorCode`] — typed error code enum covering the entire compiler
//!   pipeline (E0xxx through E9xxx, W0xxx).
//! - [`Severity`] — error, warning, or hint classification.
//! - [`DiagnosticFormat`] — rich (multi-line with source context) or short
//!   (machine-readable one-liner) output format.

pub mod error_codes;

pub use error_codes::ErrorCode;

/// A half-open byte-offset range `[start, end)` identifying a region in source code.
///
/// Spans are used throughout the compiler to track the origin of tokens, AST nodes,
/// and diagnostics back to the original source text.
///
/// # Fields
///
/// * `start` — inclusive byte offset of the first character.
/// * `end` — exclusive byte offset one past the last character.
///
/// # Examples
///
/// ```
/// use cjc_diag::Span;
///
/// let span = Span::new(0, 5);
/// assert_eq!(span.start, 0);
/// assert_eq!(span.end, 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Inclusive byte offset of the first character in the spanned region.
    pub start: usize,
    /// Exclusive byte offset one past the last character in the spanned region.
    pub end: usize,
}

impl Span {
    /// Creates a new span covering the byte range `[start, end)`.
    ///
    /// # Arguments
    ///
    /// * `start` — inclusive byte offset of the first character.
    /// * `end` — exclusive byte offset one past the last character.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Returns the smallest span that covers both `self` and `other`.
    ///
    /// The resulting span starts at `min(self.start, other.start)` and ends at
    /// `max(self.end, other.end)`.
    ///
    /// # Arguments
    ///
    /// * `other` — the span to merge with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cjc_diag::Span;
    ///
    /// let merged = Span::new(5, 10).merge(Span::new(8, 15));
    /// assert_eq!(merged, Span::new(5, 15));
    /// ```
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Creates a zero-length dummy span at offset 0.
    ///
    /// Useful as a placeholder when no meaningful source location is available
    /// (e.g., compiler-generated nodes or test scaffolding).
    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

// ── SourceMap: precomputed line-start table for O(log n) lookups ─────

/// Precomputed line-start table for O(log n) byte-offset to (line, column) lookups.
///
/// `SourceMap` scans the source text once at construction time, recording the byte
/// offset of every line start. Subsequent lookups use binary search over this table,
/// giving O(log n) performance where n is the number of lines.
///
/// All line and column numbers returned by this type are **1-based**.
///
/// # Examples
///
/// ```
/// use cjc_diag::SourceMap;
///
/// let smap = SourceMap::new("hello\nworld\n");
/// assert_eq!(smap.offset_to_line_col(0), (1, 1));   // 'h'
/// assert_eq!(smap.offset_to_line_col(6), (2, 1));   // 'w'
/// assert_eq!(smap.line_count(), 3);
/// assert_eq!(smap.get_line(1), "hello");
/// ```
#[derive(Debug, Clone)]
pub struct SourceMap<'a> {
    source: &'a str,
    /// Byte offset of each line start. `line_starts[0]` is always 0.
    line_starts: Vec<usize>,
}

impl<'a> SourceMap<'a> {
    /// Builds a [`SourceMap`] by scanning `source` once for newline positions.
    ///
    /// # Arguments
    ///
    /// * `source` — the full source text to index. The reference must outlive
    ///   the returned `SourceMap`.
    pub fn new(source: &'a str) -> Self {
        let mut line_starts = vec![0usize];
        for (i, b) in source.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self { source, line_starts }
    }

    /// Converts a byte offset to a 1-based `(line, column)` pair.
    ///
    /// Uses binary search over the precomputed line-start table for O(log n)
    /// performance. If `offset` exceeds the source length it is clamped to
    /// `source.len()`.
    ///
    /// Columns are counted in **characters** (not bytes), so multi-byte UTF-8
    /// sequences count as a single column.
    ///
    /// # Arguments
    ///
    /// * `offset` — byte offset into the source text.
    ///
    /// # Returns
    ///
    /// A `(line, column)` tuple, both 1-based.
    pub fn offset_to_line_col(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.source.len());
        // binary search: find the last line_start <= offset
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(exact) => exact,
            Err(insert) => insert.saturating_sub(1),
        };
        let line = line_idx + 1; // 1-based
        let line_start = self.line_starts[line_idx];
        // Column = count characters (not bytes) from line start to offset
        let col = self.source[line_start..offset].chars().count() + 1;
        (line, col)
    }

    /// Returns the source text for a given 1-based line number.
    ///
    /// Trailing newline (`\n`) and carriage-return (`\r`) characters are stripped
    /// from the returned slice. Returns an empty string if `line_num` is 0 or
    /// exceeds the total number of lines.
    ///
    /// # Arguments
    ///
    /// * `line_num` — 1-based line number.
    pub fn get_line(&self, line_num: usize) -> &str {
        if line_num == 0 || line_num > self.line_starts.len() {
            return "";
        }
        let start = self.line_starts[line_num - 1];
        let end = if line_num < self.line_starts.len() {
            // strip the trailing newline
            self.line_starts[line_num].saturating_sub(1)
        } else {
            self.source.len()
        };
        // Handle \r\n line endings
        let slice = &self.source[start..end];
        slice.strip_suffix('\r').unwrap_or(slice)
    }

    /// Returns the total number of lines in the source.
    ///
    /// A trailing newline adds an extra (empty) logical line, matching the
    /// convention used by most text editors.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Returns the underlying source text that this map indexes.
    pub fn source(&self) -> &str {
        self.source
    }
}

/// Classification of a diagnostic's importance.
///
/// Determines how the diagnostic is rendered (color, prefix) and whether it
/// causes compilation to abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A fatal problem that prevents successful compilation.
    Error,
    /// A potential issue that does not prevent compilation.
    Warning,
    /// An informational suggestion for the user.
    Hint,
}

/// A secondary annotation attached to a [`Diagnostic`].
///
/// Labels mark additional source regions relevant to the primary diagnostic
/// (e.g., "expected type declared here" or "conflicting definition found here").
/// The [`DiagnosticRenderer`] underlines each label and displays its message
/// next to the source context.
#[derive(Debug, Clone)]
pub struct Label {
    /// Source region this label highlights.
    pub span: Span,
    /// Explanatory text displayed alongside the underline.
    pub message: String,
}

/// A machine-applicable fix suggestion attached to a [`Diagnostic`].
///
/// Encodes a textual replacement: the source text covered by [`span`](Self::span)
/// should be replaced with [`replacement`](Self::replacement). The renderer shows
/// the suggested line with the replacement applied, Elm-style.
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    /// Source region to be replaced.
    pub span: Span,
    /// Text that should replace the source region.
    pub replacement: String,
    /// Human-readable description of the fix (e.g., "change type annotation to `str`").
    pub message: String,
}

/// A single compiler diagnostic (error, warning, or hint).
///
/// A `Diagnostic` bundles a severity level, an error code string, a primary
/// message, the originating [`Span`], and optional secondary [`Label`]s,
/// free-text hints, and machine-applicable [`FixSuggestion`]s.
///
/// Construct diagnostics via the convenience constructors [`Diagnostic::error`]
/// and [`Diagnostic::warning`], or use [`DiagnosticBuilder`] for typed
/// [`ErrorCode`]-based construction.
///
/// # Examples
///
/// ```
/// use cjc_diag::{Diagnostic, Span};
///
/// let diag = Diagnostic::error("E1000", "unexpected token", Span::new(5, 6))
///     .with_label(Span::new(5, 6), "found `}`")
///     .with_hint("did you forget to close a previous block?");
/// ```
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// How severe this diagnostic is.
    pub severity: Severity,
    /// Error code string (e.g., `"E1000"`, `"W0001"`).
    pub code: String,
    /// Primary human-readable message.
    pub message: String,
    /// Source region this diagnostic originates from.
    pub span: Span,
    /// Secondary annotations highlighting related source regions.
    pub labels: Vec<Label>,
    /// Free-text hints offering guidance to the user.
    pub hints: Vec<String>,
    /// Machine-applicable replacement suggestions.
    pub fix_suggestions: Vec<FixSuggestion>,
    /// Source filename for multi-file diagnostics. `None` = single-file context.
    pub filename: Option<String>,
}

impl Diagnostic {
    /// Creates an error-severity diagnostic.
    ///
    /// # Arguments
    ///
    /// * `code` — error code string (e.g., `"E1000"`).
    /// * `message` — primary human-readable message.
    /// * `span` — source region this error originates from.
    pub fn error(code: impl Into<String>, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            span,
            labels: Vec::new(),
            hints: Vec::new(),
            fix_suggestions: Vec::new(),
            filename: None,
        }
    }

    /// Creates a warning-severity diagnostic.
    ///
    /// # Arguments
    ///
    /// * `code` — error code string (e.g., `"W0001"`).
    /// * `message` — primary human-readable message.
    /// * `span` — source region this warning originates from.
    pub fn warning(code: impl Into<String>, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            span,
            labels: Vec::new(),
            hints: Vec::new(),
            fix_suggestions: Vec::new(),
            filename: None,
        }
    }

    /// Attaches a source filename for multi-file diagnostics.
    ///
    /// When set, the [`DiagnosticRenderer`] uses this filename instead of its
    /// default, allowing a single renderer to display diagnostics from multiple
    /// source files.
    ///
    /// # Arguments
    ///
    /// * `name` — the filename to display (e.g., `"math.cjc"`).
    pub fn with_filename(mut self, name: impl Into<String>) -> Self {
        self.filename = Some(name.into());
        self
    }

    /// Adds a secondary label highlighting an additional source region.
    ///
    /// # Arguments
    ///
    /// * `span` — source region to underline.
    /// * `message` — explanatory text displayed alongside the underline.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
        });
        self
    }

    /// Adds a free-text hint offering guidance to the user.
    ///
    /// # Arguments
    ///
    /// * `hint` — the hint text (e.g., "did you mean `foo`?").
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    /// Adds a machine-applicable fix suggestion.
    ///
    /// The renderer shows the source line with the replacement applied,
    /// Elm-style.
    ///
    /// # Arguments
    ///
    /// * `span` — source region to replace.
    /// * `replacement` — text that should replace the spanned region.
    /// * `message` — human-readable description of the fix.
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

/// Fluent builder for constructing [`Diagnostic`]s from typed [`ErrorCode`] values.
///
/// Unlike the `Diagnostic::error` / `Diagnostic::warning` constructors, the
/// builder derives severity and a default message template automatically from
/// the [`ErrorCode`]. Call [`build`](Self::build) to finalize.
///
/// # Examples
///
/// ```
/// use cjc_diag::{DiagnosticBuilder, ErrorCode, Span};
///
/// let diag = DiagnosticBuilder::new(ErrorCode::E2001, Span::new(10, 15))
///     .message("expected `i64`, found `str`")
///     .label(Span::new(10, 15), "this expression")
///     .hint("consider converting the type")
///     .build();
///
/// assert_eq!(diag.code, "E2001");
/// ```
pub struct DiagnosticBuilder {
    code: ErrorCode,
    span: Span,
    message: Option<String>,
    labels: Vec<Label>,
    hints: Vec<String>,
    fix_suggestions: Vec<FixSuggestion>,
    filename: Option<String>,
}

impl DiagnosticBuilder {
    /// Creates a new builder from a typed [`ErrorCode`] and source [`Span`].
    ///
    /// The builder starts with the code's default message template
    /// (see [`ErrorCode::message_template`]) and severity
    /// (see [`ErrorCode::severity`]). Both can be overridden before calling
    /// [`build`](Self::build).
    ///
    /// # Arguments
    ///
    /// * `code` — the typed error code.
    /// * `span` — source region this diagnostic originates from.
    pub fn new(code: ErrorCode, span: Span) -> Self {
        Self {
            code,
            span,
            message: None,
            labels: Vec::new(),
            hints: Vec::new(),
            fix_suggestions: Vec::new(),
            filename: None,
        }
    }

    /// Overrides the default message template from [`ErrorCode::message_template`].
    ///
    /// # Arguments
    ///
    /// * `msg` — custom message text.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Adds a secondary label at a specific source [`Span`].
    ///
    /// # Arguments
    ///
    /// * `span` — source region to underline.
    /// * `msg` — explanatory text displayed alongside the underline.
    pub fn label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: msg.into(),
        });
        self
    }

    /// Adds a free-text hint.
    ///
    /// # Arguments
    ///
    /// * `hint` — the hint text (e.g., "try using `as` to cast").
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    /// Adds a machine-applicable fix suggestion.
    ///
    /// # Arguments
    ///
    /// * `span` — source region to replace.
    /// * `replacement` — text that should replace the spanned region.
    /// * `msg` — human-readable description of the fix.
    pub fn fix(mut self, span: Span, replacement: impl Into<String>, msg: impl Into<String>) -> Self {
        self.fix_suggestions.push(FixSuggestion {
            span,
            replacement: replacement.into(),
            message: msg.into(),
        });
        self
    }

    /// Attaches a source filename for multi-file diagnostics.
    ///
    /// See [`Diagnostic::with_filename`] for details.
    ///
    /// # Arguments
    ///
    /// * `name` — the filename to display.
    pub fn filename(mut self, name: impl Into<String>) -> Self {
        self.filename = Some(name.into());
        self
    }

    /// Consumes the builder and returns the finalized [`Diagnostic`].
    ///
    /// If no custom message was set via [`message`](Self::message), the
    /// [`ErrorCode::message_template`] is used as the diagnostic message.
    /// Severity is always derived from [`ErrorCode::severity`].
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
            filename: self.filename,
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

/// Controls the output format used by [`DiagnosticRenderer`] and
/// [`DiagnosticBag::render_all_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticFormat {
    /// Rich output with source context, underlines, labels, hints, and fixes.
    Rich,
    /// Machine-readable one-liner: `file:line:col: severity[CODE]: message`
    /// Parseable by VS Code, grep, and other tools.
    Short,
}

/// Renders [`Diagnostic`]s to human-readable strings with source context.
///
/// Output follows a Rust+Elm hybrid style: a machine-readable header line,
/// source context with underlines and carets, secondary labels, Elm-style fix
/// suggestions, and free-text hints. Supports both [`Rich`](DiagnosticFormat::Rich)
/// (multi-line with context) and [`Short`](DiagnosticFormat::Short) (single-line,
/// machine-parseable) formats.
///
/// ANSI color output is optional and controlled at construction time.
pub struct DiagnosticRenderer<'a> {
    smap: SourceMap<'a>,
    filename: &'a str,
    use_color: bool,
    format: DiagnosticFormat,
}

impl<'a> DiagnosticRenderer<'a> {
    /// Creates a new renderer with color disabled and [`Rich`](DiagnosticFormat::Rich) format.
    ///
    /// This is the backward-compatible default constructor.
    ///
    /// # Arguments
    ///
    /// * `source` — full source text used for context snippets.
    /// * `filename` — default filename shown in diagnostic headers.
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self { smap: SourceMap::new(source), filename, use_color: false, format: DiagnosticFormat::Rich }
    }

    /// Creates a new renderer with explicit color control and [`Rich`](DiagnosticFormat::Rich) format.
    ///
    /// # Arguments
    ///
    /// * `source` — full source text used for context snippets.
    /// * `filename` — default filename shown in diagnostic headers.
    /// * `use_color` — when `true`, ANSI escape codes are emitted for severity
    ///   colors, line-number gutters, and underlines.
    pub fn new_with_color(source: &'a str, filename: &'a str, use_color: bool) -> Self {
        Self { smap: SourceMap::new(source), filename, use_color, format: DiagnosticFormat::Rich }
    }

    /// Creates a new renderer with full configuration.
    ///
    /// # Arguments
    ///
    /// * `source` — full source text used for context snippets.
    /// * `filename` — default filename shown in diagnostic headers.
    /// * `use_color` — when `true`, ANSI escape codes are emitted.
    /// * `format` — [`Rich`](DiagnosticFormat::Rich) for multi-line output or
    ///   [`Short`](DiagnosticFormat::Short) for machine-readable one-liners.
    pub fn new_with_options(source: &'a str, filename: &'a str, use_color: bool, format: DiagnosticFormat) -> Self {
        Self { smap: SourceMap::new(source), filename, use_color, format }
    }

    /// Wrap `text` with ANSI color codes if color is enabled.
    fn colorize(&self, code: &str, text: &str) -> String {
        if self.use_color {
            format!("{}{}{}", code, text, RESET)
        } else {
            text.to_string()
        }
    }

    /// Renders a single diagnostic in the machine-readable short format.
    ///
    /// Output is a single line: `file:line:col: severity[CODE]: message\n`.
    /// This format is parseable by VS Code, grep, and similar tools.
    ///
    /// # Arguments
    ///
    /// * `diag` — the diagnostic to render.
    pub fn render_short(&self, diag: &Diagnostic) -> String {
        let (line, col) = self.smap.offset_to_line_col(diag.span.start);
        let display_filename = diag.filename.as_deref().unwrap_or(self.filename);
        let severity_str = match diag.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Hint => "hint",
        };
        format!(
            "{}:{}:{}: {}[{}]: {}\n",
            display_filename, line, col, severity_str, diag.code, diag.message
        )
    }

    /// Renders a single diagnostic according to the configured [`DiagnosticFormat`].
    ///
    /// In [`Rich`](DiagnosticFormat::Rich) mode the output includes:
    /// - A machine-readable header line (`file:line:col: severity[CODE]: message`).
    /// - Source context with line-number gutters and caret underlines.
    /// - Secondary labels, fix suggestions, and hints.
    ///
    /// In [`Short`](DiagnosticFormat::Short) mode, delegates to
    /// [`render_short`](Self::render_short).
    ///
    /// # Arguments
    ///
    /// * `diag` — the diagnostic to render.
    pub fn render(&self, diag: &Diagnostic) -> String {
        if self.format == DiagnosticFormat::Short {
            return self.render_short(diag);
        }

        let mut out = String::new();
        let (line, col) = self.smap.offset_to_line_col(diag.span.start);
        let (end_line, _end_col) = self.smap.offset_to_line_col(diag.span.end);

        // Use the diagnostic's per-file filename if present, otherwise fall back
        // to the renderer's default filename.
        let display_filename = diag.filename.as_deref().unwrap_or(self.filename);

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

        // Machine-readable first line (VS Code / terminal link compatible)
        // Format: file:line:col: severity[CODE]: message
        out.push_str(&format!(
            "{}:{}:{}: {}[{}]: {}\n",
            display_filename, line, col,
            self.colorize(severity_color, severity_str),
            self.colorize(BOLD, &diag.code),
            diag.message
        ));

        // Location: --> file:line:col
        out.push_str(&format!(
            "  {} {}:{}:{}\n",
            self.colorize(BOLD_BLUE, "-->"),
            display_filename, line, col
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
            let start_source = self.smap.get_line(line);
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
                    let ml_source = self.smap.get_line(ml);
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
                let end_source = self.smap.get_line(end_line);
                out.push_str(&format!(
                    "{} {} {}\n",
                    self.colorize(BOLD_BLUE, &format!("{:>width$}", end_line, width = line_num_width)),
                    self.colorize(BOLD_BLUE, "|"),
                    end_source
                ));
            }
        } else {
            // Single-line span rendering (original logic)
            let source_line = self.smap.get_line(line);

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
            let (l_line, l_col) = self.smap.offset_to_line_col(label.span.start);
            let l_source = self.smap.get_line(l_line);
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
            let (f_line, f_col) = self.smap.offset_to_line_col(fix.span.start);
            out.push_str(&format!("{} {}\n", padding, self.colorize(BOLD_BLUE, "|")));
            out.push_str(&format!(
                "{} = {}: {}\n",
                padding,
                self.colorize(BOLD_MAGENTA, "fix"),
                self.colorize(BOLD_MAGENTA, &fix.message)
            ));
            let fix_source = self.smap.get_line(f_line);
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

/// Growable collection of [`Diagnostic`]s emitted during compilation.
///
/// `DiagnosticBag` is the primary accumulator passed through every compiler
/// phase. It supports an optional error-count limit to suppress cascading
/// errors, and provides batch rendering to formatted strings.
///
/// # Examples
///
/// ```
/// use cjc_diag::{DiagnosticBag, Diagnostic, Span};
///
/// let mut bag = DiagnosticBag::new();
/// bag.emit(Diagnostic::error("E1000", "unexpected token", Span::new(5, 6)));
/// assert!(bag.has_errors());
/// assert_eq!(bag.error_count(), 1);
/// ```
pub struct DiagnosticBag {
    /// The accumulated diagnostics (errors, warnings, hints).
    pub diagnostics: Vec<Diagnostic>,
    /// Maximum number of error-severity diagnostics before further errors are
    /// suppressed. Default: 50. Set to 0 for unlimited.
    pub error_limit: usize,
}

impl DiagnosticBag {
    /// Creates an empty diagnostic bag with the default error limit of 50.
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            error_limit: 50,
        }
    }

    /// Appends a diagnostic to the bag.
    ///
    /// If the diagnostic has [`Severity::Error`] and the current error count
    /// has already reached [`error_limit`](Self::error_limit) (and the limit
    /// is non-zero), the diagnostic is silently dropped to prevent cascading
    /// error floods.
    ///
    /// # Arguments
    ///
    /// * `diag` — the diagnostic to emit.
    pub fn emit(&mut self, diag: Diagnostic) {
        if diag.severity == Severity::Error && self.error_limit > 0 {
            let current_errors = self.error_count();
            if current_errors >= self.error_limit {
                return; // suppress cascading errors beyond limit
            }
        }
        self.diagnostics.push(diag);
    }

    /// Builds and emits a diagnostic from a [`DiagnosticBuilder`].
    ///
    /// Equivalent to `self.emit(builder.build())`.
    ///
    /// # Arguments
    ///
    /// * `builder` — the builder to finalize and emit.
    pub fn emit_coded(&mut self, builder: DiagnosticBuilder) {
        let diag = builder.build();
        self.emit(diag);
    }

    /// Returns `true` if the bag contains at least one error-severity diagnostic.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    /// Returns the number of error-severity diagnostics in the bag.
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    /// Returns the total number of diagnostics (all severities) in the bag.
    pub fn count(&self) -> usize {
        self.diagnostics.len()
    }

    /// Truncates the diagnostic list to at most `len` entries.
    ///
    /// Useful for rolling back speculative parse attempts that may have
    /// emitted tentative diagnostics.
    ///
    /// # Arguments
    ///
    /// * `len` — the maximum number of diagnostics to keep.
    pub fn truncate(&mut self, len: usize) {
        self.diagnostics.truncate(len);
    }

    /// Returns the number of warning-severity diagnostics in the bag.
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count()
    }

    /// Renders all diagnostics in [`Rich`](DiagnosticFormat::Rich) format without color.
    ///
    /// Shorthand for `render_all_color(source, filename, false)`.
    ///
    /// # Arguments
    ///
    /// * `source` — full source text for context snippets.
    /// * `filename` — filename shown in diagnostic headers.
    pub fn render_all(&self, source: &str, filename: &str) -> String {
        self.render_all_color(source, filename, false)
    }

    /// Renders all diagnostics in [`Rich`](DiagnosticFormat::Rich) format with optional color.
    ///
    /// # Arguments
    ///
    /// * `source` — full source text for context snippets.
    /// * `filename` — filename shown in diagnostic headers.
    /// * `use_color` — when `true`, ANSI escape codes are emitted.
    pub fn render_all_color(&self, source: &str, filename: &str, use_color: bool) -> String {
        self.render_all_with_options(source, filename, use_color, DiagnosticFormat::Rich)
    }

    /// Renders all diagnostics in [`Short`](DiagnosticFormat::Short) format without color.
    ///
    /// Each diagnostic produces a single machine-readable line. Includes a
    /// summary footer.
    ///
    /// # Arguments
    ///
    /// * `source` — full source text (used for line/column computation).
    /// * `filename` — filename shown in each output line.
    pub fn render_all_short(&self, source: &str, filename: &str) -> String {
        self.render_all_with_options(source, filename, false, DiagnosticFormat::Short)
    }

    /// Renders all diagnostics with full control over color and format.
    ///
    /// Iterates over every diagnostic in the bag, renders it with a
    /// [`DiagnosticRenderer`], and appends a summary footer line
    /// (e.g., "error: aborting due to 3 errors; 1 warning").
    ///
    /// # Arguments
    ///
    /// * `source` — full source text for context snippets.
    /// * `filename` — default filename shown in diagnostic headers.
    /// * `use_color` — when `true`, ANSI escape codes are emitted.
    /// * `format` — [`Rich`](DiagnosticFormat::Rich) or
    ///   [`Short`](DiagnosticFormat::Short).
    pub fn render_all_with_options(
        &self,
        source: &str,
        filename: &str,
        use_color: bool,
        format: DiagnosticFormat,
    ) -> String {
        let renderer = DiagnosticRenderer::new_with_options(source, filename, use_color, format);
        let mut out = String::new();
        for diag in &self.diagnostics {
            out.push_str(&renderer.render(diag));
            if format == DiagnosticFormat::Rich {
                out.push('\n');
            }
        }
        // Summary footer
        let errs = self.error_count();
        let warns = self.warning_count();
        if errs > 0 || warns > 0 {
            let colorize = |color: &str, text: &str| -> String {
                if use_color { format!("{}{}{}", color, text, RESET) } else { text.to_string() }
            };
            let mut parts = Vec::new();
            if errs > 0 {
                parts.push(format!(
                    "{} error{}",
                    errs, if errs == 1 { "" } else { "s" }
                ));
            }
            if warns > 0 {
                parts.push(format!(
                    "{} warning{}",
                    warns, if warns == 1 { "" } else { "s" }
                ));
            }
            if errs > 0 {
                out.push_str(&format!(
                    "{}: aborting due to {}\n",
                    colorize(BOLD_RED, "error"),
                    parts.join("; ")
                ));
            } else {
                out.push_str(&format!(
                    "{}: generated {}\n",
                    colorize(BOLD_YELLOW, "warning"),
                    parts.join("; ")
                ));
            }
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

    #[test]
    fn test_diagnostic_filename_default_none() {
        let diag = Diagnostic::error("E0001", "test", Span::new(0, 1));
        assert!(diag.filename.is_none());

        let diag = Diagnostic::warning("W0001", "test", Span::new(0, 1));
        assert!(diag.filename.is_none());
    }

    #[test]
    fn test_diagnostic_with_filename() {
        let diag = Diagnostic::error("E0001", "test", Span::new(0, 1))
            .with_filename("math.cjc");
        assert_eq!(diag.filename.as_deref(), Some("math.cjc"));
    }

    #[test]
    fn test_render_uses_diagnostic_filename() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("E0001", "test error", Span::new(0, 3))
            .with_filename("other_file.cjc");

        // Renderer has "main.cjc" but the diagnostic overrides to "other_file.cjc"
        let renderer = DiagnosticRenderer::new(source, "main.cjc");
        let output = renderer.render(&diag);
        assert!(output.contains("other_file.cjc:1:1"));
        assert!(!output.contains("main.cjc"));
    }

    #[test]
    fn test_render_falls_back_to_renderer_filename() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("E0001", "test error", Span::new(0, 3));
        // No filename on diagnostic => falls back to renderer's filename
        assert!(diag.filename.is_none());

        let renderer = DiagnosticRenderer::new(source, "main.cjc");
        let output = renderer.render(&diag);
        assert!(output.contains("main.cjc:1:1"));
    }

    #[test]
    fn test_builder_filename() {
        let diag = DiagnosticBuilder::new(ErrorCode::E1000, Span::new(0, 1))
            .filename("module.cjc")
            .build();
        assert_eq!(diag.filename.as_deref(), Some("module.cjc"));
    }

    #[test]
    fn test_builder_default_no_filename() {
        let diag = DiagnosticBuilder::new(ErrorCode::E1000, Span::new(0, 1)).build();
        assert!(diag.filename.is_none());
    }

    // ── SourceMap tests ──────────────────────────────────────────────

    #[test]
    fn test_source_map_single_line() {
        let smap = SourceMap::new("hello world");
        assert_eq!(smap.line_count(), 1);
        assert_eq!(smap.offset_to_line_col(0), (1, 1));
        assert_eq!(smap.offset_to_line_col(5), (1, 6));
        assert_eq!(smap.get_line(1), "hello world");
    }

    #[test]
    fn test_source_map_multi_line() {
        let src = "line one\nline two\nline three\n";
        let smap = SourceMap::new(src);
        assert_eq!(smap.line_count(), 4); // trailing \n creates a 4th empty line start
        assert_eq!(smap.offset_to_line_col(0), (1, 1));
        assert_eq!(smap.offset_to_line_col(9), (2, 1)); // 'l' of "line two"
        assert_eq!(smap.offset_to_line_col(14), (2, 6)); // 't' of "two"
        assert_eq!(smap.get_line(1), "line one");
        assert_eq!(smap.get_line(2), "line two");
        assert_eq!(smap.get_line(3), "line three");
    }

    #[test]
    fn test_source_map_empty_source() {
        let smap = SourceMap::new("");
        assert_eq!(smap.line_count(), 1);
        assert_eq!(smap.offset_to_line_col(0), (1, 1));
        assert_eq!(smap.get_line(1), "");
    }

    #[test]
    fn test_source_map_offset_at_newline() {
        let src = "ab\ncd\n";
        let smap = SourceMap::new(src);
        // offset 2 is '\n' — still line 1
        assert_eq!(smap.offset_to_line_col(2), (1, 3));
        // offset 3 is 'c' — line 2
        assert_eq!(smap.offset_to_line_col(3), (2, 1));
    }

    #[test]
    fn test_source_map_crlf() {
        let src = "abc\r\ndef\r\n";
        let smap = SourceMap::new(src);
        assert_eq!(smap.get_line(1), "abc");
        assert_eq!(smap.get_line(2), "def");
    }

    #[test]
    fn test_source_map_out_of_bounds_line() {
        let smap = SourceMap::new("hello");
        assert_eq!(smap.get_line(0), ""); // 0 is out of range
        assert_eq!(smap.get_line(99), ""); // way out of range
    }

    // ── Machine-readable (short) format tests ────────────────────────

    #[test]
    fn test_render_short_format() {
        let source = "let x = 42 +;\n";
        let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14));

        let renderer = DiagnosticRenderer::new_with_options(
            source, "test.cjc", false, DiagnosticFormat::Short
        );
        let output = renderer.render(&diag);
        assert_eq!(output, "test.cjc:1:14: error[E0001]: unexpected token\n");
    }

    #[test]
    fn test_render_short_warning() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::warning("W0001", "unused variable", Span::new(4, 5));

        let renderer = DiagnosticRenderer::new_with_options(
            source, "test.cjc", false, DiagnosticFormat::Short
        );
        let output = renderer.render(&diag);
        assert_eq!(output, "test.cjc:1:5: warning[W0001]: unused variable\n");
    }

    #[test]
    fn test_render_short_multi_file() {
        let source = "let x = 1;\n";
        let diag = Diagnostic::error("E0001", "test", Span::new(0, 3))
            .with_filename("other.cjc");

        let renderer = DiagnosticRenderer::new_with_options(
            source, "main.cjc", false, DiagnosticFormat::Short
        );
        let output = renderer.render(&diag);
        assert!(output.starts_with("other.cjc:"));
        assert!(!output.contains("main.cjc"));
    }

    #[test]
    fn test_render_all_short() {
        let mut bag = DiagnosticBag::new();
        bag.emit(Diagnostic::error("E0001", "bad token", Span::new(0, 1)));
        bag.emit(Diagnostic::error("E1000", "unexpected", Span::new(5, 6)));

        let output = bag.render_all_short("abcdefgh", "test.cjc");
        assert!(output.contains("test.cjc:1:1: error[E0001]: bad token\n"));
        assert!(output.contains("test.cjc:1:6: error[E1000]: unexpected\n"));
        assert!(output.contains("aborting due to 2 errors"));
    }

    // ── Rich format machine-readable first line tests ────────────────

    #[test]
    fn test_rich_format_has_machine_line() {
        let source = "let x = 42 +;\n";
        let diag = Diagnostic::error("E0001", "unexpected token", Span::new(13, 14));

        let renderer = DiagnosticRenderer::new(source, "test.cjc");
        let output = renderer.render(&diag);
        let first_line = output.lines().next().unwrap();
        // First line should be: test.cjc:1:14: error[E0001]: unexpected token
        assert!(first_line.starts_with("test.cjc:1:14:"), "first line: {}", first_line);
        assert!(first_line.contains("error[E0001]"));
        assert!(first_line.contains("unexpected token"));
    }

    // ── Footer tests ─────────────────────────────────────────────────

    #[test]
    fn test_footer_errors_and_warnings() {
        let mut bag = DiagnosticBag::new();
        bag.emit(Diagnostic::error("E0001", "err", Span::new(0, 1)));
        bag.emit(Diagnostic::warning("W0001", "warn", Span::new(0, 1)));

        let output = bag.render_all("x", "test.cjc");
        assert!(output.contains("aborting due to 1 error; 1 warning"));
    }

    #[test]
    fn test_footer_warnings_only() {
        let mut bag = DiagnosticBag::new();
        bag.emit(Diagnostic::warning("W0001", "warn1", Span::new(0, 1)));
        bag.emit(Diagnostic::warning("W0002", "warn2", Span::new(0, 1)));

        let output = bag.render_all("x", "test.cjc");
        assert!(output.contains("warning: generated 2 warnings"));
        assert!(!output.contains("aborting"));
    }

    #[test]
    fn test_warning_count() {
        let mut bag = DiagnosticBag::new();
        assert_eq!(bag.warning_count(), 0);
        bag.emit(Diagnostic::warning("W0001", "w", Span::new(0, 1)));
        bag.emit(Diagnostic::warning("W0002", "w", Span::new(0, 1)));
        bag.emit(Diagnostic::error("E0001", "e", Span::new(0, 1)));
        assert_eq!(bag.warning_count(), 2);
        assert_eq!(bag.error_count(), 1);
    }
}
