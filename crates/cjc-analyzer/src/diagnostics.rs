//! Diagnostics bridge — converts CJC DiagnosticBag entries to LSP Diagnostics.
//!
//! This module takes the parser/type-checker diagnostics from `cjc-diag`
//! and converts them into `lsp_types::Diagnostic` for the LSP client.

use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};

/// Convert CJC parser diagnostics to LSP diagnostics.
///
/// Takes source text and diagnostic messages, produces LSP-compatible
/// diagnostics with line/column positions.
pub fn diagnostics_from_parse(
    source: &str,
    diags: &cjc_diag::DiagnosticBag,
) -> Vec<Diagnostic> {
    let line_starts = compute_line_starts(source);

    diags
        .diagnostics
        .iter()
        .map(|d| {
            let severity = match d.severity {
                cjc_diag::Severity::Error => DiagnosticSeverity::ERROR,
                cjc_diag::Severity::Warning => DiagnosticSeverity::WARNING,
                cjc_diag::Severity::Hint => DiagnosticSeverity::HINT,
            };
            let (line, col) = offset_to_line_col(&line_starts, d.span.start);
            let (end_line, end_col) = offset_to_line_col(&line_starts, d.span.end);

            Diagnostic {
                range: Range {
                    start: Position {
                        line: line as u32,
                        character: col as u32,
                    },
                    end: Position {
                        line: end_line as u32,
                        character: end_col as u32,
                    },
                },
                severity: Some(severity),
                source: Some("cjc".to_string()),
                message: d.message.clone(),
                ..Default::default()
            }
        })
        .collect()
}

/// Compute byte offsets of line starts for efficient line/col lookup.
fn compute_line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}

/// Convert a byte offset to (line, col), both 0-indexed.
fn offset_to_line_col(line_starts: &[usize], offset: usize) -> (usize, usize) {
    let line = line_starts
        .partition_point(|&start| start <= offset)
        .saturating_sub(1);
    let col = offset.saturating_sub(line_starts[line]);
    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_starts() {
        let src = "line1\nline2\nline3";
        let starts = compute_line_starts(src);
        assert_eq!(starts, vec![0, 6, 12]);
    }

    #[test]
    fn test_offset_to_line_col() {
        let starts = vec![0, 6, 12];
        assert_eq!(offset_to_line_col(&starts, 0), (0, 0));
        assert_eq!(offset_to_line_col(&starts, 3), (0, 3));
        assert_eq!(offset_to_line_col(&starts, 6), (1, 0));
        assert_eq!(offset_to_line_col(&starts, 14), (2, 2));
    }

    #[test]
    fn test_parse_diagnostics() {
        let src = "fn bad(";
        let (_, diags) = cjc_parser::parse_source(src);
        let lsp_diags = diagnostics_from_parse(src, &diags);
        // Parser should report an error for incomplete function
        assert!(!lsp_diags.is_empty());
        assert_eq!(lsp_diags[0].severity, Some(DiagnosticSeverity::ERROR));
    }
}
