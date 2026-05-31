//! Hand-written, zero-dependency JSON serialization for `LockeReport`.
//!
//! Why hand-written: the workspace's "no external deps" ethos rules
//! out pulling `serde_json` just for Locke. The set of types we
//! serialize is small and closed, so the cost is bounded.
//!
//! Determinism:
//! - All `BTreeMap`s emit in key-sorted order (the map's natural iteration).
//! - All `Vec<finding>` emit in `sort_key()` order (already enforced by
//!   `LockeReport::new`).
//! - Floats use Rust's `{:?}` debug formatting which round-trips to
//!   bit-identical f64 via `f64::from_str`. NaN and infinities are
//!   serialized as JSON strings ("NaN", "Infinity", "-Infinity") since
//!   JSON-spec floats can't represent them.
//! - Strings are minimally JSON-escaped (`"`, `\\`, control chars).
//!
//! Schema versioning: the top-level `schema_version` field is the
//! `LockeReport::SCHEMA_VERSION` constant. v0.4 ships schema version 1.

use crate::report::{
    FindingEvidence, FindingSeverity, LockeInputSummary, LockeReport, SeverityCounts,
    ValidationFinding,
};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt::Write as _;

// ─── Static-code interner ─────────────────────────────────────────────────────
//
// `ValidationFinding::code` is `&'static str` by design (it's a stable error
// code, not user input). When `parse_locke_report_json` reconstructs a report
// from JSON, the code string starts life as a heap `String` and must be
// converted to `&'static str` to fit the type. Pre-fix: `box_leak_str` leaked
// per finding, so a 10K-finding parse leaked 10K small `String`s permanently
// — monotonic growth in any long-lived process.
//
// Post-fix: a thread-local `BTreeSet<&'static str>` dedupes leaks. Distinct
// codes are leaked at most once per thread. The set is internal state used
// only for memoization — its iteration is never observed by output, so
// determinism is unaffected. `BTreeSet` is used rather than `HashSet` to
// align with the project's "no HashSet/HashMap" rule.
thread_local! {
    static STATIC_CODE_INTERNER: RefCell<BTreeSet<&'static str>> =
        RefCell::new(BTreeSet::new());
}

/// Intern `s` as a `&'static str`. Distinct strings leak at most once per
/// thread; subsequent calls with an equal string return the previously
/// leaked pointer. Used only by `parse_locke_report_json` for the
/// `ValidationFinding::code` field.
fn intern_static_code(s: String) -> &'static str {
    STATIC_CODE_INTERNER.with(|interner| {
        if let Some(existing) = interner.borrow().get(s.as_str()) {
            return *existing;
        }
        let leaked: &'static str = Box::leak(s.into_boxed_str());
        interner.borrow_mut().insert(leaked);
        leaked
    })
}

/// Rough size estimate for preallocation. 256 bytes per finding is a
/// generous average that avoids early-resize reallocs for typical reports.
fn estimated_report_size(report: &LockeReport) -> usize {
    256 + report.findings.len() * 256 + report.assumptions.len() * 64
}

/// Serialize a `LockeReport` to a canonical JSON string. Repeated calls
/// over equal reports produce byte-identical output.
pub fn emit_locke_report_json(report: &LockeReport) -> String {
    let mut out = String::with_capacity(estimated_report_size(report));
    out.push('{');
    out.push_str("\"schema_version\":");
    write!(out, "{}", report.schema_version).unwrap();
    out.push(',');
    write!(out, "\"run_id\":\"{}\"", report.run_id).unwrap();
    out.push(',');
    out.push_str("\"input\":");
    write_input_summary(&mut out, &report.input);
    out.push(',');
    out.push_str("\"severity_counts\":");
    write_severity_counts(&mut out, &report.severity_counts);
    out.push(',');
    out.push_str("\"findings\":");
    write_findings(&mut out, &report.findings);
    out.push(',');
    out.push_str("\"assumptions\":");
    write_string_array(&mut out, &report.assumptions);
    out.push('}');
    out
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_string_array(out: &mut String, xs: &[String]) {
    out.push('[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_string(out, x);
    }
    out.push(']');
}

fn write_float(out: &mut String, x: f64) {
    if x.is_nan() {
        out.push_str("\"NaN\"");
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "\"Infinity\"" } else { "\"-Infinity\"" });
    } else {
        write!(out, "{:?}", x).unwrap();
    }
}

fn write_input_summary(out: &mut String, input: &LockeInputSummary) {
    out.push('{');
    out.push_str("\"dataset_label\":");
    write_string(out, &input.dataset_label);
    out.push(',');
    write!(out, "\"n_rows\":{}", input.n_rows).unwrap();
    out.push(',');
    write!(out, "\"n_cols\":{}", input.n_cols).unwrap();
    out.push(',');
    out.push_str("\"column_types\":{");
    for (i, (k, v)) in input.column_types.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_string(out, k);
        out.push(':');
        write_string(out, v);
    }
    out.push_str("}}");
}

fn write_severity_counts(out: &mut String, c: &SeverityCounts) {
    write!(
        out,
        "{{\"info\":{},\"notice\":{},\"warning\":{},\"error\":{}}}",
        c.info, c.notice, c.warning, c.error
    )
    .unwrap();
}

fn write_findings(out: &mut String, fs: &[ValidationFinding]) {
    out.push('[');
    for (i, f) in fs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_finding(out, f);
    }
    out.push(']');
}

fn severity_str(s: FindingSeverity) -> &'static str {
    s.as_str()
}

fn write_finding(out: &mut String, f: &ValidationFinding) {
    out.push('{');
    write!(out, "\"id\":\"{}\"", f.id).unwrap();
    out.push(',');
    out.push_str("\"code\":\"");
    out.push_str(f.code);
    out.push('"');
    out.push(',');
    out.push_str("\"severity\":\"");
    out.push_str(severity_str(f.severity));
    out.push('"');
    out.push(',');
    out.push_str("\"message\":");
    write_string(out, &f.message);
    out.push(',');
    out.push_str("\"column\":");
    match &f.column {
        Some(c) => write_string(out, c),
        None => out.push_str("null"),
    }
    out.push(',');
    out.push_str("\"row_range\":");
    match f.row_range {
        Some((lo, hi)) => {
            write!(out, "[{},{}]", lo, hi).unwrap();
        }
        None => out.push_str("null"),
    }
    out.push(',');
    write!(out, "\"sample_size\":{}", f.sample_size).unwrap();
    out.push(',');
    out.push_str("\"evidence\":");
    write_evidence_array(out, &f.evidence);
    out.push(',');
    out.push_str("\"assumptions\":");
    write_string_array(out, &f.assumptions);
    out.push(',');
    out.push_str("\"suggested_next_checks\":");
    write_string_array(out, &f.suggested_next_checks);
    out.push('}');
}

fn write_evidence_array(out: &mut String, es: &[FindingEvidence]) {
    out.push('[');
    for (i, e) in es.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_evidence(out, e);
    }
    out.push(']');
}

fn write_evidence(out: &mut String, e: &FindingEvidence) {
    out.push('{');
    match e {
        FindingEvidence::Count { label, value } => {
            out.push_str("\"kind\":\"count\",\"label\":");
            write_string(out, label);
            write!(out, ",\"value\":{}", value).unwrap();
        }
        FindingEvidence::Ratio { label, value } => {
            out.push_str("\"kind\":\"ratio\",\"label\":");
            write_string(out, label);
            out.push_str(",\"value\":");
            write_float(out, *value);
        }
        FindingEvidence::Range { label, min, max } => {
            out.push_str("\"kind\":\"range\",\"label\":");
            write_string(out, label);
            out.push_str(",\"min\":");
            write_float(out, *min);
            out.push_str(",\"max\":");
            write_float(out, *max);
        }
        FindingEvidence::Metric { label, value } => {
            out.push_str("\"kind\":\"metric\",\"label\":");
            write_string(out, label);
            out.push_str(",\"value\":");
            write_float(out, *value);
        }
        FindingEvidence::Sample { label, value } => {
            out.push_str("\"kind\":\"sample\",\"label\":");
            write_string(out, label);
            out.push_str(",\"value\":");
            write_string(out, value);
        }
    }
    out.push('}');
}

/// Parse a Locke-emitted JSON string back into a `LockeReport`.
///
/// **Scope**: this is a *targeted* parser tuned to the canonical
/// emission above — not a general-purpose JSON parser. It accepts the
/// exact byte sequences `emit_locke_report_json` produces and rejects
/// most other valid JSON with structured errors. v0.5 may relax this
/// to handle whitespace / key reordering if real demand appears.
///
/// Surrogate-pair `\uXXXX\uYYYY` escapes are accepted for human-edited
/// input even though emit never produces them (emit writes codepoints
/// above U+FFFF as raw UTF-8 chars). Lone surrogates are rejected.
pub fn parse_locke_report_json(input: &str) -> Result<LockeReport, String> {
    let mut p = Parser::new(input);
    p.expect('{')?;
    p.expect_key("schema_version")?;
    let schema_version = p.parse_u64()?;
    if schema_version > u32::MAX as u64 || (schema_version as u32) != LockeReport::SCHEMA_VERSION {
        return Err(format!(
            "schema_version mismatch: got {}, expected {}",
            schema_version,
            LockeReport::SCHEMA_VERSION
        ));
    }
    p.expect(',')?;
    p.expect_key("run_id")?;
    let _run_id = p.parse_string()?; // Will be recomputed from input + findings on rebuild
    p.expect(',')?;
    p.expect_key("input")?;
    let input_summary = parse_input_summary(&mut p)?;
    p.expect(',')?;
    p.expect_key("severity_counts")?;
    let _ = parse_severity_counts(&mut p)?; // Will be recomputed on rebuild
    p.expect(',')?;
    p.expect_key("findings")?;
    let findings = parse_findings(&mut p)?;
    p.expect(',')?;
    p.expect_key("assumptions")?;
    let assumptions = parse_string_array(&mut p)?;
    p.expect('}')?;
    Ok(LockeReport::new(
        input_summary,
        findings,
        std::collections::BTreeMap::new(),
        assumptions,
    ))
}

fn parse_input_summary(p: &mut Parser) -> Result<LockeInputSummary, String> {
    p.expect('{')?;
    p.expect_key("dataset_label")?;
    let dataset_label = p.parse_string()?;
    p.expect(',')?;
    p.expect_key("n_rows")?;
    let n_rows = p.parse_u64()?;
    p.expect(',')?;
    p.expect_key("n_cols")?;
    let n_cols = p.parse_u64()?;
    p.expect(',')?;
    p.expect_key("column_types")?;
    let column_types = parse_string_map(p)?;
    p.expect('}')?;
    Ok(LockeInputSummary {
        dataset_label,
        n_rows,
        n_cols,
        column_types,
    })
}

fn parse_severity_counts(p: &mut Parser) -> Result<SeverityCounts, String> {
    p.expect('{')?;
    p.expect_key("info")?;
    let info = p.parse_u64()?;
    p.expect(',')?;
    p.expect_key("notice")?;
    let notice = p.parse_u64()?;
    p.expect(',')?;
    p.expect_key("warning")?;
    let warning = p.parse_u64()?;
    p.expect(',')?;
    p.expect_key("error")?;
    let error = p.parse_u64()?;
    p.expect('}')?;
    Ok(SeverityCounts {
        info,
        notice,
        warning,
        error,
    })
}

fn parse_findings(p: &mut Parser) -> Result<Vec<ValidationFinding>, String> {
    p.expect('[')?;
    let mut out = Vec::new();
    if p.peek() == Some(']') {
        p.expect(']')?;
        return Ok(out);
    }
    loop {
        out.push(parse_finding(p)?);
        if p.peek() == Some(',') {
            p.expect(',')?;
        } else {
            break;
        }
    }
    p.expect(']')?;
    Ok(out)
}

fn parse_finding(p: &mut Parser) -> Result<ValidationFinding, String> {
    p.expect('{')?;
    p.expect_key("id")?;
    let _id = p.parse_string()?; // recomputed
    p.expect(',')?;
    p.expect_key("code")?;
    let code_owned = p.parse_string()?;
    let code = intern_static_code(code_owned);
    p.expect(',')?;
    p.expect_key("severity")?;
    let severity = parse_severity(&p.parse_string()?)?;
    p.expect(',')?;
    p.expect_key("message")?;
    let message = p.parse_string()?;
    p.expect(',')?;
    p.expect_key("column")?;
    let column = p.parse_nullable_string()?;
    p.expect(',')?;
    p.expect_key("row_range")?;
    let row_range = parse_row_range(p)?;
    p.expect(',')?;
    p.expect_key("sample_size")?;
    let sample_size = p.parse_u64()?;
    p.expect(',')?;
    p.expect_key("evidence")?;
    let evidence = parse_evidence_array(p)?;
    p.expect(',')?;
    p.expect_key("assumptions")?;
    let assumptions = parse_string_array(p)?;
    p.expect(',')?;
    p.expect_key("suggested_next_checks")?;
    let suggested_next_checks = parse_string_array(p)?;
    p.expect('}')?;
    Ok(ValidationFinding::new(
        code,
        severity,
        message,
        column,
        row_range,
        evidence,
        sample_size,
        assumptions,
        suggested_next_checks,
    ))
}

fn parse_severity(s: &str) -> Result<FindingSeverity, String> {
    match s {
        "info" => Ok(FindingSeverity::Info),
        "notice" => Ok(FindingSeverity::Notice),
        "warning" => Ok(FindingSeverity::Warning),
        "error" => Ok(FindingSeverity::Error),
        other => Err(format!("unknown severity: {}", other)),
    }
}

fn parse_row_range(p: &mut Parser) -> Result<Option<(usize, usize)>, String> {
    if p.peek() == Some('n') {
        p.expect_literal("null")?;
        Ok(None)
    } else {
        p.expect('[')?;
        let lo = p.parse_usize()?;
        p.expect(',')?;
        let hi = p.parse_usize()?;
        p.expect(']')?;
        Ok(Some((lo, hi)))
    }
}

fn parse_evidence_array(p: &mut Parser) -> Result<Vec<FindingEvidence>, String> {
    p.expect('[')?;
    let mut out = Vec::new();
    if p.peek() == Some(']') {
        p.expect(']')?;
        return Ok(out);
    }
    loop {
        out.push(parse_evidence(p)?);
        if p.peek() == Some(',') {
            p.expect(',')?;
        } else {
            break;
        }
    }
    p.expect(']')?;
    Ok(out)
}

fn parse_evidence(p: &mut Parser) -> Result<FindingEvidence, String> {
    p.expect('{')?;
    p.expect_key("kind")?;
    let kind = p.parse_string()?;
    p.expect(',')?;
    p.expect_key("label")?;
    let label = p.parse_string()?;
    let ev = match kind.as_str() {
        "count" => {
            p.expect(',')?;
            p.expect_key("value")?;
            let value = p.parse_u64()?;
            FindingEvidence::Count { label, value }
        }
        "ratio" => {
            p.expect(',')?;
            p.expect_key("value")?;
            let value = p.parse_float()?;
            FindingEvidence::Ratio { label, value }
        }
        "range" => {
            p.expect(',')?;
            p.expect_key("min")?;
            let min = p.parse_float()?;
            p.expect(',')?;
            p.expect_key("max")?;
            let max = p.parse_float()?;
            FindingEvidence::Range { label, min, max }
        }
        "metric" => {
            p.expect(',')?;
            p.expect_key("value")?;
            let value = p.parse_float()?;
            FindingEvidence::Metric { label, value }
        }
        "sample" => {
            p.expect(',')?;
            p.expect_key("value")?;
            let value = p.parse_string()?;
            FindingEvidence::Sample { label, value }
        }
        other => return Err(format!("unknown evidence kind: {}", other)),
    };
    p.expect('}')?;
    Ok(ev)
}

fn parse_string_array(p: &mut Parser) -> Result<Vec<String>, String> {
    p.expect('[')?;
    let mut out = Vec::new();
    if p.peek() == Some(']') {
        p.expect(']')?;
        return Ok(out);
    }
    loop {
        out.push(p.parse_string()?);
        if p.peek() == Some(',') {
            p.expect(',')?;
        } else {
            break;
        }
    }
    p.expect(']')?;
    Ok(out)
}

fn parse_string_map(p: &mut Parser) -> Result<std::collections::BTreeMap<String, String>, String> {
    p.expect('{')?;
    let mut out = std::collections::BTreeMap::new();
    if p.peek() == Some('}') {
        p.expect('}')?;
        return Ok(out);
    }
    loop {
        let k = p.parse_string()?;
        p.expect(':')?;
        let v = p.parse_string()?;
        out.insert(k, v);
        if p.peek() == Some(',') {
            p.expect(',')?;
        } else {
            break;
        }
    }
    p.expect('}')?;
    Ok(out)
}

// ─── Minimal JSON parser tuned to our emit ────────────────────────────────────

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }
    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }
    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        match self.advance() {
            Some(x) if x == c => Ok(()),
            Some(x) => Err(format!("expected '{}', got '{}' at pos {}", c, x, self.pos)),
            None => Err(format!("expected '{}', got EOF", c)),
        }
    }
    fn expect_literal(&mut self, lit: &str) -> Result<(), String> {
        if self.src[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("expected literal `{}` at pos {}", lit, self.pos))
        }
    }
    fn expect_key(&mut self, key: &str) -> Result<(), String> {
        let s = self.parse_string()?;
        if s == key {
            self.expect(':')?;
            Ok(())
        } else {
            Err(format!("expected key `{}`, got `{}`", key, s))
        }
    }
    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            let c = self.advance().ok_or("unexpected EOF in string")?;
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let esc = self.advance().ok_or("unexpected EOF after \\")?;
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let cp = self.parse_u_hex4()?;
                            // RFC 8259 §7: surrogate pair handling. A high
                            // surrogate (U+D800..=U+DBFF) MUST be followed
                            // by `\u` + a low surrogate (U+DC00..=U+DFFF).
                            // The combined codepoint is computed via the
                            // UTF-16 encoding formula.
                            if (0xD800..=0xDBFF).contains(&cp) {
                                if self.advance() != Some('\\') || self.advance() != Some('u') {
                                    return Err(format!(
                                        "high surrogate U+{:04X} not followed by \\u low surrogate at pos {}",
                                        cp, self.pos
                                    ));
                                }
                                let lo = self.parse_u_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(format!(
                                        "expected low surrogate (U+DC00..U+DFFF), got U+{:04X} at pos {}",
                                        lo, self.pos
                                    ));
                                }
                                let full = 0x10000u32 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                if let Some(ch) = char::from_u32(full) {
                                    out.push(ch);
                                } else {
                                    return Err(format!(
                                        "bad surrogate-pair codepoint U+{:06X}",
                                        full
                                    ));
                                }
                            } else if (0xDC00..=0xDFFF).contains(&cp) {
                                return Err(format!(
                                    "unexpected lone low surrogate U+{:04X} at pos {}",
                                    cp, self.pos
                                ));
                            } else if let Some(ch) = char::from_u32(cp) {
                                out.push(ch);
                            } else {
                                return Err(format!("bad codepoint U+{:04X}", cp));
                            }
                        }
                        other => return Err(format!("bad escape \\{}", other)),
                    }
                }
                c => out.push(c),
            }
        }
    }
    /// Read exactly 4 ASCII-hex digits and parse them as a `u32`. Errors
    /// on EOF mid-sequence or any non-hex character.
    fn parse_u_hex4(&mut self) -> Result<u32, String> {
        let mut acc: u32 = 0;
        for i in 0..4 {
            let c = self
                .advance()
                .ok_or_else(|| format!("EOF mid \\u escape after {} hex digits", i))?;
            let d = c.to_digit(16).ok_or_else(|| {
                format!("non-hex char '{}' in \\u escape at pos {}", c, self.pos)
            })?;
            acc = (acc << 4) | d;
        }
        Ok(acc)
    }
    fn parse_nullable_string(&mut self) -> Result<Option<String>, String> {
        if self.peek() == Some('n') {
            self.expect_literal("null")?;
            Ok(None)
        } else {
            Ok(Some(self.parse_string()?))
        }
    }
    /// Parse an unsigned integer. Rejects a leading `-` explicitly rather
    /// than relying on `i64::parse + as u64` (which silently wraps
    /// negatives to enormous values). All Locke `u64` fields go through
    /// this method.
    fn parse_u64(&mut self) -> Result<u64, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            return Err(format!(
                "expected unsigned integer, got leading `-` at pos {}",
                self.pos
            ));
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(format!("expected unsigned integer at pos {}", self.pos));
        }
        let s = &self.src[start..self.pos];
        s.parse::<u64>().map_err(|e| format!("bad u64 `{}`: {}", s, e))
    }
    /// Parse a `usize`, checking the value fits the host's pointer width.
    /// On 32-bit hosts this rejects values that would silently truncate.
    fn parse_usize(&mut self) -> Result<usize, String> {
        let v = self.parse_u64()?;
        if v > usize::MAX as u64 {
            return Err(format!(
                "value {} exceeds usize::MAX ({}) on this host",
                v,
                usize::MAX
            ));
        }
        Ok(v as usize)
    }
    fn parse_float(&mut self) -> Result<f64, String> {
        // Handle the special string forms "NaN" / "Infinity" / "-Infinity".
        if self.peek() == Some('"') {
            let s = self.parse_string()?;
            return match s.as_str() {
                "NaN" => Ok(f64::NAN),
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                other => Err(format!("bad float string: {}", other)),
            };
        }
        let start = self.pos;
        if self.peek() == Some('-') {
            self.advance();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.advance();
            } else {
                break;
            }
        }
        let s = &self.src[start..self.pos];
        s.parse::<f64>().map_err(|e| format!("bad float `{}`: {}", s, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{FindingEvidence, FindingSeverity, ValidationFinding};
    use std::collections::BTreeMap;

    fn make_finding() -> ValidationFinding {
        ValidationFinding::new(
            "E9001",
            FindingSeverity::Warning,
            "test message",
            Some("age".into()),
            Some((0, 10)),
            vec![
                FindingEvidence::Count {
                    label: "n_missing".into(),
                    value: 3,
                },
                FindingEvidence::Ratio {
                    label: "rate".into(),
                    value: 0.5,
                },
            ],
            100,
            vec!["NaN treated as missing".into()],
            vec!["check column".into()],
        )
    }

    fn make_report() -> LockeReport {
        let mut types = BTreeMap::new();
        types.insert("age".into(), "Float".into());
        let input = LockeInputSummary {
            dataset_label: "test".into(),
            n_rows: 100,
            n_cols: 1,
            column_types: types,
        };
        LockeReport::new(input, vec![make_finding()], BTreeMap::new(), vec![])
    }

    #[test]
    fn emit_then_parse_is_bit_identical() {
        let r1 = make_report();
        let s1 = emit_locke_report_json(&r1);
        let r2 = parse_locke_report_json(&s1).expect("round trip");
        let s2 = emit_locke_report_json(&r2);
        assert_eq!(s1, s2, "byte-identical round trip");
    }

    #[test]
    fn emit_is_deterministic_across_repeated_calls() {
        let r = make_report();
        let s1 = emit_locke_report_json(&r);
        let s2 = emit_locke_report_json(&r);
        assert_eq!(s1, s2);
    }

    #[test]
    fn empty_report_round_trips() {
        let input = LockeInputSummary::default();
        let r = LockeReport::new(input, vec![], BTreeMap::new(), vec![]);
        let s = emit_locke_report_json(&r);
        let r2 = parse_locke_report_json(&s).expect("round trip");
        let s2 = emit_locke_report_json(&r2);
        assert_eq!(s, s2);
    }

    #[test]
    fn nan_and_infinity_serialize_as_strings() {
        let f = ValidationFinding::new(
            "E9039",
            FindingSeverity::Warning,
            "test",
            Some("x".into()),
            None,
            vec![FindingEvidence::Metric {
                label: "v".into(),
                value: f64::NAN,
            }],
            10,
            vec![],
            vec![],
        );
        let input = LockeInputSummary::default();
        let r = LockeReport::new(input, vec![f], BTreeMap::new(), vec![]);
        let s = emit_locke_report_json(&r);
        assert!(s.contains("\"NaN\""));
    }

    #[test]
    fn special_characters_in_strings_are_escaped() {
        let f = ValidationFinding::new(
            "E9001",
            FindingSeverity::Notice,
            "has \"quotes\" and\nnewlines",
            Some("x".into()),
            None,
            vec![],
            10,
            vec![],
            vec![],
        );
        let input = LockeInputSummary::default();
        let r = LockeReport::new(input, vec![f], BTreeMap::new(), vec![]);
        let s = emit_locke_report_json(&r);
        assert!(s.contains("\\\"quotes\\\""));
        assert!(s.contains("\\n"));
        let r2 = parse_locke_report_json(&s).expect("round trip");
        assert!(r2.findings[0].message.contains("\"quotes\""));
        assert!(r2.findings[0].message.contains("\n"));
    }

    #[test]
    fn schema_version_mismatch_is_an_error() {
        let bad = r#"{"schema_version":99,"run_id":"x","input":{}}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err());
    }

    // ─── B3.1: negative-int wrap regression tests ────────────────────────

    #[test]
    fn parse_u64_rejects_negative_n_rows() {
        // Pre-fix: parse_int returns -1, then `as u64` gives u64::MAX,
        // so the parser "succeeds" on a hostile input with absurd n_rows.
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"t","n_rows":-1,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "negative n_rows must error, got {:?}", r);
        let msg = r.unwrap_err();
        assert!(
            msg.contains("leading `-`") || msg.contains("unsigned"),
            "expected leading-minus error, got: {}",
            msg
        );
    }

    #[test]
    fn parse_u64_rejects_oversized_n_rows() {
        // 99...99 (20 nines) overflows u64 (max ~1.8e19, this is 1e20).
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"t","n_rows":99999999999999999999,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "oversized n_rows must error");
        let msg = r.unwrap_err();
        assert!(msg.contains("bad u64"), "expected u64 overflow error, got: {}", msg);
    }

    #[test]
    fn parse_u64_rejects_negative_sample_size() {
        // Same shape attack, but at the per-finding sample_size field.
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"t","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[{"id":"abc","code":"E9001","severity":"info","message":"m","column":null,"row_range":null,"sample_size":-5,"evidence":[],"assumptions":[],"suggested_next_checks":[]}],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "negative sample_size must error, got {:?}", r);
    }

    #[test]
    fn parse_u64_accepts_zero_and_legitimate_max() {
        // Zero and u64::MAX are the boundary legitimate values.
        let ok_zero = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"t","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        assert!(parse_locke_report_json(ok_zero).is_ok());
        // u64::MAX = 18446744073709551615 (20 digits).
        let ok_max = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"t","n_rows":18446744073709551615,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let parsed = parse_locke_report_json(ok_max).expect("u64::MAX must parse");
        assert_eq!(parsed.input.n_rows, u64::MAX);
    }

    // ─── B3.2: interner dedup regression test ───────────────────────────

    #[test]
    fn intern_static_code_dedupes_repeated_strings() {
        // Same logical code interned twice → same pointer.
        let a = intern_static_code(String::from("E9999_INTERNER_TEST"));
        let b = intern_static_code(String::from("E9999_INTERNER_TEST"));
        assert!(std::ptr::eq(a, b), "interner must return same pointer for equal strings");
    }

    #[test]
    fn intern_static_code_distinct_strings_get_distinct_leaks() {
        let a = intern_static_code(String::from("E9999_INTERNER_A"));
        let b = intern_static_code(String::from("E9999_INTERNER_B"));
        assert!(
            !std::ptr::eq(a, b),
            "distinct strings must get distinct interned pointers"
        );
        assert_eq!(a, "E9999_INTERNER_A");
        assert_eq!(b, "E9999_INTERNER_B");
    }

    // ─── B3.3: surrogate pair regression tests ──────────────────────────

    #[test]
    fn parse_string_handles_surrogate_pair_for_emoji() {
        // 😀 = U+1F600 = surrogate pair (U+D83D, U+DE00) in UTF-16.
        // Emit doesn't produce this, but human-edited JSON can.
        // Construct a minimal valid report with a surrogate-pair emoji
        // in the dataset_label, then verify it parses to 😀.
        let json = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"hi \uD83D\uDE00 there","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(json).expect("surrogate pair must parse");
        assert!(
            r.input.dataset_label.contains('😀'),
            "expected emoji in dataset_label, got: {}",
            r.input.dataset_label
        );
    }

    #[test]
    fn parse_string_rejects_lone_high_surrogate() {
        // Lone high surrogate with no follow-up \u — must error.
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"\uD83D","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "lone high surrogate must error");
        let msg = r.unwrap_err();
        assert!(
            msg.contains("high surrogate"),
            "expected high-surrogate error, got: {}",
            msg
        );
    }

    #[test]
    fn parse_string_rejects_lone_low_surrogate() {
        // Lone low surrogate (no preceding high surrogate).
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"\uDE00","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "lone low surrogate must error");
        let msg = r.unwrap_err();
        assert!(
            msg.contains("lone low surrogate") || msg.contains("low surrogate"),
            "expected low-surrogate error, got: {}",
            msg
        );
    }

    #[test]
    fn parse_string_rejects_high_surrogate_then_non_low() {
        // High surrogate followed by \u + non-low (here A = 'A').
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"\uD83DA","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "high surrogate + non-low must error");
    }

    #[test]
    fn parse_u_hex4_rejects_non_hex_chars() {
        // Bad hex in \u escape — must error, not silently consume garbage.
        let bad = r#"{"schema_version":1,"run_id":"x","input":{"dataset_label":"\uZZZZ","n_rows":0,"n_cols":0,"column_types":{}},"severity_counts":{"info":0,"notice":0,"warning":0,"error":0},"findings":[],"assumptions":[]}"#;
        let r = parse_locke_report_json(bad);
        assert!(r.is_err(), "non-hex in \\u escape must error");
    }

    // ─── B3.4: byte-identity regression test ────────────────────────────

    // ─── B3.5 tripwire: detect future stdlib drift in {:?} f64 format ───
    //
    // The canonical Locke JSON byte-output relies on Rust's `f64::Debug`
    // (the `{:?}` format spec) producing the same shortest-round-trip
    // representation across compiler versions. This has been stable in
    // practice since Rust 1.0, but is not a formal stdlib guarantee.
    //
    // This test pins the output bytes for a battery of edge-case f64
    // values. If a future toolchain bump changes any of them, this test
    // fires loudly at test time — far better than discovering it via a
    // broken audit chain in production. The expected strings below were
    // captured against Rust 1.91.1; update only after deliberate review.

    #[test]
    fn write_float_canonical_bytes_for_edge_cases() {
        let cases: &[(f64, &str)] = &[
            (0.0_f64, "0.0"),
            (-0.0_f64, "-0.0"),
            (1.0_f64, "1.0"),
            (-1.0_f64, "-1.0"),
            (0.5_f64, "0.5"),
            (0.1_f64, "0.1"),
            (1e30_f64, "1e30"),
            (1e-30_f64, "1e-30"),
            (f64::EPSILON, "2.220446049250313e-16"),
            (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
            (f64::MAX, "1.7976931348623157e308"),
            (f64::MIN, "-1.7976931348623157e308"),
            (f64::from_bits(1), "5e-324"), // smallest subnormal
            ((1u64 << 53) as f64, "9007199254740992.0"),
            (std::f64::consts::PI, "3.141592653589793"),
        ];
        for (x, expected) in cases {
            let mut out = String::new();
            write_float(&mut out, *x);
            assert_eq!(
                out,
                *expected,
                "write_float({}) drifted from frozen golden — toolchain f64::Debug changed",
                x
            );
        }
    }

    #[test]
    fn write_float_nan_and_infinity_canonical_bytes() {
        // These don't go through {:?} (we wrap them as JSON strings).
        for (x, expected) in [
            (f64::NAN, "\"NaN\""),
            (f64::INFINITY, "\"Infinity\""),
            (f64::NEG_INFINITY, "\"-Infinity\""),
        ] {
            let mut out = String::new();
            write_float(&mut out, x);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn emit_output_matches_known_golden() {
        // Lock in the exact byte sequence for a known finding so the
        // write!()-based emit doesn't drift from format!()-based emit.
        // Any drift in this string means the canonical output changed and
        // every audit chain ID downstream would shift.
        let r = make_report();
        let s = emit_locke_report_json(&r);
        // Spot-check a few specific substrings rather than the whole
        // string (run_id is content-addressed and depends on input).
        assert!(s.starts_with(r#"{"schema_version":1,"run_id":""#));
        assert!(s.contains(r#""input":{"dataset_label":"test","n_rows":100,"n_cols":1"#));
        assert!(s.contains(r#""severity_counts":{"info":0,"notice":0,"warning":1,"error":0}"#));
        assert!(s.contains(r#""code":"E9001""#));
        assert!(s.contains(r#""severity":"warning""#));
        assert!(s.contains(r#""sample_size":100"#));
        assert!(s.contains(r#""row_range":[0,10]"#));
        // Evidence value:0.5 — verify {:?} format produces "0.5" not "0.5e0".
        assert!(s.contains(r#""value":0.5"#));
        assert!(s.ends_with("}"));
    }
}
