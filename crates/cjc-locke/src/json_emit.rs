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

/// Serialize a `LockeReport` to a canonical JSON string. Repeated calls
/// over equal reports produce byte-identical output.
pub fn emit_locke_report_json(report: &LockeReport) -> String {
    let mut out = String::new();
    out.push('{');
    write_string_field(&mut out, "schema_version", &report.schema_version.to_string(), true);
    out.push(',');
    write_field(&mut out, "run_id", &format!("\"{}\"", report.run_id));
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

fn write_field(out: &mut String, key: &str, value_json: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(value_json);
}

fn write_string_field(out: &mut String, key: &str, value: &str, raw: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    if raw {
        out.push_str(value);
    } else {
        write_string(out, value);
    }
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
                out.push_str(&format!("\\u{:04x}", c as u32));
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
        // {:?} gives a round-trippable representation.
        out.push_str(&format!("{:?}", x));
    }
}

fn write_input_summary(out: &mut String, input: &LockeInputSummary) {
    out.push('{');
    out.push_str("\"dataset_label\":");
    write_string(out, &input.dataset_label);
    out.push(',');
    out.push_str(&format!("\"n_rows\":{}", input.n_rows));
    out.push(',');
    out.push_str(&format!("\"n_cols\":{}", input.n_cols));
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
    out.push('{');
    out.push_str(&format!(
        "\"info\":{},\"notice\":{},\"warning\":{},\"error\":{}",
        c.info, c.notice, c.warning, c.error
    ));
    out.push('}');
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
    out.push_str(&format!("\"id\":\"{}\"", f.id));
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
        Some((lo, hi)) => out.push_str(&format!("[{},{}]", lo, hi)),
        None => out.push_str("null"),
    }
    out.push(',');
    out.push_str(&format!("\"sample_size\":{}", f.sample_size));
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
            out.push_str(&format!(",\"value\":{}", value));
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
pub fn parse_locke_report_json(input: &str) -> Result<LockeReport, String> {
    let mut p = Parser::new(input);
    p.expect('{')?;
    p.expect_key("schema_version")?;
    let schema_version: u32 = p.parse_int()? as u32;
    if schema_version != LockeReport::SCHEMA_VERSION {
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
    let n_rows = p.parse_int()? as u64;
    p.expect(',')?;
    p.expect_key("n_cols")?;
    let n_cols = p.parse_int()? as u64;
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
    let info = p.parse_int()? as u64;
    p.expect(',')?;
    p.expect_key("notice")?;
    let notice = p.parse_int()? as u64;
    p.expect(',')?;
    p.expect_key("warning")?;
    let warning = p.parse_int()? as u64;
    p.expect(',')?;
    p.expect_key("error")?;
    let error = p.parse_int()? as u64;
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
    let code = box_leak_str(code_owned);
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
    let sample_size = p.parse_int()? as u64;
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

/// Leak a `String` into a `&'static str`. Used because `ValidationFinding::code`
/// is `&'static str` by design (it's a stable error code, not user input).
/// The leak is bounded by the number of distinct codes in a report.
fn box_leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
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
        let lo = p.parse_int()? as usize;
        p.expect(',')?;
        let hi = p.parse_int()? as usize;
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
            let value = p.parse_int()? as u64;
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
                            let hex: String =
                                (0..4).filter_map(|_| self.advance()).collect();
                            let cp = u32::from_str_radix(&hex, 16)
                                .map_err(|_| format!("bad \\u escape: {}", hex))?;
                            if let Some(ch) = char::from_u32(cp) {
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
    fn parse_nullable_string(&mut self) -> Result<Option<String>, String> {
        if self.peek() == Some('n') {
            self.expect_literal("null")?;
            Ok(None)
        } else {
            Ok(Some(self.parse_string()?))
        }
    }
    fn parse_int(&mut self) -> Result<i64, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.advance();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        let s = &self.src[start..self.pos];
        s.parse::<i64>().map_err(|e| format!("bad int `{}`: {}", s, e))
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
}
