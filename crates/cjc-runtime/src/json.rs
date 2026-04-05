//! Hand-rolled JSON parser and emitter for CJC.
//!
//! Design decisions:
//! - Uses `BTreeMap` for object keys → deterministic, sorted output
//! - No external dependencies (no serde)
//! - Converts JSON ↔ CJC `Value` directly
//! - Strings always use `\uXXXX` escaping for non-ASCII in output

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::Value;

// ---------------------------------------------------------------------------
// JSON Value (intermediate representation)
// ---------------------------------------------------------------------------

/// A JSON value, using BTreeMap for deterministic key ordering.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// A simple recursive-descent JSON parser.
struct JsonParser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        match self.advance() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(format!(
                "expected '{}', found '{}' at position {}",
                expected as char, ch as char, self.pos - 1
            )),
            None => Err(format!("unexpected end of input, expected '{}'", expected as char)),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        match self.peek() {
            None => Err("unexpected end of input".into()),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') => self.parse_literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.parse_literal("null", JsonValue::Null),
            Some(ch) if ch == b'-' || ch.is_ascii_digit() => self.parse_number(),
            Some(ch) => Err(format!("unexpected character '{}' at position {}", ch as char, self.pos)),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut result = String::new();
        loop {
            match self.advance() {
                None => return Err("unterminated string".into()),
                Some(b'"') => return Ok(result),
                Some(b'\\') => {
                    match self.advance() {
                        Some(b'"') => result.push('"'),
                        Some(b'\\') => result.push('\\'),
                        Some(b'/') => result.push('/'),
                        Some(b'b') => result.push('\u{0008}'),
                        Some(b'f') => result.push('\u{000C}'),
                        Some(b'n') => result.push('\n'),
                        Some(b'r') => result.push('\r'),
                        Some(b't') => result.push('\t'),
                        Some(b'u') => {
                            let hex = self.parse_hex4()?;
                            if let Some(ch) = char::from_u32(hex) {
                                result.push(ch);
                            } else {
                                result.push('\u{FFFD}');
                            }
                        }
                        Some(ch) => return Err(format!("invalid escape '\\{}'", ch as char)),
                        None => return Err("unterminated escape sequence".into()),
                    }
                }
                Some(ch) => result.push(ch as char),
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut value = 0u32;
        for _ in 0..4 {
            let ch = self.advance().ok_or("unexpected end in \\uXXXX")?;
            let digit = match ch {
                b'0'..=b'9' => (ch - b'0') as u32,
                b'a'..=b'f' => (ch - b'a' + 10) as u32,
                b'A'..=b'F' => (ch - b'A' + 10) as u32,
                _ => return Err(format!("invalid hex digit '{}' in \\uXXXX", ch as char)),
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        // Optional minus
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // Integer part
        if self.peek() == Some(b'0') {
            self.pos += 1;
        } else {
            if !self.peek().map_or(false, |c| c.is_ascii_digit()) {
                return Err("expected digit".into());
            }
            while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        // Fractional part
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        // Exponent
        if self.peek() == Some(b'e') || self.peek() == Some(b'E') {
            self.pos += 1;
            if self.peek() == Some(b'+') || self.peek() == Some(b'-') {
                self.pos += 1;
            }
            while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let num_str = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| "invalid UTF-8 in number")?;
        let value: f64 = num_str
            .parse()
            .map_err(|_| format!("invalid number: {}", num_str))?;
        Ok(JsonValue::Number(value))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect(b'[')?;
        self.skip_whitespace();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err("expected ',' or ']' in array".into()),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect(b'{')?;
        self.skip_whitespace();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => return Err("expected ',' or '}' in object".into()),
            }
        }
    }

    fn parse_literal(&mut self, expected: &str, value: JsonValue) -> Result<JsonValue, String> {
        for byte in expected.as_bytes() {
            match self.advance() {
                Some(ch) if ch == *byte => {}
                _ => return Err(format!("expected '{}'", expected)),
            }
        }
        Ok(value)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a JSON string into a CJC `Value`.
///
/// Mapping:
/// - JSON null → `Value::Void`
/// - JSON bool → `Value::Bool`
/// - JSON number → `Value::Float` (or `Value::Int` if integer-valued)
/// - JSON string → `Value::String`
/// - JSON array → `Value::Array`
/// - JSON object → `Value::Struct { name: "Json", fields }` with sorted keys
pub fn json_parse(input: &str) -> Result<Value, String> {
    let mut parser = JsonParser::new(input);
    let json = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.pos < parser.input.len() {
        return Err(format!(
            "trailing content at position {}",
            parser.pos
        ));
    }
    Ok(json_to_value(json))
}

/// Convert a CJC `Value` to a JSON string with sorted keys.
pub fn json_stringify(value: &Value) -> Result<String, String> {
    let json = value_to_json(value)?;
    Ok(emit_json(&json))
}

// ---------------------------------------------------------------------------
// Conversions: JsonValue ↔ CJC Value
// ---------------------------------------------------------------------------

/// Convert a [`JsonValue`] into a CJC [`Value`].
///
/// JSON objects become `Value::Struct` with name `"Json"` and [`BTreeMap`]
/// fields (sorted keys). JSON arrays become `Value::Array`. Integer-valued
/// numbers (no fractional part, within `i64` range) become `Value::Int`;
/// all others become `Value::Float`.
fn json_to_value(json: JsonValue) -> Value {
    match json {
        JsonValue::Null => Value::Void,
        JsonValue::Bool(b) => Value::Bool(b),
        JsonValue::Number(n) => {
            // If the number is an exact integer and within i64 range, use Int
            if n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
                Value::Int(n as i64)
            } else {
                Value::Float(n)
            }
        }
        JsonValue::String(s) => Value::String(Rc::new(s)),
        JsonValue::Array(items) => {
            let vals: Vec<Value> = items.into_iter().map(json_to_value).collect();
            Value::Array(Rc::new(vals))
        }
        JsonValue::Object(map) => {
            let mut fields = std::collections::BTreeMap::new();
            for (key, val) in map {
                fields.insert(key, json_to_value(val));
            }
            Value::Struct {
                name: "Json".to_string(),
                fields,
            }
        }
    }
}

fn value_to_json(value: &Value) -> Result<JsonValue, String> {
    match value {
        Value::Void => Ok(JsonValue::Null),
        Value::Bool(b) => Ok(JsonValue::Bool(*b)),
        Value::Int(n) => Ok(JsonValue::Number(*n as f64)),
        Value::Float(n) => {
            if n.is_nan() || n.is_infinite() {
                Ok(JsonValue::Null) // JSON has no NaN/Inf
            } else {
                Ok(JsonValue::Number(*n))
            }
        }
        Value::String(s) => Ok(JsonValue::String((**s).clone())),
        Value::Array(arr) => {
            let items: Result<Vec<JsonValue>, String> =
                arr.iter().map(value_to_json).collect();
            Ok(JsonValue::Array(items?))
        }
        Value::Struct { fields, .. } => {
            let mut map = BTreeMap::new();
            // Use sorted iteration for deterministic output
            let mut sorted_keys: Vec<&String> = fields.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                if let Some(val) = fields.get(key) {
                    map.insert(key.clone(), value_to_json(val)?);
                }
            }
            Ok(JsonValue::Object(map))
        }
        Value::Tuple(items) => {
            let json_items: Result<Vec<JsonValue>, String> =
                items.iter().map(value_to_json).collect();
            Ok(JsonValue::Array(json_items?))
        }
        _ => Err(format!("cannot convert {} to JSON", value.type_name())),
    }
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

fn emit_json(json: &JsonValue) -> String {
    let mut out = String::new();
    emit_value(&mut out, json);
    out
}

fn emit_value(out: &mut String, json: &JsonValue) {
    match json {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(true) => out.push_str("true"),
        JsonValue::Bool(false) => out.push_str("false"),
        JsonValue::Number(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                // Emit integers without decimal point
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{}", n));
            }
        }
        JsonValue::String(s) => emit_string(out, s),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_value(out, item);
            }
            out.push(']');
        }
        JsonValue::Object(map) => {
            out.push('{');
            // BTreeMap iterates in sorted order — deterministic!
            for (i, (key, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_string(out, key);
                out.push(':');
                emit_value(out, val);
            }
            out.push('}');
        }
    }
}

fn emit_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if c < '\u{0020}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_null() {
        let v = json_parse("null").unwrap();
        assert!(matches!(v, Value::Void));
    }

    #[test]
    fn test_parse_bool() {
        assert!(matches!(json_parse("true").unwrap(), Value::Bool(true)));
        assert!(matches!(json_parse("false").unwrap(), Value::Bool(false)));
    }

    #[test]
    fn test_parse_integer() {
        match json_parse("42").unwrap() {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_float() {
        match json_parse("3.14").unwrap() {
            Value::Float(n) => assert!((n - 3.14).abs() < 1e-10),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_string() {
        match json_parse(r#""hello world""#).unwrap() {
            Value::String(s) => assert_eq!(&*s, "hello world"),
            other => panic!("expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_string_escapes() {
        match json_parse(r#""line\nbreak\ttab""#).unwrap() {
            Value::String(s) => assert_eq!(&*s, "line\nbreak\ttab"),
            other => panic!("expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_array() {
        let v = json_parse("[1, 2, 3]").unwrap();
        match v {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert!(matches!(arr[0], Value::Int(1)));
                assert!(matches!(arr[1], Value::Int(2)));
                assert!(matches!(arr[2], Value::Int(3)));
            }
            other => panic!("expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_object_sorted_keys() {
        let v = json_parse(r#"{"z": 1, "a": 2, "m": 3}"#).unwrap();
        match v {
            Value::Struct { name, fields } => {
                assert_eq!(name, "Json");
                assert_eq!(fields.len(), 3);
                assert!(matches!(fields.get("a"), Some(Value::Int(2))));
                assert!(matches!(fields.get("z"), Some(Value::Int(1))));
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_nested() {
        let v = json_parse(r#"{"items": [1, {"nested": true}]}"#).unwrap();
        match v {
            Value::Struct { fields, .. } => {
                match fields.get("items") {
                    Some(Value::Array(arr)) => {
                        assert_eq!(arr.len(), 2);
                    }
                    other => panic!("expected Array in items, got {:?}", other),
                }
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn test_stringify_roundtrip() {
        let input = r#"{"a":1,"b":"hello","c":[true,null]}"#;
        let v = json_parse(input).unwrap();
        let output = json_stringify(&v).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_stringify_sorted_keys() {
        // Keys must be alphabetically sorted in output
        let v = json_parse(r#"{"z":1,"a":2}"#).unwrap();
        let output = json_stringify(&v).unwrap();
        assert_eq!(output, r#"{"a":2,"z":1}"#);
    }

    #[test]
    fn test_roundtrip_empty_object() {
        let v = json_parse("{}").unwrap();
        let output = json_stringify(&v).unwrap();
        assert_eq!(output, "{}");
    }

    #[test]
    fn test_roundtrip_empty_array() {
        let v = json_parse("[]").unwrap();
        let output = json_stringify(&v).unwrap();
        assert_eq!(output, "[]");
    }

    #[test]
    fn test_parse_negative_number() {
        match json_parse("-42").unwrap() {
            Value::Int(n) => assert_eq!(n, -42),
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_scientific_notation() {
        match json_parse("1.5e2").unwrap() {
            Value::Int(n) => assert_eq!(n, 150),
            other => panic!("expected Int(150), got {:?}", other),
        }
    }

    #[test]
    fn test_stringify_nan_becomes_null() {
        let output = json_stringify(&Value::Float(f64::NAN)).unwrap();
        assert_eq!(output, "null");
    }
}
