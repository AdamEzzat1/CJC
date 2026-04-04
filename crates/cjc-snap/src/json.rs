//! Value-to-JSON converter for cross-language interop.
//!
//! Unlike `cjc_runtime::json::json_stringify` (which only handles basic types),
//! this module handles ALL snap-encodable types with tagged wrappers for
//! non-JSON-native types (Tensor, Complex, Enum, Map, Bytes, Bf16, F16).
//!
//! Python consumers can use `json.loads()` on the output.

use cjc_runtime::Value;

/// Convert a CJC `Value` to a JSON string.
///
/// Handles all snap-encodable types. Non-JSON-native types are wrapped in
/// objects with a `"__type"` discriminator so consumers can reconstruct them.
pub fn snap_to_json(value: &Value) -> Result<String, String> {
    let mut buf = String::with_capacity(256);
    write_json(value, &mut buf)?;
    Ok(buf)
}

fn write_json(value: &Value, buf: &mut String) -> Result<(), String> {
    match value {
        Value::Void => buf.push_str("null"),
        Value::Na => buf.push_str("null"),
        Value::Bool(b) => buf.push_str(if *b { "true" } else { "false" }),
        Value::Int(n) => buf.push_str(&n.to_string()),
        Value::Float(f) => {
            if f.is_nan() {
                buf.push_str("\"NaN\"");
            } else if f.is_infinite() {
                if *f > 0.0 {
                    buf.push_str("\"Infinity\"");
                } else {
                    buf.push_str("\"-Infinity\"");
                }
            } else {
                let s = format!("{}", f);
                buf.push_str(&s);
                // Ensure it looks like a float
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    buf.push_str(".0");
                }
            }
        }
        Value::U8(v) => buf.push_str(&v.to_string()),
        Value::String(s) => write_json_string(s.as_str(), buf),
        Value::Bytes(b) => {
            let data = b.borrow();
            buf.push_str("{\"__type\":\"Bytes\",\"hex\":\"");
            for &byte in data.iter() {
                buf.push_str(&format!("{:02x}", byte));
            }
            buf.push_str("\"}");
        }
        Value::ByteSlice(b) => {
            buf.push_str("{\"__type\":\"Bytes\",\"hex\":\"");
            for &byte in b.iter() {
                buf.push_str(&format!("{:02x}", byte));
            }
            buf.push_str("\"}");
        }
        Value::StrView(b) => {
            // StrView is validated UTF-8 — encode as string
            let s = std::str::from_utf8(b).unwrap_or("");
            write_json_string(s, buf);
        }
        Value::Array(arr) => {
            buf.push('[');
            for (i, elem) in arr.iter().enumerate() {
                if i > 0 { buf.push(','); }
                write_json(elem, buf)?;
            }
            buf.push(']');
        }
        Value::Tuple(elems) => {
            // Tuples as JSON arrays
            buf.push('[');
            for (i, elem) in elems.iter().enumerate() {
                if i > 0 { buf.push(','); }
                write_json(elem, buf)?;
            }
            buf.push(']');
        }
        Value::Struct { name, fields } => {
            // Sort fields for determinism
            let mut sorted: Vec<(&String, &Value)> = fields.iter().collect();
            sorted.sort_by_key(|(k, _)| *k);
            buf.push('{');
            buf.push_str("\"__type\":\"Struct\",\"name\":");
            write_json_string(name, buf);
            buf.push_str(",\"fields\":{");
            for (i, (key, val)) in sorted.iter().enumerate() {
                if i > 0 { buf.push(','); }
                write_json_string(key, buf);
                buf.push(':');
                write_json(val, buf)?;
            }
            buf.push_str("}}");
        }
        Value::Tensor(t) => {
            let shape = t.shape();
            let data = t.to_vec();
            buf.push_str("{\"__type\":\"Tensor\",\"shape\":[");
            for (i, &dim) in shape.iter().enumerate() {
                if i > 0 { buf.push(','); }
                buf.push_str(&dim.to_string());
            }
            buf.push_str("],\"data\":[");
            for (i, &val) in data.iter().enumerate() {
                if i > 0 { buf.push(','); }
                if val.is_nan() {
                    buf.push_str("\"NaN\"");
                } else if val.is_infinite() {
                    if val > 0.0 {
                        buf.push_str("\"Infinity\"");
                    } else {
                        buf.push_str("\"-Infinity\"");
                    }
                } else {
                    let s = format!("{}", val);
                    buf.push_str(&s);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        buf.push_str(".0");
                    }
                }
            }
            buf.push_str("]}");
        }
        Value::Enum { enum_name, variant, fields } => {
            buf.push_str("{\"__type\":\"Enum\",\"enum\":");
            write_json_string(enum_name, buf);
            buf.push_str(",\"variant\":");
            write_json_string(variant, buf);
            buf.push_str(",\"fields\":[");
            for (i, field) in fields.iter().enumerate() {
                if i > 0 { buf.push(','); }
                write_json(field, buf)?;
            }
            buf.push_str("]}");
        }
        Value::Complex(z) => {
            buf.push_str("{\"__type\":\"Complex\",\"re\":");
            write_json(&Value::Float(z.re), buf)?;
            buf.push_str(",\"im\":");
            write_json(&Value::Float(z.im), buf)?;
            buf.push('}');
        }
        Value::Bf16(v) => {
            buf.push_str("{\"__type\":\"Bf16\",\"value\":");
            let f = v.to_f32() as f64;
            write_json(&Value::Float(f), buf)?;
            buf.push('}');
        }
        Value::F16(v) => {
            buf.push_str("{\"__type\":\"F16\",\"value\":");
            let f = v.to_f32() as f64;
            write_json(&Value::Float(f), buf)?;
            buf.push('}');
        }
        Value::Map(m) => {
            let map = m.borrow();
            // Sort entries by JSON key for determinism
            let mut entries: Vec<(&Value, &Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| {
                let mut ka = String::new();
                let _ = write_json(a, &mut ka);
                let mut kb = String::new();
                let _ = write_json(b, &mut kb);
                ka.cmp(&kb)
            });
            buf.push_str("{\"__type\":\"Map\",\"entries\":[");
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 { buf.push(','); }
                buf.push_str("{\"key\":");
                write_json(key, buf)?;
                buf.push_str(",\"value\":");
                write_json(val, buf)?;
                buf.push('}');
            }
            buf.push_str("]}");
        }

        // Non-serializable runtime types
        _ => {
            return Err(format!("snap_to_json: cannot convert {} to JSON", value.type_name()));
        }
    }
    Ok(())
}

/// Write a JSON-escaped string (with quotes).
fn write_json_string(s: &str, buf: &mut String) {
    buf.push('"');
    for ch in s.chars() {
        match ch {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                buf.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => buf.push(c),
        }
    }
    buf.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_runtime::Tensor;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    #[test]
    fn test_json_int() {
        let json = snap_to_json(&Value::Int(42)).unwrap();
        assert_eq!(json, "42");
    }

    #[test]
    fn test_json_float() {
        let json = snap_to_json(&Value::Float(3.14)).unwrap();
        assert!(json.starts_with("3.14"));
    }

    #[test]
    fn test_json_bool() {
        assert_eq!(snap_to_json(&Value::Bool(true)).unwrap(), "true");
        assert_eq!(snap_to_json(&Value::Bool(false)).unwrap(), "false");
    }

    #[test]
    fn test_json_string() {
        let json = snap_to_json(&Value::String(Rc::new("hello".into()))).unwrap();
        assert_eq!(json, "\"hello\"");
    }

    #[test]
    fn test_json_string_escapes() {
        let json = snap_to_json(&Value::String(Rc::new("a\"b\\c\n".into()))).unwrap();
        assert_eq!(json, "\"a\\\"b\\\\c\\n\"");
    }

    #[test]
    fn test_json_void() {
        assert_eq!(snap_to_json(&Value::Void).unwrap(), "null");
    }

    #[test]
    fn test_json_array() {
        let val = Value::Array(Rc::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));
        let json = snap_to_json(&val).unwrap();
        assert_eq!(json, "[1,2,3]");
    }

    #[test]
    fn test_json_struct() {
        let mut fields = BTreeMap::new();
        fields.insert("x".to_string(), Value::Float(1.0));
        fields.insert("y".to_string(), Value::Float(2.0));
        let val = Value::Struct { name: "Point".to_string(), fields };
        let json = snap_to_json(&val).unwrap();
        assert!(json.contains("\"__type\":\"Struct\""));
        assert!(json.contains("\"name\":\"Point\""));
        assert!(json.contains("\"x\":"));
        assert!(json.contains("\"y\":"));
    }

    #[test]
    fn test_json_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let json = snap_to_json(&Value::Tensor(t)).unwrap();
        assert!(json.contains("\"__type\":\"Tensor\""));
        assert!(json.contains("\"shape\":[2,2]"));
        assert!(json.contains("\"data\":[1.0,2.0,3.0,4.0]"));
    }

    #[test]
    fn test_json_nan_inf() {
        let json = snap_to_json(&Value::Float(f64::NAN)).unwrap();
        assert_eq!(json, "\"NaN\"");
        let json = snap_to_json(&Value::Float(f64::INFINITY)).unwrap();
        assert_eq!(json, "\"Infinity\"");
    }

    #[test]
    fn test_json_struct_sorted() {
        // Fields must be sorted by name regardless of insertion order
        let mut f1 = BTreeMap::new();
        f1.insert("z".to_string(), Value::Int(3));
        f1.insert("a".to_string(), Value::Int(1));
        let json1 = snap_to_json(&Value::Struct { name: "S".into(), fields: f1 }).unwrap();

        let mut f2 = BTreeMap::new();
        f2.insert("a".to_string(), Value::Int(1));
        f2.insert("z".to_string(), Value::Int(3));
        let json2 = snap_to_json(&Value::Struct { name: "S".into(), fields: f2 }).unwrap();

        assert_eq!(json1, json2, "JSON struct output must be deterministic");
    }
}
