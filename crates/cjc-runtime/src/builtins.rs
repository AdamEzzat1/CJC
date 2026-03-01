//! Shared builtin function implementations for CJC eval and MIR-exec.
//!
//! These are pure (or nearly pure) dispatchers that return
//! `Result<Value, String>`. Each caller wraps the `String` into its own
//! error type (e.g. `EvalError::Runtime` or `MirExecError::Runtime`).
//!
//! **Contract:** Functions in this module must NOT depend on interpreter
//! state. Anything that needs `&mut self` on an interpreter (print, gc_*,
//! clock, Tensor.randn) stays in each executor.

use std::cell::RefCell;
use std::rc::Rc;

use crate::complex::ComplexF64;
use crate::scratchpad::Scratchpad;
use crate::paged_kv::PagedKvCache;
use crate::tensor::Tensor;
use crate::value::{Bf16, Value};

// ---------------------------------------------------------------------------
// Value conversion helpers
// ---------------------------------------------------------------------------

/// Convert a `Value::Array` of ints into a `Vec<usize>` (shape).
pub fn value_to_shape(val: &Value) -> Result<Vec<usize>, String> {
    match val {
        Value::Array(arr) => {
            let mut shape = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                shape.push(value_to_usize(v)?);
            }
            Ok(shape)
        }
        _ => Err(format!("expected Array for shape, got {}", val.type_name())),
    }
}

/// Convert a `Value::Int` to `usize`, rejecting negatives.
pub fn value_to_usize(val: &Value) -> Result<usize, String> {
    match val {
        Value::Int(i) => {
            if *i < 0 {
                Err(format!("expected non-negative integer, got {i}"))
            } else {
                Ok(*i as usize)
            }
        }
        _ => Err(format!("expected Int, got {}", val.type_name())),
    }
}

/// Convert a `Value::Array` of floats/ints to `Vec<f64>`.
pub fn value_to_f64_vec(val: &Value) -> Result<Vec<f64>, String> {
    match val {
        Value::Array(arr) => {
            let mut data = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                match v {
                    Value::Float(f) => data.push(*f),
                    Value::Int(i) => data.push(*i as f64),
                    _ => {
                        return Err(format!(
                            "expected numeric values in array, got {}",
                            v.type_name()
                        ));
                    }
                }
            }
            Ok(data)
        }
        _ => Err(format!("expected Array, got {}", val.type_name())),
    }
}

/// Convert a `Value::Array` of ints to `Vec<usize>`.
pub fn value_to_usize_vec(val: &Value) -> Result<Vec<usize>, String> {
    match val {
        Value::Array(arr) => {
            let mut indices = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                indices.push(value_to_usize(v)?);
            }
            Ok(indices)
        }
        _ => Err(format!("expected Array for indices, got {}", val.type_name())),
    }
}

/// Extract a `&Tensor` from a `Value::Tensor`.
pub fn value_to_tensor(val: &Value) -> Result<&Tensor, String> {
    match val {
        Value::Tensor(t) => Ok(t),
        _ => Err(format!("expected Tensor, got {}", val.type_name())),
    }
}

/// Extract bytes from a `ByteSlice`, `Bytes`, or `String` value.
pub fn value_to_bytes(val: &Value) -> Result<Vec<u8>, String> {
    match val {
        Value::ByteSlice(b) => Ok(b.as_ref().clone()),
        Value::Bytes(b) => Ok(b.borrow().clone()),
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        _ => Err(format!("expected ByteSlice or Bytes, got {}", val.type_name())),
    }
}

/// Structural equality comparison for assertion builtins.
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Void, Value::Void) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Stateless builtin functions
// ---------------------------------------------------------------------------

/// Dispatch a stateless builtin function by name.
/// Returns `Ok(Some(value))` if handled, `Ok(None)` if not a known builtin.
pub fn dispatch_builtin(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    match name {
        "Complex" => {
            let re = match args.get(0) {
                Some(Value::Float(v)) => *v,
                Some(Value::Int(v)) => *v as f64,
                _ => return Err("Complex() requires numeric re argument".into()),
            };
            let im = match args.get(1) {
                Some(Value::Float(v)) => *v,
                Some(Value::Int(v)) => *v as f64,
                None => 0.0,
                _ => return Err("Complex() requires numeric im argument".into()),
            };
            Ok(Some(Value::Complex(ComplexF64::new(re, im))))
        }
        // f16 conversion builtins
        "f16_to_f64" => match &args[0] {
            Value::F16(v) => Ok(Some(Value::Float(v.to_f64()))),
            _ => Err("f16_to_f64 expects f16".into()),
        },
        "f64_to_f16" => match &args[0] {
            Value::Float(v) => Ok(Some(Value::F16(crate::f16::F16::from_f64(*v)))),
            Value::Int(v) => Ok(Some(Value::F16(crate::f16::F16::from_f64(*v as f64)))),
            _ => Err("f64_to_f16 expects f64".into()),
        },
        "f16_to_f32" => match &args[0] {
            Value::F16(v) => Ok(Some(Value::Float(v.to_f32() as f64))),
            _ => Err("f16_to_f32 expects f16".into()),
        },
        "f32_to_f16" => match &args[0] {
            Value::Float(v) => Ok(Some(Value::F16(crate::f16::F16::from_f32(*v as f32)))),
            _ => Err("f32_to_f16 expects f32".into()),
        },
        // bf16 conversion builtins
        "bf16_to_f32" => match &args[0] {
            Value::Bf16(v) => Ok(Some(Value::Float(v.to_f32() as f64))),
            _ => Err("bf16_to_f32 expects bf16".into()),
        },
        "f32_to_bf16" => match &args[0] {
            Value::Float(v) => Ok(Some(Value::Bf16(Bf16::from_f32(*v as f32)))),
            _ => Err("f32_to_bf16 expects f32".into()),
        },
        // Tensor constructors (stateless ones)
        "Tensor.zeros" => {
            let shape = value_to_shape(&args[0])?;
            Ok(Some(Value::Tensor(Tensor::zeros(&shape))))
        }
        "Tensor.ones" => {
            let shape = value_to_shape(&args[0])?;
            Ok(Some(Value::Tensor(Tensor::ones(&shape))))
        }
        "Tensor.from_vec" => {
            if args.len() != 2 {
                return Err("Tensor.from_vec requires 2 arguments: data and shape".into());
            }
            let data = value_to_f64_vec(&args[0])?;
            let shape = value_to_shape(&args[1])?;
            let t = Tensor::from_vec(data, &shape).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tensor(t)))
        }
        "matmul" => {
            if args.len() != 2 {
                return Err("matmul requires 2 Tensor arguments".into());
            }
            let a = value_to_tensor(&args[0])?;
            let b = value_to_tensor(&args[1])?;
            Ok(Some(Value::Tensor(a.matmul(b).map_err(|e| format!("{e}"))?)))
        }
        "attention" => {
            if args.len() != 3 {
                return Err("attention requires 3 Tensor arguments: queries, keys, values".into());
            }
            let q = value_to_tensor(&args[0])?;
            let k = value_to_tensor(&args[1])?;
            let v = value_to_tensor(&args[2])?;
            Ok(Some(Value::Tensor(
                Tensor::scaled_dot_product_attention(q, k, v).map_err(|e| format!("{e}"))?,
            )))
        }
        "Buffer.alloc" => {
            if args.is_empty() {
                return Err("Buffer.alloc requires a length argument".into());
            }
            let len = value_to_usize(&args[0])?;
            Ok(Some(Value::Tensor(Tensor::zeros(&[len]))))
        }
        "Tensor.from_bytes" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(
                    "Tensor.from_bytes requires 2-3 arguments: bytes, shape, [dtype='f64']".into(),
                );
            }
            let bytes = match &args[0] {
                Value::ByteSlice(b) => b.clone(),
                Value::Bytes(b) => Rc::new(b.borrow().clone()),
                _ => {
                    return Err(
                        "Tensor.from_bytes: first argument must be ByteSlice or Bytes".into(),
                    )
                }
            };
            let shape = value_to_shape(&args[1])?;
            let dtype = if args.len() == 3 {
                match &args[2] {
                    Value::String(s) => s.as_str().to_string(),
                    _ => return Err("Tensor.from_bytes: dtype must be a string".into()),
                }
            } else {
                "f64".to_string()
            };
            Ok(Some(Value::Tensor(
                Tensor::from_bytes(&bytes, &shape, &dtype).map_err(|e| format!("{e}"))?,
            )))
        }
        "Scratchpad.new" => {
            if args.len() != 2 {
                return Err("Scratchpad.new requires 2 arguments: max_seq_len, dim".into());
            }
            let max_seq_len = value_to_usize(&args[0])?;
            let dim = value_to_usize(&args[1])?;
            Ok(Some(Value::Scratchpad(Rc::new(RefCell::new(
                Scratchpad::new(max_seq_len, dim),
            )))))
        }
        "PagedKvCache.new" => {
            if args.len() != 2 {
                return Err("PagedKvCache.new requires 2 arguments: max_tokens, dim".into());
            }
            let max_tokens = value_to_usize(&args[0])?;
            let dim = value_to_usize(&args[1])?;
            Ok(Some(Value::PagedKvCache(Rc::new(RefCell::new(
                PagedKvCache::new(max_tokens, dim),
            )))))
        }
        "AlignedByteSlice.from_bytes" => {
            if args.len() != 1 {
                return Err("AlignedByteSlice.from_bytes requires 1 argument: bytes".into());
            }
            let bytes = match &args[0] {
                Value::ByteSlice(b) => b.clone(),
                Value::Bytes(b) => Rc::new(b.borrow().clone()),
                _ => {
                    return Err(
                        "AlignedByteSlice.from_bytes: argument must be ByteSlice or Bytes".into(),
                    )
                }
            };
            Ok(Some(Value::AlignedBytes(
                crate::aligned_pool::AlignedByteSlice::from_bytes(bytes),
            )))
        }
        "to_string" => {
            if args.len() != 1 {
                return Err("to_string requires exactly 1 argument".into());
            }
            Ok(Some(Value::String(Rc::new(format!("{}", args[0])))))
        }
        "len" => {
            if args.len() != 1 {
                return Err("len requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Array(arr) => Ok(Some(Value::Int(arr.len() as i64))),
                Value::String(s) => Ok(Some(Value::Int(s.len() as i64))),
                Value::Tensor(t) => Ok(Some(Value::Int(t.len() as i64))),
                Value::Tuple(t) => Ok(Some(Value::Int(t.len() as i64))),
                other => Err(format!("len not supported for {}", other.type_name())),
            }
        }
        "push" => {
            if args.len() != 2 {
                return Err("push requires 2 arguments: array and value".into());
            }
            match (&args[0], &args[1]) {
                (Value::Array(a), val) => {
                    let mut new_arr = (**a).clone();
                    new_arr.push(val.clone());
                    Ok(Some(Value::Array(Rc::new(new_arr))))
                }
                _ => Err("push requires an Array as first argument".into()),
            }
        }
        "sort" => {
            if args.len() != 1 {
                return Err("sort requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Array(arr) => {
                    let mut sorted: Vec<Value> = (**arr).clone();
                    sorted.sort_by(|a, b| {
                        let fa = match a {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f64,
                            _ => f64::NAN,
                        };
                        let fb = match b {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f64,
                            _ => f64::NAN,
                        };
                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    Ok(Some(Value::Array(Rc::new(sorted))))
                }
                _ => Err(format!("sort requires an Array, got {}", args[0].type_name())),
            }
        }
        "sqrt" => {
            if args.len() != 1 {
                return Err("sqrt requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.sqrt()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).sqrt()))),
                _ => Err(format!("sqrt requires a number, got {}", args[0].type_name())),
            }
        }
        "floor" => {
            if args.len() != 1 {
                return Err("floor requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.floor()))),
                Value::Int(i) => Ok(Some(Value::Int(*i))),
                _ => Err(format!("floor requires a number, got {}", args[0].type_name())),
            }
        }
        "int" => {
            if args.len() != 1 {
                return Err("int requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Int(*f as i64))),
                Value::Int(i) => Ok(Some(Value::Int(*i))),
                _ => Err(format!("int requires a number, got {}", args[0].type_name())),
            }
        }
        "float" => {
            if args.len() != 1 {
                return Err("float requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Int(i) => Ok(Some(Value::Float(*i as f64))),
                Value::Float(f) => Ok(Some(Value::Float(*f))),
                _ => Err(format!("float requires a number, got {}", args[0].type_name())),
            }
        }
        "isnan" => {
            if args.len() != 1 {
                return Err("isnan requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Bool(f.is_nan()))),
                Value::Int(_) => Ok(Some(Value::Bool(false))),
                _ => Err(format!("isnan requires a number, got {}", args[0].type_name())),
            }
        }
        "isinf" => {
            if args.len() != 1 {
                return Err("isinf requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Bool(f.is_infinite()))),
                Value::Int(_) => Ok(Some(Value::Bool(false))),
                _ => Err(format!("isinf requires a number, got {}", args[0].type_name())),
            }
        }
        "abs" => {
            if args.len() != 1 {
                return Err("abs requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.abs()))),
                Value::Int(i) => Ok(Some(Value::Int(i.abs()))),
                _ => Err(format!("abs requires a number, got {}", args[0].type_name())),
            }
        }
        "assert" => {
            if args.len() != 1 {
                return Err("assert requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Bool(true) => Ok(Some(Value::Void)),
                Value::Bool(false) => Err("assertion failed".into()),
                other => Err(format!("assert requires Bool, got {}", other.type_name())),
            }
        }
        "assert_eq" => {
            if args.len() != 2 {
                return Err("assert_eq requires exactly 2 arguments".into());
            }
            if values_equal(&args[0], &args[1]) {
                Ok(Some(Value::Void))
            } else {
                Err(format!("assertion failed: `{}` != `{}`", args[0], args[1]))
            }
        }
        // ── JSON builtins ──────────────────────────────────────────
        "json_parse" => {
            if args.len() != 1 {
                return Err("json_parse requires exactly 1 argument".into());
            }
            let s = match &args[0] {
                Value::String(s) => s.as_str(),
                _ => return Err(format!("json_parse requires String, got {}", args[0].type_name())),
            };
            crate::json::json_parse(s).map(Some)
        }
        "json_stringify" => {
            if args.len() != 1 {
                return Err("json_stringify requires exactly 1 argument".into());
            }
            crate::json::json_stringify(&args[0]).map(|s| Some(Value::String(Rc::new(s))))
        }

        // ── DateTime builtins (pure arithmetic, except datetime_now) ──
        "datetime_from_epoch" => {
            if args.len() != 1 {
                return Err("datetime_from_epoch requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_from_epoch(*n)))),
                _ => Err(format!("datetime_from_epoch requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_from_parts" => {
            if args.len() != 6 {
                return Err("datetime_from_parts requires 6 arguments (year, month, day, hour, min, sec)".into());
            }
            let mut vals = [0i64; 6];
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    Value::Int(n) => vals[i] = *n,
                    _ => return Err(format!("datetime_from_parts arg {} must be Int", i)),
                }
            }
            Ok(Some(Value::Int(crate::datetime::datetime_from_parts(
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
            ))))
        }
        "datetime_year" => {
            if args.len() != 1 { return Err("datetime_year requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_year(*n)))),
                _ => Err(format!("datetime_year requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_month" => {
            if args.len() != 1 { return Err("datetime_month requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_month(*n)))),
                _ => Err(format!("datetime_month requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_day" => {
            if args.len() != 1 { return Err("datetime_day requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_day(*n)))),
                _ => Err(format!("datetime_day requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_hour" => {
            if args.len() != 1 { return Err("datetime_hour requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_hour(*n)))),
                _ => Err(format!("datetime_hour requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_minute" => {
            if args.len() != 1 { return Err("datetime_minute requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_minute(*n)))),
                _ => Err(format!("datetime_minute requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_second" => {
            if args.len() != 1 { return Err("datetime_second requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::Int(crate::datetime::datetime_second(*n)))),
                _ => Err(format!("datetime_second requires Int, got {}", args[0].type_name())),
            }
        }
        "datetime_diff" => {
            if args.len() != 2 { return Err("datetime_diff requires 2 arguments".into()); }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Some(Value::Int(crate::datetime::datetime_diff(*a, *b)))),
                _ => Err("datetime_diff requires two Int arguments".into()),
            }
        }
        "datetime_add_millis" => {
            if args.len() != 2 { return Err("datetime_add_millis requires 2 arguments".into()); }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Some(Value::Int(crate::datetime::datetime_add_millis(*a, *b)))),
                _ => Err("datetime_add_millis requires two Int arguments".into()),
            }
        }
        "datetime_format" => {
            if args.len() != 1 { return Err("datetime_format requires 1 argument".into()); }
            match &args[0] {
                Value::Int(n) => Ok(Some(Value::String(Rc::new(crate::datetime::datetime_format(*n))))),
                _ => Err(format!("datetime_format requires Int, got {}", args[0].type_name())),
            }
        }

        // ── File I/O builtins ─────────────────────────────────────────
        "file_read" => {
            if args.len() != 1 { return Err("file_read requires 1 argument".into()); }
            match &args[0] {
                Value::String(path) => {
                    let content = std::fs::read_to_string(path.as_str())
                        .map_err(|e| format!("file_read error: {}", e))?;
                    Ok(Some(Value::String(Rc::new(content))))
                }
                _ => Err(format!("file_read requires String path, got {}", args[0].type_name())),
            }
        }
        "file_write" => {
            if args.len() != 2 { return Err("file_write requires 2 arguments (path, content)".into()); }
            match (&args[0], &args[1]) {
                (Value::String(path), Value::String(content)) => {
                    std::fs::write(path.as_str(), content.as_str())
                        .map_err(|e| format!("file_write error: {}", e))?;
                    Ok(Some(Value::Void))
                }
                _ => Err("file_write requires (String, String) arguments".into()),
            }
        }
        "file_exists" => {
            if args.len() != 1 { return Err("file_exists requires 1 argument".into()); }
            match &args[0] {
                Value::String(path) => Ok(Some(Value::Bool(std::path::Path::new(path.as_str()).exists()))),
                _ => Err(format!("file_exists requires String path, got {}", args[0].type_name())),
            }
        }
        "file_lines" => {
            if args.len() != 1 { return Err("file_lines requires 1 argument".into()); }
            match &args[0] {
                Value::String(path) => {
                    let content = std::fs::read_to_string(path.as_str())
                        .map_err(|e| format!("file_lines error: {}", e))?;
                    let lines: Vec<Value> = content
                        .lines()
                        .map(|l| Value::String(Rc::new(l.to_string())))
                        .collect();
                    Ok(Some(Value::Array(Rc::new(lines))))
                }
                _ => Err(format!("file_lines requires String path, got {}", args[0].type_name())),
            }
        }

        // ── Window function builtins ──────────────────────────────────
        "window_sum" | "window_mean" | "window_min" | "window_max" => {
            if args.len() != 2 {
                return Err(format!("{name} requires 2 arguments (array, window_size)"));
            }
            let data = value_to_f64_vec(&args[0])?;
            let ws = match &args[1] {
                Value::Int(i) => {
                    if *i < 0 {
                        return Err(format!("{name}: window_size must be non-negative, got {i}"));
                    }
                    *i as usize
                }
                _ => return Err(format!("{name} requires Int window_size, got {}", args[1].type_name())),
            };
            let result = match name {
                "window_sum" => crate::window::window_sum(&data, ws),
                "window_mean" => crate::window::window_mean(&data, ws),
                "window_min" => crate::window::window_min(&data, ws),
                "window_max" => crate::window::window_max(&data, ws),
                _ => unreachable!(),
            };
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }

        // ── Stats builtins ───────────────────────────────────────────
        "variance" => {
            if args.len() != 1 { return Err("variance requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::variance(&data)?)))
        }
        "sd" => {
            if args.len() != 1 { return Err("sd requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::sd(&data)?)))
        }
        "se" => {
            if args.len() != 1 { return Err("se requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::se(&data)?)))
        }
        "median" => {
            if args.len() != 1 { return Err("median requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::median(&data)?)))
        }
        "quantile" => {
            if args.len() != 2 { return Err("quantile requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let p = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("quantile: p must be a number".into()),
            };
            Ok(Some(Value::Float(crate::stats::quantile(&data, p)?)))
        }
        "iqr" => {
            if args.len() != 1 { return Err("iqr requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::iqr(&data)?)))
        }
        "skewness" => {
            if args.len() != 1 { return Err("skewness requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::skewness(&data)?)))
        }
        "kurtosis" => {
            if args.len() != 1 { return Err("kurtosis requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::kurtosis(&data)?)))
        }
        "z_score" => {
            if args.len() != 1 { return Err("z_score requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::z_score(&data)?;
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "standardize" => {
            if args.len() != 1 { return Err("standardize requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::standardize(&data)?;
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "n_distinct" => {
            if args.len() != 1 { return Err("n_distinct requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Int(crate::stats::n_distinct(&data) as i64)))
        }
        // ── Correlation builtins ─────────────────────────────────────
        "cor" => {
            if args.len() != 2 { return Err("cor requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::cor(&x, &y)?)))
        }
        "cov" => {
            if args.len() != 2 { return Err("cov requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::cov(&x, &y)?)))
        }
        // ── Distribution builtins ────────────────────────────────────
        "normal_cdf" => {
            if args.len() != 1 { return Err("normal_cdf requires 1 argument".into()); }
            let x = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("normal_cdf requires a number".into()),
            };
            Ok(Some(Value::Float(crate::distributions::normal_cdf(x))))
        }
        "normal_pdf" => {
            if args.len() != 1 { return Err("normal_pdf requires 1 argument".into()); }
            let x = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("normal_pdf requires a number".into()),
            };
            Ok(Some(Value::Float(crate::distributions::normal_pdf(x))))
        }
        "normal_ppf" => {
            if args.len() != 1 { return Err("normal_ppf requires 1 argument".into()); }
            let p = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("normal_ppf requires a number".into()),
            };
            Ok(Some(Value::Float(crate::distributions::normal_ppf(p)?)))
        }
        "t_cdf" => {
            if args.len() != 2 { return Err("t_cdf requires 2 arguments (x, df)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_cdf: x must be a number".into()) };
            let df = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_cdf: df must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::t_cdf(x, df))))
        }
        "chi2_cdf" => {
            if args.len() != 2 { return Err("chi2_cdf requires 2 arguments (x, df)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("chi2_cdf: x must be a number".into()) };
            let df = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("chi2_cdf: df must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::chi2_cdf(x, df))))
        }
        "f_cdf" => {
            if args.len() != 3 { return Err("f_cdf requires 3 arguments (x, df1, df2)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("f_cdf: x must be a number".into()) };
            let df1 = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("f_cdf: df1 must be a number".into()) };
            let df2 = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("f_cdf: df2 must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::f_cdf(x, df1, df2))))
        }
        // ── Hypothesis test builtins ─────────────────────────────────
        "t_test" => {
            if args.len() != 2 { return Err("t_test requires 2 arguments (data, mu)".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let mu = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_test: mu must be a number".into()) };
            let r = crate::hypothesis::t_test(&data, mu)?;
            let mut fields = std::collections::HashMap::new();
            fields.insert("t_statistic".into(), Value::Float(r.t_statistic));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            fields.insert("df".into(), Value::Float(r.df));
            fields.insert("mean".into(), Value::Float(r.mean));
            fields.insert("se".into(), Value::Float(r.se));
            Ok(Some(Value::Struct { name: "TTestResult".into(), fields }))
        }
        "t_test_two_sample" => {
            if args.len() != 2 { return Err("t_test_two_sample requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let r = crate::hypothesis::t_test_two_sample(&x, &y)?;
            let mut fields = std::collections::HashMap::new();
            fields.insert("t_statistic".into(), Value::Float(r.t_statistic));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            fields.insert("df".into(), Value::Float(r.df));
            Ok(Some(Value::Struct { name: "TTestResult".into(), fields }))
        }
        "chi_squared_test" => {
            if args.len() != 2 { return Err("chi_squared_test requires 2 arguments".into()); }
            let obs = value_to_f64_vec(&args[0])?;
            let exp = value_to_f64_vec(&args[1])?;
            let r = crate::hypothesis::chi_squared_test(&obs, &exp)?;
            let mut fields = std::collections::HashMap::new();
            fields.insert("chi2".into(), Value::Float(r.chi2));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            fields.insert("df".into(), Value::Float(r.df));
            Ok(Some(Value::Struct { name: "ChiSquaredResult".into(), fields }))
        }
        // ── Linalg builtins ──────────────────────────────────────────
        "det" => {
            if args.len() != 1 { return Err("det requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Float(t.det().map_err(|e| format!("{e}"))?)))
        }
        "solve" => {
            if args.len() != 2 { return Err("solve requires 2 Tensor arguments".into()); }
            let a = value_to_tensor(&args[0])?;
            let b = value_to_tensor(&args[1])?;
            Ok(Some(Value::Tensor(a.solve(b).map_err(|e| format!("{e}"))?)))
        }
        "lstsq" => {
            if args.len() != 2 { return Err("lstsq requires 2 Tensor arguments".into()); }
            let a = value_to_tensor(&args[0])?;
            let b = value_to_tensor(&args[1])?;
            Ok(Some(Value::Tensor(a.lstsq(b).map_err(|e| format!("{e}"))?)))
        }
        "trace" => {
            if args.len() != 1 { return Err("trace requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Float(t.trace().map_err(|e| format!("{e}"))?)))
        }
        "norm_frobenius" => {
            if args.len() != 1 { return Err("norm_frobenius requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Float(t.norm_frobenius().map_err(|e| format!("{e}"))?)))
        }
        "eigh" => {
            if args.len() != 1 { return Err("eigh requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            let (vals, vecs) = t.eigh().map_err(|e| format!("{e}"))?;
            let val_values: Vec<Value> = vals.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Tuple(Rc::new(vec![
                Value::Array(Rc::new(val_values)),
                Value::Tensor(vecs),
            ]))))
        }
        "matrix_rank" => {
            if args.len() != 1 { return Err("matrix_rank requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Int(t.matrix_rank().map_err(|e| format!("{e}"))? as i64)))
        }
        "kron" => {
            if args.len() != 2 { return Err("kron requires 2 Tensor arguments".into()); }
            let a = value_to_tensor(&args[0])?;
            let b = value_to_tensor(&args[1])?;
            Ok(Some(Value::Tensor(a.kron(b).map_err(|e| format!("{e}"))?)))
        }
        // ── ML builtins ──────────────────────────────────────────────
        "mse_loss" => {
            if args.len() != 2 { return Err("mse_loss requires 2 arguments".into()); }
            let pred = value_to_f64_vec(&args[0])?;
            let target = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::ml::mse_loss(&pred, &target)?)))
        }
        "cross_entropy_loss" => {
            if args.len() != 2 { return Err("cross_entropy_loss requires 2 arguments".into()); }
            let pred = value_to_f64_vec(&args[0])?;
            let target = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::ml::cross_entropy_loss(&pred, &target)?)))
        }
        "huber_loss" => {
            if args.len() != 3 { return Err("huber_loss requires 3 arguments".into()); }
            let pred = value_to_f64_vec(&args[0])?;
            let target = value_to_f64_vec(&args[1])?;
            let delta = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("huber_loss: delta must be a number".into()) };
            Ok(Some(Value::Float(crate::ml::huber_loss(&pred, &target, delta)?)))
        }
        // ── Cumulative builtins ──────────────────────────────────────
        "cumsum" => {
            if args.len() != 1 { return Err("cumsum requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::cumsum(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "cumprod" => {
            if args.len() != 1 { return Err("cumprod requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::cumprod(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "cummax" => {
            if args.len() != 1 { return Err("cummax requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::cummax(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "cummin" => {
            if args.len() != 1 { return Err("cummin requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::cummin(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "lag" => {
            if args.len() != 2 { return Err("lag requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let n = value_to_usize(&args[1])?;
            let result = crate::stats::lag(&data, n);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "lead" => {
            if args.len() != 2 { return Err("lead requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let n = value_to_usize(&args[1])?;
            let result = crate::stats::lead(&data, n);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "rank" => {
            if args.len() != 1 { return Err("rank requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::rank(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "dense_rank" => {
            if args.len() != 1 { return Err("dense_rank requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::dense_rank(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "histogram" => {
            if args.len() != 2 { return Err("histogram requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let n_bins = value_to_usize(&args[1])?;
            let (edges, counts) = crate::stats::histogram(&data, n_bins)?;
            let edge_values: Vec<Value> = edges.into_iter().map(Value::Float).collect();
            let count_values: Vec<Value> = counts.into_iter().map(|c| Value::Int(c as i64)).collect();
            Ok(Some(Value::Tuple(Rc::new(vec![
                Value::Array(Rc::new(edge_values)),
                Value::Array(Rc::new(count_values)),
            ]))))
        }
        // ── Additional stats builtins ───────────────────────────────
        "sample_variance" => {
            if args.len() != 1 { return Err("sample_variance requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::sample_variance(&data)?)))
        }
        "sample_sd" => {
            if args.len() != 1 { return Err("sample_sd requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::sample_sd(&data)?)))
        }
        "sample_cov" => {
            if args.len() != 2 { return Err("sample_cov requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::sample_cov(&x, &y)?)))
        }
        "row_number" => {
            if args.len() != 1 { return Err("row_number requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::row_number(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        // ── Distribution PPF builtins ───────────────────────────────
        "t_ppf" => {
            if args.len() != 2 { return Err("t_ppf requires 2 arguments (p, df)".into()); }
            let p = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_ppf: p must be a number".into()) };
            let df = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_ppf: df must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::t_ppf(p, df)?)))
        }
        "chi2_ppf" => {
            if args.len() != 2 { return Err("chi2_ppf requires 2 arguments (p, df)".into()); }
            let p = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("chi2_ppf: p must be a number".into()) };
            let df = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("chi2_ppf: df must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::chi2_ppf(p, df)?)))
        }
        "f_ppf" => {
            if args.len() != 3 { return Err("f_ppf requires 3 arguments (p, df1, df2)".into()); }
            let p = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("f_ppf: p must be a number".into()) };
            let df1 = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("f_ppf: df1 must be a number".into()) };
            let df2 = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("f_ppf: df2 must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::f_ppf(p, df1, df2)?)))
        }
        // ── Discrete distribution builtins ──────────────────────────
        "binomial_pmf" => {
            if args.len() != 3 { return Err("binomial_pmf requires 3 arguments (k, n, p)".into()); }
            let k = value_to_usize(&args[0])? as u64;
            let n = value_to_usize(&args[1])? as u64;
            let p = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("binomial_pmf: p must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::binomial_pmf(k, n, p))))
        }
        "binomial_cdf" => {
            if args.len() != 3 { return Err("binomial_cdf requires 3 arguments (k, n, p)".into()); }
            let k = value_to_usize(&args[0])? as u64;
            let n = value_to_usize(&args[1])? as u64;
            let p = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("binomial_cdf: p must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::binomial_cdf(k, n, p))))
        }
        "poisson_pmf" => {
            if args.len() != 2 { return Err("poisson_pmf requires 2 arguments (k, lambda)".into()); }
            let k = value_to_usize(&args[0])? as u64;
            let lambda = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("poisson_pmf: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::poisson_pmf(k, lambda))))
        }
        "poisson_cdf" => {
            if args.len() != 2 { return Err("poisson_cdf requires 2 arguments (k, lambda)".into()); }
            let k = value_to_usize(&args[0])? as u64;
            let lambda = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("poisson_cdf: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::poisson_cdf(k, lambda))))
        }
        // ── Hypothesis test builtins (additional) ───────────────────
        "t_test_paired" => {
            if args.len() != 2 { return Err("t_test_paired requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let r = crate::hypothesis::t_test_paired(&x, &y)?;
            let mut fields = std::collections::HashMap::new();
            fields.insert("t_statistic".into(), Value::Float(r.t_statistic));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            fields.insert("df".into(), Value::Float(r.df));
            Ok(Some(Value::Struct { name: "TTestResult".into(), fields }))
        }
        "anova_oneway" => {
            if args.len() < 2 { return Err("anova_oneway requires at least 2 group arguments".into()); }
            let mut groups = Vec::new();
            let mut group_vecs = Vec::new();
            for a in args.iter() {
                group_vecs.push(value_to_f64_vec(a)?);
            }
            for gv in &group_vecs {
                groups.push(gv.as_slice());
            }
            let r = crate::hypothesis::anova_oneway(&groups)?;
            let mut fields = std::collections::HashMap::new();
            fields.insert("f_statistic".into(), Value::Float(r.f_statistic));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            fields.insert("df_between".into(), Value::Float(r.df_between));
            fields.insert("df_within".into(), Value::Float(r.df_within));
            fields.insert("ss_between".into(), Value::Float(r.ss_between));
            fields.insert("ss_within".into(), Value::Float(r.ss_within));
            Ok(Some(Value::Struct { name: "AnovaResult".into(), fields }))
        }
        "f_test" => {
            if args.len() != 2 { return Err("f_test requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let (f_stat, p_val) = crate::hypothesis::f_test(&x, &y)?;
            let mut fields = std::collections::HashMap::new();
            fields.insert("f_statistic".into(), Value::Float(f_stat));
            fields.insert("p_value".into(), Value::Float(p_val));
            Ok(Some(Value::Struct { name: "FTestResult".into(), fields }))
        }
        "lm" => {
            // lm(X, y, n, p) — linear model
            if args.len() != 4 { return Err("lm requires 4 arguments (X_flat, y, n, p)".into()); }
            let x_flat = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let n = value_to_usize(&args[2])?;
            let p = value_to_usize(&args[3])?;
            let r = crate::hypothesis::lm(&x_flat, &y, n, p)?;
            let coef_values: Vec<Value> = r.coefficients.into_iter().map(Value::Float).collect();
            let se_values: Vec<Value> = r.std_errors.into_iter().map(Value::Float).collect();
            let t_values: Vec<Value> = r.t_values.into_iter().map(Value::Float).collect();
            let p_values: Vec<Value> = r.p_values.into_iter().map(Value::Float).collect();
            let resid_values: Vec<Value> = r.residuals.into_iter().map(Value::Float).collect();
            let mut fields = std::collections::HashMap::new();
            fields.insert("coefficients".into(), Value::Array(Rc::new(coef_values)));
            fields.insert("std_errors".into(), Value::Array(Rc::new(se_values)));
            fields.insert("t_values".into(), Value::Array(Rc::new(t_values)));
            fields.insert("p_values".into(), Value::Array(Rc::new(p_values)));
            fields.insert("r_squared".into(), Value::Float(r.r_squared));
            fields.insert("adj_r_squared".into(), Value::Float(r.adj_r_squared));
            fields.insert("residuals".into(), Value::Array(Rc::new(resid_values)));
            fields.insert("f_statistic".into(), Value::Float(r.f_statistic));
            Ok(Some(Value::Struct { name: "LmResult".into(), fields }))
        }
        // ── ML builtins (additional) ────────────────────────────────
        "binary_cross_entropy" => {
            if args.len() != 2 { return Err("binary_cross_entropy requires 2 arguments".into()); }
            let pred = value_to_f64_vec(&args[0])?;
            let target = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::ml::binary_cross_entropy(&pred, &target)?)))
        }
        "hinge_loss" => {
            if args.len() != 2 { return Err("hinge_loss requires 2 arguments".into()); }
            let pred = value_to_f64_vec(&args[0])?;
            let target = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::ml::hinge_loss(&pred, &target)?)))
        }
        "confusion_matrix" => {
            if args.len() != 2 { return Err("confusion_matrix requires 2 arguments".into()); }
            let pred = value_to_f64_vec(&args[0])?;
            let actual = value_to_f64_vec(&args[1])?;
            let pred_bool: Vec<bool> = pred.iter().map(|&x| x > 0.5).collect();
            let actual_bool: Vec<bool> = actual.iter().map(|&x| x > 0.5).collect();
            let cm = crate::ml::confusion_matrix(&pred_bool, &actual_bool);
            let mut fields = std::collections::HashMap::new();
            fields.insert("tp".into(), Value::Int(cm.tp as i64));
            fields.insert("fp".into(), Value::Int(cm.fp as i64));
            fields.insert("tn".into(), Value::Int(cm.tn as i64));
            fields.insert("fn_count".into(), Value::Int(cm.fn_count as i64));
            fields.insert("precision".into(), Value::Float(crate::ml::precision(&cm)));
            fields.insert("recall".into(), Value::Float(crate::ml::recall(&cm)));
            fields.insert("f1_score".into(), Value::Float(crate::ml::f1_score(&cm)));
            fields.insert("accuracy".into(), Value::Float(crate::ml::accuracy(&cm)));
            Ok(Some(Value::Struct { name: "ConfusionMatrix".into(), fields }))
        }
        "auc_roc" => {
            if args.len() != 2 { return Err("auc_roc requires 2 arguments (scores, labels)".into()); }
            let scores = value_to_f64_vec(&args[0])?;
            let labels_f = value_to_f64_vec(&args[1])?;
            let labels: Vec<bool> = labels_f.iter().map(|&x| x > 0.5).collect();
            Ok(Some(Value::Float(crate::ml::auc_roc(&scores, &labels)?)))
        }
        // ── Tensor activation builtins ──────────────────────────────
        "sigmoid" => {
            if args.len() != 1 { return Err("sigmoid requires 1 argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.sigmoid())))
        }
        "tanh_activation" => {
            if args.len() != 1 { return Err("tanh_activation requires 1 argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.tanh_activation())))
        }
        "leaky_relu" => {
            if args.len() != 2 { return Err("leaky_relu requires 2 arguments (tensor, alpha)".into()); }
            let t = value_to_tensor(&args[0])?;
            let alpha = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("leaky_relu: alpha must be a number".into()) };
            Ok(Some(Value::Tensor(t.leaky_relu(alpha))))
        }
        "silu" => {
            if args.len() != 1 { return Err("silu requires 1 argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.silu())))
        }
        "mish" => {
            if args.len() != 1 { return Err("mish requires 1 argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.mish())))
        }
        "argmax" => {
            if args.len() != 1 { return Err("argmax requires 1 argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Int(t.argmax() as i64)))
        }
        "argmin" => {
            if args.len() != 1 { return Err("argmin requires 1 argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Int(t.argmin() as i64)))
        }
        "clamp" => {
            if args.len() != 3 { return Err("clamp requires 3 arguments (tensor, min, max)".into()); }
            let t = value_to_tensor(&args[0])?;
            let min_v = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("clamp: min must be a number".into()) };
            let max_v = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("clamp: max must be a number".into()) };
            Ok(Some(Value::Tensor(t.clamp(min_v, max_v))))
        }
        "one_hot" => {
            if args.len() != 2 { return Err("one_hot requires 2 arguments (indices, depth)".into()); }
            let indices = value_to_usize_vec(&args[0])?;
            let depth = value_to_usize(&args[1])?;
            Ok(Some(Value::Tensor(Tensor::one_hot(&indices, depth).map_err(|e| format!("{e}"))?)))
        }
        // ── FFT builtins ────────────────────────────────────────────
        "rfft" => {
            if args.len() != 1 { return Err("rfft requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::fft::rfft(&data);
            let pairs: Vec<Value> = result.iter().map(|&(re, im)| {
                Value::Tuple(Rc::new(vec![Value::Float(re), Value::Float(im)]))
            }).collect();
            Ok(Some(Value::Array(Rc::new(pairs))))
        }
        "psd" => {
            if args.len() != 1 { return Err("psd requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::fft::psd(&data);
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }

        _ => Ok(None), // Not a shared builtin
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_constructor() {
        let result = dispatch_builtin("Complex", &[Value::Float(3.0), Value::Float(4.0)]);
        match result {
            Ok(Some(Value::Complex(c))) => {
                assert_eq!(c.re, 3.0);
                assert_eq!(c.im, 4.0);
            }
            _ => panic!("expected Complex value"),
        }
    }

    #[test]
    fn test_complex_constructor_from_ints() {
        let result = dispatch_builtin("Complex", &[Value::Int(1), Value::Int(-2)]);
        match result {
            Ok(Some(Value::Complex(c))) => {
                assert_eq!(c.re, 1.0);
                assert_eq!(c.im, -2.0);
            }
            _ => panic!("expected Complex value"),
        }
    }

    #[test]
    fn test_complex_real_only() {
        let result = dispatch_builtin("Complex", &[Value::Float(5.0)]);
        match result {
            Ok(Some(Value::Complex(c))) => {
                assert_eq!(c.re, 5.0);
                assert_eq!(c.im, 0.0);
            }
            _ => panic!("expected Complex value"),
        }
    }

    #[test]
    fn test_to_string() {
        let result = dispatch_builtin("to_string", &[Value::Int(42)]);
        match result {
            Ok(Some(Value::String(s))) => assert_eq!(s.as_str(), "42"),
            _ => panic!("expected String value"),
        }
    }

    #[test]
    fn test_len_array() {
        let arr = Value::Array(Rc::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));
        let result = dispatch_builtin("len", &[arr]);
        assert!(matches!(result, Ok(Some(Value::Int(3)))));
    }

    #[test]
    fn test_len_string() {
        let s = Value::String(Rc::new("hello".to_string()));
        let result = dispatch_builtin("len", &[s]);
        assert!(matches!(result, Ok(Some(Value::Int(5)))));
    }

    #[test]
    fn test_assert_pass() {
        let result = dispatch_builtin("assert", &[Value::Bool(true)]);
        assert!(matches!(result, Ok(Some(Value::Void))));
    }

    #[test]
    fn test_assert_fail() {
        let result = dispatch_builtin("assert", &[Value::Bool(false)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_eq_pass() {
        let result = dispatch_builtin("assert_eq", &[Value::Int(42), Value::Int(42)]);
        assert!(matches!(result, Ok(Some(Value::Void))));
    }

    #[test]
    fn test_assert_eq_fail() {
        let result = dispatch_builtin("assert_eq", &[Value::Int(1), Value::Int(2)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sqrt() {
        let result = dispatch_builtin("sqrt", &[Value::Float(4.0)]);
        match result {
            Ok(Some(Value::Float(v))) => assert_eq!(v, 2.0),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_abs_float() {
        let result = dispatch_builtin("abs", &[Value::Float(-3.14)]);
        match result {
            Ok(Some(Value::Float(v))) => assert_eq!(v, 3.14),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_abs_int() {
        let result = dispatch_builtin("abs", &[Value::Int(-42)]);
        assert!(matches!(result, Ok(Some(Value::Int(42)))));
    }

    #[test]
    fn test_floor() {
        let result = dispatch_builtin("floor", &[Value::Float(3.7)]);
        match result {
            Ok(Some(Value::Float(v))) => assert_eq!(v, 3.0),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_int_conversion() {
        let result = dispatch_builtin("int", &[Value::Float(3.9)]);
        assert!(matches!(result, Ok(Some(Value::Int(3)))));
    }

    #[test]
    fn test_float_conversion() {
        let result = dispatch_builtin("float", &[Value::Int(42)]);
        match result {
            Ok(Some(Value::Float(v))) => assert_eq!(v, 42.0),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_isnan() {
        let result = dispatch_builtin("isnan", &[Value::Float(f64::NAN)]);
        assert!(matches!(result, Ok(Some(Value::Bool(true)))));

        let result = dispatch_builtin("isnan", &[Value::Float(1.0)]);
        assert!(matches!(result, Ok(Some(Value::Bool(false)))));

        let result = dispatch_builtin("isnan", &[Value::Int(0)]);
        assert!(matches!(result, Ok(Some(Value::Bool(false)))));
    }

    #[test]
    fn test_isinf() {
        let result = dispatch_builtin("isinf", &[Value::Float(f64::INFINITY)]);
        assert!(matches!(result, Ok(Some(Value::Bool(true)))));

        let result = dispatch_builtin("isinf", &[Value::Float(1.0)]);
        assert!(matches!(result, Ok(Some(Value::Bool(false)))));
    }

    #[test]
    fn test_push() {
        let arr = Value::Array(Rc::new(vec![Value::Int(1)]));
        let result = dispatch_builtin("push", &[arr, Value::Int(2)]);
        match result {
            Ok(Some(Value::Array(a))) => {
                assert_eq!(a.len(), 2);
                assert!(matches!(&a[1], Value::Int(2)));
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_sort() {
        let arr = Value::Array(Rc::new(vec![
            Value::Float(3.0),
            Value::Float(1.0),
            Value::Float(2.0),
        ]));
        let result = dispatch_builtin("sort", &[arr]);
        match result {
            Ok(Some(Value::Array(a))) => {
                assert!(matches!(&a[0], Value::Float(v) if *v == 1.0));
                assert!(matches!(&a[1], Value::Float(v) if *v == 2.0));
                assert!(matches!(&a[2], Value::Float(v) if *v == 3.0));
            }
            _ => panic!("expected sorted Array"),
        }
    }

    #[test]
    fn test_unknown_builtin_returns_none() {
        let result = dispatch_builtin("unknown_function", &[]);
        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn test_tensor_zeros() {
        let shape = Value::Array(Rc::new(vec![Value::Int(2), Value::Int(3)]));
        let result = dispatch_builtin("Tensor.zeros", &[shape]);
        match result {
            Ok(Some(Value::Tensor(t))) => {
                assert_eq!(t.shape(), &[2, 3]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Int(42), &Value::Int(42)));
        assert!(!values_equal(&Value::Int(1), &Value::Int(2)));
        assert!(values_equal(&Value::Float(3.14), &Value::Float(3.14)));
        assert!(values_equal(&Value::Bool(true), &Value::Bool(true)));
        assert!(values_equal(&Value::Void, &Value::Void));
        assert!(!values_equal(&Value::Int(1), &Value::Float(1.0)));
    }

    #[test]
    fn test_value_to_shape() {
        let arr = Value::Array(Rc::new(vec![Value::Int(2), Value::Int(3), Value::Int(4)]));
        assert_eq!(value_to_shape(&arr).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn test_value_to_shape_negative_rejected() {
        let arr = Value::Array(Rc::new(vec![Value::Int(-1)]));
        assert!(value_to_shape(&arr).is_err());
    }

    #[test]
    fn test_value_to_f64_vec() {
        let arr = Value::Array(Rc::new(vec![
            Value::Float(1.0),
            Value::Int(2),
            Value::Float(3.5),
        ]));
        assert_eq!(value_to_f64_vec(&arr).unwrap(), vec![1.0, 2.0, 3.5]);
    }
}
