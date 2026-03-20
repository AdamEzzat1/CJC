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

use crate::accumulator::BinnedAccumulatorF64;
use crate::complex::ComplexF64;
use crate::scratchpad::Scratchpad;
use crate::paged_kv::PagedKvCache;
use crate::tensor::Tensor;
use crate::tensor_simd::UnaryOp;
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

/// Convert a `Value::Array` of complex tuples (re, im) to `Vec<(f64, f64)>`.
pub fn value_to_complex_vec(val: &Value) -> Result<Vec<(f64, f64)>, String> {
    match val {
        Value::Array(arr) => {
            let mut data = Vec::with_capacity(arr.len());
            for v in arr.iter() {
                match v {
                    Value::Tuple(t) if t.len() == 2 => {
                        let re = match &t[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("complex tuple element must be numeric".into()) };
                        let im = match &t[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("complex tuple element must be numeric".into()) };
                        data.push((re, im));
                    }
                    _ => return Err("expected array of (re, im) tuples".into()),
                }
            }
            Ok(data)
        }
        _ => Err(format!("expected Array of complex tuples, got {}", val.type_name())),
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

/// Convert a `Value::Float` or `Value::Int` to `f64`.
pub fn value_to_f64(val: &Value) -> Result<f64, String> {
    match val {
        Value::Float(v) => Ok(*v),
        Value::Int(v) => Ok(*v as f64),
        _ => Err(format!("expected Float or Int, got {}", val.type_name())),
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
// Deterministic categorical sampling (needs external RNG)
// ---------------------------------------------------------------------------

/// Sample an index from a 1-D probability tensor using the given uniform
/// random value `u` in [0, 1). Returns the selected index (0-based).
/// This is a general-purpose RL primitive, not domain-specific.
pub fn categorical_sample_with_u(probs: &Tensor, u: f64) -> Result<i64, String> {
    if probs.ndim() == 0 {
        return Err("categorical_sample requires at least a 1-D tensor".into());
    }
    let data = probs.to_vec();
    if data.is_empty() {
        return Err("categorical_sample: empty probability tensor".into());
    }
    let mut cumsum = 0.0;
    for (i, &p) in data.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return Ok(i as i64);
        }
    }
    // Numerical safety: return last valid index
    Ok((data.len() - 1) as i64)
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

        // ── String manipulation builtins ────────────────────────────────
        "str_upper" => {
            if args.len() != 1 { return Err("str_upper requires 1 argument".into()); }
            match &args[0] {
                Value::String(s) => Ok(Some(Value::String(Rc::new(s.to_uppercase())))),
                _ => Err("str_upper: argument must be a string".into()),
            }
        }
        "str_lower" => {
            if args.len() != 1 { return Err("str_lower requires 1 argument".into()); }
            match &args[0] {
                Value::String(s) => Ok(Some(Value::String(Rc::new(s.to_lowercase())))),
                _ => Err("str_lower: argument must be a string".into()),
            }
        }
        "str_trim" => {
            if args.len() != 1 { return Err("str_trim requires 1 argument".into()); }
            match &args[0] {
                Value::String(s) => Ok(Some(Value::String(Rc::new(s.trim().to_string())))),
                _ => Err("str_trim: argument must be a string".into()),
            }
        }
        "str_contains" => {
            if args.len() != 2 { return Err("str_contains requires 2 arguments".into()); }
            match (&args[0], &args[1]) {
                (Value::String(haystack), Value::String(needle)) => {
                    Ok(Some(Value::Bool(haystack.contains(needle.as_str()))))
                }
                _ => Err("str_contains: both arguments must be strings".into()),
            }
        }
        "str_replace" => {
            if args.len() != 3 { return Err("str_replace requires 3 arguments (str, from, to)".into()); }
            match (&args[0], &args[1], &args[2]) {
                (Value::String(s), Value::String(from), Value::String(to)) => {
                    // Replace first occurrence only (matches tidy/stringr semantics).
                    // Use str_replace_all for global replacement.
                    Ok(Some(Value::String(Rc::new(s.replacen(from.as_str(), to.as_str(), 1)))))
                }
                _ => Err("str_replace: all arguments must be strings".into()),
            }
        }
        "str_split" => {
            if args.len() != 2 { return Err("str_split requires 2 arguments (str, delimiter)".into()); }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(delim)) => {
                    let parts: Vec<Value> = s.split(delim.as_str())
                        .map(|p| Value::String(Rc::new(p.to_string())))
                        .collect();
                    Ok(Some(Value::Array(Rc::new(parts))))
                }
                _ => Err("str_split: both arguments must be strings".into()),
            }
        }
        "str_join" => {
            if args.len() != 2 { return Err("str_join requires 2 arguments (array, delimiter)".into()); }
            match (&args[0], &args[1]) {
                (Value::Array(arr), Value::String(delim)) => {
                    let parts: Vec<String> = arr.iter()
                        .map(|v| format!("{}", v))
                        .collect();
                    Ok(Some(Value::String(Rc::new(parts.join(delim.as_str())))))
                }
                _ => Err("str_join: first arg must be array, second must be string".into()),
            }
        }
        "str_starts_with" => {
            if args.len() != 2 { return Err("str_starts_with requires 2 arguments".into()); }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(prefix)) => {
                    Ok(Some(Value::Bool(s.starts_with(prefix.as_str()))))
                }
                _ => Err("str_starts_with: both arguments must be strings".into()),
            }
        }
        "str_ends_with" => {
            if args.len() != 2 { return Err("str_ends_with requires 2 arguments".into()); }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(suffix)) => {
                    Ok(Some(Value::Bool(s.ends_with(suffix.as_str()))))
                }
                _ => Err("str_ends_with: both arguments must be strings".into()),
            }
        }
        "str_repeat" => {
            if args.len() != 2 { return Err("str_repeat requires 2 arguments (str, count)".into()); }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::Int(n)) => {
                    if *n < 0 { return Err("str_repeat: count must be non-negative".into()); }
                    Ok(Some(Value::String(Rc::new(s.repeat(*n as usize)))))
                }
                _ => Err("str_repeat: first arg must be string, second must be integer".into()),
            }
        }
        "str_chars" => {
            if args.len() != 1 { return Err("str_chars requires 1 argument".into()); }
            match &args[0] {
                Value::String(s) => {
                    let chars: Vec<Value> = s.chars()
                        .map(|c| Value::String(Rc::new(c.to_string())))
                        .collect();
                    Ok(Some(Value::Array(Rc::new(chars))))
                }
                _ => Err("str_chars: argument must be a string".into()),
            }
        }
        "str_substr" => {
            if args.len() != 3 { return Err("str_substr requires 3 arguments (str, start, len)".into()); }
            match (&args[0], &args[1], &args[2]) {
                (Value::String(s), Value::Int(start), Value::Int(len)) => {
                    let start = (*start).max(0) as usize;
                    let len = (*len).max(0) as usize;
                    let result: String = s.chars().skip(start).take(len).collect();
                    Ok(Some(Value::String(Rc::new(result))))
                }
                _ => Err("str_substr: (str, int, int) expected".into()),
            }
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
        "log" => {
            if args.len() != 1 {
                return Err("log requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.ln()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).ln()))),
                _ => Err(format!("log requires a number, got {}", args[0].type_name())),
            }
        }
        "exp" => {
            if args.len() != 1 {
                return Err("exp requires exactly 1 argument".into());
            }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.exp()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).exp()))),
                _ => Err(format!("exp requires a number, got {}", args[0].type_name())),
            }
        }
        // ---- Mathematics Hardening Phase: Trigonometric ----
        "sin" => {
            if args.len() != 1 { return Err("sin requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.sin()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).sin()))),
                _ => Err(format!("sin requires a number, got {}", args[0].type_name())),
            }
        }
        "cos" => {
            if args.len() != 1 { return Err("cos requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.cos()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).cos()))),
                _ => Err(format!("cos requires a number, got {}", args[0].type_name())),
            }
        }
        "tan" => {
            if args.len() != 1 { return Err("tan requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.tan()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).tan()))),
                _ => Err(format!("tan requires a number, got {}", args[0].type_name())),
            }
        }
        "asin" => {
            if args.len() != 1 { return Err("asin requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.asin()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).asin()))),
                _ => Err(format!("asin requires a number, got {}", args[0].type_name())),
            }
        }
        "acos" => {
            if args.len() != 1 { return Err("acos requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.acos()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).acos()))),
                _ => Err(format!("acos requires a number, got {}", args[0].type_name())),
            }
        }
        "atan" => {
            if args.len() != 1 { return Err("atan requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.atan()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).atan()))),
                _ => Err(format!("atan requires a number, got {}", args[0].type_name())),
            }
        }
        "atan2" => {
            if args.len() != 2 { return Err("atan2 requires exactly 2 arguments".into()); }
            let y = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("atan2 requires numbers, got {}", args[0].type_name())),
            };
            let x = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("atan2 requires numbers, got {}", args[1].type_name())),
            };
            Ok(Some(Value::Float(y.atan2(x))))
        }
        // ---- Mathematics Hardening Phase: Hyperbolic ----
        "sinh" => {
            if args.len() != 1 { return Err("sinh requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.sinh()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).sinh()))),
                _ => Err(format!("sinh requires a number, got {}", args[0].type_name())),
            }
        }
        "cosh" => {
            if args.len() != 1 { return Err("cosh requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.cosh()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).cosh()))),
                _ => Err(format!("cosh requires a number, got {}", args[0].type_name())),
            }
        }
        "tanh_scalar" => {
            if args.len() != 1 { return Err("tanh_scalar requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.tanh()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).tanh()))),
                _ => Err(format!("tanh_scalar requires a number, got {}", args[0].type_name())),
            }
        }
        // ---- Mathematics Hardening Phase: Exponentiation & Logarithms ----
        "pow" => {
            if args.len() != 2 { return Err("pow requires exactly 2 arguments".into()); }
            let base = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("pow requires numbers, got {}", args[0].type_name())),
            };
            let exp = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("pow requires numbers, got {}", args[1].type_name())),
            };
            Ok(Some(Value::Float(base.powf(exp))))
        }
        "log2" => {
            if args.len() != 1 { return Err("log2 requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.log2()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).log2()))),
                _ => Err(format!("log2 requires a number, got {}", args[0].type_name())),
            }
        }
        "log10" => {
            if args.len() != 1 { return Err("log10 requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.log10()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).log10()))),
                _ => Err(format!("log10 requires a number, got {}", args[0].type_name())),
            }
        }
        "log1p" => {
            if args.len() != 1 { return Err("log1p requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.ln_1p()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).ln_1p()))),
                _ => Err(format!("log1p requires a number, got {}", args[0].type_name())),
            }
        }
        "expm1" => {
            if args.len() != 1 { return Err("expm1 requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.exp_m1()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).exp_m1()))),
                _ => Err(format!("expm1 requires a number, got {}", args[0].type_name())),
            }
        }
        // ---- Mathematics Hardening Phase: Rounding ----
        "ceil" => {
            if args.len() != 1 { return Err("ceil requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.ceil()))),
                Value::Int(i) => Ok(Some(Value::Int(*i))),
                _ => Err(format!("ceil requires a number, got {}", args[0].type_name())),
            }
        }
        "round" => {
            if args.len() != 1 { return Err("round requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.round()))),
                Value::Int(i) => Ok(Some(Value::Int(*i))),
                _ => Err(format!("round requires a number, got {}", args[0].type_name())),
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
        // ---- Mathematics Hardening Phase: Comparison & Sign ----
        "min" => {
            if args.len() != 2 { return Err("min requires exactly 2 arguments".into()); }
            let a = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("min requires numbers, got {}", args[0].type_name())),
            };
            let b = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("min requires numbers, got {}", args[1].type_name())),
            };
            Ok(Some(Value::Float(a.min(b))))
        }
        "max" => {
            if args.len() != 2 { return Err("max requires exactly 2 arguments".into()); }
            let a = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("max requires numbers, got {}", args[0].type_name())),
            };
            let b = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("max requires numbers, got {}", args[1].type_name())),
            };
            Ok(Some(Value::Float(a.max(b))))
        }
        "sign" => {
            if args.len() != 1 { return Err("sign requires exactly 1 argument".into()); }
            match &args[0] {
                Value::Float(f) => Ok(Some(Value::Float(f.signum()))),
                Value::Int(i) => Ok(Some(Value::Float((*i as f64).signum()))),
                _ => Err(format!("sign requires a number, got {}", args[0].type_name())),
            }
        }
        // ---- Mathematics Hardening Phase: Precision Helpers ----
        "hypot" => {
            if args.len() != 2 { return Err("hypot requires exactly 2 arguments".into()); }
            let x = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("hypot requires numbers, got {}", args[0].type_name())),
            };
            let y = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err(format!("hypot requires numbers, got {}", args[1].type_name())),
            };
            Ok(Some(Value::Float(x.hypot(y))))
        }
        // ---- Mathematics Hardening Phase: Constants ----
        "PI" => {
            if !args.is_empty() { return Err("PI takes no arguments".into()); }
            Ok(Some(Value::Float(std::f64::consts::PI)))
        }
        "E" => {
            if !args.is_empty() { return Err("E takes no arguments".into()); }
            Ok(Some(Value::Float(std::f64::consts::E)))
        }
        "TAU" => {
            if !args.is_empty() { return Err("TAU takes no arguments".into()); }
            Ok(Some(Value::Float(std::f64::consts::TAU)))
        }
        "INF" => {
            if !args.is_empty() { return Err("INF takes no arguments".into()); }
            Ok(Some(Value::Float(f64::INFINITY)))
        }
        "NAN_VAL" => {
            if !args.is_empty() { return Err("NAN_VAL takes no arguments".into()); }
            Ok(Some(Value::Float(f64::NAN)))
        }
        // ---- Mathematics Hardening Phase: Vector Operations ----
        "dot" => {
            if args.len() != 2 { return Err("dot requires exactly 2 arguments".into()); }
            let a = match &args[0] {
                Value::Tensor(t) => t,
                _ => return Err(format!("dot requires tensors, got {}", args[0].type_name())),
            };
            let b = match &args[1] {
                Value::Tensor(t) => t,
                _ => return Err(format!("dot requires tensors, got {}", args[1].type_name())),
            };
            if a.ndim() != 1 || b.ndim() != 1 {
                return Err("dot requires 1D tensors".into());
            }
            if a.len() != b.len() {
                return Err(format!("dot: length mismatch ({} vs {})", a.len(), b.len()));
            }
            let av = a.to_vec();
            let bv = b.to_vec();
            let products: Vec<f64> = av.iter().zip(bv.iter()).map(|(x, y)| x * y).collect();
            let sum = crate::accumulator::binned_sum_f64(&products);
            Ok(Some(Value::Float(sum)))
        }
        "outer" => {
            if args.len() != 2 { return Err("outer requires exactly 2 arguments".into()); }
            let a = match &args[0] {
                Value::Tensor(t) => t,
                _ => return Err(format!("outer requires tensors, got {}", args[0].type_name())),
            };
            let b = match &args[1] {
                Value::Tensor(t) => t,
                _ => return Err(format!("outer requires tensors, got {}", args[1].type_name())),
            };
            if a.ndim() != 1 || b.ndim() != 1 {
                return Err("outer requires 1D tensors".into());
            }
            let av = a.to_vec();
            let bv = b.to_vec();
            let m = av.len();
            let n = bv.len();
            let mut data = Vec::with_capacity(m * n);
            for ai in &av {
                for bj in &bv {
                    data.push(ai * bj);
                }
            }
            Ok(Some(Value::Tensor(Tensor::from_vec(data, &[m, n]).map_err(|e| format!("{e}"))?)))
        }
        "cross" => {
            if args.len() != 2 { return Err("cross requires exactly 2 arguments".into()); }
            let a = match &args[0] {
                Value::Tensor(t) => t,
                _ => return Err(format!("cross requires tensors, got {}", args[0].type_name())),
            };
            let b = match &args[1] {
                Value::Tensor(t) => t,
                _ => return Err(format!("cross requires tensors, got {}", args[1].type_name())),
            };
            if a.ndim() != 1 || b.ndim() != 1 || a.len() != 3 || b.len() != 3 {
                return Err("cross requires two 3-element 1D tensors".into());
            }
            let av = a.to_vec();
            let bv = b.to_vec();
            let result = vec![
                av[1] * bv[2] - av[2] * bv[1],
                av[2] * bv[0] - av[0] * bv[2],
                av[0] * bv[1] - av[1] * bv[0],
            ];
            Ok(Some(Value::Tensor(Tensor::from_vec(result, &[3]).map_err(|e| format!("{e}"))?)))
        }
        "norm" => {
            if args.len() < 1 || args.len() > 2 { return Err("norm requires 1-2 arguments".into()); }
            let t = match &args[0] {
                Value::Tensor(t) => t,
                _ => return Err(format!("norm requires a tensor, got {}", args[0].type_name())),
            };
            let ord = if args.len() == 2 {
                match &args[1] {
                    Value::Int(i) => *i,
                    Value::Float(f) => *f as i64,
                    _ => return Err("norm: ord must be an integer".into()),
                }
            } else {
                2 // default: L2 norm
            };
            let data = t.to_vec();
            let result = match ord {
                1 => {
                    let abs_vals: Vec<f64> = data.iter().map(|x| x.abs()).collect();
                    crate::accumulator::binned_sum_f64(&abs_vals)
                }
                2 => {
                    let sq_vals: Vec<f64> = data.iter().map(|x| x * x).collect();
                    crate::accumulator::binned_sum_f64(&sq_vals).sqrt()
                }
                _ => {
                    let p = ord as f64;
                    let pow_vals: Vec<f64> = data.iter().map(|x| x.abs().powf(p)).collect();
                    crate::accumulator::binned_sum_f64(&pow_vals).powf(1.0 / p)
                }
            };
            Ok(Some(Value::Float(result)))
        }
        // ---- Mathematics Hardening Phase: Tensor Constructors ----
        "Tensor.linspace" => {
            if args.len() != 3 { return Err("Tensor.linspace requires 3 arguments (start, end, n)".into()); }
            let start = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("Tensor.linspace: start must be a number".into()),
            };
            let end = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("Tensor.linspace: end must be a number".into()),
            };
            let n = match &args[2] {
                Value::Int(i) => *i as usize,
                _ => return Err("Tensor.linspace: n must be an integer".into()),
            };
            if n == 0 {
                return Ok(Some(Value::Tensor(Tensor::from_vec(vec![], &[0]).map_err(|e| format!("{e}"))?)));
            }
            if n == 1 {
                return Ok(Some(Value::Tensor(Tensor::from_vec(vec![start], &[1]).map_err(|e| format!("{e}"))?)));
            }
            let step = (end - start) / (n as f64 - 1.0);
            let data: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
            Ok(Some(Value::Tensor(Tensor::from_vec(data, &[n]).map_err(|e| format!("{e}"))?)))
        }
        "Tensor.arange" => {
            if args.len() < 2 || args.len() > 3 { return Err("Tensor.arange requires 2-3 arguments (start, end, step?)".into()); }
            let start = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("Tensor.arange: start must be a number".into()),
            };
            let end = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("Tensor.arange: end must be a number".into()),
            };
            let step = if args.len() == 3 {
                match &args[2] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Err("Tensor.arange: step must be a number".into()),
                }
            } else {
                1.0
            };
            if step == 0.0 { return Err("Tensor.arange: step cannot be zero".into()); }
            let mut data = Vec::new();
            let mut val = start;
            if step > 0.0 {
                while val < end { data.push(val); val += step; }
            } else {
                while val > end { data.push(val); val += step; }
            }
            let n = data.len();
            Ok(Some(Value::Tensor(Tensor::from_vec(data, &[n]).map_err(|e| format!("{e}"))?)))
        }
        "Tensor.eye" => {
            if args.len() != 1 { return Err("Tensor.eye requires 1 argument (n)".into()); }
            let n = match &args[0] {
                Value::Int(i) => *i as usize,
                _ => return Err("Tensor.eye: n must be an integer".into()),
            };
            let mut data = vec![0.0; n * n];
            for i in 0..n {
                data[i * n + i] = 1.0;
            }
            Ok(Some(Value::Tensor(Tensor::from_vec(data, &[n, n]).map_err(|e| format!("{e}"))?)))
        }
        "Tensor.full" => {
            if args.len() != 2 { return Err("Tensor.full requires 2 arguments (shape, value)".into()); }
            let shape = match &args[0] {
                Value::Array(arr) => {
                    let mut s = Vec::new();
                    for v in arr.iter() {
                        match v {
                            Value::Int(i) => s.push(*i as usize),
                            _ => return Err("Tensor.full: shape must be an array of ints".into()),
                        }
                    }
                    s
                }
                _ => return Err("Tensor.full: shape must be an array".into()),
            };
            let fill_val = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("Tensor.full: value must be a number".into()),
            };
            let total: usize = shape.iter().product();
            let data = vec![fill_val; total];
            Ok(Some(Value::Tensor(Tensor::from_vec(data, &shape).map_err(|e| format!("{e}"))?)))
        }
        "Tensor.diag" => {
            if args.len() != 1 { return Err("Tensor.diag requires 1 argument".into()); }
            let t = match &args[0] {
                Value::Tensor(t) => t,
                _ => return Err("Tensor.diag requires a tensor".into()),
            };
            match t.ndim() {
                1 => {
                    // 1D -> 2D diagonal matrix
                    let data = t.to_vec();
                    let n = data.len();
                    let mut out = vec![0.0; n * n];
                    for i in 0..n {
                        out[i * n + i] = data[i];
                    }
                    Ok(Some(Value::Tensor(Tensor::from_vec(out, &[n, n]).map_err(|e| format!("{e}"))?)))
                }
                2 => {
                    // 2D -> 1D diagonal extraction
                    let rows = t.shape()[0];
                    let cols = t.shape()[1];
                    let n = rows.min(cols);
                    let mut data = Vec::with_capacity(n);
                    for i in 0..n {
                        data.push(t.get(&[i, i]).map_err(|e| format!("{e}"))?);
                    }
                    Ok(Some(Value::Tensor(Tensor::from_vec(data, &[n]).map_err(|e| format!("{e}"))?)))
                }
                _ => Err("Tensor.diag requires a 1D or 2D tensor".into()),
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
        "mean" => {
            if args.len() != 1 { return Err("mean requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            if data.is_empty() { return Err("mean: empty data".into()); }
            Ok(Some(Value::Float(cjc_repro::kahan_sum_f64(&data) / data.len() as f64)))
        }
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
        // ── Bastion primitives ──────────────────────────────────────
        "nth_element" => {
            if args.len() != 2 { return Err("nth_element requires 2 arguments: data, k".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let k = value_to_usize(&args[1])?;
            Ok(Some(Value::Float(crate::stats::nth_element_copy(&data, k)?)))
        }
        "median_fast" => {
            if args.len() != 1 { return Err("median_fast requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::median_fast(&data)?)))
        }
        "quantile_fast" => {
            if args.len() != 2 { return Err("quantile_fast requires 2 arguments: data, p".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let p = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("quantile_fast: p must be a number".into()),
            };
            Ok(Some(Value::Float(crate::stats::quantile_fast(&data, p)?)))
        }
        "filter_mask" => {
            if args.len() != 2 { return Err("filter_mask requires 2 arguments: data, mask".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let mask: Vec<bool> = match &args[1] {
                Value::Array(arr) => arr.iter().map(|v| match v {
                    Value::Bool(b) => Ok(*b),
                    Value::Int(i) => Ok(*i != 0),
                    _ => Err("filter_mask: mask must be array of bools".to_string()),
                }).collect::<Result<Vec<_>, _>>()?,
                _ => return Err("filter_mask: mask must be an array".into()),
            };
            let result = crate::stats::filter_mask(&data, &mask)?;
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "erf" => {
            if args.len() != 1 { return Err("erf requires 1 argument".into()); }
            let x = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("erf requires a number".into()),
            };
            Ok(Some(Value::Float(crate::distributions::erf(x))))
        }
        "erfc" => {
            if args.len() != 1 { return Err("erfc requires 1 argument".into()); }
            let x = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("erfc requires a number".into()),
            };
            Ok(Some(Value::Float(crate::distributions::erfc(x))))
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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
            let mut fields = std::collections::BTreeMap::new();
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

        // ── B1: Weighted & robust statistics ──────────────────────────
        "weighted_mean" => {
            if args.len() != 2 { return Err("weighted_mean requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let weights = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::weighted_mean(&data, &weights)?)))
        }
        "weighted_var" => {
            if args.len() != 2 { return Err("weighted_var requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let weights = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::weighted_var(&data, &weights)?)))
        }
        "trimmed_mean" => {
            if args.len() != 2 { return Err("trimmed_mean requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let prop = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("trimmed_mean: proportion must be a number".into()) };
            Ok(Some(Value::Float(crate::stats::trimmed_mean(&data, prop)?)))
        }
        "winsorize" => {
            if args.len() != 2 { return Err("winsorize requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let prop = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("winsorize: proportion must be a number".into()) };
            let result = crate::stats::winsorize(&data, prop)?;
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "mad" => {
            if args.len() != 1 { return Err("mad requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::mad(&data)?)))
        }
        "mode" => {
            if args.len() != 1 { return Err("mode requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            Ok(Some(Value::Float(crate::stats::mode(&data)?)))
        }
        "percentile_rank" => {
            if args.len() != 2 { return Err("percentile_rank requires 2 arguments".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let value = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("percentile_rank: value must be a number".into()) };
            Ok(Some(Value::Float(crate::stats::percentile_rank(&data, value)?)))
        }

        // ── B4: ML training extensions ─────────────────────────────────
        "cat" => {
            if args.len() != 2 { return Err("cat requires 2 arguments (array of tensors, axis)".into()); }
            let tensors_arr = match &args[0] {
                Value::Array(arr) => arr.iter().map(|v| match v {
                    Value::Tensor(t) => Ok(t),
                    _ => Err("cat: first argument must be array of tensors".to_string()),
                }).collect::<Result<Vec<&Tensor>, String>>()?,
                _ => return Err("cat: first argument must be array of tensors".into()),
            };
            let axis = value_to_usize(&args[1])?;
            let refs: Vec<&Tensor> = tensors_arr;
            Ok(Some(Value::Tensor(Tensor::cat(&refs, axis).map_err(|e| format!("{e}"))?)))
        }
        "stack" => {
            if args.len() != 2 { return Err("stack requires 2 arguments (array of tensors, axis)".into()); }
            let tensors_arr = match &args[0] {
                Value::Array(arr) => arr.iter().map(|v| match v {
                    Value::Tensor(t) => Ok(t),
                    _ => Err("stack: first argument must be array of tensors".to_string()),
                }).collect::<Result<Vec<&Tensor>, String>>()?,
                _ => return Err("stack: first argument must be array of tensors".into()),
            };
            let axis = value_to_usize(&args[1])?;
            Ok(Some(Value::Tensor(Tensor::stack(&tensors_arr, axis).map_err(|e| format!("{e}"))?)))
        }
        "topk" => {
            if args.len() != 2 { return Err("topk requires 2 arguments (tensor, k)".into()); }
            let t = value_to_tensor(&args[0])?;
            let k = value_to_usize(&args[1])?;
            let (vals, idxs) = t.topk(k).map_err(|e| format!("{e}"))?;
            let idx_values: Vec<Value> = idxs.into_iter().map(|i| Value::Int(i as i64)).collect();
            Ok(Some(Value::Tuple(Rc::new(vec![Value::Tensor(vals), Value::Array(Rc::new(idx_values))]))))
        }
        "batch_norm" => {
            if args.len() != 6 { return Err("batch_norm requires 6 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let mean = value_to_f64_vec(&args[1])?;
            let var = value_to_f64_vec(&args[2])?;
            let gamma = value_to_f64_vec(&args[3])?;
            let beta = value_to_f64_vec(&args[4])?;
            let eps = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("batch_norm: eps must be a number".into()) };
            let result = crate::ml::batch_norm(&x, &mean, &var, &gamma, &beta, eps)?;
            let values: Vec<Value> = result.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "dropout_mask" => {
            if args.len() != 3 { return Err("dropout_mask requires 3 arguments (n, prob, seed)".into()); }
            let n = value_to_usize(&args[0])?;
            let prob = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("dropout_mask: prob must be a number".into()) };
            let seed = match &args[2] { Value::Int(i) => *i as u64, _ => return Err("dropout_mask: seed must be an integer".into()) };
            let mask = crate::ml::dropout_mask(n, prob, seed);
            let values: Vec<Value> = mask.into_iter().map(Value::Float).collect();
            Ok(Some(Value::Array(Rc::new(values))))
        }
        "lr_step_decay" => {
            if args.len() != 4 { return Err("lr_step_decay requires 4 arguments".into()); }
            let lr = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr_step_decay: lr must be a number".into()) };
            let rate = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr_step_decay: rate must be a number".into()) };
            let epoch = value_to_usize(&args[2])?;
            let step = value_to_usize(&args[3])?;
            Ok(Some(Value::Float(crate::ml::lr_step_decay(lr, rate, epoch, step))))
        }
        "lr_cosine" => {
            if args.len() != 4 { return Err("lr_cosine requires 4 arguments".into()); }
            let max_lr = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr_cosine: max_lr must be a number".into()) };
            let min_lr = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr_cosine: min_lr must be a number".into()) };
            let epoch = value_to_usize(&args[2])?;
            let total = value_to_usize(&args[3])?;
            Ok(Some(Value::Float(crate::ml::lr_cosine(max_lr, min_lr, epoch, total))))
        }
        "lr_linear_warmup" => {
            if args.len() != 3 { return Err("lr_linear_warmup requires 3 arguments".into()); }
            let lr = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr_linear_warmup: lr must be a number".into()) };
            let epoch = value_to_usize(&args[1])?;
            let warmup = value_to_usize(&args[2])?;
            Ok(Some(Value::Float(crate::ml::lr_linear_warmup(lr, epoch, warmup))))
        }
        "l1_penalty" => {
            if args.len() != 2 { return Err("l1_penalty requires 2 arguments".into()); }
            let params = value_to_f64_vec(&args[0])?;
            let lambda = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("l1_penalty: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::ml::l1_penalty(&params, lambda))))
        }
        "l2_penalty" => {
            if args.len() != 2 { return Err("l2_penalty requires 2 arguments".into()); }
            let params = value_to_f64_vec(&args[0])?;
            let lambda = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("l2_penalty: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::ml::l2_penalty(&params, lambda))))
        }

        // ── B3: Linear algebra extensions ─────────────────────────────
        "cond" => {
            if args.len() != 1 { return Err("cond requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Float(t.cond().map_err(|e| format!("{e}"))?)))
        }
        "norm_1" => {
            if args.len() != 1 { return Err("norm_1 requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Float(t.norm_1().map_err(|e| format!("{e}"))?)))
        }
        "norm_inf" => {
            if args.len() != 1 { return Err("norm_inf requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Float(t.norm_inf().map_err(|e| format!("{e}"))?)))
        }
        "schur" => {
            if args.len() != 1 { return Err("schur requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            let (q, t_mat) = t.schur().map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tuple(Rc::new(vec![Value::Tensor(q), Value::Tensor(t_mat)]))))
        }
        "matrix_exp" => {
            if args.len() != 1 { return Err("matrix_exp requires 1 Tensor argument".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.matrix_exp().map_err(|e| format!("{e}"))?)))
        }

        // ── B2: Rank correlations & partial correlation ────────────────
        "spearman_cor" => {
            if args.len() != 2 { return Err("spearman_cor requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::spearman_cor(&x, &y)?)))
        }
        "kendall_cor" => {
            if args.len() != 2 { return Err("kendall_cor requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            Ok(Some(Value::Float(crate::stats::kendall_cor(&x, &y)?)))
        }
        "partial_cor" => {
            if args.len() != 3 { return Err("partial_cor requires 3 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let z = value_to_f64_vec(&args[2])?;
            Ok(Some(Value::Float(crate::stats::partial_cor(&x, &y, &z)?)))
        }
        "cor_ci" => {
            if args.len() != 3 { return Err("cor_ci requires 3 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let alpha = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("cor_ci: alpha must be a number".into()) };
            let (lo, hi) = crate::stats::cor_ci(&x, &y, alpha)?;
            Ok(Some(Value::Tuple(Rc::new(vec![Value::Float(lo), Value::Float(hi)]))))
        }

        // ── B6: Advanced FFT & Distributions ─────────────────────────
        "hann" => {
            if args.len() != 1 { return Err("hann requires 1 argument (n)".into()); }
            let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("hann: n must be an integer".into()) };
            let w = crate::fft::hann_window(n);
            Ok(Some(Value::Array(Rc::new(w.into_iter().map(Value::Float).collect()))))
        }
        "hamming" => {
            if args.len() != 1 { return Err("hamming requires 1 argument (n)".into()); }
            let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("hamming: n must be an integer".into()) };
            let w = crate::fft::hamming_window(n);
            Ok(Some(Value::Array(Rc::new(w.into_iter().map(Value::Float).collect()))))
        }
        "blackman" => {
            if args.len() != 1 { return Err("blackman requires 1 argument (n)".into()); }
            let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("blackman: n must be an integer".into()) };
            let w = crate::fft::blackman_window(n);
            Ok(Some(Value::Array(Rc::new(w.into_iter().map(Value::Float).collect()))))
        }
        "fft_arbitrary" => {
            if args.len() != 1 { return Err("fft_arbitrary requires 1 argument (complex array)".into()); }
            let data = value_to_complex_vec(&args[0])?;
            let result = crate::fft::fft_arbitrary(&data);
            let pairs: Vec<Value> = result.iter().map(|&(re, im)| {
                Value::Tuple(Rc::new(vec![Value::Float(re), Value::Float(im)]))
            }).collect();
            Ok(Some(Value::Array(Rc::new(pairs))))
        }
        "fft_2d" => {
            if args.len() != 3 { return Err("fft_2d requires 3 arguments (data, rows, cols)".into()); }
            let data = value_to_complex_vec(&args[0])?;
            let rows = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("fft_2d: rows must be an integer".into()) };
            let cols = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("fft_2d: cols must be an integer".into()) };
            let result = crate::fft::fft_2d(&data, rows, cols)?;
            let pairs: Vec<Value> = result.iter().map(|&(re, im)| {
                Value::Tuple(Rc::new(vec![Value::Float(re), Value::Float(im)]))
            }).collect();
            Ok(Some(Value::Array(Rc::new(pairs))))
        }
        "ifft_2d" => {
            if args.len() != 3 { return Err("ifft_2d requires 3 arguments (data, rows, cols)".into()); }
            let data = value_to_complex_vec(&args[0])?;
            let rows = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("ifft_2d: rows must be an integer".into()) };
            let cols = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("ifft_2d: cols must be an integer".into()) };
            let result = crate::fft::ifft_2d(&data, rows, cols)?;
            let pairs: Vec<Value> = result.iter().map(|&(re, im)| {
                Value::Tuple(Rc::new(vec![Value::Float(re), Value::Float(im)]))
            }).collect();
            Ok(Some(Value::Array(Rc::new(pairs))))
        }
        "beta_pdf" => {
            if args.len() != 3 { return Err("beta_pdf requires 3 arguments (x, a, b)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta_pdf: x must be a number".into()) };
            let a = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta_pdf: a must be a number".into()) };
            let b = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta_pdf: b must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::beta_pdf(x, a, b))))
        }
        "beta_cdf" => {
            if args.len() != 3 { return Err("beta_cdf requires 3 arguments (x, a, b)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta_cdf: x must be a number".into()) };
            let a = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta_cdf: a must be a number".into()) };
            let b = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta_cdf: b must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::beta_cdf(x, a, b))))
        }
        "gamma_pdf" => {
            if args.len() != 3 { return Err("gamma_pdf requires 3 arguments (x, k, theta)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma_pdf: x must be a number".into()) };
            let k = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma_pdf: k must be a number".into()) };
            let theta = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma_pdf: theta must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::gamma_pdf(x, k, theta))))
        }
        "gamma_cdf" => {
            if args.len() != 3 { return Err("gamma_cdf requires 3 arguments (x, k, theta)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma_cdf: x must be a number".into()) };
            let k = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma_cdf: k must be a number".into()) };
            let theta = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma_cdf: theta must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::gamma_cdf(x, k, theta))))
        }
        "exp_pdf" => {
            if args.len() != 2 { return Err("exp_pdf requires 2 arguments (x, lambda)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("exp_pdf: x must be a number".into()) };
            let lambda = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("exp_pdf: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::exp_pdf(x, lambda))))
        }
        "exp_cdf" => {
            if args.len() != 2 { return Err("exp_cdf requires 2 arguments (x, lambda)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("exp_cdf: x must be a number".into()) };
            let lambda = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("exp_cdf: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::exp_cdf(x, lambda))))
        }
        "weibull_pdf" => {
            if args.len() != 3 { return Err("weibull_pdf requires 3 arguments (x, k, lambda)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("weibull_pdf: x must be a number".into()) };
            let k = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("weibull_pdf: k must be a number".into()) };
            let lambda = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("weibull_pdf: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::weibull_pdf(x, k, lambda))))
        }
        "weibull_cdf" => {
            if args.len() != 3 { return Err("weibull_cdf requires 3 arguments (x, k, lambda)".into()); }
            let x = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("weibull_cdf: x must be a number".into()) };
            let k = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("weibull_cdf: k must be a number".into()) };
            let lambda = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("weibull_cdf: lambda must be a number".into()) };
            Ok(Some(Value::Float(crate::distributions::weibull_cdf(x, k, lambda))))
        }

        // ── B5: Analyst QoL extensions ─────────────────────────────
        "case_when" => {
            if args.len() != 3 { return Err("case_when requires 3 arguments (conditions, values, default)".into()); }
            let conditions = match &args[0] {
                Value::Array(arr) => arr.iter().map(|v| match v {
                    Value::Bool(b) => Ok(*b),
                    _ => Err("case_when conditions must be booleans".into()),
                }).collect::<Result<Vec<bool>, String>>()?,
                _ => return Err("case_when conditions must be an array".into()),
            };
            let values = match &args[1] {
                Value::Array(arr) => arr.as_ref().clone(),
                _ => return Err("case_when values must be an array".into()),
            };
            if conditions.len() != values.len() {
                return Err("case_when conditions and values must have same length".into());
            }
            for (i, &cond) in conditions.iter().enumerate() {
                if cond { return Ok(Some(values[i].clone())); }
            }
            Ok(Some(args[2].clone())) // default
        }
        "ntile" => {
            if args.len() != 2 { return Err("ntile requires 2 arguments (data, n)".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let n = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("ntile: n must be an integer".into()) };
            let result = crate::stats::ntile(&data, n)?;
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }
        "percent_rank" => {
            if args.len() != 1 { return Err("percent_rank requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::percent_rank_fn(&data)?;
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }
        "cume_dist" => {
            if args.len() != 1 { return Err("cume_dist requires 1 argument".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let result = crate::stats::cume_dist(&data)?;
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }
        "wls" => {
            if args.len() != 5 { return Err("wls requires 5 arguments (X, y, weights, n, p)".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let w = value_to_f64_vec(&args[2])?;
            let n = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("wls: n must be an integer".into()) };
            let p = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("wls: p must be an integer".into()) };
            let r = crate::hypothesis::wls(&x, &y, &w, n, p)?;
            let fields = std::collections::BTreeMap::from([
                ("coefficients".to_string(), Value::Array(Rc::new(r.coefficients.into_iter().map(Value::Float).collect()))),
                ("r_squared".to_string(), Value::Float(r.r_squared)),
                ("residuals".to_string(), Value::Array(Rc::new(r.residuals.into_iter().map(Value::Float).collect()))),
            ]);
            Ok(Some(Value::Struct { name: "LmResult".to_string(), fields }))
        }

        // ── B7: Non-parametric tests & multiple comparisons ────────
        "tukey_hsd" => {
            let groups: Vec<Vec<f64>> = args.iter()
                .map(|a| value_to_f64_vec(a))
                .collect::<Result<Vec<_>, _>>()?;
            let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
            let results = crate::hypothesis::tukey_hsd(&group_refs)?;
            let result_values: Vec<Value> = results.iter().map(|pair| {
                let mut fields = std::collections::BTreeMap::new();
                fields.insert("group_i".into(), Value::Int(pair.group_i as i64));
                fields.insert("group_j".into(), Value::Int(pair.group_j as i64));
                fields.insert("mean_diff".into(), Value::Float(pair.mean_diff));
                fields.insert("q_statistic".into(), Value::Float(pair.q_statistic));
                fields.insert("p_value".into(), Value::Float(pair.p_value));
                Value::Struct { name: "TukeyHsdPair".into(), fields }
            }).collect();
            Ok(Some(Value::Array(Rc::new(result_values))))
        }
        "mann_whitney" => {
            if args.len() != 2 { return Err("mann_whitney requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let r = crate::hypothesis::mann_whitney(&x, &y)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("u_statistic".into(), Value::Float(r.u_statistic));
            fields.insert("z_score".into(), Value::Float(r.z_score));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            Ok(Some(Value::Struct { name: "MannWhitneyResult".into(), fields }))
        }
        "kruskal_wallis" => {
            let groups: Vec<Vec<f64>> = args.iter()
                .map(|a| value_to_f64_vec(a))
                .collect::<Result<Vec<_>, _>>()?;
            let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
            let r = crate::hypothesis::kruskal_wallis(&group_refs)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("h_statistic".into(), Value::Float(r.h_statistic));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            fields.insert("df".into(), Value::Float(r.df));
            Ok(Some(Value::Struct { name: "KruskalWallisResult".into(), fields }))
        }
        "wilcoxon_signed_rank" => {
            if args.len() != 2 { return Err("wilcoxon_signed_rank requires 2 arguments".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let r = crate::hypothesis::wilcoxon_signed_rank(&x, &y)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("w_statistic".into(), Value::Float(r.w_statistic));
            fields.insert("z_score".into(), Value::Float(r.z_score));
            fields.insert("p_value".into(), Value::Float(r.p_value));
            Ok(Some(Value::Struct { name: "WilcoxonResult".into(), fields }))
        }
        "bonferroni" => {
            if args.len() != 1 { return Err("bonferroni requires 1 argument (p_values array)".into()); }
            let pvals = value_to_f64_vec(&args[0])?;
            let adj = crate::hypothesis::bonferroni(&pvals);
            Ok(Some(Value::Array(Rc::new(adj.into_iter().map(Value::Float).collect()))))
        }
        "fdr_bh" => {
            if args.len() != 1 { return Err("fdr_bh requires 1 argument (p_values array)".into()); }
            let pvals = value_to_f64_vec(&args[0])?;
            let adj = crate::hypothesis::fdr_bh(&pvals);
            Ok(Some(Value::Array(Rc::new(adj.into_iter().map(Value::Float).collect()))))
        }
        "logistic_regression" => {
            if args.len() != 4 { return Err("logistic_regression requires 4 arguments (X, y, n, p)".into()); }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let n = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("logistic_regression: n must be an integer".into()) };
            let p = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("logistic_regression: p must be an integer".into()) };
            let r = crate::hypothesis::logistic_regression(&x, &y, n, p)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("coefficients".into(), Value::Array(Rc::new(r.coefficients.into_iter().map(Value::Float).collect())));
            fields.insert("std_errors".into(), Value::Array(Rc::new(r.std_errors.into_iter().map(Value::Float).collect())));
            fields.insert("z_values".into(), Value::Array(Rc::new(r.z_values.into_iter().map(Value::Float).collect())));
            fields.insert("p_values".into(), Value::Array(Rc::new(r.p_values.into_iter().map(Value::Float).collect())));
            fields.insert("log_likelihood".into(), Value::Float(r.log_likelihood));
            fields.insert("aic".into(), Value::Float(r.aic));
            fields.insert("iterations".into(), Value::Int(r.iterations as i64));
            Ok(Some(Value::Struct { name: "LogisticResult".into(), fields }))
        }

        // ── Stationarity tests ────────────────────────────────────────
        "adf_test" => {
            if args.len() != 1 { return Err("adf_test requires 1 argument: data".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let (t_stat, p_val) = crate::stationarity::adf_test(&data)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("statistic".into(), Value::Float(t_stat));
            fields.insert("p_value".into(), Value::Float(p_val));
            Ok(Some(Value::Struct { name: "AdfResult".into(), fields }))
        }
        "kpss_test" => {
            if args.len() != 1 { return Err("kpss_test requires 1 argument: data".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let (stat, p_val) = crate::stationarity::kpss_test(&data)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("statistic".into(), Value::Float(stat));
            fields.insert("p_value".into(), Value::Float(p_val));
            Ok(Some(Value::Struct { name: "KpssResult".into(), fields }))
        }
        "pp_test" => {
            if args.len() != 1 { return Err("pp_test requires 1 argument: data".into()); }
            let data = value_to_f64_vec(&args[0])?;
            let (z_t, p_val) = crate::stationarity::pp_test(&data)?;
            let mut fields = std::collections::BTreeMap::new();
            fields.insert("statistic".into(), Value::Float(z_t));
            fields.insert("p_value".into(), Value::Float(p_val));
            Ok(Some(Value::Struct { name: "PpResult".into(), fields }))
        }

        // Phase C4: Sorting & Tensor Indexing
        "argsort" => {
            if args.len() != 1 { return Err("argsort requires 1 arg: Tensor".into()); }
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.argsort())))
        }
        "gather" => {
            if args.len() != 3 { return Err("gather requires 3 args: tensor, dim, indices".into()); }
            let t = value_to_tensor(&args[0])?;
            let dim = value_to_usize(&args[1])?;
            let indices = value_to_tensor(&args[2])?;
            Ok(Some(Value::Tensor(t.gather(dim, &indices).map_err(|e| format!("{e}"))?)))
        }
        "scatter" => {
            if args.len() != 4 { return Err("scatter requires 4 args: tensor, dim, indices, src".into()); }
            let t = value_to_tensor(&args[0])?;
            let dim = value_to_usize(&args[1])?;
            let indices = value_to_tensor(&args[2])?;
            let src = value_to_tensor(&args[3])?;
            Ok(Some(Value::Tensor(t.scatter(dim, &indices, &src).map_err(|e| format!("{e}"))?)))
        }
        "index_select" => {
            if args.len() != 3 { return Err("index_select requires 3 args: tensor, dim, indices".into()); }
            let t = value_to_tensor(&args[0])?;
            let dim = value_to_usize(&args[1])?;
            let indices = value_to_tensor(&args[2])?;
            Ok(Some(Value::Tensor(t.index_select(dim, &indices).map_err(|e| format!("{e}"))?)))
        }

        // Phase C6: Collection utilities
        "array_push" => {
            if args.len() != 2 { return Err("array_push requires 2 args: array, value".into()); }
            let mut arr_rc = match &args[0] { Value::Array(a) => Rc::clone(a), _ => return Err("array_push: first arg must be Array".into()) };
            // COW: Rc::make_mut only clones if refcount > 1.
            // For `arr = array_push(arr, val)` where old binding is overwritten,
            // refcount is 1 → zero-copy push (amortized O(1) instead of O(n)).
            Rc::make_mut(&mut arr_rc).push(args[1].clone());
            Ok(Some(Value::Array(arr_rc)))
        }
        "array_pop" => {
            if args.len() != 1 { return Err("array_pop requires 1 arg: array".into()); }
            let arr = match &args[0] { Value::Array(a) => (**a).clone(), _ => return Err("array_pop: expected Array".into()) };
            if arr.is_empty() { return Err("array_pop: empty array".into()); }
            let mut new_arr = arr;
            let last = new_arr.pop().unwrap();
            Ok(Some(Value::Tuple(Rc::new(vec![last, Value::Array(Rc::new(new_arr))]))))
        }
        "array_contains" => {
            if args.len() != 2 { return Err("array_contains requires 2 args: array, value".into()); }
            let arr = match &args[0] { Value::Array(a) => a, _ => return Err("array_contains: first arg must be Array".into()) };
            let needle = &args[1];
            let found = arr.iter().any(|v| format!("{v}") == format!("{needle}"));
            Ok(Some(Value::Bool(found)))
        }
        "array_reverse" => {
            if args.len() != 1 { return Err("array_reverse requires 1 arg: array".into()); }
            let arr = match &args[0] { Value::Array(a) => (**a).clone(), _ => return Err("array_reverse: expected Array".into()) };
            let mut new_arr = arr;
            new_arr.reverse();
            Ok(Some(Value::Array(Rc::new(new_arr))))
        }
        "array_flatten" => {
            if args.len() != 1 { return Err("array_flatten requires 1 arg: array".into()); }
            let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("array_flatten: expected Array".into()) };
            let mut result = Vec::new();
            fn flatten_recursive(arr: &[Value], result: &mut Vec<Value>) {
                for v in arr {
                    match v {
                        Value::Array(inner) => flatten_recursive(inner, result),
                        _ => result.push(v.clone()),
                    }
                }
            }
            flatten_recursive(&arr, &mut result);
            Ok(Some(Value::Array(Rc::new(result))))
        }
        "array_len" => {
            if args.len() != 1 { return Err("array_len requires 1 arg: array".into()); }
            match &args[0] {
                Value::Array(a) => Ok(Some(Value::Int(a.len() as i64))),
                _ => Err("array_len: expected Array".into()),
            }
        }
        "array_slice" => {
            if args.len() != 3 { return Err("array_slice requires 3 args: array, start, end".into()); }
            let arr = match &args[0] { Value::Array(a) => a, _ => return Err("array_slice: expected Array".into()) };
            let start = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("array_slice: start must be Int".into()) };
            let end = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("array_slice: end must be Int".into()) };
            if start > end || end > arr.len() {
                return Err(format!("array_slice: bounds [{start}, {end}) out of range for len {}", arr.len()));
            }
            Ok(Some(Value::Array(Rc::new(arr[start..end].to_vec()))))
        }

        // Phase C5: Map & Set constructors
        "Map.new" => {
            if !args.is_empty() { return Err("Map.new takes 0 arguments".into()); }
            Ok(Some(Value::Map(Rc::new(RefCell::new(crate::det_map::DetMap::new())))))
        }
        "Set.new" => {
            if !args.is_empty() { return Err("Set.new takes 0 arguments".into()); }
            Ok(Some(Value::Map(Rc::new(RefCell::new(crate::det_map::DetMap::new())))))
        }

        // Phase C3: Bitwise operations
        "bit_and" => {
            if args.len() != 2 { return Err("bit_and requires 2 Int args".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_and: expected Int".into()) };
            let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_and: expected Int".into()) };
            Ok(Some(Value::Int(a & b)))
        }
        "bit_or" => {
            if args.len() != 2 { return Err("bit_or requires 2 Int args".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_or: expected Int".into()) };
            let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_or: expected Int".into()) };
            Ok(Some(Value::Int(a | b)))
        }
        "bit_xor" => {
            if args.len() != 2 { return Err("bit_xor requires 2 Int args".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_xor: expected Int".into()) };
            let b = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_xor: expected Int".into()) };
            Ok(Some(Value::Int(a ^ b)))
        }
        "bit_not" => {
            if args.len() != 1 { return Err("bit_not requires 1 Int arg".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_not: expected Int".into()) };
            Ok(Some(Value::Int(!a)))
        }
        "bit_shl" => {
            if args.len() != 2 { return Err("bit_shl requires 2 Int args".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_shl: expected Int".into()) };
            let n = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_shl: expected Int".into()) };
            if n < 0 || n > 63 { return Err("bit_shl: shift amount must be 0-63".into()); }
            Ok(Some(Value::Int(((a as u64) << n) as i64)))
        }
        "bit_shr" => {
            if args.len() != 2 { return Err("bit_shr requires 2 Int args".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("bit_shr: expected Int".into()) };
            let n = match &args[1] { Value::Int(i) => *i, _ => return Err("bit_shr: expected Int".into()) };
            if n < 0 || n > 63 { return Err("bit_shr: shift amount must be 0-63".into()); }
            Ok(Some(Value::Int(((a as u64) >> n) as i64)))
        }
        "popcount" => {
            if args.len() != 1 { return Err("popcount requires 1 Int arg".into()); }
            let a = match &args[0] { Value::Int(i) => *i, _ => return Err("popcount: expected Int".into()) };
            Ok(Some(Value::Int((a as u64).count_ones() as i64)))
        }

        // Phase C2: Optimizer constructors
        "Adam.new" => {
            if args.len() < 2 || args.len() > 4 {
                return Err("Adam.new requires 2-4 args: n_params, lr, [beta1], [beta2]".into());
            }
            let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("Adam.new: n_params must be Int".into()) };
            let lr = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("Adam.new: lr must be Float".into()) };
            let beta1 = if args.len() > 2 {
                match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("Adam.new: beta1 must be Float".into()) }
            } else { 0.9 };
            let beta2 = if args.len() > 3 {
                match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("Adam.new: beta2 must be Float".into()) }
            } else { 0.999 };
            let mut state = crate::ml::AdamState::new(n, lr);
            state.beta1 = beta1;
            state.beta2 = beta2;
            let erased: Rc<RefCell<dyn std::any::Any>> = Rc::new(RefCell::new(state));
            Ok(Some(Value::OptimizerState(erased)))
        }
        "Sgd.new" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("Sgd.new requires 2-3 args: n_params, lr, [momentum]".into());
            }
            let n = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("Sgd.new: n_params must be Int".into()) };
            let lr = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("Sgd.new: lr must be Float".into()) };
            let momentum = if args.len() > 2 {
                match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("Sgd.new: momentum must be Float".into()) }
            } else { 0.0 };
            let state = crate::ml::SgdState::new(n, lr, momentum);
            let erased: Rc<RefCell<dyn std::any::Any>> = Rc::new(RefCell::new(state));
            Ok(Some(Value::OptimizerState(erased)))
        }

        // ---- ML Autodiff Builtins ----
        // stop_gradient: returns x unchanged; in AD context, gradients don't flow through
        "stop_gradient" => {
            if args.len() != 1 { return Err("stop_gradient requires exactly 1 argument".into()); }
            Ok(Some(args[0].clone()))
        }
        // grad_checkpoint: returns x unchanged; semantic marker for memory checkpointing
        "grad_checkpoint" => {
            if args.len() != 1 { return Err("grad_checkpoint requires exactly 1 argument".into()); }
            Ok(Some(args[0].clone()))
        }
        // clip_grad: clips a gradient value to [min_val, max_val] range
        "clip_grad" => {
            if args.len() != 3 { return Err("clip_grad requires 3 arguments (value, min, max)".into()); }
            let val = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => return Err("clip_grad requires numeric arguments".into()),
            };
            let min_val = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("clip_grad min must be numeric".into()) };
            let max_val = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("clip_grad max must be numeric".into()) };
            Ok(Some(Value::Float(val.max(min_val).min(max_val))))
        }
        // grad_scale: scales a gradient value by a scalar factor
        "grad_scale" => {
            if args.len() != 2 { return Err("grad_scale requires 2 arguments (value, scale)".into()); }
            let val = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("grad_scale requires numeric first arg".into()) };
            let scale = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("grad_scale requires numeric scale".into()) };
            Ok(Some(Value::Float(val * scale)))
        }

        // ── v0.1 Broadcasting builtins ──────────────────────────────

        "broadcast" => {
            if args.len() != 2 {
                return Err("broadcast requires 2 arguments (fn_name, tensor)".into());
            }
            let fn_name = match &args[0] {
                Value::String(s) => s.clone(),
                _ => return Err("broadcast: first argument must be a function name string".into()),
            };
            let t = value_to_tensor(&args[1])?;

            // SIMD-accelerated path for known unary operations that can be
            // vectorized with AVX2 (bit-identical to scalar).
            match fn_name.as_str() {
                "sqrt" => return Ok(Some(Value::Tensor(t.map_simd(UnaryOp::Sqrt)))),
                "abs"  => return Ok(Some(Value::Tensor(t.map_simd(UnaryOp::Abs)))),
                "neg"  => return Ok(Some(Value::Tensor(t.map_simd(UnaryOp::Neg)))),
                "relu" => return Ok(Some(Value::Tensor(t.map_simd(UnaryOp::Relu)))),
                _ => {} // fall through to scalar path
            }

            // Scalar path for transcendental functions (sin, cos, exp, etc.)
            // These cannot be trivially SIMD-vectorized while preserving
            // bit-identical results with libm scalar implementations.
            let f: Box<dyn Fn(f64) -> f64> = match fn_name.as_str() {
                "sin"     => Box::new(|x: f64| x.sin()),
                "cos"     => Box::new(|x: f64| x.cos()),
                "tan"     => Box::new(|x: f64| x.tan()),
                "asin"    => Box::new(|x: f64| x.asin()),
                "acos"    => Box::new(|x: f64| x.acos()),
                "atan"    => Box::new(|x: f64| x.atan()),
                "exp"     => Box::new(|x: f64| x.exp()),
                "ln"      => Box::new(|x: f64| x.ln()),
                "log"     => Box::new(|x: f64| x.ln()),
                "log2"    => Box::new(|x: f64| x.log2()),
                "log10"   => Box::new(|x: f64| x.log10()),
                "log1p"   => Box::new(|x: f64| x.ln_1p()),
                "expm1"   => Box::new(|x: f64| x.exp_m1()),
                "floor"   => Box::new(|x: f64| x.floor()),
                "ceil"    => Box::new(|x: f64| x.ceil()),
                "round"   => Box::new(|x: f64| x.round()),
                "sigmoid" => Box::new(|x: f64| 1.0 / (1.0 + (-x).exp())),
                "tanh"    => Box::new(|x: f64| x.tanh()),
                "sign"    => Box::new(|x: f64| {
                    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
                }),
                _ => return Err(format!("broadcast: unknown unary function '{fn_name}'")),
            };
            Ok(Some(Value::Tensor(t.map(f))))
        }

        "broadcast2" => {
            if args.len() != 3 {
                return Err("broadcast2 requires 3 arguments (fn_name, tensor1, tensor2)".into());
            }
            let fn_name = match &args[0] {
                Value::String(s) => s.clone(),
                _ => return Err("broadcast2: first argument must be a function name string".into()),
            };
            let t1 = value_to_tensor(&args[1])?;
            let t2 = value_to_tensor(&args[2])?;
            let result = match fn_name.as_str() {
                "add"   => t1.add(&t2),
                "sub"   => t1.sub(&t2),
                "mul"   => t1.mul_elem(&t2),
                "div"   => t1.div_elem(&t2),
                "pow"   => t1.elem_pow(&t2),
                "min"   => t1.elem_min(&t2),
                "max"   => t1.elem_max(&t2),
                "atan2" => t1.elem_atan2(&t2),
                "hypot" => t1.elem_hypot(&t2),
                _ => return Err(format!("broadcast2: unknown binary function '{fn_name}'")),
            };
            match result {
                Ok(t) => Ok(Some(Value::Tensor(t))),
                Err(e) => Err(format!("broadcast2: {e}")),
            }
        }

        // ── Peak RSS memory tracking ─────────────────────────────────
        "peak_rss" => {
            Ok(Some(Value::Int(peak_rss_kb() as i64)))
        }

        // ── Fused broadcast operations (eliminate intermediate tensors) ──
        "broadcast_fma" => {
            // broadcast_fma(a, b, c) = a * b + c element-wise in one pass.
            // Eliminates the intermediate tensor that broadcast2("mul") would create.
            if args.len() != 3 {
                return Err("broadcast_fma requires 3 arguments (a, b, c)".into());
            }
            let a = value_to_tensor(&args[0])?;
            let b = value_to_tensor(&args[1])?;
            let c = value_to_tensor(&args[2])?;
            let result = a.fused_mul_add(&b, &c)
                .map_err(|e| format!("broadcast_fma: {e}"))?;
            Ok(Some(Value::Tensor(result)))
        }

        // -- Phase 2: Tensor boolean/masking ops --------------------------------
        "tensor_where" => {
            let a = value_to_tensor(&args[0])?;
            let cond = value_to_tensor(&args[1])?;
            let other = value_to_tensor(&args[2])?;
            Ok(Some(Value::Tensor(a.tensor_where(cond, other).map_err(|e| format!("{e}"))?)))
        }
        "tensor_any" => {
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Bool(t.any())))
        }
        "tensor_all" => {
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Bool(t.all())))
        }
        "tensor_nonzero" => {
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.nonzero())))
        }
        "tensor_masked_fill" => {
            let t = value_to_tensor(&args[0])?;
            let mask = value_to_tensor(&args[1])?;
            let val = value_to_f64(&args[2])?;
            Ok(Some(Value::Tensor(t.masked_fill(mask, val).map_err(|e| format!("{e}"))?)))
        }
        // -- Phase 2: Axis reductions ------------------------------------------
        "tensor_mean_axis" => {
            let t = value_to_tensor(&args[0])?;
            let axis = value_to_usize(&args[1])?;
            let keepdim = matches!(args.get(2), Some(Value::Bool(true)));
            Ok(Some(Value::Tensor(t.mean_axis(axis, keepdim).map_err(|e| format!("{e}"))?)))
        }
        "tensor_var_axis" => {
            let t = value_to_tensor(&args[0])?;
            let axis = value_to_usize(&args[1])?;
            let keepdim = matches!(args.get(2), Some(Value::Bool(true)));
            Ok(Some(Value::Tensor(t.var_axis(axis, keepdim).map_err(|e| format!("{e}"))?)))
        }
        "tensor_std_axis" => {
            let t = value_to_tensor(&args[0])?;
            let axis = value_to_usize(&args[1])?;
            let keepdim = matches!(args.get(2), Some(Value::Bool(true)));
            Ok(Some(Value::Tensor(t.std_axis(axis, keepdim).map_err(|e| format!("{e}"))?)))
        }
        "tensor_prod_axis" => {
            let t = value_to_tensor(&args[0])?;
            let axis = value_to_usize(&args[1])?;
            let keepdim = matches!(args.get(2), Some(Value::Bool(true)));
            Ok(Some(Value::Tensor(t.prod_axis(axis, keepdim).map_err(|e| format!("{e}"))?)))
        }
        // -- Phase 2: Sort ops -------------------------------------------------
        "tensor_sort" => {
            let t = value_to_tensor(&args[0])?;
            let axis = value_to_usize(&args[1])?;
            let desc = matches!(args.get(2), Some(Value::Bool(true)));
            Ok(Some(Value::Tensor(t.sort_axis(axis, desc).map_err(|e| format!("{e}"))?)))
        }
        "tensor_argsort_axis" => {
            let t = value_to_tensor(&args[0])?;
            let axis = value_to_usize(&args[1])?;
            let desc = matches!(args.get(2), Some(Value::Bool(true)));
            Ok(Some(Value::Tensor(t.argsort_axis(axis, desc).map_err(|e| format!("{e}"))?)))
        }
        // -- Phase 2: Einsum ---------------------------------------------------
        "einsum" => {
            let notation = match &args[0] {
                Value::String(s) => s.as_str().to_string(),
                _ => return Err("einsum: first arg must be notation string".into()),
            };
            let tensors: Vec<&Tensor> = args[1..].iter()
                .map(value_to_tensor)
                .collect::<Result<_, _>>()?;
            let result = Tensor::einsum(&notation, &tensors).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tensor(result)))
        }
        // -- Phase 2: Reshape enhancements -------------------------------------
        "tensor_unsqueeze" => {
            let t = value_to_tensor(&args[0])?;
            let dim = value_to_usize(&args[1])?;
            Ok(Some(Value::Tensor(t.unsqueeze(dim).map_err(|e| format!("{e}"))?)))
        }
        "tensor_squeeze" => {
            let t = value_to_tensor(&args[0])?;
            let dim = args.get(1).map(|v| value_to_usize(v)).transpose()?;
            Ok(Some(Value::Tensor(t.squeeze(dim).map_err(|e| format!("{e}"))?)))
        }
        "tensor_flatten" => {
            let t = value_to_tensor(&args[0])?;
            let start = value_to_usize(&args[1])?;
            let end = value_to_usize(&args[2])?;
            Ok(Some(Value::Tensor(t.flatten(start, end).map_err(|e| format!("{e}"))?)))
        }
        "tensor_chunk" => {
            let t = value_to_tensor(&args[0])?;
            let n = value_to_usize(&args[1])?;
            let dim = value_to_usize(&args[2])?;
            let chunks = t.chunk(n, dim).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Array(Rc::new(chunks.into_iter().map(Value::Tensor).collect()))))
        }
        // -- Phase 3: SVD, PCA, Pseudoinverse ----------------------------------
        "svd" => {
            let t = value_to_tensor(&args[0])?;
            let (u, s, vt) = t.svd().map_err(|e| format!("{e}"))?;
            let s_tensor = Tensor::from_vec(s, &[u.shape()[1]]).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tuple(Rc::new(vec![
                Value::Tensor(u),
                Value::Tensor(s_tensor),
                Value::Tensor(vt),
            ]))))
        }
        "pinv" => {
            let t = value_to_tensor(&args[0])?;
            Ok(Some(Value::Tensor(t.pinv().map_err(|e| format!("{e}"))?)))
        }
        "pca" => {
            let t = value_to_tensor(&args[0])?;
            let n_components = value_to_usize(&args[1])?;
            let (transformed, components, variance) = crate::ml::pca(&t, n_components).map_err(|e| format!("{e}"))?;
            let vlen = variance.len();
            let var_tensor = Tensor::from_vec(variance, &[vlen]).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tuple(Rc::new(vec![
                Value::Tensor(transformed),
                Value::Tensor(components),
                Value::Tensor(var_tensor),
            ]))))
        }
        // -- Phase 7: Sparse operations ----------------------------------------
        "sparse_add" => {
            let a = value_to_sparse(&args[0])?;
            let b = value_to_sparse(&args[1])?;
            Ok(Some(Value::SparseTensor(crate::sparse::sparse_add(a, b).map_err(|e| e)?)))
        }
        "sparse_sub" => {
            let a = value_to_sparse(&args[0])?;
            let b = value_to_sparse(&args[1])?;
            Ok(Some(Value::SparseTensor(crate::sparse::sparse_sub(a, b).map_err(|e| e)?)))
        }
        "sparse_matmul" => {
            let a = value_to_sparse(&args[0])?;
            let b = value_to_sparse(&args[1])?;
            Ok(Some(Value::SparseTensor(crate::sparse::sparse_matmul(a, b).map_err(|e| e)?)))
        }
        "sparse_transpose" => {
            let a = value_to_sparse(&args[0])?;
            Ok(Some(Value::SparseTensor(crate::sparse::sparse_transpose(a))))
        }
        // -- Phase 9: Clustering -----------------------------------------------
        "kmeans" => {
            let data = value_to_f64_vec(&args[0])?;
            let n_samples = value_to_usize(&args[1])?;
            let n_features = value_to_usize(&args[2])?;
            let k = value_to_usize(&args[3])?;
            let max_iter = value_to_usize(&args[4])?;
            let seed = match &args[5] { Value::Int(v) => *v as u64, _ => 42 };
            let (centroids, labels, inertia) = crate::clustering::kmeans(&data, n_samples, n_features, k, max_iter, seed);
            let label_vals: Vec<Value> = labels.iter().map(|&l| Value::Int(l as i64)).collect();
            let centroid_t = Tensor::from_vec(centroids, &[k, n_features]).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tuple(Rc::new(vec![
                Value::Tensor(centroid_t),
                Value::Array(Rc::new(label_vals)),
                Value::Float(inertia),
            ]))))
        }
        "dbscan" => {
            let data = value_to_f64_vec(&args[0])?;
            let n_samples = value_to_usize(&args[1])?;
            let n_features = value_to_usize(&args[2])?;
            let eps = value_to_f64(&args[3])?;
            let min_samples = value_to_usize(&args[4])?;
            let labels = crate::clustering::dbscan(&data, n_samples, n_features, eps, min_samples);
            let label_vals: Vec<Value> = labels.iter().map(|&l| Value::Int(l)).collect();
            Ok(Some(Value::Array(Rc::new(label_vals))))
        }
        // -- Phase 10: Categorical encoding ------------------------------------
        "label_encode" => {
            // label_encode: convert array of strings to (sorted_levels, codes)
            // Uses BTreeSet for deterministic sorted order
            let strs: Vec<String> = match &args[0] {
                Value::Array(arr) => arr.iter().map(|v| match v {
                    Value::String(s) => Ok(s.as_str().to_string()),
                    _ => Err("label_encode: expected array of strings".to_string()),
                }).collect::<Result<_, _>>()?,
                _ => return Err("label_encode: expected array".into()),
            };
            let mut level_set = std::collections::BTreeSet::new();
            for s in &strs { level_set.insert(s.clone()); }
            let levels: Vec<String> = level_set.into_iter().collect();
            let level_map: std::collections::BTreeMap<&str, u32> = levels.iter().enumerate()
                .map(|(i, s)| (s.as_str(), i as u32)).collect();
            let codes: Vec<u32> = strs.iter().map(|s| level_map[s.as_str()]).collect();
            let level_vals: Vec<Value> = levels.iter().map(|s| Value::String(Rc::new(s.clone()))).collect();
            let code_vals: Vec<Value> = codes.iter().map(|&c| Value::Int(c as i64)).collect();
            Ok(Some(Value::Tuple(Rc::new(vec![
                Value::Array(Rc::new(level_vals)),
                Value::Array(Rc::new(code_vals)),
            ]))))
        }
        // -- Phase 11: Time series ---------------------------------------------
        "acf" => {
            let data = value_to_f64_vec(&args[0])?;
            let max_lag = value_to_usize(&args[1])?;
            let result = crate::timeseries::acf(&data, max_lag);
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }
        "ewma" => {
            let data = value_to_f64_vec(&args[0])?;
            let alpha = value_to_f64(&args[1])?;
            let result = crate::timeseries::ewma(&data, alpha);
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }
        "diff" => {
            let data = value_to_f64_vec(&args[0])?;
            let periods = value_to_usize(&args[1])?;
            let result = crate::timeseries::diff(&data, periods);
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }
        // -- Phase 5: Optimization root finding --------------------------------
        "bisect" => {
            // bisect expects a closure as first arg — handled by executor, not here
            Ok(None)
        }
        // -- Phase 6: Interpolation --------------------------------------------
        "polyfit" => {
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let degree = value_to_usize(&args[2])?;
            let coeffs = crate::interpolate::polyfit(&x, &y, degree).map_err(|e| e)?;
            Ok(Some(Value::Array(Rc::new(coeffs.into_iter().map(Value::Float).collect()))))
        }
        "polyval" => {
            let coeffs = value_to_f64_vec(&args[0])?;
            let x = value_to_f64_vec(&args[1])?;
            let result = crate::interpolate::polyval(&coeffs, &x);
            Ok(Some(Value::Array(Rc::new(result.into_iter().map(Value::Float).collect()))))
        }

        _ => Ok(None), // Not a shared builtin
    }
}

/// Extract a SparseCsr reference from a Value.
fn value_to_sparse(val: &Value) -> Result<&crate::sparse::SparseCsr, String> {
    match val {
        Value::SparseTensor(s) => Ok(s),
        _ => Err(format!("expected SparseTensor, got {}", val.type_name())),
    }
}

// ---------------------------------------------------------------------------
// Peak RSS memory tracking (platform-specific)
// ---------------------------------------------------------------------------

/// Returns peak resident set size in kilobytes.
///
/// Platform support:
/// - **Windows**: `GetProcessMemoryInfo` → `PeakWorkingSetSize`
/// - **Linux**: Reads `/proc/self/status` → `VmHWM`
/// - **macOS**: `getrusage(RUSAGE_SELF)` → `ru_maxrss` (in bytes on macOS)
/// - **Other**: Returns 0
pub fn peak_rss_kb() -> u64 {
    #[cfg(target_os = "windows")]
    {
        peak_rss_windows()
    }
    #[cfg(target_os = "linux")]
    {
        peak_rss_linux()
    }
    #[cfg(target_os = "macos")]
    {
        peak_rss_macos()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

#[cfg(target_os = "windows")]
fn peak_rss_windows() -> u64 {
    use std::mem::{size_of, MaybeUninit};

    #[repr(C)]
    #[allow(non_snake_case)]
    struct ProcessMemoryCounters {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }

    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            hProcess: isize,
            ppsmemCounters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    unsafe {
        let mut pmc = MaybeUninit::<ProcessMemoryCounters>::zeroed();
        let pmc_ref = pmc.as_mut_ptr();
        (*pmc_ref).cb = size_of::<ProcessMemoryCounters>() as u32;
        let handle = GetCurrentProcess();
        if K32GetProcessMemoryInfo(handle, pmc_ref, (*pmc_ref).cb) != 0 {
            let pmc = pmc.assume_init();
            (pmc.PeakWorkingSetSize / 1024) as u64
        } else {
            0
        }
    }
}

#[cfg(target_os = "linux")]
fn peak_rss_linux() -> u64 {
    // Read VmHWM from /proc/self/status (peak resident set size in kB).
    if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmHWM:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<u64>() {
                        return kb;
                    }
                }
            }
        }
    }
    0
}

#[cfg(target_os = "macos")]
fn peak_rss_macos() -> u64 {
    #[repr(C)]
    struct Rusage {
        ru_utime: [i64; 2],  // timeval (tv_sec, tv_usec)
        ru_stime: [i64; 2],  // timeval
        ru_maxrss: i64,      // max resident set size (bytes on macOS)
        // ... remaining fields omitted (we only need maxrss)
        _padding: [i64; 11],
    }

    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    unsafe {
        let mut usage = std::mem::MaybeUninit::<Rusage>::zeroed();
        if getrusage(0 /* RUSAGE_SELF */, usage.as_mut_ptr()) == 0 {
            let usage = usage.assume_init();
            // macOS reports in bytes, convert to KB
            (usage.ru_maxrss as u64) / 1024
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_rss_returns_nonzero() {
        let rss = peak_rss_kb();
        // On any platform we support, peak RSS should be > 0 for a running process.
        assert!(rss > 0, "peak_rss_kb() should return non-zero, got {rss}");
    }

    #[test]
    fn test_peak_rss_builtin_dispatch() {
        let result = dispatch_builtin("peak_rss", &[]);
        match result {
            Ok(Some(Value::Int(kb))) => {
                assert!(kb > 0, "peak_rss should return positive value, got {kb}");
            }
            other => panic!("Expected Ok(Some(Int)), got: {other:?}"),
        }
    }

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

    #[test]
    fn test_log_float() {
        let result = dispatch_builtin("log", &[Value::Float(1.0)]);
        match result {
            Ok(Some(Value::Float(v))) => assert!((v - 0.0).abs() < 1e-15),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_log_e() {
        let result = dispatch_builtin("log", &[Value::Float(std::f64::consts::E)]);
        match result {
            Ok(Some(Value::Float(v))) => assert!((v - 1.0).abs() < 1e-15),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_exp_float() {
        let result = dispatch_builtin("exp", &[Value::Float(0.0)]);
        match result {
            Ok(Some(Value::Float(v))) => assert!((v - 1.0).abs() < 1e-15),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_exp_one() {
        let result = dispatch_builtin("exp", &[Value::Float(1.0)]);
        match result {
            Ok(Some(Value::Float(v))) => assert!((v - std::f64::consts::E).abs() < 1e-15),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_categorical_sample_deterministic() {
        let probs = Tensor::from_vec(vec![0.0, 0.0, 1.0], &[3]).unwrap();
        // u=0.5, cumsum: 0.0, 0.0, 1.0 → picks index 2
        assert_eq!(categorical_sample_with_u(&probs, 0.5).unwrap(), 2);
    }

    #[test]
    fn test_categorical_sample_first() {
        let probs = Tensor::from_vec(vec![0.5, 0.3, 0.2], &[3]).unwrap();
        // u=0.1 → picks index 0 (cumsum 0.5 > 0.1)
        assert_eq!(categorical_sample_with_u(&probs, 0.1).unwrap(), 0);
    }

    #[test]
    fn test_categorical_sample_middle() {
        let probs = Tensor::from_vec(vec![0.2, 0.5, 0.3], &[3]).unwrap();
        // u=0.6 → cumsum 0.2, 0.7 → picks index 1
        assert_eq!(categorical_sample_with_u(&probs, 0.6).unwrap(), 1);
    }

    #[test]
    fn test_categorical_sample_last() {
        let probs = Tensor::from_vec(vec![0.2, 0.3, 0.5], &[3]).unwrap();
        // u=0.99 → cumsum 0.2, 0.5, 1.0 → picks index 2
        assert_eq!(categorical_sample_with_u(&probs, 0.99).unwrap(), 2);
    }
}
