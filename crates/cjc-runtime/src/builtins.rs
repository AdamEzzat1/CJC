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
