//! Decoding of canonically-encoded CJC values.
//!
//! Reads the binary format produced by `encode::snap_encode` and reconstructs
//! a `Value`. Returns `SnapError` on malformed input.

use std::collections::BTreeMap;
use std::rc::Rc;
use std::cell::RefCell;

use cjc_runtime::value::Bf16;
use cjc_runtime::complex::ComplexF64;
use cjc_runtime::f16::F16;
use cjc_runtime::{Tensor, Value};

use crate::encode::*;
use crate::SnapError;

// ---------------------------------------------------------------------------
// Cursor -- a simple reader over a byte slice
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    #[allow(dead_code)]
    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_byte(&mut self) -> Result<u8, SnapError> {
        if self.pos >= self.data.len() {
            return Err(SnapError::UnexpectedEof);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], SnapError> {
        if self.pos + n > self.data.len() {
            return Err(SnapError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u16_le(&mut self) -> Result<u16, SnapError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u64_le(&mut self) -> Result<u64, SnapError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_i64_le(&mut self) -> Result<i64, SnapError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_f64_le(&mut self) -> Result<f64, SnapError> {
        let bits = self.read_u64_le()?;
        Ok(f64::from_bits(bits))
    }

    fn read_string(&mut self) -> Result<String, SnapError> {
        let len = self.read_u64_le()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| SnapError::Utf8Error)
    }

    fn read_raw_bytes(&mut self) -> Result<Vec<u8>, SnapError> {
        let len = self.read_u64_le()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(bytes.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Public decoder
// ---------------------------------------------------------------------------

/// Decode a byte slice back into a CJC `Value`.
///
/// Returns `SnapError` if the data is malformed, truncated, or contains an
/// invalid tag byte.
pub fn snap_decode(data: &[u8]) -> Result<Value, SnapError> {
    let mut cursor = Cursor::new(data);
    let value = decode_value(&mut cursor)?;
    Ok(value)
}

fn decode_value(cursor: &mut Cursor<'_>) -> Result<Value, SnapError> {
    let tag = cursor.read_byte()?;
    match tag {
        TAG_VOID => Ok(Value::Void),

        TAG_INT => {
            let v = cursor.read_i64_le()?;
            Ok(Value::Int(v))
        }

        TAG_FLOAT => {
            let v = cursor.read_f64_le()?;
            Ok(Value::Float(v))
        }

        TAG_BOOL => {
            let b = cursor.read_byte()?;
            Ok(Value::Bool(b != 0))
        }

        TAG_STRING => {
            let s = cursor.read_string()?;
            Ok(Value::String(Rc::new(s)))
        }

        TAG_BYTES => {
            let data = cursor.read_raw_bytes()?;
            Ok(Value::Bytes(Rc::new(RefCell::new(data))))
        }

        TAG_BYTESLICE => {
            let data = cursor.read_raw_bytes()?;
            Ok(Value::ByteSlice(Rc::new(data)))
        }

        TAG_STRVIEW => {
            let data = cursor.read_raw_bytes()?;
            // Validate UTF-8
            if std::str::from_utf8(&data).is_err() {
                return Err(SnapError::Utf8Error);
            }
            Ok(Value::StrView(Rc::new(data)))
        }

        TAG_U8 => {
            let v = cursor.read_byte()?;
            Ok(Value::U8(v))
        }

        TAG_ARRAY => {
            let len = cursor.read_u64_le()? as usize;
            let mut elems = Vec::with_capacity(len);
            for _ in 0..len {
                elems.push(decode_value(cursor)?);
            }
            Ok(Value::Array(Rc::new(elems)))
        }

        TAG_TUPLE => {
            let len = cursor.read_u64_le()? as usize;
            let mut elems = Vec::with_capacity(len);
            for _ in 0..len {
                elems.push(decode_value(cursor)?);
            }
            Ok(Value::Tuple(Rc::new(elems)))
        }

        TAG_STRUCT => {
            let name = cursor.read_string()?;
            let count = cursor.read_u64_le()? as usize;
            let mut fields = BTreeMap::new();
            for _ in 0..count {
                let key = cursor.read_string()?;
                let val = decode_value(cursor)?;
                fields.insert(key, val);
            }
            Ok(Value::Struct { name, fields })
        }

        TAG_TENSOR => {
            let ndim = cursor.read_u64_le()? as usize;
            let mut shape = Vec::with_capacity(ndim);
            let mut numel: usize = 1;
            for _ in 0..ndim {
                let dim = cursor.read_u64_le()? as usize;
                numel = numel.saturating_mul(dim);
                shape.push(dim);
            }
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                data.push(cursor.read_f64_le()?);
            }
            let tensor = Tensor::from_vec(data, &shape)
                .map_err(|_| SnapError::UnexpectedEof)?;
            Ok(Value::Tensor(tensor))
        }

        TAG_ENUM => {
            let enum_name = cursor.read_string()?;
            let variant = cursor.read_string()?;
            let count = cursor.read_u64_le()? as usize;
            let mut fields = Vec::with_capacity(count);
            for _ in 0..count {
                fields.push(decode_value(cursor)?);
            }
            Ok(Value::Enum {
                enum_name,
                variant,
                fields,
            })
        }

        TAG_BF16 => {
            let bits = cursor.read_u16_le()?;
            Ok(Value::Bf16(Bf16(bits)))
        }

        TAG_F16 => {
            let bits = cursor.read_u16_le()?;
            Ok(Value::F16(F16(bits)))
        }

        TAG_COMPLEX => {
            let re = cursor.read_f64_le()?;
            let im = cursor.read_f64_le()?;
            Ok(Value::Complex(ComplexF64 { re, im }))
        }

        TAG_MAP => {
            let count = cursor.read_u64_le()? as usize;
            let map = cjc_runtime::DetMap::new();
            let map_ref = Rc::new(RefCell::new(map));
            for _ in 0..count {
                let key = decode_value(cursor)?;
                let val = decode_value(cursor)?;
                map_ref.borrow_mut().insert(key, val);
            }
            Ok(Value::Map(map_ref))
        }

        other => Err(SnapError::InvalidTag(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::snap_encode;

    /// Round-trip helper: encode then decode, verify no error.
    fn roundtrip(value: &Value) -> Value {
        let bytes = snap_encode(value);
        snap_decode(&bytes).expect("decode failed")
    }

    #[test]
    fn test_roundtrip_void() {
        let v = roundtrip(&Value::Void);
        assert!(matches!(v, Value::Void));
    }

    #[test]
    fn test_roundtrip_int() {
        let v = roundtrip(&Value::Int(42));
        assert!(matches!(v, Value::Int(42)));
    }

    #[test]
    fn test_roundtrip_negative_int() {
        let v = roundtrip(&Value::Int(-999));
        assert!(matches!(v, Value::Int(-999)));
    }

    #[test]
    fn test_roundtrip_float() {
        let v = roundtrip(&Value::Float(3.14));
        match v {
            Value::Float(f) => assert_eq!(f, 3.14),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_roundtrip_nan() {
        let v = roundtrip(&Value::Float(f64::NAN));
        match v {
            Value::Float(f) => assert!(f.is_nan()),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_roundtrip_bool() {
        assert!(matches!(roundtrip(&Value::Bool(true)), Value::Bool(true)));
        assert!(matches!(roundtrip(&Value::Bool(false)), Value::Bool(false)));
    }

    #[test]
    fn test_roundtrip_string() {
        let orig = Value::String(Rc::new("hello world".to_string()));
        let decoded = roundtrip(&orig);
        match decoded {
            Value::String(s) => assert_eq!(s.as_str(), "hello world"),
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn test_roundtrip_array() {
        let orig = Value::Array(Rc::new(vec![
            Value::Int(1),
            Value::Float(2.0),
            Value::Bool(true),
        ]));
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert!(matches!(arr[0], Value::Int(1)));
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_roundtrip_tuple() {
        let orig = Value::Tuple(Rc::new(vec![Value::Int(10), Value::Int(20)]));
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Tuple(t) => assert_eq!(t.len(), 2),
            _ => panic!("expected Tuple"),
        }
    }

    #[test]
    fn test_roundtrip_struct() {
        let mut fields = BTreeMap::new();
        fields.insert("x".to_string(), Value::Int(1));
        fields.insert("y".to_string(), Value::Float(2.0));
        let orig = Value::Struct {
            name: "Point".to_string(),
            fields,
        };
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Struct { name, fields } => {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
                assert!(matches!(fields.get("x"), Some(Value::Int(1))));
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_roundtrip_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let orig = Value::Tensor(t);
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            _ => panic!("expected Tensor"),
        }
    }

    #[test]
    fn test_roundtrip_enum() {
        let orig = Value::Enum {
            enum_name: "Option".to_string(),
            variant: "Some".to_string(),
            fields: vec![Value::Int(42)],
        };
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Enum {
                enum_name,
                variant,
                fields,
            } => {
                assert_eq!(enum_name, "Option");
                assert_eq!(variant, "Some");
                assert_eq!(fields.len(), 1);
            }
            _ => panic!("expected Enum"),
        }
    }

    #[test]
    fn test_roundtrip_u8() {
        let v = roundtrip(&Value::U8(255));
        assert!(matches!(v, Value::U8(255)));
    }

    #[test]
    fn test_roundtrip_bytes() {
        let orig = Value::Bytes(Rc::new(RefCell::new(vec![0xDE, 0xAD, 0xBE, 0xEF])));
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Bytes(b) => assert_eq!(*b.borrow(), vec![0xDE, 0xAD, 0xBE, 0xEF]),
            _ => panic!("expected Bytes"),
        }
    }

    #[test]
    fn test_roundtrip_bf16() {
        let orig = Value::Bf16(Bf16(0x4000));
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Bf16(v) => assert_eq!(v.0, 0x4000),
            _ => panic!("expected Bf16"),
        }
    }

    #[test]
    fn test_roundtrip_complex() {
        let orig = Value::Complex(ComplexF64 { re: 1.0, im: -2.0 });
        let decoded = roundtrip(&orig);
        match decoded {
            Value::Complex(z) => {
                assert_eq!(z.re, 1.0);
                assert_eq!(z.im, -2.0);
            }
            _ => panic!("expected Complex"),
        }
    }

    #[test]
    fn test_invalid_tag() {
        let data = vec![0xFF];
        let result = snap_decode(&data);
        assert!(matches!(result, Err(SnapError::InvalidTag(0xFF))));
    }

    #[test]
    fn test_unexpected_eof() {
        let data = vec![TAG_INT, 0x01]; // Only 2 bytes, need 9
        let result = snap_decode(&data);
        assert!(matches!(result, Err(SnapError::UnexpectedEof)));
    }

    #[test]
    fn test_empty_input() {
        let result = snap_decode(&[]);
        assert!(matches!(result, Err(SnapError::UnexpectedEof)));
    }
}
