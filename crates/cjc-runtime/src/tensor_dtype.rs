//! Typed tensor infrastructure: DType enum and TypedStorage.
//!
//! This module provides the byte-first storage layer for multi-typed tensors.
//! The core idea: raw bytes are the primary representation, typed views are
//! computed on demand. This enables zero-copy serialization (snap), SIMD-friendly
//! aligned buffers, and memory-efficient storage for non-f64 types.
//!
//! ## Byte-First Philosophy
//!
//! - `TypedStorage` stores raw `Vec<u8>` + a `DType` tag
//! - Typed access via `as_f64()`, `as_i64()`, etc. reinterprets bytes in-place
//! - Serialization = memcpy the byte buffer (no conversion)
//! - COW semantics via `Rc<RefCell<Vec<u8>>>` (same pattern as Buffer<T>)

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use crate::accumulator::binned_sum_f64;
use crate::complex::ComplexF64;
use crate::error::RuntimeError;
use crate::value::Bf16;

// ---------------------------------------------------------------------------
// DType — element type tag for typed tensors
// ---------------------------------------------------------------------------

/// Element type for typed tensor storage.
///
/// Each variant determines how the raw byte buffer is interpreted.
/// Byte widths are fixed and platform-independent (little-endian canonical).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 64-bit IEEE 754 float (8 bytes per element)
    F64,
    /// 32-bit IEEE 754 float (4 bytes per element)
    F32,
    /// 64-bit signed integer (8 bytes per element)
    I64,
    /// 32-bit signed integer (4 bytes per element)
    I32,
    /// 8-bit unsigned integer (1 byte per element)
    U8,
    /// Boolean (1 byte per element; 0x00 = false, 0x01 = true)
    /// Note: NOT packed bits — 1 byte per bool for simplicity and alignment.
    /// Packed-bit BoolTensor can be a future optimization.
    Bool,
    /// Brain float 16-bit (2 bytes per element)
    Bf16,
    /// IEEE 754 half-precision float (2 bytes per element)
    F16,
    /// Complex f64 pair (16 bytes per element: 8 re + 8 im)
    Complex,
}

impl DType {
    /// Bytes per element for this dtype.
    pub fn byte_width(&self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::U8 | DType::Bool => 1,
            DType::Complex => 16,
        }
    }

    /// Human-readable name for display and error messages.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::U8 => "u8",
            DType::Bool => "bool",
            DType::Bf16 => "bf16",
            DType::F16 => "f16",
            DType::Complex => "complex",
        }
    }

    /// Whether this dtype represents a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F64 | DType::F32 | DType::Bf16 | DType::F16)
    }

    /// Whether this dtype represents an integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::U8)
    }

    /// Whether this dtype supports arithmetic operations.
    pub fn is_numeric(&self) -> bool {
        !matches!(self, DType::Bool)
    }

    /// Tag byte used in snap serialization.
    pub fn snap_tag(&self) -> u8 {
        match self {
            DType::F64 => 0,
            DType::F32 => 1,
            DType::I64 => 2,
            DType::I32 => 3,
            DType::U8 => 4,
            DType::Bool => 5,
            DType::Bf16 => 6,
            DType::F16 => 7,
            DType::Complex => 8,
        }
    }

    /// Reconstruct DType from a snap tag byte.
    pub fn from_snap_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(DType::F64),
            1 => Ok(DType::F32),
            2 => Ok(DType::I64),
            3 => Ok(DType::I32),
            4 => Ok(DType::U8),
            5 => Ok(DType::Bool),
            6 => Ok(DType::Bf16),
            7 => Ok(DType::F16),
            8 => Ok(DType::Complex),
            _ => Err(format!("unknown dtype snap tag: {tag}")),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// TypedStorage — byte-first tensor backing store
// ---------------------------------------------------------------------------

/// Byte-first tensor storage with COW (copy-on-write) semantics.
///
/// The raw byte buffer is the canonical representation. Typed views are
/// created on demand via `as_f64()`, `as_i64()`, etc. This enables:
///
/// - Zero-copy snap serialization (bytes ARE the encoded form)
/// - Memory-mapped I/O (load bytes, interpret in-place)
/// - SIMD-friendly aligned byte buffers
/// - Memory-efficient storage (f32 = 50% of f64, bool = 12.5%)
///
/// COW: Cloning a `TypedStorage` increments the Rc refcount (zero copy).
/// Mutation triggers a deep copy if shared.
#[derive(Debug)]
pub struct TypedStorage {
    /// Raw byte buffer. Alignment: elements are naturally aligned within
    /// the Vec<u8> because Vec guarantees pointer alignment ≥ 8 bytes
    /// (on 64-bit platforms). For SIMD (16-byte alignment), use
    /// AlignedByteSlice for hot paths.
    bytes: Rc<RefCell<Vec<u8>>>,
    /// Element type determines byte interpretation.
    dtype: DType,
    /// Number of logical elements (NOT bytes).
    len: usize,
}

impl TypedStorage {
    // -- Construction -------------------------------------------------------

    /// Create storage filled with zeros.
    pub fn zeros(dtype: DType, len: usize) -> Self {
        let nbytes = len * dtype.byte_width();
        TypedStorage {
            bytes: Rc::new(RefCell::new(vec![0u8; nbytes])),
            dtype,
            len,
        }
    }

    /// Create storage from an existing byte buffer.
    /// Returns error if byte length doesn't match dtype × element count.
    pub fn from_bytes(bytes: Vec<u8>, dtype: DType, len: usize) -> Result<Self, String> {
        let expected = len * dtype.byte_width();
        if bytes.len() != expected {
            return Err(format!(
                "TypedStorage::from_bytes: expected {} bytes ({} × {} elements), got {}",
                expected,
                dtype.byte_width(),
                len,
                bytes.len()
            ));
        }
        Ok(TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype,
            len,
        })
    }

    /// Create f64 storage from a Vec<f64>.
    pub fn from_f64_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        let bytes = f64_vec_to_bytes(data);
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::F64,
            len,
        }
    }

    /// Create i64 storage from a Vec<i64>.
    pub fn from_i64_vec(data: Vec<i64>) -> Self {
        let len = data.len();
        let bytes = i64_vec_to_bytes(data);
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::I64,
            len,
        }
    }

    /// Create f32 storage from a Vec<f32>.
    pub fn from_f32_vec(data: Vec<f32>) -> Self {
        let len = data.len();
        let bytes = f32_vec_to_bytes(data);
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::F32,
            len,
        }
    }

    /// Create i32 storage from a Vec<i32>.
    pub fn from_i32_vec(data: Vec<i32>) -> Self {
        let len = data.len();
        let bytes = i32_vec_to_bytes(data);
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::I32,
            len,
        }
    }

    /// Create u8 storage from a Vec<u8>.
    pub fn from_u8_vec(data: Vec<u8>) -> Self {
        let len = data.len();
        TypedStorage {
            bytes: Rc::new(RefCell::new(data)),
            dtype: DType::U8,
            len,
        }
    }

    /// Create bool storage from a Vec<bool>.
    pub fn from_bool_vec(data: Vec<bool>) -> Self {
        let len = data.len();
        let bytes: Vec<u8> = data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::Bool,
            len,
        }
    }

    /// Create complex storage from a Vec<ComplexF64>.
    pub fn from_complex_vec(data: Vec<ComplexF64>) -> Self {
        let len = data.len();
        let mut bytes = Vec::with_capacity(len * 16);
        for c in &data {
            bytes.extend_from_slice(&c.re.to_le_bytes());
            bytes.extend_from_slice(&c.im.to_le_bytes());
        }
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::Complex,
            len,
        }
    }

    /// Create bf16 storage from a Vec<Bf16>.
    pub fn from_bf16_vec(data: Vec<Bf16>) -> Self {
        let len = data.len();
        let mut bytes = Vec::with_capacity(len * 2);
        for v in &data {
            bytes.extend_from_slice(&v.0.to_le_bytes());
        }
        TypedStorage {
            bytes: Rc::new(RefCell::new(bytes)),
            dtype: DType::Bf16,
            len,
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of logical elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total byte count of the raw buffer.
    pub fn byte_len(&self) -> usize {
        self.len * self.dtype.byte_width()
    }

    /// Number of live references to the underlying byte buffer.
    pub fn refcount(&self) -> usize {
        Rc::strong_count(&self.bytes)
    }

    /// Borrow the raw byte buffer.
    pub fn borrow_bytes(&self) -> Ref<Vec<u8>> {
        self.bytes.borrow()
    }

    /// Clone the raw bytes out (for serialization, etc.).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bytes.borrow().clone()
    }

    // -- Typed views (read-only) -------------------------------------------

    /// Interpret bytes as f64 slice. Panics if dtype != F64.
    pub fn as_f64_vec(&self) -> Vec<f64> {
        assert_eq!(self.dtype, DType::F64, "as_f64_vec: dtype is {}", self.dtype);
        bytes_to_f64_vec(&self.bytes.borrow())
    }

    /// Interpret bytes as i64 slice. Panics if dtype != I64.
    pub fn as_i64_vec(&self) -> Vec<i64> {
        assert_eq!(self.dtype, DType::I64, "as_i64_vec: dtype is {}", self.dtype);
        bytes_to_i64_vec(&self.bytes.borrow())
    }

    /// Interpret bytes as f32 slice. Panics if dtype != F32.
    pub fn as_f32_vec(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DType::F32, "as_f32_vec: dtype is {}", self.dtype);
        bytes_to_f32_vec(&self.bytes.borrow())
    }

    /// Interpret bytes as i32 slice. Panics if dtype != I32.
    pub fn as_i32_vec(&self) -> Vec<i32> {
        assert_eq!(self.dtype, DType::I32, "as_i32_vec: dtype is {}", self.dtype);
        bytes_to_i32_vec(&self.bytes.borrow())
    }

    /// Interpret bytes as bool slice. Panics if dtype != Bool.
    pub fn as_bool_vec(&self) -> Vec<bool> {
        assert_eq!(self.dtype, DType::Bool, "as_bool_vec: dtype is {}", self.dtype);
        self.bytes.borrow().iter().map(|&b| b != 0).collect()
    }

    /// Interpret bytes as u8 slice (trivial — bytes ARE u8). Panics if dtype != U8.
    pub fn as_u8_vec(&self) -> Vec<u8> {
        assert_eq!(self.dtype, DType::U8, "as_u8_vec: dtype is {}", self.dtype);
        self.bytes.borrow().clone()
    }

    /// Interpret bytes as ComplexF64 slice. Panics if dtype != Complex.
    pub fn as_complex_vec(&self) -> Vec<ComplexF64> {
        assert_eq!(self.dtype, DType::Complex, "as_complex_vec: dtype is {}", self.dtype);
        let raw = self.bytes.borrow();
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let off = i * 16;
            let re = f64::from_le_bytes(raw[off..off + 8].try_into().unwrap());
            let im = f64::from_le_bytes(raw[off + 8..off + 16].try_into().unwrap());
            result.push(ComplexF64 { re, im });
        }
        result
    }

    /// Interpret bytes as Bf16 slice. Panics if dtype != Bf16.
    pub fn as_bf16_vec(&self) -> Vec<Bf16> {
        assert_eq!(self.dtype, DType::Bf16, "as_bf16_vec: dtype is {}", self.dtype);
        let raw = self.bytes.borrow();
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let off = i * 2;
            let bits = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
            result.push(Bf16(bits));
        }
        result
    }

    /// Convert any numeric dtype to f64 vec (for operations that need f64).
    /// Bool: false→0.0, true→1.0.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self.dtype {
            DType::F64 => self.as_f64_vec(),
            DType::F32 => self.as_f32_vec().into_iter().map(|v| v as f64).collect(),
            DType::I64 => self.as_i64_vec().into_iter().map(|v| v as f64).collect(),
            DType::I32 => self.as_i32_vec().into_iter().map(|v| v as f64).collect(),
            DType::U8 => self.as_u8_vec().into_iter().map(|v| v as f64).collect(),
            DType::Bool => self.as_bool_vec().into_iter().map(|v| if v { 1.0 } else { 0.0 }).collect(),
            DType::Bf16 => self.as_bf16_vec().into_iter().map(|v| v.to_f32() as f64).collect(),
            DType::F16 => {
                let raw = self.bytes.borrow();
                let mut result = Vec::with_capacity(self.len);
                for i in 0..self.len {
                    let off = i * 2;
                    let bits = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
                    result.push(crate::f16::F16(bits).to_f64());
                }
                result
            }
            DType::Complex => {
                // Return real parts only for scalar operations
                self.as_complex_vec().into_iter().map(|c| c.re).collect()
            }
        }
    }

    // -- Element access -----------------------------------------------------

    /// Get a single f64 value at index. Works for any numeric dtype (converts).
    pub fn get_as_f64(&self, idx: usize) -> Result<f64, RuntimeError> {
        if idx >= self.len {
            return Err(RuntimeError::IndexOutOfBounds { index: idx, length: self.len });
        }
        let raw = self.bytes.borrow();
        let bw = self.dtype.byte_width();
        let off = idx * bw;
        Ok(match self.dtype {
            DType::F64 => f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()),
            DType::F32 => f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as f64,
            DType::I64 => i64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) as f64,
            DType::I32 => i32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as f64,
            DType::U8 => raw[off] as f64,
            DType::Bool => if raw[off] != 0 { 1.0 } else { 0.0 },
            DType::Bf16 => {
                let bits = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
                Bf16(bits).to_f32() as f64
            }
            DType::F16 => {
                let bits = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
                crate::f16::F16(bits).to_f64()
            }
            DType::Complex => {
                f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) // real part
            }
        })
    }

    /// Set a single f64 value at index. Converts to storage dtype.
    /// Triggers COW if shared.
    pub fn set_from_f64(&mut self, idx: usize, val: f64) -> Result<(), RuntimeError> {
        if idx >= self.len {
            return Err(RuntimeError::IndexOutOfBounds { index: idx, length: self.len });
        }
        self.make_unique();
        let bw = self.dtype.byte_width();
        let off = idx * bw;
        let mut raw = self.bytes.borrow_mut();
        match self.dtype {
            DType::F64 => raw[off..off + 8].copy_from_slice(&val.to_le_bytes()),
            DType::F32 => raw[off..off + 4].copy_from_slice(&(val as f32).to_le_bytes()),
            DType::I64 => raw[off..off + 8].copy_from_slice(&(val as i64).to_le_bytes()),
            DType::I32 => raw[off..off + 4].copy_from_slice(&(val as i32).to_le_bytes()),
            DType::U8 => raw[off] = val as u8,
            DType::Bool => raw[off] = if val != 0.0 { 1 } else { 0 },
            DType::Bf16 => {
                let bits = Bf16::from_f32(val as f32).0;
                raw[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            }
            DType::F16 => {
                let bits = crate::f16::F16::from_f64(val).0;
                raw[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            }
            DType::Complex => {
                raw[off..off + 8].copy_from_slice(&val.to_le_bytes());
                raw[off + 8..off + 16].copy_from_slice(&0.0f64.to_le_bytes());
            }
        }
        Ok(())
    }

    // -- COW ----------------------------------------------------------------

    /// Ensure exclusive ownership. If shared, deep-copy the byte buffer.
    pub fn make_unique(&mut self) {
        if Rc::strong_count(&self.bytes) > 1 {
            let data = self.bytes.borrow().clone();
            self.bytes = Rc::new(RefCell::new(data));
        }
    }

    /// Force a deep copy, returning a new TypedStorage that does not share.
    pub fn deep_clone(&self) -> TypedStorage {
        TypedStorage {
            bytes: Rc::new(RefCell::new(self.bytes.borrow().clone())),
            dtype: self.dtype,
            len: self.len,
        }
    }

    // -- Reductions ---------------------------------------------------------

    /// Sum all elements as f64. Uses BinnedAccumulator for float types.
    pub fn sum_f64(&self) -> f64 {
        let data = self.to_f64_vec();
        if self.dtype.is_float() || self.dtype == DType::Complex {
            binned_sum_f64(&data)
        } else {
            // Integer types: exact sum (no accumulator needed)
            data.iter().sum()
        }
    }

    /// Mean of all elements as f64.
    pub fn mean_f64(&self) -> f64 {
        if self.len == 0 {
            return f64::NAN;
        }
        self.sum_f64() / self.len as f64
    }

    // -- Type casting -------------------------------------------------------

    /// Cast to a different dtype. Returns a new TypedStorage.
    pub fn cast(&self, target: DType) -> TypedStorage {
        if self.dtype == target {
            return self.deep_clone();
        }
        let f64_data = self.to_f64_vec();
        match target {
            DType::F64 => TypedStorage::from_f64_vec(f64_data),
            DType::F32 => TypedStorage::from_f32_vec(f64_data.into_iter().map(|v| v as f32).collect()),
            DType::I64 => TypedStorage::from_i64_vec(f64_data.into_iter().map(|v| v as i64).collect()),
            DType::I32 => TypedStorage::from_i32_vec(f64_data.into_iter().map(|v| v as i32).collect()),
            DType::U8 => TypedStorage::from_u8_vec(f64_data.into_iter().map(|v| v as u8).collect()),
            DType::Bool => TypedStorage::from_bool_vec(f64_data.into_iter().map(|v| v != 0.0).collect()),
            DType::Bf16 => TypedStorage::from_bf16_vec(f64_data.into_iter().map(|v| Bf16::from_f32(v as f32)).collect()),
            DType::F16 => {
                let mut bytes = Vec::with_capacity(f64_data.len() * 2);
                for v in &f64_data {
                    let bits = crate::f16::F16::from_f64(*v).0;
                    bytes.extend_from_slice(&bits.to_le_bytes());
                }
                TypedStorage {
                    bytes: Rc::new(RefCell::new(bytes)),
                    dtype: DType::F16,
                    len: f64_data.len(),
                }
            }
            DType::Complex => TypedStorage::from_complex_vec(
                f64_data.into_iter().map(|v| ComplexF64 { re: v, im: 0.0 }).collect()
            ),
        }
    }
}

impl Clone for TypedStorage {
    /// Cloning increments refcount — zero copy (COW).
    fn clone(&self) -> Self {
        TypedStorage {
            bytes: Rc::clone(&self.bytes),
            dtype: self.dtype,
            len: self.len,
        }
    }
}

// ---------------------------------------------------------------------------
// Byte conversion helpers (little-endian, deterministic)
// ---------------------------------------------------------------------------

fn f64_vec_to_bytes(data: Vec<f64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
    let n = bytes.len() / 8;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 8;
        result.push(f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
    }
    result
}

fn i64_vec_to_bytes(data: Vec<i64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_i64_vec(bytes: &[u8]) -> Vec<i64> {
    let n = bytes.len() / 8;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 8;
        result.push(i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
    }
    result
}

fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        result.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
    }
    result
}

fn i32_vec_to_bytes(data: Vec<i32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_i32_vec(bytes: &[u8]) -> Vec<i32> {
    let n = bytes.len() / 4;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        result.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_byte_width() {
        assert_eq!(DType::F64.byte_width(), 8);
        assert_eq!(DType::F32.byte_width(), 4);
        assert_eq!(DType::I64.byte_width(), 8);
        assert_eq!(DType::I32.byte_width(), 4);
        assert_eq!(DType::U8.byte_width(), 1);
        assert_eq!(DType::Bool.byte_width(), 1);
        assert_eq!(DType::Bf16.byte_width(), 2);
        assert_eq!(DType::F16.byte_width(), 2);
        assert_eq!(DType::Complex.byte_width(), 16);
    }

    #[test]
    fn test_dtype_snap_roundtrip() {
        for dt in &[DType::F64, DType::F32, DType::I64, DType::I32,
                    DType::U8, DType::Bool, DType::Bf16, DType::F16, DType::Complex] {
            assert_eq!(DType::from_snap_tag(dt.snap_tag()).unwrap(), *dt);
        }
    }

    #[test]
    fn test_f64_storage_roundtrip() {
        let data = vec![1.5, -2.3, 0.0, f64::INFINITY, f64::NEG_INFINITY];
        let storage = TypedStorage::from_f64_vec(data.clone());
        assert_eq!(storage.dtype(), DType::F64);
        assert_eq!(storage.len(), 5);
        assert_eq!(storage.as_f64_vec(), data);
    }

    #[test]
    fn test_i64_storage_roundtrip() {
        let data = vec![1i64, -2, 0, i64::MAX, i64::MIN];
        let storage = TypedStorage::from_i64_vec(data.clone());
        assert_eq!(storage.dtype(), DType::I64);
        assert_eq!(storage.as_i64_vec(), data);
    }

    #[test]
    fn test_f32_storage_roundtrip() {
        let data = vec![1.0f32, -2.5, 0.0, 3.14];
        let storage = TypedStorage::from_f32_vec(data.clone());
        assert_eq!(storage.dtype(), DType::F32);
        assert_eq!(storage.as_f32_vec(), data);
    }

    #[test]
    fn test_i32_storage_roundtrip() {
        let data = vec![42i32, -1, 0, i32::MAX];
        let storage = TypedStorage::from_i32_vec(data.clone());
        assert_eq!(storage.as_i32_vec(), data);
    }

    #[test]
    fn test_u8_storage_roundtrip() {
        let data = vec![0u8, 127, 255];
        let storage = TypedStorage::from_u8_vec(data.clone());
        assert_eq!(storage.as_u8_vec(), data);
    }

    #[test]
    fn test_bool_storage_roundtrip() {
        let data = vec![true, false, true, true, false];
        let storage = TypedStorage::from_bool_vec(data.clone());
        assert_eq!(storage.as_bool_vec(), data);
    }

    #[test]
    fn test_complex_storage_roundtrip() {
        let data = vec![
            ComplexF64 { re: 1.0, im: 2.0 },
            ComplexF64 { re: -3.0, im: 0.5 },
        ];
        let storage = TypedStorage::from_complex_vec(data.clone());
        let back = storage.as_complex_vec();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].re, 1.0);
        assert_eq!(back[0].im, 2.0);
        assert_eq!(back[1].re, -3.0);
        assert_eq!(back[1].im, 0.5);
    }

    #[test]
    fn test_bf16_storage_roundtrip() {
        let data = vec![Bf16::from_f32(1.0), Bf16::from_f32(-0.5)];
        let storage = TypedStorage::from_bf16_vec(data.clone());
        let back = storage.as_bf16_vec();
        assert_eq!(back[0].to_f32(), 1.0);
        assert_eq!(back[1].to_f32(), -0.5);
    }

    #[test]
    fn test_cow_semantics() {
        let s1 = TypedStorage::from_f64_vec(vec![1.0, 2.0, 3.0]);
        let s2 = s1.clone();
        assert_eq!(s1.refcount(), 2);
        assert_eq!(s2.refcount(), 2);

        let s3 = s1.deep_clone();
        assert_eq!(s3.refcount(), 1);
        assert_eq!(s1.refcount(), 2); // s1 and s2 still share
    }

    #[test]
    fn test_cow_mutation() {
        let s1 = TypedStorage::from_f64_vec(vec![1.0, 2.0, 3.0]);
        let mut s2 = s1.clone();
        assert_eq!(s1.refcount(), 2);

        s2.set_from_f64(0, 99.0).unwrap();
        assert_eq!(s1.refcount(), 1); // s1 no longer shared
        assert_eq!(s2.refcount(), 1);
        assert_eq!(s1.as_f64_vec()[0], 1.0); // unchanged
        assert_eq!(s2.as_f64_vec()[0], 99.0); // mutated copy
    }

    #[test]
    fn test_get_set_f64() {
        let mut storage = TypedStorage::from_f64_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(storage.get_as_f64(0).unwrap(), 10.0);
        assert_eq!(storage.get_as_f64(2).unwrap(), 30.0);
        assert!(storage.get_as_f64(3).is_err());

        storage.set_from_f64(1, 99.0).unwrap();
        assert_eq!(storage.get_as_f64(1).unwrap(), 99.0);
    }

    #[test]
    fn test_get_set_i64() {
        let mut storage = TypedStorage::from_i64_vec(vec![10, 20, 30]);
        assert_eq!(storage.get_as_f64(0).unwrap(), 10.0);
        storage.set_from_f64(1, 42.0).unwrap();
        assert_eq!(storage.as_i64_vec()[1], 42);
    }

    #[test]
    fn test_to_f64_vec_conversion() {
        let storage = TypedStorage::from_i32_vec(vec![1, 2, 3]);
        assert_eq!(storage.to_f64_vec(), vec![1.0, 2.0, 3.0]);

        let storage = TypedStorage::from_bool_vec(vec![true, false, true]);
        assert_eq!(storage.to_f64_vec(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sum_f64() {
        let storage = TypedStorage::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!((storage.sum_f64() - 10.0).abs() < 1e-12);

        let storage = TypedStorage::from_i64_vec(vec![1, 2, 3, 4]);
        assert!((storage.sum_f64() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_cast_f64_to_i64() {
        let s = TypedStorage::from_f64_vec(vec![1.5, -2.7, 3.0]);
        let c = s.cast(DType::I64);
        assert_eq!(c.dtype(), DType::I64);
        assert_eq!(c.as_i64_vec(), vec![1, -2, 3]);
    }

    #[test]
    fn test_cast_i64_to_f32() {
        let s = TypedStorage::from_i64_vec(vec![1, 2, 3]);
        let c = s.cast(DType::F32);
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.as_f32_vec(), vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_zeros_all_dtypes() {
        for dt in &[DType::F64, DType::F32, DType::I64, DType::I32,
                    DType::U8, DType::Bool, DType::Bf16, DType::F16, DType::Complex] {
            let s = TypedStorage::zeros(*dt, 10);
            assert_eq!(s.len(), 10);
            assert_eq!(s.byte_len(), 10 * dt.byte_width());
            // All zero bytes → all zero values
            assert!((s.get_as_f64(0).unwrap()).abs() < 1e-15 || s.get_as_f64(0).unwrap() == 0.0);
        }
    }

    #[test]
    fn test_byte_determinism() {
        // Same data → identical bytes (deterministic encoding)
        let s1 = TypedStorage::from_f64_vec(vec![1.0, 2.0, 3.0]);
        let s2 = TypedStorage::from_f64_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(s1.to_bytes(), s2.to_bytes());

        let s3 = TypedStorage::from_i64_vec(vec![42, -1, 0]);
        let s4 = TypedStorage::from_i64_vec(vec![42, -1, 0]);
        assert_eq!(s3.to_bytes(), s4.to_bytes());
    }

    #[test]
    fn test_from_bytes_roundtrip() {
        let original = TypedStorage::from_f64_vec(vec![1.5, -2.3, 0.0]);
        let bytes = original.to_bytes();
        let restored = TypedStorage::from_bytes(bytes, DType::F64, 3).unwrap();
        assert_eq!(original.as_f64_vec(), restored.as_f64_vec());
    }

    #[test]
    fn test_from_bytes_size_mismatch() {
        assert!(TypedStorage::from_bytes(vec![0u8; 10], DType::F64, 2).is_err());
    }
}
