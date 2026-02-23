use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

use crate::aligned_pool::AlignedByteSlice;
use crate::complex;
use crate::det_map::DetMap;
use crate::gc::GcRef;
use crate::paged_kv::PagedKvCache;
use crate::scratchpad::Scratchpad;
use crate::sparse::SparseCsr;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 7. Value enum for the interpreter
// ---------------------------------------------------------------------------

/// bf16 brain-float: u16-backed storage, deterministic f32 conversions.
/// Arithmetic is performed by widening to f32, computing, then narrowing back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// Convert f32 to bf16 by truncating lower 16 mantissa bits.
    pub fn from_f32(v: f32) -> Self {
        Bf16((v.to_bits() >> 16) as u16)
    }

    /// Convert bf16 to f32 by shifting left 16 bits.
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    pub fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }

    pub fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }

    pub fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }

    pub fn neg(self) -> Self {
        Self::from_f32(-self.to_f32())
    }
}

impl fmt::Display for Bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// A CJC function value — either a named function or a closure.
#[derive(Debug, Clone)]
pub struct FnValue {
    /// Function name (or `"<lambda>"` for anonymous closures).
    pub name: String,
    /// Number of parameters.
    pub arity: usize,
    /// Opaque identifier used by the interpreter to locate the function body.
    pub body_id: usize,
}

/// The universal value type for the CJC interpreter.
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(Rc<String>),
    /// Owning byte buffer.
    Bytes(Rc<RefCell<Vec<u8>>>),
    /// Non-owning byte slice view. In the interpreter this is an owning
    /// snapshot (Vec<u8>) since we can't borrow across eval boundaries.
    /// The compiler/MIR path can use true zero-copy slices.
    ByteSlice(Rc<Vec<u8>>),
    /// Validated UTF-8 string view (same representation as ByteSlice but
    /// guaranteed valid UTF-8).
    StrView(Rc<Vec<u8>>),
    /// Single byte value (u8).
    U8(u8),
    Tensor(Tensor),
    SparseTensor(SparseCsr),
    Map(Rc<RefCell<DetMap>>),
    /// Copy-on-write array. `Rc` provides O(1) clone; `Rc::make_mut()`
    /// triggers a deep copy only when the array is mutated and shared.
    Array(Rc<Vec<Value>>),
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    /// Copy-on-write tuple. Same COW semantics as Array.
    Tuple(Rc<Vec<Value>>),
    ClassRef(GcRef),
    Fn(FnValue),
    /// A closure: a function name + captured environment values.
    Closure {
        fn_name: String,
        env: Vec<Value>,
        /// Arity of the *original* lambda params (not including captures).
        arity: usize,
    },
    /// Enum variant value: `Some(42)`, `None`, `Ok(v)`, `Err(e)`
    Enum {
        enum_name: String,
        variant: String,
        fields: Vec<Value>,
    },
    /// Compiled regex pattern: (pattern, flags)
    Regex { pattern: String, flags: String },
    /// bf16 brain-float: u16-backed, deterministic f32 conversions
    Bf16(Bf16),
    /// f16 half-precision: u16-backed IEEE 754 binary16
    F16(crate::f16::F16),
    /// Complex f64: deterministic fixed-sequence arithmetic
    Complex(complex::ComplexF64),
    /// Pre-allocated KV-cache scratchpad for zero-allocation inference.
    Scratchpad(Rc<RefCell<Scratchpad>>),
    /// Block-paged KV-cache (vLLM-style).
    PagedKvCache(Rc<RefCell<PagedKvCache>>),
    /// Aligned byte slice with 16-byte alignment guarantee.
    AlignedBytes(AlignedByteSlice),
    Void,
}

impl Value {
    /// Returns a human-readable type name for error messages.
    pub fn type_name(&self) -> &str {
        match self {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::String(_) => "String",
            Value::Bytes(_) => "Bytes",
            Value::ByteSlice(_) => "ByteSlice",
            Value::StrView(_) => "StrView",
            Value::U8(_) => "u8",
            Value::Tensor(_) => "Tensor",
            Value::SparseTensor(_) => "SparseTensor",
            Value::Map(_) => "Map",
            Value::Array(_) => "Array",
            Value::Tuple(_) => "Tuple",
            Value::Struct { .. } => "Struct",
            Value::Enum { .. } => "Enum",
            Value::ClassRef(_) => "ClassRef",
            Value::Fn(_) => "Fn",
            Value::Closure { .. } => "Closure",
            Value::Regex { .. } => "Regex",
            Value::Bf16(_) => "Bf16",
            Value::F16(_) => "F16",
            Value::Complex(_) => "Complex",
            Value::Scratchpad(_) => "Scratchpad",
            Value::PagedKvCache(_) => "PagedKvCache",
            Value::AlignedBytes(_) => "AlignedBytes",
            Value::Void => "Void",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "{v}"),
            Value::Bytes(b) => {
                let b = b.borrow();
                write!(f, "Bytes([")?;
                for (i, byte) in b.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{byte}")?;
                }
                write!(f, "])")
            }
            Value::ByteSlice(b) => {
                // Try to display as UTF-8, fall back to hex
                match std::str::from_utf8(b) {
                    Ok(s) => write!(f, "b\"{s}\""),
                    Err(_) => {
                        write!(f, "b\"")?;
                        for &byte in b.iter() {
                            if byte.is_ascii_graphic() || byte == b' ' {
                                write!(f, "{}", byte as char)?;
                            } else {
                                write!(f, "\\x{byte:02x}")?;
                            }
                        }
                        write!(f, "\"")
                    }
                }
            }
            Value::StrView(b) => {
                // StrView is guaranteed valid UTF-8
                let s = std::str::from_utf8(b).unwrap_or("<invalid utf8>");
                write!(f, "{s}")
            }
            Value::U8(v) => write!(f, "{v}"),
            Value::Tensor(t) => write!(f, "{t}"),
            Value::SparseTensor(s) => write!(f, "SparseTensor({}x{}, nnz={})", s.nrows, s.ncols, s.nnz()),
            Value::Map(m) => {
                let m = m.borrow();
                write!(f, "Map({{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}})")
            }
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Tuple(elems) => {
                write!(f, "(")?;
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{name} {{ ")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, " }}")
            }
            Value::Enum {
                enum_name: _,
                variant,
                fields,
            } => {
                write!(f, "{variant}")?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{v}")?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Regex { pattern, flags } => {
                write!(f, "/{pattern}/")?;
                if !flags.is_empty() {
                    write!(f, "{flags}")?;
                }
                Ok(())
            }
            Value::Bf16(v) => write!(f, "{}", v.to_f32()),
            Value::F16(v) => write!(f, "{}", v.to_f64()),
            Value::Complex(z) => write!(f, "{z}"),
            Value::ClassRef(r) => write!(f, "<object@{}>", r.index),
            Value::Fn(fv) => write!(f, "<fn {}({})>", fv.name, fv.arity),
            Value::Closure {
                fn_name, arity, ..
            } => write!(f, "<closure {}({})>", fn_name, arity),
            Value::Scratchpad(s) => write!(f, "{}", s.borrow()),
            Value::PagedKvCache(c) => write!(f, "{}", c.borrow()),
            Value::AlignedBytes(a) => write!(f, "{}", a),
            Value::Void => write!(f, "void"),
        }
    }
}

