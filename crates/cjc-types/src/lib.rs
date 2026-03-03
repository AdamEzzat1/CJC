use std::collections::HashMap;
use std::fmt;

// ── Core Type Representation ────────────────────────────────────

/// Unique type identifier.
pub type TypeId = usize;

/// Substitution map: TypeVarId -> concrete Type.
pub type TypeSubst = HashMap<TypeVarId, Type>;

/// Substitution map for symbolic shape variables.
pub type ShapeSubst = HashMap<String, usize>;

/// The CJC type system representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Primitive types.
    I32,
    I64,
    U8,
    F32,
    F64,
    Bool,
    Str,
    Void,

    /// Owning byte buffer.
    Bytes,

    /// Non-owning byte slice view (zero-copy).
    ByteSlice,

    /// Validated UTF-8 string view (zero-copy).
    StrView,

    /// Tensor type with element type and optional shape.
    Tensor {
        elem: Box<Type>,
        shape: Option<Vec<ShapeDim>>,
    },

    /// Buffer type.
    Buffer {
        elem: Box<Type>,
    },

    /// Array type with element type and length.
    Array {
        elem: Box<Type>,
        len: usize,
    },

    /// Tuple type.
    Tuple(Vec<Type>),

    /// User-defined struct (value type).
    Struct(StructType),

    /// User-defined class (GC reference type).
    Class(ClassType),

    /// User-defined enum / ADT (value type, stack-allocatable).
    Enum(EnumType),

    /// Brain float 16-bit type.
    Bf16,

    /// IEEE 754 half-precision 16-bit float type.
    F16,

    /// Range type: `start..end` (half-open interval).
    ///
    /// **Status:** Type-system placeholder. No `Value::Range` exists yet;
    /// the runtime represents ranges via for-loop desugaring. This type is
    /// reserved for future range-literal support.
    Range {
        elem: Box<Type>,
    },

    /// Generic slice (non-owning view into an array).
    ///
    /// **Status:** Type-system placeholder. No `Value::Slice` exists yet;
    /// the runtime uses `ByteSlice` / `StrView` for concrete slice-like
    /// values. This type is reserved for future generic slice support.
    Slice {
        elem: Box<Type>,
    },

    /// Complex number type (f64 real + f64 imaginary).
    Complex,

    /// Compiled regex pattern type.
    Regex,

    /// Function type.
    Fn {
        params: Vec<Type>,
        ret: Box<Type>,
    },

    /// Map type with key and value types.
    Map {
        key: Box<Type>,
        value: Box<Type>,
    },

    /// Sparse tensor type.
    SparseTensor {
        elem: Box<Type>,
    },

    /// Tidy data view (lazy bitmask + projection over a DataFrame).
    TidyView,

    /// Grouped tidy data view (TidyView partitioned by key columns).
    GroupedTidyView,

    /// Type variable (for generics).
    Var(TypeVarId),

    /// A named type that hasn't been resolved yet.
    Unresolved(String),

    /// Error type (for error recovery — unifies with everything).
    Error,
}

impl Type {
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::I32 | Type::I64 | Type::U8 | Type::F32 | Type::F64 | Type::Bf16 | Type::F16 | Type::Complex)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64 | Type::Bf16 | Type::F16)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Type::I32 | Type::I64 | Type::U8)
    }

    pub fn is_value_type(&self) -> bool {
        matches!(
            self,
            Type::I32
                | Type::I64
                | Type::U8
                | Type::F32
                | Type::F64
                | Type::Bf16
                | Type::F16
                | Type::Bool
                | Type::Str
                | Type::Void
                | Type::Bytes
                | Type::ByteSlice
                | Type::StrView
                | Type::Tensor { .. }
                | Type::Buffer { .. }
                | Type::Array { .. }
                | Type::Tuple(_)
                | Type::Struct(_)
                | Type::Enum(_)
                | Type::Map { .. }
                | Type::SparseTensor { .. }
                | Type::Complex
                | Type::Regex
                | Type::Range { .. }
                | Type::Slice { .. }
        )
    }

    /// Returns true if this type is NoGC-safe (no hidden allocations).
    pub fn is_nogc_safe(&self) -> bool {
        matches!(
            self,
            Type::I32
                | Type::I64
                | Type::U8
                | Type::F32
                | Type::F64
                | Type::Bf16
                | Type::F16
                | Type::Bool
                | Type::Complex
                | Type::Void
                | Type::ByteSlice
                | Type::StrView
        )
    }

    pub fn is_gc_type(&self) -> bool {
        matches!(self, Type::Class(_))
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Type::Error)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::U8 => write!(f, "u8"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "String"),
            Type::Void => write!(f, "void"),
            Type::Bytes => write!(f, "Bytes"),
            Type::ByteSlice => write!(f, "ByteSlice"),
            Type::StrView => write!(f, "StrView"),
            Type::Tensor { elem, shape } => {
                write!(f, "Tensor<{}", elem)?;
                if let Some(dims) = shape {
                    write!(f, ", [")?;
                    for (i, dim) in dims.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", dim)?;
                    }
                    write!(f, "]")?;
                }
                write!(f, ">")
            }
            Type::Buffer { elem } => write!(f, "Buffer<{}>", elem),
            Type::Array { elem, len } => write!(f, "[{}; {}]", elem, len),
            Type::Tuple(elems) => {
                write!(f, "(")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, ")")
            }
            Type::Struct(s) => write!(f, "{}", s.name),
            Type::Class(c) => write!(f, "{}", c.name),
            Type::Enum(e) => write!(f, "{}", e.name),
            Type::Bf16 => write!(f, "bf16"),
            Type::F16 => write!(f, "f16"),
            Type::Range { elem } => write!(f, "Range<{}>", elem),
            Type::Slice { elem } => write!(f, "Slice<{}>", elem),
            Type::Complex => write!(f, "Complex"),
            Type::Regex => write!(f, "Regex"),
            Type::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Map { key, value } => write!(f, "Map<{}, {}>", key, value),
            Type::SparseTensor { elem } => write!(f, "SparseTensor<{}>", elem),
            Type::TidyView => write!(f, "TidyView"),
            Type::GroupedTidyView => write!(f, "GroupedTidyView"),
            Type::Var(id) => write!(f, "T{}", id.0),
            Type::Unresolved(name) => write!(f, "?{}", name),
            Type::Error => write!(f, "<error>"),
        }
    }
}

// ── Shape Dimensions ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeDim {
    Known(usize),
    Symbolic(String),
}

impl fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeDim::Known(n) => write!(f, "{}", n),
            ShapeDim::Symbolic(name) => write!(f, "{}", name),
        }
    }
}

// ── Type Unification ────────────────────────────────────────────

/// Unify two types, producing bindings for type variables.
/// Returns the unified (most specific) type, or an error message.
pub fn unify(a: &Type, b: &Type, subst: &mut TypeSubst) -> Result<Type, String> {
    // Error type unifies with everything (error recovery)
    if a.is_error() {
        return Ok(b.clone());
    }
    if b.is_error() {
        return Ok(a.clone());
    }

    match (a, b) {
        // Type variable on the left
        (Type::Var(id), _) => {
            if let Some(bound) = subst.get(id).cloned() {
                unify(&bound, b, subst)
            } else {
                // Occurs check: prevent infinite types
                if occurs_in(*id, b, subst) {
                    return Err(format!("infinite type: T{} occurs in {}", id.0, b));
                }
                subst.insert(*id, b.clone());
                Ok(b.clone())
            }
        }
        // Type variable on the right
        (_, Type::Var(id)) => {
            if let Some(bound) = subst.get(id).cloned() {
                unify(a, &bound, subst)
            } else {
                if occurs_in(*id, a, subst) {
                    return Err(format!("infinite type: T{} occurs in {}", id.0, a));
                }
                subst.insert(*id, a.clone());
                Ok(a.clone())
            }
        }
        // Identical primitives
        (Type::I32, Type::I32)
        | (Type::I64, Type::I64)
        | (Type::F32, Type::F32)
        | (Type::F64, Type::F64)
        | (Type::Bf16, Type::Bf16)
        | (Type::F16, Type::F16)
        | (Type::Complex, Type::Complex)
        | (Type::Bool, Type::Bool)
        | (Type::Str, Type::Str)
        | (Type::Void, Type::Void) => Ok(a.clone()),

        // Range: unify element types
        (Type::Range { elem: e1 }, Type::Range { elem: e2 }) => {
            let elem = unify(e1, e2, subst)?;
            Ok(Type::Range {
                elem: Box::new(elem),
            })
        }

        // Slice: unify element types
        (Type::Slice { elem: e1 }, Type::Slice { elem: e2 }) => {
            let elem = unify(e1, e2, subst)?;
            Ok(Type::Slice {
                elem: Box::new(elem),
            })
        }

        // Tensor: unify element types (shapes handled separately)
        (
            Type::Tensor { elem: e1, shape: s1 },
            Type::Tensor { elem: e2, shape: s2 },
        ) => {
            let elem = unify(e1, e2, subst)?;
            // For shape unification: if both have shapes, verify compatibility
            let shape = match (s1, s2) {
                (Some(sh1), Some(sh2)) => {
                    let mut shape_subst = ShapeSubst::new();
                    Some(unify_shapes(sh1, sh2, &mut shape_subst)?)
                }
                (Some(s), None) | (None, Some(s)) => Some(s.clone()),
                (None, None) => None,
            };
            Ok(Type::Tensor {
                elem: Box::new(elem),
                shape,
            })
        }

        // Buffer
        (Type::Buffer { elem: e1 }, Type::Buffer { elem: e2 }) => {
            let elem = unify(e1, e2, subst)?;
            Ok(Type::Buffer {
                elem: Box::new(elem),
            })
        }

        // Array
        (Type::Array { elem: e1, len: l1 }, Type::Array { elem: e2, len: l2 }) => {
            if l1 != l2 {
                return Err(format!("array length mismatch: {} vs {}", l1, l2));
            }
            let elem = unify(e1, e2, subst)?;
            Ok(Type::Array {
                elem: Box::new(elem),
                len: *l1,
            })
        }

        // Tuple
        (Type::Tuple(t1), Type::Tuple(t2)) => {
            if t1.len() != t2.len() {
                return Err(format!(
                    "tuple length mismatch: {} vs {}",
                    t1.len(),
                    t2.len()
                ));
            }
            let elems: Result<Vec<Type>, String> = t1
                .iter()
                .zip(t2.iter())
                .map(|(a, b)| unify(a, b, subst))
                .collect();
            Ok(Type::Tuple(elems?))
        }

        // Function
        (
            Type::Fn { params: p1, ret: r1 },
            Type::Fn { params: p2, ret: r2 },
        ) => {
            if p1.len() != p2.len() {
                return Err(format!(
                    "function arity mismatch: {} params vs {}",
                    p1.len(),
                    p2.len()
                ));
            }
            let params: Result<Vec<Type>, String> = p1
                .iter()
                .zip(p2.iter())
                .map(|(a, b)| unify(a, b, subst))
                .collect();
            let ret = unify(r1, r2, subst)?;
            Ok(Type::Fn {
                params: params?,
                ret: Box::new(ret),
            })
        }

        // Struct (nominal)
        (Type::Struct(s1), Type::Struct(s2)) => {
            if s1.name == s2.name {
                Ok(a.clone())
            } else {
                Err(format!(
                    "type mismatch: struct `{}` vs struct `{}`",
                    s1.name, s2.name
                ))
            }
        }

        // Class (nominal)
        (Type::Class(c1), Type::Class(c2)) => {
            if c1.name == c2.name {
                Ok(a.clone())
            } else {
                Err(format!(
                    "type mismatch: class `{}` vs class `{}`",
                    c1.name, c2.name
                ))
            }
        }

        // Enum (nominal — same name means same enum, unify variant fields pairwise)
        (Type::Enum(e1), Type::Enum(e2)) => {
            if e1.name == e2.name {
                // If both have concrete variant info, unify fields pairwise
                if e1.variants.len() == e2.variants.len() {
                    let mut unified_variants = Vec::new();
                    for (v1, v2) in e1.variants.iter().zip(e2.variants.iter()) {
                        if v1.name != v2.name || v1.fields.len() != v2.fields.len() {
                            return Err(format!(
                                "enum variant mismatch in `{}`: `{}` vs `{}`",
                                e1.name, v1.name, v2.name
                            ));
                        }
                        let mut unified_fields = Vec::new();
                        for (f1, f2) in v1.fields.iter().zip(v2.fields.iter()) {
                            unified_fields.push(unify(f1, f2, subst)?);
                        }
                        unified_variants.push(EnumVariant {
                            name: v1.name.clone(),
                            fields: unified_fields,
                        });
                    }
                    Ok(Type::Enum(EnumType {
                        name: e1.name.clone(),
                        type_params: e1.type_params.clone(),
                        variants: unified_variants,
                    }))
                } else {
                    Ok(a.clone())
                }
            } else {
                Err(format!(
                    "type mismatch: enum `{}` vs enum `{}`",
                    e1.name, e2.name
                ))
            }
        }

        _ => Err(format!("type mismatch: expected `{}`, found `{}`", a, b)),
    }
}

/// Span-aware unification: like `unify()` but emits a spanned `Diagnostic`
/// into `diag` on failure instead of returning `Err(String)`.
///
/// Returns `Ok(unified_type)` on success, or `Ok(Type::Error)` after emitting
/// a spanned error diagnostic on failure.  The caller never sees an `Err` —
/// error recovery proceeds with `Type::Error` which unifies with everything.
pub fn unify_spanned(
    a: &Type,
    b: &Type,
    subst: &mut TypeSubst,
    span: cjc_diag::Span,
    diag: &mut cjc_diag::DiagnosticBag,
) -> Type {
    match unify(a, b, subst) {
        Ok(ty) => ty,
        Err(msg) => {
            diag.emit(
                cjc_diag::Diagnostic::error(
                    "E0100",
                    msg,
                    span,
                )
                .with_hint(format!(
                    "expected `{}`, found `{}`",
                    a, b
                )),
            );
            Type::Error
        }
    }
}

/// Occurs check: does type variable `id` occur in type `ty`?
fn occurs_in(id: TypeVarId, ty: &Type, subst: &TypeSubst) -> bool {
    match ty {
        Type::Var(other) => {
            if *other == id {
                return true;
            }
            if let Some(bound) = subst.get(other) {
                occurs_in(id, bound, subst)
            } else {
                false
            }
        }
        Type::Tensor { elem, .. } => occurs_in(id, elem, subst),
        Type::Buffer { elem } => occurs_in(id, elem, subst),
        Type::Array { elem, .. } => occurs_in(id, elem, subst),
        Type::Tuple(elems) => elems.iter().any(|e| occurs_in(id, e, subst)),
        Type::Fn { params, ret } => {
            params.iter().any(|p| occurs_in(id, p, subst)) || occurs_in(id, ret, subst)
        }
        Type::Enum(e) => e
            .variants
            .iter()
            .any(|v| v.fields.iter().any(|f| occurs_in(id, f, subst))),
        _ => false,
    }
}

/// Apply a type substitution, replacing all bound Var(id) with their concrete types.
pub fn apply_subst(ty: &Type, subst: &TypeSubst) -> Type {
    match ty {
        Type::Var(id) => {
            if let Some(bound) = subst.get(id) {
                // Recursively apply in case of chained substitutions
                apply_subst(bound, subst)
            } else {
                ty.clone()
            }
        }
        Type::Tensor { elem, shape } => Type::Tensor {
            elem: Box::new(apply_subst(elem, subst)),
            shape: shape.clone(),
        },
        Type::Buffer { elem } => Type::Buffer {
            elem: Box::new(apply_subst(elem, subst)),
        },
        Type::Array { elem, len } => Type::Array {
            elem: Box::new(apply_subst(elem, subst)),
            len: *len,
        },
        Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| apply_subst(e, subst)).collect()),
        Type::Fn { params, ret } => Type::Fn {
            params: params.iter().map(|p| apply_subst(p, subst)).collect(),
            ret: Box::new(apply_subst(ret, subst)),
        },
        Type::Enum(e) => Type::Enum(EnumType {
            name: e.name.clone(),
            type_params: e.type_params.clone(),
            variants: e
                .variants
                .iter()
                .map(|v| EnumVariant {
                    name: v.name.clone(),
                    fields: v.fields.iter().map(|f| apply_subst(f, subst)).collect(),
                })
                .collect(),
        }),
        _ => ty.clone(),
    }
}

// ── Shape Unification ──────────────────────────────────────────

/// Unify two shape dimensions.
pub fn unify_shape_dim(
    a: &ShapeDim,
    b: &ShapeDim,
    subst: &mut ShapeSubst,
) -> Result<ShapeDim, String> {
    match (a, b) {
        (ShapeDim::Known(n1), ShapeDim::Known(n2)) => {
            if n1 == n2 {
                Ok(ShapeDim::Known(*n1))
            } else {
                Err(format!(
                    "shape dimension mismatch: expected {}, found {}",
                    n1, n2
                ))
            }
        }
        (ShapeDim::Symbolic(s), ShapeDim::Known(n)) | (ShapeDim::Known(n), ShapeDim::Symbolic(s)) => {
            if let Some(&bound) = subst.get(s.as_str()) {
                if bound == *n {
                    Ok(ShapeDim::Known(*n))
                } else {
                    Err(format!(
                        "symbolic shape `{}` already bound to {}, cannot unify with {}",
                        s, bound, n
                    ))
                }
            } else {
                subst.insert(s.clone(), *n);
                Ok(ShapeDim::Known(*n))
            }
        }
        (ShapeDim::Symbolic(s1), ShapeDim::Symbolic(s2)) => {
            if s1 == s2 {
                Ok(ShapeDim::Symbolic(s1.clone()))
            } else {
                // If one is bound, use that
                match (subst.get(s1).copied(), subst.get(s2).copied()) {
                    (Some(n1), Some(n2)) => {
                        if n1 == n2 {
                            Ok(ShapeDim::Known(n1))
                        } else {
                            Err(format!(
                                "shape conflict: `{}` = {} but `{}` = {}",
                                s1, n1, s2, n2
                            ))
                        }
                    }
                    (Some(n), None) => {
                        subst.insert(s2.clone(), n);
                        Ok(ShapeDim::Known(n))
                    }
                    (None, Some(n)) => {
                        subst.insert(s1.clone(), n);
                        Ok(ShapeDim::Known(n))
                    }
                    (None, None) => {
                        // Both unbound: keep symbolic, they are equated
                        Ok(ShapeDim::Symbolic(s1.clone()))
                    }
                }
            }
        }
    }
}

/// Unify two shape vectors dimension by dimension.
pub fn unify_shapes(
    a: &[ShapeDim],
    b: &[ShapeDim],
    subst: &mut ShapeSubst,
) -> Result<Vec<ShapeDim>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "rank mismatch: {}-D tensor vs {}-D tensor",
            a.len(),
            b.len()
        ));
    }
    a.iter()
        .zip(b.iter())
        .enumerate()
        .map(|(i, (da, db))| {
            unify_shape_dim(da, db, subst).map_err(|e| format!("dimension {}: {}", i, e))
        })
        .collect()
}

/// Compute broadcast-compatible shape from two shapes (NumPy rules).
/// Returns the broadcast result shape, or an error if incompatible.
pub fn broadcast_shapes(
    a: &Option<Vec<ShapeDim>>,
    b: &Option<Vec<ShapeDim>>,
) -> Result<Option<Vec<ShapeDim>>, String> {
    match (a, b) {
        (Some(sa), Some(sb)) => {
            let max_rank = sa.len().max(sb.len());
            let mut result = Vec::with_capacity(max_rank);

            // Right-align: pad shorter shape with Known(1) on the left
            for i in 0..max_rank {
                let da = if i < max_rank - sa.len() {
                    &ShapeDim::Known(1)
                } else {
                    &sa[i - (max_rank - sa.len())]
                };
                let db = if i < max_rank - sb.len() {
                    &ShapeDim::Known(1)
                } else {
                    &sb[i - (max_rank - sb.len())]
                };

                let dim = broadcast_dim(da, db, i)?;
                result.push(dim);
            }
            Ok(Some(result))
        }
        _ => Ok(None), // No shape info: pass through
    }
}

/// Broadcast a single dimension pair.
fn broadcast_dim(a: &ShapeDim, b: &ShapeDim, dim_idx: usize) -> Result<ShapeDim, String> {
    match (a, b) {
        (ShapeDim::Known(1), other) | (other, ShapeDim::Known(1)) => Ok(other.clone()),
        (ShapeDim::Known(n1), ShapeDim::Known(n2)) => {
            if n1 == n2 {
                Ok(ShapeDim::Known(*n1))
            } else {
                Err(format!(
                    "broadcast incompatible at dimension {}: {} vs {}",
                    dim_idx, n1, n2
                ))
            }
        }
        (ShapeDim::Symbolic(s), ShapeDim::Known(n)) | (ShapeDim::Known(n), ShapeDim::Symbolic(s)) => {
            if *n == 1 {
                Ok(ShapeDim::Symbolic(s.clone()))
            } else {
                // Symbolic could be 1 (broadcasts) or n (equal); conservatively allow
                Ok(ShapeDim::Known(*n))
            }
        }
        (ShapeDim::Symbolic(s1), ShapeDim::Symbolic(s2)) => {
            if s1 == s2 {
                Ok(ShapeDim::Symbolic(s1.clone()))
            } else {
                // Two different symbolic dims: conservatively keep first
                Ok(ShapeDim::Symbolic(s1.clone()))
            }
        }
    }
}

// ── Struct, Class, and Enum Type Info ────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    pub name: String,
    pub type_params: Vec<String>,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassType {
    pub name: String,
    pub type_params: Vec<String>,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumType {
    pub name: String,
    pub type_params: Vec<String>,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<Type>,
}

// ── Type Variables ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVarId(pub usize);

// ── Traits / Typeclasses ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub super_traits: Vec<String>,
    pub methods: Vec<MethodSig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MethodSig {
    pub name: String,
    pub params: Vec<Type>,
    pub ret: Type,
}

/// A trait implementation: "Type implements Trait".
#[derive(Debug, Clone)]
pub struct TraitImpl {
    pub trait_name: String,
    pub target_type: Type,
    pub type_args: Vec<Type>,
}

// ── Built-in Trait Hierarchy ────────────────────────────────────

pub const TRAIT_NUMERIC: &str = "Numeric";
pub const TRAIT_FLOAT: &str = "Float";
pub const TRAIT_INT: &str = "Int";
pub const TRAIT_DIFFERENTIABLE: &str = "Differentiable";

/// Built-in trait hierarchy:
///   Numeric
///   ├── Int
///   └── Float
///       └── Differentiable
pub fn builtin_trait_defs() -> Vec<TraitDef> {
    vec![
        TraitDef {
            name: TRAIT_NUMERIC.into(),
            type_params: vec![],
            super_traits: vec![],
            methods: vec![
                MethodSig {
                    name: "zero".into(),
                    params: vec![],
                    ret: Type::Var(TypeVarId(0)), // Self
                },
                MethodSig {
                    name: "one".into(),
                    params: vec![],
                    ret: Type::Var(TypeVarId(0)),
                },
            ],
        },
        TraitDef {
            name: TRAIT_INT.into(),
            type_params: vec![],
            super_traits: vec![TRAIT_NUMERIC.into()],
            methods: vec![],
        },
        TraitDef {
            name: TRAIT_FLOAT.into(),
            type_params: vec![],
            super_traits: vec![TRAIT_NUMERIC.into()],
            methods: vec![
                MethodSig {
                    name: "sqrt".into(),
                    params: vec![Type::Var(TypeVarId(0))],
                    ret: Type::Var(TypeVarId(0)),
                },
                MethodSig {
                    name: "ln".into(),
                    params: vec![Type::Var(TypeVarId(0))],
                    ret: Type::Var(TypeVarId(0)),
                },
                MethodSig {
                    name: "exp".into(),
                    params: vec![Type::Var(TypeVarId(0))],
                    ret: Type::Var(TypeVarId(0)),
                },
            ],
        },
        TraitDef {
            name: TRAIT_DIFFERENTIABLE.into(),
            type_params: vec![],
            super_traits: vec![TRAIT_FLOAT.into()],
            methods: vec![],
        },
    ]
}

/// Built-in trait implementations.
pub fn builtin_trait_impls() -> Vec<TraitImpl> {
    vec![
        // i32: Numeric, Int
        TraitImpl { trait_name: TRAIT_NUMERIC.into(), target_type: Type::I32, type_args: vec![] },
        TraitImpl { trait_name: TRAIT_INT.into(), target_type: Type::I32, type_args: vec![] },
        // i64: Numeric, Int
        TraitImpl { trait_name: TRAIT_NUMERIC.into(), target_type: Type::I64, type_args: vec![] },
        TraitImpl { trait_name: TRAIT_INT.into(), target_type: Type::I64, type_args: vec![] },
        // f32: Numeric, Float, Differentiable
        TraitImpl { trait_name: TRAIT_NUMERIC.into(), target_type: Type::F32, type_args: vec![] },
        TraitImpl { trait_name: TRAIT_FLOAT.into(), target_type: Type::F32, type_args: vec![] },
        TraitImpl { trait_name: TRAIT_DIFFERENTIABLE.into(), target_type: Type::F32, type_args: vec![] },
        // f64: Numeric, Float, Differentiable
        TraitImpl { trait_name: TRAIT_NUMERIC.into(), target_type: Type::F64, type_args: vec![] },
        TraitImpl { trait_name: TRAIT_FLOAT.into(), target_type: Type::F64, type_args: vec![] },
        TraitImpl { trait_name: TRAIT_DIFFERENTIABLE.into(), target_type: Type::F64, type_args: vec![] },
    ]
}

// ── Type Environment ────────────────────────────────────────────

/// Type checking environment.
pub struct TypeEnv {
    /// Named type definitions.
    pub type_defs: HashMap<String, Type>,
    /// Trait definitions.
    pub trait_defs: HashMap<String, TraitDef>,
    /// Trait implementations.
    pub trait_impls: Vec<TraitImpl>,
    /// Variable scopes (stack of scope frames).
    /// Each entry stores (type, is_mutable).
    scopes: Vec<HashMap<String, (Type, bool)>>,
    /// Const definitions (name -> value type). Always immutable.
    pub const_defs: HashMap<String, Type>,
    /// Function signatures.
    pub fn_sigs: HashMap<String, Vec<FnSigEntry>>,
    /// Next type variable ID.
    next_var: usize,
}

// ---------------------------------------------------------------------------
// Effect Classification
// ---------------------------------------------------------------------------

/// Bitflag effect classification for builtin functions.
///
/// Each flag represents a category of side effect. A function with `PURE` (no
/// flags set) is guaranteed to be deterministic and allocation-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectSet {
    bits: u16,
}

impl Default for EffectSet {
    fn default() -> Self {
        Self::PURE
    }
}

impl EffectSet {
    /// No effects — pure, deterministic, no allocation.
    pub const PURE: Self = Self { bits: 0 };

    // Individual effect flags.
    pub const IO: u16 = 0b0000_0001; // print, file read/write
    pub const ALLOC: u16 = 0b0000_0010; // heap allocation (Rc/String/Vec/Tensor)
    pub const GC: u16 = 0b0000_0100; // triggers GC (gc_alloc, gc_collect)
    pub const NONDET: u16 = 0b0000_1000; // nondeterministic (randn, clock, hash-order)
    pub const MUTATES: u16 = 0b0001_0000; // mutates arguments (push, Map.insert)
    pub const ARENA_OK: u16 = 0b0010_0000; // result can safely live on arena
    pub const CAPTURES: u16 = 0b0100_0000; // may capture/store arguments beyond call

    /// Create an EffectSet from raw bits.
    pub const fn new(bits: u16) -> Self {
        Self { bits }
    }

    /// Check whether a specific flag is set.
    pub const fn has(&self, flag: u16) -> bool {
        self.bits & flag != 0
    }

    /// Return a new set with the given flag added.
    pub const fn with(self, flag: u16) -> Self {
        Self {
            bits: self.bits | flag,
        }
    }

    /// True if no flags are set (pure function).
    pub const fn is_pure(&self) -> bool {
        self.bits == 0
    }

    /// True if the function does NOT trigger GC.
    pub const fn is_nogc_safe(&self) -> bool {
        !self.has(Self::GC)
    }

    /// True if the function may allocate (heap or GC).
    pub const fn may_alloc(&self) -> bool {
        self.bits & (Self::ALLOC | Self::GC) != 0
    }

    /// Raw bits (for serialization / debug).
    pub const fn bits(&self) -> u16 {
        self.bits
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pure() {
            return write!(f, "PURE");
        }
        let mut parts = Vec::new();
        if self.has(Self::IO) {
            parts.push("IO");
        }
        if self.has(Self::ALLOC) {
            parts.push("ALLOC");
        }
        if self.has(Self::GC) {
            parts.push("GC");
        }
        if self.has(Self::NONDET) {
            parts.push("NONDET");
        }
        if self.has(Self::MUTATES) {
            parts.push("MUTATES");
        }
        if self.has(Self::ARENA_OK) {
            parts.push("ARENA_OK");
        }
        if self.has(Self::CAPTURES) {
            parts.push("CAPTURES");
        }
        write!(f, "{}", parts.join("|"))
    }
}

// ---------------------------------------------------------------------------
// Effect Registry
// ---------------------------------------------------------------------------

pub mod effect_registry;

// ---------------------------------------------------------------------------
// Function Signature Entry
// ---------------------------------------------------------------------------

/// Entry in the function signature table (for multiple dispatch).
#[derive(Debug, Clone)]
pub struct FnSigEntry {
    pub name: String,
    pub type_params: Vec<(String, Vec<String>)>, // (name, trait bounds)
    pub params: Vec<(String, Type)>,
    pub ret: Type,
    pub is_nogc: bool,
    /// Effect classification for this function. Used by the nogc verifier,
    /// escape analysis, and optimizer to reason about side effects.
    pub effects: EffectSet,
}

impl TypeEnv {
    pub fn new() -> Self {
        let mut env = Self {
            type_defs: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: Vec::new(),
            scopes: vec![HashMap::new()],
            fn_sigs: HashMap::new(),
            next_var: 0,
            const_defs: HashMap::new(),
        };

        // Register built-in types
        env.type_defs.insert("i32".into(), Type::I32);
        env.type_defs.insert("i64".into(), Type::I64);
        env.type_defs.insert("f32".into(), Type::F32);
        env.type_defs.insert("f64".into(), Type::F64);
        env.type_defs.insert("bf16".into(), Type::Bf16);
        env.type_defs.insert("f16".into(), Type::F16);
        env.type_defs.insert("Complex".into(), Type::Complex);
        env.type_defs.insert("bool".into(), Type::Bool);
        env.type_defs.insert("String".into(), Type::Str);

        // Register prelude enum types: Option<T> and Result<T, E>
        env.type_defs.insert(
            "Option".into(),
            Type::Enum(EnumType {
                name: "Option".into(),
                type_params: vec!["T".into()],
                variants: vec![
                    EnumVariant {
                        name: "Some".into(),
                        fields: vec![Type::Unresolved("T".into())],
                    },
                    EnumVariant {
                        name: "None".into(),
                        fields: vec![],
                    },
                ],
            }),
        );
        env.type_defs.insert(
            "Result".into(),
            Type::Enum(EnumType {
                name: "Result".into(),
                type_params: vec!["T".into(), "E".into()],
                variants: vec![
                    EnumVariant {
                        name: "Ok".into(),
                        fields: vec![Type::Unresolved("T".into())],
                    },
                    EnumVariant {
                        name: "Err".into(),
                        fields: vec![Type::Unresolved("E".into())],
                    },
                ],
            }),
        );

        // Register Error struct (for error metadata)
        env.type_defs.insert(
            "Error".into(),
            Type::Struct(StructType {
                name: "Error".into(),
                type_params: vec![],
                fields: vec![
                    ("span".into(), Type::I64),
                    ("message".into(), Type::Str),
                ],
            }),
        );

        // Register built-in traits
        for def in builtin_trait_defs() {
            env.trait_defs.insert(def.name.clone(), def);
        }

        // Register built-in impls
        env.trait_impls = builtin_trait_impls();

        // Register bf16 trait impls
        env.trait_impls.push(TraitImpl {
            trait_name: TRAIT_NUMERIC.into(),
            target_type: Type::Bf16,
            type_args: vec![],
        });
        env.trait_impls.push(TraitImpl {
            trait_name: TRAIT_FLOAT.into(),
            target_type: Type::Bf16,
            type_args: vec![],
        });

        // Register f16 trait impls
        env.trait_impls.push(TraitImpl {
            trait_name: TRAIT_NUMERIC.into(),
            target_type: Type::F16,
            type_args: vec![],
        });
        env.trait_impls.push(TraitImpl {
            trait_name: TRAIT_FLOAT.into(),
            target_type: Type::F16,
            type_args: vec![],
        });

        // Register prelude variant constructors as functions
        // Some(v) -> Option<T>
        env.fn_sigs.entry("Some".into()).or_default().push(FnSigEntry {
            name: "Some".into(),
            type_params: vec![("T".into(), vec![])],
            params: vec![("_0".into(), Type::Unresolved("T".into()))],
            ret: Type::Enum(EnumType {
                name: "Option".into(),
                type_params: vec!["T".into()],
                variants: vec![
                    EnumVariant { name: "Some".into(), fields: vec![Type::Unresolved("T".into())] },
                    EnumVariant { name: "None".into(), fields: vec![] },
                ],
            }),
            is_nogc: false,
            effects: EffectSet::default(),
        });
        // None -> Option<T>
        env.fn_sigs.entry("None".into()).or_default().push(FnSigEntry {
            name: "None".into(),
            type_params: vec![("T".into(), vec![])],
            params: vec![],
            ret: Type::Enum(EnumType {
                name: "Option".into(),
                type_params: vec!["T".into()],
                variants: vec![
                    EnumVariant { name: "Some".into(), fields: vec![Type::Unresolved("T".into())] },
                    EnumVariant { name: "None".into(), fields: vec![] },
                ],
            }),
            is_nogc: false,
            effects: EffectSet::default(),
        });
        // Ok(v) -> Result<T, E>
        env.fn_sigs.entry("Ok".into()).or_default().push(FnSigEntry {
            name: "Ok".into(),
            type_params: vec![("T".into(), vec![]), ("E".into(), vec![])],
            params: vec![("_0".into(), Type::Unresolved("T".into()))],
            ret: Type::Enum(EnumType {
                name: "Result".into(),
                type_params: vec!["T".into(), "E".into()],
                variants: vec![
                    EnumVariant { name: "Ok".into(), fields: vec![Type::Unresolved("T".into())] },
                    EnumVariant { name: "Err".into(), fields: vec![Type::Unresolved("E".into())] },
                ],
            }),
            is_nogc: false,
            effects: EffectSet::default(),
        });
        // Err(e) -> Result<T, E>
        env.fn_sigs.entry("Err".into()).or_default().push(FnSigEntry {
            name: "Err".into(),
            type_params: vec![("T".into(), vec![]), ("E".into(), vec![])],
            params: vec![("_0".into(), Type::Unresolved("E".into()))],
            ret: Type::Enum(EnumType {
                name: "Result".into(),
                type_params: vec!["T".into(), "E".into()],
                variants: vec![
                    EnumVariant { name: "Ok".into(), fields: vec![Type::Unresolved("T".into())] },
                    EnumVariant { name: "Err".into(), fields: vec![Type::Unresolved("E".into())] },
                ],
            }),
            is_nogc: false,
            effects: EffectSet::default(),
        });

        env
    }

    pub fn fresh_var(&mut self) -> TypeVarId {
        let id = TypeVarId(self.next_var);
        self.next_var += 1;
        id
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Define an immutable variable in the current scope.
    pub fn define_var(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), (ty, false));
        }
    }

    /// Define a mutable variable in the current scope.
    pub fn define_var_mut(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), (ty, true));
        }
    }

    pub fn lookup_var(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some((ty, _)) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Look up a variable and return its (type, is_mutable) pair.
    pub fn lookup_var_entry(&self, name: &str) -> Option<(&Type, bool)> {
        for scope in self.scopes.iter().rev() {
            if let Some((ty, mutable)) = scope.get(name) {
                return Some((ty, *mutable));
            }
        }
        None
    }

    /// Returns true if the named variable exists AND is mutable.
    pub fn is_var_mutable(&self, name: &str) -> bool {
        self.lookup_var_entry(name).map(|(_, m)| m).unwrap_or(false)
    }

    pub fn resolve_type_name(&self, name: &str) -> Option<Type> {
        self.type_defs.get(name).cloned()
    }

    /// Check if a type satisfies a trait bound.
    pub fn satisfies_trait(&self, ty: &Type, trait_name: &str) -> bool {
        // Error type satisfies all traits (for error recovery)
        if ty.is_error() {
            return true;
        }

        // Check direct implementations
        for imp in &self.trait_impls {
            if imp.trait_name == trait_name && self.types_match(&imp.target_type, ty) {
                return true;
            }
        }

        // Check sub-trait relationships
        if let Some(trait_def) = self.trait_defs.get(trait_name) {
            // If the type satisfies a sub-trait, it also satisfies super-traits
            for sub_trait_name in self.trait_defs.keys().cloned().collect::<Vec<_>>() {
                if sub_trait_name == trait_name {
                    continue;
                }
                if let Some(sub_def) = self.trait_defs.get(&sub_trait_name) {
                    if sub_def.super_traits.contains(&trait_name.to_string()) {
                        // If type implements the sub-trait, it implements this trait
                        for imp in &self.trait_impls {
                            if imp.trait_name == sub_trait_name
                                && self.types_match(&imp.target_type, ty)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            // Also check if the trait has super_traits that the type satisfies
            let _ = trait_def;
        }

        false
    }

    /// Check if two types match (simple structural equality for v1).
    pub fn types_match(&self, a: &Type, b: &Type) -> bool {
        if a.is_error() || b.is_error() {
            return true;
        }
        match (a, b) {
            (Type::I32, Type::I32)
            | (Type::I64, Type::I64)
            | (Type::F32, Type::F32)
            | (Type::F64, Type::F64)
            | (Type::Bool, Type::Bool)
            | (Type::Str, Type::Str)
            | (Type::Void, Type::Void) => true,
            (Type::Tensor { elem: e1, .. }, Type::Tensor { elem: e2, .. }) => {
                self.types_match(e1, e2)
            }
            (Type::Buffer { elem: e1 }, Type::Buffer { elem: e2 }) => self.types_match(e1, e2),
            (
                Type::Array {
                    elem: e1, len: l1, ..
                },
                Type::Array {
                    elem: e2, len: l2, ..
                },
            ) => l1 == l2 && self.types_match(e1, e2),
            (Type::Struct(s1), Type::Struct(s2)) => s1.name == s2.name,
            (Type::Class(c1), Type::Class(c2)) => c1.name == c2.name,
            (Type::Enum(e1), Type::Enum(e2)) => e1.name == e2.name,
            (Type::Bf16, Type::Bf16) => true,
            (Type::F16, Type::F16) => true,
            (Type::Range { elem: e1 }, Type::Range { elem: e2 }) => self.types_match(e1, e2),
            (Type::Slice { elem: e1 }, Type::Slice { elem: e2 }) => self.types_match(e1, e2),
            (
                Type::Fn {
                    params: p1,
                    ret: r1,
                },
                Type::Fn {
                    params: p2,
                    ret: r2,
                },
            ) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2).all(|(a, b)| self.types_match(a, b))
                    && self.types_match(r1, r2)
            }
            (Type::Map { key: k1, value: v1 }, Type::Map { key: k2, value: v2 }) => {
                self.types_match(k1, k2) && self.types_match(v1, v2)
            }
            (Type::SparseTensor { elem: e1 }, Type::SparseTensor { elem: e2 }) => {
                self.types_match(e1, e2)
            }
            (Type::Var(a), Type::Var(b)) => a == b,
            _ => false,
        }
    }

    /// Check shapes are compatible for matmul: [M, K] x [K, N] -> [M, N].
    pub fn check_matmul_shapes(
        &self,
        a_shape: &Option<Vec<ShapeDim>>,
        b_shape: &Option<Vec<ShapeDim>>,
    ) -> Result<Option<Vec<ShapeDim>>, String> {
        match (a_shape, b_shape) {
            (Some(a), Some(b)) => {
                if a.len() != 2 || b.len() != 2 {
                    return Err(format!(
                        "matmul requires 2D tensors, got {}D and {}D",
                        a.len(),
                        b.len()
                    ));
                }
                // Check inner dimensions match
                match (&a[1], &b[0]) {
                    (ShapeDim::Known(k1), ShapeDim::Known(k2)) => {
                        if k1 != k2 {
                            return Err(format!(
                                "matmul inner dimension mismatch: {} vs {}",
                                k1, k2
                            ));
                        }
                    }
                    (ShapeDim::Symbolic(s1), ShapeDim::Symbolic(s2)) => {
                        if s1 != s2 {
                            return Err(format!(
                                "matmul inner dimension mismatch: {} vs {}",
                                s1, s2
                            ));
                        }
                    }
                    _ => {} // Mixed known/symbolic: allow (runtime check)
                }
                Ok(Some(vec![a[0].clone(), b[1].clone()]))
            }
            _ => Ok(None), // No shape info: pass through
        }
    }

    /// Check if a concrete type satisfies all listed trait bounds.
    pub fn check_bounds(&self, ty: &Type, bounds: &[String]) -> bool {
        bounds.iter().all(|b| self.satisfies_trait(ty, b))
    }

    /// Register a function signature.
    pub fn register_fn(&mut self, entry: FnSigEntry) {
        self.fn_sigs
            .entry(entry.name.clone())
            .or_default()
            .push(entry);
    }

    /// Register a struct type.
    pub fn register_struct(&mut self, name: &str, st: StructType) {
        self.type_defs.insert(name.to_string(), Type::Struct(st));
    }

    /// Register a class type.
    pub fn register_class(&mut self, name: &str, ct: ClassType) {
        self.type_defs.insert(name.to_string(), Type::Class(ct));
    }

    /// Register an enum type.
    pub fn register_enum_type(&mut self, name: &str, et: EnumType) {
        self.type_defs.insert(name.to_string(), Type::Enum(et));
    }

    /// Look up which enum a variant belongs to (by variant name).
    /// Returns the enum type if found.
    pub fn lookup_variant_enum(&self, variant_name: &str) -> Option<&EnumType> {
        for ty in self.type_defs.values() {
            if let Type::Enum(et) = ty {
                if et.variants.iter().any(|v| v.name == variant_name) {
                    return Some(et);
                }
            }
        }
        None
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ── Type Checker ────────────────────────────────────────────────

use cjc_ast::*;
use cjc_diag::{Diagnostic, DiagnosticBag};

fn to_diag_span(span: cjc_ast::Span) -> cjc_diag::Span {
    cjc_diag::Span::new(span.start, span.end)
}

/// P2-3: Returns true if an expression is a compile-time constant (pure literal).
/// Only literal values are accepted as const initializers.
fn is_const_expr(expr: &cjc_ast::Expr) -> bool {
    use cjc_ast::ExprKind;
    match &expr.kind {
        ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::RawStringLit(_)
        | ExprKind::ByteCharLit(_) => true,
        ExprKind::Unary { op: cjc_ast::UnaryOp::Neg, operand } => is_const_expr(operand),
        _ => false,
    }
}

pub struct TypeChecker {
    pub env: TypeEnv,
    pub diagnostics: DiagnosticBag,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            diagnostics: DiagnosticBag::new(),
        }
    }

    pub fn check_program(&mut self, program: &Program) {
        // First pass: register all type and function declarations
        for decl in &program.declarations {
            self.register_decl(decl);
        }

        // Second pass: type-check function bodies
        for decl in &program.declarations {
            self.check_decl(decl);
        }
    }

    fn register_decl(&mut self, decl: &Decl) {
        match &decl.kind {
            DeclKind::Struct(s) => {
                let fields: Vec<(String, Type)> = s
                    .fields
                    .iter()
                    .map(|f| (f.name.name.clone(), self.resolve_type_expr(&f.ty)))
                    .collect();
                let st = StructType {
                    name: s.name.name.clone(),
                    type_params: s.type_params.iter().map(|p| p.name.name.clone()).collect(),
                    fields,
                };
                self.env.register_struct(&s.name.name, st);
            }
            DeclKind::Class(c) => {
                let fields: Vec<(String, Type)> = c
                    .fields
                    .iter()
                    .map(|f| (f.name.name.clone(), self.resolve_type_expr(&f.ty)))
                    .collect();
                let ct = ClassType {
                    name: c.name.name.clone(),
                    type_params: c.type_params.iter().map(|p| p.name.name.clone()).collect(),
                    fields,
                };
                self.env.register_class(&c.name.name, ct);
            }
            DeclKind::Fn(f) => {
                let params: Vec<(String, Type)> = f
                    .params
                    .iter()
                    .map(|p| (p.name.name.clone(), self.resolve_type_expr(&p.ty)))
                    .collect();
                let ret = f
                    .return_type
                    .as_ref()
                    .map(|t| self.resolve_type_expr(t))
                    .unwrap_or(Type::Void);
                let type_params: Vec<(String, Vec<String>)> = f
                    .type_params
                    .iter()
                    .map(|tp| {
                        (
                            tp.name.name.clone(),
                            tp.bounds.iter().map(|b| self.type_expr_name(b)).collect(),
                        )
                    })
                    .collect();
                self.env.register_fn(FnSigEntry {
                    name: f.name.name.clone(),
                    type_params,
                    params,
                    ret,
                    is_nogc: f.is_nogc,
                    effects: EffectSet::default(),
                });
            }
            DeclKind::Trait(t) => {
                let methods: Vec<MethodSig> = t
                    .methods
                    .iter()
                    .map(|m| MethodSig {
                        name: m.name.name.clone(),
                        params: m.params.iter().map(|p| self.resolve_type_expr(&p.ty)).collect(),
                        ret: m
                            .return_type
                            .as_ref()
                            .map(|t| self.resolve_type_expr(t))
                            .unwrap_or(Type::Void),
                    })
                    .collect();
                let def = TraitDef {
                    name: t.name.name.clone(),
                    type_params: t.type_params.iter().map(|p| p.name.name.clone()).collect(),
                    super_traits: t.super_traits.iter().map(|s| self.type_expr_name(s)).collect(),
                    methods,
                };
                self.env.trait_defs.insert(t.name.name.clone(), def);
            }
            DeclKind::Impl(i) => {
                if let Some(ref trait_ref) = i.trait_ref {
                    let trait_name = self.type_expr_name(trait_ref);
                    let target = self.resolve_type_expr(&i.target);
                    self.env.trait_impls.push(TraitImpl {
                        trait_name,
                        target_type: target,
                        type_args: vec![],
                    });
                }
                // Register methods
                for method in &i.methods {
                    let target_name = self.type_expr_name(&i.target);
                    let params: Vec<(String, Type)> = method
                        .params
                        .iter()
                        .map(|p| (p.name.name.clone(), self.resolve_type_expr(&p.ty)))
                        .collect();
                    let ret = method
                        .return_type
                        .as_ref()
                        .map(|t| self.resolve_type_expr(t))
                        .unwrap_or(Type::Void);
                    let qualified_name = format!("{}.{}", target_name, method.name.name);
                    self.env.register_fn(FnSigEntry {
                        name: qualified_name,
                        type_params: vec![],
                        params,
                        ret,
                        is_nogc: method.is_nogc,
                        effects: EffectSet::default(),
                    });
                }
            }
            DeclKind::Enum(e) => {
                self.register_enum(e);
            }
            _ => {}
        }
    }

    /// Register an enum type and its variant constructors in the type environment.
    fn register_enum(&mut self, e: &cjc_ast::EnumDecl) {
        let type_params: Vec<String> =
            e.type_params.iter().map(|p| p.name.name.clone()).collect();
        let variants: Vec<EnumVariant> = e
            .variants
            .iter()
            .map(|v| EnumVariant {
                name: v.name.name.clone(),
                fields: v.fields.iter().map(|f| self.resolve_type_expr(f)).collect(),
            })
            .collect();
        let enum_type = EnumType {
            name: e.name.name.clone(),
            type_params: type_params.clone(),
            variants: variants.clone(),
        };
        self.env
            .type_defs
            .insert(e.name.name.clone(), Type::Enum(enum_type.clone()));

        // Register each variant as a constructor function
        let ret_type = Type::Enum(enum_type);
        for variant in &variants {
            let params: Vec<(String, Type)> = variant
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| (format!("_{}", i), f.clone()))
                .collect();
            let tp: Vec<(String, Vec<String>)> =
                type_params.iter().map(|t| (t.clone(), vec![])).collect();
            self.env.register_fn(FnSigEntry {
                name: variant.name.clone(),
                type_params: tp,
                params,
                ret: ret_type.clone(),
                is_nogc: false,
                effects: EffectSet::default(),
            });
        }
    }

    fn check_decl(&mut self, decl: &Decl) {
        match &decl.kind {
            DeclKind::Fn(f) => self.check_fn(f),
            DeclKind::Let(l) => {
                self.check_let(l, decl.span);
            }
            DeclKind::Impl(i) => {
                self.check_impl(i, decl.span);
            }
            // P2-3: Const expressions — type-check and register in env.
            DeclKind::Const(c) => {
                self.check_const_decl(c, decl.span);
            }
            _ => {}
        }
    }

    /// P2-3: Type-check a compile-time constant declaration.
    ///
    /// Evaluates only pure literal expressions; emits E0400 if the
    /// initializer is not a compile-time constant.
    fn check_const_decl(&mut self, c: &cjc_ast::ConstDecl, span: cjc_ast::Span) {
        let declared_ty = self.resolve_type_expr(&c.ty);
        let init_ty = self.check_expr(&c.value);

        // Verify the initializer is a compile-time constant (pure literal).
        if !is_const_expr(&c.value) {
            self.diagnostics.emit(
                Diagnostic::error(
                    "E0400",
                    format!(
                        "const `{}` initializer is not a compile-time constant expression",
                        c.name.name
                    ),
                    to_diag_span(span),
                )
                .with_hint("const initializers must be literal values (integer, float, bool, or string)"),
            );
        }

        // Type-check the declared type vs initializer type.
        if !declared_ty.is_error() && !init_ty.is_error() && !self.env.types_match(&declared_ty, &init_ty) {
            self.diagnostics.emit(
                Diagnostic::error(
                    "E0401",
                    format!(
                        "const `{}` type mismatch: declared `{}`, initialized with `{}`",
                        c.name.name, declared_ty, init_ty
                    ),
                    to_diag_span(span),
                ),
            );
        }

        // Register in const_defs AND as an immutable var in the top-level scope.
        self.env.const_defs.insert(c.name.name.clone(), declared_ty.clone());
        self.env.define_var(&c.name.name, declared_ty);
    }

    /// Type-check an `impl` block.
    ///
    /// When a `trait_ref` is present (`impl Trait for Type`), enforces:
    /// 1. The trait must be defined (known to the type environment).
    /// 2. All required trait methods must be present in the impl.
    /// 3. Duplicate impls for the same (trait, type) pair are rejected.
    ///
    /// Also type-checks all method bodies regardless of whether a trait is named.
    fn check_impl(&mut self, i: &cjc_ast::ImplDecl, span: cjc_ast::Span) {
        let target_name = self.type_expr_name(&i.target);

        if let Some(ref trait_ref) = i.trait_ref {
            let trait_name = self.type_expr_name(trait_ref);

            // (1) Verify the trait is defined.
            if !self.env.trait_defs.contains_key(&trait_name) {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0200",
                        format!("trait `{}` is not defined", trait_name),
                        to_diag_span(span),
                    )
                    .with_hint(format!(
                        "declare the trait with `trait {} {{ ... }}`",
                        trait_name
                    )),
                );
                // Still type-check method bodies for good diagnostics
                for method in &i.methods {
                    self.check_fn(method);
                }
                return;
            }

            // (2) Check for duplicate impls: same trait for same type.
            let impl_count = self.env.trait_impls.iter().filter(|imp| {
                imp.trait_name == trait_name
                    && self.env.types_match(&imp.target_type, &self.env.resolve_type_name(&target_name).unwrap_or(Type::Unresolved(target_name.clone())))
            }).count();
            if impl_count > 1 {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0201",
                        format!(
                            "duplicate implementation of trait `{}` for type `{}`",
                            trait_name, target_name
                        ),
                        to_diag_span(span),
                    )
                    .with_hint("only one impl of a trait per type is allowed"),
                );
            }

            // (3) Verify all required trait methods are present.
            let required_methods: Vec<String> = self.env
                .trait_defs
                .get(&trait_name)
                .map(|def| def.methods.iter().map(|m| m.name.clone()).collect())
                .unwrap_or_default();

            let provided_methods: Vec<String> =
                i.methods.iter().map(|m| m.name.name.clone()).collect();

            for required in &required_methods {
                if !provided_methods.contains(required) {
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0202",
                            format!(
                                "impl of `{}` for `{}` is missing method `{}`",
                                trait_name, target_name, required
                            ),
                            to_diag_span(span),
                        )
                        .with_hint(format!(
                            "add `fn {}(...)` to satisfy the trait requirement",
                            required
                        )),
                    );
                }
            }
        }

        // Type-check all method bodies.
        for method in &i.methods {
            self.check_fn(method);
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        self.env.push_scope();

        // Bind parameters
        for param in &f.params {
            let ty = self.resolve_type_expr(&param.ty);
            self.env.define_var(&param.name.name, ty);
        }

        // Bind type parameters
        for tp in &f.type_params {
            let var = self.env.fresh_var();
            self.env.define_var(&tp.name.name, Type::Var(var));
        }

        let expected_ret = f
            .return_type
            .as_ref()
            .map(|t| self.resolve_type_expr(t))
            .unwrap_or(Type::Void);

        // Check nogc constraints
        if f.is_nogc {
            self.check_nogc_block(&f.body);
        }

        // Check body
        let body_type = self.check_block(&f.body);

        // Check return type
        if !expected_ret.is_error()
            && !body_type.is_error()
            && expected_ret != Type::Void
            && !self.env.types_match(&body_type, &expected_ret)
        {
            self.diagnostics.emit(
                Diagnostic::error(
                    "E0103",
                    format!(
                        "mismatched return type: expected `{}`, found `{}`",
                        expected_ret, body_type
                    ),
                    to_diag_span(f.body.span),
                )
                .with_hint(format!("function `{}` should return `{}`", f.name.name, expected_ret)),
            );
        }

        self.env.pop_scope();
    }

    fn check_let(&mut self, l: &LetStmt, span: cjc_ast::Span) {
        let init_type = self.check_expr(&l.init);

        if let Some(ref ty_expr) = l.ty {
            let declared = self.resolve_type_expr(ty_expr);
            if !declared.is_error()
                && !init_type.is_error()
                && !self.env.types_match(&declared, &init_type)
            {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0104",
                        format!(
                            "type mismatch: declared `{}`, initialized with `{}`",
                            declared, init_type
                        ),
                        to_diag_span(span),
                    )
                    .with_hint(format!(
                        "change the type annotation to `{}` or fix the initializer",
                        init_type
                    )),
                );
            }
            if l.mutable {
                self.env.define_var_mut(&l.name.name, declared);
            } else {
                self.env.define_var(&l.name.name, declared);
            }
        } else if l.mutable {
            self.env.define_var_mut(&l.name.name, init_type);
        } else {
            self.env.define_var(&l.name.name, init_type);
        }
    }

    fn check_block(&mut self, block: &Block) -> Type {
        self.env.push_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }

        let result = if let Some(ref expr) = block.expr {
            self.check_expr(expr)
        } else {
            Type::Void
        };

        self.env.pop_scope();
        result
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let(l) => self.check_let(l, stmt.span),
            StmtKind::Expr(e) => {
                self.check_expr(e);
            }
            StmtKind::Return(e) => {
                if let Some(expr) = e {
                    self.check_expr(expr);
                }
            }
            StmtKind::If(if_stmt) => self.check_if(if_stmt),
            StmtKind::While(w) => {
                let cond_type = self.check_expr(&w.condition);
                if !cond_type.is_error() && cond_type != Type::Bool {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0105",
                        format!("while condition must be `bool`, found `{}`", cond_type),
                        to_diag_span(stmt.span),
                    ));
                }
                self.check_block(&w.body);
            }
            StmtKind::For(f) => {
                // Type-check the iterator expressions
                match &f.iter {
                    cjc_ast::ForIter::Range { start, end } => {
                        self.check_expr(start);
                        self.check_expr(end);
                    }
                    cjc_ast::ForIter::Expr(expr) => {
                        self.check_expr(expr);
                    }
                }
                self.check_block(&f.body);
            }
            StmtKind::NoGcBlock(block) => {
                self.check_nogc_block(block);
                self.check_block(block);
            }
            // Break/Continue are validated at parse time (loop depth check).
            // No type checking needed — they produce no values.
            StmtKind::Break | StmtKind::Continue => {}
        }
    }

    fn check_if(&mut self, if_stmt: &IfStmt) {
        let cond_type = self.check_expr(&if_stmt.condition);
        if !cond_type.is_error() && cond_type != Type::Bool {
            self.diagnostics.emit(Diagnostic::error(
                "E0105",
                format!("if condition must be `bool`, found `{}`", cond_type),
                to_diag_span(if_stmt.condition.span),
            ));
        }
        self.check_block(&if_stmt.then_block);
        if let Some(ref else_branch) = if_stmt.else_branch {
            match else_branch {
                ElseBranch::ElseIf(elif) => self.check_if(elif),
                ElseBranch::Else(block) => {
                    self.check_block(block);
                }
            }
        }
    }

    fn check_nogc_block(&mut self, block: &Block) {
        // Check that no GC types are allocated in the block
        for stmt in &block.stmts {
            if let StmtKind::Let(ref l) = stmt.kind {
                let ty = self.check_expr(&l.init);
                if ty.is_gc_type() {
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0201",
                            format!(
                                "cannot allocate GC-managed type `{}` in `nogc` context",
                                ty
                            ),
                            to_diag_span(stmt.span),
                        )
                        .with_label(
                            to_diag_span(stmt.span),
                            "GC allocation not allowed here",
                        )
                        .with_hint(
                            "use value types (struct, Tensor, Buffer) in nogc zones; \
                             move GC-managed objects outside the nogc block",
                        ),
                    );
                }
            }
        }
    }

    /// Static exhaustiveness checking for match over enum types.
    /// Emits a diagnostic if not all variants are covered and no wildcard/binding exists.
    fn check_match_exhaustiveness(
        &mut self,
        scrutinee_type: &Type,
        arms: &[MatchArm],
        span: cjc_ast::Span,
    ) {
        let enum_type = match scrutinee_type {
            Type::Enum(e) => e,
            _ => return, // Only check exhaustiveness for enum types
        };

        // Collect all variant names for the enum under scrutiny
        let all_variant_names: Vec<String> = enum_type
            .variants
            .iter()
            .map(|v| v.name.clone())
            .collect();

        // Collect covered variant names and check for wildcard/binding patterns.
        // A `Binding(name)` is treated as a variant pattern (not a wildcard) if
        // `name` matches one of the enum's declared variant names.  This handles
        // the common case where bare enum variant names (e.g., `Red`, `Green`) are
        // parsed as `Binding` rather than `Variant` by the parser when written
        // without parentheses.
        let mut covered_variants: Vec<String> = Vec::new();
        let mut has_wildcard = false;

        for arm in arms {
            match &arm.pattern.kind {
                PatternKind::Wildcard => {
                    has_wildcard = true;
                }
                PatternKind::Binding(name) => {
                    if all_variant_names.contains(&name.name) {
                        // Name matches an enum variant — treat as variant pattern.
                        covered_variants.push(name.name.clone());
                    } else {
                        // Pure binding variable — covers all remaining arms.
                        has_wildcard = true;
                    }
                }
                PatternKind::Variant {
                    variant, ..
                } => {
                    covered_variants.push(variant.name.clone());
                }
                _ => {}
            }
        }

        if has_wildcard {
            return; // Wildcard covers everything
        }

        // Check for missing variants
        let missing: Vec<&str> = enum_type
            .variants
            .iter()
            .filter(|v| !covered_variants.contains(&v.name))
            .map(|v| v.name.as_str())
            .collect();

        if !missing.is_empty() {
            self.diagnostics.emit(
                Diagnostic::error(
                    "E0130",
                    format!(
                        "non-exhaustive match on enum `{}`: missing variant(s) {}",
                        enum_type.name,
                        missing
                            .iter()
                            .map(|n| format!("`{}`", n))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    to_diag_span(span),
                )
                .with_hint("add the missing variant(s) or a wildcard `_` arm"),
            );
        }
    }

    pub fn check_expr(&mut self, expr: &Expr) -> Type {
        match &expr.kind {
            ExprKind::IntLit(_) => Type::I64,
            ExprKind::FloatLit(_) => Type::F64,
            ExprKind::StringLit(_) => Type::Str,
            ExprKind::ByteStringLit(_) => Type::ByteSlice,
            ExprKind::ByteCharLit(_) => Type::U8,
            ExprKind::RawStringLit(_) => Type::Str,
            ExprKind::RawByteStringLit(_) => Type::ByteSlice,
            ExprKind::FStringLit(segments) => {
                // P2-5: Type-check each interpolated expression; result type is always Str.
                for (_lit, interp) in segments {
                    if let Some(e) = interp {
                        self.check_expr(e);
                    }
                }
                Type::Str
            }
            ExprKind::RegexLit { .. } => Type::Regex,
            ExprKind::TensorLit { rows } => {
                // Type-check all elements, infer element type as f64
                for row in rows {
                    for expr in row {
                        self.check_expr(expr);
                    }
                }
                Type::Tensor {
                    elem: Box::new(Type::F64),
                    shape: None,
                }
            }
            ExprKind::BoolLit(_) => Type::Bool,
            ExprKind::Ident(id) => {
                if let Some(ty) = self.env.lookup_var(&id.name) {
                    ty.clone()
                } else {
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0106",
                            format!("undefined variable `{}`", id.name),
                            to_diag_span(id.span),
                        )
                        .with_hint("check for typos or declare the variable with `let`"),
                    );
                    Type::Error
                }
            }
            ExprKind::Binary { op, left, right } => {
                let lt = self.check_expr(left);
                let rt = self.check_expr(right);
                self.check_binary_op(*op, &lt, &rt, expr.span)
            }
            ExprKind::Unary { op, operand } => {
                let t = self.check_expr(operand);
                match op {
                    UnaryOp::Neg => {
                        if t.is_numeric() || t.is_error() {
                            t
                        } else {
                            self.diagnostics.emit(Diagnostic::error(
                                "E0107",
                                format!("cannot negate type `{}`", t),
                                to_diag_span(expr.span),
                            ));
                            Type::Error
                        }
                    }
                    UnaryOp::Not => {
                        if t == Type::Bool || t.is_error() {
                            Type::Bool
                        } else {
                            self.diagnostics.emit(Diagnostic::error(
                                "E0107",
                                format!("cannot apply `!` to type `{}`", t),
                                to_diag_span(expr.span),
                            ));
                            Type::Error
                        }
                    }
                    UnaryOp::BitNot => {
                        if t.is_int() || t.is_error() {
                            t
                        } else {
                            self.diagnostics.emit(Diagnostic::error(
                                "E0107",
                                format!("cannot apply `~` to type `{}`; bitwise NOT requires an integer", t),
                                to_diag_span(expr.span),
                            ));
                            Type::Error
                        }
                    }
                }
            }
            ExprKind::Call { callee, args } => {
                // Simple function call resolution
                if let ExprKind::Ident(id) = &callee.kind {
                    self.check_fn_call(&id.name, args, expr.span)
                } else if let ExprKind::Field { object, name } = &callee.kind {
                    let obj_ty = self.check_expr(object);
                    let qualified = format!("{}.{}", obj_ty, name.name);
                    self.check_fn_call(&qualified, args, expr.span)
                } else {
                    let _callee_ty = self.check_expr(callee);
                    for arg in args {
                        self.check_expr(&arg.value);
                    }
                    Type::Error
                }
            }
            ExprKind::Field { object, name } => {
                let obj_ty = self.check_expr(object);
                match &obj_ty {
                    Type::Struct(st) => {
                        if let Some((_, ft)) = st.fields.iter().find(|(n, _)| n == &name.name) {
                            ft.clone()
                        } else {
                            self.diagnostics.emit(
                                Diagnostic::error(
                                    "E0108",
                                    format!(
                                        "no field `{}` on struct `{}`",
                                        name.name, st.name
                                    ),
                                    to_diag_span(name.span),
                                )
                                .with_hint(format!(
                                    "available fields: {}",
                                    st.fields
                                        .iter()
                                        .map(|(n, _)| n.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )),
                            );
                            Type::Error
                        }
                    }
                    Type::Class(ct) => {
                        if let Some((_, ft)) = ct.fields.iter().find(|(n, _)| n == &name.name) {
                            ft.clone()
                        } else {
                            self.diagnostics.emit(Diagnostic::error(
                                "E0108",
                                format!("no field `{}` on class `{}`", name.name, ct.name),
                                to_diag_span(name.span),
                            ));
                            Type::Error
                        }
                    }
                    Type::Tensor { .. } => {
                        match name.name.as_str() {
                            "shape" => Type::Array {
                                elem: Box::new(Type::I64),
                                len: 0,
                            },
                            "buffer" => Type::Buffer {
                                elem: Box::new(Type::F64),
                            },
                            _ => {
                                // Could be a method — return Error for now
                                Type::Error
                            }
                        }
                    }
                    Type::Error => Type::Error,
                    _ => {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0108",
                            format!("cannot access field `{}` on type `{}`", name.name, obj_ty),
                            to_diag_span(name.span),
                        ));
                        Type::Error
                    }
                }
            }
            ExprKind::Index { object, index } => {
                let obj_ty = self.check_expr(object);
                let _idx_ty = self.check_expr(index);
                match &obj_ty {
                    Type::Array { elem, .. } => *elem.clone(),
                    Type::Tensor { elem, .. } => *elem.clone(),
                    Type::Buffer { elem } => *elem.clone(),
                    Type::Error => Type::Error,
                    _ => {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0109",
                            format!("cannot index into type `{}`", obj_ty),
                            to_diag_span(expr.span),
                        ));
                        Type::Error
                    }
                }
            }
            ExprKind::MultiIndex { object, indices } => {
                let obj_ty = self.check_expr(object);
                for idx in indices {
                    self.check_expr(idx);
                }
                match &obj_ty {
                    Type::Tensor { elem, .. } => *elem.clone(),
                    Type::Error => Type::Error,
                    _ => {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0109",
                            format!("multi-index not supported on type `{}`", obj_ty),
                            to_diag_span(expr.span),
                        ));
                        Type::Error
                    }
                }
            }
            ExprKind::Assign { target, value } => {
                let _tt = self.check_expr(target);
                let _vt = self.check_expr(value);
                // P0-3: Enforce immutable binding — emit E0150 if target is a
                // bare identifier that was declared without `mut`.
                if let ExprKind::Ident(id) = &target.kind {
                    if let Some((_ty, is_mut)) = self.env.lookup_var_entry(&id.name) {
                        if !is_mut {
                            self.diagnostics.emit(
                                Diagnostic::error(
                                    "E0150",
                                    format!(
                                        "cannot assign to immutable variable `{}`",
                                        id.name
                                    ),
                                    to_diag_span(expr.span),
                                )
                                .with_hint(format!(
                                    "consider making `{}` mutable: `let mut {} = ...`",
                                    id.name, id.name
                                )),
                            );
                        }
                    }
                }
                Type::Void
            }
            ExprKind::Pipe { left, right } => {
                let _lt = self.check_expr(left);
                // Pipe semantics: result type is the result of the right expression
                self.check_expr(right)
            }
            ExprKind::Block(block) => self.check_block(block),
            ExprKind::StructLit { name, fields } => {
                if let Some(ty) = self.env.resolve_type_name(&name.name) {
                    if let Type::Struct(ref st) = ty {
                        for field in fields {
                            let ft = self.check_expr(&field.value);
                            if let Some((_, expected)) =
                                st.fields.iter().find(|(n, _)| n == &field.name.name)
                            {
                                if !ft.is_error()
                                    && !expected.is_error()
                                    && !self.env.types_match(&ft, expected)
                                {
                                    self.diagnostics.emit(Diagnostic::error(
                                        "E0110",
                                        format!(
                                            "field `{}` type mismatch: expected `{}`, found `{}`",
                                            field.name.name, expected, ft
                                        ),
                                        to_diag_span(field.span),
                                    ));
                                }
                            }
                        }
                    }
                    ty
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0111",
                        format!("undefined type `{}`", name.name),
                        to_diag_span(name.span),
                    ));
                    Type::Error
                }
            }
            ExprKind::ArrayLit(elems) => {
                if elems.is_empty() {
                    Type::Array {
                        elem: Box::new(Type::Error),
                        len: 0,
                    }
                } else {
                    let first_type = self.check_expr(&elems[0]);
                    for elem in elems.iter().skip(1) {
                        let et = self.check_expr(elem);
                        if !et.is_error()
                            && !first_type.is_error()
                            && !self.env.types_match(&et, &first_type)
                        {
                            self.diagnostics.emit(Diagnostic::error(
                                "E0112",
                                format!(
                                    "array element type mismatch: expected `{}`, found `{}`",
                                    first_type, et
                                ),
                                to_diag_span(elem.span),
                            ));
                        }
                    }
                    Type::Array {
                        elem: Box::new(first_type),
                        len: elems.len(),
                    }
                }
            }
            ExprKind::Col(_) => {
                // Data DSL column reference — returns an opaque Expr type
                Type::Unresolved("ColumnExpr".into())
            }
            ExprKind::Lambda { params, body } => {
                self.env.push_scope();
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| {
                        let ty = self.resolve_type_expr(&p.ty);
                        self.env.define_var(&p.name.name, ty.clone());
                        ty
                    })
                    .collect();
                let ret_type = self.check_expr(body);
                self.env.pop_scope();
                Type::Fn {
                    params: param_types,
                    ret: Box::new(ret_type),
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                let scrut_ty = self.check_expr(scrutinee);
                // Check exhaustiveness for enum types
                self.check_match_exhaustiveness(&scrut_ty, arms, expr.span);
                // Type of a match = type of the first arm body (simplified)
                if arms.is_empty() {
                    Type::Void
                } else {
                    let first = self.check_expr(&arms[0].body);
                    for arm in arms.iter().skip(1) {
                        self.check_expr(&arm.body);
                    }
                    first
                }
            }
            ExprKind::TupleLit(elems) => {
                let types: Vec<Type> = elems.iter().map(|e| self.check_expr(e)).collect();
                Type::Tuple(types)
            }
            ExprKind::Try(inner) => {
                // `expr?` — inner must be Result<T, E>, expression type is T
                let inner_ty = self.check_expr(inner);
                match &inner_ty {
                    Type::Enum(e) if e.name == "Result" => {
                        // Extract T from Result<T, E>
                        if let Some(ok_variant) = e.variants.iter().find(|v| v.name == "Ok") {
                            if let Some(t) = ok_variant.fields.first() {
                                t.clone()
                            } else {
                                Type::Void
                            }
                        } else {
                            Type::Error
                        }
                    }
                    Type::Error => Type::Error,
                    _ => {
                        self.diagnostics.emit(
                            Diagnostic::error(
                                "E0120",
                                format!(
                                    "`?` operator requires `Result<T, E>`, found `{}`",
                                    inner_ty
                                ),
                                to_diag_span(expr.span),
                            )
                            .with_hint("the `?` operator can only be applied to Result types"),
                        );
                        Type::Error
                    }
                }
            }
            ExprKind::VariantLit {
                enum_name,
                variant,
                fields,
            } => {
                // Type-check variant literal by looking up enum and variant
                let field_types: Vec<Type> = fields.iter().map(|f| self.check_expr(f)).collect();

                // Try to find the enum via explicit enum_name or variant lookup
                let enum_name_str = enum_name.as_ref().map(|id| id.name.as_str());
                let found_enum = if let Some(name) = enum_name_str {
                    self.env.resolve_type_name(name)
                } else {
                    // Look up by variant name
                    self.env
                        .lookup_variant_enum(&variant.name)
                        .map(|e| Type::Enum(e.clone()))
                };

                if let Some(Type::Enum(et)) = found_enum {
                    if let Some(v) = et.variants.iter().find(|v| v.name == variant.name) {
                        if v.fields.len() != field_types.len() {
                            self.diagnostics.emit(Diagnostic::error(
                                "E0121",
                                format!(
                                    "variant `{}` expects {} field(s), found {}",
                                    variant.name,
                                    v.fields.len(),
                                    field_types.len()
                                ),
                                to_diag_span(expr.span),
                            ));
                            Type::Error
                        } else {
                            Type::Enum(et)
                        }
                    } else {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0122",
                            format!("no variant `{}` in enum `{}`", variant.name, et.name),
                            to_diag_span(variant.span),
                        ));
                        Type::Error
                    }
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0122",
                        format!("unknown enum variant `{}`", variant.name),
                        to_diag_span(variant.span),
                    ));
                    Type::Error
                }
            }
            ExprKind::CompoundAssign { op, target, value } => {
                let tt = self.check_expr(target);
                let vt = self.check_expr(value);
                // Type-check the binary operation (target op value)
                let _result_ty = self.check_binary_op(*op, &tt, &vt, expr.span);
                // Check mutability like a regular assignment
                if let ExprKind::Ident(id) = &target.kind {
                    if let Some((_ty, is_mut)) = self.env.lookup_var_entry(&id.name) {
                        if !is_mut {
                            self.diagnostics.emit(
                                Diagnostic::error(
                                    "E0150",
                                    format!(
                                        "cannot assign to immutable variable `{}`",
                                        id.name
                                    ),
                                    to_diag_span(expr.span),
                                )
                                .with_hint(format!(
                                    "consider making `{}` mutable: `let mut {} = ...`",
                                    id.name, id.name
                                )),
                            );
                        }
                    }
                }
                Type::Void
            }
            ExprKind::IfExpr { condition, then_block, else_branch } => {
                let cond_type = self.check_expr(condition);
                if !cond_type.is_error() && cond_type != Type::Bool {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0105",
                        format!("if condition must be `bool`, found `{}`", cond_type),
                        to_diag_span(condition.span),
                    ));
                }
                let then_ty = self.check_block(then_block);
                if let Some(ref eb) = else_branch {
                    match eb {
                        ElseBranch::ElseIf(elif) => {
                            self.check_if(elif);
                        }
                        ElseBranch::Else(block) => {
                            self.check_block(block);
                        }
                    }
                }
                then_ty
            }
        }
    }

    fn check_binary_op(
        &mut self,
        op: BinOp,
        lt: &Type,
        rt: &Type,
        span: cjc_ast::Span,
    ) -> Type {
        if lt.is_error() || rt.is_error() {
            return Type::Error;
        }

        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                if lt.is_numeric() && self.env.types_match(lt, rt) {
                    lt.clone()
                } else if lt.is_numeric() && rt.is_numeric() {
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0101",
                            format!(
                                "cannot apply `{}` to `{}` and `{}`",
                                op, lt, rt
                            ),
                            to_diag_span(span),
                        )
                        .with_hint(format!(
                            "both operands must have the same numeric type; \
                             consider casting one to `{}`",
                            lt
                        )),
                    );
                    Type::Error
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0101",
                        format!("cannot apply `{}` to `{}` and `{}`", op, lt, rt),
                        to_diag_span(span),
                    ));
                    Type::Error
                }
            }
            BinOp::Eq | BinOp::Ne => {
                if self.env.types_match(lt, rt) {
                    Type::Bool
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0102",
                        format!(
                            "cannot compare `{}` and `{}` for equality",
                            lt, rt
                        ),
                        to_diag_span(span),
                    ));
                    Type::Error
                }
            }
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                if lt.is_numeric() && self.env.types_match(lt, rt) {
                    Type::Bool
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0102",
                        format!("cannot compare `{}` and `{}`", lt, rt),
                        to_diag_span(span),
                    ));
                    Type::Error
                }
            }
            BinOp::And | BinOp::Or => {
                if *lt == Type::Bool && *rt == Type::Bool {
                    Type::Bool
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0101",
                        format!(
                            "`{}` requires `bool` operands, found `{}` and `{}`",
                            op, lt, rt
                        ),
                        to_diag_span(span),
                    ));
                    Type::Error
                }
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                if lt.is_int() && self.env.types_match(lt, rt) {
                    lt.clone()
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0101",
                        format!(
                            "cannot apply `{}` to `{}` and `{}`; bitwise ops require integer operands",
                            op, lt, rt
                        ),
                        to_diag_span(span),
                    ));
                    Type::Error
                }
            }
            BinOp::Match | BinOp::NotMatch => {
                // LHS should be ByteSlice/String/StrView, RHS should be Regex
                if *rt == Type::Regex {
                    Type::Bool
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0101",
                        format!(
                            "`{}` requires a Regex on the right side, found `{}`",
                            op, rt
                        ),
                        to_diag_span(span),
                    ));
                    Type::Error
                }
            }
        }
    }

    fn check_fn_call(&mut self, name: &str, args: &[CallArg], span: cjc_ast::Span) -> Type {
        // Check argument types
        let arg_types: Vec<Type> = args.iter().map(|a| self.check_expr(&a.value)).collect();

        // Look up function
        if let Some(sigs) = self.env.fn_sigs.get(name).cloned() {
            // Find matching signature with type unification
            for sig in &sigs {
                if sig.params.len() != arg_types.len() {
                    continue;
                }

                // If function has type parameters, attempt unification
                if !sig.type_params.is_empty() {
                    let mut subst = TypeSubst::new();

                    // Create fresh type variables for each type parameter
                    let mut param_var_map: HashMap<String, TypeVarId> = HashMap::new();
                    for (tp_name, _bounds) in &sig.type_params {
                        let var_id = self.env.fresh_var();
                        param_var_map.insert(tp_name.clone(), var_id);
                    }

                    // Replace type parameter names in param types with fresh vars
                    let sig_param_types: Vec<Type> = sig
                        .params
                        .iter()
                        .map(|(_, ty)| self.substitute_type_params(ty, &param_var_map))
                        .collect();
                    let sig_ret = self.substitute_type_params(&sig.ret, &param_var_map);

                    // Try to unify each argument type with the parameter type
                    let mut ok = true;
                    for (arg_ty, param_ty) in arg_types.iter().zip(sig_param_types.iter()) {
                        if unify(arg_ty, param_ty, &mut subst).is_err() {
                            ok = false;
                            break;
                        }
                    }

                    if ok {
                        // P0-2: Check that all inferred types satisfy their trait bounds.
                        // Emits E0300: trait bound not satisfied (call-site enforcement).
                        let mut bounds_ok = true;
                        for (tp_name, bounds) in &sig.type_params {
                            if let Some(&var_id) = param_var_map.get(tp_name) {
                                let inferred = apply_subst(&Type::Var(var_id), &subst);
                                if !matches!(inferred, Type::Var(_)) {
                                    if !self.env.check_bounds(&inferred, bounds) {
                                        self.diagnostics.emit(
                                            Diagnostic::error(
                                                "E0300",
                                                format!(
                                                    "trait bound not satisfied: type `{}` does not implement `{}` (required by type parameter `{}`)",
                                                    inferred,
                                                    bounds.join(" + "),
                                                    tp_name
                                                ),
                                                to_diag_span(span),
                                            )
                                            .with_hint(format!(
                                                "type parameter `{}` requires: `{}`",
                                                tp_name,
                                                bounds.join(" + ")
                                            )),
                                        );
                                        bounds_ok = false;
                                    }
                                }
                            }
                        }
                        if bounds_ok {
                            return apply_subst(&sig_ret, &subst);
                        }
                    }
                } else {
                    // Non-generic function: verify argument types with span-aware unification.
                    let mut all_ok = true;
                    for ((_, param_ty), arg_ty) in sig.params.iter().zip(arg_types.iter()) {
                        let mut subst_local = TypeSubst::new();
                        let result = unify_spanned(
                            arg_ty,
                            param_ty,
                            &mut subst_local,
                            to_diag_span(span),
                            &mut self.diagnostics,
                        );
                        if result.is_error() && !arg_ty.is_error() {
                            all_ok = false;
                        }
                    }
                    if all_ok || arg_types.iter().any(|t| t.is_error()) {
                        return sig.ret.clone();
                    }
                    return Type::Error;
                }
            }
            self.diagnostics.emit(
                Diagnostic::error(
                    "E0113",
                    format!(
                        "no matching overload for `{}` with {} argument(s)",
                        name,
                        arg_types.len()
                    ),
                    to_diag_span(span),
                )
                .with_hint(format!(
                    "available overloads: {}",
                    sigs.iter()
                        .map(|s| format!("{}({})", s.name, s.params.len()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
            );
            Type::Error
        } else {
            // Built-in function check
            match name {
                "print" => Type::Void,
                "mean" | "sum" | "max" | "min" | "count" => {
                    // Data DSL aggregation functions
                    Type::F64
                }
                "bf16_to_f32" => Type::F32,
                "f32_to_bf16" => Type::Bf16,
                "f16_to_f64" => Type::F64,
                "f64_to_f16" => Type::F16,
                "f16_to_f32" => Type::F32,
                "f32_to_f16" => Type::F16,
                _ => {
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0114",
                            format!("undefined function `{}`", name),
                            to_diag_span(span),
                        )
                        .with_hint("check for typos or define the function"),
                    );
                    Type::Error
                }
            }
        }
    }

    /// Replace type parameter names (like "T") with fresh type variables in a type.
    fn substitute_type_params(&self, ty: &Type, map: &HashMap<String, TypeVarId>) -> Type {
        match ty {
            Type::Unresolved(name) => {
                if let Some(&var_id) = map.get(name) {
                    Type::Var(var_id)
                } else {
                    ty.clone()
                }
            }
            Type::Tensor { elem, shape } => Type::Tensor {
                elem: Box::new(self.substitute_type_params(elem, map)),
                shape: shape.clone(),
            },
            Type::Buffer { elem } => Type::Buffer {
                elem: Box::new(self.substitute_type_params(elem, map)),
            },
            Type::Array { elem, len } => Type::Array {
                elem: Box::new(self.substitute_type_params(elem, map)),
                len: *len,
            },
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.substitute_type_params(e, map)).collect())
            }
            Type::Fn { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.substitute_type_params(p, map)).collect(),
                ret: Box::new(self.substitute_type_params(ret, map)),
            },
            Type::Enum(e) => Type::Enum(EnumType {
                name: e.name.clone(),
                type_params: e.type_params.clone(),
                variants: e
                    .variants
                    .iter()
                    .map(|v| EnumVariant {
                        name: v.name.clone(),
                        fields: v
                            .fields
                            .iter()
                            .map(|f| self.substitute_type_params(f, map))
                            .collect(),
                    })
                    .collect(),
            }),
            _ => ty.clone(),
        }
    }

    fn resolve_type_expr(&self, ty: &TypeExpr) -> Type {
        match &ty.kind {
            TypeExprKind::Named { name, args } => {
                let base_name = &name.name;
                match base_name.as_str() {
                    "i32" => Type::I32,
                    "i64" => Type::I64,
                    "u8" => Type::U8,
                    "f32" => Type::F32,
                    "f64" => Type::F64,
                    "bool" => Type::Bool,
                    "String" => Type::Str,
                    "Bytes" => Type::Bytes,
                    "ByteSlice" => Type::ByteSlice,
                    "StrView" => Type::StrView,
                    "Regex" => Type::Regex,
                    "Tensor" => {
                        let elem = if !args.is_empty() {
                            if let TypeArg::Type(t) = &args[0] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::F64
                            }
                        } else {
                            Type::F64
                        };
                        let shape = if args.len() > 1 {
                            if let TypeArg::Shape(dims) = &args[1] {
                                Some(
                                    dims.iter()
                                        .map(|d| match d {
                                            cjc_ast::ShapeDim::Name(n) => {
                                                ShapeDim::Symbolic(n.name.clone())
                                            }
                                            cjc_ast::ShapeDim::Lit(v) => {
                                                ShapeDim::Known(*v as usize)
                                            }
                                        })
                                        .collect(),
                                )
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        Type::Tensor {
                            elem: Box::new(elem),
                            shape,
                        }
                    }
                    "Buffer" => {
                        let elem = if !args.is_empty() {
                            if let TypeArg::Type(t) = &args[0] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::F64
                            }
                        } else {
                            Type::F64
                        };
                        Type::Buffer {
                            elem: Box::new(elem),
                        }
                    }
                    "bf16" => Type::Bf16,
                    "f16" => Type::F16,
                    "Complex" => Type::Complex,
                    "Range" => {
                        let elem = if !args.is_empty() {
                            if let TypeArg::Type(t) = &args[0] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::I64
                            }
                        } else {
                            Type::I64
                        };
                        Type::Range {
                            elem: Box::new(elem),
                        }
                    }
                    "Slice" => {
                        let elem = if !args.is_empty() {
                            if let TypeArg::Type(t) = &args[0] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::F64
                            }
                        } else {
                            Type::F64
                        };
                        Type::Slice {
                            elem: Box::new(elem),
                        }
                    }
                    "Option" => {
                        let inner = if !args.is_empty() {
                            if let TypeArg::Type(t) = &args[0] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::Unresolved("T".into())
                            }
                        } else {
                            Type::Unresolved("T".into())
                        };
                        Type::Enum(EnumType {
                            name: "Option".into(),
                            type_params: vec!["T".into()],
                            variants: vec![
                                EnumVariant { name: "Some".into(), fields: vec![inner] },
                                EnumVariant { name: "None".into(), fields: vec![] },
                            ],
                        })
                    }
                    "Result" => {
                        let ok_ty = if !args.is_empty() {
                            if let TypeArg::Type(t) = &args[0] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::Unresolved("T".into())
                            }
                        } else {
                            Type::Unresolved("T".into())
                        };
                        let err_ty = if args.len() > 1 {
                            if let TypeArg::Type(t) = &args[1] {
                                self.resolve_type_expr(t)
                            } else {
                                Type::Unresolved("E".into())
                            }
                        } else {
                            Type::Unresolved("E".into())
                        };
                        Type::Enum(EnumType {
                            name: "Result".into(),
                            type_params: vec!["T".into(), "E".into()],
                            variants: vec![
                                EnumVariant { name: "Ok".into(), fields: vec![ok_ty] },
                                EnumVariant { name: "Err".into(), fields: vec![err_ty] },
                            ],
                        })
                    }
                    _ => {
                        if let Some(ty) = self.env.type_defs.get(base_name) {
                            ty.clone()
                        } else {
                            Type::Unresolved(base_name.clone())
                        }
                    }
                }
            }
            TypeExprKind::Array { elem, size } => {
                let elem_ty = self.resolve_type_expr(elem);
                let len = if let ExprKind::IntLit(v) = size.kind {
                    v as usize
                } else {
                    0
                };
                Type::Array {
                    elem: Box::new(elem_ty),
                    len,
                }
            }
            TypeExprKind::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.resolve_type_expr(e)).collect())
            }
            TypeExprKind::Fn { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.resolve_type_expr(p)).collect(),
                ret: Box::new(self.resolve_type_expr(ret)),
            },
            TypeExprKind::ShapeLit(_) => Type::Error, // Shape literals only valid in type args
        }
    }

    fn type_expr_name(&self, ty: &TypeExpr) -> String {
        match &ty.kind {
            TypeExprKind::Named { name, .. } => name.name.clone(),
            _ => "<unknown>".into(),
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_types() {
        let env = TypeEnv::new();
        assert_eq!(env.resolve_type_name("i32"), Some(Type::I32));
        assert_eq!(env.resolve_type_name("f64"), Some(Type::F64));
        assert_eq!(env.resolve_type_name("bool"), Some(Type::Bool));
    }

    #[test]
    fn test_trait_satisfaction() {
        let env = TypeEnv::new();
        assert!(env.satisfies_trait(&Type::I32, "Numeric"));
        assert!(env.satisfies_trait(&Type::I32, "Int"));
        assert!(!env.satisfies_trait(&Type::I32, "Float"));

        assert!(env.satisfies_trait(&Type::F64, "Numeric"));
        assert!(env.satisfies_trait(&Type::F64, "Float"));
        assert!(env.satisfies_trait(&Type::F64, "Differentiable"));
        assert!(!env.satisfies_trait(&Type::F64, "Int"));

        assert!(!env.satisfies_trait(&Type::Bool, "Numeric"));
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", Type::I32), "i32");
        assert_eq!(format!("{}", Type::F64), "f64");
        assert_eq!(
            format!(
                "{}",
                Type::Tensor {
                    elem: Box::new(Type::F32),
                    shape: Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)])
                }
            ),
            "Tensor<f32, [3, 4]>"
        );
    }

    #[test]
    fn test_matmul_shape_check() {
        let env = TypeEnv::new();

        // Valid: [3, 4] x [4, 5] -> [3, 5]
        let result = env.check_matmul_shapes(
            &Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]),
            &Some(vec![ShapeDim::Known(4), ShapeDim::Known(5)]),
        );
        assert!(result.is_ok());
        let shape = result.unwrap().unwrap();
        assert_eq!(shape, vec![ShapeDim::Known(3), ShapeDim::Known(5)]);

        // Invalid: [3, 4] x [5, 6] -> error
        let result = env.check_matmul_shapes(
            &Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]),
            &Some(vec![ShapeDim::Known(5), ShapeDim::Known(6)]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_scope() {
        let mut env = TypeEnv::new();
        env.define_var("x", Type::I32);
        assert_eq!(env.lookup_var("x"), Some(&Type::I32));

        env.push_scope();
        env.define_var("y", Type::F64);
        assert_eq!(env.lookup_var("x"), Some(&Type::I32));
        assert_eq!(env.lookup_var("y"), Some(&Type::F64));

        env.pop_scope();
        assert_eq!(env.lookup_var("x"), Some(&Type::I32));
        assert_eq!(env.lookup_var("y"), None);
    }

    #[test]
    fn test_types_match() {
        let env = TypeEnv::new();
        assert!(env.types_match(&Type::I32, &Type::I32));
        assert!(!env.types_match(&Type::I32, &Type::I64));
        assert!(env.types_match(&Type::Error, &Type::I32)); // Error matches everything
    }

    // ── Unification Tests ────────────────────────────────────────

    #[test]
    fn test_unify_concrete_types() {
        let mut subst = TypeSubst::new();
        assert_eq!(unify(&Type::I32, &Type::I32, &mut subst).unwrap(), Type::I32);
        assert_eq!(unify(&Type::F64, &Type::F64, &mut subst).unwrap(), Type::F64);
        assert_eq!(unify(&Type::Bool, &Type::Bool, &mut subst).unwrap(), Type::Bool);
        assert_eq!(unify(&Type::Str, &Type::Str, &mut subst).unwrap(), Type::Str);
    }

    #[test]
    fn test_unify_var_binds() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        let result = unify(&Type::Var(var0), &Type::F64, &mut subst).unwrap();
        assert_eq!(result, Type::F64);
        assert_eq!(subst.get(&var0), Some(&Type::F64));
    }

    #[test]
    fn test_unify_var_right_side() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        let result = unify(&Type::I32, &Type::Var(var0), &mut subst).unwrap();
        assert_eq!(result, Type::I32);
        assert_eq!(subst.get(&var0), Some(&Type::I32));
    }

    #[test]
    fn test_unify_mismatch_fails() {
        let mut subst = TypeSubst::new();
        assert!(unify(&Type::I32, &Type::F64, &mut subst).is_err());
        assert!(unify(&Type::Bool, &Type::Str, &mut subst).is_err());
    }

    #[test]
    fn test_unify_tensor_elem() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        let a = Type::Tensor {
            elem: Box::new(Type::Var(var0)),
            shape: None,
        };
        let b = Type::Tensor {
            elem: Box::new(Type::F32),
            shape: None,
        };
        let result = unify(&a, &b, &mut subst).unwrap();
        assert_eq!(
            result,
            Type::Tensor {
                elem: Box::new(Type::F32),
                shape: None,
            }
        );
        assert_eq!(subst.get(&var0), Some(&Type::F32));
    }

    #[test]
    fn test_unify_fn_types() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        let a = Type::Fn {
            params: vec![Type::Var(var0)],
            ret: Box::new(Type::Var(var0)),
        };
        let b = Type::Fn {
            params: vec![Type::I64],
            ret: Box::new(Type::I64),
        };
        let result = unify(&a, &b, &mut subst).unwrap();
        assert_eq!(
            result,
            Type::Fn {
                params: vec![Type::I64],
                ret: Box::new(Type::I64),
            }
        );
    }

    #[test]
    fn test_unify_error_recovery() {
        let mut subst = TypeSubst::new();
        assert_eq!(unify(&Type::Error, &Type::I32, &mut subst).unwrap(), Type::I32);
        assert_eq!(unify(&Type::F64, &Type::Error, &mut subst).unwrap(), Type::F64);
    }

    #[test]
    fn test_apply_subst_basic() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        subst.insert(var0, Type::F32);
        assert_eq!(apply_subst(&Type::Var(var0), &subst), Type::F32);
        assert_eq!(apply_subst(&Type::I32, &subst), Type::I32);
    }

    #[test]
    fn test_apply_subst_nested() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        subst.insert(var0, Type::F32);
        let tensor = Type::Tensor {
            elem: Box::new(Type::Var(var0)),
            shape: None,
        };
        assert_eq!(
            apply_subst(&tensor, &subst),
            Type::Tensor {
                elem: Box::new(Type::F32),
                shape: None,
            }
        );
    }

    #[test]
    fn test_apply_subst_chained() {
        let mut subst = TypeSubst::new();
        let var0 = TypeVarId(0);
        let var1 = TypeVarId(1);
        subst.insert(var0, Type::Var(var1));
        subst.insert(var1, Type::I64);
        assert_eq!(apply_subst(&Type::Var(var0), &subst), Type::I64);
    }

    #[test]
    fn test_check_bounds_numeric() {
        let env = TypeEnv::new();
        assert!(env.check_bounds(&Type::I32, &["Numeric".into()]));
        assert!(env.check_bounds(&Type::F64, &["Numeric".into()]));
        assert!(!env.check_bounds(&Type::Bool, &["Numeric".into()]));
    }

    #[test]
    fn test_check_bounds_differentiable() {
        let env = TypeEnv::new();
        assert!(env.check_bounds(&Type::F64, &["Differentiable".into()]));
        assert!(env.check_bounds(&Type::F32, &["Differentiable".into()]));
        assert!(!env.check_bounds(&Type::I32, &["Differentiable".into()]));
    }

    #[test]
    fn test_check_bounds_multiple() {
        let env = TypeEnv::new();
        assert!(env.check_bounds(&Type::F64, &["Numeric".into(), "Float".into()]));
        assert!(!env.check_bounds(&Type::I32, &["Numeric".into(), "Float".into()]));
    }

    // ── Shape Unification Tests ────────────────────────────────

    #[test]
    fn test_unify_shape_known_match() {
        let mut subst = ShapeSubst::new();
        let result = unify_shape_dim(&ShapeDim::Known(3), &ShapeDim::Known(3), &mut subst);
        assert_eq!(result.unwrap(), ShapeDim::Known(3));
    }

    #[test]
    fn test_unify_shape_known_mismatch() {
        let mut subst = ShapeSubst::new();
        let result = unify_shape_dim(&ShapeDim::Known(3), &ShapeDim::Known(4), &mut subst);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mismatch"));
    }

    #[test]
    fn test_unify_shape_symbolic_binds() {
        let mut subst = ShapeSubst::new();
        let result = unify_shape_dim(
            &ShapeDim::Symbolic("N".into()),
            &ShapeDim::Known(5),
            &mut subst,
        );
        assert_eq!(result.unwrap(), ShapeDim::Known(5));
        assert_eq!(subst.get("N"), Some(&5));
    }

    #[test]
    fn test_unify_shape_symbolic_conflict() {
        let mut subst = ShapeSubst::new();
        subst.insert("N".into(), 3);
        let result = unify_shape_dim(
            &ShapeDim::Symbolic("N".into()),
            &ShapeDim::Known(5),
            &mut subst,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already bound"));
    }

    #[test]
    fn test_unify_shapes_rank_mismatch() {
        let mut subst = ShapeSubst::new();
        let a = vec![ShapeDim::Known(3), ShapeDim::Known(4)];
        let b = vec![ShapeDim::Known(3)];
        let result = unify_shapes(&a, &b, &mut subst);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("rank mismatch"));
    }

    // ── Broadcasting Tests ─────────────────────────────────────

    #[test]
    fn test_broadcast_same_shape() {
        let a = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
        let b = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
        let result = broadcast_shapes(&a, &b).unwrap().unwrap();
        assert_eq!(result, vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
    }

    #[test]
    fn test_broadcast_scalar_to_matrix() {
        let a = Some(vec![ShapeDim::Known(1)]);
        let b = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
        let result = broadcast_shapes(&a, &b).unwrap().unwrap();
        assert_eq!(result, vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
    }

    #[test]
    fn test_broadcast_row_to_matrix() {
        let a = Some(vec![ShapeDim::Known(1), ShapeDim::Known(4)]);
        let b = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
        let result = broadcast_shapes(&a, &b).unwrap().unwrap();
        assert_eq!(result, vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
    }

    #[test]
    fn test_broadcast_col_to_matrix() {
        let a = Some(vec![ShapeDim::Known(3), ShapeDim::Known(1)]);
        let b = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
        let result = broadcast_shapes(&a, &b).unwrap().unwrap();
        assert_eq!(result, vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        let a = Some(vec![ShapeDim::Known(3), ShapeDim::Known(5)]);
        let b = Some(vec![ShapeDim::Known(3), ShapeDim::Known(4)]);
        let result = broadcast_shapes(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_symbolic_with_one() {
        let a = Some(vec![ShapeDim::Symbolic("N".into())]);
        let b = Some(vec![ShapeDim::Known(1)]);
        let result = broadcast_shapes(&a, &b).unwrap().unwrap();
        assert_eq!(result, vec![ShapeDim::Symbolic("N".into())]);
    }

    #[test]
    fn test_broadcast_none_passthrough() {
        assert_eq!(broadcast_shapes(&None, &None).unwrap(), None);
        let a = Some(vec![ShapeDim::Known(3)]);
        assert_eq!(broadcast_shapes(&a, &None).unwrap(), None);
    }

    #[test]
    fn test_value_vs_gc_type() {
        assert!(Type::I32.is_value_type());
        assert!(Type::F64.is_value_type());
        assert!(Type::Struct(StructType {
            name: "Foo".into(),
            type_params: vec![],
            fields: vec![]
        })
        .is_value_type());

        assert!(Type::Class(ClassType {
            name: "Bar".into(),
            type_params: vec![],
            fields: vec![]
        })
        .is_gc_type());

        assert!(!Type::I32.is_gc_type());
        assert!(!Type::Class(ClassType {
            name: "Bar".into(),
            type_params: vec![],
            fields: vec![]
        })
        .is_value_type());
    }
}
