# CJC Language — Syntax & Type System Reference

> **Audit revision:** Post-Hardening (Phase 5) — 2025 tests passing, 0 failures
> **Auditor role:** Role 1 (Syntax), Role 2 (Types), Role 5 (Determinism Guardian)
> **Status:** READ-ONLY audit — no source modifications

---

## Table of Contents

1. [Lexical Structure](#1-lexical-structure)
2. [Tokens and Keywords](#2-tokens-and-keywords)
3. [Operators](#3-operators)
4. [Literals](#4-literals)
5. [Top-Level Declarations](#5-top-level-declarations)
6. [Statements](#6-statements)
7. [Expressions](#7-expressions)
8. [Patterns](#8-patterns)
9. [Type System](#9-type-system)
10. [Generics and Type Variables](#10-generics-and-type-variables)
11. [Type Unification](#11-type-unification)
12. [Span-Aware Diagnostics](#12-span-aware-diagnostics)
13. [Type Checker](#13-type-checker)
14. [Match Exhaustiveness](#14-match-exhaustiveness)
15. [Trait Resolution](#15-trait-resolution)
16. [Error Codes](#16-error-codes)

---

## 1. Lexical Structure

### Source Encoding
CJC source files are ASCII/UTF-8. The lexer operates on raw bytes for performance.

### Comments
```cjc
// Single-line comment

/* Block comment */

/* Nested /* block */ comment supported — depth-tracked */
```

Block comments are **nested** (depth-counted), matching modern language conventions.

### Whitespace
Whitespace (spaces, tabs, newlines, carriage returns) is ignored between tokens. There is no significant-whitespace rule (unlike Python or Haskell).

---

## 2. Tokens and Keywords

**Keywords** (reserved, cannot be used as identifiers):

| Keyword  | Purpose                        |
|----------|--------------------------------|
| `struct` | Value-type aggregate definition |
| `class`  | GC reference-type definition   |
| `fn`     | Function definition             |
| `trait`  | Trait (interface) definition    |
| `impl`   | Implementation block           |
| `let`    | Variable binding               |
| `mut`    | Mutable binding marker         |
| `return` | Explicit return statement      |
| `if`     | Conditional                    |
| `else`   | Conditional alternative        |
| `while`  | Loop                           |
| `for`    | Range/collection iteration     |
| `in`     | Iterator keyword (for loops)   |
| `nogc`   | NoGC region marker             |
| `col`    | Column reference (data DSL)    |
| `import` | Module import                  |
| `as`     | Alias (import / as-cast)       |
| `sealed` | Sealed trait modifier          |
| `match`  | Pattern matching               |
| `enum`   | Algebraic data type definition |
| `true`   | Boolean literal                |
| `false`  | Boolean literal                |

**Special tokens:**

| Token | Description |
|-------|-------------|
| `_`   | Wildcard pattern / placeholder identifier |
| `..`  | Range operator |
| `\|>`  | Pipe operator |
| `\|`   | Lambda parameter delimiter |
| `[|`  | Tensor literal open |
| `\|]`  | Tensor literal close |
| `~=`  | Regex match operator |
| `!~`  | Regex non-match operator |
| `->`  | Return type arrow / function type arrow |
| `=>`  | Match arm fat arrow |

---

## 3. Operators

### Binary Operators (precedence high → low)

| Category      | Operators                    | Notes                        |
|---------------|------------------------------|------------------------------|
| Multiplicative | `*`, `/`, `%`               | Standard arithmetic          |
| Additive       | `+`, `-`                    | Standard arithmetic          |
| Comparison     | `<`, `>`, `<=`, `>=`        | Returns `bool`               |
| Equality       | `==`, `!=`                  | Returns `bool`               |
| Logical AND    | `&&`                        | Short-circuit evaluation     |
| Logical OR     | `\|\|`                       | Short-circuit evaluation     |
| Regex          | `~=`, `!~`                  | `String ~= Regex → bool`    |
| Pipe           | `\|>`                        | `expr |> f(args)` sugar      |
| Assignment     | `=`                         | Mutates variable             |

### Unary Operators

| Operator | Usage    | Types        |
|----------|----------|--------------|
| `-`      | Negation | `i32 i64 f32 f64 bf16` |
| `!`      | Logical NOT | `bool` |

### Postfix Operators

| Operator | Usage | Notes |
|----------|-------|-------|
| `?`      | Try operator | Desugars to `match` on `Result` |
| `[]`     | Index       | Single index into array/tensor |
| `[i, j]` | Multi-index | Two-dimensional tensor index |
| `.field` | Field access | Struct/class field |
| `.method(...)` | Method call | Dispatched on object type |

---

## 4. Literals

### Integer Literals
```cjc
42          // i64 (default)
1_000_000   // underscore separators allowed
100i64      // explicit i64 suffix
100i32      // explicit i32 suffix
```

### Float Literals
```cjc
3.14        // f64 (default)
2.5e10      // scientific notation
1.0f64      // explicit f64 suffix
1.0f32      // explicit f32 suffix
```

### Boolean Literals
```cjc
true
false
```

### String Literals
```cjc
"hello world"         // Standard string (escape processed)
r"raw \n no escape"   // Raw string (no escape processing)
r#"contains "quotes""#  // Raw string with hash delimiters
```

**Supported escape sequences in standard strings:**
`\n`, `\t`, `\r`, `\\`, `\"`, `\0`

### Byte String Literals
```cjc
b"hello"              // ByteSlice (Vec<u8>)
b"a\nb\x41"           // with hex escape \xNN
br"raw \n bytes"      // raw byte string
br#"raw with "quotes""#
```

### Byte Char Literals
```cjc
b'A'      // u8 value 65
b'\n'     // u8 value 10
b'\xff'   // u8 value 255
b'\\'     // u8 value 92
```

### Regex Literals
```cjc
/\d+/          // Regex literal
/hello/i       // case-insensitive flag
/pattern/igms  // multiple flags: i, g, m, s
```

**Supported flags:** `i` (case-insensitive), `g` (global), `m` (multiline), `s` (dotall), `x` (extended/whitespace-ignore)

> **Implementation note:** Regex literals use context-sensitive disambiguation — `/` is only lexed as regex start when not preceded by a value-producing token.

### Tensor Literals
```cjc
[| 1.0, 2.0, 3.0 |]             // 1D tensor, shape [3]
[| 1.0, 2.0; 3.0, 4.0 |]        // 2D tensor, shape [2,2]
```

### Array Literals
```cjc
[1, 2, 3]         // Array<i64>
[1.0, 2.0, 3.0]   // Array<f64>
```

### Tuple Literals
```cjc
(1, 2.0, true)    // Tuple(i64, f64, bool)
(x, y)            // Tuple of variables
```

---

## 5. Top-Level Declarations

### Function Declaration
```cjc
fn add(a: i64, b: i64) -> i64 {
    a + b
}

// Generic function with type bounds
fn dot<T: Float>(a: Tensor<T>, b: Tensor<T>) -> T {
    // ...
}

// NoGC function — guarantees no GC allocation
nogc fn fast_sum(data: ByteSlice) -> f64 {
    // ...
}
```

### Struct Declaration (value type)
```cjc
struct Point {
    x: f64,
    y: f64,
}

// Generic struct
struct Pair<A, B> {
    first: A,
    second: B,
}
```

### Class Declaration (GC reference type)
```cjc
class Node {
    value: i64,
    next: Node,  // self-referential via GC
}
```

### Enum Declaration (Algebraic Data Type)
```cjc
enum Direction {
    North,
    South,
    East,
    West,
}

// Enum with payload variants
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Enum with multiple fields per variant
enum Shape {
    Circle(f64),          // radius
    Rectangle(f64, f64),  // width, height
}
```

### Trait Declaration
```cjc
trait Area {
    fn area(self: Circle) -> f64;
}

// Trait with super-traits
trait Printable: Display + Debug {
    fn pretty_print(self: Self) -> String;
}

// Generic trait
trait Container<T> {
    fn get(self: Self, idx: i64) -> T;
    fn len(self: Self) -> i64;
}
```

### Impl Declaration
```cjc
// Bare impl (method implementations for a type)
impl Point {
    fn distance(p: Point, q: Point) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy)   // sqrt not shown
    }
}

// Impl with trait reference (AST-supported; parser has known limitation)
impl Circle : Area {
    fn area(self: Circle) -> f64 {
        3.14159 * self.radius * self.radius
    }
}
```

> **Parser limitation (documented):** `impl Trait for Type` syntax does not fully parse. Use `impl Type : Trait` or bare `impl Type`. This is a known open item tracked in the Roadmap.

### Import Declaration
```cjc
import std.math
import std.io as io
import my_module.utils
```

---

## 6. Statements

### Let Binding
```cjc
let x = 42;           // immutable binding, type inferred as i64
let y: f64 = 3.14;    // with explicit type annotation
let mut z = 0;        // mutable binding
z = z + 1;            // assignment to mutable
```

### Expression Statement
```cjc
foo(x);               // side-effecting call; result discarded
x + 1;                // valid but result is Void
```

### Return Statement
```cjc
return 42;
return;               // equivalent to return Void
```

### If Statement / If Expression
```cjc
if x > 0 {
    positive()
} else if x < 0 {
    negative()
} else {
    zero()
}

// If as expression (both branches must produce same type)
let result = if flag { 1 } else { 0 };
```

### While Loop
```cjc
while condition {
    // body
}

let mut i = 0;
while i < 10 {
    i = i + 1;
}
```

### For Loop
```cjc
// Range iteration (exclusive end)
for i in 0..10 {
    print(i);
}

// Collection iteration
for item in collection {
    process(item);
}
```

### NoGC Block
```cjc
nogc {
    // All operations here must be NoGC-safe
    // No GC allocation allowed
    let sum = 0.0;
    // ...
}
```

---

## 7. Expressions

### Block Expression
```cjc
let result = {
    let a = compute_a();
    let b = compute_b();
    a + b       // tail expression — block value
};
```

### Function Call
```cjc
foo(1, 2, 3)
obj.method(arg)
callee(named_arg: value)   // named arguments supported
```

### Struct Literal
```cjc
let p = Point { x: 1.0, y: 2.0 };
```

### Lambda / Closure
```cjc
|x: f64| x * 2.0
|a: i64, b: i64| a + b

// Closure capturing environment
let threshold = 0.5;
let filter = |x: f64| x > threshold;   // captures threshold
```

> **Implementation:** Closures are lambda-lifted during HIR→MIR lowering. Captured variables become leading parameters of a synthetic `__closure_N` function. At call site, `MakeClosure { fn_name, captures }` bundles the env.

### Match Expression
```cjc
match direction {
    North => 0,
    South => 180,
    East  => 90,
    West  => 270,
}

// With payload binding
match shape {
    Circle(r)        => 3.14159 * r * r,
    Rectangle(w, h)  => w * h,
}

// With wildcard
match value {
    Ok(v)  => v,
    _      => default,
}
```

### Pipe Expression
```cjc
data |> filter(predicate) |> map(transform) |> collect()
// Equivalent to: collect(map(transform(filter(data, predicate))))
```

### Try Expression
```cjc
let value = risky_operation()?;
// Desugars to: match risky_operation() { Ok(v) => v, Err(e) => return Err(e) }
```

### Variant Constructor
```cjc
Ok(42)
Err("something failed")
Circle(3.14)
None
```

---

## 8. Patterns

### Wildcard Pattern
```cjc
_    // matches anything, binds nothing
```

### Binding Pattern
```cjc
x    // matches anything, binds matched value to x
```

> **Important semantic note (hardening fix):** Bare identifier patterns (`Red`, `Green`) in match arms where the identifier names an enum variant are treated as **variant patterns** by the type checker's exhaustiveness analyzer. This was fixed in the Production Hardening phase — previously they were incorrectly treated as wildcard bindings.

### Literal Patterns
```cjc
42      // integer literal
3.14    // float literal
true    // boolean
"hello" // string
```

### Tuple Destructuring
```cjc
(a, b)          // binds both fields
(x, _)          // discard second
(_, _, z)       // discard first two
```

### Struct Destructuring
```cjc
Point { x, y }          // shorthand: binds x and y
Point { x: px, y: py }  // renames: binds px and py
```

### Enum Variant Pattern
```cjc
Circle(r)                    // binds radius
Rectangle(w, h)              // binds both fields
Ok(value)                    // binds inner value
None                         // unit variant
```

---

## 9. Type System

### Primitive Types

| CJC Type | Size | Description |
|----------|------|-------------|
| `i32`    | 32-bit | Signed integer |
| `i64`    | 64-bit | Signed integer (default) |
| `u8`     | 8-bit  | Unsigned byte |
| `f32`    | 32-bit | IEEE 754 single precision |
| `f64`    | 64-bit | IEEE 754 double precision (default) |
| `bf16`   | 16-bit | Brain float (ML-oriented) |
| `bool`   | 1-bit  | Boolean (`true`/`false`) |
| `String` | heap  | UTF-8 owned string |
| `void`   | —     | No value |

### Memory Layers (Three-Layer Model)

```
Layer 1: @nogc — Static borrow, zero allocation
  Types: i32, i64, u8, f32, f64, bf16, bool, void, ByteSlice, StrView

Layer 2: COW — Buffer<T>, deterministic RC allocation
  Types: Bytes, Buffer<T>, Tensor<T>, Array<T>, String, Struct, Enum

Layer 3: GC — Mark-sweep garbage collector
  Types: Class (reference types, self-referential)
```

### Compound Types

#### Tensor Type
```cjc
Tensor<f64>              // element type only
Tensor<f64, [M, N]>      // with symbolic shape
Tensor<f32, [3, 3]>      // with concrete shape
```

#### Buffer Type
```cjc
Buffer<f64>   // COW reference-counted buffer
Buffer<u8>    // byte buffer
```

#### Array Type
```cjc
[i64; 10]     // fixed-length array of i64
[f64; 256]    // fixed-length array of f64
```

#### Tuple Type
```cjc
(i64, f64)         // 2-tuple
(bool, String, i32) // 3-tuple
```

#### Function Type
```cjc
fn(i64, i64) -> i64
fn(f64) -> bool
fn() -> void
```

#### Map Type
```cjc
Map<String, i64>    // hash map
Map<i64, f64>       // integer-keyed
```

#### Sparse Tensor Type
```cjc
SparseTensor<f64>   // COO-format sparse tensor
```

#### Regex Type
```cjc
Regex    // compiled regex pattern (from /pattern/flags literal)
```

#### Bytes / ByteSlice / StrView
```cjc
Bytes      // owned byte buffer (Vec<u8> wrapper)
ByteSlice  // non-owning byte slice (zero-copy view)
StrView    // validated UTF-8 string view (zero-copy)
```

### User-Defined Types

#### Struct (value type — Layer 2)
```cjc
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}
```
Structs are stack-allocated value types. Assignment copies all fields.

#### Class (reference type — Layer 3)
```cjc
class TreeNode {
    value: i64,
    left: TreeNode,
    right: TreeNode,
}
```
Classes are GC-managed. Assignment copies the reference, not the data.

#### Enum (ADT — Layer 2)
```cjc
enum Option<T> {
    Some(T),
    None,
}
```
Enums are stack-allocated with a discriminant tag. Sum types with per-variant payload fields.

---

## 10. Generics and Type Variables

### Type Parameters
```cjc
fn identity<T>(x: T) -> T { x }

struct Wrapper<T> {
    value: T,
}

trait Container<T> {
    fn get(self: Self) -> T;
}
```

### Type Bounds
```cjc
fn sum<T: Numeric>(values: Tensor<T>) -> T { ... }
fn print_all<T: Display + Debug>(items: [T; N]) { ... }
```

### Shape Parameters
```cjc
fn matmul<T: Float, M: Shape, K: Shape, N: Shape>(
    a: Tensor<T, [M, K]>,
    b: Tensor<T, [K, N]>
) -> Tensor<T, [M, N]> { ... }
```

Shape dimensions can be:
- `Known(n)` — concrete integer (e.g., `[3, 3]`)
- `Symbolic(name)` — symbolic dimension variable (e.g., `M`, `N`, `batch`)

### Type Variable IDs
Internally, each type variable is assigned a `TypeVarId(usize)`. The type substitution map (`TypeSubst = HashMap<TypeVarId, Type>`) tracks type variable bindings during unification.

---

## 11. Type Unification

The CJC type checker uses **Hindley-Milner style unification** with an occurs check to prevent infinite types.

### `unify(a, b, subst) -> Result<Type, String>`
Core unification function. Returns the **most specific unified type** on success, or an error string on failure.

**Rules:**
- `Error` type unifies with everything (error recovery — prevents cascade errors)
- `Var(id)` unifies with any type by binding `id → T` in `subst` (occurs check applied)
- Identical primitives unify trivially
- `Tensor<A>` unifies with `Tensor<B>` if `A` unifies with `B`; shapes unified separately
- `Buffer<A>` unifies with `Buffer<B>` if elements unify
- `Array<A, L>` unifies with `Array<B, L>` if elements unify and lengths match
- `Tuple(...)` unifies pointwise if same arity
- `Fn(P...) -> R` unifies if arities match and params/return unify
- `Struct`, `Class`, `Enum` — nominal: same name = compatible, else fail

### `unify_spanned(a, b, subst, span, diag) -> Type`
**Span-aware wrapper** added in Production Hardening Phase 1. Instead of returning `Err(String)`, this:
1. Calls `unify(a, b, subst)`
2. On failure, emits `Diagnostic::error("E0100", msg, span).with_hint(...)` into `diag`
3. Returns `Type::Error` for downstream error recovery

This enables precise error pointing with source locations, without breaking existing call sites.

### Shape Unification
```
unify_shapes(sh1, sh2, shape_subst) -> Result<Vec<ShapeDim>, String>
```
Shape dimensions are unified separately from element types:
- `Known(M)` vs `Known(N)`: must be equal
- `Symbolic(name)` vs anything: binds name in `shape_subst`
- Symbolic vs Symbolic: unify symbolically

---

## 12. Span-Aware Diagnostics

### `Span` Type
```rust
pub struct Span {
    pub start: usize,  // byte offset in source
    pub end: usize,    // byte offset (exclusive)
}
```

Operations:
- `Span::new(start, end)` — construct
- `Span::merge(other)` — union of two spans (min start, max end)
- `Span::dummy()` — zero span for synthesized nodes

### `Diagnostic` Type
```rust
pub struct Diagnostic {
    pub severity: Severity,  // Error, Warning, Hint
    pub code: String,        // "E0001", "W0001", etc.
    pub message: String,
    pub span: Span,
    pub labels: Vec<Label>,
    pub hints: Vec<String>,
}
```

Builder methods: `.with_label(span, message)`, `.with_hint(hint)`

### `DiagnosticBag`
Collects multiple diagnostics. Key methods:
- `emit(diag)` — add a diagnostic
- `has_errors()` — true if any Error severity
- `error_count()` — count of Error severity
- `count()` — total count (all severities)
- `truncate(n)` — discard diagnostics past index n (used for backtracking in lexer)
- `render_all(source, filename)` — format all diagnostics with source context

### Rendered Format
```
error[E0100]: type mismatch: expected `i64`, found `f64`
  --> example.cjc:5:12
   |
 5 | let x: i64 = 3.14;
   |              ^^^^ expected `i64`, found `f64`
   |
   = hint: expected `i64`, found `f64`
```

---

## 13. Type Checker

**`TypeChecker`** (in `cjc_types`) performs semantic analysis over a parsed AST Program.

### Key Fields
```rust
pub struct TypeChecker {
    pub env: TypeEnv,           // type environment
    pub diagnostics: DiagnosticBag,
    next_var: TypeVarId,
    subst: TypeSubst,
}

pub struct TypeEnv {
    pub types: HashMap<String, Type>,      // named types (struct, class, enum)
    pub functions: HashMap<String, Type>,  // function signatures
    pub trait_defs: HashMap<String, TraitDef>, // registered traits
    pub trait_impls: Vec<TraitImpl>,       // registered trait implementations
}
```

### Entry Point
```rust
checker.check_program(&prog)
```
Iterates all declarations, calling:
- `check_fn()` — type-check function body
- `check_decl()` — dispatches on DeclKind
- `check_impl()` — trait resolution enforcement *(added in hardening)*

### `check_impl()` — Trait Resolution Enforcement
Validates an `impl` block:
- **E0200**: Trait referenced in `impl Type : Trait` is undefined in `env.trait_defs`
- **E0201**: Duplicate impl detected (same type + same trait already registered)
- **E0202**: Method required by trait but missing from impl body
- **Type-checks all method bodies** regardless of trait reference

### Builtin Function Types
The type checker pre-registers builtin types for: `print`, `matmul`, `zeros`, `rand_tensor`, `len`, `assert`, `sqrt`, `abs`, `floor`, `ceil`, `min`, `max`, `sin`, `cos`, `exp`, `log`.

---

## 14. Match Exhaustiveness

**Error code: `E0130`**

The type checker enforces exhaustive enum matches as **compile-time errors** (not warnings).

### Algorithm

1. Collect all variant names for the scrutinee enum type
2. Walk match arms:
   - `PatternKind::Wildcard` → sets `has_wildcard = true` (covers all)
   - `PatternKind::Binding(name)`:
     - If `name` is a known enum variant → add to `covered_variants`
     - Otherwise (true binding) → sets `has_wildcard = true`
   - `PatternKind::Variant { variant }` → add variant name to `covered_variants`
3. If `has_wildcard` is true → exhaustive (no error)
4. Otherwise compute `uncovered = all_variants - covered_variants`
5. If `uncovered` is non-empty → emit `Diagnostic::error("E0130", ..., span)`

### Example
```cjc
enum Color { Red, Green, Blue, }

// ERROR: E0130 — missing variant Blue
fn name(c: Color) -> i64 {
    match c {
        Red   => 1,
        Green => 2,
    }
}

// OK: wildcard covers Blue
fn name2(c: Color) -> i64 {
    match c {
        Red => 1,
        _   => 0,  // covers Green and Blue
    }
}
```

### Integration
- `type_check_program(prog) -> Result<(), MirExecError>` — standalone gate that returns `Err(TypeErrors(Vec<String>))` if any errors
- `run_program_type_checked(prog, seed) -> ...` — gated pipeline (runs type checker before execution)
- `run_program(prog, seed)` — **not** gated (backward compatible, allows programs with unknown builtins)

---

## 15. Trait Resolution

### Trait Definition
```rust
pub struct TraitDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub methods: Vec<TraitMethodSig>,
}

pub struct TraitMethodSig {
    pub name: String,
    pub param_types: Vec<Type>,
    pub return_type: Type,
}
```

Registered in `TypeEnv::trait_defs` when `DeclKind::Trait` is processed.

### Impl Checking (E0200, E0201, E0202)
When `DeclKind::Impl(i)` is processed:

1. If `i.trait_ref` is `Some(tr)`:
   - Look up `tr.name` in `env.trait_defs` → missing = **E0200** (undefined trait)
   - Check if `(target_type, trait_name)` already registered → **E0201** (duplicate impl)
   - For each method in trait → check impl has matching method → missing = **E0202** (missing method)
2. Type-check all method bodies (regardless of trait_ref)
3. Register in `env.trait_impls`

---

## 16. Error Codes

### Lexer Errors
| Code   | Description |
|--------|-------------|
| E0002  | Unexpected character |
| E0003  | Unterminated literal (string, byte string, char, regex) |
| E0004  | Unknown escape sequence |
| E0005  | Invalid numeric suffix |
| E0010  | Expected `"` after `br` and hash delimiters |
| E0011  | Unterminated or malformed regex literal |

### Parser Errors
| Code   | Description |
|--------|-------------|
| E0001  | Unexpected token / parse error |
| E0020  | Expected token (e.g., expected `:` in field) |
| E0021  | Expected expression |
| E0022  | Unclosed delimiter |

### Type Checker Errors
| Code   | Description |
|--------|-------------|
| E0100  | Type mismatch (from `unify_spanned`) |
| E0101  | Return type mismatch |
| E0103  | Type error in expression |
| E0130  | Non-exhaustive match (missing enum variants) |
| E0200  | Undefined trait in impl |
| E0201  | Duplicate impl for the same type+trait |
| E0202  | Missing required trait method in impl |

---

*Generated by the CJC Full Repo Audit (Read-Only, Post-Hardening Phase 5)*
*Total passing tests at time of audit: **2025***
