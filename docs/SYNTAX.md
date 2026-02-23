# CJC Language Syntax Reference

This document describes the syntax of CJC as implemented in the repository source code and verified against the test suite. CJC is a scientific computing language with a Rust-flavored syntax, featuring tensors, automatic differentiation, a data DSL, and a garbage-collector/no-GC split type system.

Every syntax claim below is grounded in the lexer, parser, AST, HIR, or test code. Claims are labeled:

- **Proven** — exercised by at least one test.
- **Observed** — present in parser/lexer code but no dedicated test found.
- **Inferred** — deduced from code structure; may not be directly tested.

---

## 1. Repo Syntax Map

| Layer | Crate | Key File | Role |
|-------|-------|----------|------|
| Tokens | `cjc-lexer` | `crates/cjc-lexer/src/lib.rs` | Tokenization, keyword recognition |
| Parsing | `cjc-parser` | `crates/cjc-parser/src/lib.rs` | Pratt parser, grammar rules |
| AST | `cjc-ast` | `crates/cjc-ast/src/lib.rs` | Node definitions (`ExprKind`, `StmtKind`, `PatternKind`) |
| HIR | `cjc-hir` | `crates/cjc-hir/src/lib.rs` | Desugaring (pipes, for-loops, try), capture analysis |
| MIR | `cjc-mir` | `crates/cjc-mir/src/lib.rs` | Lambda lifting, pattern compilation |
| MIR Optimizer | `cjc-mir` | `crates/cjc-mir/src/optimize.rs` | Constant folding + dead code elimination |
| NoGC Verifier | `cjc-mir` | `crates/cjc-mir/src/nogc_verify.rs` | Static GC-freedom verification |
| Type System | `cjc-types` | `crates/cjc-types/src/lib.rs` | Types, unification, shape inference, broadcasting |
| Eval (v1) | `cjc-eval` | `crates/cjc-eval/src/lib.rs` | Direct AST interpreter |
| MIR Exec (v2) | `cjc-mir-exec` | `crates/cjc-mir-exec/src/lib.rs` | MIR-based execution |
| Diagnostics | `cjc-diag` | `crates/cjc-diag/src/lib.rs` | Span tracking, error rendering |
| Dispatch | `cjc-dispatch` | `crates/cjc-dispatch/src/lib.rs` | Overload resolution, coherence |
| Runtime | `cjc-runtime` | `crates/cjc-runtime/src/lib.rs` | Builtins: `print`, `len`, `push`, etc. |
| Regex Engine | `cjc-regex` | `crates/cjc-regex/src/lib.rs` | NFA-based byte regex matcher |
| Data DSL | `cjc-data` | `crates/cjc-data/src/lib.rs` | Column-oriented pipelines |
| AD | `cjc-ad` | `crates/cjc-ad/src/lib.rs` | Automatic differentiation |
| CLI | `cjc-cli` | `crates/cjc-cli/src/main.rs` | Entry point, `--mir-opt` flag |

**Tests:** `tests/` directory (624+ tests across 20+ files, plus `tests/milestone_2_4/`, `tests/milestone_2_5/`, `tests/milestone_2_6/`).

---

## 2. Lexical Structure

### 2.1 Keywords

All keywords are reserved; an identifier matching a keyword always produces the keyword token.

| Keyword | Token | Purpose |
|---------|-------|---------|
| `struct` | `Struct` | Value-type definition |
| `class` | `Class` | GC-managed reference type |
| `fn` | `Fn` | Function definition |
| `trait` | `Trait` | Trait definition |
| `impl` | `Impl` | Implementation block |
| `let` | `Let` | Variable binding |
| `mut` | `Mut` | Mutable modifier |
| `return` | `Return` | Return statement |
| `if` | `If` | Conditional |
| `else` | `Else` | Else branch |
| `while` | `While` | While loop |
| `for` | `For` | For loop |
| `in` | `In` | Iterator keyword |
| `nogc` | `NoGc` | No-GC annotation |
| `match` | `Match` | Pattern match |
| `enum` | `Enum` | Enum definition |
| `import` | `Import` | Import declaration |
| `as` | `As` | Aliasing |
| `sealed` | `Sealed` | Sealed trait modifier |
| `col` | `Col` | Data DSL column keyword |
| `true` | `True` | Boolean literal |
| `false` | `False` | Boolean literal |

The token `_` (`Underscore`) is matched in `lex_ident()` and produces a dedicated token kind, but it is **not** included in `is_keyword()`. It functions as a special pattern token rather than a true keyword.

**Evidence:** `cjc-lexer/src/lib.rs`, `lex_ident()` match table, `is_keyword()` method.

### 2.2 Operators and Punctuation

| Token | Symbol | Category |
|-------|--------|----------|
| `Plus` | `+` | Arithmetic |
| `Minus` | `-` | Arithmetic / Unary neg |
| `Star` | `*` | Arithmetic |
| `Slash` | `/` | Arithmetic |
| `Percent` | `%` | Arithmetic |
| `EqEq` | `==` | Comparison |
| `BangEq` | `!=` | Comparison |
| `Lt` | `<` | Comparison |
| `Gt` | `>` | Comparison |
| `LtEq` | `<=` | Comparison |
| `GtEq` | `>=` | Comparison |
| `AmpAmp` | `&&` | Logical AND |
| `PipePipe` | `\|\|` | Logical OR |
| `Bang` | `!` | Logical NOT |
| `Eq` | `=` | Assignment |
| `Pipe` | `\|` | Lambda delimiter |
| `PipeGt` | `\|>` | Pipe operator |
| `TildeEq` | `~=` | Regex match |
| `BangTilde` | `!~` | Regex not-match |
| `Tilde` | `~` | Reserved |
| `Question` | `?` | Try operator |
| `Arrow` | `->` | Return type |
| `FatArrow` | `=>` | Match arm |
| `Dot` | `.` | Field access |
| `DotDot` | `..` | Range |
| `Comma` | `,` | Separator |
| `Colon` | `:` | Type annotation |
| `Semicolon` | `;` | Statement terminator |

**Delimiters:** `(` `)` `{` `}` `[` `]`

**Disambiguation rules** (maximal munch, longest match):

| Sequence | Resolution |
|----------|------------|
| `->` | `Arrow` (not `Minus` + `Gt`) |
| `=>` | `FatArrow` (not `Eq` + `Gt`) |
| `==` | `EqEq` (not `Eq` + `Eq`) |
| `!=` | `BangEq` (not `Bang` + `Eq`) |
| `<=` | `LtEq` |
| `>=` | `GtEq` |
| `&&` | `AmpAmp` (single `&` is an error) |
| `\|\|` | `PipePipe` |
| `\|>` | `PipeGt` |
| `..` | `DotDot` |
| `~=` | `TildeEq` (not `Tilde` + `Eq`) |
| `!~` | `BangTilde` (not `Bang` + `Tilde`) |

**Evidence:** `cjc-lexer/src/lib.rs`, `next_token()` match arms.

### 2.3 Literals

**Integers:**
```
42          // decimal
1_000       // underscores for readability (stripped)
100i64      // with type suffix
42i32       // with type suffix
```

**Floats:**
```
3.14        // decimal point
2.5e10      // exponent notation
1.0E-3      // exponent with sign
1f32        // with type suffix
3.14f64     // with type suffix
```

Valid numeric suffixes: `f32`, `f64`, `i32`, `i64`. Others produce error E0005.

**Strings:**
```
"hello"
"line\nnewline"
"tab\there"
"escaped\\backslash"
"quote\"inside"
"null\0char"
```

Supported escape sequences: `\n` `\t` `\r` `\\` `\"` `\0`. Unknown escapes produce error E0004.

**Byte String Literals:**
```
b"hello"          // ByteStringLit → ByteSlice value
b"\x00\xff"       // hex escapes in byte strings
```

**Byte Char Literals:**
```
b'A'              // ByteCharLit → u8 value (0x41)
b'\n'             // escape sequences supported
```

**Raw String Literals:**
```
r"no \n escapes"          // RawStringLit
r#"can contain "quotes""# // hash-delimited raw strings
br"raw bytes"             // RawByteStringLit
br#"raw "byte" string"#  // hash-delimited raw byte strings
```

**Regex Literals:**
```
/pattern/         // RegexLit (no flags)
/\d+/             // with metacharacters
/hello/i          // with flags (case-insensitive)
/\w+@\w+\.\w+/im // multiple flags
```

Flags: `i` (case-insensitive), `g` (global), `m` (multiline), `s` (dotall), `x` (extended).

Context-sensitive `/` disambiguation: The `/` character starts a regex literal only when the previous token is *not* value-producing (i.e., not an identifier, literal, `)`, `]`, or `}`). After value-producing tokens, `/` is the division operator. A space immediately after `/` also forces division interpretation.

Token text format: `"pattern"` (no flags) or `"pattern\0flags"` (NUL-separated).

**Booleans:** `true`, `false`

**Evidence:** `cjc-lexer/src/lib.rs`, `lex_number()`, `lex_string()`, `lex_regex()`. Tests: `tests/test_lexer.rs`, `tests/test_regex.rs`, `tests/test_bytes_strings.rs`.

### 2.4 Identifiers

Identifiers start with an ASCII letter or `_`, followed by ASCII letters, digits, or `_`. The single `_` is the `Underscore` token (wildcard pattern), not an identifier. All keyword matches take priority over identifier.

**Evidence:** `cjc-lexer/src/lib.rs`, `lex_ident()`.

### 2.5 Comments

```
// Line comment (to end of line)

/* Block comment */

/* Nested /* block */ comments supported */
```

Block comments track nesting depth, so `/* /* */ */` is valid.

**Evidence:** `cjc-lexer/src/lib.rs`, `skip_line_comment()`, `skip_block_comment()`. Tests: `tests/test_lexer.rs`.

### 2.6 Whitespace

Spaces, tabs, carriage returns, and newlines are skipped between tokens. CJC is not whitespace-sensitive.

---

## 3. Expressions

### 3.1 Operator Precedence and Associativity

From lowest to highest binding power:

| Precedence | Operators | Associativity | Example |
|------------|-----------|---------------|---------|
| 2 (lowest) | `=` | Right | `x = y = 1` |
| 4 | `\|>` | Left | `x \|> f \|> g` |
| 6 | `\|\|` | Left | `a \|\| b` |
| 8 | `&&` | Left | `a && b` |
| 10 | `==` `!=` `~=` `!~` | Left | `a == b`, `s ~= /pat/` |
| 12 | `<` `>` `<=` `>=` | Left | `a < b` |
| 14 | `+` `-` | Left | `a + b - c` |
| 16 | `*` `/` `%` | Left | `a * b / c` |
| 18 | Prefix `-` `!` | Right (unary) | `-x`, `!flag` |
| 20 (highest) | `.` `[` `(` `?` | Left (postfix) | `a.b[c](d)?` |

The parser uses Pratt (binding power) parsing: `parse_expr_bp(min_bp)`.

**Evidence:** `cjc-parser/src/lib.rs`, `mod prec` (lines 34-55), `parse_expr_bp()`.

### 3.2 Literals

```
42              // IntLit
3.14            // FloatLit
"hello"         // StringLit
true            // BoolLit
false           // BoolLit
```

**AST:** `ExprKind::IntLit(i64)`, `ExprKind::FloatLit(f64)`, `ExprKind::StringLit(String)`, `ExprKind::BoolLit(bool)`, `ExprKind::ByteStringLit(Vec<u8>)`, `ExprKind::ByteCharLit(u8)`, `ExprKind::RawStringLit(String)`, `ExprKind::RawByteStringLit(Vec<u8>)`, `ExprKind::RegexLit { pattern, flags }`.

### 3.3 Identifiers

```
x
foo_bar
_temp
```

**AST:** `ExprKind::Ident(Ident)`.

### 3.4 Unary Operators

```
-x          // Negation (numeric types)
!flag       // Logical NOT (bool only)
```

**AST:** `ExprKind::Unary { op: UnaryOp, operand }`.

### 3.5 Binary Operators

```
a + b       a - b       a * b       a / b       a % b
a == b      a != b      a < b       a > b       a <= b      a >= b
a && b      a || b
s ~= /pat/              // regex match (Bool)
s !~ /pat/              // regex not-match (Bool)
```

**AST:** `ExprKind::Binary { op: BinOp, left, right }`.

The `~=` (Match) and `!~` (NotMatch) operators require the left operand to be `String`, `ByteSlice`, or `StrView` and the right operand to be `Regex`. They return `Bool`.

### 3.6 Assignment

```
x = 42
a[i] = val
obj.field = val
```

Right-associative. Target must be a variable, index expression, or field access. Requires the target to be declared `mut`.

**AST:** `ExprKind::Assign { target, value }`.

**Evidence:** Tests: `tests/test_eval.rs` (`test_variable_assignment`).

### 3.7 Function Calls

```
foo(a, b)                 // Positional arguments
create(width: 10, height: 20)  // Named arguments
obj.method(x)             // Method-style call
```

**AST:** `ExprKind::Call { callee, args: Vec<CallArg> }` where `CallArg { name: Option<Ident>, value: Expr }`.

**Evidence:** `cjc-parser/src/lib.rs`, `parse_call_args()`. Tests: `tests/test_parser.rs` (`test_parse_call_with_named_args`).

### 3.8 Field Access

```
point.x
obj.field.subfield
```

**AST:** `ExprKind::Field { object, name }`.

### 3.9 Indexing

```
arr[0]          // Single index
tensor[i, j]    // Multi-dimensional index
```

Single-index produces `ExprKind::Index { object, index }`. Multi-index (comma-separated) produces `ExprKind::MultiIndex { object, indices }`.

**Evidence:** Tests: `tests/test_eval.rs` (`test_array_literal_and_indexing`, `test_tensor_multi_index`).

### 3.10 Pipe Operator

```
data |> filter(pred)
5 |> double |> add_one
10 |> add(5)
```

Left-associative. Desugared in HIR:
- `x |> f(y)` becomes `f(x, y)` (left value prepended as first argument)
- `x |> f` becomes `f(x)`

**AST:** `ExprKind::Pipe { left, right }`.
**HIR:** Desugared to `HirExprKind::Call`.

**Evidence:** Tests: `tests/test_eval.rs` (`test_pipe_operator`, `test_pipe_with_extra_args`).

### 3.11 Try Operator

```
risky_call()?
```

Postfix `?`. Desugared in HIR to:

```
match risky_call() {
    Ok(__try_v)  => __try_v,
    Err(__try_e) => return Err(__try_e),
}
```

**AST:** `ExprKind::Try(Box<Expr>)`.

**Evidence:** `cjc-hir/src/lib.rs`, try desugaring. Tests: `tests/milestone_2_6/option_result.rs`.

### 3.12 Block Expressions

```
{
    let x = 1;
    let y = 2;
    x + y       // tail expression (no semicolon) = block value
}
```

A block contains zero or more statements, optionally followed by a tail expression (without `;`). The tail expression is the value of the block. If there is no tail expression, the block produces no value.

**AST:** `ExprKind::Block(Block)` where `Block { stmts, expr: Option<Box<Expr>> }`.

**Evidence:** `cjc-parser/src/lib.rs`, `parse_block()`.

### 3.13 Array Literals

```
[1, 2, 3]
[10, 20, 30,]   // trailing comma allowed
```

**AST:** `ExprKind::ArrayLit(Vec<Expr>)`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_array_literal`).

### 3.14 Tuple Literals

```
(1, 2, 3)
(a, b)
()              // unit tuple
```

A single parenthesized expression `(x)` is just grouping; two or more comma-separated expressions produce a tuple.

**AST:** `ExprKind::TupleLit(Vec<Expr>)`.

**Evidence:** Tests: `tests/test_match_patterns.rs`.

### 3.15 Struct Literals

```
Point { x: 1.0, y: 2.0 }
Config { width: 800, height: 600, title: "CJC" }
```

Struct literals are parsed as postfix on an identifier. They are **disabled** inside `if`/`while` conditions and `match` scrutinees to avoid ambiguity with block bodies.

**AST:** `ExprKind::StructLit { name, fields: Vec<FieldInit> }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_struct_literal`), `tests/test_eval.rs` (`test_struct_creation_and_field_access`).

### 3.16 Lambda Expressions

```
|x: f64| x * 2.0              // single param with type
|x: f64, y: f64| x + y        // multiple params
|| 42                          // no params
|x: f64| { let y = x; y }     // block body
```

Parameters optionally have type annotations. The body is a single expression (which can be a block).

**Captures:** If the lambda body references variables from an enclosing scope, HIR capture analysis produces a `Closure` node with `captures: Vec<HirCapture>`. In `nogc` context, captures use `Clone` mode; otherwise `Ref` mode.

**AST:** `ExprKind::Lambda { params, body }`.
**HIR:** `HirExprKind::Lambda` (no captures) or `HirExprKind::Closure` (with captures).

**Evidence:** Tests: `tests/test_closures.rs`.

### 3.17 Match Expressions

```
match value {
    pattern1 => expr1,
    pattern2 => expr2,
    _ => default_expr,
}
```

Comma-separated arms. Trailing comma optional. See Section 8 for pattern syntax.

**AST:** `ExprKind::Match { scrutinee, arms: Vec<MatchArm> }`.

**Evidence:** Tests: `tests/test_match_patterns.rs` (51 tests), `tests/milestone_2_6/enums.rs`.

### 3.18 Enum Variant Construction

```
None                    // Unit variant
Some(42)                // Single-field variant
Rect(10.0, 20.0)       // Multi-field variant
```

Prelude variants (`Some`, `None`, `Ok`, `Err`) are auto-resolved to their enums (`Option`, `Result`). User-defined variants are registered during HIR lowering.

**AST:** `ExprKind::VariantLit { enum_name, variant, fields }` or resolved from `ExprKind::Ident` / `ExprKind::Call` during HIR lowering.

**Evidence:** Tests: `tests/milestone_2_6/enums.rs`, `tests/milestone_2_6/option_result.rs`.

### 3.19 Column Expression (Data DSL)

```
col("price")
```

**AST:** `ExprKind::Col(String)`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_col`), `tests/test_data.rs`.

---

## 4. Statements

### 4.1 Let Bindings

```
let x = 42;                    // immutable
let mut count = 0;             // mutable
let x: i64 = 42;               // with type annotation
let mut total: f64 = 0.0;      // mutable with type
```

Every `let` requires an initializer (`= expr`). Type annotation is optional.

**AST:** `StmtKind::Let(LetStmt)` where `LetStmt { name, mutable, ty: Option<TypeExpr>, init }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_let`, `test_parse_let_mut`).

### 4.2 Expression Statements

```
print("hello");
foo(x);
```

Any expression followed by `;` is an expression statement. Without `;`, an expression at the end of a block becomes the tail expression (block value).

**AST:** `StmtKind::Expr(Expr)`.

### 4.3 Return Statements

```
return 42;
return;
```

Optional return value. Used for early return from functions.

**AST:** `StmtKind::Return(Option<Expr>)`.

**Evidence:** Tests: `tests/test_eval.rs` (`test_early_return`).

### 4.4 If / Else

```
if condition {
    body
}

if condition {
    then_body
} else {
    else_body
}

if x > 0 {
    1
} else if x == 0 {
    0
} else {
    -1
}
```

`else if` chains are supported via `ElseBranch::ElseIf`. Struct literals are disabled in the condition to avoid ambiguity with the block body `{`.

**AST:** `StmtKind::If(IfStmt)` where `IfStmt { condition, then_block, else_branch: Option<ElseBranch> }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_if_else_if_else`), `tests/test_eval.rs` (`test_if_else`).

### 4.5 While Loops

```
while condition {
    body
}
```

Struct literals disabled in the condition.

**AST:** `StmtKind::While(WhileStmt)` where `WhileStmt { condition, body }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_while`), `tests/test_eval.rs` (`test_while_loop`).

### 4.6 For Loops

**Range form (exclusive end):**
```
for i in 0..10 {
    print(i);
}
```

**Expression form:**
```
for x in array {
    print(x);
}
```

Both forms are **desugared in HIR** to while loops (see Section 7).

**AST:** `StmtKind::For(ForStmt)` where `ForStmt { ident, iter: ForIter, body }` and `ForIter` is either `Range { start, end }` or `Expr(Box<Expr>)`.

**Evidence:** Tests: `tests/test_for_loops.rs`.

### 4.7 NoGC Blocks

```
nogc {
    // Only value-type operations allowed here
    let t = Tensor.zeros([3, 3]);
}
```

Restricts code to GC-free operations. Closures inside `nogc` blocks capture by `Clone` instead of `Ref`.

**AST:** `StmtKind::NoGcBlock(Block)`.

**Evidence:** Tests: `tests/milestone_2_4/nogc_verifier/`.

---

## 5. Declarations

### 5.1 Function Declarations

```
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn greet(name: String) {
    print(name)
}

nogc fn fast_compute(x: f64) -> f64 {
    x * x
}
```

- Optional return type annotation (`-> Type`).
- Optional type parameters (`<T, U: Bound>`).
- `nogc` prefix marks function as GC-free.
- Parameters always have type annotations.

**AST:** `DeclKind::Fn(FnDecl)` where `FnDecl { name, type_params, params, return_type, body, is_nogc }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_fn_simple`, `test_parse_fn_no_return_type`, `test_parse_fn_nogc`).

### 5.2 Struct Declarations

```
struct Point {
    x: f64,
    y: f64,
}

struct Pair<T: Clone, U> {
    first: T,
    second: U,
}

struct Config {
    width: i64,
    height: i64,
    scale: f64 = 1.0,    // default value
}
```

Fields have `name: Type` and optionally `= default_expr`.

**AST:** `DeclKind::Struct(StructDecl)` where `StructDecl { name, type_params, fields: Vec<FieldDecl> }` and `FieldDecl { name, ty, default: Option<Expr> }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_struct_simple`, `test_parse_struct_generic`).

### 5.3 Class Declarations

```
class Node<T> {
    value: T,
    next: Node<T>,
}
```

Classes are GC-managed reference types. Same field syntax as structs.

**AST:** `DeclKind::Class(ClassDecl)`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_class`).

### 5.4 Enum Declarations

```
enum Color { Red, Green, Blue }

enum Shape {
    Circle(f64),
    Rect(f64, f64),
}

enum Option<T> {
    Some(T),
    None,
}
```

Variants can be unit (`Name`) or tuple-like (`Name(Type, ...)`). No struct-like variants.

**AST:** `DeclKind::Enum(EnumDecl)` where `EnumDecl { name, type_params, variants: Vec<VariantDecl> }` and `VariantDecl { name, fields: Vec<TypeExpr> }`.

**Evidence:** Tests: `tests/milestone_2_6/enums.rs`.

### 5.5 Trait Declarations

```
trait Numeric: Add + Mul {
    fn zero() -> Self;
    fn one() -> Self;
}

sealed trait Shape {
    fn area() -> f64;
}
```

- Optional super-trait bounds (`: Trait1 + Trait2`).
- Methods are signatures only (no bodies).
- `sealed` modifier restricts implementors.

**AST:** `DeclKind::Trait(TraitDecl)` where `TraitDecl { name, type_params, super_traits, methods: Vec<FnSig> }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_trait`).

### 5.6 Impl Blocks

```
impl Point {
    fn distance(self: Point) -> f64 { ... }
}

impl<T> Vec<T> : Iterable {
    fn len(self: Vec<T>) -> i64 { 0 }
}
```

- Optional type parameters.
- Optional trait reference (`: TraitName`).
- Methods can be marked `nogc`.

**AST:** `DeclKind::Impl(ImplDecl)` where `ImplDecl { type_params, target, trait_ref, methods }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_impl`).

### 5.7 Import Declarations

```
import std.io.File as F
import math.linalg
```

Dot-separated module path with optional `as` alias.

**AST:** `DeclKind::Import(ImportDecl)` where `ImportDecl { path: Vec<Ident>, alias: Option<Ident> }`.

**Evidence:** Tests: `tests/test_parser.rs` (`test_parse_import`, `test_parse_import_no_alias`).

---

## 6. Types

### 6.1 Primitive Types

| Syntax | Description |
|--------|-------------|
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |
| `f32` | 32-bit float |
| `f64` | 64-bit float |
| `bool` | Boolean |
| `Str` / `String` | String type |
| `ByteSlice` | Non-owning byte view (`Rc<Vec<u8>>` pre-LLVM) |
| `StrView` | Validated UTF-8 byte view |
| `u8` | Unsigned byte |
| `Regex` | Compiled regex pattern |
| `Void` | Unit/void type |

**Evidence:** `cjc-types/src/lib.rs`, `Type` enum.

### 6.2 Composite Types

```
Tensor<f32, [3, 4]>      // Tensor with element type and shape
Buffer<f64>               // Buffer with element type
[i64; 5]                  // Fixed-size array
(i64, f64, bool)          // Tuple type
fn(i64, i64) -> i64       // Function type
Map<String, i64>          // Map type
```

**AST type expressions:**
- `TypeExprKind::Named { name, args }` — named types with optional generic args
- `TypeExprKind::Array { elem, size }` — `[T; N]`
- `TypeExprKind::Tuple(Vec<TypeExpr>)` — `(T, U, ...)`
- `TypeExprKind::Fn { params, ret }` — `fn(T, U) -> V`
- `TypeExprKind::ShapeLit(Vec<ShapeDim>)` — `[M, N]` shape literal

### 6.3 Generic Type Parameters

```
fn identity<T>(x: T) -> T { x }
struct Pair<T: Clone, U> { first: T, second: U }
trait Container<T: Eq + Hash> { ... }
```

Type parameters can have bounds separated by `+`.

**Type Arguments** in angle brackets can be types, expressions (for sizes), or shape literals:
```
Tensor<f32, [M, N]>      // Type + shape args
[i64; 10]                 // Expression arg for array size
```

**AST:** `TypeParam { name, bounds }`, `TypeArg::Type | TypeArg::Expr | TypeArg::Shape`.

**Evidence:** `cjc-parser/src/lib.rs`, `parse_optional_type_params()`, `parse_type_arg_list()`.

### 6.4 Shape Dimensions

Shapes describe tensor dimensions. Each dimension is either a literal integer or a symbolic name:

```
[3, 4]          // Two known dimensions
[M, N]          // Two symbolic dimensions
[batch, 784]    // Mixed
```

**AST:** `ShapeDim::Lit(i64)` or `ShapeDim::Name(Ident)`.

---

## 7. Desugaring Notes

### 7.1 Pipe Desugaring (AST → HIR)

```
x |> f(y, z)    →    f(x, y, z)     // prepend LHS as first arg
x |> f          →    f(x)           // call with LHS as sole arg
```

**Evidence:** `cjc-hir/src/lib.rs`, `lower_expr()` match on `ExprKind::Pipe`.

### 7.2 For-Loop Desugaring (AST → HIR)

**Range form:**
```
for i in start..end { body }
```
desugars to:
```
{
    let __for_end_N = end;
    let mut __for_idx_N = start;
    while __for_idx_N < __for_end_N {
        let i = __for_idx_N;
        body
        __for_idx_N = __for_idx_N + 1;
    }
}
```

**Expression form:**
```
for x in arr { body }
```
desugars to:
```
{
    let __for_arr_N = arr;
    let __for_len_N = len(__for_arr_N);
    let mut __for_idx_N = 0;
    while __for_idx_N < __for_len_N {
        let x = __for_arr_N[__for_idx_N];
        body
        __for_idx_N = __for_idx_N + 1;
    }
}
```

Generated names use `__for_` prefix with unique IDs to avoid collisions.

**Evidence:** `cjc-hir/src/lib.rs`, `desugar_for_range()`, `desugar_for_expr()`, `gensym()`.

### 7.3 Try Operator Desugaring (AST → HIR)

```
expr?
```
desugars to:
```
match expr {
    Ok(__try_v) => __try_v,
    Err(__try_e) => return Err(__try_e),
}
```

**Evidence:** `cjc-hir/src/lib.rs`, try desugaring block.

### 7.4 Closure Capture Analysis (AST → HIR)

Lambdas that reference variables from enclosing scopes are promoted from `Lambda` to `Closure` nodes. The analysis:

1. Collects all variable references in the body.
2. Excludes: lambda parameters, body-local definitions, known function names, builtins.
3. Remaining references are captures.
4. Capture mode: `Ref` normally, `Clone` inside `nogc` context.

**Builtins excluded from capture:** `print`, `Tensor`, `matmul`, `Buffer`, `len`, `push`, `assert`, `assert_eq`, `clock`, `gc_alloc`, `gc_collect`, `gc_live_count`, `Some`, `None`, `Ok`, `Err`, `bf16_to_f32`, `f32_to_bf16`, `true`, `false`.

**Evidence:** `cjc-hir/src/lib.rs`, capture analysis in `Lambda` lowering. Tests: `tests/test_closures.rs`.

### 7.5 Lambda Lifting (HIR → MIR)

In MIR, closures are lambda-lifted: the closure body becomes a top-level `MirFunction` and the closure site becomes `MirExprKind::MakeClosure { fn_name, captures }`.

**Evidence:** `cjc-mir/src/lib.rs`, `HirToMir`.

### 7.6 Variant Resolution (AST → HIR)

Identifiers matching known variant names are resolved to `VariantLit` expressions. Prelude variants are auto-registered:

| Variant | Enum |
|---------|------|
| `Some` | `Option` |
| `None` | `Option` |
| `Ok` | `Result` |
| `Err` | `Result` |

User-defined enum variants are registered during the HIR pre-scan pass. If a variant name collides with a local variable, the local variable takes precedence.

**Evidence:** `cjc-hir/src/lib.rs`, `variant_names` HashMap, `lower_expr()` Ident/Call resolution.

---

## 8. Patterns

Patterns appear in `match` arms.

### 8.1 Wildcard Pattern

```
_ => default_value
```

Matches anything, binds nothing.

**AST:** `PatternKind::Wildcard`.

### 8.2 Binding Pattern

```
x => x + 1
n => n * 2
```

Matches anything, binds the value to the name.

**AST:** `PatternKind::Binding(Ident)`.

Note: If the name matches a known unit variant (e.g., `None`), it is lowered as a variant pattern in HIR, not a binding.

### 8.3 Literal Patterns

```
42 => "forty-two"
-3 => "negative three"
3.14 => "pi"
true => "yes"
false => "no"
"hello" => "greeting"
```

Negative integer/float literals are supported (parsed as prefix `-` + literal).

**AST:** `PatternKind::LitInt(i64)`, `PatternKind::LitFloat(f64)`, `PatternKind::LitBool(bool)`, `PatternKind::LitString(String)`.

### 8.4 Tuple Patterns

```
(a, b) => a + b
(x, _, z) => x + z
(1, y) => y
```

**AST:** `PatternKind::Tuple(Vec<Pattern>)`.

### 8.5 Struct Patterns

```
Point { x, y } => x + y                    // shorthand (binds field name)
Point { x: px, y: py } => px + py          // explicit binding
Point { x: 0, y } => y                     // mix literal + shorthand
```

**AST:** `PatternKind::Struct { name, fields: Vec<PatternField> }` where `PatternField { name, pattern: Option<Pattern> }`. Missing `pattern` means shorthand binding.

### 8.6 Enum Variant Patterns

```
Some(x) => x
None => 0
Ok(value) => value
Err(e) => handle(e)
Circle(r) => 3.14 * r * r
Rect(w, h) => w * h
```

**AST:** `PatternKind::Variant { enum_name, variant, fields: Vec<Pattern> }`.

**Evidence:** Tests: `tests/test_match_patterns.rs` (51 tests), `tests/milestone_2_6/enums.rs`.

---

## 9. Pitfalls and Diagnostics

### 9.1 Error Code Reference

**Lexer Errors (E0002-E0005):**

| Code | Message | Fix |
|------|---------|-----|
| E0002 | `unexpected character '&', did you mean '&&'?` | Use `&&` for logical AND |
| E0002 | `unexpected character '{c}'` | Remove or fix invalid character |
| E0003 | `unterminated string literal` | Close the string with `"` |
| E0004 | `unknown escape sequence '\{c}'` | Use valid escapes: `\n \t \r \\ \" \0` |
| E0005 | `invalid numeric suffix '{s}'` | Use `f32`, `f64`, `i32`, or `i64` |

**Type Checker Errors (E0101-E0201):**

| Code | Meaning |
|------|---------|
| E0101 | Binary/logical operator applied to wrong types |
| E0102 | Equality or comparison on incompatible types |
| E0103 | Return type mismatch |
| E0104 | Let binding type annotation doesn't match initializer |
| E0105 | Non-boolean condition in `if` or `while` |
| E0106 | Undefined variable |
| E0107 | Wrong type for unary operator |
| E0108 | Field access on wrong type or nonexistent field |
| E0109 | Index into non-indexable type |
| E0110 | Struct literal field type mismatch |
| E0111 | Undefined type name |
| E0112 | Heterogeneous array element types |
| E0113 | No matching function overload |
| E0114 | Undefined function |
| E0115 | Generic type parameter violates trait bound |
| E0120 | `?` on non-`Result` type |
| E0121 | Wrong number of enum variant fields |
| E0122 | Unknown enum variant |
| E0130 | Non-exhaustive match (missing variant coverage) |
| E0201 | GC allocation inside `nogc` context |

**Dispatch Errors (E0300-E0303):**

| Code | Meaning |
|------|---------|
| E0301 | Ambiguous method resolution |
| E0302 | No matching function for argument types |
| E0303 | Overlapping function definitions (coherence) |

**Parser Errors (E1000-E1002):**

| Code | Meaning |
|------|---------|
| E1000 | General parse error |
| E1001 | Expected token X, found token Y |
| E1002 | Parse error with hint |

### 9.2 Common Pitfalls

1. **Single `&` is an error.** CJC has no bitwise AND; use `&&` for logical AND.

2. **Struct literals in conditions.** `if val == Foo { x: 1 } { ... }` is ambiguous. The parser disables struct literals inside `if`/`while` conditions and `match` scrutinees. Use a `let` binding instead:
   ```
   let expected = Foo { x: 1 };
   if val == expected { ... }
   ```

3. **`||` for zero-parameter lambdas.** The lexer tokenizes `||` as `PipePipe`. The parser special-cases this in `parse_lambda_no_params()` to recognize zero-parameter lambdas.

4. **Trailing semicolons change block values.** `{ x + y }` evaluates to `x + y`; `{ x + y; }` evaluates to void (the expression becomes a statement).

5. **`for` range is exclusive.** `for i in 0..3` iterates `i = 0, 1, 2` (not 3).

6. **No `else if` keyword.** `else if` is parsed as `else` followed by an `if` statement; they are two separate tokens.

7. **Named arguments in calls.** `foo(name: value)` uses `name:` syntax. This can't be used in lambda parameter position.

8. **Match exhaustiveness.** For enum types, the type checker (E0130) requires all variants to be covered. A wildcard `_` or binding pattern satisfies this.

### 9.3 Error Recovery

The parser implements error recovery via `synchronize()`: on a parse error, it skips tokens until reaching a synchronization point (`;`, `}`, EOF, or a declaration keyword). This allows multiple errors to be reported in one pass.

**Evidence:** `cjc-parser/src/lib.rs`, `synchronize()`.

---

## 10. Grammar (EBNF-ish)

This section presents an approximate grammar derived from the parser implementation. Terminals are in `"quotes"` or UPPER_CASE token names.

```ebnf
program         = decl* EOF

decl            = struct_decl
                | class_decl
                | enum_decl
                | fn_decl
                | nogc_fn_decl
                | trait_decl
                | impl_decl
                | import_decl
                | let_decl ";"
                | stmt

(* ── Declarations ─────────────────────────────────── *)

struct_decl     = "struct" IDENT type_params? "{" field_list "}"
class_decl      = "class" IDENT type_params? "{" field_list "}"
enum_decl       = "enum" IDENT type_params? "{" variant_list "}"

field_list      = (field_decl ("," field_decl)* ","?)?
field_decl      = IDENT ":" type_expr ("=" expr)?

variant_list    = (variant_decl ("," variant_decl)* ","?)?
variant_decl    = IDENT ("(" type_list ")")?
type_list       = type_expr ("," type_expr)* ","?

fn_decl         = "fn" IDENT type_params? "(" param_list ")"
                  ("->" type_expr)? block
nogc_fn_decl    = "nogc" fn_decl

param_list      = (param ("," param)* ","?)?
param           = IDENT ":" type_expr

trait_decl      = "sealed"? "trait" IDENT type_params?
                  (":" trait_bounds)? "{" fn_sig* "}"
trait_bounds     = type_expr ("+" type_expr)*
fn_sig          = "fn" IDENT type_params? "(" param_list ")"
                  ("->" type_expr)? ";"

impl_decl       = type_params? "impl" type_expr (":" type_expr)?
                  "{" fn_decl* "}"

import_decl     = "import" IDENT ("." IDENT)* ("as" IDENT)?

let_decl        = "let" "mut"? IDENT (":" type_expr)? "=" expr

(* ── Statements ───────────────────────────────────── *)

block           = "{" stmt* expr? "}"

stmt            = let_stmt
                | return_stmt
                | if_stmt
                | while_stmt
                | for_stmt
                | nogc_block
                | expr_stmt

let_stmt        = "let" "mut"? IDENT (":" type_expr)? "=" expr ";"
return_stmt     = "return" expr? ";"
expr_stmt       = expr ";"

if_stmt         = "if" expr block ("else" (if_stmt | block))?
while_stmt      = "while" expr block
for_stmt        = "for" IDENT "in" for_iter block
for_iter        = expr ".." expr        (* range *)
                | expr                  (* expression *)
nogc_block      = "nogc" block

(* ── Expressions (by precedence, low to high) ─────── *)

expr            = assign_expr

assign_expr     = pipe_expr ("=" assign_expr)?           (* right-assoc *)
pipe_expr       = or_expr ("|>" or_expr)*
or_expr         = and_expr ("||" and_expr)*
and_expr        = eq_expr ("&&" eq_expr)*
eq_expr         = cmp_expr (("==" | "!=" | "~=" | "!~") cmp_expr)*
cmp_expr        = add_expr (("<" | ">" | "<=" | ">=") add_expr)*
add_expr        = mul_expr (("+" | "-") mul_expr)*
mul_expr        = unary_expr (("*" | "/" | "%") unary_expr)*
unary_expr      = ("-" | "!") unary_expr | postfix_expr
postfix_expr    = atom (postfix_op)*
postfix_op      = "." IDENT
                | "(" call_args ")"
                | "[" index_args "]"
                | "?"
                | "{" field_inits "}"   (* struct literal, conditional *)

atom            = INT_LIT | FLOAT_LIT | STRING_LIT
                | BYTE_STRING_LIT | BYTE_CHAR_LIT
                | RAW_STRING_LIT | RAW_BYTE_STRING_LIT
                | REGEX_LIT
                | "true" | "false"
                | IDENT
                | "col" "(" STRING_LIT ")"
                | "(" tuple_or_expr ")"
                | "[" array_elems "]"
                | "|" param_list "|" expr
                | "||" expr                             (* zero-param lambda *)
                | "match" expr "{" match_arms "}"
                | block

call_args       = (call_arg ("," call_arg)* ","?)?
call_arg        = (IDENT ":")? expr
index_args      = expr ("," expr)*
field_inits     = (field_init ("," field_init)* ","?)?
field_init      = IDENT ":" expr

tuple_or_expr   = /* empty */                           (* unit tuple *)
                | expr                                  (* grouping *)
                | expr "," (expr ("," expr)*)?          (* tuple *)

array_elems     = (expr ("," expr)* ","?)?

match_arms      = (match_arm ("," match_arm)* ","?)?
match_arm       = pattern "=>" expr

(* ── Patterns ─────────────────────────────────────── *)

pattern         = "_"                                   (* wildcard *)
                | "true" | "false"                      (* bool literal *)
                | STRING_LIT                            (* string literal *)
                | "-"? INT_LIT                          (* int literal *)
                | "-"? FLOAT_LIT                        (* float literal *)
                | "(" pattern ("," pattern)* ")"        (* tuple *)
                | IDENT "{" pat_fields "}"              (* struct *)
                | IDENT "(" pattern ("," pattern)* ")"  (* variant *)
                | IDENT                                 (* binding or unit variant *)

pat_fields      = (pat_field ("," pat_field)* ","?)?
pat_field       = IDENT (":" pattern)?

(* ── Type Expressions ─────────────────────────────── *)

type_expr       = named_type
                | tuple_type
                | array_type
                | fn_type

named_type      = IDENT ("<" type_arg_list ">")?
type_arg_list   = type_arg ("," type_arg)*
type_arg        = "[" shape_dims "]"                    (* shape arg *)
                | INT_LIT                               (* expr arg *)
                | type_expr                             (* type arg *)

tuple_type      = "(" (type_expr ("," type_expr)*)? ")"
array_type      = "[" type_expr ";" expr "]"
fn_type         = "fn" "(" (type_expr ("," type_expr)*)? ")" "->" type_expr
shape_dims      = shape_dim ("," shape_dim)*
shape_dim       = IDENT | INT_LIT

type_params     = "<" type_param ("," type_param)* ">"
type_param      = IDENT (":" type_expr ("+" type_expr)*)?
```

**Evidence:** `cjc-parser/src/lib.rs`, all `parse_*` functions.

---

## 11. Examples

### 11.1 Arithmetic and Variables

```cjc
let x = 2 + 3 * 4;        // 14 (precedence: * before +)
let mut sum = 0;
sum = sum + x;
```
*(see: tests/test_eval.rs, test_basic_arithmetic_int)*

### 11.2 Functions and Recursion

```cjc
fn factorial(n: i64) {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
fn main() { factorial(5) }
```
*(see: tests/test_eval.rs, test_recursive_function)*

### 11.3 Structs

```cjc
struct Point { x: f64, y: f64 }

fn main() {
    let p = Point { x: 10.0, y: 20.0 };
    p.x + p.y
}
```
*(see: tests/test_eval.rs, test_struct_creation_and_field_access)*

### 11.4 Enums and Match

```cjc
enum Shape {
    Circle(f64),
    Rect(f64, f64),
}

fn area(s: Shape) -> f64 {
    match s {
        Circle(r) => 3.14159 * r * r,
        Rect(w, h) => w * h,
    }
}

fn main() -> f64 {
    let s = Circle(5.0);
    area(s)
}
```
*(see: tests/milestone_2_6/enums.rs, enum_match_payload_with_binding)*

### 11.5 Closures with Capture

```cjc
let scale = 2.0;
let f = |x: f64| x * scale;
f(21.0)
```
*(see: tests/test_closures.rs, test_hir_closure_single_capture_ref)*

### 11.6 For Loops

```cjc
let mut sum = 0;
for i in 0..10 {
    sum = sum + i;
}
sum
```
*(see: tests/test_for_loops.rs, test_parse_for_range_basic)*

### 11.7 Pipe Operator

```cjc
fn double(x: i64) { x * 2 }
fn add_one(x: i64) { x + 1 }
fn main() { 5 |> double |> add_one }
```
Result: `11`. *(see: tests/test_eval.rs, test_pipe_operator)*

### 11.8 Pattern Matching on Tuples

```cjc
fn main() {
    let pair = (10, 20);
    match pair {
        (a, b) => a + b,
    }
}
```
*(see: tests/test_match_patterns.rs, test_parse_match_with_tuple_pattern)*

### 11.9 Struct Destructuring in Match

```cjc
struct Point { x: f64, y: f64 }

fn main() {
    let p = Point { x: 3.0, y: 4.0 };
    match p {
        Point { x, y } => x + y,
    }
}
```
*(see: tests/test_match_patterns.rs, test_parse_match_with_struct_pattern)*

### 11.10 Option and Result

```cjc
fn safe_div(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("division by zero")
    } else {
        Ok(a / b)
    }
}
```
*(see: tests/milestone_2_6/option_result.rs)*

### 11.11 NoGC Functions

```cjc
nogc fn dot_product(a: Tensor<f64, [N]>, b: Tensor<f64, [N]>) -> f64 {
    // GC-free computation only
    matmul(a, b)
}
```
*(see: tests/milestone_2_4/nogc_verifier/)*

### 11.12 Traits and Impls

```cjc
trait Numeric: Add + Mul {
    fn zero() -> Self;
    fn one() -> Self;
}

impl Point {
    fn magnitude(self: Point) -> f64 {
        self.x * self.x + self.y * self.y
    }
}
```
*(see: tests/test_parser.rs, test_parse_trait, test_parse_impl)*

---

## 12. Syntax Coverage Checklist

| Feature | Status | Evidence |
|---------|--------|----------|
| Integer literals | Proven | `test_lexer.rs`, `test_eval.rs` |
| Float literals | Proven | `test_lexer.rs`, `test_eval.rs` |
| String literals + escapes | Proven | `test_lexer.rs` |
| Boolean literals | Proven | `test_eval.rs` |
| Numeric type suffixes (f32/f64/i32/i64) | Proven | `test_lexer.rs` |
| Line comments `//` | Proven | `test_lexer.rs` |
| Block comments `/* */` (nested) | Proven | `test_lexer.rs` |
| All binary operators | Proven | `test_eval.rs` |
| Unary `-` and `!` | Proven | `test_eval.rs` |
| Operator precedence | Proven | `test_eval.rs` (arithmetic order) |
| Assignment `=` | Proven | `test_eval.rs` |
| Let bindings (immutable) | Proven | `test_parser.rs`, `test_eval.rs` |
| Let bindings (mutable) | Proven | `test_parser.rs`, `test_eval.rs` |
| Let with type annotation | Proven | `test_parser.rs` |
| If / else | Proven | `test_parser.rs`, `test_eval.rs` |
| Else-if chains | Proven | `test_parser.rs` |
| While loops | Proven | `test_parser.rs`, `test_eval.rs` |
| For loops (range) | Proven | `test_for_loops.rs` |
| For loops (expression) | Proven | `test_for_loops.rs` |
| Return statement | Proven | `test_eval.rs` |
| Block expressions (tail expr) | Proven | `test_eval.rs` |
| Function definitions | Proven | `test_parser.rs`, `test_eval.rs` |
| Function calls | Proven | `test_eval.rs` |
| Named call arguments | Proven | `test_parser.rs` |
| Recursive functions | Proven | `test_eval.rs` |
| Struct declarations | Proven | `test_parser.rs` |
| Struct field defaults | Proven | `test_parser.rs` |
| Struct literals | Proven | `test_parser.rs`, `test_eval.rs` |
| Field access `.` | Proven | `test_eval.rs` |
| Class declarations | Proven | `test_parser.rs` |
| Enum declarations | Proven | `milestone_2_6/enums.rs` |
| Enum variant construction | Proven | `milestone_2_6/enums.rs` |
| Match expressions | Proven | `test_match_patterns.rs` (51 tests) |
| Wildcard pattern `_` | Proven | `test_match_patterns.rs` |
| Binding patterns | Proven | `test_match_patterns.rs` |
| Literal patterns (int/float/bool/string) | Proven | `test_match_patterns.rs` |
| Tuple patterns | Proven | `test_match_patterns.rs` |
| Struct patterns (shorthand) | Proven | `test_match_patterns.rs` |
| Struct patterns (explicit) | Proven | `test_match_patterns.rs` |
| Enum variant patterns | Proven | `milestone_2_6/enums.rs` |
| Match exhaustiveness check | Proven | `milestone_2_6/exhaustiveness.rs` |
| Lambda expressions | Proven | `test_closures.rs` |
| Zero-param lambdas `\|\| expr` | Proven | `test_closures.rs` |
| Closure capture analysis | Proven | `test_closures.rs` |
| NoGC clone captures | Proven | `test_closures.rs` |
| Array literals | Proven | `test_parser.rs`, `test_eval.rs` |
| Array indexing `a[i]` | Proven | `test_eval.rs` |
| Multi-index `a[i,j]` | Proven | `test_eval.rs` |
| Tuple literals | Proven | `test_match_patterns.rs` |
| Pipe operator `\|>` | Proven | `test_eval.rs` |
| Try operator `?` | Proven | `milestone_2_6/option_result.rs` |
| Trait declarations | Proven | `test_parser.rs` |
| Super-trait bounds | Proven | `test_parser.rs` |
| Sealed traits | Observed | Parser handles `sealed` keyword |
| Impl blocks | Proven | `test_parser.rs` |
| Impl with trait | Proven | `test_parser.rs` |
| Generic type parameters | Proven | `test_parser.rs` |
| Type parameter bounds | Proven | `test_parser.rs` |
| Import declarations | Proven | `test_parser.rs` |
| Import with alias | Proven | `test_parser.rs` |
| Col expression (data DSL) | Proven | `test_parser.rs`, `test_data.rs` |
| NoGC functions | Proven | `milestone_2_4/nogc_verifier/` |
| NoGC blocks | Proven | `test_parser.rs` |
| String concatenation `+` | Proven | `test_eval.rs` |
| Tensor operations | Proven | `test_eval.rs` |
| Function types `fn(T) -> U` | Proven | `test_parser.rs` |
| Array types `[T; N]` | Proven | `test_parser.rs` |
| Tuple types `(T, U)` | Observed | Parser handles them |
| Shape literals `[M, N]` | Proven | `test_types.rs` |
| Option prelude (Some/None) | Proven | `milestone_2_6/option_result.rs` |
| Result prelude (Ok/Err) | Proven | `milestone_2_6/option_result.rs` |
| Error recovery | Proven | `test_parser.rs` |
| Negative literal patterns | Proven | `test_match_patterns.rs` |
| Automatic differentiation | Proven | `test_ad.rs` |
| Method dispatch | Proven | `test_dispatch.rs` |
| Byte string literals `b"..."` | Proven | `test_bytes_strings.rs` |
| Byte char literals `b'c'` | Proven | `test_bytes_strings.rs` |
| Raw string literals `r"..."` | Proven | `test_bytes_strings.rs` |
| Raw byte string literals `br"..."` | Proven | `test_bytes_strings.rs` |
| Regex literals `/pattern/flags` | Proven | `test_regex.rs` (77 tests) |
| Regex match `~=` | Proven | `test_regex.rs` |
| Regex not-match `!~` | Proven | `test_regex.rs` |
| Context-sensitive `/` (division vs regex) | Proven | `test_regex.rs` |
| ByteSlice methods | Proven | `test_bytes_strings.rs` |
| StrView methods | Proven | `test_bytes_strings.rs` |
| Regex/MIR-exec parity | Proven | `test_regex.rs` (12 parity tests) |
| Deterministic hashing (murmurhash3) | Proven | `test_determinism.rs`, `test_regression_gate.rs` |

---

## Appendix: Files Relied Upon

| File | Lines | Role in this document |
|------|-------|-----------------------|
| `crates/cjc-lexer/src/lib.rs` | ~581 | Sections 2 (Lexical Structure) |
| `crates/cjc-parser/src/lib.rs` | ~2483 | Sections 3-5, 10 (Grammar) |
| `crates/cjc-ast/src/lib.rs` | ~1145 | All AST node references |
| `crates/cjc-hir/src/lib.rs` | ~2205 | Section 7 (Desugaring) |
| `crates/cjc-mir/src/lib.rs` | ~820 | Section 7.5 (Lambda lifting) |
| `crates/cjc-types/src/lib.rs` | ~2797 | Section 6 (Types), 9 (Diagnostics) |
| `crates/cjc-diag/src/lib.rs` | — | Section 9 (Diagnostics) |
| `crates/cjc-dispatch/src/lib.rs` | — | Section 9 (Dispatch errors) |
| `tests/test_*.rs` | — | Evidence throughout |
| `tests/milestone_2_*/**` | — | Evidence throughout |

**Major TODOs not implemented (or not found in parser):**
- No `else if` as a single keyword (parsed as `else` + `if`; works fine but no dedicated AST node).
- No character literals (e.g., `'c'`); use `b'c'` for byte chars.
- No bitwise operators (`&`, `|`, `^`, `<<`, `>>`); single `&` is an error.
- No range-inclusive syntax (`..=`).
- No `break` / `continue` in loops.
- No pattern guards (`pattern if condition =>`).
- No `or` patterns (`A | B =>`).
- No string interpolation.
- No `async` / `await`.
- Tensor literal syntax `[| ... |]` not yet implemented.
- Transformer kernels (matmul, softmax, layernorm) not yet in Layer 2.
