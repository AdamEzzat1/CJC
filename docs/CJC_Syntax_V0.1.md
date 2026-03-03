# CJC Syntax Reference — V 0.1

**Version:** 0.1.0
**Date:** 2026-03-02
**Status:** Complete reference for all implemented syntax

---

## Table of Contents

1. [Lexical Structure](#1-lexical-structure)
2. [Declarations](#2-declarations)
3. [Statements](#3-statements)
4. [Expressions](#4-expressions)
5. [Patterns](#5-patterns)
6. [Types](#6-types)
7. [Operators & Precedence](#7-operators--precedence)
8. [Built-in Types](#8-built-in-types)
9. [Prelude](#9-prelude)
10. [Grammar Summary](#10-grammar-summary)

---

## 1. Lexical Structure

### 1.1 Keywords (26)

```
struct   class    fn       trait    impl     enum
let      mut      const    return   break    continue
if       else     while    for      in       match
import   as       sealed   nogc     col      true
false    _
```

### 1.2 Integer Literals

```cjc
42                  // decimal
0xFF                // hexadecimal (case-insensitive: 0xff, 0XFF)
0b1010              // binary (0b or 0B prefix)
0o777               // octal (0o or 0O prefix)
1_000_000           // underscore separators (ignored)
0xFF_FF             // separators in any base
```

**Type suffixes:** `i32`, `i64`, `f32`, `f64`

```cjc
42i64               // explicit i64
3f64                // explicit f64
```

### 1.3 Float Literals

```cjc
3.14                // standard
2.0                 // requires digit after dot
1.5e10              // scientific notation
2.5E-3              // negative exponent
```

### 1.4 String Literals

**Standard strings** — escape sequences supported:

```cjc
"hello world"
"line1\nline2"      // newline
"tab\there"         // tab
"quote: \""         // escaped quote
"null: \0"          // null byte
"backslash: \\"     // escaped backslash
```

**Escape sequences:** `\n`, `\t`, `\r`, `\\`, `\"`, `\0`

**Format strings** — interpolation with `{}`:

```cjc
f"Hello, {name}!"
f"result = {x + y}"
f"escaped brace: {{}}"   // prints literal {
f"{a} + {b} = {a + b}"
```

**Raw strings** — no escape processing:

```cjc
r"C:\Users\path"          // backslashes are literal
r#"contains "quotes""#    // hash delimiters for embedded quotes
```

**Byte strings:**

```cjc
b"hello"                  // byte string
b'A'                      // single byte (u8)
br"raw\bytes"             // raw byte string
```

**Regex literals:**

```cjc
/pattern/                 // basic regex
/\d+/                     // digits
/hello/i                  // case-insensitive flag
/^start.*end$/gm          // multiple flags
```

**Regex flags:** `i` (case-insensitive), `g` (global), `m` (multiline), `s` (dotall), `x` (extended)

### 1.5 Boolean Literals

```cjc
true
false
```

### 1.6 Tensor Literals

```cjc
[| 1.0, 2.0, 3.0 |]              // 1-D tensor (shape: [3])
[| 1.0, 2.0; 3.0, 4.0 |]        // 2-D tensor (shape: [2, 2])
[| 1, 2, 3; 4, 5, 6; 7, 8, 9 |] // 3x3 matrix
```

Semicolons (`;`) separate rows. Commas (`,`) separate elements within a row.

### 1.7 Comments

```cjc
// Single-line comment

/* Multi-line
   block comment */

/* Nested /* comments */ are supported */
```

Block comments nest correctly — inner `/* */` pairs are tracked.

### 1.8 Whitespace

Spaces, tabs, newlines, and carriage returns are whitespace. Whitespace is insignificant except inside strings.

---

## 2. Declarations

### 2.1 Functions

```
fn NAME(PARAMS) -> RETURN_TYPE { BODY }
fn NAME(PARAMS) { BODY }                    // implicit return
nogc fn NAME(PARAMS) -> RETURN_TYPE { BODY } // no-GC function
```

Parameters **require** type annotations:

```cjc
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn greet(name: str) {
    print(f"Hello, {name}!");
}

nogc fn fast_square(x: f64) -> f64 {
    x * x
}
```

### 2.2 Structs

```
struct NAME { FIELD: TYPE, ... }
struct NAME<TYPE_PARAMS> { FIELD: TYPE, ... }
```

```cjc
struct Point {
    x: f64,
    y: f64,
}

struct Pair<T, U> {
    first: T,
    second: U,
}

struct Config {
    width: i64,
    height: i64,
    title: str,
}
```

Fields can have default values:

```cjc
struct Options {
    verbose: bool = false,
    seed: i64 = 42,
}
```

### 2.3 Classes

```
class NAME { FIELD: TYPE, ... }
class NAME<TYPE_PARAMS> { FIELD: TYPE, ... }
```

```cjc
class Node<T> {
    value: T,
    next: Node<T>,
}
```

### 2.4 Enums

```
enum NAME { VARIANT, ... }
enum NAME<TYPE_PARAMS> { VARIANT, ... }
```

**Unit variants:**

```cjc
enum Color {
    Red,
    Green,
    Blue,
}
```

**Variants with payloads:**

```cjc
enum Shape {
    Circle(f64),
    Rect(f64, f64),
    Point,
}
```

**Generic enums:**

```cjc
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### 2.5 Traits

```
trait NAME { FN_SIGNATURES }
trait NAME: SUPER_TRAITS { FN_SIGNATURES }
sealed trait NAME { ... }
```

```cjc
trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
}

trait Printable: Display {
    fn to_string(self) -> str;
}

sealed trait Internal {
    fn secret() -> i64;
}
```

### 2.6 Implementations

```
impl TYPE { METHODS }
impl TRAIT for TYPE { METHODS }
impl<TYPE_PARAMS> TYPE { METHODS }
```

```cjc
impl Point {
    fn distance(self: Point) -> f64 {
        sqrt(self.x ** 2.0 + self.y ** 2.0)
    }
}

impl Numeric for i64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}
```

### 2.7 Imports

```
import MODULE_PATH
import MODULE_PATH as ALIAS
```

```cjc
import math.linalg
import utils.helpers as h
```

### 2.8 Constants

```
const NAME: TYPE = VALUE;
```

```cjc
const MAX_SIZE: i64 = 1024;
const PI: f64 = 3.14159265358979;
```

---

## 3. Statements

### 3.1 Let Binding

```
let NAME: TYPE = EXPR;
let mut NAME: TYPE = EXPR;
let NAME = EXPR;                  // type inferred from literal
```

```cjc
let x: i64 = 42;
let mut count: i64 = 0;
let name: str = "CJC";
let pi = 3.14;                    // inferred f64
```

### 3.2 Expression Statement

Any expression followed by `;`:

```cjc
print("hello");
x = x + 1;
foo(bar);
```

### 3.3 Return

```
return EXPR;
return;
```

```cjc
fn abs(x: i64) -> i64 {
    if x < 0 {
        return -x;
    }
    x
}
```

### 3.4 Break and Continue

```
break;
continue;
```

Valid only inside `while` and `for` loops.

### 3.5 If Statement

```
if CONDITION { BODY }
if CONDITION { BODY } else { BODY }
if CONDITION { BODY } else if CONDITION { BODY } else { BODY }
```

```cjc
if x > 0 {
    print("positive");
} else if x == 0 {
    print("zero");
} else {
    print("negative");
}
```

**No semicolon** after the closing `}` of if/else blocks inside function bodies.

### 3.6 While Loop

```
while CONDITION { BODY }
```

```cjc
let mut i: i64 = 0;
while i < 10 {
    print(i);
    i += 1;
}
```

### 3.7 For Loop

```
for NAME in START..END { BODY }     // range (exclusive end)
for NAME in EXPR { BODY }           // iterator
```

```cjc
// Range iteration
for i in 0..10 {
    print(i);
}

// Array iteration
let items = [1, 2, 3, 4, 5];
for item in items {
    print(item);
}
```

### 3.8 NoGC Block

```
nogc { BODY }
```

```cjc
nogc {
    let result = x * y + z;
    result
}
```

Statically verified to contain no allocating operations.

---

## 4. Expressions

### 4.1 Literals

```cjc
42                          // integer
3.14                        // float
"hello"                     // string
f"x = {x}"                 // format string
true                        // boolean
[1, 2, 3]                  // array
(1, "two", 3.0)            // tuple
[| 1.0, 2.0; 3.0, 4.0 |]  // tensor
```

### 4.2 Identifiers

```cjc
x
my_variable
_unused
camelCase
CONSTANT
```

### 4.3 Binary Operations

```cjc
a + b       // add
a - b       // subtract
a * b       // multiply
a / b       // divide
a % b       // modulo
a ** b      // power (right-associative)

a == b      // equal
a != b      // not equal
a < b       // less than
a > b       // greater than
a <= b      // less or equal
a >= b      // greater or equal

a && b      // logical AND (short-circuit)
a || b      // logical OR (short-circuit)

a & b       // bitwise AND (integers only)
a | b       // bitwise OR
a ^ b       // bitwise XOR
a << n      // left shift
a >> n      // right shift

a ~= pat    // regex match (returns bool)
a !~ pat    // regex not match
```

### 4.4 Unary Operations

```cjc
-x          // numeric negation
!x          // logical NOT
~x          // bitwise NOT (integers only)
```

### 4.5 Assignment

```cjc
x = 10;

// Compound assignments
x += 5;     x -= 3;     x *= 2;
x /= 4;     x %= 7;     x **= 2;
x &= mask;  x |= flag;  x ^= bits;
x <<= 1;    x >>= 1;
```

### 4.6 Function Call

```cjc
foo(a, b)
bar()
create(width: 10, height: 20)   // named arguments
```

### 4.7 Method Call

```cjc
point.distance()
tensor.reshape([2, 3])
list.push(item)
text.len()
```

### 4.8 Field Access

```cjc
point.x
config.width
result.value
```

### 4.9 Indexing

```cjc
arr[0]              // single index
arr[i]              // variable index
matrix[i, j]        // multi-index (tensors)
```

### 4.10 Lambda / Closure

```
|PARAMS| BODY
```

```cjc
|x: i64| x * 2
|a: f64, b: f64| a + b
|| 42                          // no parameters
|x: i64| { let y = x * 2; y } // block body
```

Closures capture outer variables automatically:

```cjc
fn make_adder(n: i64) -> fn(i64) -> i64 {
    |x: i64| x + n     // captures `n`
}
```

### 4.11 If Expression

```cjc
let result = if condition { value_a } else { value_b };

let x = if n > 0 { n } else if n == 0 { 0 } else { -n };
```

Returns the value of the taken branch. Returns `Void` if no else branch and condition is false.

### 4.12 Match Expression

```
match SCRUTINEE {
    PATTERN => EXPR,
    PATTERN => EXPR,
    ...
}
```

```cjc
let label = match x {
    0 => "zero",
    1 => "one",
    _ => "other",
};

match shape {
    Circle(r) => 3.14 * r * r,
    Rect(w, h) => w * h,
    _ => 0.0,
}

match option_val {
    Some(x) => x * 2,
    None => 0,
}
```

### 4.13 Pipe Operator

```
EXPR |> CALL
```

The pipe operator threads the left-hand value as the first argument of the right-hand function call:

```cjc
// These are equivalent:
data |> transform() |> output()
output(transform(data))

// Data pipeline
df |> filter(col("age") > 18)
   |> select("name", "age")
   |> arrange(desc("age"))
```

### 4.14 Block Expression

```cjc
let result = {
    let a = 10;
    let b = 20;
    a + b           // tail expression (no semicolon) = return value
};
```

The last expression in a block (without trailing `;`) is the block's value.

### 4.15 Struct Literal

```cjc
Point { x: 1.0, y: 2.0 }
Config { width: 800, height: 600, title: "App" }
```

### 4.16 Tuple Literal

```cjc
(1, 2, 3)
("name", 42, true)
()                      // unit/empty tuple
```

### 4.17 Array Literal

```cjc
[1, 2, 3, 4, 5]
["alpha", "beta", "gamma"]
[]                      // empty array
```

### 4.18 Enum Variant Construction

```cjc
// Unit variants
None
Red
Blue

// Payload variants (function-call syntax)
Some(42)
Ok("success")
Err("failure")
Circle(3.14)
```

Qualified syntax:

```cjc
Color::Red
Option::Some(42)
Shape::Circle(3.14)
```

### 4.19 Try Operator

```cjc
let value = risky_operation()?;
```

Desugars to matching on `Result`: returns `Err` early if the expression is `Err`, unwraps `Ok` otherwise.

### 4.20 Range Expression

```cjc
0..10       // range from 0 to 9 (exclusive end)
start..end  // variable range
```

Used primarily in `for` loops.

### 4.21 Column Reference (Data DSL)

```cjc
col("column_name")     // reference a DataFrame column
desc("column_name")    // descending sort specification
asc("column_name")     // ascending sort specification
```

---

## 5. Patterns

Patterns are used in `match` arms and destructuring.

### 5.1 Wildcard

```cjc
_ => default_value
```

Matches anything, binds nothing.

### 5.2 Binding

```cjc
x => x * 2          // binds the matched value to `x`
name => print(name)
```

### 5.3 Literal Patterns

```cjc
42 => "forty-two"
3.14 => "pi"
true => "yes"
false => "no"
"hello" => "greeting"
```

### 5.4 Tuple Destructuring

```cjc
(a, b) => a + b
(x, y, z) => x * y * z
(_, second) => second       // ignore first element
```

### 5.5 Struct Destructuring

```cjc
Point { x, y } => x + y                    // shorthand
Point { x: px, y: py } => px + py          // renamed bindings
```

### 5.6 Variant Patterns

```cjc
Some(x) => x
None => 0
Circle(r) => r * r
Rect(w, h) => w * h
Ok(value) => value
Err(msg) => print(msg)
```

### 5.7 Nested Patterns

```cjc
Some((a, b)) => a + b       // variant containing tuple
Some(Point { x, y }) => x   // variant containing struct
```

---

## 6. Types

### 6.1 Primitive Types

| Type | Description |
|------|-------------|
| `i64` | 64-bit signed integer |
| `i32` | 32-bit signed integer |
| `f64` | 64-bit floating point |
| `f32` | 32-bit floating point |
| `bool` | Boolean (true/false) |
| `str` | String |
| `()` | Unit type (void) |

### 6.2 Collection Types

```cjc
[i64]           // array of i64
[str]           // array of strings
Vec<i64>        // vector
Map<str, i64>   // ordered map
Set<str>        // ordered set
```

### 6.3 Tensor Types

```cjc
Tensor              // dynamic-shape tensor (f64)
Tensor<f32>         // typed tensor
Tensor<f64, [3, 4]> // shaped tensor (3x4)
```

### 6.4 Tuple Types

```cjc
(i64, f64)
(str, i64, bool)
()                  // unit
```

### 6.5 Function Types

```cjc
fn(i64) -> i64
fn(f64, f64) -> f64
fn() -> bool
```

### 6.6 Generic Types

```cjc
Option<i64>
Result<str, str>
Vec<Point>
Pair<i64, str>
```

### 6.7 Type Parameters

```cjc
fn identity<T>(x: T) -> T { x }

struct Container<T: Clone> {
    value: T,
}

impl<T> Container<T> {
    fn get(self: Container<T>) -> T { self.value }
}
```

### 6.8 Special Types

| Type | Description |
|------|-------------|
| `Any` | Dynamic type (bypasses type checking) |
| `Self` | Current impl type |
| `Option<T>` | Optional value (Some/None) |
| `Result<T, E>` | Error-capable value (Ok/Err) |
| `Complex` | Complex number (f64 real + imaginary) |

---

## 7. Operators & Precedence

Operators listed from **highest** to **lowest** precedence:

| Precedence | Category | Operators | Associativity |
|------------|----------|-----------|---------------|
| 26 | Postfix | `.` `[` `()` `?` | Left |
| 24 | Unary | `-` `!` `~` | Right (prefix) |
| 22 | Power | `**` | **Right** |
| 20 | Multiplicative | `*` `/` `%` | Left |
| 18 | Additive | `+` `-` | Left |
| 16 | Shift | `<<` `>>` | Left |
| 14 | Comparison | `<` `>` `<=` `>=` | Left |
| 12 | Equality | `==` `!=` `~=` `!~` | Left |
| 11 | Bitwise AND | `&` | Left |
| 10 | Bitwise XOR | `^` | Left |
| 9 | Bitwise OR | `|` | Left |
| 8 | Logical AND | `&&` | Left |
| 6 | Logical OR | `||` | Left |
| 4 | Pipe | `|>` | Left |
| 2 | Assignment | `=` `+=` `-=` `*=` `/=` `%=` `**=` `&=` `|=` `^=` `<<=` `>>=` | **Right** |

### Precedence Examples

```cjc
2 * 3 ** 2          // = 2 * 9 = 18    (** binds tighter than *)
2 ** 3 ** 2         // = 2 ** 9 = 512  (** is right-associative)
a + b * c           // = a + (b * c)
a && b || c         // = (a && b) || c
x & 0xFF | flag     // = (x & 0xFF) | flag
data |> f() |> g()  // = g(f(data))
```

---

## 8. Built-in Types

### 8.1 Tensor

```cjc
// Constructors
Tensor.zeros([3, 3])        // all zeros
Tensor.ones([2, 4])         // all ones
Tensor.eye(3)               // 3x3 identity
Tensor.randn([2, 3])        // random normal
Tensor.uniform([10])        // random uniform [0,1)
Tensor.full([2, 2], 5.0)   // filled with value
Tensor.linspace(0.0, 1.0, 10)  // evenly spaced
Tensor.arange(0.0, 10.0, 1.0)  // range with step
Tensor.from_vec(data, shape)    // from vector
Tensor.diag(values)             // diagonal matrix

// Methods
t.sum()                     // Kahan-stable sum
t.mean()                    // mean
t.reshape([4, 2])          // zero-copy reshape
t.transpose()              // transpose
t.add(other)               // element-wise add
t.mul_elem(other)          // element-wise multiply
t.softmax()                // softmax
t.relu()                   // ReLU activation
t.shape()                  // shape as array
t.len()                    // total elements
t.get(indices)             // element access
t.set(indices, value)      // element mutation
```

### 8.2 Complex Numbers

```cjc
let z = Complex(3.0, 4.0);     // 3 + 4i
z.re()                          // real part (3.0)
z.im()                          // imaginary part (4.0)
z.abs()                         // magnitude (5.0)
z.conj()                        // conjugate (3 - 4i)
z.norm_sq()                     // |z|^2 (25.0)

// Arithmetic: +, -, *, / all supported
let w = z + Complex(1.0, -2.0);
let product = z * w;
```

### 8.3 Map and Set

```cjc
let m = Map.new();              // ordered map
// Methods: get, set, contains, keys, values, len

let s = Set.new();              // ordered set
// Methods: insert, contains, remove, len
```

---

## 9. Prelude

The following types and variants are available without import:

### Option

```cjc
Some(value)     // wraps a value
None            // absence of value

// Methods
opt.unwrap()            // extract or panic
opt.unwrap_or(default)  // extract or default
opt.is_some()           // returns bool
opt.is_none()           // returns bool
```

### Result

```cjc
Ok(value)       // success
Err(error)      // failure

// Try operator
let x = operation()?;   // returns Err early on failure
```

---

## 10. Grammar Summary

### Program Structure

```
Program     = Declaration*
Declaration = FnDecl | StructDecl | ClassDecl | EnumDecl
            | TraitDecl | ImplDecl | ImportDecl | ConstDecl
            | LetStmt | Statement
```

### Statements

```
Statement   = LetStmt | ExprStmt | ReturnStmt
            | BreakStmt | ContinueStmt
            | IfStmt | WhileStmt | ForStmt | NoGcBlock
```

### Expressions

```
Expression  = Literal | Ident | BinaryExpr | UnaryExpr
            | CallExpr | MethodExpr | FieldExpr | IndexExpr
            | LambdaExpr | MatchExpr | IfExpr | BlockExpr
            | AssignExpr | CompoundAssignExpr
            | StructLitExpr | TupleLitExpr | ArrayLitExpr
            | TensorLitExpr | PipeExpr | RangeExpr | TryExpr
```

### Patterns

```
Pattern     = WildcardPat | BindingPat | LiteralPat
            | TuplePat | StructPat | VariantPat
```

### Types

```
TypeExpr    = NamedType | GenericType | TupleType
            | ArrayType | FnType | ShapeType
```

---

## Syntax Rules & Conventions

1. **Semicolons** end statements. Block statements (`if`, `while`, `for`) do **not** require trailing semicolons inside function bodies.

2. **Tail expressions** — the last expression in a block without a trailing `;` becomes the block's return value.

3. **Type annotations** are required on function parameters. Let bindings can often omit types when the literal type is unambiguous.

4. **Immutable by default** — use `let mut` to declare mutable variables.

5. **Right-associative operators**: `**` (power) and `=`/`+=`/etc. (assignment).

6. **Short-circuit evaluation**: `&&` and `||` do not evaluate the right operand if the left determines the result.

7. **Bitwise operators** work only on integer types. Using them on floats produces a runtime error.

8. **The pipe operator** `|>` inserts the left-hand value as the first argument of the right-hand function call.

9. **Format strings** use `{}` for interpolation. Use `{{` and `}}` for literal braces.

10. **Tensor literals** use `[| ... |]` delimiters. Semicolons separate rows; commas separate elements.
