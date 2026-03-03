# CJC Syntax Reference

**Version:** 0.1
**Status:** Complete reference for all implemented syntax constructs

---

## Table of Contents

1. [Lexical Structure](#1-lexical-structure)
2. [Declarations](#2-declarations)
3. [Statements](#3-statements)
4. [Expressions](#4-expressions)
5. [Patterns](#5-patterns)
6. [Types](#6-types)
7. [Operators and Precedence](#7-operators-and-precedence)
8. [Built-in Functions](#8-built-in-functions)
9. [Prelude](#9-prelude)
10. [Syntax Rules and Conventions](#10-syntax-rules-and-conventions)
11. [Grammar Summary](#11-grammar-summary)

---

## 1. Lexical Structure

### 1.1 Keywords (26)

```
struct   class    record   enum     fn       trait
impl     let      mut      const    import   as
return   break    continue if       else     while
for      in       match    nogc     col      sealed
true     false
```

The underscore `_` is a reserved identifier used as the wildcard pattern.

### 1.2 Integer Literals

```cjc
42                  // decimal
0xFF                // hexadecimal (case-insensitive: 0xff, 0XFF)
0b1010              // binary (0b or 0B prefix)
0o777               // octal (0o or 0O prefix)
1_000_000           // underscore separators (ignored by the compiler)
0xFF_FF             // separators work in any base
```

**Type suffixes** select a specific integer type:

```cjc
42i64               // explicit i64
42i32               // explicit i32
```

Without a suffix, integer literals default to `i64`.

### 1.3 Float Literals

```cjc
3.14                // standard float
2.0                 // requires digit after the decimal point
1.5e10              // scientific notation
2.5E-3              // negative exponent (case-insensitive E)
```

**Type suffixes:**

```cjc
3.14f64             // explicit f64
3.14f32             // explicit f32
3f64                // integer form with float suffix
```

Without a suffix, float literals default to `f64`.

### 1.4 String Literals

**Standard strings** support escape sequences:

```cjc
"hello world"
"line1\nline2"          // newline
"tab\there"             // tab
"quote: \""             // escaped double quote
"null: \0"              // null byte
"backslash: \\"         // escaped backslash
"carriage return: \r"   // carriage return
```

| Escape | Meaning         |
|--------|-----------------|
| `\n`   | Newline         |
| `\t`   | Tab             |
| `\r`   | Carriage return |
| `\\`   | Backslash       |
| `\"`   | Double quote    |
| `\0`   | Null byte       |

**Format strings** use the `f` prefix and `{}` for interpolation:

```cjc
f"Hello, {name}!"
f"result = {x + y}"
f"escaped brace: {{}}"     // prints literal {
f"{a} + {b} = {a + b}"
```

**Raw strings** skip all escape processing:

```cjc
r"C:\Users\path"               // backslashes are literal
r#"contains "quotes" inside"#  // hash delimiters for embedded quotes
```

**Byte strings** produce byte data:

```cjc
b"hello"                // byte string (Vec<u8>)
b'A'                    // single byte literal (u8 value 65)
b'\n'                   // byte 10
b'\xff'                 // byte 255 (hex escape)
br"raw\bytes"           // raw byte string (no escape processing)
br#"raw with "quotes""# // raw byte string with hash delimiters
```

### 1.5 Boolean Literals

```cjc
true
false
```

### 1.6 Regex Literals

```cjc
/pattern/               // basic regex
/\d+/                   // match digits
/hello/i                // case-insensitive
/^start.*end$/gm        // multiple flags
```

| Flag | Meaning                     |
|------|-----------------------------|
| `i`  | Case-insensitive            |
| `g`  | Global (match all)          |
| `m`  | Multiline                   |
| `s`  | Dotall (`.` matches newline)|
| `x`  | Extended (ignore whitespace)|

### 1.7 Tensor Literals

```cjc
[| 1.0, 2.0, 3.0 |]                // 1-D tensor, shape [3]
[| 1.0, 2.0; 3.0, 4.0 |]           // 2-D tensor, shape [2, 2]
[| 1, 2, 3; 4, 5, 6; 7, 8, 9 |]   // 3x3 matrix
```

Semicolons (`;`) separate rows. Commas (`,`) separate elements within a row.

### 1.8 Comments

```cjc
// Single-line comment

/* Multi-line
   block comment */

/* Nested /* comments */ are supported */
```

Block comments nest correctly -- inner `/* */` pairs are tracked by depth.

### 1.9 Whitespace

Spaces, tabs, newlines, and carriage returns are whitespace. Whitespace is insignificant except inside string literals. There are no indentation rules.

---

## 2. Declarations

### 2.1 Functions

```
fn NAME(PARAMS) -> RETURN_TYPE { BODY }
fn NAME(PARAMS) { BODY }                         // void return
fn NAME<TYPE_PARAMS>(PARAMS) -> RET { BODY }     // generic
fn NAME(PARAMS) -> RET / EFFECT { BODY }         // with effect annotation
nogc fn NAME(PARAMS) -> RET { BODY }             // no-GC guarantee
```

Parameters **require** type annotations:

```cjc
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn greet(name: str) {
    print(f"Hello, {name}!");
}

// Generic function
fn identity<T>(x: T) -> T {
    x
}

// NoGC function -- statically verified to perform no allocation
nogc fn fast_square(x: f64) -> f64 {
    x * x
}

// Effect annotation
fn pure_add(a: i64, b: i64) -> i64 / pure {
    a + b
}
```

### 2.2 Structs (value type)

```
struct NAME { FIELD: TYPE, ... }
struct NAME<TYPE_PARAMS> { FIELD: TYPE, ... }
```

```cjc
struct Point {
    x: f64,
    y: f64,
}

// Generic struct
struct Pair<T, U> {
    first: T,
    second: U,
}

// Fields with default values
struct Options {
    verbose: bool = false,
    seed: i64 = 42,
}
```

Structs are stack-allocated value types. Assignment copies all fields.

### 2.3 Records (immutable value type)

```
record NAME { FIELD: TYPE, ... }
```

```cjc
record Color {
    r: u8,
    g: u8,
    b: u8,
}
```

Records are immutable structs -- all fields are read-only after construction.

### 2.4 Classes (GC reference type)

```
class NAME { FIELD: TYPE, ... }
class NAME<TYPE_PARAMS> { FIELD: TYPE, ... }
```

```cjc
class Node<T> {
    value: T,
    next: Node<T>,
}

class TreeNode {
    value: i64,
    left: TreeNode,
    right: TreeNode,
}
```

Classes are garbage-collected reference types. Assignment copies the reference, not the data. Self-referential types require `class`.

### 2.5 Enums (algebraic data types)

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

**Tuple variants (with payloads):**

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

### 2.6 Traits

```
trait NAME { FN_SIGNATURES }
trait NAME: SUPER_TRAITS { FN_SIGNATURES }
sealed trait NAME { FN_SIGNATURES }
```

```cjc
trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
}

// Trait with super-trait bounds
trait Printable: Display {
    fn to_string(self: Self) -> str;
}

// Sealed trait -- cannot be implemented outside its defining module
sealed trait Internal {
    fn secret() -> i64;
}

// Generic trait
trait Container<T> {
    fn get(self: Self, idx: i64) -> T;
    fn len(self: Self) -> i64;
}
```

### 2.7 Implementations

```
impl TYPE { METHODS }
impl TRAIT for TYPE { METHODS }
impl TYPE : TRAIT { METHODS }
impl<TYPE_PARAMS> TYPE { METHODS }
```

```cjc
// Methods on a type
impl Point {
    fn distance(self: Point) -> f64 {
        sqrt(self.x ** 2.0 + self.y ** 2.0)
    }
}

// Trait implementation
impl Numeric for i64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

// Alternative trait impl syntax
impl Circle : Area {
    fn area(self: Circle) -> f64 {
        3.14159 * self.radius * self.radius
    }
}
```

### 2.8 Imports

```
import MODULE_PATH
import MODULE_PATH as ALIAS
```

```cjc
import math.linalg
import utils.helpers as h
```

### 2.9 Constants

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
let NAME = EXPR;               // type inferred
```

Variables are **immutable by default**. Use `let mut` for mutable bindings.

```cjc
let x: i64 = 42;
let mut count: i64 = 0;
let name: str = "CJC";
let pi = 3.14;                 // inferred as f64
```

### 3.2 Expression Statement

Any expression followed by a semicolon:

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
    x       // tail expression also works as return
}
```

### 3.4 Break and Continue

```
break;
continue;
```

Valid only inside `while` and `for` loops.

```cjc
let mut i: i64 = 0;
while i < 100 {
    i += 1;
    if i % 2 == 0 {
        continue;
    }
    if i > 50 {
        break;
    }
    print(i);
}
```

### 3.5 If / Else

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

No semicolon after the closing `}` of if/else blocks inside function bodies.

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
for NAME in EXPR { BODY }          // collection iteration
```

```cjc
// Range iteration (0 through 9)
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

All operations inside a `nogc` block are statically verified to contain no allocating operations (no GC, no heap allocation).

---

## 4. Expressions

### 4.1 Literals

```cjc
42                              // integer
3.14                            // float
"hello"                         // string
f"x = {x}"                     // format string
true                            // boolean
[1, 2, 3]                      // array
(1, "two", 3.0)                // tuple
[| 1.0, 2.0; 3.0, 4.0 |]      // tensor
/\d+/                           // regex
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

**Arithmetic:**

```cjc
a + b           // addition
a - b           // subtraction
a * b           // multiplication
a / b           // division
a % b           // modulo (remainder)
a ** b          // power (right-associative)
```

**Comparison:**

```cjc
a == b          // equal
a != b          // not equal
a < b           // less than
a > b           // greater than
a <= b          // less than or equal
a >= b          // greater than or equal
```

**Logical (short-circuit):**

```cjc
a && b          // logical AND
a || b          // logical OR
```

**Bitwise (integer types only):**

```cjc
a & b           // bitwise AND
a | b           // bitwise OR
a ^ b           // bitwise XOR
a << n          // left shift
a >> n          // right shift
```

**Regex:**

```cjc
text ~= /pattern/      // regex match (returns bool)
text !~ /pattern/       // regex non-match (returns bool)
```

### 4.4 Unary Operations

```cjc
-x              // numeric negation
!x              // logical NOT
~x              // bitwise NOT (integer types only)
```

### 4.5 Assignment

```cjc
x = 10;

// Compound assignment operators
x += 5;         x -= 3;         x *= 2;
x /= 4;         x %= 7;         x **= 2;
x &= mask;      x |= flag;      x ^= bits;
x <<= 1;        x >>= 1;
```

Assignment targets must be `let mut` bindings.

### 4.6 Function Call

```cjc
foo(a, b)
bar()
create(width: 10, height: 20)      // named arguments
```

### 4.7 Method Call

```cjc
point.distance()
tensor.reshape([2, 3])
text.len()
```

Method calls can also be written as qualified static calls:

```cjc
Point.distance(point)
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
matrix[i, j]        // multi-dimensional index (tensors)
```

### 4.10 Lambda / Closure

```
|PARAMS| BODY
```

```cjc
|x: i64| x * 2
|a: f64, b: f64| a + b
|| 42                               // no parameters
|x: i64| { let y = x * 2; y }      // block body
```

Closures capture outer variables automatically:

```cjc
fn make_adder(n: i64) -> fn(i64) -> i64 {
    |x: i64| x + n         // captures n from enclosing scope
}

let add5 = make_adder(5);
add5(10)                    // returns 15
```

### 4.11 If Expression

When used as an expression, `if` returns the value of the taken branch:

```cjc
let result = if condition { value_a } else { value_b };

let sign = if n > 0 { 1 } else if n == 0 { 0 } else { -1 };
```

Both branches must produce the same type. Returns `Void` if no else branch and condition is false.

### 4.12 Match Expression

```
match SCRUTINEE {
    PATTERN => EXPR,
    PATTERN => EXPR,
    ...
}
```

```cjc
// Literal patterns
let label = match x {
    0 => "zero",
    1 => "one",
    _ => "other",
};

// Enum variant patterns
match shape {
    Circle(r) => 3.14 * r * r,
    Rect(w, h) => w * h,
    _ => 0.0,
}

// Option matching
match option_val {
    Some(x) => x * 2,
    None => 0,
}

// Result matching
match result_val {
    Ok(v) => print(v),
    Err(e) => print(f"Error: {e}"),
}
```

Match expressions on enum types must be **exhaustive** -- all variants must be covered, or a wildcard `_` must be present. Non-exhaustive matches produce a compile-time error.

### 4.13 Pipe Operator

```
EXPR |> CALL
```

The pipe operator threads the left-hand value as the first argument of the right-hand function call:

```cjc
// These are equivalent:
data |> transform() |> output()
output(transform(data))

// Data pipeline example
df |> filter(col("age") > 18)
   |> select("name", "age")
   |> arrange(desc("age"))
```

### 4.14 Block Expression

A block evaluates to the value of its last expression (the "tail expression"), which must not have a trailing semicolon:

```cjc
let result = {
    let a = 10;
    let b = 20;
    a + b           // tail expression = block return value
};
```

### 4.15 Struct Literal

```cjc
Point { x: 1.0, y: 2.0 }
Config { width: 800, height: 600, title: "App" }
Options { verbose: true }      // unmentioned fields use defaults
```

### 4.16 Tuple Literal

```cjc
(1, 2, 3)
("name", 42, true)
()                      // unit / empty tuple
```

### 4.17 Array Literal

```cjc
[1, 2, 3, 4, 5]
["alpha", "beta", "gamma"]
[]                      // empty array
```

**Important:** Arrays are immutable values. Use `array_push(arr, val)` which returns a **new** array:

```cjc
let mut items: [i64] = [];
items = array_push(items, 1);
items = array_push(items, 2);
// items is now [1, 2]
```

### 4.18 Enum Variant Construction

```cjc
// Unit variants
None
Red
Blue

// Tuple variants (function-call syntax)
Some(42)
Ok("success")
Err("failure")
Circle(3.14)

// Qualified syntax
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
0..10           // range from 0 to 9 (exclusive end)
start..end      // variable range
```

Used primarily in `for` loops.

### 4.21 Column Reference (Data DSL)

```cjc
col("column_name")         // reference a DataFrame column
desc("column_name")        // descending sort specification
asc("column_name")         // ascending sort specification
```

---

## 5. Patterns

Patterns are used in `match` arms and destructuring bindings.

### 5.1 Wildcard

```cjc
_ => default_value
```

Matches anything, binds nothing. Ensures exhaustiveness.

### 5.2 Binding

```cjc
x => x * 2
name => print(name)
```

Matches anything and binds the matched value to the given name.

**Note:** If the binding name matches a known enum variant name, it is treated as a variant pattern, not a binding.

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
(_, second) => second           // ignore first element
```

### 5.5 Struct Destructuring

```cjc
Point { x, y } => x + y                    // shorthand (field name = binding name)
Point { x: px, y: py } => px + py          // renamed bindings
```

### 5.6 Enum Variant Patterns

```cjc
Some(x) => x
None => 0
Circle(r) => r * r
Rect(w, h) => w * h
Ok(value) => value
Err(msg) => print(msg)
```

### 5.7 Nested Patterns

Patterns compose:

```cjc
Some((a, b)) => a + b               // variant containing a tuple
Some(Point { x, y }) => x           // variant containing a struct
```

---

## 6. Types

### 6.1 Primitive Types

| Type   | Size    | Description                        |
|--------|---------|------------------------------------|
| `i32`  | 32-bit  | Signed integer                     |
| `i64`  | 64-bit  | Signed integer (default for int literals) |
| `u8`   | 8-bit   | Unsigned byte                      |
| `f32`  | 32-bit  | IEEE 754 single-precision float    |
| `f64`  | 64-bit  | IEEE 754 double-precision float (default for float literals) |
| `f16`  | 16-bit  | IEEE 754 half-precision float      |
| `bf16` | 16-bit  | Brain float (ML-oriented)          |
| `bool` |         | Boolean (`true` / `false`)         |
| `str`  |         | UTF-8 string                       |
| `()`   |         | Unit type (void / no value)        |

### 6.2 Collection Types

```cjc
[i64]               // array of i64
[str]               // array of strings
Map<str, i64>       // ordered map
Set<str>            // ordered set
```

### 6.3 Tensor Types

```cjc
Tensor               // dynamic-shape tensor (f64 elements)
Tensor<f32>          // tensor with explicit element type
Tensor<f64, [3, 4]>  // shaped tensor (3x4 matrix)
```

### 6.4 Tuple Types

```cjc
(i64, f64)
(str, i64, bool)
()                   // unit
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
Pair<i64, str>
Container<f64>
```

### 6.7 Type Parameters and Bounds

```cjc
fn identity<T>(x: T) -> T { x }

struct Container<T: Clone> {
    value: T,
}

fn sum<T: Numeric>(values: Tensor<T>) -> T {
    // ...
}

// Multiple bounds
fn print_all<T: Display + Debug>(items: [T]) {
    // ...
}
```

### 6.8 Special Types

| Type          | Description                                   |
|---------------|-----------------------------------------------|
| `Any`         | Dynamic type -- bypasses static type checking |
| `Self`        | The implementing type (inside `impl` blocks)  |
| `Option<T>`   | Optional value (`Some(T)` or `None`)          |
| `Result<T,E>` | Success/failure (`Ok(T)` or `Err(E)`)         |
| `Complex`     | Complex number (f64 real + imaginary parts)   |
| `Regex`       | Compiled regex pattern                        |
| `Bytes`       | Owned byte buffer                             |
| `ByteSlice`   | Non-owning byte slice (zero-copy view)        |
| `StrView`     | Validated UTF-8 string view (zero-copy)       |
| `Buffer<T>`   | COW reference-counted buffer                  |
| `SparseTensor<f64>` | COO-format sparse tensor               |

### 6.9 Memory Layers

CJC uses a three-layer memory model:

| Layer    | Allocation       | Types                                                  |
|----------|------------------|--------------------------------------------------------|
| `@nogc`  | Zero allocation  | `i32`, `i64`, `u8`, `f32`, `f64`, `bf16`, `bool`, `ByteSlice`, `StrView` |
| COW      | Deterministic RC | `Tensor`, `Array`, `String`, `Struct`, `Enum`, `Buffer<T>`, `Bytes` |
| GC       | Mark-sweep GC    | `class` types (reference types, self-referential)      |

---

## 7. Operators and Precedence

Operators are listed from **highest** to **lowest** precedence:

| Prec | Category       | Operators                                                      | Associativity |
|------|----------------|----------------------------------------------------------------|---------------|
| 26   | Postfix        | `.` `[]` `()` `?`                                              | Left          |
| 24   | Unary          | `-` `!` `~`                                                    | Right (prefix)|
| 22   | Power          | `**`                                                           | **Right**     |
| 20   | Multiplicative | `*` `/` `%`                                                    | Left          |
| 18   | Additive       | `+` `-`                                                        | Left          |
| 16   | Shift          | `<<` `>>`                                                      | Left          |
| 14   | Comparison     | `<` `>` `<=` `>=`                                              | Left          |
| 12   | Equality       | `==` `!=` `~=` `!~`                                           | Left          |
| 11   | Bitwise AND    | `&`                                                            | Left          |
| 10   | Bitwise XOR    | `^`                                                            | Left          |
| 9    | Bitwise OR     | `\|`                                                           | Left          |
| 8    | Logical AND    | `&&`                                                           | Left          |
| 6    | Logical OR     | `\|\|`                                                         | Left          |
| 4    | Pipe           | `\|>`                                                          | Left          |
| 2    | Assignment     | `=` `+=` `-=` `*=` `/=` `%=` `**=` `&=` `\|=` `^=` `<<=` `>>=` | **Right**     |

### Precedence Examples

```cjc
2 * 3 ** 2              // = 2 * 9 = 18        (** binds tighter than *)
2 ** 3 ** 2             // = 2 ** 9 = 512      (** is right-associative)
a + b * c               // = a + (b * c)
a && b || c             // = (a && b) || c
x & 0xFF | flag         // = (x & 0xFF) | flag
data |> f() |> g()      // = g(f(data))
```

### Special Tokens

| Token | Description                        |
|-------|------------------------------------|
| `..`  | Range operator                     |
| `\|>` | Pipe operator                      |
| `\|`  | Lambda parameter delimiter         |
| `[|`  | Tensor literal open                |
| `\|]` | Tensor literal close               |
| `~=`  | Regex match operator               |
| `!~`  | Regex non-match operator           |
| `->`  | Return type / function type arrow  |
| `=>`  | Match arm fat arrow                |

---

## 8. Built-in Functions

### 8.1 I/O

| Function       | Signature              | Description                    |
|----------------|------------------------|--------------------------------|
| `print`        | `(Any...) -> ()`       | Print values to stdout         |
| `read_line`    | `() -> str`            | Read a line from stdin         |
| `to_string`    | `(Any) -> str`         | Convert any value to string    |

### 8.2 Math -- Core

| Function    | Signature               | Description                 |
|-------------|-------------------------|-----------------------------|
| `sqrt`      | `(f64) -> f64`          | Square root                 |
| `abs`       | `(f64) -> f64`          | Absolute value              |
| `floor`     | `(f64) -> f64`          | Floor                       |
| `ceil`      | `(f64) -> f64`          | Ceiling                     |
| `round`     | `(f64) -> f64`          | Round to nearest integer    |
| `min`       | `(f64, f64) -> f64`     | Minimum                     |
| `max`       | `(f64, f64) -> f64`     | Maximum                     |
| `sign`      | `(f64) -> f64`          | Sign (-1, 0, or 1)         |
| `pow`       | `(f64, f64) -> f64`     | Power                       |
| `hypot`     | `(f64, f64) -> f64`     | Hypotenuse                  |
| `clamp`     | `(f64, f64, f64) -> f64`| Clamp value to range        |

### 8.3 Math -- Trigonometric

| Function   | Description            |
|------------|------------------------|
| `sin`      | Sine                   |
| `cos`      | Cosine                 |
| `tan`      | Tangent                |
| `asin`     | Inverse sine           |
| `acos`     | Inverse cosine         |
| `atan`     | Inverse tangent        |
| `atan2`    | Two-argument arctangent|
| `sinh`     | Hyperbolic sine        |
| `cosh`     | Hyperbolic cosine      |
| `tanh_scalar` | Hyperbolic tangent  |

### 8.4 Math -- Logarithmic and Exponential

| Function   | Description                    |
|------------|--------------------------------|
| `log`      | Natural logarithm              |
| `log2`     | Base-2 logarithm               |
| `log10`    | Base-10 logarithm              |
| `log1p`    | ln(1 + x), accurate for small x |
| `exp`      | Exponential (e^x)              |
| `expm1`    | e^x - 1, accurate for small x |

### 8.5 Math -- Constants

| Name      | Value                    |
|-----------|--------------------------|
| `PI`      | 3.14159265358979...      |
| `E`       | 2.71828182845904...      |
| `TAU`     | 6.28318530717958... (2 * PI) |
| `INF`     | Positive infinity        |
| `NAN_VAL` | Not-a-Number             |

### 8.6 Type Checking and Conversion

| Function   | Description                  |
|------------|------------------------------|
| `int`      | Convert to integer           |
| `float`    | Convert to float             |
| `isnan`    | Check if value is NaN        |
| `isinf`    | Check if value is infinite   |

### 8.7 Assertions

| Function    | Signature                        | Description                      |
|-------------|----------------------------------|----------------------------------|
| `assert`    | `(bool) -> ()`                   | Panic if condition is false      |
| `assert_eq` | `(Any, Any) -> ()`              | Panic if values are not equal    |

### 8.8 Array Operations

| Function         | Signature                     | Description                            |
|------------------|-------------------------------|----------------------------------------|
| `len`            | `([T]) -> i64`                | Array length                           |
| `array_push`     | `([T], T) -> [T]`            | Return new array with element appended |
| `array_pop`      | `([T]) -> [T]`               | Return new array with last element removed |
| `array_contains` | `([T], T) -> bool`           | Check if array contains element        |
| `array_reverse`  | `([T]) -> [T]`               | Return reversed array                  |
| `array_flatten`  | `([[T]]) -> [T]`             | Flatten nested array                   |
| `array_len`      | `([T]) -> i64`               | Array length (alias for `len`)         |
| `array_slice`    | `([T], i64, i64) -> [T]`    | Slice from start to end                |
| `push`           | `([T], T) -> [T]`            | Alias for `array_push`                 |
| `sort`           | `([T]) -> [T]`               | Return sorted array                    |

### 8.9 Tensor Constructors

| Function             | Description                          |
|----------------------|--------------------------------------|
| `Tensor.zeros(shape)`   | All-zeros tensor                 |
| `Tensor.ones(shape)`    | All-ones tensor                  |
| `Tensor.eye(n)`         | n x n identity matrix            |
| `Tensor.randn(shape)`   | Random normal distribution       |
| `Tensor.uniform(shape)` | Random uniform [0, 1)            |
| `Tensor.full(shape, v)` | Filled with value v              |
| `Tensor.linspace(start, end, n)` | n evenly spaced values |
| `Tensor.arange(start, end, step)` | Range with step        |
| `Tensor.from_vec(data, shape)`    | From flat data + shape  |
| `Tensor.diag(values)`   | Diagonal matrix from values      |
| `Tensor.from_bytes(b)`  | From byte data                   |

### 8.10 Tensor Methods

| Method              | Description                       |
|---------------------|-----------------------------------|
| `t.sum()`           | Kahan-compensated sum             |
| `t.mean()`          | Mean                              |
| `t.reshape(shape)`  | Zero-copy reshape                 |
| `t.transpose()`     | Matrix transpose                  |
| `t.add(other)`      | Element-wise addition             |
| `t.mul_elem(other)` | Element-wise multiplication       |
| `t.softmax()`       | Softmax activation                |
| `t.relu()`          | ReLU activation                   |
| `t.shape()`         | Shape as array                    |
| `t.len()`           | Total number of elements          |
| `t.get(indices)`    | Element access                    |
| `t.set(indices, v)` | Element mutation                  |
| `matmul(a, b)`      | Matrix multiplication             |

### 8.11 Tensor Activations

| Function           | Description                      |
|--------------------|----------------------------------|
| `sigmoid`          | Sigmoid activation               |
| `tanh_activation`  | Tanh activation (tensor)         |
| `leaky_relu`       | Leaky ReLU                       |
| `silu`             | SiLU / Swish activation          |
| `mish`             | Mish activation                  |

### 8.12 Tensor Utilities

| Function       | Description                        |
|----------------|------------------------------------|
| `argmax`       | Index of maximum element           |
| `argmin`       | Index of minimum element           |
| `one_hot`      | One-hot encoding                   |
| `cat`          | Concatenate tensors                |
| `stack`        | Stack tensors along new axis       |
| `topk`         | Top-k elements                     |
| `argsort`      | Indices that sort the tensor       |
| `gather`       | Gather elements by index           |
| `scatter`      | Scatter elements by index          |
| `index_select` | Select elements by index           |

### 8.13 Linear Algebra

| Function             | Description                          |
|----------------------|--------------------------------------|
| `dot(a, b)`          | Dot product                          |
| `outer(a, b)`        | Outer product                        |
| `cross(a, b)`        | Cross product                        |
| `norm(t)`            | Vector norm                          |
| `det(m)`             | Determinant                          |
| `trace(m)`           | Matrix trace                         |
| `solve(a, b)`        | Solve linear system Ax = b          |
| `lstsq(a, b)`        | Least-squares solution               |
| `norm_frobenius(m)`  | Frobenius norm                       |
| `eigh(m)`            | Eigendecomposition (symmetric)       |
| `matrix_rank(m)`     | Matrix rank                          |
| `kron(a, b)`         | Kronecker product                    |
| `cond(m)`            | Condition number                     |
| `norm_1(m)`          | 1-norm                               |
| `norm_inf(m)`        | Infinity norm                        |
| `schur(m)`           | Schur decomposition                  |
| `matrix_exp(m)`      | Matrix exponential                   |
| `linalg.lu(m)`       | LU decomposition                     |
| `linalg.qr(m)`       | QR decomposition                     |
| `linalg.cholesky(m)` | Cholesky decomposition               |
| `linalg.inv(m)`      | Matrix inverse                       |

### 8.14 Complex Numbers

```cjc
let z = Complex(3.0, 4.0);         // 3 + 4i
z.re()                              // real part: 3.0
z.im()                              // imaginary part: 4.0
z.abs()                             // magnitude: 5.0
z.conj()                            // conjugate: 3 - 4i
z.norm_sq()                         // |z|^2: 25.0

// Arithmetic: +, -, *, / all supported
let w = z + Complex(1.0, -2.0);
let product = z * w;
```

### 8.15 Map and Set

```cjc
// Map (ordered)
let m = Map.new();
Map.insert(m, "key", 42);
let val = Map.get(m, "key");
Map.remove(m, "key");
Map.len(m);
Map.contains_key(m, "key");

// Set (ordered)
let s = Set.new();
```

### 8.16 Statistics -- Descriptive

| Function       | Description                          |
|----------------|--------------------------------------|
| `variance`     | Population variance                  |
| `sd`           | Population standard deviation        |
| `se`           | Standard error                       |
| `median`       | Median                               |
| `quantile`     | Quantile (0.0 to 1.0)               |
| `iqr`          | Interquartile range                  |
| `skewness`     | Skewness                            |
| `kurtosis`     | Kurtosis                            |
| `z_score`      | Z-score normalization                |
| `standardize`  | Standardize data                     |
| `n_distinct`   | Count of distinct values             |
| `sample_variance` | Sample variance (Bessel-corrected) |
| `sample_sd`    | Sample standard deviation            |
| `sample_cov`   | Sample covariance                    |
| `mode`         | Most frequent value                  |
| `mad`          | Median absolute deviation            |
| `weighted_mean`| Weighted mean                        |
| `weighted_var` | Weighted variance                    |
| `trimmed_mean` | Trimmed mean                         |
| `winsorize`    | Winsorized values                    |
| `percentile_rank` | Percentile rank                   |

### 8.17 Statistics -- Correlation and Inference

| Function             | Description                                  |
|----------------------|----------------------------------------------|
| `cor`                | Pearson correlation                          |
| `cov`               | Covariance                                   |
| `spearman_cor`       | Spearman rank correlation                    |
| `kendall_cor`        | Kendall rank correlation                     |
| `partial_cor`        | Partial correlation                          |
| `cor_ci`             | Correlation confidence interval              |
| `t_test`             | One-sample t-test                            |
| `t_test_two_sample`  | Two-sample t-test                            |
| `t_test_paired`      | Paired t-test                                |
| `chi_squared_test`   | Chi-squared test                             |
| `anova_oneway`       | One-way ANOVA                                |
| `f_test`             | F-test                                       |
| `mann_whitney`       | Mann-Whitney U test                          |
| `kruskal_wallis`     | Kruskal-Wallis test                          |
| `wilcoxon_signed_rank` | Wilcoxon signed-rank test                  |
| `tukey_hsd`          | Tukey HSD post-hoc test                      |
| `bonferroni`         | Bonferroni correction                        |
| `fdr_bh`             | Benjamini-Hochberg FDR correction            |

### 8.18 Distributions

| Function        | Description                          |
|-----------------|--------------------------------------|
| `normal_cdf`    | Normal CDF                           |
| `normal_pdf`    | Normal PDF                           |
| `normal_ppf`    | Normal percent point function        |
| `t_cdf`         | Student's t CDF                      |
| `t_ppf`         | Student's t PPF                      |
| `chi2_cdf`      | Chi-squared CDF                      |
| `chi2_ppf`      | Chi-squared PPF                      |
| `f_cdf`         | F-distribution CDF                   |
| `f_ppf`         | F-distribution PPF                   |
| `binomial_pmf`  | Binomial PMF                         |
| `binomial_cdf`  | Binomial CDF                         |
| `poisson_pmf`   | Poisson PMF                          |
| `poisson_cdf`   | Poisson CDF                          |
| `beta_pdf`      | Beta PDF                             |
| `beta_cdf`      | Beta CDF                             |
| `gamma_pdf`     | Gamma PDF                            |
| `gamma_cdf`     | Gamma CDF                            |
| `exp_pdf`       | Exponential PDF                      |
| `exp_cdf`       | Exponential CDF                      |
| `weibull_pdf`   | Weibull PDF                          |
| `weibull_cdf`   | Weibull CDF                          |

### 8.19 Regression

| Function              | Description                           |
|-----------------------|---------------------------------------|
| `lm`                  | Linear regression (OLS)               |
| `wls`                 | Weighted least squares                |
| `logistic_regression` | Logistic regression                   |

### 8.20 ML -- Loss Functions

| Function               | Description                          |
|------------------------|--------------------------------------|
| `mse_loss`             | Mean squared error loss              |
| `cross_entropy_loss`   | Cross-entropy loss                   |
| `huber_loss`           | Huber loss                           |
| `binary_cross_entropy` | Binary cross-entropy loss            |
| `hinge_loss`           | Hinge loss (SVM)                     |

### 8.21 ML -- Metrics and Utilities

| Function           | Description                         |
|--------------------|-------------------------------------|
| `confusion_matrix` | Confusion matrix                    |
| `auc_roc`          | Area under ROC curve                |
| `batch_norm`       | Batch normalization                 |
| `dropout_mask`     | Generate dropout mask               |
| `lr_step_decay`    | Step decay learning rate schedule   |
| `lr_cosine`        | Cosine annealing learning rate      |
| `lr_linear_warmup` | Linear warmup learning rate         |
| `l1_penalty`       | L1 regularization penalty           |
| `l2_penalty`       | L2 regularization penalty           |

### 8.22 ML -- Autodiff

| Function           | Description                         |
|--------------------|-------------------------------------|
| `GradGraph.new`    | Create new gradient computation graph |
| `Adam.new`         | Create Adam optimizer               |
| `Sgd.new`          | Create SGD optimizer                |
| `stop_gradient`    | Detach from computation graph       |
| `grad_checkpoint`  | Gradient checkpointing              |
| `clip_grad`        | Gradient clipping                   |
| `grad_scale`       | Gradient scaling                    |
| `categorical_sample` | Sample from categorical distribution |

### 8.23 Window and Cumulative Functions

| Function      | Description                      |
|---------------|----------------------------------|
| `cumsum`      | Cumulative sum                   |
| `cumprod`     | Cumulative product               |
| `cummax`      | Cumulative maximum               |
| `cummin`      | Cumulative minimum               |
| `lag`         | Lag (previous value)             |
| `lead`        | Lead (next value)                |
| `rank`        | Rank values                      |
| `dense_rank`  | Dense rank (no gaps)             |
| `row_number`  | Row number                       |
| `ntile`       | N-tile bucketing                 |
| `percent_rank`| Percent rank                     |
| `cume_dist`   | Cumulative distribution          |
| `window_sum`  | Windowed sum                     |
| `window_mean` | Windowed mean                    |
| `window_min`  | Windowed minimum                 |
| `window_max`  | Windowed maximum                 |
| `histogram`   | Histogram                        |

### 8.24 FFT and Signal Processing

| Function        | Description                        |
|-----------------|------------------------------------|
| `rfft`          | Real FFT                           |
| `psd`           | Power spectral density             |
| `fft_arbitrary` | Arbitrary-length FFT               |
| `fft_2d`        | 2-D FFT                            |
| `ifft_2d`       | Inverse 2-D FFT                    |
| `hann`          | Hann window function               |
| `hamming`       | Hamming window function            |
| `blackman`      | Blackman window function           |

### 8.25 String Functions

| Function          | Description                          |
|-------------------|--------------------------------------|
| `str_detect`      | Check if string matches pattern      |
| `str_extract`     | Extract first match                  |
| `str_extract_all` | Extract all matches                  |
| `str_replace`     | Replace first match                  |
| `str_replace_all` | Replace all matches                  |
| `str_split`       | Split string by delimiter            |
| `str_count`       | Count pattern occurrences            |
| `str_trim`        | Trim whitespace                      |
| `str_to_upper`    | Convert to uppercase                 |
| `str_to_lower`    | Convert to lowercase                 |
| `str_starts`      | Check prefix                         |
| `str_ends`        | Check suffix                         |
| `str_sub`         | Substring extraction                 |
| `str_len`         | String length                        |

### 8.26 Data -- Tidy/DataFrame DSL

| Function       | Description                           |
|----------------|---------------------------------------|
| `col`          | Column reference                      |
| `desc`         | Descending sort specification         |
| `asc`          | Ascending sort specification          |
| `case_when`    | Conditional case expression           |
| `tidy_count`   | Count rows                            |
| `tidy_sum`     | Sum aggregation                       |
| `tidy_mean`    | Mean aggregation                      |
| `tidy_min`     | Min aggregation                       |
| `tidy_max`     | Max aggregation                       |
| `tidy_first`   | First value                           |
| `tidy_last`    | Last value                            |

### 8.27 Data -- CSV and JSON

| Function           | Description                       |
|--------------------|-----------------------------------|
| `Csv.parse`        | Parse CSV string                  |
| `Csv.parse_tsv`    | Parse TSV string                  |
| `Csv.stream_sum`   | Streaming CSV sum                 |
| `Csv.stream_minmax`| Streaming CSV min/max             |
| `json_parse`       | Parse JSON string                 |
| `json_stringify`    | Serialize to JSON string          |

### 8.28 Data -- File I/O

| Function      | Description                       |
|---------------|-----------------------------------|
| `file_read`   | Read file contents as string      |
| `file_write`  | Write string to file              |
| `file_exists` | Check if file exists              |
| `file_lines`  | Read file as array of lines       |

### 8.29 Data -- DateTime

| Function              | Description                       |
|-----------------------|-----------------------------------|
| `datetime_now`        | Current timestamp                 |
| `datetime_from_epoch` | DateTime from epoch millis        |
| `datetime_from_parts` | DateTime from y/m/d/h/m/s        |
| `datetime_year`       | Extract year                      |
| `datetime_month`      | Extract month                     |
| `datetime_day`        | Extract day                       |
| `datetime_hour`       | Extract hour                      |
| `datetime_minute`     | Extract minute                    |
| `datetime_second`     | Extract second                    |
| `datetime_diff`       | Difference between two DateTimes  |
| `datetime_add_millis` | Add milliseconds                  |
| `datetime_format`     | Format as string                  |

### 8.30 Numeric Precision Conversion

| Function       | Description                   |
|----------------|-------------------------------|
| `f16_to_f64`   | Half to double precision      |
| `f64_to_f16`   | Double to half precision      |
| `f16_to_f32`   | Half to single precision      |
| `f32_to_f16`   | Single to half precision      |
| `bf16_to_f32`  | Brain float to single         |
| `f32_to_bf16`  | Single to brain float         |

### 8.31 Bitwise Operations (functional)

| Function    | Description                  |
|-------------|------------------------------|
| `bit_and`   | Bitwise AND                  |
| `bit_or`    | Bitwise OR                   |
| `bit_xor`   | Bitwise XOR                  |
| `bit_not`   | Bitwise NOT                  |
| `bit_shl`   | Bitwise shift left           |
| `bit_shr`   | Bitwise shift right          |
| `popcount`  | Population count (set bits)  |

### 8.32 Miscellaneous

| Function      | Description                       |
|---------------|-----------------------------------|
| `clock`       | Current wall-clock time (seconds) |
| `gc_alloc`    | Allocate GC object                |
| `gc_collect`  | Trigger garbage collection        |
| `gc_live_count` | Count of live GC objects        |
| `snap`        | Snapshot a value                  |
| `restore`     | Restore from snapshot             |
| `snap_hash`   | Hash a snapshot                   |
| `snap_save`   | Save snapshot to storage          |
| `snap_load`   | Load snapshot from storage        |
| `snap_to_json`| Convert snapshot to JSON          |
| `memo_call`   | Memoized function call            |

---

## 9. Prelude

The following types and variants are available without any `import` statement.

### Option

```cjc
Some(value)                 // wraps a value
None                        // absence of a value

// Methods
opt.unwrap()                // extract value or panic
opt.unwrap_or(default)      // extract value or return default
opt.is_some()               // returns true if Some
opt.is_none()               // returns true if None
```

### Result

```cjc
Ok(value)                   // success
Err(error)                  // failure

// Try operator for early return
let x = operation()?;       // returns Err early on failure
```

---

## 10. Syntax Rules and Conventions

1. **Semicolons** end statements. Block statements (`if`, `while`, `for`) do **not** require trailing semicolons inside function bodies.

2. **Tail expressions** -- the last expression in a block without a trailing `;` becomes the block's return value:
   ```cjc
   fn double(x: i64) -> i64 {
       x * 2       // no semicolon = return value
   }
   ```

3. **Type annotations** are required on function parameters. Let bindings can omit types when the literal type is unambiguous:
   ```cjc
   fn add(a: i64, b: i64) -> i64 { a + b }     // required
   let x = 42;                                    // inferred as i64
   ```

4. **Immutable by default** -- use `let mut` to declare mutable variables:
   ```cjc
   let x = 10;           // immutable
   let mut y = 20;       // mutable
   y = 30;               // OK
   // x = 15;            // ERROR: x is immutable
   ```

5. **Right-associative operators**: `**` (power) and all assignment operators (`=`, `+=`, etc.).

6. **Short-circuit evaluation**: `&&` and `||` do not evaluate the right operand if the left determines the result.

7. **Bitwise operators** work only on integer types. Using them on floats produces a runtime error.

8. **The pipe operator** `|>` inserts the left-hand value as the first argument of the right-hand function call.

9. **Format strings** use `{}` for interpolation. Use `{{` and `}}` for literal braces.

10. **Tensor literals** use `[| ... |]` delimiters. Semicolons separate rows; commas separate elements.

11. **Arrays are values** -- `array_push` and similar operations return a new array rather than mutating in place:
    ```cjc
    let mut arr: [i64] = [];
    arr = array_push(arr, 1);      // must reassign
    ```

12. **The `Any` type** bypasses static type checking for dynamic/polymorphic code.

---

## 11. Grammar Summary

### Program Structure

```
Program     = Declaration*
Declaration = FnDecl | StructDecl | RecordDecl | ClassDecl | EnumDecl
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
