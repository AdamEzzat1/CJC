# Java vs CJC-Style Classes

CJC takes inspiration from Java's class system but diverges in key areas to
support deterministic scientific computing, zero-GC execution paths, and
effect-safe composition.

## 1. Overview

| Concept | Java | CJC |
|---------|------|-----|
| Data carrier | `record` (Java 16+) | `record` (immutable value type) |
| Mutable aggregate | `class` | `struct` (value) or `class` (GC ref) |
| Polymorphism | Interfaces + inheritance | Traits only (no inheritance) |
| Default semantics | Reference (heap) | Value (stack-copied) |
| Effect tracking | Checked exceptions | Effect annotations (`/ pure`, `/ io`, ...) |
| Garbage collection | Always-on | Opt-in (`class` uses GC; `struct`/`record` do not) |

## 2. Records

Both Java and CJC have `record` as a concise data carrier.

**Java:**
```java
record Point(double x, double y) {}
```

**CJC:**
```
record Point {
    x: f64,
    y: f64
}
```

Key differences:
- CJC records are **value types** (stack-allocated, copied on assignment).
  Java records are reference types on the heap.
- CJC enforces immutability at **both compile time (E0160) and runtime**.
  Java records have final fields but the object itself is a mutable reference.
- CJC records support **structural equality** by default.  Java records
  auto-generate `equals()` / `hashCode()` similarly.
- CJC records can be **pattern-matched** with `match` and struct destructuring.

## 3. Traits vs Interfaces and Inheritance

CJC replaces Java's interface + class inheritance model with **traits only**.

**Java:**
```java
interface Drawable { void draw(); }
class Circle extends Shape implements Drawable { ... }
```

**CJC:**
```
trait Drawable {
    fn draw(self: Any) -> i64;
}
struct Circle { radius: f64 }
impl Drawable for Circle {
    fn draw(self: Circle) -> i64 { 1 }
}
```

Key differences:
- CJC has **no `extends` keyword**. The parser rejects it.
- CJC has **no `super` keyword**. There is no inheritance chain.
- All polymorphism comes through **trait bounds** on generics:
  `fn render<T: Drawable>(shape: T)`.
- CJC supports two `impl` syntaxes:
  - `impl Trait for Type { ... }` (Rust-style)
  - `impl Type : Trait { ... }` (CJC alternative)
  - `impl Type { ... }` (bare methods, no trait)

### Why No Inheritance

1. **Composition over inheritance** — traits compose without diamond problems.
2. **Determinism** — no virtual dispatch tables; method resolution is static.
3. **NoGC compatibility** — inheritance hierarchies typically require heap
   allocation and GC, which conflicts with CJC's zero-GC execution paths.

## 4. Value vs Reference Semantics

In Java, all objects are reference types.  In CJC, the default is **value**.

| Type | Semantics | Memory | GC Required |
|------|-----------|--------|-------------|
| `struct` | Value (mutable) | Stack / inline | No |
| `record` | Value (immutable) | Stack / inline | No |
| `class` | Reference | GC heap | Yes |

**CJC value semantics:**
```
struct Pair { x: i64, y: i64 }
let mut a = Pair { x: 1, y: 2 };
let mut b = a;       // b is an independent copy
b.x = 99;           // does NOT affect a
print(a.x);         // prints 1
```

In Java, the equivalent would alias — `b` and `a` would refer to the same
object on the heap.  CJC's value-by-default design eliminates an entire class
of aliasing bugs.

## 5. Effect System vs Checked Exceptions

Java uses checked exceptions to track side effects.  CJC uses an **effect
annotation system** that is more fine-grained.

**Java:**
```java
void readFile() throws IOException { ... }
```

**CJC:**
```
fn read_file(path: str) -> str / io {
    // IO operations allowed here
}

fn pure_compute(x: f64) -> f64 / pure {
    // Only pure operations allowed — no IO, no allocation
    sqrt(x)
}
```

CJC effect flags: `pure`, `io`, `alloc`, `gc`, `nondet`, `mutates`,
`arena_ok`, `captures`.

Combined effects use `+`: `fn save(data: Any) / io + alloc { ... }`

### Effect Checking Rules

1. **Annotated functions/methods** are checked: calling a function with
   effects not in the annotation produces **E4002**.
2. **Unannotated functions** are backward-compatible: any effect is allowed.
3. Effects propagate transitively: a `/ pure` function cannot call a `/ io`
   function (whether builtin or user-defined).

## 6. Method Effects

Methods on `impl` blocks support the same effect annotations as top-level
functions.

```
struct Sensor { reading: f64 }
impl Sensor {
    fn calibrate(self: Sensor) -> f64 / pure {
        self.reading * 1.05    // OK — pure arithmetic
    }

    fn log_reading(self: Sensor) / io {
        print(self.reading);   // OK — IO annotation allows print
    }
}
```

The type checker verifies method effects through the same `check_fn()` path
as top-level functions.  If a `/ pure` method calls `print()`, the type
checker emits **E4002** with a message identifying the violating call.

## 7. Migration Guide for Java Developers

| Java Pattern | CJC Equivalent |
|-------------|----------------|
| `class Foo { ... }` | `struct Foo { ... }` (value) or `class Foo { ... }` (ref) |
| `record Foo(int x)` | `record Foo { x: i64 }` |
| `interface Bar { ... }` | `trait Bar { ... }` |
| `class A extends B` | Not supported — use trait composition |
| `implements Foo` | `impl Foo for Type { ... }` |
| `throws IOException` | `/ io` effect annotation |
| `final` fields | `record` (all fields immutable by default) |
| `new Foo()` | `Foo { field: value }` (brace-literal construction) |

## 8. Test Coverage

The Role 9 test suite (`tests/role9_classes/`) contains 52 tests across 4
files:

| File | Tests | Requirement |
|------|-------|-------------|
| `test_r9_records.rs` | 15 | R1: Records as data carrier |
| `test_r9_traits.rs` | 12 | R2: Traits over inheritance |
| `test_r9_value_semantics.rs` | 10 | R3: Value classes default |
| `test_r9_method_effects.rs` | 15 | R4: Effect tags on methods |

All tests verify both eval and MIR-exec parity where applicable.
