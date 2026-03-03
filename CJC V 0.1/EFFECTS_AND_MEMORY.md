# Effects, Memory Model, and Determinism

## Effect System

CJC tracks computational effects through annotations on functions and methods.
This enables static verification that pure computations don't perform I/O,
and that NoGC regions don't allocate.

### Effect Annotations

```
fn pure_add(a: i64, b: i64) -> i64 / pure {
    a + b
}

fn greet(name: str) / io {
    print(name);
}

fn process(data: f64) -> f64 / io + alloc {
    print(data);
    outer(data, data)
}
```

### Effect Flags

| Flag | Meaning |
|------|---------|
| `pure` | No side effects at all |
| `io` | Performs I/O (print, file ops) |
| `alloc` | Allocates memory (outer product, array ops) |
| `gc` | Triggers garbage collection |
| `nondet` | Non-deterministic (RNG, datetime_now) |
| `mutates` | Mutates shared state |
| `arena_ok` | Safe for arena allocation |
| `captures` | Captures environment (closures) |

### Effect Checking Rules

1. **Annotated functions** are checked: calling a function whose effects exceed
   the annotation produces **E4002**.
2. **Unannotated functions** are backward-compatible: any effect is allowed.
3. Effects propagate transitively: a `/ pure` function cannot call a `/ io`
   function, whether built-in or user-defined.
4. Combined effects use `+`: `/ io + alloc` permits both I/O and allocation.

### Method Effects

Methods on `impl` blocks support the same effect annotations:

```
struct Sensor { reading: f64 }
impl Sensor {
    fn calibrate(self: Sensor) -> f64 / pure {
        self.reading * 1.05    // OK: pure arithmetic
    }
    fn log_reading(self: Sensor) / io {
        print(self.reading);   // OK: IO annotation allows print
    }
}
```

### Effect Registry

Every built-in function has a registered effect:
- `sqrt`, `abs`, `dot`, `norm` — pure
- `print`, `file_write` — io
- `outer`, `array_push` — alloc
- `Tensor.randn`, `categorical_sample` — nondet
- `datetime_now`, `clock` — nondet + io

## Three-Layer Memory Model

CJC uses a three-layer memory architecture:

### Layer 1: NoGC (Stack)

Value types (`struct`, `record`, primitives) live on the stack. No heap
allocation, no garbage collection. Verified with `nogc fn` and `nogc { }`.

```
nogc fn dot_product(a: Any, b: Any, n: i64) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..n {
        sum = sum + a[i] * b[i];
    }
    sum
}
```

The NoGC verifier performs transitive call-graph analysis: if `dot_product`
calls another function, that function must also be NoGC-safe.

### Layer 2: COW Buffers (Rc)

Tensors and arrays use copy-on-write (COW) storage backed by `Rc<RefCell<Vec>>`.
Multiple tensors can share the same underlying buffer. Deep copies happen only
when a shared buffer is mutated:

```
let a = Tensor.zeros([1000, 1000]);  // allocates once
let b = a;                           // shares buffer (Rc clone)
// b is read-only → no copy needed
```

### Layer 3: GC Heap (Mark-Sweep)

Only `class` instances use the garbage-collected heap. The GC is a simple
mark-sweep collector that runs when the heap grows beyond a threshold.

```
class Node {
    value: i64,
    next: Any
}
// Node instances are GC-managed
```

`struct` and `record` types never touch the GC.

## Value vs Reference Semantics

| Type | Semantics | Memory | GC Required |
|------|-----------|--------|-------------|
| `struct` | Value (mutable) | Stack / inline | No |
| `record` | Value (immutable) | Stack / inline | No |
| `class` | Reference | GC heap | Yes |

```
struct Pair { x: i64, y: i64 }
let mut a = Pair { x: 1, y: 2 };
let mut b = a;     // b is an independent COPY
b.x = 99;          // does NOT affect a
print(a.x);        // 1
```

## Determinism Guarantees

CJC is deterministic by design:

### Seeded RNG

All random operations use a SplitMix64 PRNG seeded by `--seed N` (default: 42).
Same seed = same results, always.

### Stable Summation

Three summation strategies prevent floating-point non-determinism:
- **Kahan summation** — compensated accumulation for `sum()`
- **Exponent-binned accumulation** — groups by exponent for extreme ranges
- **Pairwise summation** — recursive halving for balanced accuracy

### Ordered Collections

Maps and Sets use `BTreeMap`/`BTreeSet` (not hash maps), ensuring iteration
order is always deterministic. Float comparisons use IEEE 754 `total_cmp` for
consistent NaN ordering.

### Reproducibility Flag

```bash
cjc run simulation.cjc --seed 42 --reproducible
```

The `--reproducible` flag enables additional checks to ensure cross-platform
reproducibility.
