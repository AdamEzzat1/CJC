# CJC V 0.1 — Deterministic Scientific Computing Language

CJC is a statically-informed, deterministic programming language for scientific
computing. Built entirely in Rust with zero external dependencies, it ships
300+ built-in functions spanning tensors, linear algebra, statistics, machine
learning, data wrangling, and automatic differentiation — all with reproducible,
seed-controlled execution.

## Installation

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (1.70+)

### Install

```bash
git clone <repo-url> CJC
cd CJC
cargo install --path crates/cjc-cli
```

### Verify

```bash
cjc --version          # cjc 0.1.0
cjc repl               # starts interactive REPL
```

## Your First Program

Create a file `hello.cjc`:

```
print("Hello, CJC!");
print(2 + 2);
```

Run it:

```bash
cjc run hello.cjc
```

Output:

```
Hello, CJC!
4
```

## Quick Tour

### Variables and Types

```
let x: i64 = 42;
let mut y: f64 = 3.14;
y = y * 2.0;
print(y);                  // 6.28
```

### Functions

```
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}
print(fibonacci(10));      // 55
```

### Tensors and Linear Algebra

```
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let B = Tensor.eye(2);
let C = matmul(A, B);
print(C);
```

### Statistics

```
let data = [23.1, 25.4, 22.8, 24.5, 26.1, 23.9];
print(mean(data));         // 24.3
print(sd(data));           // standard deviation
print(t_test(data, 24.0)); // one-sample t-test
```

### Records and Pattern Matching

```
record Point { x: f64, y: f64 }

fn distance(p: Any) -> f64 {
    sqrt(p.x * p.x + p.y * p.y)
}

let p = Point { x: 3.0, y: 4.0 };
print(distance(p));        // 5
```

### Interactive REPL

```
$ cjc repl
CJC REPL v0.1.0  (type :help for commands, :quit to exit)
cjc> let x = 42;
cjc> print(x * 2);
84
cjc> :quit
```

## What's In This Folder

| File | Description |
|------|-------------|
| **README.md** | This file — getting started guide |
| **SYNTAX_REFERENCE.md** | Complete language syntax (keywords, operators, types, grammar) |
| **BUILTIN_REFERENCE.md** | All 300+ built-in functions organized by domain |
| **TENSOR_AND_ML_GUIDE.md** | Tensors, automatic differentiation, neural networks |
| **DATA_SCIENCE_GUIDE.md** | Tidy DSL, CSV, strings, regex, statistics |
| **EFFECTS_AND_MEMORY.md** | Effect system, memory model, NoGC, determinism |
| **SNAP_SERIALIZATION.md** | Content-addressable serialization system |
| **CLI_AND_REPL_GUIDE.md** | CLI commands, flags, REPL meta-commands |
| **ARCHITECTURE.md** | Compiler pipeline, crate map, design decisions |
| **KNOWN_LIMITATIONS.md** | Honest assessment of gaps and roadmap |
| **examples/** | 12 runnable example programs |

## Capabilities at a Glance

- **26 keywords** — struct, record, class, enum, fn, trait, impl, match, ...
- **30+ types** — i64, f64, bool, str, Tensor, Complex, bf16, Map, Set, ...
- **30+ operators** — arithmetic, comparison, logical, bitwise, regex, pipe
- **300+ builtins** across:
  - Mathematics (40+): sqrt, log, exp, sin, cos, abs, floor, ...
  - Statistics (80+): mean, sd, t_test, anova, normal_cdf, ...
  - Linear algebra (20+): matmul, solve, det, eigenvalues, ...
  - Machine learning (50+): cross_entropy, Adam, attention, conv2d, ...
  - Tensors (50+): construction, shape ops, broadcasting, activations
  - Data wrangling (73+): tidy verbs, group_by, joins, pivots
  - String ops (14): str_detect, str_replace, str_split, ...
  - I/O, CSV, JSON, DateTime, Snap serialization
- **Two execution engines** — AST interpreter + MIR executor with optimizer
- **Automatic differentiation** — forward-mode (Dual) + reverse-mode (GradGraph)
- **Deterministic by design** — seeded RNG, Kahan summation, ordered collections
- **Interactive REPL** — history, multi-line input, 8 meta-commands
- **Color diagnostics** — structured error codes (E0xxx-E8xxx)

## Running Examples

```bash
cjc run "CJC V 0.1/examples/01_hello_world.cjc"
cjc run "CJC V 0.1/examples/05_tensors_and_linalg.cjc"
cjc run "CJC V 0.1/examples/08_statistics.cjc"
```

## License

MIT
