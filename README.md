# CJC — Deterministic Scientific Computing Language

**Version 0.1.0** | **Rust** | **Zero External Dependencies** | **MIT License**

CJC is a statically-typed programming language built from scratch in Rust for scientific computing, machine learning, and data analysis. It provides 221+ built-in functions, numerically stable tensor operations, and guaranteed deterministic execution — all with zero external dependencies.

```cjc
fn main() -> i64 {
    let data = [| 1.0, 2.0, 3.0, 4.0, 5.0 |];
    let mean_val: f64 = mean(data);
    let std_val: f64 = sd(data);
    print(f"mean: {mean_val}, std: {std_val}");
    0
}
```

---

## Key Features

- **Deterministic Execution** — Same seed produces identical results on every run. Kahan/binned summation prevents floating-point drift. All collections use ordered maps.
- **221+ Built-in Functions** — Math, statistics, linear algebra, ML, signal processing, data wrangling, and more — all built in, no packages needed.
- **Zero-Copy Tensors** — Copy-on-write buffer system with no garbage collector pauses. Reshape, transpose, and slice without allocation.
- **Two Execution Engines** — Tree-walk interpreter (eval) and optimized MIR executor with tail-call optimization and arena allocation.
- **73+ Data Wrangling Operations** — Tidyverse-compatible: filter, select, mutate, group_by, summarize, joins, pivots, window functions.
- **Interactive REPL** — Explore the language interactively with persistent state across lines.
- **Color Diagnostics** — ANSI-colored error messages with source context, spans, and actionable hints.

---

## Quick Start

### Build

```bash
git clone https://github.com/your-repo/CJC.git
cd CJC
cargo build --release
```

### Run a Program

```bash
# Via cargo
cargo run --bin cjc -- run examples/hello.cjc

# Or use the built binary directly
./target/release/cjc run myfile.cjc
```

### Install Globally

```bash
cargo install --path crates/cjc-cli
cjc run myfile.cjc
cjc repl
```

---

## CLI Usage

```
cjc run <file.cjc>     Run a CJC program
cjc repl                Start the interactive REPL
cjc lex <file.cjc>     Tokenize and print tokens
cjc parse <file.cjc>   Parse and pretty-print AST
cjc check <file.cjc>   Type-check without running
```

### Flags

| Flag | Description |
|------|-------------|
| `--help`, `-h` | Print usage and exit |
| `--version`, `-V` | Print version and exit |
| `--color` | Force color output |
| `--no-color` | Disable color output |
| `--seed N` | Set RNG seed (default: 42) |
| `--time` | Print execution time |
| `--mir-opt` | Enable MIR optimizations |
| `--mir-mono` | Enable monomorphization |
| `--multi-file` | Enable module resolution |

---

## Language Overview

### Variables and Types

```cjc
let x: i64 = 42;
let pi: f64 = 3.14159;
let name: str = "CJC";
let flag: bool = true;
let mut counter: i64 = 0;   // mutable
counter += 1;
```

### Functions

```cjc
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn greet(name: str) {
    print(f"Hello, {name}!");
}
```

### Structs

```cjc
struct Point {
    x: f64,
    y: f64,
}

fn distance(p: Point) -> f64 {
    sqrt(p.x ** 2.0 + p.y ** 2.0)
}

fn main() -> i64 {
    let p = Point { x: 3.0, y: 4.0 };
    print(f"distance: {distance(p)}");
    0
}
```

### Enums and Pattern Matching

```cjc
enum Shape {
    Circle(f64),
    Rect(f64, f64),
}

fn area(s: Shape) -> f64 {
    match s {
        Circle(r) => 3.14159 * r ** 2.0,
        Rect(w, h) => w * h,
        _ => 0.0,
    }
}
```

### Closures

```cjc
fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

fn main() -> i64 {
    let double = |x: i64| x * 2;
    apply(double, 21)   // returns 42
}
```

### Control Flow

```cjc
// If/else
if x > 0 {
    print("positive");
} else if x == 0 {
    print("zero");
} else {
    print("negative");
}

// If as expression
let abs_val = if x >= 0 { x } else { -x };

// While loop
while count < 100 {
    count += 1;
}

// For loop (range)
for i in 0..10 {
    print(i);
}
```

### Operators

```cjc
// Arithmetic
x + y    x - y    x * y    x / y    x % y    x ** y

// Comparison
x == y   x != y   x < y    x > y    x <= y   x >= y

// Logical
x && y   x || y   !x

// Bitwise
x & y    x | y    x ^ y    ~x    x << n    x >> n

// Compound assignment
x += 1   x -= 1   x *= 2   x /= 2   x %= 3   x **= 2

// Pipe
data |> transform() |> output()
```

### Number Literals

```cjc
42              // decimal integer
3.14            // float
0xFF            // hexadecimal (255)
0b1010          // binary (10)
0o777           // octal (511)
1_000_000       // underscore separators
0xFF_FF         // hex with separators
```

### String Variants

```cjc
"hello world"           // standard string
f"value = {x + 1}"     // format string (interpolation)
r"no\escape"            // raw string
b"byte string"          // byte string
```

---

## Tensor Operations

CJC has first-class tensor support with zero-copy semantics and numerically stable operations.

```cjc
// Create tensors
let a = Tensor.zeros([3, 3]);
let b = Tensor.ones([3, 3]);
let c = Tensor.randn([2, 4]);          // random normal
let d = Tensor.eye(3);                  // identity matrix
let e = [| 1.0, 2.0; 3.0, 4.0 |];     // tensor literal (2x2)

// Operations
let sum = a.add(b);
let product = matmul(a, b);
let transposed = c.transpose();
let reshaped = c.reshape([4, 2]);       // zero-copy view

// Reductions (Kahan-stable)
let total: f64 = e.sum();
let avg: f64 = e.mean();

// Deep learning operations
let out = attention(query, key, value);
let conv_out = conv2d(input, kernel, stride, padding);
let normed = layer_norm(x, gamma, beta, epsilon);
```

---

## Built-in Function Categories

### Mathematics (19 functions)
`sqrt`, `log`, `exp`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `pow`, `log2`, `log10`, `ceil`, `floor`, `round`, `abs`, `sign`, `hypot`, `clamp`

**Constants:** `PI`, `E`, `TAU`, `INF`, `NAN_VAL`

### Statistics (35+ functions)
`mean`, `variance`, `sd`, `median`, `quantile`, `cor`, `cov`, `iqr`, `skewness`, `kurtosis`, `z_score`, `standardize`, `trimmed_mean`, `weighted_mean`, `mad`, `spearman_cor`, `kendall_cor`, `partial_cor`

### Distributions (24 functions)
Normal, t, Chi-squared, F, Beta, Exponential, Gamma, Weibull, Poisson, Binomial — each with PDF/CDF/PPF variants.

### Hypothesis Testing (24 functions)
`t_test`, `t_test_two_sample`, `t_test_paired`, `chi_squared_test`, `f_test`, `anova_oneway`, `mann_whitney`, `wilcoxon_signed_rank`, `kruskal_wallis`, `tukey_hsd`, `bonferroni`, `fdr_bh`

### Linear Algebra (9+ functions)
`matmul`, `dot`, `cross`, `outer`, `kron`, `det`, `solve`, `lstsq`, `eigh`, `trace`, `matrix_rank`, `matrix_exp`

### Machine Learning (40+ functions)

**Loss functions:** `mse_loss`, `cross_entropy_loss`, `huber_loss`, `binary_cross_entropy`, `hinge_loss`

**Activations:** `sigmoid`, `relu`, `gelu`, `tanh_activation`, `leaky_relu`, `mish`, `silu`

**Layers:** `layer_norm`, `batch_norm`, `dropout_mask`, `conv1d`, `conv2d`, `maxpool2d`, `attention`

**Optimization:** `Adam.new`, `Sgd.new`, `lr_cosine`, `lr_linear_warmup`, `lr_step_decay`

**Gradient utilities:** `stop_gradient`, `grad_checkpoint`, `clip_grad`, `grad_scale`

### Signal Processing (14+ functions)
FFT (Cooley-Tukey), RFFT, IFFT, 2D-FFT, Bluestein, PSD, window functions (Hanning, Hamming, Blackman)

### Data Wrangling (73+ functions)
`filter`, `select`, `mutate`, `group_by`, `summarize`, `arrange`, `distinct`, `inner_join`, `left_join`, `right_join`, `anti_join`, `semi_join`, `pivot_longer`, `pivot_wider`, `lag`, `lead`, `rank`, `dense_rank`, `ntile`, `window_sum`, `window_mean`, `str_detect`, `str_extract`, `str_replace`, `str_split`, `datetime_from_epoch`, `datetime_diff`

### I/O and Utilities
`print`, `file_read`, `file_write`, `file_exists`, `json_parse`, `json_stringify`, `assert`, `assert_eq`

---

## Data Manipulation (TidyView DSL)

CJC includes a tidyverse-compatible data manipulation layer for tabular data.

```cjc
// Load and transform data
let df = Csv.parse("data.csv");

// Filter, select, mutate
df |> filter(col("price") > 100)
   |> select("name", "price", "category")
   |> mutate("tax", col("price") * 0.08)

// Group and summarize
df |> group_by("category")
   |> summarize("avg_price", mean(col("price")))
   |> arrange(desc("avg_price"))

// Joins
inner_join(orders, customers, "customer_id")
left_join(products, categories, "category_id")

// Pivoting
df |> pivot_longer("Q1", "Q2", "Q3", "Q4", names_to: "quarter", values_to: "sales")
df |> pivot_wider(names_from: "metric", values_from: "value")
```

---

## Module System

CJC supports multi-file projects with a module system.

```cjc
// math/linalg.cjc
fn dot_product(a: Tensor, b: Tensor) -> f64 {
    dot(a, b)
}

// main.cjc
import math.linalg

fn main() -> i64 {
    let a = [| 1.0, 2.0, 3.0 |];
    let b = [| 4.0, 5.0, 6.0 |];
    print(linalg::dot_product(a, b));
    0
}
```

Run with `cjc run main.cjc --multi-file`.

---

## Architecture

CJC is built as 17 Rust crates with a clean compilation pipeline:

```
Source Code
    |
    v
[Lexer] ──> Tokens
    |
    v
[Parser] ──> AST (Abstract Syntax Tree)
    |
    v
[Type Checker] ──> Typed AST
    |
    ├──> [Eval] ──> Direct interpretation (v1)
    |
    └──> [HIR Lowering] ──> HIR (desugaring, capture analysis)
              |
              v
         [MIR Lowering] ──> MIR (lambda-lifting, pattern compilation)
              |
              v
         [Optimizer] ──> Optimized MIR (constant folding, DCE)
              |
              v
         [MIR-Exec] ──> Execution (v2, with TCO + arena allocation)
```

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `cjc-lexer` | Tokenization (59 token kinds) |
| `cjc-parser` | Pratt parser (13-level precedence) |
| `cjc-ast` | Abstract syntax tree definitions |
| `cjc-types` | Type checking and unification |
| `cjc-dispatch` | Multi-method dispatch with coherence |
| `cjc-eval` | Tree-walk interpreter (v1 engine) |
| `cjc-hir` | High-level IR with desugaring |
| `cjc-mir` | Mid-level IR, optimizer, NoGC verifier |
| `cjc-mir-exec` | MIR executor with TCO (v2 engine) |
| `cjc-runtime` | 221+ builtins, tensor system |
| `cjc-data` | TidyView data manipulation layer |
| `cjc-ad` | Automatic differentiation |
| `cjc-repro` | Deterministic RNG and accumulation |
| `cjc-diag` | Diagnostic rendering with ANSI color |
| `cjc-regex` | Regular expression support |
| `cjc-module` | Multi-file module resolution |
| `cjc-cli` | Command-line interface and REPL |

---

## Memory Model

CJC uses a **deterministic, GC-free memory model**:

- **Copy-on-Write Buffers** — Tensors share backing storage via reference counting. Mutation triggers a deep copy only when shared.
- **No Garbage Collector** — RC-based memory with explicit lifecycle. No GC pauses or non-determinism.
- **Arena Allocation** — Per-stack-frame arenas for fast allocation and bulk deallocation.
- **NoGC Verification** — Static analysis proves that `nogc` functions never trigger allocation, verified through transitive call graph analysis.

---

## Numerical Guarantees

- **Kahan Summation** — All floating-point reductions use compensated summation to prevent catastrophic cancellation.
- **Seeded RNG** — SplitMix64 random number generator produces identical sequences from the same seed.
- **Deterministic Ordering** — All collections use `BTreeMap`/`BTreeSet` (no hash randomization).
- **IEEE 754 Compliance** — `total_cmp()` ensures deterministic NaN ordering across platforms.

---

## Test Suite

CJC has a comprehensive test suite with 3,118 passing tests:

```bash
cargo test                    # Run all tests
cargo test --test test_eval   # Run interpreter tests
cargo test --test test_mir_exec  # Run MIR executor tests
```

| Suite | Tests | Description |
|-------|-------|-------------|
| Language Hardening | 83 | Syntax extensions, operators, ML builtins |
| Mathematics | 120 | Stats, linalg, distributions |
| Chess RL Benchmark | 49 | End-to-end reinforcement learning |
| Data Science | 100+ | Tidy operations, CSV, joins |
| Parity Gates | 50+ | Eval vs MIR-exec equivalence |
| Full Workspace | **3,118** | All tests passing, 0 failures |

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/CJC_GRADE_REPORT.md`](docs/CJC_GRADE_REPORT.md) | Comprehensive grading and analysis |
| [`docs/CJC_Syntax_V0.1.md`](docs/CJC_Syntax_V0.1.md) | Complete syntax reference |
| [`docs/CJC_Feature_Capabilities.md`](docs/CJC_Feature_Capabilities.md) | Feature matrix and status |
| [`docs/SYNTAX.md`](docs/SYNTAX.md) | Detailed grammar specification |
| [`docs/CJC_STAGE2_SPEC.md`](docs/CJC_STAGE2_SPEC.md) | Stage 2 language specification |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records (12 ADRs) |

---

## License

MIT
