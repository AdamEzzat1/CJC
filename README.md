# CJC-Lang — A Deterministic Numerical Programming Language

*Computational Jacobian Core*

**Version 0.1.4** | **Rust** | **Zero External Dependencies** | **MIT License**

> **Note:** Starting with v0.1.4, the project is now **CJC-Lang** and the CLI command has changed from `cjc` to `cjcl`. Install with `cargo install cjc-lang`.

CJC-Lang is a programming language being built for reproducible numerical computation, machine learning, and data analysis. It is implemented entirely in Rust across 21 workspace crates (~96K lines), with zero external runtime dependencies. The language is under active development — it works, but it is not production-ready.

```cjcl
fn main() -> i64 {
    let data = [| 1.0, 2.0, 3.0, 4.0, 5.0 |];
    let mean_val: f64 = mean(data);
    let std_val: f64 = sd(data);
    print(f"mean: {mean_val}, std: {std_val}");
    0
}
```

---

## What CJC-Lang Is Trying to Do

CJC-Lang exists to answer a specific question: *can a single language handle numerical computing, machine learning, and data analysis with guaranteed reproducibility — without depending on any external libraries?*

The design priorities are:

1. **Deterministic execution** — Same seed produces bit-identical output across runs. No hash map iteration order surprises, no non-deterministic floating-point reductions.
2. **Zero external dependencies** — The entire toolchain, from lexer to executor, is self-contained. No LLVM, no libc math, no third-party crates.
3. **Numerical correctness** — Kahan and binned accumulators for floating-point summation. SplitMix64 for reproducible RNG. No fused multiply-add in SIMD kernels.
4. **Dual execution backends** — An AST tree-walk interpreter (v1) and a MIR register-machine executor (v2). Both must produce identical results for every program.

These are goals being worked toward, not finished claims.

---

## What CJC-Lang Has Proven So Far

The test suite currently has **6,715+ tests** across the workspace — 1,913 unit tests in `crates/*/src/` plus 4,802 integration tests in `tests/`, with 22 `proptest!` property-test macros and 38 `bolero` fuzz targets on top. **5,353 pass in the default `cargo test --workspace --release` build**; the remainder are feature-gated (parallel, quantum backends, etc.). Coverage includes:

### Language Fundamentals (Passing)
- Lexer, parser, and type system correctness
- Variable binding, control flow (`if`/`while`/`for`), closures with capture analysis
- Pattern matching with structural destructuring (tuples and structs)
- First-class functions and higher-order functions
- String, integer, float, boolean, array, and tensor types

### Compiler Pipeline (Passing)
- AST → HIR lowering with capture analysis
- HIR → MIR lowering with register allocation
- MIR optimizer (constant folding, dead code elimination)
- NoGC static verifier (proves absence of GC allocations in marked paths)
- **Parity tests**: AST interpreter and MIR executor produce identical output for all programs

### Numerical Infrastructure (Passing)
- Kahan summation and binned accumulators
- SplitMix64 deterministic RNG with explicit seed threading
- Automatic differentiation (forward-mode dual numbers, reverse-mode tape)
- Tensor operations (create, reshape, element access, arithmetic)
- Deterministic linear algebra operations
- 440+ built-in functions (363 in `cjc-runtime` + 83 in `cjc-quantum`) covering math, statistics, ML, signal processing, data wrangling, linear algebra, and quantum simulation

### Data and Visualization (Passing)
- 73+ DataFrame operations (filter, group_by, join, select, pivot, window functions)
- Grammar-of-graphics visualization library (80 chart types, SVG output)
- NFA-based regex engine
- Binary serialization

### What Is Not Yet Working
- Default function parameters
- Decorators
- MIR-level autodiff integration
- Browser compilation target

---

## The Chess RL Demo

The chess reinforcement learning demo is a capability benchmark — it stress-tests CJC-Lang's ability to handle a non-trivial numerical computing workload end-to-end.

### Which Parts Are Written in CJC

The **CJC-Lang backend** (`tests/chess_rl_project/`, `tests/chess_rl_advanced/`) implements:

- **Complete chess engine** — Board representation, legal move generation for all piece types, castling, en passant, pawn promotion, check and checkmate detection, draw rules (stalemate, 50-move rule, insufficient material, threefold repetition)
- **Neural network forward pass** — `forward_move()` computing linear layers with tanh activation through CJC's tensor and arithmetic primitives
- **Training loop** — REINFORCE policy gradient with reward shaping, weight updates, gradient computation
- **Board encoding** — Board state to feature tensor conversion
- **Action selection** — Softmax probability distribution over legal moves with temperature-based exploration

All of the above runs through both CJC-Lang execution backends (AST interpreter and MIR executor) and produces identical results. This is verified by **216 dedicated tests** including:
- Move generation correctness (all piece types, all special moves)
- Training determinism (same seed → identical weights after N episodes)
- Network output validity (no NaN, bounded values)
- Parity between AST and MIR execution
- Fuzz testing across multiple random seeds

### Which Parts Are Written in JavaScript

The **browser frontend** (`examples/chess_rl_platform.html`, ~3,800 lines) is a self-contained HTML file that mirrors the CJC chess engine in JavaScript to provide an interactive experience. CJC does not yet compile to WebAssembly or have a GUI toolkit, so the browser UI is written in plain JavaScript with no frameworks or libraries.

The JS frontend includes a more advanced version of the agent (CJC-Lang does not yet compile to WebAssembly):

- **Actor-Critic network** — Residual blocks, GELU activation, He initialization, dual policy + value heads (~110K parameters)
- **A2C with GAE** — Advantage Actor-Critic with Generalized Advantage Estimation (γ=0.99, λ=0.95), entropy regularization, gradient clipping
- **288-dimensional features** — Board state, attack maps, piece-square tables, pawn structure, material balance, king safety, game phase
- **Curriculum learning** — Three training phases: vs Random → vs Heuristic → vs Self-Play
- **Training dashboard** — Real-time charts for reward, win rate, loss, entropy
- **4 baseline opponents** — Random, Heuristic (greedy material), 1-ply Minimax, Untrained network
- **Tactical puzzle suite** — 5 predefined positions with known best moves

The JS frontend demonstrates what CJC-Lang will eventually support natively. The neural network math is hand-written — no TensorFlow, no PyTorch, no ML libraries. Every gradient is derived manually.

### Training Results (500 Episodes)

| Metric | Value |
|--------|-------|
| vs Random | 60% win rate (12W / 8D / 0L) |
| vs Heuristic | 20% win rate (4W / 16D / 0L) |
| vs 1-ply Minimax | 20% win rate (2W / 5D / 3L) |
| Tactical Puzzles | 1/5 (pawn promotion) |
| Value Loss | Decreased ~47% over training |
| Policy Entropy | Decreased ~16% (policy specializing) |

These results are modest. The agent learns basic material awareness and consistently beats a random player, but does not play strong chess. 110K parameters with 500 episodes of policy gradient training is not enough for expert play. The demo proves the training pipeline works end-to-end, not that it produces a grandmaster.

### What This Demo Proves About CJC-Lang

1. **CJC-Lang can express non-trivial algorithms** — A complete chess engine with all standard rules runs correctly through both execution backends
2. **CJC-Lang can do numerical computing** — Neural network forward passes, gradient computation, and weight updates work through CJC-Lang's runtime
3. **Determinism holds under load** — Training with the same seed produces identical results across runs, verified by dedicated tests
4. **Both backends agree** — Every chess RL test produces identical output from the AST interpreter and MIR executor
5. **The test infrastructure scales** — 216 chess RL tests including fuzz testing and multi-episode training sequences

### What This Demo Does Not Prove

- CJC-Lang is not ready for production use
- The language still lacks features expected for real ML work (module system, variadic functions, browser target)
- The JS frontend does the heavy lifting for the interactive experience
- 500 episodes of A2C is not enough to produce competitive chess play
- The CJC-Lang-native network (V1) is simpler than the JS network (V2)

### Running the Demo

**CJC-Lang Tests:**
```bash
cargo test --workspace                    # 5,353 default-build tests (6,715+ markers incl. feature-gated)
cargo test --test test_chess_rl_project   # Chess engine + V1 RL (150 tests)
cargo test --test test_chess_rl_advanced  # ML upgrade tests (66 tests)
```

**Interactive Browser Demo:**
Open `examples/chess_rl_platform.html` in any browser. No server, no build step, no dependencies.
- Click **Play Agent** to play against the neural network
- Click the **Training** tab, select 500 episodes, and click Train
- Watch learning curves update in real time

---

## Language Overview

### Variables and Types

```cjcl
let x: i64 = 42;
let pi: f64 = 3.14159;
let name: str = "CJC-Lang";
let flag: bool = true;
let mut counter: i64 = 0;
counter += 1;
```

### Functions

```cjcl
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn greet(name: str) {
    print(f"Hello, {name}!");
}
```

### Structs

```cjcl
struct Point {
    x: f64,
    y: f64,
}

fn distance(p: Point) -> f64 {
    sqrt(p.x ** 2.0 + p.y ** 2.0)
}
```

### Enums and Pattern Matching

```cjcl
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

```cjcl
fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

fn main() -> i64 {
    let double = |x: i64| x * 2;
    apply(double, 21)   // returns 42
}
```

### Control Flow

```cjcl
if x > 0 {
    print("positive");
} else {
    print("non-positive");
}

while count < 100 {
    count += 1;
}

for i in 0..10 {
    print(i);
}
```

### Operators

```cjcl
// Arithmetic: + - * / % **
// Comparison: == != < > <= >=
// Logical: && || !
// Bitwise: & | ^ ~ << >>
// Pipe: data |> transform() |> output()
```

---

## Tensor Operations

```cjcl
let a = Tensor.zeros([3, 3]);
let b = Tensor.ones([3, 3]);
let c = Tensor.randn([2, 4]);
let d = Tensor.eye(3);
let e = [| 1.0, 2.0; 3.0, 4.0 |];

let product = matmul(a, b);
let transposed = c.transpose();
let total: f64 = e.sum();       // Kahan-stable
```

---

## Built-in Function Categories

| Category | Count | Examples |
|----------|-------|---------|
| Mathematics | 19+ | `sqrt`, `log`, `exp`, `sin`, `cos`, `abs`, `clamp` |
| Statistics | 35+ | `mean`, `sd`, `median`, `cor`, `quantile`, `z_score` |
| Distributions | 24 | Normal, t, Chi-squared, F, Beta, Gamma (PDF/CDF/PPF) |
| Hypothesis Tests | 24 | `t_test`, `anova_oneway`, `chi_squared_test`, `tukey_hsd` |
| Linear Algebra | 9+ | `matmul`, `det`, `solve`, `lstsq`, `eigh` |
| ML / Deep Learning | 40+ | `relu`, `gelu`, `attention`, `conv2d`, `Adam.new` |
| Signal Processing | 14+ | FFT, RFFT, IFFT, PSD, window functions |
| Data Wrangling | 73+ | `filter`, `group_by`, `join`, `pivot_longer`, `window_sum` |

---

## Architecture

```
Source → [Lexer] → Tokens → [Parser] → AST → [TypeChecker] → Typed AST
                                                    |
                              ┌─────────────────────┼──────────────────────┐
                              v                                            v
                         [Eval] (v1)                              [HIR Lowering]
                    Tree-walk interpreter                               |
                                                                        v
                                                                  [MIR Lowering]
                                                                        |
                                                                        v
                                                                  [Optimizer]
                                                                 CF + DCE passes
                                                                        |
                                                                        v
                                                                  [MIR-Exec] (v2)
                                                               Register machine
```

### Workspace Crates (20)

| Crate | Purpose |
|-------|---------|
| `cjc-lexer` | Tokenization |
| `cjc-parser` | Pratt parser |
| `cjc-ast` | AST node definitions |
| `cjc-types` | Type system and inference |
| `cjc-diag` | Diagnostic infrastructure |
| `cjc-hir` | AST → HIR lowering, capture analysis |
| `cjc-mir` | HIR → MIR lowering, optimizer, NoGC verifier |
| `cjc-eval` | AST tree-walk interpreter (v1) |
| `cjc-mir-exec` | MIR register-machine executor (v2) |
| `cjc-dispatch` | Operator dispatch layer |
| `cjc-runtime` | 363 builtins, tensor system, COW buffers |
| `cjc-ad` | Automatic differentiation (forward + reverse) |
| `cjc-data` | DataFrame DSL (filter, group_by, join) |
| `cjc-repro` | Deterministic RNG, Kahan/Binned accumulators |
| `cjc-regex` | NFA-based regex engine |
| `cjc-snap` | Binary serialization |
| `cjc-vizor` | Grammar-of-graphics visualization |
| `cjc-module` | Multi-file module system (wired via `cjcl run --multi-file`) |
| `cjc-cli` | CLI frontend and REPL |
| `cjc-analyzer` | Language server (experimental) |

---

## Memory Model

- **Copy-on-Write Buffers** — Tensors share backing storage via reference counting. Mutation triggers a copy only when shared.
- **No Garbage Collector** — RC-based memory with explicit lifecycle. No GC pauses.
- **Arena Allocation** — Per-frame arenas in the MIR executor for fast allocation.
- **NoGC Verification** — Static analysis proves that `nogc`-marked functions never trigger allocation.

## Numerical Guarantees

- **Kahan Summation** — All floating-point reductions use compensated summation.
- **Seeded RNG** — SplitMix64 produces identical sequences from the same seed.
- **Deterministic Ordering** — `BTreeMap`/`BTreeSet` everywhere, no hash randomization.
- **IEEE 754 Compliance** — `total_cmp()` for deterministic NaN ordering.

---

## CLI Usage

```
cjcl run <file.cjcl>    Run a CJC-Lang program
cjcl repl               Start the interactive REPL
cjcl lex <file.cjcl>    Tokenize and print tokens
cjcl parse <file.cjcl>  Parse and pretty-print AST
cjcl check <file.cjcl>  Type-check without running
```

| Flag | Description |
|------|-------------|
| `--seed N` | Set RNG seed (default: 42) |
| `--mir-opt` | Enable MIR optimizations |
| `--time` | Print execution time |
| `--color` / `--no-color` | Control color output |

---

## Building

```bash
git clone <repo-url>
cd CJC
cargo build --workspace
cargo test --workspace
```

Requires a Rust toolchain (rustc, cargo). No other dependencies.

## Project Structure

```
crates/                  — 20 Rust crates (language implementation)
tests/                   — 4,802 integration-test markers across 118 test binaries
  chess_rl_project/      — Chess RL CJC tests (69 tests)
  chess_rl_advanced/     — ML upgrade tests (84 tests)
  chess_rl_hardening/    — Robustness tests (104 tests)
  chess_rl_playability/  — End-to-end gameplay (62 tests)
examples/
  chess_rl_platform.html — Interactive browser demo (self-contained)
docs/                    — Design documents and specifications
gallery/                 — 80 Vizor-generated SVG visualizations
```

## Test Suite

```bash
cargo test --workspace         # Run everything
cargo test -p cjc-runtime      # Run a specific crate's tests
cargo test --test test_eval    # Run interpreter tests
```

| Suite | Tests | Description |
|-------|-------|-------------|
| Raw `#[test]` markers | **6,715** | 1,913 unit + 4,802 integration |
| Default build passing | **5,353** | `cargo test --workspace --release` (118 binaries, 0 failures) |
| Property tests | 22 `proptest!` macros | Each generates hundreds of random cases per run |
| Fuzz targets | 38 `bolero::check` | Structure-aware fuzzing |
| Chess RL | 319 | Engine, training, determinism, fuzz (across 4 suites) |
| Dedicated parity files | 179 across 13 files | Eval vs MIR-exec bit-identity |
| Language Hardening | 83 | Syntax, operators, ML builtins |
| Data Science | 100+ | Tidy operations, CSV, joins |
| Mathematics | 120 | Stats, linalg, distributions |

---

## License

MIT
