> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [REBRAND_NOTICE.md](REBRAND_NOTICE.md) for the full mapping.

# CJC Codebase Mentor Plan — Learning to Navigate, Understand, and Extend CJC

**For:** Adam Ezzat
**Goal:** Build deep understanding of the CJC codebase so you can independently make improvements, fix bugs, and add features.

---

## How This Plan Works

Each level builds on the previous one. Start at Level 1 and work through sequentially. Each level has:
- **Reading assignments** — specific files to read, with annotations
- **Exercises** — small tasks to do yourself (with hints, not answers)
- **Checkpoints** — how to verify you understand the material

Estimated time: ~2-3 weeks at 1-2 hours/day.

---

## Level 1: The 10,000-Foot View (Day 1-2)

### Goal: Understand the pipeline and how data flows through CJC

**Read these files in order:**

1. **`Cargo.toml`** (workspace root) — See all 20 crates and their dependencies
   - *Note which crates depend on which. Draw the dependency graph on paper.*

2. **`crates/cjc-cli/src/main.rs`** (lines 1-100, then 550-650) — The entry point
   - *This is where `cjc run file.cjc` starts. Follow the code path.*

3. **`crates/cjc-lexer/src/lib.rs`** (lines 1-50, then the `tokenize()` method) — Tokenization
   - *The lexer turns `"let x = 5"` into `[Let, Ident("x"), Eq, IntLit(5)]`*

4. **`crates/cjc-parser/src/lib.rs`** (lines 1-100, then `parse_source()`) — Parsing
   - *The parser turns tokens into an AST (abstract syntax tree)*

5. **`crates/cjc-ast/src/lib.rs`** (skim the full file) — AST node definitions
   - *This is the "vocabulary" of CJC. Every language construct is an enum variant here.*

6. **`crates/cjc-eval/src/lib.rs`** (lines 1-100, then `exec()` and `eval_expr()`) — The v1 interpreter
   - *This walks the AST tree and evaluates it directly*

### Exercise 1.1: Trace a Simple Program
Take this program:
```cjc
fn main() -> i64 {
    let x: i64 = 5;
    let y: i64 = x + 3;
    print(y);
    0
}
```
Trace it through:
1. What tokens does the lexer produce?
2. What AST nodes does the parser create?
3. How does eval execute it step by step?

*Hint: You can use `cjc lex file.cjc` and `cjc parse file.cjc` to see actual output.*

### Exercise 1.2: Add a Trivial Builtin
Add a builtin function `hello_world()` that returns the string `"Hello from CJC!"`.
You need to touch exactly 3 files (the wiring pattern):
1. `crates/cjc-runtime/src/builtins.rs` — The stateless dispatch
2. `crates/cjc-eval/src/lib.rs` — Wire it in `is_known_builtin`
3. `crates/cjc-mir-exec/src/lib.rs` — Wire it in `is_known_builtin`

*This exercise teaches the most important pattern in CJC: the three-place wiring.*

### Checkpoint
You should be able to answer:
- [ ] What are the 5 stages of the pipeline?
- [ ] What is the difference between eval (v1) and mir-exec (v2)?
- [ ] Why do builtins need to be wired in 3 places?
- [ ] What does `--mir-opt` do?

---

## Level 2: The Type System and AST (Day 3-4)

### Goal: Understand how CJC represents and checks types

**Read these files:**

1. **`crates/cjc-ast/src/lib.rs`** — Focus on:
   - `ExprKind` enum (line ~345) — Every expression type
   - `StmtKind` enum (line ~280) — Every statement type
   - `TypeAnnotation` (search for it) — How types are written in source

2. **`crates/cjc-types/src/lib.rs`** — The type system
   - `Type` enum — All types CJC knows about
   - Type inference rules

3. **`crates/cjc-runtime/src/value.rs`** (or wherever `Value` is defined) — Runtime values
   - `Value` enum — What values look like at runtime

### Exercise 2.1: Map AST → Value
For each AST expression kind, identify what `Value` variant it produces:
- `ExprKind::IntLit(5)` → `Value::Int(5)`
- `ExprKind::FloatLit(3.14)` → ?
- `ExprKind::ArrayLit([1,2,3])` → ?
- `ExprKind::Call("mean", [arr])` → ?
- `ExprKind::IfExpr { ... }` → ?

### Exercise 2.2: Read a Parser Rule
Find the `parse_if` method in `crates/cjc-parser/src/lib.rs`. Read it and answer:
- How does it distinguish `if` as a statement vs `if` as an expression?
- What happens when there's no `else` branch?
- How is `else if` handled (chain or nesting)?

### Checkpoint
- [ ] Can you list 10 `ExprKind` variants from memory?
- [ ] What is `Value::Void` used for?
- [ ] What is the difference between `TypeAnnotation` and `Type`?

---

## Level 3: The MIR Pipeline (Day 5-7)

### Goal: Understand how AST becomes MIR and how MIR is executed

**Read these files:**

1. **`crates/cjc-hir/src/lib.rs`** — HIR lowering
   - *AST → HIR is mainly about capture analysis for closures*
   - Focus on: What does HIR add that AST doesn't have?

2. **`crates/cjc-mir/src/lib.rs`** — MIR definition
   - `MirInstr` enum — The instruction set of the register machine
   - *Each instruction operates on numbered registers (r0, r1, ...)*

3. **`crates/cjc-mir-exec/src/lib.rs`** — MIR execution
   - `step()` or `exec_instr()` — How one MIR instruction is executed
   - *Compare this to eval's `eval_expr()` — same semantics, different representation*

4. **`crates/cjc-mir/src/optimize.rs`** — MIR optimizer
   - Constant folding and dead code elimination
   - *These are the `--mir-opt` passes*

### Exercise 3.1: Trace MIR
Use `cjc run --mir file.cjc` (or however MIR output is shown) to see the MIR for:
```cjc
fn add(a: i64, b: i64) -> i64 {
    a + b
}
fn main() -> i64 {
    let result: i64 = add(3, 4);
    print(result);
    0
}
```
Questions:
- How many registers are used?
- What MIR instruction does `a + b` become?
- How is the function call represented?

### Exercise 3.2: Understand Parity
Run the same program through both executors and compare:
```rust
// In a test file:
let src = "fn main() -> i64 { 5 + 3 }";
let eval_result = /* run through eval */;
let mir_result = /* run through mir-exec */;
assert_eq!(eval_result, mir_result);
```
*This is what "parity" means — both executors must agree.*

### Checkpoint
- [ ] What is the difference between HIR and MIR?
- [ ] What does constant folding do?
- [ ] Why is parity testing important?

---

## Level 4: The Runtime (Day 8-10)

### Goal: Understand values, tensors, dispatch, and determinism

**Read these files:**

1. **`crates/cjc-runtime/src/builtins.rs`** — The big dispatch table
   - *This is the largest file. Don't read all of it — use search to find specific builtins.*
   - Find `"mean"`, `"dot"`, `"array_push"` — see the pattern

2. **`crates/cjc-runtime/src/tensor.rs`** — Tensor operations
   - How tensors are stored (flat `Vec<f64>` + shape)
   - BLAS-like operations (matmul, transpose)

3. **`crates/cjc-repro/src/lib.rs`** — Determinism primitives
   - `SplitMix64` — The RNG
   - `KahanAccumulatorF64` — Compensated summation
   - *Why Kahan? Because `sum([1e16, 1.0, -1e16])` should be `1.0`, not `0.0`*

4. **`crates/cjc-dispatch/src/lib.rs`** — Operator dispatch
   - How `a + b` becomes the right operation based on types

### Exercise 4.1: Add a Builtin (Real)
Add `array_reverse(arr)` that returns a reversed copy of an array.
1. Add to `dispatch_builtin` in `builtins.rs`
2. Wire in both executors
3. Write a test that verifies parity

### Exercise 4.2: Understand Determinism
Read `crates/cjc-repro/src/lib.rs` and answer:
- Why SplitMix64 instead of thread_rng?
- Why BTreeMap instead of HashMap?
- Why no FMA in SIMD?
- What does `BinnedAccumulatorF64::merge()` enable?

### Checkpoint
- [ ] Can you add a new builtin without help?
- [ ] Can you explain why `HashMap` is banned?
- [ ] What is the COW (copy-on-write) pattern used for arrays?

---

## Level 5: The Data DSL (Day 11-13)

### Goal: Understand TidyView and the DataFrame system

**Read these files:**

1. **`crates/cjc-data/src/lib.rs`** — The core DataFrame + TidyView
   - `DataFrame` struct — Column-oriented storage
   - `TidyView` — Lazy-ish operations (filter, select, mutate, etc.)
   - `DExpr` enum — Data expressions (column references, math, aggregations)

2. **`crates/cjc-data/src/tidy_dispatch.rs`** — How CJC code calls TidyView methods

3. **`crates/cjc-data/src/lazy.rs`** — The new lazy query IR
   - `ViewNode` — The query plan
   - `LazyView` — Builder pattern
   - `optimize()` — Predicate pushdown, filter merging

4. **`crates/cjc-data/src/agg_kernels.rs`** — Specialized aggregation
5. **`crates/cjc-data/src/column_meta.rs`** — Zone maps + sorted flags

### Exercise 5.1: Trace a TidyView Pipeline
```cjc
let df = dataframe(columns: ["name", "age", "score"],
                   data: [["Alice", 30, 95], ["Bob", 25, 87], ["Carol", 35, 92]]);
let view = df.view();
let filtered = view.filter(col("age") > 28);
let result = filtered.select(["name", "score"]);
```
Trace what happens internally:
- What does `df.view()` create?
- What does `.filter()` store? (Hint: it's lazy)
- When does actual computation happen?

### Exercise 5.2: Add a New TidyAgg
Add `TidyAgg::Product(String)` that computes the product of a column.
- Add variant to `TidyAgg` enum
- Add reduction logic in `agg_reduce` / `eval_agg_over_groups_fast`
- Wire in `tidy_dispatch.rs`
- Write a test

### Checkpoint
- [ ] What is the difference between eager and lazy evaluation in TidyView?
- [ ] Why does group_by use BTreeMap?
- [ ] What are zone maps and when do they help?

---

## Level 6: Making Real Changes (Day 14+)

### Goal: Independently identify and implement improvements

At this point, you should be able to:

1. **Read any file** and understand what it does
2. **Trace any CJC program** through the full pipeline
3. **Add builtins** following the three-place wiring pattern
4. **Write parity tests** ensuring both executors agree
5. **Understand determinism** invariants and why they matter

### Suggested First Solo Projects

**Easy (1-2 hours):**
- Add `array_zip(a, b)` builtin → returns array of [a[i], b[i]] pairs
- Add `string_split(s, delimiter)` builtin
- Add `math_clamp(x, min, max)` builtin
- Improve an error message you find confusing

**Medium (half day):**
- Add `cjc init` command that creates a project skeleton
- Add tab completion to the REPL for builtin names
- Add `--verbose` flag that prints timing for each pipeline stage
- Add a new DExpr window function (e.g., `Percent_Rank`)

**Hard (1-2 days):**
- Add `cjc fmt` command that pretty-prints CJC source code (PrettyPrinter exists in AST)
- Expand the LSP analyzer with go-to-definition
- Add a new optimization pass to the MIR optimizer
- Implement `cjc test` command that discovers and runs `.cjc` test files

---

## Reference Card

### Key Files (Bookmark These)

| What | File |
|------|------|
| Entry point | `crates/cjc-cli/src/main.rs` |
| Lexer | `crates/cjc-lexer/src/lib.rs` |
| Parser | `crates/cjc-parser/src/lib.rs` |
| AST definitions | `crates/cjc-ast/src/lib.rs` |
| Type system | `crates/cjc-types/src/lib.rs` |
| Eval (v1) | `crates/cjc-eval/src/lib.rs` |
| MIR-exec (v2) | `crates/cjc-mir-exec/src/lib.rs` |
| Builtins dispatch | `crates/cjc-runtime/src/builtins.rs` |
| DataFrame/TidyView | `crates/cjc-data/src/lib.rs` |
| Determinism | `crates/cjc-repro/src/lib.rs` |
| Diagnostics | `crates/cjc-diag/src/lib.rs` |

### Key Patterns

1. **Three-place wiring:** builtins.rs + eval + mir-exec
2. **Parity testing:** Always test both executors produce identical output
3. **Determinism invariants:** BTreeMap, Kahan, SplitMix64, no FMA
4. **COW arrays:** `array_push` returns new array, doesn't mutate
5. **Column-oriented data:** DataFrame stores `Vec<Column>`, not `Vec<Row>`

### Useful Commands

```bash
# Run a CJC program
cjc run file.cjc

# Run with MIR optimizer
cjc run --mir-opt file.cjc

# See tokens
cjc lex file.cjc

# See AST
cjc parse file.cjc

# Type check only
cjc check file.cjc

# Start REPL
cjc repl

# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p cjc-data --lib

# Run specific test file
cargo test --test test_beta_hardening

# Check compilation
cargo check --workspace
```

### How to Find Things

```bash
# Find where a builtin is implemented
grep -n '"mean"' crates/cjc-runtime/src/builtins.rs

# Find where an AST node is handled in eval
grep -n 'ExprKind::IfExpr' crates/cjc-eval/src/lib.rs

# Find all tests for a feature
grep -rn 'fn test_.*filter' tests/

# Find all usages of a type
grep -rn 'TidyAgg' crates/cjc-data/src/
```

---

## Weekly Check-In Template

After each week, answer these questions:

1. What 3 files did I read this week?
2. What 1 exercise did I complete?
3. What 1 thing surprised me about the codebase?
4. What 1 thing would I change if I could?
5. What's my plan for next week?

---

## The Meta-Lesson

CJC is ~96K LOC across 20 crates. You will NOT understand it all at once. The key insight is:

**Follow the data.**

Every CJC program is data flowing through a pipeline:
```
Source text → Tokens → AST → HIR → MIR → Execution → Value
```

If you can trace a piece of data through this pipeline, you can understand any feature. If you can't, narrow your focus until you can.

Start small. Add a builtin. Write a test. Read one file. Understanding compounds.
