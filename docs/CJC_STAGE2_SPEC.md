# CJC Stage 2 Specification
## IR Pipeline, Closures, Pattern Matching, For-Loops

**Version:** 2.0-draft
**Date:** 2025-01-XX
**Status:** SPECIFICATION — requires sign-off before implementation

---

## Table of Contents

1. [Stage 2 MVP Scope](#1-stage-2-mvp-scope)
2. [IR-Exec Backend Choice](#2-ir-exec-backend-choice)
3. [HIR / MIR / Kernel IR Structures](#3-hir--mir--kernel-ir-structures)
4. [Closure Capture Rules](#4-closure-capture-rules)
5. [Pattern Matching MVP](#5-pattern-matching-mvp)
6. [For-Loop MVP](#6-for-loop-mvp)
7. [Parity Gates for LLVM Expansion](#7-parity-gates-for-llvm-expansion)
8. [Milestone Plan 2.0 → 2.5](#8-milestone-plan-20--25)

---

## 1. Stage 2 MVP Scope

### 1.1 What Stage 2 IS

Stage 2 transforms CJC from a tree-walk interpreted language into a
**compiled-IR language** with proper closures, control flow, and an
optimization pipeline — while preserving the 3-layer memory guarantee.

**IN scope:**

| Feature | Justification |
|---------|--------------|
| HIR → MIR → Kernel IR pipeline | Required foundation for optimization and codegen |
| MIR interpreter (reference backend) | Executes MIR directly; replaces tree-walk eval |
| Closures with explicit capture | Enables higher-order programming (map, filter, fold) |
| `for` loops with range and iterator protocol | Basic iteration without recursion workarounds |
| `match` with literal, ident, tuple, struct patterns | Discriminated control flow for enums/tagged data |
| `nogc` verifier in MIR | Enforces no-GC-allocation at IR level (not just type-check) |
| Constant folding + dead-code elimination | Minimum viable optimizer passes |
| 138-test parity gate | All existing tests must pass on the new backend |

**OUT of scope (deferred to Stage 3+):**

| Feature | Reason |
|---------|--------|
| LLVM codegen | Stage 2 proves the IR is correct; LLVM is Stage 3 |
| Enums / algebraic data types | Match works on existing types first |
| Async / concurrency | Orthogonal; needs runtime redesign |
| Module system / imports from files | Current single-file model is sufficient |
| Trait objects / dynamic dispatch | Multiple dispatch already works at call sites |
| SIMD / vectorized kernels | Optimization, not semantics |
| `while let` / `if let` | Sugar; can be added post-match |

### 1.2 What Exists Today (Stage 1 Baseline)

Grounded in the actual codebase as of this writing:

```
Crate            Lines   Key Facts
─────────────    ─────   ──────────────────────────────────────
cjc-lexer          485   53 TokenKinds; `For` keyword exists (unused);
                         `FatArrow` (=>) exists; single `|` → error
cjc-ast            430   8 DeclKind, 6 StmtKind, 17 ExprKind;
                         Lambda{params,body} exists; no Pattern type
cjc-parser       2,059   Pratt precedence parser; no for/match parsing;
                         Lambda never parsed (lexer blocks `|`)
cjc-types          520   Type enum: I32..Void, Tensor, Buffer, Array,
                         Tuple, Struct, Class, Fn{params,ret}, Var, Error
cjc-eval         1,800   Tree-walk interpreter; scope stack
                         (Vec<HashMap<String,Value>>); 10 Value variants;
                         Lambda → synthetic FnDecl (no capture)
cjc-runtime        780   Buffer<T> COW via Rc<RefCell<Vec<T>>>;
                         GcHeap mark-sweep (allocated but never used by
                         interpreter for structs — they are RAII values)
cjc-dispatch       320   Multi-dispatch: name+arity → Vec<Candidate>
cjc-ad             650   Dual (forward), GradGraph (reverse); 15 GradOps
cjc-data           420   DataFrame, Series, GroupBy
cjc-repro          180   Seeded Rng (Xoshiro256**)
cjc-diag           200   Diagnostic, Severity, Span
cjc-cli            150   CLI entry point; --time flag

Total           ~8,000   138 tests passing; 3 demos; 4 benchmarks
```

### 1.3 Anti-Feature-Creep Rules

1. **No new types in Value enum** unless required by a listed feature.
2. **No new token kinds** beyond what closures/match/for need.
3. **No changes to Buffer/Tensor/COW semantics** — they are stable.
4. **No changes to the AD engine** — it stays as a Rust-only library.
5. **Every new line of code must be covered by at least one test.**

---

## 2. IR-Exec Backend Choice

### 2.1 Decision: MIR Tree Interpreter (NOT Bytecode VM)

**Chosen: MIR Interpreter** — a tree-walking evaluator over the MIR
data structure (not a flat bytecode VM).

### 2.2 Rationale

| Criterion | Bytecode VM | MIR Interpreter | Winner |
|-----------|-------------|-----------------|--------|
| Implementation complexity | High (need encoder, decoder, stack machine, jump patching) | Low (recursive match over MIR enum) | **MIR** |
| Debuggability | Hard (offsets, disassembly) | Easy (print MIR tree, step through) | **MIR** |
| Path to LLVM | Must re-derive structure from flat bytecode | MIR maps ~1:1 to LLVM IR Basic Blocks | **MIR** |
| Optimization passes | Operate on bytecode (peephole) | Operate on structured MIR (standard SSA transforms) | **MIR** |
| Runtime speed | Faster dispatch (computed goto) | Slightly slower (enum match overhead) | VM |
| Existing codebase fit | Must discard tree-walk eval entirely | Tree-walk style is already proven in cjc-eval | **MIR** |

**Key insight:** CJC's Stage 2 goal is **correctness + IR design**, not peak
interpreter speed. A bytecode VM optimizes the wrong thing — we'd spend weeks
on encoding/decoding/jump-patching when that effort is wasted once LLVM
codegen arrives in Stage 3. The MIR interpreter is a *reference implementation*
that validates our IR is sound.

### 2.3 Architecture Diagram

```
                    STAGE 2 PIPELINE
                    ════════════════

  Source (.cjc)
       │
       ▼
  ┌─────────┐
  │  Lexer   │  cjc-lexer  (existing, minor token additions)
  └────┬─────┘
       │ Vec<Token>
       ▼
  ┌─────────┐
  │  Parser  │  cjc-parser (add for/match/closure parsing)
  └────┬─────┘
       │ AST (Program)
       ▼
  ┌─────────┐
  │ Lowering │  cjc-hir  (NEW crate: AST → HIR)
  └────┬─────┘
       │ HIR
       ▼
  ┌─────────┐
  │ MIR Gen  │  cjc-mir  (NEW crate: HIR → MIR, closure conversion,
  └────┬─────┘           pattern compilation, desugaring)
       │ MIR
       ▼
  ┌─────────────┐
  │ MIR Optimize │  (constant fold, dead-code elim, nogc verify)
  └──────┬───────┘
         │ MIR (optimized)
         ▼
  ┌──────────────┐
  │ MIR Executor  │  cjc-mir-exec  (NEW crate: reference interpreter)
  └──────────────┘
         │
         ▼
      Output
```

**New crates for Stage 2:**

| Crate | Purpose |
|-------|---------|
| `cjc-hir` | HIR data structures + AST → HIR lowering |
| `cjc-mir` | MIR data structures + HIR → MIR lowering + optimizer passes |
| `cjc-mir-exec` | MIR reference interpreter |

---

## 3. HIR / MIR / Kernel IR Structures

### 3.1 HIR (High-level IR)

HIR is a *desugared AST*. It removes syntactic sugar but preserves
high-level types and structure. Every HIR node carries a `TypeId`
resolved by the type checker.

```rust
// cjc-hir/src/lib.rs

/// Unique ID for every HIR node (for diagnostics, debug).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirId(pub u32);

/// Type-annotated identifier.
#[derive(Debug, Clone)]
pub struct HirIdent {
    pub name: String,
    pub hir_id: HirId,
    pub ty: TypeId,
}

/// Top-level item after desugaring.
#[derive(Debug, Clone)]
pub enum HirItem {
    Fn(HirFn),
    Struct(HirStructDef),
    // Class, Trait, Impl kept from AST with types resolved
}

#[derive(Debug, Clone)]
pub struct HirFn {
    pub name: String,
    pub params: Vec<HirParam>,
    pub ret_ty: TypeId,
    pub body: HirBlock,
    pub is_nogc: bool,
    pub hir_id: HirId,
}

#[derive(Debug, Clone)]
pub struct HirParam {
    pub name: String,
    pub ty: TypeId,
    pub hir_id: HirId,
}

#[derive(Debug, Clone)]
pub struct HirBlock {
    pub stmts: Vec<HirStmt>,
    pub expr: Option<Box<HirExpr>>,   // trailing expression = block value
    pub hir_id: HirId,
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    Let {
        name: String,
        ty: TypeId,
        init: HirExpr,
        hir_id: HirId,
    },
    Expr(HirExpr),
    // No Return here — return is an expression in HIR
}

#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: TypeId,
    pub hir_id: HirId,
}

#[derive(Debug, Clone)]
pub enum HirExprKind {
    /// Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),

    /// Variable reference (resolved to definition site)
    Var(String),

    /// Binary op (desugared: no Pipe — pipe is lowered to Call)
    Binary { op: BinOp, left: Box<HirExpr>, right: Box<HirExpr> },

    /// Unary op
    Unary { op: UnaryOp, operand: Box<HirExpr> },

    /// Function call (callee resolved to function name or closure var)
    Call { callee: Box<HirExpr>, args: Vec<HirExpr> },

    /// Field access
    Field { object: Box<HirExpr>, name: String },

    /// Index
    Index { object: Box<HirExpr>, index: Box<HirExpr> },

    /// Block
    Block(HirBlock),

    /// If expression (always has else — parser inserts `else { void }`)
    If {
        cond: Box<HirExpr>,
        then_branch: HirBlock,
        else_branch: HirBlock,
    },

    /// While loop
    While { cond: Box<HirExpr>, body: HirBlock },

    /// For loop (desugared from `for x in iter { ... }`)
    /// See §6 for desugaring rules.
    For {
        var: String,
        iter: Box<HirExpr>,
        body: HirBlock,
    },

    /// Match expression
    /// See §5 for pattern compilation.
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
    },

    /// Closure (with explicit capture list after closure conversion)
    Closure {
        params: Vec<HirParam>,
        captures: Vec<HirCapture>,
        body: Box<HirExpr>,
        fn_ty: TypeId,
    },

    /// Struct literal
    StructLit { name: String, fields: Vec<(String, HirExpr)> },

    /// Array literal
    ArrayLit(Vec<HirExpr>),

    /// Return
    Return(Option<Box<HirExpr>>),

    /// Assign
    Assign { target: Box<HirExpr>, value: Box<HirExpr> },

    /// NoGc block (carries nogc flag for verifier)
    NoGcBlock(HirBlock),
}

/// One arm of a match expression.
#[derive(Debug, Clone)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<Box<HirExpr>>,   // optional `if cond`
    pub body: HirExpr,
}

/// See §5 for pattern details.
#[derive(Debug, Clone)]
pub enum HirPattern {
    /// `_` — matches anything, binds nothing
    Wildcard,
    /// `x` — matches anything, binds to `x`
    Binding(String),
    /// `42`, `3.14`, `true`, `"hello"` — exact match
    Literal(HirLiteral),
    /// `(a, b, c)` — tuple destructuring
    Tuple(Vec<HirPattern>),
    /// `Foo { x, y }` — struct destructuring
    Struct { name: String, fields: Vec<(String, HirPattern)> },
}

#[derive(Debug, Clone)]
pub enum HirLiteral {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

/// A captured variable in a closure.
#[derive(Debug, Clone)]
pub struct HirCapture {
    pub name: String,
    pub ty: TypeId,
    pub mode: CaptureMode,
}

#[derive(Debug, Clone, Copy)]
pub enum CaptureMode {
    /// Immutable borrow (default for non-mut variables)
    Ref,
    /// Clone the value into the closure (default for `nogc` closures)
    Clone,
}
```

### 3.2 MIR (Mid-level IR)

MIR is a **control-flow graph** (CFG) of basic blocks. Every value is
an explicit temporary. This is the level where:
- Pattern matching is compiled to decision trees
- Closures are lambda-lifted to top-level functions with env parameters
- `nogc` verification runs
- Optimization passes operate

```rust
// cjc-mir/src/lib.rs

/// A MIR program is a collection of functions.
#[derive(Debug, Clone)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
    pub structs: Vec<MirStructDef>,
    pub entry: MirFnId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MirFnId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TempId(pub u32);

#[derive(Debug, Clone)]
pub struct MirFunction {
    pub id: MirFnId,
    pub name: String,
    pub params: Vec<MirLocal>,
    pub ret_ty: TypeId,
    pub locals: Vec<MirLocal>,        // all temporaries
    pub blocks: Vec<MirBlock>,        // basic blocks
    pub is_nogc: bool,
}

#[derive(Debug, Clone)]
pub struct MirLocal {
    pub id: TempId,
    pub name: Option<String>,         // None for compiler temporaries
    pub ty: TypeId,
}

#[derive(Debug, Clone)]
pub struct MirBlock {
    pub id: BlockId,
    pub stmts: Vec<MirStmt>,
    pub terminator: MirTerminator,
}

/// A MIR statement produces a value in a temporary.
#[derive(Debug, Clone)]
pub enum MirStmt {
    /// `dest = constant`
    Const { dest: TempId, value: MirConst },

    /// `dest = src` (copy/move)
    Copy { dest: TempId, src: TempId },

    /// `dest = op(a, b)` or `dest = op(a)`
    BinOp { dest: TempId, op: BinOp, left: TempId, right: TempId },
    UnaryOp { dest: TempId, op: UnaryOp, operand: TempId },

    /// `dest = callee(args...)`
    Call { dest: TempId, callee: MirCallee, args: Vec<TempId> },

    /// `dest = object.field`
    FieldGet { dest: TempId, object: TempId, field: String },

    /// `object.field = value`
    FieldSet { object: TempId, field: String, value: TempId },

    /// `dest = object[index]`
    IndexGet { dest: TempId, object: TempId, index: TempId },

    /// `object[index] = value`
    IndexSet { object: TempId, index: TempId, value: TempId },

    /// `dest = StructName { field: val, ... }`
    StructLit { dest: TempId, name: String, fields: Vec<(String, TempId)> },

    /// `dest = [a, b, c, ...]`
    ArrayLit { dest: TempId, elems: Vec<TempId> },

    /// `dest = make_closure(fn_id, captures...)`
    MakeClosure { dest: TempId, fn_id: MirFnId, captures: Vec<TempId> },

    /// GC allocation marker (for nogc verification)
    GcAlloc { dest: TempId, ty: TypeId },
}

#[derive(Debug, Clone)]
pub enum MirCallee {
    /// Direct call to a known function
    Direct(MirFnId),
    /// Indirect call through a function value
    Indirect(TempId),
    /// Builtin function
    Builtin(String),
}

#[derive(Debug, Clone)]
pub enum MirConst {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Void,
}

/// Block terminator — controls flow between blocks.
#[derive(Debug, Clone)]
pub enum MirTerminator {
    /// Unconditional jump
    Goto(BlockId),

    /// Conditional branch
    Branch {
        cond: TempId,
        then_block: BlockId,
        else_block: BlockId,
    },

    /// Return from function
    Return(TempId),

    /// Switch (compiled from match)
    Switch {
        scrutinee: TempId,
        cases: Vec<(MirConst, BlockId)>,
        default: BlockId,
    },

    /// Unreachable (after exhaustive match)
    Unreachable,
}
```

### 3.3 Kernel IR (Stage 3 — Stub Only)

Kernel IR is a **SIMD-oriented loop IR** for tight numerical code inside
`nogc` blocks. It is **not implemented in Stage 2** — we only define the
data structures so the MIR→Kernel lowering boundary is clear.

```rust
// Reserved for Stage 3. Sketch only.
pub enum KernelOp {
    Load { dest: KReg, addr: KAddr },
    Store { addr: KAddr, src: KReg },
    FAdd { dest: KReg, a: KReg, b: KReg },
    FMul { dest: KReg, a: KReg, b: KReg },
    Loop { var: KReg, start: KReg, end: KReg, body: Vec<KernelOp> },
    // ... SIMD ops in Stage 3
}
```

---

## 4. Closure Capture Rules

### 4.1 Current State

The Lambda AST node exists:
```rust
ExprKind::Lambda { params: Vec<Param>, body: Box<Expr> }
```
But the lexer rejects single `|` (treats it as error), and the
interpreter converts lambdas to synthetic named functions with **no
capture** — they can only reference globals.

### 4.2 Stage 2 Design

**Syntax** (unchanged from AST):
```cjc
let f = |x: f64, y: f64| x + y
let g = |x: f64| {
    let sq = x * x
    sq + 1.0
}
```

**Lexer change:** Single `|` (pipe) becomes a valid token `TokenKind::Pipe`.
The lexer currently has this code:
```rust
b'|' => {
    if self.peek() == b'|' { ... PipePipe ... }
    else if self.peek() == b'>' { ... PipeGt ... }
    else { ERROR }  // ← change this to emit Pipe token
}
```

**Parser change:** When the parser sees `Pipe` in expression position,
it enters lambda-parsing mode:
```
|  param_list  |  expr
|  param_list  |  { block }
```

### 4.3 Capture Mode

**Default: Immutable Reference (`Ref`)**

Closures capture variables from the enclosing scope by immutable
reference by default. This means:

```cjc
let scale = 2.0
let f = |x: f64| x * scale   // `scale` captured by Ref
f(3.0)  // → 6.0
```

The captured `scale` is a read-only reference to the outer scope's value.
Attempting to assign to a captured variable is a compile error.

**Clone capture (explicit or `nogc`-forced):**

In `nogc` blocks, GC references cannot be held, so all captures are
**cloned** (deep-copied) into the closure:

```cjc
nogc {
    let bias = 1.0
    let f = |x: f64| x + bias   // `bias` captured by Clone (nogc rule)
    f(3.0)  // → 4.0
}
```

### 4.4 Capture Analysis Algorithm

During HIR→MIR lowering (closure conversion):

1. Walk the closure body and collect all `Var` references.
2. For each reference, check if it is:
   a. A parameter of the closure → skip (local)
   b. Defined in the closure body → skip (local)
   c. A known function name → skip (direct call)
   d. A builtin → skip
   e. Otherwise → **captured variable**
3. For each captured variable, determine the capture mode:
   - If inside a `nogc` context → `CaptureMode::Clone`
   - If the variable is `mut` in the outer scope → **error** (Stage 2
     does not support mutable captures; this prevents iterator invalidation)
   - Otherwise → `CaptureMode::Ref`
4. Lambda-lift the closure to a top-level MIR function with an extra
   `env` parameter (a struct containing all captures).

### 4.5 Runtime Representation

```rust
/// In the MIR executor:
pub enum MirValue {
    // ... existing variants ...
    Closure {
        fn_id: MirFnId,
        env: Vec<MirValue>,    // captured values
    },
}
```

When calling a `Closure`, the executor prepends the `env` values to
the argument list and dispatches to the lifted function.

### 4.6 Nogc Constraints

- A closure **defined inside** a `nogc` block must capture by `Clone`.
- A closure **called inside** a `nogc` block is fine — function pointers
  are not GC-allocated; they are stack values.
- A closure that captures a `ClassRef` (GC reference) **cannot** be
  created inside a `nogc` block — this is a compile error.

---

## 5. Pattern Matching MVP

### 5.1 Syntax

```cjc
match expr {
    pattern1 => body1,
    pattern2 if guard => body2,
    _ => default_body,
}
```

**Match is an expression** — every arm must produce the same type (or
the last arm can diverge).

### 5.2 MVP Patterns

| Pattern | Example | Semantics |
|---------|---------|-----------|
| Wildcard | `_` | Match anything, bind nothing |
| Binding | `x` | Match anything, bind to `x` |
| Int literal | `42` | Exact equality check |
| Float literal | `3.14` | Exact equality check |
| Bool literal | `true` | Exact equality check |
| String literal | `"hello"` | Exact equality check |
| Tuple | `(a, b, c)` | Destructure tuple, recurse into sub-patterns |
| Struct | `Point { x, y }` | Destructure struct fields |
| Struct with rename | `Point { x: px, y: py }` | Bind struct fields to different names |

**NOT in MVP:** Range patterns (`1..10`), slice patterns (`[a, b, ..]`),
enum variant patterns (no enums yet), or-patterns (`A | B`), nested
struct patterns deeper than 1 level.

### 5.3 Lexer/Parser Changes

**Lexer:** Add `Match` keyword to the keyword list. The `FatArrow` (`=>`)
token already exists.

```rust
// Add to keyword matching:
"match" => TokenKind::Match,
```

**Parser:** Add `parse_match_expr()`:

```
match_expr   := "match" expr "{" match_arms "}"
match_arms   := match_arm ("," match_arm)* ","?
match_arm    := pattern ("if" expr)? "=>" expr
pattern      := "_"
              | ident
              | literal
              | "(" pattern ("," pattern)* ")"
              | ident "{" field_pat ("," field_pat)* "}"
field_pat    := ident (":" pattern)?
```

### 5.4 Lowering Strategy: Decision Trees

Match compilation follows the standard **decision tree** approach
(as in Rust/OCaml compilers):

1. **HIR:** Match expressions are kept as-is with patterns.
2. **MIR:** Match is compiled to a tree of `Branch` and `Switch`
   terminators:

```
// Source:
match point {
    Point { x: 0, y } => y,
    Point { x, y: 0 } => x,
    _ => -1,
}

// MIR (simplified):
bb0:
    t0 = point.x
    branch (t0 == 0) → bb1, bb2

bb1:                          // x == 0
    t1 = point.y
    return t1

bb2:                          // x != 0
    t2 = point.y
    branch (t2 == 0) → bb3, bb4

bb3:                          // y == 0
    t3 = point.x
    return t3

bb4:                          // wildcard
    t4 = const -1
    return t4
```

### 5.5 Exhaustiveness Checking

Stage 2 MVP: **no exhaustiveness checking**. The compiler inserts an
implicit `_ => panic("non-exhaustive match")` at the end of every match.
Users should always write an explicit wildcard arm. Exhaustiveness
analysis is deferred to Stage 3 (requires enum support to be truly useful).

---

## 6. For-Loop MVP

### 6.1 Syntax

```cjc
// Range-based (exclusive upper bound):
for i in 0..10 {
    print(i)
}

// Iterator-based (arrays, tensors):
for x in my_array {
    print(x)
}
```

### 6.2 Lexer Changes

Add `DotDot` token for range syntax:

```rust
// In the lexer, when we see `.`:
b'.' => {
    if self.peek() == b'.' {
        self.advance();
        Token::new(TokenKind::DotDot, ...)   // NEW
    } else {
        Token::new(TokenKind::Dot, ...)
    }
}
```

Add `In` keyword:
```rust
"in" => TokenKind::In,
```

### 6.3 Parser

```
for_stmt := "for" ident "in" expr ".." expr block
          | "for" ident "in" expr block
```

The first form is range iteration; the second is iterator iteration.

**AST node addition:**
```rust
// In StmtKind:
For {
    var: Ident,
    iter: ForIter,
    body: Block,
}

pub enum ForIter {
    Range { start: Box<Expr>, end: Box<Expr> },
    Expr(Box<Expr>),
}
```

### 6.4 Desugaring (HIR Level)

**Range for-loop:**
```cjc
for i in 0..n { body }
```
desugars to:
```
let mut __iter_i = 0
while __iter_i < n {
    let i = __iter_i
    body
    __iter_i = __iter_i + 1
}
```

**Array for-loop:**
```cjc
for x in arr { body }
```
desugars to:
```
let __arr = arr
let __len = len(__arr)
let mut __idx = 0
while __idx < __len {
    let x = __arr[__idx]
    body
    __idx = __idx + 1
}
```

**Tensor for-loop:**
```cjc
for x in tensor { body }
```
desugars to the same pattern as array, with `len(tensor)` returning
the total element count and `tensor[i]` returning element `i` as a float.

### 6.5 Nogc Compatibility

For-loops inside `nogc` blocks are fully compatible — the desugared
form uses only `while`, `let`, integer arithmetic, and indexing, none of
which allocate on the GC heap.

---

## 7. Parity Gates for LLVM Expansion

### 7.1 Purpose

Before Stage 3 begins LLVM codegen, the MIR interpreter must pass
**every existing test and benchmark** to prove the IR pipeline is
semantically correct. These are non-negotiable gates.

### 7.2 Gate Definitions

| Gate ID | Description | Pass Criteria |
|---------|-------------|---------------|
| **G-1** | Unit test parity | All 138 existing tests pass when executed through the new pipeline (Lexer→Parser→HIR→MIR→MIR-Exec) |
| **G-2** | Demo parity | All 3 demos (`demo1_matmul.cjc`, `demo2_gradient.cjc`, `demo3_pipeline.cjc`) produce identical output |
| **G-3** | Matmul benchmark parity | `bench/matmul_bench.cjc` produces correct results; timing within 2x of Stage 1 tree-walk |
| **G-4** | Memory pressure parity | `bench/memory_pressure.cjc` produces correct results; GC isolation behavior preserved |
| **G-5** | Closure tests | New test suite: 20+ tests covering capture, nested closures, closures in nogc blocks |
| **G-6** | Match tests | New test suite: 15+ tests covering all MVP pattern kinds, guards, nested patterns |
| **G-7** | For-loop tests | New test suite: 10+ tests covering range iteration, array iteration, nested loops |
| **G-8** | Nogc verifier | MIR-level verification rejects GC allocations inside nogc blocks; test suite with positive and negative cases |
| **G-9** | AD benchmark parity | `bench/ad_bench` (Rust binary) still compiles and produces correct results (AD engine is unchanged) |
| **G-10** | Optimizer soundness | Constant-folded and dead-code-eliminated MIR produces identical results to unoptimized MIR for all tests |

### 7.3 Gate Enforcement

Gates are checked by a CI-like script:

```bash
# bench/run_parity_gates.sh
#!/bin/bash
set -e

echo "=== PARITY GATE CHECK ==="

echo "[G-1] Unit tests..."
cargo test --workspace

echo "[G-2] Demo parity..."
cargo run -p cjc-cli -- run demos/demo1_matmul.cjc | diff - docs/expected/demo1.txt
cargo run -p cjc-cli -- run demos/demo2_gradient.cjc | diff - docs/expected/demo2.txt
cargo run -p cjc-cli -- run demos/demo3_pipeline.cjc | diff - docs/expected/demo3.txt

echo "[G-3] Matmul benchmark..."
cargo run -p cjc-cli -- run bench/matmul_bench.cjc --time

echo "[G-5-7] New feature tests..."
cargo test -p cjc-hir
cargo test -p cjc-mir
cargo test -p cjc-mir-exec

echo "[G-9] AD benchmark..."
cargo run -p ad-bench

echo "=== ALL GATES PASSED ==="
```

### 7.4 Definition of "Pass"

- **G-1 through G-4**: Byte-identical output (excluding timing values).
- **G-5 through G-8**: All new tests in `#[test]` pass (`cargo test`).
- **G-9**: Compiles and runs without error; correctness checks pass.
- **G-10**: MIR executor output matches for both optimized and unoptimized paths.

---

## 8. Milestone Plan 2.0 → 2.5

### 8.1 Milestone Overview

```
  2.0       2.1        2.2        2.3        2.4         2.5
   │         │          │          │          │           │
   ▼         ▼          ▼          ▼          ▼           ▼
 HIR +    Closures   Match +    For-loop   Optimizer   PARITY
 MIR      in MIR     Patterns   + Desugar  + NoGC      GATE
 Skeleton                                   Verifier    SIGN-OFF
```

### 8.2 Milestone Details

---

#### Milestone 2.0 — IR Skeleton + MIR Executor MVP

**Goal:** Replace tree-walk interpreter with HIR→MIR→MIR-Exec pipeline
for the *existing* language (no new features).

**Deliverables:**
- [ ] `cjc-hir` crate: HIR data structures + AST→HIR lowering for all
  existing AST nodes (no match, no for, no closures yet)
- [ ] `cjc-mir` crate: MIR data structures + HIR→MIR lowering for
  straight-line code, if/else, while, function calls
- [ ] `cjc-mir-exec` crate: MIR reference interpreter that evaluates
  MIR functions using a value stack and scope map
- [ ] `cjc-cli` updated to use new pipeline (with `--legacy` flag to
  keep tree-walk path for comparison)
- [ ] **Gate G-1 passes** (all 138 tests via new pipeline)
- [ ] **Gate G-2 passes** (all 3 demos produce identical output)

**Estimated effort:** ~2,500 lines across 3 new crates

**Definition of Done:** `cargo test --workspace` passes. All demos
produce identical output on both `--legacy` and new pipeline.

---

#### Milestone 2.1 — Closures

**Goal:** Full closure support with capture analysis.

**Deliverables:**
- [ ] Lexer: `Pipe` token for single `|`
- [ ] Parser: Lambda expression parsing (`|params| body`)
- [ ] HIR: `HirExprKind::Closure` with capture list
- [ ] Closure conversion in HIR→MIR: lambda-lifting with env struct
- [ ] MIR executor: `MirValue::Closure` variant + closure dispatch
- [ ] Capture analysis: `Ref` and `Clone` modes, mutable-capture error
- [ ] **Gate G-5 passes** (20+ closure tests)

**Estimated effort:** ~800 lines

**Definition of Done:** Gates G-1, G-2, G-5 pass. Higher-order
functions work:
```cjc
fn apply(f: fn(f64) -> f64, x: f64) -> f64 { f(x) }
let sq = |x: f64| x * x
print(apply(sq, 5.0))   // → 25.0
```

---

#### Milestone 2.2 — Pattern Matching

**Goal:** `match` expressions with all MVP patterns.

**Deliverables:**
- [ ] Lexer: `Match` keyword
- [ ] Parser: `parse_match_expr()` with all MVP pattern forms
- [ ] AST: `ExprKind::Match`, `Pattern` enum
- [ ] HIR: `HirExprKind::Match` preserved
- [ ] MIR: Decision tree compilation (pattern → Branch/Switch)
- [ ] **Gate G-6 passes** (15+ match tests)

**Estimated effort:** ~600 lines

**Definition of Done:** Gates G-1, G-2, G-5, G-6 pass.
```cjc
match value {
    0 => print("zero"),
    1 => print("one"),
    x => print(x),
}
```

---

#### Milestone 2.3 — For-Loops

**Goal:** Range and iterator for-loops.

**Deliverables:**
- [ ] Lexer: `DotDot` token, `In` keyword
- [ ] Parser: `parse_for_stmt()` with range and expression forms
- [ ] AST: `StmtKind::For` with `ForIter`
- [ ] HIR: Desugaring to `while` (see §6.4)
- [ ] **Gate G-7 passes** (10+ for-loop tests)

**Estimated effort:** ~400 lines

**Definition of Done:** Gates G-1, G-2, G-5, G-6, G-7 pass.
```cjc
for i in 0..5 {
    print(i)
}
```

---

#### Milestone 2.4 — Optimizer + NoGC Verifier

**Goal:** MIR-level optimization and nogc enforcement.

**Deliverables:**
- [ ] Constant folding pass: fold `Const op Const → Const` in MIR
- [ ] Dead code elimination: remove unreachable blocks, unused temps
- [ ] NoGC verifier: walk MIR and reject `GcAlloc` inside `is_nogc`
  functions/blocks
- [ ] **Gate G-8 passes** (nogc positive + negative tests)
- [ ] **Gate G-10 passes** (optimized = unoptimized output)

**Estimated effort:** ~500 lines

**Definition of Done:** Gates G-1 through G-8, G-10 pass.

---

#### Milestone 2.5 — Parity Gate Sign-Off

**Goal:** Every single gate passes. Stage 2 is complete.

**Deliverables:**
- [ ] `bench/run_parity_gates.sh` script runs clean
- [ ] All 10 gates (G-1 through G-10) documented as passing
- [ ] Performance comparison: MIR-exec vs tree-walk (document, don't
  necessarily beat — MIR correctness is the goal)
- [ ] `docs/STAGE2_SIGNOFF.md` with gate results, line counts, test counts
- [ ] Archive Stage 1 tree-walk interpreter (keep `--legacy` flag but
  mark it as deprecated)

**Definition of Done:** Zero gate failures. `docs/STAGE2_SIGNOFF.md`
written. Ready to begin Stage 3 (LLVM codegen).

---

### 8.3 Total Estimated Effort

| Milestone | New Lines | New Tests | Cumulative Gates |
|-----------|-----------|-----------|------------------|
| 2.0 | ~2,500 | ~0 (reuse 138) | G-1, G-2 |
| 2.1 | ~800 | ~20 | G-1, G-2, G-5 |
| 2.2 | ~600 | ~15 | G-1, G-2, G-5, G-6 |
| 2.3 | ~400 | ~10 | G-1, G-2, G-5, G-6, G-7 |
| 2.4 | ~500 | ~15 | G-1 through G-8, G-10 |
| 2.5 | ~100 | ~0 | ALL (G-1 through G-10) |
| **Total** | **~4,900** | **~60 new** | **10/10 gates** |

Final Stage 2 target: **~13,000 lines** total (8,000 existing + 4,900 new),
**~200 tests** (138 existing + 60 new).

---

## Appendix A: New Tokens Summary

| Token | Lexeme | Used By |
|-------|--------|---------|
| `Pipe` | `\|` | Closure parameters |
| `Match` | `match` | Match expressions |
| `In` | `in` | For-loops |
| `DotDot` | `..` | Range expressions |

Note: `For`, `FatArrow` (`=>`), and `As` already exist.

## Appendix B: New AST Nodes Summary

| Node | Location | Used By |
|------|----------|---------|
| `ExprKind::Match { scrutinee, arms }` | cjc-ast | Match expressions |
| `Pattern` enum (Wildcard, Binding, Literal, Tuple, Struct) | cjc-ast | Match arms |
| `MatchArm { pattern, guard, body }` | cjc-ast | Match arms |
| `StmtKind::For { var, iter, body }` | cjc-ast | For-loops |
| `ForIter` enum (Range, Expr) | cjc-ast | For-loop iteration source |

Note: `ExprKind::Lambda` already exists.

## Appendix C: New Crates Summary

| Crate | Purpose | Depends On |
|-------|---------|------------|
| `cjc-hir` | HIR structures + AST→HIR lowering | cjc-ast, cjc-types |
| `cjc-mir` | MIR structures + HIR→MIR lowering + optimizer | cjc-hir, cjc-types |
| `cjc-mir-exec` | MIR reference interpreter | cjc-mir, cjc-runtime, cjc-dispatch |

---

*End of Stage 2 Specification*
