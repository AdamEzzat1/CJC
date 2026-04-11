> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [REBRAND_NOTICE.md](REBRAND_NOTICE.md) for the full mapping.

# CJC v1 MVP Specification

## 1. Thesis Statement

CJC proves that a single language can deliver **deterministic systems-grade numeric performance** (Layer 1), **flexible polymorphism via multiple dispatch** (Layer 2), and **ergonomic high-level data/ML abstractions** (Layer 3), with a clean, enforced boundary between GC-managed graphs and no-GC numeric buffers — all while providing reproducible execution and Swift-quality error messages.

The v1 MVP proves this thesis with a minimal but complete vertical slice.

---

## 2. v1 MVP Scope

### MUST (Ship-blocking)

| # | Feature | Proves |
|---|---------|--------|
| M1 | Lexer + recursive-descent parser for CJC subset | Syntax is real, parseable, produces AST |
| M2 | Core type system: `i32`, `i64`, `f32`, `f64`, `bool`, `String` | Numeric foundation |
| M3 | `struct` (value type) + `class` (GC reference type) | Value/ref boundary works |
| M4 | `Tensor<T>` as value struct backed by `Buffer<T>` | First-class tensors exist |
| M5 | Numeric traits: `Numeric`, `Float`, `Differentiable` | Trait-constrained generics |
| M6 | Lightweight shape annotations: `Tensor<f32, [N, M]>` | Dimension mismatch caught at compile time |
| M7 | Multiple dispatch with static specialization | Dispatch tier 1 works |
| M8 | `nogc` function annotation + static enforcement | Layer 1/3 boundary enforced |
| M9 | Forward-mode AD via `Dual<T>` | AD works, validates against finite diff |
| M10 | Reverse-mode AD via `GradNode` graph | Backprop works on tiny models |
| M11 | Data DSL: `filter`, `group_by`, `summarize` on DataFrame | Expression trees build + lower |
| M12 | Logical plan → optimized plan → column-kernel execution | Lowering pipeline works |
| M13 | `--reproducible` mode: deterministic RNG + stable reductions | Reproducibility is real |
| M14 | Error messages with source spans, hints, and fix suggestions | Diagnostic quality is high |
| M15 | Automated test suite with golden-file tests | Everything verified |

### SHOULD (v1.1)

| # | Feature |
|---|---------|
| S1 | Runtime dispatch (tier 2) with `dynamic` keyword |
| S2 | Sealed method tables |
| S3 | Sparse matrix types |
| S4 | Units-of-measure (opt-in) |
| S5 | BLAS/LAPACK FFI bindings |
| S6 | GPU device awareness (CPU-only kernels in v1) |
| S7 | Distribution types (Normal, Bernoulli, etc.) |
| S8 | Variational inference / HMC |

### LATER (v2+)

| # | Feature |
|---|---------|
| L1 | LLVM/MLIR code generation |
| L2 | Full GPU kernel compilation |
| L3 | Package manager + lockfiles |
| L4 | Language server protocol |
| L5 | Mixed-precision training |
| L6 | Probabilistic programming DSL |

### Trade-offs Taken

| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| Interpreted/tree-walk execution in v1 | LLVM codegen | Proves semantics without compiler backend complexity. Codegen is additive later. |
| Recursive-descent parser | Parser generator (LALR/PEG) | Better error messages, easier error recovery, full control. |
| Tracing GC for Layer 3 | Reference counting | Handles cycles in AD graphs naturally. RC would need weak refs for GradNode cycles. |
| Monomorphization for static dispatch | Dictionary passing | Predictable performance. Cost: binary size (irrelevant for interpreted v1). |
| Shape annotations are optional | Mandatory shapes | 80/20 — catch errors when annotated, don't force annotation everywhere. |

---

## 3. Minimal Syntax Subset

### 3.1 Grammar (EBNF — Core Subset)

```ebnf
program        = { declaration } ;
declaration    = struct_decl | class_decl | fn_decl | trait_decl | let_stmt ;

struct_decl    = "struct" IDENT [ type_params ] "{" { field_decl } "}" ;
class_decl     = "class" IDENT [ type_params ] "{" { field_decl } "}" ;
field_decl     = IDENT ":" type_expr [ "=" expr ] ;

trait_decl     = "trait" IDENT [ type_params ] [ ":" trait_bounds ]
                 "{" { fn_sig } "}" ;
trait_bounds   = type_expr { "+" type_expr } ;

fn_decl        = [ "nogc" ] "fn" IDENT [ type_params ] "(" [ params ] ")"
                 [ "->" type_expr ] block ;
fn_sig         = "fn" IDENT [ type_params ] "(" [ params ] ")"
                 [ "->" type_expr ] ";" ;
params         = param { "," param } ;
param          = IDENT ":" type_expr ;

type_params    = "<" type_param { "," type_param } ">" ;
type_param     = IDENT [ ":" trait_bounds ] ;
type_expr      = IDENT [ "<" type_arg { "," type_arg } ">" ]
               | "[" type_expr ";" expr "]"          (* fixed array *)
               | "(" type_expr { "," type_expr } ")" (* tuple *)
               | "fn" "(" [ type_expr { "," type_expr } ] ")" "->" type_expr ;
type_arg       = type_expr | expr ;  (* expr for shape params *)

block          = "{" { statement } [ expr ] "}" ;
statement      = let_stmt | expr_stmt | return_stmt | if_stmt | while_stmt ;
let_stmt       = "let" [ "mut" ] IDENT [ ":" type_expr ] "=" expr ";" ;
expr_stmt      = expr ";" ;
return_stmt    = "return" [ expr ] ";" ;
if_stmt        = "if" expr block [ "else" ( if_stmt | block ) ] ;
while_stmt     = "while" expr block ;

expr           = assignment ;
assignment     = pipe_expr [ "=" expr ] ;
pipe_expr      = logical_or { "|>" call_expr } ;
logical_or     = logical_and { "||" logical_and } ;
logical_and    = equality { "&&" equality } ;
equality       = comparison { ( "==" | "!=" ) comparison } ;
comparison     = addition { ( "<" | ">" | "<=" | ">=" ) addition } ;
addition       = multiplication { ( "+" | "-" ) multiplication } ;
multiplication = unary { ( "*" | "/" | "%" ) unary } ;
unary          = ( "-" | "!" ) unary | postfix ;
postfix        = primary { "." IDENT [ call_args ] | "[" expr "]" | call_args } ;
call_args      = "(" [ arg { "," arg } ] ")" ;
arg            = [ IDENT "=" ] expr ;
primary        = INT_LIT | FLOAT_LIT | STRING_LIT | BOOL_LIT
               | IDENT | "(" expr ")" | block
               | "col" "(" STRING_LIT ")" ;   (* Data DSL column ref *)

BOOL_LIT       = "true" | "false" ;
```

### 3.2 Keywords (v1)

```
struct  class  fn  trait  let  mut  return  if  else  while
nogc  true  false  impl  for  col  import  as
```

### 3.3 Code Examples

#### Example 1: Tensor matmul with shape checking + nogc

```cjc
struct Tensor<T: Float, Shape> {
    buffer: Buffer<T>
    shape: Shape
    strides: [i64; 2]
}

// nogc enforces: no GC allocation inside this function body
nogc fn matmul<T: Float>(a: Tensor<T, [M, K]>, b: Tensor<T, [K, N]>) -> Tensor<T, [M, N]> {
    let out = Buffer.alloc<T>(M * N);
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < N {
            let mut sum = T.zero();
            let mut k = 0;
            while k < K {
                sum = sum + a[i, k] * b[k, j];
                k = k + 1;
            }
            out[i * N + j] = sum;
            j = j + 1;
        }
        i = i + 1;
    }
    Tensor { buffer: out, shape: [M, N], strides: [N, 1] }
}

fn main() {
    let a = Tensor.zeros<f64>([3, 4]);
    let b = Tensor.zeros<f64>([4, 5]);
    let c = matmul(a, b);  // c : Tensor<f64, [3, 5]> — shape inferred

    // Compile error: shape mismatch
    // let bad = Tensor.zeros<f64>([4, 3]);
    // matmul(a, bad);  // ERROR: expected [K, N] but got [4, 3] where K=4 but first dim ≠ K
}
```

#### Example 2: Forward-mode AD + Data DSL pipeline

```cjc
// Forward-mode AD: Dual numbers are a value type
struct Dual<T: Float> {
    value: T
    deriv: T
}

impl<T: Float> Dual<T> : Numeric {
    fn zero() -> Dual<T> { Dual { value: T.zero(), deriv: T.zero() } }
    fn one()  -> Dual<T> { Dual { value: T.one(),  deriv: T.zero() } }
}

// Arithmetic on duals propagates derivatives
fn add<T: Float>(a: Dual<T>, b: Dual<T>) -> Dual<T> {
    Dual { value: a.value + b.value, deriv: a.deriv + b.deriv }
}

fn mul<T: Float>(a: Dual<T>, b: Dual<T>) -> Dual<T> {
    Dual { value: a.value * b.value, deriv: a.value * b.deriv + a.deriv * b.value }
}

// Data DSL pipeline — builds expression tree, lowered to nogc kernels
fn analyze_data() {
    let df = DataFrame.read_csv("data.csv");

    let result = df
        |> filter(col("age") > 18)
        |> group_by("department")
        |> summarize(
            avg_salary = mean(col("salary")),
            headcount  = count()
        );

    // Bridge to tensor for ML
    let features = result.to_tensor(cols: ["avg_salary", "headcount"]);
}
```

#### Example 3: Reverse-mode AD training loop

```cjc
// GradNode is a GC-managed class (Layer 3)
class GradNode {
    op: Op
    inputs: [GradNode]
    tensor: Tensor<f32>
    grad: Tensor<f32>
}

fn linear(x: GradNode, w: GradNode, b: GradNode) -> GradNode {
    matmul_node(x, w) + b  // operator overloading builds graph
}

fn mse_loss(pred: GradNode, target: Tensor<f32>) -> GradNode {
    let diff = pred - target;
    mean(diff * diff)
}

fn train() {
    let w = GradNode.parameter(Tensor.randn<f32>([4, 1]));
    let b = GradNode.parameter(Tensor.zeros<f32>([1]));

    let x_data = Tensor.randn<f32>([100, 4]);
    let y_data = Tensor.randn<f32>([100, 1]);

    let lr = 0.01;
    let mut epoch = 0;
    while epoch < 100 {
        let x = GradNode.input(x_data);
        let pred = linear(x, w, b);
        let loss = mse_loss(pred, y_data);

        loss.backward();  // reverse-mode AD — graph walk in GC layer
                          // gradient kernels run in nogc

        // SGD update — operates on underlying tensors (nogc)
        nogc {
            w.tensor = w.tensor - lr * w.grad;
            b.tensor = b.tensor - lr * b.grad;
        }
        w.zero_grad();
        b.zero_grad();

        epoch = epoch + 1;
    }
}
```

---

## 4. Runtime Memory Split

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CJC Process                          │
│                                                         │
│  ┌──────────────────────┐  ┌─────────────────────────┐  │
│  │   Buffer Allocator   │  │     Tracing GC Heap     │  │
│  │   (No GC — Layer 1)  │  │     (Layer 3 only)      │  │
│  │                      │  │                         │  │
│  │  ┌────────────────┐  │  │  ┌───────────────────┐  │  │
│  │  │ Buffer<f32>    │  │  │  │ GradNode          │  │  │
│  │  │ Buffer<f64>    │  │  │  │ Expr              │  │  │
│  │  │ Tensor.buffer  │◄─┼──┼──│ Plan              │  │  │
│  │  │ Matrix.buffer  │  │  │  │ Model             │  │  │
│  │  └────────────────┘  │  │  │ DataFrame.schema  │  │  │
│  │                      │  │  └───────────────────┘  │  │
│  │  Allocation:         │  │                         │  │
│  │  - Arena per scope   │  │  GC roots:              │  │
│  │  - Explicit alloc/   │  │  - Stack refs to class  │  │
│  │    free for long-    │  │    instances             │  │
│  │    lived buffers     │  │  - Global model regs     │  │
│  │  - Ref counting for  │  │                         │  │
│  │    shared views      │  │  Collection:             │  │
│  │    (COW on mutate)   │  │  - Mark-sweep            │  │
│  │                      │  │  - Incremental (planned) │  │
│  └──────────────────────┘  └─────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │                   nogc Zones                        ││
│  │  - Static check: no GC-type allocation              ││
│  │  - Runtime check: GC collection paused              ││
│  │  - Only Buffer/Tensor/Matrix types allowed           ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### 4.2 Concrete Types & Their Allocation

| Type | Semantics | Allocated On | Copyable? | Notes |
|------|-----------|-------------|-----------|-------|
| `i32`, `i64`, `f32`, `f64`, `bool` | Value | Stack | Yes (bitwise) | Primitives |
| `Buffer<T>` | Value (handle) | Buffer allocator | Ref-counted; COW on mutation | Underlying memory is heap-allocated, metadata is value |
| `Tensor<T, S>` | Value (view) | Stack (metadata) + Buffer allocator (data) | Yes (shares buffer via refcount) | Copy = new view, mutate = COW |
| `struct` (user) | Value | Stack or inline | Memberwise copy | Deep copy of value fields, refcount bump for buffer fields |
| `class` (user) | Reference | GC heap | No copy; reference shared | GC-managed, tracing collector |
| `GradNode` | Reference | GC heap | No copy | Points to Tensor buffer (cross-boundary ref) |
| `Expr`, `Plan` | Reference | GC heap | No copy | Data DSL intermediate objects |
| `String` | Value (handle) | Buffer allocator | COW | Like Buffer<u8> with UTF-8 invariant |

### 4.3 Cross-Boundary Rules

1. **GC → Buffer**: A GC object may hold a reference (handle) to a Buffer. The Buffer's refcount includes GC-side references. When GC collects the object, it decrements the Buffer refcount.

2. **Buffer → GC**: **Never.** Value types cannot hold references to GC objects. This is enforced by the type system.

3. **nogc zones**: The compiler statically verifies that no expression inside a `nogc fn` or `nogc { ... }` block can allocate a GC type or call a function that isn't also `nogc`. At runtime, GC collection is suspended during nogc execution.

4. **Pinning**: When a GC object holds a Buffer handle, the Buffer is not GC-managed — it's refcounted independently. No pinning needed. The GC only traces GC-heap pointers.

### 4.4 Buffer Allocator Detail

```
Buffer<T> internals:
  ┌──────────────────┐
  │ ptr: *mut T      │  → points to allocation in buffer heap
  │ len: usize       │
  │ capacity: usize  │
  │ refcount: AtomicU32 │
  │ allocator: AllocatorId │  → arena or global
  └──────────────────┘

Arena allocation:
  - Scoped arenas for temporary computation buffers
  - Arena freed in bulk at scope exit
  - No individual deallocation within arena

Global allocation:
  - Long-lived buffers (model parameters, dataset columns)
  - Freed when refcount → 0
```

### 4.5 GC Design (v1)

- **Algorithm**: Simple mark-sweep (v1). Upgrade to incremental mark-sweep in v1.1.
- **Roots**: Stack frames (local variables of class type), global registries.
- **Tracing**: GC objects only point to other GC objects or to Buffer handles (which are NOT traced — refcounted separately).
- **Trigger**: Allocation count threshold (default: 10,000 allocations between collections, tunable).
- **nogc interaction**: GC is paused (no collection) during nogc zone execution. Allocations in nogc are compile-error.

---

## 5. v1 Multiple Dispatch Resolution Rules

### 5.1 Method Definition

Methods are defined with `fn` at module scope. Multiple methods may share a name with different parameter types:

```cjc
fn add(a: f32, b: f32) -> f32 { a + b }
fn add(a: f64, b: f64) -> f64 { a + b }
fn add<T: Numeric>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> { /* element-wise */ }
fn add<T: Numeric>(a: Tensor<T>, b: T) -> Tensor<T> { /* broadcast scalar */ }
```

### 5.2 Dispatch Resolution Algorithm

Given a call `f(arg1, arg2, ..., argN)`:

**Step 1 — Candidate Collection**
Collect all methods named `f` visible in the current scope (local module + imports).

**Step 2 — Applicability Filtering**
A method is **applicable** if each argument type is a subtype of (or satisfies the trait bound of) the corresponding parameter type after type variable substitution.

**Step 3 — Specificity Ordering**
Among applicable methods, rank by specificity:
- A concrete type is more specific than a generic type.
- A constrained generic `<T: Float>` is more specific than an unconstrained `<T>`.
- A more derived trait bound is more specific (`Float` > `Numeric` > `Any`).
- Specificity is computed per-parameter and combined lexicographically (left to right).

**Step 4 — Ambiguity Check**
If two or more methods are equally specific (no single most-specific method), the compiler emits an **ambiguity error** with:
- The call site with source span
- The conflicting method signatures
- A hint suggesting how to disambiguate (add a more specific overload, or use explicit type annotation)

**Step 5 — Static vs. Runtime**
In v1, **all dispatch is static**. The compiler resolves the method at compile time using the statically-known types. If the types are not known statically (e.g., trait object), v1 emits an error suggesting runtime dispatch (`dynamic` keyword, planned for v1.1).

### 5.3 Coherence Rules

1. **No orphan implementations**: You can only define methods for a type in the same module as the type, or in the module that defines the trait.

2. **No overlapping implementations**: Two `impl` blocks for the same trait + type combination are a compile error.

3. **Sealed modules**: A module may be marked `sealed`, which prevents external modules from adding methods to its types. This gives performance guarantees (method table is closed, can be fully monomorphized).

```cjc
sealed module linalg {
    // External code cannot add new methods dispatching on Tensor
    // within this module's method table
}
```

### 5.4 Specificity Example

```cjc
fn process(x: f64)              { /* A */ }
fn process<T: Float>(x: T)     { /* B */ }
fn process<T: Numeric>(x: T)   { /* C */ }

process(3.14)  // Resolves to A (concrete f64 > generic Float > generic Numeric)
process(3.14f32)  // Resolves to B (f32 matches Float, not the f64 overload)

// Ambiguity example:
fn combine<T: Float>(a: T, b: Tensor<T>) { /* D */ }
fn combine<T: Float>(a: Tensor<T>, b: T) { /* E */ }

// combine(tensor, tensor)  — ERROR: ambiguous, D and E both match equally
// Fix: add fn combine<T: Float>(a: Tensor<T>, b: Tensor<T>) { /* F */ }
```

### 5.5 Diagnostic Format for Dispatch Errors

```
error[E0301]: ambiguous method resolution for `combine`
  --> src/main.cjc:42:5
   |
42 |     combine(tensor, tensor)
   |     ^^^^^^^ multiple equally-specific methods match
   |
   = candidates:
     --> src/ops.cjc:10:1  fn combine<T: Float>(a: T, b: Tensor<T>)
     --> src/ops.cjc:14:1  fn combine<T: Float>(a: Tensor<T>, b: T)
   |
   = hint: add a more specific overload:
     fn combine<T: Float>(a: Tensor<T>, b: Tensor<T>) -> ...
```

---

## 6. Milestone Plan

### Milestone 0: Project Skeleton (Foundation)
- Rust project structure (cargo workspace)
- Crate layout: `cjc-lexer`, `cjc-parser`, `cjc-types`, `cjc-dispatch`, `cjc-runtime`, `cjc-eval`, `cjc-ad`, `cjc-data`, `cjc-cli`
- CI pipeline (cargo test + clippy + format check)
- Test harness with golden-file support

### Milestone 1: Parse + Typecheck + Diagnose
- Lexer: all v1 tokens
- Parser: recursive descent for full v1 grammar
- AST pretty-printer
- Type checker: primitives, structs, classes, generics, trait bounds
- Shape annotation checking (basic)
- Error messages with spans + hints
- **Demo: parse, typecheck, and diagnose errors in a CJC program**

### Milestone 2: Dispatch + Runtime + Tensors
- Multiple dispatch resolution (static, v1 rules)
- Coherence checker
- Tree-walk interpreter (eval AST directly)
- Buffer allocator (arena + global)
- Tensor runtime (alloc, index, slice, view)
- Basic ops: add, mul, matmul (naive)
- GC for class instances (mark-sweep)
- `nogc` enforcement (static check)
- **Demo: matmul on Tensors with dispatch, nogc enforcement, GC for graph nodes**

### Milestone 3: AD + Data DSL + Reproducibility
- Forward-mode AD: Dual number type + arithmetic
- Reverse-mode AD: GradNode graph, backward pass, gradient computation
- Data DSL: expression tree builder, pipe operator
- Logical plan: filter, group_by, summarize
- Plan optimizer: predicate pushdown, column pruning
- Column-buffer kernel execution
- Deterministic RNG (SplitMix64 / Xoshiro256**)
- Stable reduction (Kahan summation)
- `--reproducible` flag
- **Demo: end-to-end linear regression training + data pipeline**

---

## 7. Three Flagship Demos

### Demo 1: "Shape-Safe Matmul" (Milestone 2)

Proves: Layer 1 deterministic execution, nogc enforcement, shape checking, dispatch.

```cjc
nogc fn matmul(a: Tensor<f64, [3, 4]>, b: Tensor<f64, [4, 5]>) -> Tensor<f64, [3, 5]> {
    // naive triple loop — runs in no-GC zone
    let out = Buffer.alloc<f64>(3 * 5);
    // ... (loop body)
    Tensor { buffer: out, shape: [3, 5], strides: [5, 1] }
}

fn main() {
    let a = Tensor.randn<f64>([3, 4]);
    let b = Tensor.randn<f64>([4, 5]);
    let c = matmul(a, b);
    print(c.shape);  // [3, 5]

    // This line would produce a compile error:
    // let bad = Tensor.randn<f64>([5, 3]);
    // matmul(a, bad);
    // error[E0201]: shape mismatch in argument `b`
    //   expected Tensor<f64, [4, N]> but got Tensor<f64, [5, 3]>
    //   dimension 0: expected 4, got 5
}
```

**Acceptance criteria:**
- matmul computes correctly (validated against known result)
- `nogc` rejects any attempt to allocate a class inside the function
- Shape mismatch produces a clear error with dimension details
- Dispatch resolves the correct `matmul` overload

### Demo 2: "Gradient Descent on Linear Model" (Milestone 3)

Proves: Reverse-mode AD, GC/nogc boundary, training loop, reproducibility.

```cjc
fn main() {
    // Reproducible execution
    let rng = Rng.seeded(42);

    // Data — value types, buffer-allocated
    let x = Tensor.randn<f32>([100, 4], rng: rng);
    let y_true = Tensor.randn<f32>([100, 1], rng: rng);

    // Parameters — wrapped in GradNode (GC-managed)
    let w = GradNode.parameter(Tensor.randn<f32>([4, 1], rng: rng));
    let b = GradNode.parameter(Tensor.zeros<f32>([1]));

    let lr = 0.01;
    let mut epoch = 0;
    while epoch < 50 {
        let x_node = GradNode.input(x);
        let pred = x_node.matmul(w) + b;
        let loss = mean((pred - y_true) * (pred - y_true));

        loss.backward();

        nogc {
            w.tensor = w.tensor - lr * w.grad;
            b.tensor = b.tensor - lr * b.grad;
        }
        w.zero_grad();
        b.zero_grad();

        if epoch % 10 == 0 {
            print("epoch", epoch, "loss", loss.value());
        }
        epoch = epoch + 1;
    }
}
```

**Acceptance criteria:**
- Loss decreases monotonically (or near-monotonically)
- Gradients validated against finite-difference approximation (max error < 1e-4)
- Running twice with `--reproducible` produces bit-identical loss sequence
- GC collects stale GradNode graphs without touching tensor buffers
- `nogc` block rejects any attempt to create GradNode

### Demo 3: "Data Pipeline to Tensor" (Milestone 3)

Proves: Data DSL, expression trees, plan optimization, lowering to kernels, bridge to tensors.

```cjc
fn main() {
    let df = DataFrame.from_columns(
        name:    ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
        dept:    ["eng",   "eng", "sales", "eng",  "sales", "eng"],
        salary:  [95000.0, 102000.0, 78000.0, 110000.0, 82000.0, 98000.0],
        tenure:  [3, 7, 2, 10, 1, 5]
    );

    let summary = df
        |> filter(col("tenure") > 2)
        |> group_by("dept")
        |> summarize(
            avg_salary = mean(col("salary")),
            max_tenure = max(col("tenure")),
            headcount  = count()
        );

    print(summary);
    // dept   | avg_salary | max_tenure | headcount
    // eng    | 101250.0   | 10         | 4
    // (sales filtered out — only Carol had tenure > 2... wait, no)
    // Actually: Alice(3), Bob(7), Dave(10), Frank(5) from eng pass; Carol(2) fails, Eve(1) fails
    // eng: avg(95000+102000+110000+98000)/4 = 101250, max_tenure=10, count=4

    // Bridge to tensor
    let features = summary.to_tensor(cols: ["avg_salary", "headcount"]);
    print(features.shape);  // [1, 2] — one group (eng), two features
}
```

**Acceptance criteria:**
- Filter correctly excludes rows with tenure <= 2
- Group-by produces correct groups
- Aggregations compute correctly
- Plan optimizer applies column pruning (name column never materialized)
- Predicate pushdown moves filter before group_by in plan
- `to_tensor` produces correctly shaped tensor
- All operations except final print can run in nogc kernels

---

## 8. Definition of Done Checklist

### 8.1 Per-Feature Checklist

Every feature must satisfy ALL of the following before it is considered done:

- [ ] **Spec**: Written specification with examples and edge cases
- [ ] **Implementation**: Code complete and compiling
- [ ] **Unit tests**: ≥3 positive tests + ≥2 negative tests (error cases)
- [ ] **Golden tests**: Expected output captured and checked in
- [ ] **Error messages**: All error paths produce spans + hints
- [ ] **Documentation**: Inline doc comments on public APIs
- [ ] **Integration**: Works with other completed features (no regressions)
- [ ] **Review**: Code reviewed by at least one other "role" (cross-checked)

### 8.2 Per-Milestone Checklist

- [ ] All features in milestone pass per-feature checklist
- [ ] Flagship demo runs end-to-end without panics
- [ ] Flagship demo produces expected output (golden-file verified)
- [ ] No known P0 bugs (crashes, incorrect results, silent failures)
- [ ] Performance sanity check: flagship demo completes in <5s on reference hardware
- [ ] `cargo clippy` clean (no warnings)
- [ ] `cargo fmt` clean
- [ ] All tests pass: `cargo test --workspace`

### 8.3 v1 Release Checklist

- [ ] All 3 milestones pass per-milestone checklist
- [ ] All 3 flagship demos run and produce correct output
- [ ] `--reproducible` mode produces bit-identical output across 10 runs
- [ ] Error message quality: 10 curated error scenarios produce helpful diagnostics
- [ ] AD correctness: all gradients match finite-difference within tolerance (1e-4 for f32, 1e-8 for f64)
- [ ] Data DSL correctness: all queries match hand-computed expected results
- [ ] Dispatch correctness: all specificity/ambiguity test cases resolve correctly
- [ ] `nogc` enforcement: all violation test cases produce compile errors
- [ ] Shape checking: dimension mismatch test cases caught at compile time
- [ ] Memory safety: no leaks detected (custom leak check or valgrind)
- [ ] Fuzz testing: parser survives 10,000 random inputs without panic
- [ ] Cross-validation: tensor ops match NumPy on 5 reference computations
- [ ] README with build instructions, quick-start, and demo walkthrough

### 8.4 Test Categories & Counts (Minimum)

| Category | Min Tests | Oracle |
|----------|----------|--------|
| Lexer | 20 | Expected token streams |
| Parser (valid) | 30 | AST golden files |
| Parser (errors) | 15 | Error message golden files |
| Type checker (valid) | 25 | No errors emitted |
| Type checker (errors) | 20 | Error golden files |
| Shape checking | 10 | Correct shapes + error golden files |
| Dispatch resolution | 15 | Correct method selected |
| Dispatch ambiguity | 5 | Ambiguity error golden files |
| Tensor ops | 15 | NumPy cross-check |
| Forward AD | 10 | Finite-difference validation |
| Reverse AD | 10 | Finite-difference validation |
| Data DSL (parse) | 10 | AST golden files |
| Data DSL (execute) | 10 | Hand-computed results |
| Plan optimizer | 5 | Optimized plan golden files |
| Reproducibility | 5 | Bit-identical across runs |
| nogc enforcement | 10 | Compile error golden files |
| GC correctness | 5 | No leaks, no UAF |
| Integration (demos) | 3 | Full demo output golden files |
| Parser fuzzing | 10,000 inputs | No panics |
| **Total** | **~10,200+** | |

---

## 9. Implementation Language & Structure

### 9.1 Host Language

CJC v1 is implemented in **Rust**. Rationale:
- Memory safety for the compiler/runtime itself
- Excellent performance for tree-walk interpreter
- Strong ecosystem (logos for lexer, ariadne/miette for diagnostics)
- Dogfooding spirit (CJC aspires to Rust-level safety)

### 9.2 Crate Layout

```
cjc/
├── Cargo.toml              (workspace root)
├── crates/
│   ├── cjc-lexer/          Tokenizer
│   ├── cjc-parser/         Recursive descent parser → AST
│   ├── cjc-ast/            AST node definitions (shared)
│   ├── cjc-types/          Type system, trait resolution, shape checking
│   ├── cjc-dispatch/       Multiple dispatch resolution
│   ├── cjc-runtime/        Buffer allocator, GC, tensor runtime
│   ├── cjc-eval/           Tree-walk interpreter
│   ├── cjc-ad/             AD engine (forward + reverse)
│   ├── cjc-data/           Data DSL: expr trees, plan, optimizer, kernels
│   ├── cjc-diag/           Diagnostic formatting (errors, warnings, hints)
│   ├── cjc-repro/          Reproducibility: deterministic RNG, stable reductions
│   └── cjc-cli/            CLI entry point (cjc run, cjc check, cjc fmt)
├── tests/
│   ├── golden/             Golden-file test inputs and expected outputs
│   ├── fuzz/               Fuzz test harnesses
│   └── integration/        End-to-end demo tests
├── demos/
│   ├── demo1_matmul.cjc
│   ├── demo2_gradient.cjc
│   └── demo3_pipeline.cjc
└── docs/
    ├── CJC_v1_MVP_SPEC.md  (this document)
    ├── GRAMMAR.md
    └── ERRORS.md
```

---

## 10. Open Questions (To Resolve During Implementation)

| # | Question | Options | Leaning | Resolve By |
|---|----------|---------|---------|------------|
| Q1 | Should shape params be full dependent types or symbolic constants? | Dependent / Symbolic / Erased | Symbolic constants (like Zig's comptime) — avoids dependent type complexity | Milestone 1 |
| Q2 | Should `String` be a value type with COW or GC-managed? | Value+COW / GC | Value+COW (consistent with Buffer model) | Milestone 1 |
| Q3 | Should `impl` blocks be attached to types or free-standing? | Attached / Free | Both: `impl Type { ... }` for methods, free `fn` for dispatch | Milestone 1 |
| Q4 | What GC trigger heuristic for v1? | Alloc count / Byte threshold / Generational | Alloc count (simplest, tune later) | Milestone 2 |
| Q5 | Should reverse-mode AD use a persistent tape or per-forward graph? | Tape / Graph | Per-forward graph (cleaner, no tape lifetime issues) | Milestone 3 |

---

*This document is the single source of truth for CJC v1. All implementation work derives from it. Changes require explicit versioning and justification.*
