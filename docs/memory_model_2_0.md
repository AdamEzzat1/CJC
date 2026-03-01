# CJC Memory Model 2.0

**Priority order**: Determinism > Memory efficiency > Latency > Speed

## Design Goals

| Goal | Description | Status |
|------|-------------|--------|
| G0 | Arena first, RC last | Implemented |
| G1 | Deterministic alloc/dealloc (no stop-the-world) | Implemented |
| G2 | Policy A: strong cycles forbidden, Weak breaks cycles | Implemented |
| G3 | @no_gc as real contract (escape analysis enforcement) | Implemented |
| G4 | Concurrency determinism (deterministic layout) | Implemented |

## Architecture

### Layer 1: ObjectSlab (replaces mark-sweep GC)

**File**: `crates/cjc-runtime/src/object_slab.rs`

The mark-sweep GC has been replaced with a deterministic RC-backed slab allocator:

- `ObjectSlab`: type-erased slab with `Rc<RefCell<Box<dyn Any>>>` entries
- `SlabRef`: lightweight copyable index handle
- LIFO free list for deterministic slot reuse
- `collect_noop()`: no-op method (no stop-the-world pauses)
- `GcHeap`: backward-compatibility wrapper delegating to ObjectSlab

### Layer 2: Escape Analysis

**File**: `crates/cjc-mir/src/escape.rs`

Intraprocedural MIR annotation pass that classifies every let-binding:

- `AllocHint::Stack` -- primitive types (Int, Float, Bool, U8, Void)
- `AllocHint::Arena` -- non-escaping heap values (eligible for frame-arena)
- `AllocHint::Rc` -- escaping values (returned, captured, stored in containers)

**Escape reasons tracked**: `ReturnedFromFn`, `CapturedByClosure`, `StoredInContainer`, `PassedToUnknownFn`, `AssignedToFieldOrIndex`, `Mutable`, `CallResult`

The `alloc_hint` field is added to `MirStmt::Let` for downstream use by executors.

### Layer 3: FrameArena

**File**: `crates/cjc-runtime/src/frame_arena.rs`

Bump allocator for non-escaping values (AllocHint::Arena):

- Sequential allocation within 4 KB pages
- 8-byte alignment for safe casting
- `reset()` returns cursor to start (bulk-free)
- Pages are never returned to OS (retained for reuse)
- `ArenaStore`: type-erased entry list backed by FrameArena

### Layer 4: BinnedAllocator

**File**: `crates/cjc-runtime/src/binned_alloc.rs`

Size-class allocator with 13 bins (16 B -- 64 KB):

| Bin | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-----|---|---|---|---|---|---|---|---|---|---|----|----|-----|
| Size | 16B | 32B | 48B | 64B | 128B | 256B | 512B | 1KB | 2KB | 4KB | 8KB | 16KB | 64KB |

- LIFO free lists per bin (deterministic reuse order)
- Single contiguous backing `Vec<u8>` (never shrinks)
- Overflow list for allocations > 64 KB
- Double-free protection

### Layer 5: TiledMatmul

**File**: `crates/cjc-runtime/src/tensor_tiled.rs`

L2-friendly tiled matrix multiplication:

- Configurable tile size (default 64x64 = 32 KB at f64)
- Row-major tile iteration order (deterministic)
- `matmul(a, m, k, b, n)`: standard A*B
- `matmul_transposed_b(a, m, k, b, n)`: A*B^T (cache-friendly)
- Verified against naive O(n^3) implementation

## Cycle Policy: Policy A

Strong reference cycles are **forbidden** at the language level:

- All heap references use `Rc` (reference counting)
- `Weak<T>` available for explicit back-references
- No mark-sweep collector exists -- cycles would leak
- The `@no_gc` annotation verified by escape analysis ensures no heap allocation in critical sections

## @no_gc Contract

The `@no_gc` annotation is enforced at two levels:

1. **NoGC verifier** (`nogc_verify.rs`): rejects calls to `gc_alloc`, `gc_collect`, and transitively may-gc functions
2. **Escape analysis** (`escape.rs`): `has_heap_alloc()` detects any Arena or Rc bindings in @no_gc functions

## Determinism Guarantees

1. **Same alloc sequence = same slot indices** (ObjectSlab, BinnedAllocator)
2. **Same arena alloc sequence = same (page, offset) layout** (FrameArena)
3. **No OS memory return during execution** (all allocators)
4. **No stop-the-world pauses** (collect is a no-op)
5. **Deterministic tiled matmul** (fixed tile iteration order)
6. **Deterministic hash iteration** (DetMap uses insertion order)

## Test Coverage

| Component | Unit Tests | Integration Tests |
|-----------|-----------|-------------------|
| ObjectSlab | 6 | 3 (G0) |
| GcHeap compat | 6 | 1 (G0) |
| Escape analysis | 13 | 3 (G3) |
| FrameArena | 7 | 1 (G1) |
| ArenaStore | 5 | 1 |
| BinnedAllocator | 11 | 2 (G1, G4) |
| TiledMatmul | 8 | 2 (G4) |
| **Total** | **56** | **16** |

Full workspace: **2,273 tests, 0 failures**

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `crates/cjc-runtime/src/object_slab.rs` | ~200 | RC-backed slab (replaces GC) |
| `crates/cjc-runtime/src/gc.rs` | ~150 | Backward-compat wrapper |
| `crates/cjc-runtime/src/frame_arena.rs` | ~290 | Bump arena + ArenaStore |
| `crates/cjc-runtime/src/binned_alloc.rs` | ~310 | Size-class allocator |
| `crates/cjc-runtime/src/tensor_tiled.rs` | ~250 | Tiled matmul |
| `crates/cjc-mir/src/escape.rs` | ~570 | Escape analysis pass |
| `tests/memory_model/mod.rs` | ~270 | Integration test suite |
| `tests/test_memory_model.rs` | 2 | Test harness |

## Known Limitations

1. **Escape analysis is intraprocedural**: does not track values across function boundaries (conservative: unknown = Rc)
2. **Mutable bindings are conservative**: marked as Rc even if they don't actually escape
3. **Arena values still use Rc internally**: for compatibility with the Value type system; true arena-backed inline storage is a future optimization
4. **Tiled matmul not yet integrated into Tensor::matmul**: available as a standalone engine
5. **MIR executor lacks tidy dispatch**: tidy tests skip MIR-exec parity check
