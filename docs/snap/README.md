# CJC Snap — Content-Addressable Serialization

CJC Snap is a first-class serialization system built into the CJC language. It provides deterministic binary encoding with SHA-256 content hashing, file persistence for checkpointing, memoization, and cross-language JSON export — all with zero external dependencies.

**Format versions:** v1 (tags 0x00–0x11) and v2 (tags 0x00–0x17, adds typed tensors, chunked tensors, sparse matrices, DataFrames, and categorical encoding). Both formats are fully supported with automatic detection on load.

## Quick Start

```cjc
// Snap a value — returns a SnapBlob struct
let blob = snap(42)
print(blob.hash)   // 64-char SHA-256 hex string
print(blob.size)   // byte count of the encoding

// Restore the original value
let original = restore(blob)
print(original)    // 42

// Content-addressable: same value → same hash
let h1 = snap_hash(42)
let h2 = snap_hash(42)
assert(h1 == h2)   // always true

// Save to disk and load back
snap_save(42, "checkpoint.snap")
let loaded = snap_load("checkpoint.snap")
print(loaded)  // 42

// Export to JSON for Python consumption
let json = snap_to_json(42)
print(json)  // "42"

// Memoize expensive function calls
fn fib(n: i64) -> i64 {
    if n <= 1 { return n }
    return memo_call("fib", n - 1) + memo_call("fib", n - 2)
}
```

## Builtin Reference

CJC Snap provides 9 builtins, all available from both the AST evaluator and MIR executor:

### `snap(value) -> SnapBlob`

Encodes a value into a content-addressable binary blob.

- **Input:** Any snap-encodable value (see Supported Types below)
- **Returns:** A `SnapBlob` struct with three fields:
  - `hash` (String) — 64-character lowercase SHA-256 hex digest
  - `data` (Bytes) — canonical binary encoding
  - `size` (Int) — byte count of the encoding
- **Effect:** `alloc` (allocates the blob struct)

```cjc
let blob = snap([1, 2, 3])
print(blob.hash)  // "a1b2c3..." (64-char hex)
print(blob.size)  // 28
```

### `restore(blob) -> Value`

Decodes a SnapBlob back into the original value, verifying SHA-256 integrity.

- **Input:** A `SnapBlob` struct (as returned by `snap()`)
- **Returns:** The decoded value
- **Errors:** If the hash doesn't match the data (tampered or corrupted blob)
- **Effect:** `alloc`

```cjc
let blob = snap("hello")
let original = restore(blob)
print(original)  // "hello"
```

### `snap_hash(value) -> String`

Returns only the SHA-256 content hash of a value (without the full blob).

- **Input:** Any snap-encodable value
- **Returns:** 64-character lowercase hex SHA-256 hash
- **Effect:** `alloc`
- **Key property:** Deterministic — same logical value always produces the same hash, regardless of HashMap iteration order or memory layout

```cjc
let h1 = snap_hash(42)
let h2 = snap_hash(42)
assert(h1 == h2)  // content-addressable identity
```

### `snap_save(value, path)`

Saves a value to a `.snap` file on disk.

- **Input:** A snap-encodable value and a file path string
- **Returns:** Void
- **Effect:** `io` (writes to filesystem)
- **File format:** 48-byte header (magic + version + SHA-256 + data length) followed by the canonical encoding. See [WIRE_FORMAT.md](WIRE_FORMAT.md) for details.

```cjc
let weights = tensor_randn([784, 256])
snap_save(weights, "model_weights.snap")
```

### `snap_save_v2(value, path)`

Saves a value to a `.snap` file using the v2 format.

- **Input:** A snap-encodable value and a file path string
- **Returns:** Void
- **Effect:** `io` (writes to filesystem)
- **v2 features:** Supports all v2 tags including typed tensors, chunked tensors, sparse matrices, DataFrames, and categorical data. Uses the v2 payload header (`CJS\x01` + version + flags).

```cjc
let weights = tensor_randn([784, 256])
snap_save_v2(weights, "model_weights_v2.snap")
```

### `snap_load(path) -> Value`

Loads a value from a `.snap` file, validating magic bytes, version, and SHA-256 integrity.

- **Input:** File path string
- **Returns:** The decoded value
- **Errors:** Invalid magic bytes, unsupported version, truncated file, hash mismatch, missing file
- **Effect:** `io_alloc` (reads from filesystem + allocates)
- **Auto-detects** v1 and v2 file formats transparently

```cjc
// Works with both v1 and v2 .snap files
let weights = snap_load("model_weights.snap")
let weights_v2 = snap_load("model_weights_v2.snap")
```

### `snap_to_json(value) -> String`

Converts a value to a JSON string for cross-language interop.

- **Input:** Any snap-encodable value
- **Returns:** JSON string
- **Effect:** `alloc`
- **Non-JSON-native types** use `__type` discriminators (see JSON Format section below)

```cjc
let json = snap_to_json([1, 2, 3])
print(json)  // "[1,2,3]"
```

### `memo_call(fn_name, args...) -> Value`

Calls a function with memoization. If the same function+args combination has been seen before (determined by snap-hashing), returns the cached result without re-executing.

- **Input:** Function name (String) + any number of arguments
- **Returns:** The function's return value (cached or fresh)
- **Effect:** `alloc`
- **Cache key:** SHA-256 of the snap-encoded tuple `(fn_name, arg1, arg2, ...)`
- **Cache scope:** Per-executor instance (not persisted across runs)

```cjc
fn expensive_compute(x: i64) -> i64 {
    print("computing...")
    return x * x
}

let a = memo_call("expensive_compute", 5)  // prints "computing..."
let b = memo_call("expensive_compute", 5)  // cache hit — no print
assert(a == b)  // 25
```

## Supported Types

All CJC data types that represent pure values can be snap-encoded:

### Core Types (v1 + v2)

| Type | Tag | JSON Format |
|------|-----|-------------|
| `Void` | `0x00` | `null` |
| `Int` | `0x01` | `42` |
| `Float` | `0x02` | `3.14` (NaN → `"NaN"`, Inf → `"Infinity"`) |
| `Bool` | `0x03` | `true` / `false` |
| `String` | `0x04` | `"hello"` |
| `U8` | `0x0D` | `255` |
| `Bytes` | `0x0A` | `{"__type":"Bytes","hex":"deadbeef"}` |
| `ByteSlice` | `0x0B` | `{"__type":"Bytes","hex":"..."}` |
| `StrView` | `0x0C` | `"..."` (as JSON string) |
| `Array` | `0x05` | `[1,2,3]` |
| `Tuple` | `0x06` | `[1,2,3]` (same as Array in JSON) |
| `Struct` | `0x07` | `{"__type":"Struct","name":"...","fields":{...}}` |
| `Tensor` | `0x08` | `{"__type":"Tensor","shape":[...],"data":[...]}` |
| `Enum` | `0x09` | `{"__type":"Enum","enum":"...","variant":"...","fields":[...]}` |
| `Complex` | `0x10` | `{"__type":"Complex","re":...,"im":...}` |
| `Bf16` | `0x0E` | `{"__type":"Bf16","value":...}` |
| `F16` | `0x0F` | `{"__type":"F16","value":...}` |
| `Map` | `0x11` | `{"__type":"Map","entries":[{"key":...,"value":...},...]}` |

### Extended Types (v2 only)

| Type | Tag | Description |
|------|-----|-------------|
| `TypedTensor` | `0x12` | Multi-dtype tensor (F64, F32, I64, I32, U8, Bool, Bf16, F16, Complex) |
| `ChunkedTensor` | `0x13` | Large tensor with 4 MB chunks + per-chunk SHA-256 integrity |
| `SparseTensor` | `0x14` | Sparse CSR matrix (nrows, ncols, row_offsets, col_indices, values) |
| `Categorical` | `0x15` | Factor-encoded column (levels + codes) |
| `Schema` | `0x16` | Field metadata (name + type_tag pairs) |
| `DataFrame` | `0x17` | Columnar table (Int, Float, Str, Bool, Categorical, DateTime columns) |

**Not snap-encodable** (runtime-only): `Fn`, `Closure`, `ClassRef`, `Scratchpad`, `GradGraph`, `OptimizerState`, `PagedKvCache`, `AlignedBytes`, `TidyView`, `GroupedTidyView`, `Regex`.

Use `is_snappable()` (Rust API) to check at runtime.

## CJC Examples

### Checkpointing a Training Loop

```cjc
struct Checkpoint {
    epoch: i64,
    weights: Any,
    loss: f64,
}

fn train(data: Any) -> Any {
    let w = tensor_randn([784, 256])
    let i = 0
    while i < 100 {
        // ... training step ...
        let loss = 0.5
        if i % 10 == 0 {
            let cp = Checkpoint { epoch: i, weights: w, loss: loss }
            snap_save(cp, "checkpoint_" + to_string(i) + ".snap")
            print("Saved checkpoint at epoch " + to_string(i))
        }
        i = i + 1
    }
    return w
}

// Resume from checkpoint
let cp = snap_load("checkpoint_50.snap")
print("Resuming from epoch " + to_string(cp.epoch))
```

### Content-Addressable Data Deduplication

```cjc
fn deduplicate(items: Any) -> Any {
    let seen: Any = []
    let unique: Any = []
    let i = 0
    while i < len(items) {
        let h = snap_hash(items[i])
        if !array_contains(seen, h) {
            seen = array_push(seen, h)
            unique = array_push(unique, items[i])
        }
        i = i + 1
    }
    return unique
}
```

### Recursive Memoization

```cjc
fn fib(n: i64) -> i64 {
    if n <= 1 { return n }
    return memo_call("fib", n - 1) + memo_call("fib", n - 2)
}

print(fib(30))  // fast — O(n) instead of O(2^n)
```

## Python Interop

CJC Snap provides two paths for Python consumption:

### Path 1: JSON Export (Recommended)

Use `snap_to_json()` to export CJC values as JSON, then `json.loads()` in Python:

```cjc
// CJC side
let weights = tensor_randn([784, 256])
let json = snap_to_json(weights)
file_write("weights.json", json)
```

```python
# Python side
import json
import numpy as np

with open("weights.json") as f:
    data = json.load(f)

# Reconstruct tensor
assert data["__type"] == "Tensor"
arr = np.array(data["data"]).reshape(data["shape"])
print(arr.shape)  # (784, 256)
```

### Path 2: Binary .snap Files (Advanced)

Read `.snap` files directly using the binary format (see [WIRE_FORMAT.md](WIRE_FORMAT.md)):

```python
import struct
import hashlib
import numpy as np

def read_snap_file(path):
    with open(path, "rb") as f:
        data = f.read()

    # Parse 48-byte header
    magic = data[0:4]
    assert magic == b"CJCS", f"Invalid magic: {magic}"

    version = struct.unpack("<I", data[4:8])[0]
    assert version in (1, 2), f"Unsupported version: {version}"

    content_hash = data[8:40]
    data_len = struct.unpack("<Q", data[40:48])[0]
    payload = data[48:48 + data_len]

    # Verify integrity
    actual_hash = hashlib.sha256(payload).digest()
    assert actual_hash == content_hash, "Hash mismatch — file corrupted"

    # v2 payloads have a 6-byte header: CJS\x01 + version + flags
    pos = 0
    if payload[0:4] == b"CJS\x01":
        pos = 6  # skip v2 header

    return decode_value(payload, pos)

def decode_value(data, pos):
    tag = data[pos]
    pos += 1

    if tag == 0x00:  # Void
        return None, pos
    elif tag == 0x01:  # Int
        val = struct.unpack("<q", data[pos:pos+8])[0]
        return val, pos + 8
    elif tag == 0x02:  # Float
        val = struct.unpack("<d", data[pos:pos+8])[0]
        return val, pos + 8
    elif tag == 0x03:  # Bool
        return bool(data[pos]), pos + 1
    elif tag == 0x04:  # String
        slen = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 8
        return data[pos:pos+slen].decode("utf-8"), pos + slen
    elif tag == 0x08:  # Tensor (v1 f64-only)
        ndim = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 8
        shape = []
        for _ in range(ndim):
            dim = struct.unpack("<Q", data[pos:pos+8])[0]
            pos += 8
            shape.append(dim)
        numel = 1
        for d in shape:
            numel *= d
        values = struct.unpack(f"<{numel}d", data[pos:pos + numel*8])
        pos += numel * 8
        return np.array(values).reshape(shape), pos
    elif tag == 0x14:  # Sparse CSR (v2)
        dtype_tag = data[pos]; pos += 1
        nrows = struct.unpack("<Q", data[pos:pos+8])[0]; pos += 8
        ncols = struct.unpack("<Q", data[pos:pos+8])[0]; pos += 8
        nnz = struct.unpack("<Q", data[pos:pos+8])[0]; pos += 8
        row_ptr = list(struct.unpack(f"<{nrows+1}Q", data[pos:pos+(nrows+1)*8]))
        pos += (nrows+1)*8
        col_idx = list(struct.unpack(f"<{nnz}Q", data[pos:pos+nnz*8]))
        pos += nnz*8
        values = list(struct.unpack(f"<{nnz}d", data[pos:pos+nnz*8]))
        pos += nnz*8
        from scipy.sparse import csr_matrix
        return csr_matrix((values, col_idx, row_ptr), shape=(nrows, ncols)), pos
    # ... handle other tags (0x12, 0x13, 0x15-0x17) as needed
    else:
        raise ValueError(f"Unknown tag: 0x{tag:02x}")
```

### Type Mapping: CJC → Python

| CJC Type | JSON `__type` | Python Equivalent |
|----------|---------------|-------------------|
| Int | (bare number) | `int` |
| Float | (bare number) | `float` |
| Bool | (bare bool) | `bool` |
| String | (bare string) | `str` |
| Void | `null` | `None` |
| Array | (bare array) | `list` |
| Tuple | (bare array) | `tuple` |
| Tensor | `"Tensor"` | `numpy.ndarray` |
| Struct | `"Struct"` | `dict` (with `name` field) |
| Enum | `"Enum"` | `dict` (with `variant` field) |
| Complex | `"Complex"` | `complex` (`re + im*1j`) |
| Bytes | `"Bytes"` | `bytes.fromhex(hex)` |
| Bf16 | `"Bf16"` | `numpy.float16` (approx) |
| F16 | `"F16"` | `numpy.float16` |
| Map | `"Map"` | `dict` (from entries) |
| SparseTensor | (v2 binary only) | `scipy.sparse.csr_matrix` |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  CJC Program                                        │
│  snap() / snap_v2()  /  restore() / restore_v2()    │
│  snap_save() / snap_save_v2() / snap_load()         │
│  snap_hash() / snap_to_json() / memo_call()         │
└──────────────┬──────────────────────────────────────┘
               │ dispatch_call (stateful builtins)
┌──────────────┴──────────────────────────────────────┐
│  cjc-eval / cjc-mir-exec                            │
│  Interpreter::memo_cache / MirExecutor::memo_cache   │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────┴──────────────────────────────────────┐
│  cjc-snap crate                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ encode.rs│  │ decode.rs│  │ hash.rs  │          │
│  │ v1 + v2  │  │ v1 + v2  │  │ SHA-256  │          │
│  │ 24 tags  │  │ auto-    │  │ (FIPS    │          │
│  │ 0x00-0x17│  │ detect   │  │  180-4)  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│  ┌──────────┐  ┌──────────┐                         │
│  │ json.rs  │  │persist.rs│                         │
│  │ Value →  │  │ v1 + v2  │                         │
│  │ JSON     │  │ .snap    │                         │
│  └──────────┘  └──────────┘                         │
└─────────────────────────────────────────────────────┘
               │
┌──────────────┴──────────────────────────────────────┐
│  cjc-runtime  (Value, Tensor, SparseCsr, DType)     │
└─────────────────────────────────────────────────────┘
```

## Testing

CJC Snap is covered by 49 integration tests in `tests/language_hardening/test_lh11_snap.rs` plus ~90 unit tests across the crate modules:

```bash
# Run all snap integration tests
cargo test --test test_language_hardening -- test_lh11

# Run snap crate unit tests (90 tests)
cargo test -p cjc-snap

# Full regression suite (5,156 tests)
cargo test --workspace
```

### v2-specific tests

The v2 encoding is covered by dedicated tests:

- **Typed tensor roundtrips** — F64, F32, I64, etc. via `TAG_TYPED_TENSOR`
- **Chunked tensor** — single-chunk, multi-chunk, empty tensor, hash integrity verification, determinism
- **Sparse CSR** — roundtrip through `TAG_SPARSE_CSR`, verify nrows/ncols/nnz/values
- **DataFrame** — columnar encoding with Int, Float, Str, Bool, Categorical, DateTime columns
- **Categorical** — standalone level/code encoding
- **Schema** — field metadata roundtrip
- **v2 persist** — `snap_save_v2()` / `snap_load()` file roundtrips for all value types
- **Backward compatibility** — v1 files still load correctly via `snap_load()`

## Effect Classification

| Builtin | Effect | Reason |
|---------|--------|--------|
| `snap` | `alloc` | Allocates SnapBlob struct (v1 encoding) |
| `snap_v2` | `alloc` | Allocates SnapBlob struct (v2 encoding) |
| `restore` | `alloc` | Allocates decoded Value (v1 only) |
| `restore_v2` | `alloc` | Allocates decoded Value (auto-detects v1/v2) |
| `snap_hash` | `alloc` | Allocates hash String |
| `snap_save` | `io` | Writes v1 .snap file to filesystem |
| `snap_save_v2` | `io` | Writes v2 .snap file to filesystem |
| `snap_load` | `io_alloc` | Reads from filesystem + allocates (auto-detects v1/v2) |
| `snap_to_json` | `alloc` | Allocates JSON String |
| `memo_call` | `alloc` | Allocates cache entry + result |

## Rust API Reference

### High-Level Functions

```rust
use cjc_snap::{snap, snap_v2, restore, restore_v2, SnapBlob, SnapError};

// v1 encoding (tags 0x00-0x11)
let blob: SnapBlob = snap(&value);
let value: Value = restore(&blob)?;

// v2 encoding (tags 0x00-0x17, with CJS\x01 header)
let blob: SnapBlob = snap_v2(&value);
let value: Value = restore_v2(&blob)?;  // auto-detects v1/v2
```

### File Persistence

```rust
use cjc_snap::persist::{snap_save, snap_save_v2, snap_load};

snap_save(&value, "file.snap")?;     // v1 file format
snap_save_v2(&value, "file.snap")?;  // v2 file format
let value = snap_load("file.snap")?; // auto-detects v1/v2
```

### Specialized Encoders

```rust
use cjc_snap::{
    encode_typed_tensor, encode_chunked_tensor,
    encode_sparse_csr, encode_categorical,
    encode_schema, encode_dataframe,
    DataFrameColumnData, DEFAULT_CHUNK_SIZE,
};

// Typed tensor with explicit dtype
encode_typed_tensor(dtype_tag, &shape, &raw_bytes, &mut buf);

// Chunked tensor for large data (4 MB chunks, per-chunk SHA-256)
encode_chunked_tensor(dtype_tag, &shape, &raw_bytes, DEFAULT_CHUNK_SIZE, &mut buf);

// Sparse CSR matrix
encode_sparse_csr(nrows, ncols, &row_offsets, &col_indices, &values, &mut buf);

// Categorical column
encode_categorical(&levels, &codes, &mut buf);

// DataFrame with typed columns
encode_dataframe(&col_names, &col_types, &col_data, n_rows, &mut buf);
```
