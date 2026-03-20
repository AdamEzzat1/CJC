# CJC Snap Wire Format Specification

This document describes the binary encoding format used by CJC Snap for deterministic, content-addressable serialization of CJC values.

## Design Principles

1. **Deterministic** — The same logical value always produces the exact same byte sequence, regardless of BTreeMap iteration order, Rc pointer identity, or floating-point NaN payload.
2. **Self-describing** — Each encoded value starts with a tag byte identifying its type, enabling streaming decoding without a schema.
3. **Byte-first** — Raw bytes are the primary representation; typed views are computed on demand. This enables zero-copy access and efficient streaming.
4. **Zero dependencies** — The encoder, decoder, and SHA-256 implementation are all hand-rolled with no external crates.
5. **Content-addressable** — The SHA-256 hash of the canonical encoding serves as a unique identifier for any value.
6. **Versioned** — v1 and v2 formats coexist with automatic detection on load.

## Tag Byte Table

Every encoded value begins with a single tag byte:

### Core Tags (v1 — 0x00–0x11)

| Tag | Hex | Type | Size After Tag |
|-----|-----|------|----------------|
| `TAG_VOID` | `0x00` | Void/null | 0 bytes |
| `TAG_INT` | `0x01` | 64-bit signed integer | 8 bytes (i64 LE) |
| `TAG_FLOAT` | `0x02` | 64-bit float | 8 bytes (f64 LE, NaN canonicalized) |
| `TAG_BOOL` | `0x03` | Boolean | 1 byte (0x00 or 0x01) |
| `TAG_STRING` | `0x04` | UTF-8 string | 8-byte length + N bytes |
| `TAG_ARRAY` | `0x05` | Array | 8-byte count + N elements |
| `TAG_TUPLE` | `0x06` | Tuple | 8-byte count + N elements |
| `TAG_STRUCT` | `0x07` | Named struct | name + 8-byte field count + sorted fields |
| `TAG_TENSOR` | `0x08` | N-dimensional tensor | 8-byte ndim + dims + f64 data |
| `TAG_ENUM` | `0x09` | Enum variant | enum_name + variant + 8-byte count + fields |
| `TAG_BYTES` | `0x0A` | Byte array | 8-byte length + N bytes |
| `TAG_BYTESLICE` | `0x0B` | Byte slice (view) | 8-byte length + N bytes |
| `TAG_STRVIEW` | `0x0C` | String view (UTF-8 validated bytes) | 8-byte length + N bytes |
| `TAG_U8` | `0x0D` | Unsigned 8-bit integer | 1 byte |
| `TAG_BF16` | `0x0E` | Brain float 16 | 2 bytes (u16 LE, raw bits) |
| `TAG_F16` | `0x0F` | IEEE 754 half-precision | 2 bytes (u16 LE, raw bits) |
| `TAG_COMPLEX` | `0x10` | Complex f64 | 16 bytes (re f64 LE + im f64 LE) |
| `TAG_MAP` | `0x11` | Ordered map | 8-byte count + sorted key-value pairs |

### Extended Tags (v2 — 0x12–0x17)

| Tag | Hex | Type | Size After Tag |
|-----|-----|------|----------------|
| `TAG_TYPED_TENSOR` | `0x12` | Multi-dtype tensor | dtype + ndim + shape + byte_length + raw bytes |
| `TAG_CHUNKED_TENSOR` | `0x13` | Chunked tensor (streaming) | dtype + ndim + shape + chunk_size + n_chunks + chunks |
| `TAG_SPARSE_CSR` | `0x14` | Sparse CSR matrix | dtype + nrows + ncols + nnz + row_ptr + col_idx + values |
| `TAG_CATEGORICAL` | `0x15` | Categorical column | n_levels + level_strings + n_rows + codes |
| `TAG_SCHEMA` | `0x16` | Field schema (metadata) | n_fields + (name + type_tag) pairs |
| `TAG_DATAFRAME` | `0x17` | Columnar DataFrame | n_cols + n_rows + typed column data |

## Encoding Rules

### Primitives

**Void** (0 bytes after tag):
```
[0x00]
```

**Int** (8 bytes, little-endian signed 64-bit):
```
[0x01] [i64 LE: 8 bytes]
```

**Float** (8 bytes, little-endian f64 with NaN canonicalization):
```
[0x02] [f64 LE: 8 bytes]
```
- All NaN variants (positive, negative, signaling, quiet, with any payload) are replaced with the canonical quiet NaN bit pattern `0x7FF8_0000_0000_0000` before encoding.
- This ensures all NaN values hash identically.

**Bool** (1 byte):
```
[0x03] [0x01 for true, 0x00 for false]
```

**U8** (1 byte):
```
[0x0D] [u8 value]
```

### Strings

**String** (length-prefixed UTF-8):
```
[0x04] [u64 LE: byte length] [UTF-8 bytes]
```

The length prefix is always 8 bytes (u64 little-endian), giving a theoretical maximum string length of 2^64 - 1 bytes.

### Byte Types

**Bytes**, **ByteSlice**, **StrView** all follow the same layout:
```
[tag] [u64 LE: byte length] [raw bytes]
```

StrView data is validated as UTF-8 on decode.

### Half-Precision Floats

**Bf16** (Brain Float 16):
```
[0x0E] [u16 LE: raw bit pattern]
```

**F16** (IEEE 754 half-precision):
```
[0x0F] [u16 LE: raw bit pattern]
```

Both store the raw 16-bit representation. No NaN canonicalization is applied at the 16-bit level (the host format preserves the original bits).

### Complex

**Complex** (two f64 values):
```
[0x10] [f64 LE: real part] [f64 LE: imaginary part]
```

Both components undergo NaN canonicalization independently.

### Collections

**Array** and **Tuple**:
```
[tag] [u64 LE: element count] [element₀] [element₁] ... [elementₙ₋₁]
```

Each element is recursively encoded.

**Struct** (sorted fields for determinism):
```
[0x07] [string: struct name] [u64 LE: field count]
  [string: field₀ name] [value: field₀ value]
  [string: field₁ name] [value: field₁ value]
  ...
```

**Critical:** Fields are sorted lexicographically by field name before encoding. This ensures that two structs with the same fields inserted in different order produce identical encodings.

**Enum**:
```
[0x09] [string: enum name] [string: variant name] [u64 LE: field count]
  [value: field₀] [value: field₁] ...
```

**Map** (sorted by canonical key encoding):
```
[0x11] [u64 LE: entry count]
  [value: key₀] [value: value₀]
  [value: key₁] [value: value₁]
  ...
```

Map entries are sorted by the canonical byte encoding of their keys (lexicographic byte comparison). This ensures determinism regardless of insertion order.

### Tensor

**Tensor** (N-dimensional dense f64 array):
```
[0x08] [u64 LE: ndim]
  [u64 LE: dim₀] [u64 LE: dim₁] ... [u64 LE: dimₙ₋₁]
  [f64 LE: data₀] [f64 LE: data₁] ... [f64 LE: dataₘ₋₁]
```

Where `m = dim₀ × dim₁ × ... × dimₙ₋₁` (total element count).

Each f64 element undergoes NaN canonicalization. Data is stored in row-major (C-contiguous) order.

### Typed Tensor (v2)

**Typed Tensor** (multi-dtype N-dimensional array):
```
[0x12] [u8: dtype_tag]
  [u64 LE: ndim]
  [u64 LE: dim₀] ... [u64 LE: dimₙ₋₁]
  [u64 LE: byte_length]
  [raw bytes: byte_length bytes]
```

The `dtype_tag` identifies the element type:

| DType Tag | Type | Element Size |
|-----------|------|-------------|
| `0` | F64 | 8 bytes |
| `1` | F32 | 4 bytes |
| `2` | I64 | 8 bytes |
| `3` | I32 | 4 bytes |
| `4` | U8 | 1 byte |
| `5` | Bool | 1 byte |
| `6` | Bf16 | 2 bytes |
| `7` | F16 | 2 bytes |
| `8` | Complex | 16 bytes |

Raw bytes are stored in the native little-endian format of the element type. NaN canonicalization applies to all floating-point dtypes. The `byte_length` field enables zero-copy slicing without knowing the dtype.

### Chunked Tensor (v2)

**Chunked Tensor** (streaming-friendly large tensor encoding):
```
[0x13] [u8: dtype_tag]
  [u64 LE: ndim]
  [u64 LE: dim₀] ... [u64 LE: dimₙ₋₁]
  [u64 LE: chunk_size]       (default 4 MB = 4,194,304 bytes)
  [u64 LE: n_chunks]
  -- for each chunk:
  [u64 LE: chunk_byte_length]
  [32 bytes: SHA-256 hash of chunk data]
  [raw bytes: chunk_byte_length bytes]
```

**Design rationale:** Large tensors (e.g., model weights) benefit from chunked encoding because:
- **Per-chunk integrity:** Each chunk has its own SHA-256 hash, enabling detection of corruption at chunk granularity.
- **Resumable I/O:** A reader can verify and process chunks independently; a failed transfer only requires re-sending corrupted chunks.
- **Content-addressable chunks:** Identical chunks across different tensors share the same hash, enabling deduplication.

The default chunk size of 4 MB balances hash overhead (~0.001%) against granularity. The last chunk may be smaller than `chunk_size`.

### Sparse CSR Matrix (v2)

**Sparse CSR** (Compressed Sparse Row format):
```
[0x14] [u8: dtype_tag]
  [u64 LE: nrows]
  [u64 LE: ncols]
  [u64 LE: nnz]          (number of non-zero values)
  -- row_offsets: (nrows + 1) entries
  [u64 LE: row_ptr₀] ... [u64 LE: row_ptrₙᵣₒᵥᵥₛ]
  -- col_indices: nnz entries
  [u64 LE: col₀] ... [u64 LE: colₙₙᵤ₋₁]
  -- values: nnz f64 entries
  [f64 LE: val₀] ... [f64 LE: valₙₙᵤ₋₁]
```

CSR is the standard sparse format for row-oriented access. Row offsets are stored as absolute indices (not deltas). Values undergo NaN canonicalization.

### Categorical (v2)

**Categorical** (factor-encoded column):
```
[0x15] [u64 LE: n_levels]
  -- level strings (the category dictionary):
  [string: level₀] [string: level₁] ... [string: levelₙ₋₁]
  [u64 LE: n_rows]
  -- codes (index into levels):
  [u32 LE: code₀] [u32 LE: code₁] ... [u32 LE: codeₙ₋₁]
```

Codes are 32-bit unsigned integers indexing into the level array. This encoding is space-efficient for columns with high repetition (e.g., country codes, categories).

### Schema (v2)

**Schema** (field metadata):
```
[0x16] [u64 LE: n_fields]
  -- for each field:
  [string: field_name]
  [u8: type_tag]
```

Schema provides structural metadata about a record type without encoding the actual data. Useful for catalog entries, format negotiation, and lightweight type checking.

### DataFrame (v2)

**DataFrame** (columnar table encoding):
```
[0x17] [u64 LE: n_cols] [u64 LE: n_rows]
  -- for each column:
  [string: column_name]
  [u8: col_type]
  [column data...]
```

Column types and their data formats:

| col_type | Name | Data Format |
|----------|------|------------|
| `0` | Int | `n_rows` i64 values (LE) |
| `1` | Float | `n_rows` f64 bit patterns (LE, NaN canonicalized) |
| `2` | Str | `n_rows` length-prefixed strings |
| `3` | Bool | `n_rows` u8 values (0x00/0x01) |
| `4` | Categorical | `n_levels` + level strings + `n_rows` u32 codes |
| `5` | DateTime | `n_rows` i64 epoch-millisecond timestamps (LE) |

**Categorical column data within DataFrame:**
```
[u64 LE: n_levels]
  [string: level₀] ... [string: levelₙ₋₁]
  [u32 LE: code₀] ... [u32 LE: codeₙ₋₁]
```

**String column data within DataFrame:**
```
[string: val₀] [string: val₁] ... [string: valₙ₋₁]
```

Each string is length-prefixed as usual (u64 LE length + UTF-8 bytes).

**Design rationale:** Columnar encoding is more efficient than row-based encoding for DataFrames because:
- Homogeneous data within a column compresses better
- Column-wise operations can skip irrelevant columns entirely
- Type metadata is stored once per column, not once per cell

## v2 Payload Header

The v2 encoding wraps the tag+data payload with a version header:

```
[4 bytes: magic "CJS\x01" (0x43, 0x4A, 0x53, 0x01)]
[1 byte: version = 2]
[1 byte: flags = 0x00 (uncompressed)]
[N bytes: recursive tag+data payload]
```

The magic bytes `CJS\x01` distinguish v2 payloads from v1 payloads (which start directly with a tag byte). The `flags` byte is reserved for future use (e.g., compression).

The `restore_v2()` / `snap_decode_v2()` functions auto-detect the format:
- If the first 4 bytes are `CJS\x01`, parse as v2 (skip 6-byte header, decode payload).
- Otherwise, fall back to v1 decoding (raw tag+data).

## Content Hashing

After encoding, the SHA-256 hash of the complete byte stream is computed. This hash serves as the content address.

```
content_hash = SHA-256(encoded_bytes)
```

The SHA-256 implementation follows FIPS 180-4. It uses:
- Initial hash values: square roots of the first 8 primes
- 64 round constants: cube roots of the first 64 primes
- Standard padding: append bit '1', pad to 448 mod 512, append 64-bit big-endian message length

The resulting `SnapBlob` contains both the encoded data and its hash:
```rust
pub struct SnapBlob {
    pub content_hash: [u8; 32],  // SHA-256 digest
    pub data: Vec<u8>,           // canonical encoding
}
```

## .snap File Format

When persisted to disk via `snap_save()` or `snap_save_v2()`, the encoding is wrapped in a self-describing file format:

### v1 File Format (`snap_save`)

```
Offset  Size  Description
─────────────────────────────────────────
0       4     Magic bytes: "CJCS" (0x43, 0x4A, 0x43, 0x53)
4       4     Format version: 1 (u32 little-endian)
8       32    SHA-256 content hash
40      8     Data length in bytes (u64 little-endian)
48      N     Snap-encoded data bytes (v1 tag+data)
```

### v2 File Format (`snap_save_v2`)

```
Offset  Size  Description
─────────────────────────────────────────
0       4     Magic bytes: "CJCS" (0x43, 0x4A, 0x43, 0x53)
4       4     Format version: 2 (u32 little-endian)
8       32    SHA-256 content hash
40      8     Data length in bytes (u64 little-endian)
48      N     Snap-encoded data bytes (v2: CJS\x01 + version + flags + payload)
```

**Total header size: 48 bytes (both versions).**

The v1 and v2 file formats share the same 48-byte header layout. The difference is:
- **v1 data bytes** start directly with a tag byte (0x00–0x11)
- **v2 data bytes** start with the 6-byte v2 payload header (`CJS\x01` + version + flags), followed by tag+data

### Validation on Load

`snap_load()` auto-detects v1 and v2 files. The following checks are performed in order:

1. **Size check:** File must be at least 48 bytes
2. **Magic check:** First 4 bytes must be `CJCS`
3. **Version check:** Version must be `1` or `2`
4. **Truncation check:** File must contain at least `48 + data_length` bytes
5. **Hash check:** SHA-256 of the data bytes must match the stored hash
6. **Decode:** Data is decoded via `restore_v2()`, which auto-detects v1/v2 payload format

Any failure produces a descriptive error message.

## Example: Encoding `Int(42)`

```
Canonical encoding:
  01                         # TAG_INT
  2A 00 00 00 00 00 00 00   # 42 as i64 LE

SHA-256 of above 9 bytes:
  (computed at runtime)

.snap file (48 + 9 = 57 bytes):
  43 4A 43 53               # Magic: "CJCS"
  01 00 00 00               # Version: 1
  XX XX XX XX XX XX XX XX   # SHA-256 hash (32 bytes)
  XX XX XX XX XX XX XX XX
  XX XX XX XX XX XX XX XX
  XX XX XX XX XX XX XX XX
  09 00 00 00 00 00 00 00   # Data length: 9
  01 2A 00 00 00 00 00 00   # Encoded Int(42)
  00
```

## Example: Encoding `Struct { name: "Point", fields: {x: 1, y: 2} }`

```
07                           # TAG_STRUCT
05 00 00 00 00 00 00 00     # name length: 5
50 6F 69 6E 74              # "Point"
02 00 00 00 00 00 00 00     # field count: 2
01 00 00 00 00 00 00 00     # field name length: 1
78                           # "x"
01                           # TAG_INT
01 00 00 00 00 00 00 00     # 1 as i64 LE
01 00 00 00 00 00 00 00     # field name length: 1
79                           # "y"
01                           # TAG_INT
02 00 00 00 00 00 00 00     # 2 as i64 LE
```

Note: Fields are in sorted order ("x" before "y").

## Example: Encoding `Tensor` with shape [2, 3]

```
08                           # TAG_TENSOR
02 00 00 00 00 00 00 00     # ndim: 2
02 00 00 00 00 00 00 00     # dim[0]: 2
03 00 00 00 00 00 00 00     # dim[1]: 3
00 00 00 00 00 00 F0 3F     # data[0]: 1.0 as f64 LE
00 00 00 00 00 00 00 40     # data[1]: 2.0 as f64 LE
00 00 00 00 00 00 08 40     # data[2]: 3.0 as f64 LE
00 00 00 00 00 00 10 40     # data[3]: 4.0 as f64 LE
00 00 00 00 00 00 14 40     # data[4]: 5.0 as f64 LE
00 00 00 00 00 00 18 40     # data[5]: 6.0 as f64 LE
```

Total: 1 (tag) + 8 (ndim) + 16 (shape) + 48 (data) = 73 bytes.

## Non-Encodable Types

The following runtime-only Value variants cannot be snap-encoded and will cause a panic if passed to `snap_encode()`:

- `Fn` — Function pointer
- `Closure` — Closure with captured environment
- `ClassRef` — Class reference
- `Scratchpad` — Mutable scratch buffer
- `GradGraph` — Automatic differentiation graph
- `OptimizerState` — Optimizer state
- `PagedKvCache` — KV cache for attention
- `AlignedBytes` — Aligned memory buffer
- `TidyView` / `GroupedTidyView` — Data views
- `Regex` — Compiled regex

**Note:** `SparseTensor` (CSR format) is now encodable as of v2 via `TAG_SPARSE_CSR` (0x14).

Use `cjc_snap::is_snappable(&value)` to check before encoding. At the CJC language level, the builtins return descriptive errors rather than panicking.

## Versioning

Two format versions exist:

| Version | Tags | Payload Header | API |
|---------|------|---------------|-----|
| v1 | 0x00–0x11 | None (raw tag+data) | `snap()` / `restore()` / `snap_save()` |
| v2 | 0x00–0x17 | `CJS\x01` + version + flags | `snap_v2()` / `restore_v2()` / `snap_save_v2()` |

**Backward compatibility:**
- `restore_v2()` reads both v1 and v2 payloads (auto-detects via magic bytes)
- `snap_load()` reads both v1 and v2 `.snap` files (auto-detects via version field)
- `restore()` reads only v1 payloads
- v1 readers reject v2 files (version check fails)

**Forward compatibility:** The v2 `flags` byte (currently `0x00`) reserves space for future features such as compression (`0x01` = zstd, etc.) without a version bump.

## Security

- SHA-256 integrity checking prevents silent data corruption
- Hash verification on `restore()` / `restore_v2()` catches any tampering or bit-flip
- Per-chunk SHA-256 in chunked tensors enables granular corruption detection
- The SHA-256 implementation is a faithful port of FIPS 180-4 (verified against NIST test vectors)
- No unsafe code in the encoding/decoding path

## Example: Encoding a Chunked Tensor [3, 2] (f64)

For a small tensor that fits in a single chunk:
```
13                           # TAG_CHUNKED_TENSOR
00                           # dtype_tag: 0 = F64
02 00 00 00 00 00 00 00     # ndim: 2
03 00 00 00 00 00 00 00     # dim[0]: 3
02 00 00 00 00 00 00 00     # dim[1]: 2
00 00 40 00 00 00 00 00     # chunk_size: 4194304 (4 MB)
01 00 00 00 00 00 00 00     # n_chunks: 1
30 00 00 00 00 00 00 00     # chunk 0 byte_length: 48 (6 x 8 bytes)
XX XX XX ... (32 bytes)      # chunk 0 SHA-256 hash
XX XX XX ... (48 bytes)      # chunk 0 data (6 f64 values, LE)
```

## Example: Encoding a Sparse CSR 3x3 Identity Matrix

```
14                           # TAG_SPARSE_CSR
00                           # dtype_tag: 0 = F64
03 00 00 00 00 00 00 00     # nrows: 3
03 00 00 00 00 00 00 00     # ncols: 3
03 00 00 00 00 00 00 00     # nnz: 3
-- row_offsets (4 entries):
00 00 00 00 00 00 00 00     # row_ptr[0]: 0
01 00 00 00 00 00 00 00     # row_ptr[1]: 1
02 00 00 00 00 00 00 00     # row_ptr[2]: 2
03 00 00 00 00 00 00 00     # row_ptr[3]: 3
-- col_indices (3 entries):
00 00 00 00 00 00 00 00     # col[0]: 0
01 00 00 00 00 00 00 00     # col[1]: 1
02 00 00 00 00 00 00 00     # col[2]: 2
-- values (3 f64 entries):
00 00 00 00 00 00 F0 3F     # val[0]: 1.0
00 00 00 00 00 00 F0 3F     # val[1]: 1.0
00 00 00 00 00 00 F0 3F     # val[2]: 1.0
```

## Example: Encoding a DataFrame with 2 Columns, 3 Rows

```
17                           # TAG_DATAFRAME
02 00 00 00 00 00 00 00     # n_cols: 2
03 00 00 00 00 00 00 00     # n_rows: 3
-- column 0:
04 00 00 00 00 00 00 00     # name length: 4
6E 61 6D 65                  # "name"
02                           # col_type: 2 = Str
-- 3 strings:
05 00 00 00 00 00 00 00  41 6C 69 63 65    # "Alice"
03 00 00 00 00 00 00 00  42 6F 62          # "Bob"
07 00 00 00 00 00 00 00  43 68 61 72 6C 69 65  # "Charlie"
-- column 1:
03 00 00 00 00 00 00 00     # name length: 3
61 67 65                     # "age"
00                           # col_type: 0 = Int
-- 3 i64 values:
1E 00 00 00 00 00 00 00     # 30
19 00 00 00 00 00 00 00     # 25
23 00 00 00 00 00 00 00     # 35
```
