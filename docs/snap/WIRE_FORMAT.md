# CJC Snap Wire Format Specification

This document describes the binary encoding format used by CJC Snap for deterministic, content-addressable serialization of CJC values.

## Design Principles

1. **Deterministic** — The same logical value always produces the exact same byte sequence, regardless of HashMap iteration order, Rc pointer identity, or floating-point NaN payload.
2. **Self-describing** — Each encoded value starts with a tag byte identifying its type, enabling streaming decoding without a schema.
3. **Zero dependencies** — The encoder, decoder, and SHA-256 implementation are all hand-rolled with no external crates.
4. **Content-addressable** — The SHA-256 hash of the canonical encoding serves as a unique identifier for any value.

## Tag Byte Table

Every encoded value begins with a single tag byte:

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

When persisted to disk via `snap_save()`, the encoding is wrapped in a self-describing file format:

```
Offset  Size  Description
─────────────────────────────────────────
0       4     Magic bytes: "CJCS" (0x43, 0x4A, 0x43, 0x53)
4       4     Format version: 1 (u32 little-endian)
8       32    SHA-256 content hash
40      8     Data length in bytes (u64 little-endian)
48      N     Snap-encoded data bytes
```

**Total header size: 48 bytes.**

### Validation on Load

When loading via `snap_load()`, the following checks are performed in order:

1. **Size check:** File must be at least 48 bytes
2. **Magic check:** First 4 bytes must be `CJCS`
3. **Version check:** Version must be `1`
4. **Truncation check:** File must contain at least `48 + data_length` bytes
5. **Hash check:** SHA-256 of the data bytes must match the stored hash
6. **Decode:** Binary data is decoded back to a `Value`

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
- `SparseTensor` — Sparse tensor (COO format)
- `PagedKvCache` — KV cache for attention
- `AlignedBytes` — Aligned memory buffer
- `TidyView` / `GroupedTidyView` — Data views
- `Regex` — Compiled regex

Use `cjc_snap::is_snappable(&value)` to check before encoding. At the CJC language level, the builtins return descriptive errors rather than panicking.

## Versioning

The file format version (currently `1`) is stored in the `.snap` header. Future versions may extend the tag space or add new encoding features while maintaining backwards compatibility. Version `1` readers will reject files with higher version numbers.

## Security

- SHA-256 integrity checking prevents silent data corruption
- Hash verification on `restore()` catches any tampering or bit-flip
- The SHA-256 implementation is a faithful port of FIPS 180-4 (verified against NIST test vectors)
- No unsafe code in the encoding/decoding path
