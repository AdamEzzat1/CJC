# CJC Bytes-First + String Views Specification

**Status:** Locked
**Layer:** 1 (NoGC kernel) + 2 (ergonomics)
**Version:** 2.7-draft

---

## 1. Core Types

### 1.1 Bytes (owning)

An owning, growable byte buffer. Heap-allocated, reference-counted with COW
semantics (same model as existing `Buffer<T>`).

```
Bytes  ≡  Rc<RefCell<Vec<u8>>>   // internal representation
```

**Allocates:** Yes — construction, concatenation, `to_bytes()` conversions.

### 1.2 ByteSlice (view)

A non-owning, immutable view into a contiguous byte range. Zero-copy. Borrows
the underlying `Bytes` buffer (or a static byte literal).

```
ByteSlice  ≡  { data: *const u8, len: usize }   // conceptual
```

**Allocates:** Never. All ByteSlice operations are NoGC-safe.

### 1.3 String (owning UTF-8)

An owning, validated UTF-8 string. Heap-allocated, `Rc<String>` internally
(existing CJC `Value::String`).

**Allocates:** Yes — construction, concatenation, case conversion.

### 1.4 StrView (validated view)

A non-owning view into a validated UTF-8 byte range. Zero-copy. Produced only
via `ByteSlice.as_str_utf8()` which validates the bytes.

```
StrView  ≡  { data: *const u8, len: usize }   // conceptual; same repr as ByteSlice
```

**Allocates:** Never. StrView operations are NoGC-safe (except `to_string()`).

---

## 2. Allocation Rules

| Operation                        | Allocates? | NoGC-safe? |
|----------------------------------|-----------|------------|
| `Bytes.new()`                    | Yes       | No         |
| `Bytes.from_array(arr)`          | Yes       | No         |
| `Bytes.as_slice()`               | No        | Yes        |
| `ByteSlice.slice(a, b)`          | No        | Yes        |
| `ByteSlice.find(byte)`           | No        | Yes        |
| `ByteSlice.split_byte(delim)`    | No*       | Yes*       |
| `ByteSlice.trim_ascii()`         | No        | Yes        |
| `ByteSlice.strip_prefix(p)`      | No        | Yes        |
| `ByteSlice.strip_suffix(s)`      | No        | Yes        |
| `ByteSlice.as_str_utf8()`        | No        | Yes        |
| `ByteSlice == ByteSlice`         | No        | Yes        |
| `StrView.len_bytes()`            | No        | Yes        |
| `StrView.as_bytes()`             | No        | Yes        |
| `StrView.to_string()`            | Yes       | No         |
| `String + String`                | Yes       | No         |
| `b"..." literal`                 | No**      | Yes        |

\* `split_byte` returns an iterator of ByteSlice views; the iterator itself is
stack-allocated. If collected into an array, that array allocation is the
caller's responsibility.

\** Byte string literals are compile-time constants stored in static data.

---

## 3. Literal Syntax

### 3.1 Byte String Literals: `b"..."`

```cjc
let data: ByteSlice = b"Hello, World!\n";
```

- Produces a `ByteSlice` pointing to static (compile-time) data.
- Supports the same escape sequences as regular strings: `\n`, `\t`, `\r`,
  `\\`, `\"`, `\0`, `\xNN` (hex byte).
- Does NOT require valid UTF-8 (arbitrary bytes via `\xNN`).
- Token kind: `ByteStringLit`.

### 3.2 Char Literals: `'c'`

```cjc
let newline: u8 = b'\n';
let comma: u8 = b',';
let letter: u8 = b'A';
```

- `b'c'` produces a `u8` value (single ASCII byte).
- Supports escapes: `\n`, `\t`, `\r`, `\\`, `\'`, `\0`, `\xNN`.
- Token kind: `ByteCharLit`.

**Note:** Unicode char literals (`'c'` without `b` prefix) are reserved for
future use but not implemented in this milestone. Only `b'c'` byte char
literals are supported.

### 3.3 Raw Strings: `r"..."`

```cjc
let pattern: ByteSlice = br"(\d+)\s+(\w+)";
let path: String = r"C:\Users\data\file.txt";
```

- `r"..."` — raw string literal (no escape processing). Produces `String`.
- `br"..."` — raw byte string literal (no escape processing). Produces `ByteSlice`.
- Delimiter hashes: `r#"..."#`, `r##"..."##` etc. for strings containing `"`.
- Token kinds: `RawStringLit`, `RawByteStringLit`.

### 3.4 Slicing Syntax

```cjc
let sub: ByteSlice = buf.slice(start, end);
```

Slicing uses method syntax rather than `buf[a..b]` indexing to keep the
language grammar simple and avoid ambiguity with the existing `..` range
operator. The `slice(start, end)` method produces a zero-copy view.

Single-byte indexing uses standard index syntax:

```cjc
let byte: u8 = buf[i];        // bounds-checked, returns u8
```

---

## 4. Conversion Contracts

### 4.1 Bytes → ByteSlice (zero-copy)

```cjc
let owned: Bytes = Bytes.from_array([72, 101, 108, 108, 111]);
let view: ByteSlice = owned.as_slice();
```

### 4.2 ByteSlice → StrView (validate, zero-copy)

```cjc
let result: Result<StrView, Error> = view.as_str_utf8();
match result {
    Ok(s) => print(s),
    Err(e) => print("invalid UTF-8"),
}
```

- Returns `Result<StrView, Utf8Error>`.
- `Utf8Error` contains: `{ valid_up_to: i64, error_len: i64 }`.
- No allocation on error — error is a stack value.
- No allocation on success — StrView borrows the ByteSlice data.

### 4.3 StrView → String (allocating)

```cjc
let s: String = view.to_string();  // allocates — disallowed in nogc
```

### 4.4 String → ByteSlice (zero-copy)

```cjc
let bytes: ByteSlice = my_string.as_bytes();
```

### 4.5 Disallowed in NoGC

The following operations are rejected at compile time (or by the NoGC verifier)
inside `nogc` functions or blocks:

- `StrView.to_string()`
- `String + String` concatenation
- `Bytes.new()`, `Bytes.from_array()`
- Any operation that constructs an owning `String` or `Bytes`

---

## 5. Invalid UTF-8 Policy

- `Bytes` and `ByteSlice` can contain **any** byte sequence, including invalid
  UTF-8.
- No implicit UTF-8 validation occurs anywhere in the bytes path.
- UTF-8 validation is **explicit** and **opt-in** via `as_str_utf8()`.
- `as_str_utf8()` returns `Result` — there is no panic path.
- String literals (`"..."`) are always valid UTF-8 (compiler-guaranteed).
- Byte string literals (`b"..."`) make no UTF-8 guarantee.

---

## 6. Equality and Hashing

### 6.1 Content Equality

```
ByteSlice == ByteSlice   →  byte-by-byte comparison
StrView == StrView       →  byte-by-byte comparison (same as ByteSlice)
Bytes == Bytes           →  byte-by-byte comparison (via as_slice())
```

### 6.2 Deterministic Hashing

All byte-content types use `murmurhash3` with fixed seed `0x5f3759df` (already
implemented in `cjc_runtime`). This hash is:

- **Deterministic** across runs (no random seed).
- **Deterministic** across platforms (uses little-endian byte order).
- **Stable** — the seed and algorithm are part of the CJC spec.

Contract: `hash(a) == hash(b)` if and only if `a == b` (for byte content).

### 6.3 Canonical Ordering

For deterministic output, byte slices are sorted in **lexicographic byte
order** (unsigned `u8` comparison, left-to-right, shorter-is-less).

This ordering is:
- Total (every pair is comparable).
- Deterministic (same input → same output).
- Platform-independent.

---

## 7. ByteSlice Method Reference (NoGC-safe)

| Method                              | Returns            | Allocates? |
|-------------------------------------|--------------------|------------|
| `len() -> i64`                      | byte count         | No         |
| `is_empty() -> bool`               | true if len == 0   | No         |
| `get(i: i64) -> u8`                | byte at index      | No         |
| `slice(start: i64, end: i64) -> ByteSlice` | sub-view   | No         |
| `find(byte: u8) -> Result<i64, Error>` | first occurrence | No       |
| `find_byte(byte: u8) -> i64`       | index or -1        | No         |
| `split_byte(delim: u8) -> Array<ByteSlice>` | split views | Array alloc |
| `trim_ascii() -> ByteSlice`        | trimmed view       | No         |
| `strip_prefix(p: ByteSlice) -> Result<ByteSlice, Error>` | stripped | No |
| `strip_suffix(s: ByteSlice) -> Result<ByteSlice, Error>` | stripped | No |
| `starts_with(p: ByteSlice) -> bool` | prefix check      | No         |
| `ends_with(s: ByteSlice) -> bool`  | suffix check       | No         |
| `count_byte(b: u8) -> i64`         | occurrences        | No         |
| `as_str_utf8() -> Result<StrView, Utf8Error>` | validated view | No |
| `eq(other: ByteSlice) -> bool`     | content equality   | No         |

## 8. StrView Method Reference (NoGC-safe unless noted)

| Method                              | Returns            | Allocates? |
|-------------------------------------|--------------------|------------|
| `len_bytes() -> i64`               | byte count         | No         |
| `as_bytes() -> ByteSlice`          | byte view          | No         |
| `eq(other: StrView) -> bool`       | content equality   | No         |
| `to_string() -> String`            | owning copy        | Yes (!)    |

---

## 9. Non-Negotiables

1. **No allocations in NoGC ByteSlice operations.** Every method in §7 must be
   provably allocation-free when called in a `nogc` context.

2. **Deterministic hashing contract.** `murmurhash3` with fixed seed
   `0x5f3759df`, as specified in §6.2.

3. **Stable sort contract.** Lexicographic byte order (§6.3) used for all
   deterministic output. Sort is stable (equal elements preserve insertion
   order).

4. **No implicit UTF-8 in bytes paths.** The type system enforces the
   separation between `ByteSlice` (unvalidated) and `StrView` (validated).

5. **Result-based error handling.** `as_str_utf8()` returns `Result`, not
   panicking. Error values are stack-allocated (no heap allocation for error
   messages in NoGC).
