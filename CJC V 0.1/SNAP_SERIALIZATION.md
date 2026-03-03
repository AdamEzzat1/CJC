# Snap: Content-Addressable Serialization

Snap is CJC's built-in serialization system. It encodes any CJC value into a
binary blob with a SHA-256 content hash, enabling content-addressable storage,
memoization, and cross-language interop.

## Quick Start

```
// Encode a value
let blob = snap([1, 2, 3, 4, 5]);

// Decode it back
let restored = restore(blob);
print(restored);                   // [1, 2, 3, 4, 5]

// Get the content hash
let hash = snap_hash([1, 2, 3, 4, 5]);
print(hash);                       // SHA-256 hex string
```

## Built-in Functions

| Function | Signature | Effect | Description |
|----------|-----------|--------|-------------|
| `snap` | `snap(value) -> SnapBlob` | alloc | Encode value to binary blob |
| `restore` | `restore(blob) -> Value` | alloc | Decode blob back to value |
| `snap_hash` | `snap_hash(value) -> str` | alloc | SHA-256 hash without storing blob |
| `snap_save` | `snap_save(value, path)` | io | Encode and save to disk |
| `snap_load` | `snap_load(path) -> Value` | io | Load and decode from disk |
| `snap_to_json` | `snap_to_json(value) -> str` | alloc | Convert to JSON representation |
| `memo_call` | `memo_call(fn, args...) -> Value` | io | Memoized function call |

## Supported Types

All CJC value types are snap-encodable:

| Type | Tag Byte | Notes |
|------|----------|-------|
| `i64` | 0x01 | 8-byte little-endian |
| `f64` | 0x02 | 8-byte IEEE 754 |
| `bool` | 0x03 | 1 byte (0/1) |
| `str` | 0x04 | Length-prefixed UTF-8 |
| `Array` | 0x05 | Length + recursive elements |
| `Void` | 0x06 | Zero bytes |
| `Tensor` | 0x07 | Shape + flat f64 data |
| `Struct` | 0x08 | Name + sorted fields |
| `Tuple` | 0x09 | Length + elements |
| `Map` | 0x0A | Length + sorted key-value pairs |
| `i32` | 0x0B | 4-byte little-endian |
| `f32` | 0x0C | 4-byte IEEE 754 |
| `u8` | 0x0D | 1 byte |
| `Bytes` | 0x0E | Length-prefixed raw bytes |
| `Complex` | 0x0F | 16 bytes (re + im as f64) |
| `Enum` | 0x10 | Variant name + optional payload |

## File I/O

```
// Save a model's weights to disk
let weights = Tensor.randn([100, 50]);
snap_save(weights, "model_weights.snap");

// Load them back
let loaded = snap_load("model_weights.snap");
print(Tensor.shape(loaded));       // [100, 50]
```

### .snap File Format

```
Bytes 0-3:    Magic "CJCS" (0x43 0x4A 0x43 0x53)
Bytes 4-7:    Version (u32 little-endian, currently 1)
Bytes 8-39:   SHA-256 content hash (32 bytes)
Bytes 40-47:  Data length (u64 little-endian)
Bytes 48+:    Encoded data (tag-length-value format)
```

## JSON Export

```
let data = [1, 2, 3];
let json = snap_to_json(data);
print(json);
// {"type":"array","value":[{"type":"i64","value":1},...]}

// Save as JSON for Python/R consumption
file_write("export.json", snap_to_json(data));
```

## Memoization

`memo_call` caches function results using snap hashes of the arguments:

```
fn expensive_compute(n: i64) -> i64 {
    // Imagine this takes a long time
    let mut sum: i64 = 0;
    for i in 0..n {
        sum = sum + i * i;
    }
    sum
}

// First call: computes and caches
let r1 = memo_call(expensive_compute, 10000);

// Second call: returns cached result instantly
let r2 = memo_call(expensive_compute, 10000);

print(r1 == r2);              // true
```

## Python Interop

### Via JSON (Simplest)

```python
import json

# In CJC: snap_to_json(data) -> written to file
with open("export.json") as f:
    data = json.load(f)

# Navigate the typed JSON structure
for item in data["value"]:
    print(item["type"], item["value"])
```

### Via Binary Wire Format

```python
import struct
import hashlib

def read_snap(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"CJCS", f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        content_hash = f.read(32)
        data_len = struct.unpack("<Q", f.read(8))[0]
        data = f.read(data_len)

        # Verify integrity
        actual_hash = hashlib.sha256(data).digest()
        assert actual_hash == content_hash, "Hash mismatch!"

        return decode_value(data)
```

## Use Cases

1. **Checkpointing** — Save training state at each epoch with `snap_save`
2. **Memoization** — Cache expensive computations with `memo_call`
3. **Content-addressable dedup** — Same data always produces same hash
4. **Cross-language export** — `snap_to_json` for Python/R/Julia consumption
5. **Reproducibility** — Snap blobs are deterministic (ordered maps, stable hashes)
