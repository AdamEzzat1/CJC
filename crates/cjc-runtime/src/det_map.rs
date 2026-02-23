use crate::value::Value;

// ---------------------------------------------------------------------------
// 5. Deterministic Map (open addressing, MurmurHash3, insertion-order iteration)
// ---------------------------------------------------------------------------

/// Deterministic MurmurHash3 finalizer (64-bit, fixed seed).
pub fn murmurhash3_finalize(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Hash a byte slice using MurmurHash3 with a fixed seed.
pub fn murmurhash3(data: &[u8]) -> u64 {
    const SEED: u64 = 0x5f3759df;
    let mut h = SEED;
    // Process 8-byte chunks
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let mut k = [0u8; 8];
        k.copy_from_slice(&data[i * 8..(i + 1) * 8]);
        let k = u64::from_le_bytes(k);
        let k = k.wrapping_mul(0x87c37b91114253d5);
        let k = k.rotate_left(31);
        let k = k.wrapping_mul(0x4cf5ad432745937f);
        h ^= k;
        h = h.rotate_left(27);
        h = h.wrapping_mul(5).wrapping_add(0x52dce729);
    }
    // Tail
    let tail = &data[chunks * 8..];
    let mut k: u64 = 0;
    for (i, &b) in tail.iter().enumerate() {
        k |= (b as u64) << (i * 8);
    }
    if !tail.is_empty() {
        let k = k.wrapping_mul(0x87c37b91114253d5);
        let k = k.rotate_left(31);
        let k = k.wrapping_mul(0x4cf5ad432745937f);
        h ^= k;
    }
    h ^= data.len() as u64;
    murmurhash3_finalize(h)
}

/// Deterministic hash for a Value. Uses MurmurHash3 with fixed seed.
pub fn value_hash(val: &Value) -> u64 {
    match val {
        Value::Int(n) => murmurhash3(&n.to_le_bytes()),
        Value::Float(f) => murmurhash3(&f.to_bits().to_le_bytes()),
        Value::Bool(b) => murmurhash3(&[*b as u8]),
        Value::String(s) => murmurhash3(s.as_bytes()),
        Value::Bytes(b) => murmurhash3(&b.borrow()),
        Value::ByteSlice(b) => murmurhash3(b),
        Value::StrView(b) => murmurhash3(b),
        Value::U8(v) => murmurhash3(&[*v]),
        Value::Bf16(v) => murmurhash3(&v.0.to_le_bytes()),
        Value::Enum {
            enum_name,
            variant,
            fields,
        } => {
            // Hash enum_name + variant + each field
            let mut h = murmurhash3(enum_name.as_bytes());
            h ^= murmurhash3(variant.as_bytes());
            for f in fields {
                h ^= value_hash(f);
            }
            h
        }
        Value::Void => murmurhash3(&[0xff]),
        _ => murmurhash3(&[0xfe]),
    }
}

/// A deterministic hash map with insertion-order iteration.
/// Uses open addressing with Fibonacci hashing.
#[derive(Debug, Clone)]
pub struct DetMap {
    /// Entries: (hash, key, value). None for empty slots.
    entries: Vec<Option<(u64, Value, Value)>>,
    /// Insertion-order indices into `entries`.
    order: Vec<usize>,
    len: usize,
    capacity: usize,
}

const FIBONACCI_CONSTANT: u64 = 11400714819323198485; // 2^64 / phi

impl DetMap {
    pub fn new() -> Self {
        let capacity = 8;
        DetMap {
            entries: vec![None; capacity],
            order: Vec::new(),
            len: 0,
            capacity,
        }
    }

    fn slot_index(&self, hash: u64) -> usize {
        let shift = 64 - (self.capacity.trailing_zeros() as u64);
        (hash.wrapping_mul(FIBONACCI_CONSTANT) >> shift) as usize
    }

    pub fn insert(&mut self, key: Value, value: Value) {
        if self.len * 4 >= self.capacity * 3 {
            self.grow();
        }
        let hash = value_hash(&key);
        let mut slot = self.slot_index(hash);
        loop {
            match &self.entries[slot] {
                None => {
                    self.entries[slot] = Some((hash, key, value));
                    self.order.push(slot);
                    self.len += 1;
                    return;
                }
                Some((h, k, _)) => {
                    if *h == hash && values_equal_static(k, &key) {
                        // Overwrite value, preserve insertion order
                        self.entries[slot] = Some((hash, key, value));
                        return;
                    }
                    slot = (slot + 1) % self.capacity;
                }
            }
        }
    }

    pub fn get(&self, key: &Value) -> Option<&Value> {
        let hash = value_hash(key);
        let mut slot = self.slot_index(hash);
        let start = slot;
        loop {
            match &self.entries[slot] {
                None => return None,
                Some((h, k, v)) => {
                    if *h == hash && values_equal_static(k, key) {
                        return Some(v);
                    }
                    slot = (slot + 1) % self.capacity;
                    if slot == start {
                        return None;
                    }
                }
            }
        }
    }

    pub fn contains_key(&self, key: &Value) -> bool {
        self.get(key).is_some()
    }

    pub fn remove(&mut self, key: &Value) -> Option<Value> {
        let hash = value_hash(key);
        let mut slot = self.slot_index(hash);
        let start = slot;
        loop {
            match &self.entries[slot] {
                None => return None,
                Some((h, k, _)) => {
                    if *h == hash && values_equal_static(k, key) {
                        let (_, _, v) = self.entries[slot].take().unwrap();
                        self.len -= 1;
                        self.order.retain(|&s| s != slot);
                        // Re-insert displaced entries (Robin Hood-style cleanup)
                        // Preserve insertion-order by updating slot refs in-place
                        let mut next = (slot + 1) % self.capacity;
                        while self.entries[next].is_some() {
                            let entry = self.entries[next].take().unwrap();
                            let old_slot = next;
                            let rehash = entry.0;
                            let rkey = entry.1;
                            let rval = entry.2;
                            // Find new slot
                            let mut new_slot = self.slot_index(rehash);
                            while self.entries[new_slot].is_some() {
                                new_slot = (new_slot + 1) % self.capacity;
                            }
                            self.entries[new_slot] = Some((rehash, rkey, rval));
                            // Update order in-place: replace old_slot with new_slot
                            // This preserves the original insertion position
                            for s in &mut self.order {
                                if *s == old_slot {
                                    *s = new_slot;
                                    break;
                                }
                            }
                            next = (next + 1) % self.capacity;
                        }
                        return Some(v);
                    }
                    slot = (slot + 1) % self.capacity;
                    if slot == start {
                        return None;
                    }
                }
            }
        }
    }

    /// Iterate in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.order.iter().filter_map(move |&slot| {
            self.entries[slot]
                .as_ref()
                .map(|(_, k, v)| (k, v))
        })
    }

    pub fn keys(&self) -> Vec<Value> {
        self.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn values_vec(&self) -> Vec<Value> {
        self.iter().map(|(_, v)| v.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;
        let old_order = self.order.clone();
        let old_entries: Vec<_> = old_order
            .iter()
            .filter_map(|&slot| self.entries[slot].clone())
            .collect();

        self.entries = vec![None; new_capacity];
        self.order = Vec::new();
        self.len = 0;
        self.capacity = new_capacity;

        for (hash, key, value) in old_entries {
            let _ = hash; // Re-hash with new capacity
            self.insert(key, value);
        }
    }
}

impl Default for DetMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Static value equality (no interpreter context needed).
pub fn values_equal_static(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bytes(a), Value::Bytes(b)) => *a.borrow() == *b.borrow(),
        (Value::ByteSlice(a), Value::ByteSlice(b)) => **a == **b,
        (Value::StrView(a), Value::StrView(b)) => **a == **b,
        (Value::U8(a), Value::U8(b)) => a == b,
        (Value::Bf16(a), Value::Bf16(b)) => a.0 == b.0,
        (
            Value::Enum {
                enum_name: en1,
                variant: v1,
                fields: f1,
            },
            Value::Enum {
                enum_name: en2,
                variant: v2,
                fields: f2,
            },
        ) => {
            en1 == en2
                && v1 == v2
                && f1.len() == f2.len()
                && f1
                    .iter()
                    .zip(f2.iter())
                    .all(|(a, b)| values_equal_static(a, b))
        }
        (Value::Void, Value::Void) => true,
        _ => false,
    }
}

