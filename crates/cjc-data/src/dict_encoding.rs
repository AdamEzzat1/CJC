//! Stable dictionary encoding for string columns.
//!
//! Uses `BTreeMap` for deterministic key ordering — the same input data always
//! produces the same dictionary codes regardless of platform or run.

use std::collections::BTreeMap;

/// A dictionary-encoded string column.
///
/// Unique string values are assigned compact `u32` codes in **sorted order**
/// (via `BTreeMap`), guaranteeing deterministic encoding across runs.
#[derive(Debug, Clone)]
pub struct DictEncoding {
    /// Dictionary: sorted unique values -> compact u32 codes.
    dict: BTreeMap<String, u32>,
    /// Reverse lookup: code -> string (sorted order).
    reverse: Vec<String>,
    /// Encoded data: each row's string mapped to its u32 code.
    codes: Vec<u32>,
}

impl DictEncoding {
    /// Build a dictionary encoding from a string column.
    ///
    /// Unique values are sorted (BTreeMap natural ordering) and assigned
    /// consecutive codes starting from 0.
    pub fn encode(data: &[String]) -> Self {
        // Collect unique values in sorted order via BTreeMap.
        let mut unique: BTreeMap<String, u32> = BTreeMap::new();
        for s in data {
            unique.entry(s.clone()).or_insert(0);
        }

        // Assign consecutive codes in sorted order.
        let mut reverse = Vec::with_capacity(unique.len());
        for (i, (key, code)) in unique.iter_mut().enumerate() {
            *code = i as u32;
            reverse.push(key.clone());
        }

        // Encode the data.
        let codes: Vec<u32> = data.iter().map(|s| unique[s]).collect();

        DictEncoding {
            dict: unique,
            reverse,
            codes,
        }
    }

    /// Decode back to the original string values.
    pub fn decode(&self) -> Vec<String> {
        self.codes
            .iter()
            .map(|&c| self.reverse[c as usize].clone())
            .collect()
    }

    /// Look up the code for a string value.
    pub fn lookup(&self, value: &str) -> Option<u32> {
        self.dict.get(value).copied()
    }

    /// Number of unique values in the dictionary.
    pub fn cardinality(&self) -> usize {
        self.reverse.len()
    }

    /// Get the encoded codes slice.
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// Get the dictionary mapping (sorted).
    pub fn dict(&self) -> &BTreeMap<String, u32> {
        &self.dict
    }

    /// Get the reverse lookup table (code -> string).
    pub fn reverse(&self) -> &[String] {
        &self.reverse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let data: Vec<String> = vec!["banana", "apple", "cherry", "apple", "banana"]
            .into_iter()
            .map(String::from)
            .collect();
        let enc = DictEncoding::encode(&data);
        assert_eq!(enc.decode(), data);
    }

    #[test]
    fn test_sorted_codes() {
        let data: Vec<String> = vec!["cherry", "apple", "banana"]
            .into_iter()
            .map(String::from)
            .collect();
        let enc = DictEncoding::encode(&data);
        // BTreeMap sorts: apple=0, banana=1, cherry=2
        assert_eq!(enc.lookup("apple"), Some(0));
        assert_eq!(enc.lookup("banana"), Some(1));
        assert_eq!(enc.lookup("cherry"), Some(2));
        assert_eq!(enc.cardinality(), 3);
    }

    #[test]
    fn test_empty() {
        let enc = DictEncoding::encode(&[]);
        assert_eq!(enc.cardinality(), 0);
        assert!(enc.codes().is_empty());
        assert!(enc.decode().is_empty());
    }
}
