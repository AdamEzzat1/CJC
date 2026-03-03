//! File persistence for CJC Snap — save/load values to disk.
//!
//! ## File Format (.snap)
//!
//! ```text
//! Offset  Size  Description
//! 0       4     Magic: "CJCS" (0x43, 0x4A, 0x43, 0x53)
//! 4       4     Version: 1 (u32 LE)
//! 8       32    SHA-256 content hash
//! 40      8     Data length (u64 LE)
//! 48      N     Snap-encoded data bytes
//! ```

use cjc_runtime::Value;
use crate::{snap, restore};

/// Magic bytes identifying a CJC Snap file.
pub const MAGIC: [u8; 4] = [0x43, 0x4A, 0x43, 0x53]; // "CJCS"

/// Current file format version.
pub const VERSION: u32 = 1;

/// Header size: magic(4) + version(4) + hash(32) + data_len(8) = 48 bytes.
const HEADER_SIZE: usize = 48;

/// Save a CJC value to a `.snap` file.
///
/// The file contains a self-describing header with the SHA-256 hash,
/// followed by the canonical snap encoding. The file can be loaded
/// back with `snap_load()`, or parsed by external tools (e.g., Python).
pub fn snap_save(value: &Value, path: &str) -> Result<(), String> {
    let blob = snap(value);

    let data_len = blob.data.len() as u64;
    let mut file_bytes = Vec::with_capacity(HEADER_SIZE + blob.data.len());

    // Magic
    file_bytes.extend_from_slice(&MAGIC);
    // Version
    file_bytes.extend_from_slice(&VERSION.to_le_bytes());
    // Content hash
    file_bytes.extend_from_slice(&blob.content_hash);
    // Data length
    file_bytes.extend_from_slice(&data_len.to_le_bytes());
    // Data
    file_bytes.extend_from_slice(&blob.data);

    std::fs::write(path, &file_bytes)
        .map_err(|e| format!("snap_save: {}", e))
}

/// Load a CJC value from a `.snap` file.
///
/// Validates the magic bytes, version, and SHA-256 hash integrity.
/// Returns the decoded value or a descriptive error.
pub fn snap_load(path: &str) -> Result<Value, String> {
    let file_bytes = std::fs::read(path)
        .map_err(|e| format!("snap_load: {}", e))?;

    if file_bytes.len() < HEADER_SIZE {
        return Err(format!(
            "snap_load: file too small ({} bytes, need at least {})",
            file_bytes.len(),
            HEADER_SIZE
        ));
    }

    // Validate magic
    if file_bytes[0..4] != MAGIC {
        return Err(format!(
            "snap_load: invalid magic bytes {:02x}{:02x}{:02x}{:02x} (expected CJCS)",
            file_bytes[0], file_bytes[1], file_bytes[2], file_bytes[3]
        ));
    }

    // Validate version
    let version = u32::from_le_bytes(file_bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(format!(
            "snap_load: unsupported version {} (expected {})",
            version, VERSION
        ));
    }

    // Extract hash
    let mut content_hash = [0u8; 32];
    content_hash.copy_from_slice(&file_bytes[8..40]);

    // Extract data length and data
    let data_len = u64::from_le_bytes(file_bytes[40..48].try_into().unwrap()) as usize;
    if file_bytes.len() < HEADER_SIZE + data_len {
        return Err(format!(
            "snap_load: truncated file (header says {} data bytes, file has {})",
            data_len,
            file_bytes.len() - HEADER_SIZE
        ));
    }

    let data = file_bytes[HEADER_SIZE..HEADER_SIZE + data_len].to_vec();

    // Reconstruct blob and restore (verifies hash)
    let blob = crate::SnapBlob { content_hash, data };
    restore(&blob).map_err(|e| format!("snap_load: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_runtime::Tensor;
    use std::rc::Rc;

    fn test_file(name: &str) -> String {
        format!("__test_persist_{}.snap", name)
    }

    fn cleanup(path: &str) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_load_int() {
        let path = test_file("int");
        snap_save(&Value::Int(42), &path).unwrap();
        let loaded = snap_load(&path).unwrap();
        assert!(matches!(loaded, Value::Int(42)));
        cleanup(&path);
    }

    #[test]
    fn test_save_load_string() {
        let path = test_file("string");
        snap_save(&Value::String(Rc::new("hello CJC".into())), &path).unwrap();
        let loaded = snap_load(&path).unwrap();
        match loaded {
            Value::String(s) => assert_eq!(s.as_str(), "hello CJC"),
            _ => panic!("expected String"),
        }
        cleanup(&path);
    }

    #[test]
    fn test_save_load_tensor() {
        let path = test_file("tensor");
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        snap_save(&Value::Tensor(t), &path).unwrap();
        let loaded = snap_load(&path).unwrap();
        match loaded {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            _ => panic!("expected Tensor"),
        }
        cleanup(&path);
    }

    #[test]
    fn test_bad_magic() {
        let path = test_file("bad_magic");
        // Write a file with invalid magic but large enough to pass size check
        let mut bytes = vec![0u8; 48];
        bytes[0..4].copy_from_slice(b"XXXX"); // bad magic
        std::fs::write(&path, &bytes).unwrap();
        let result = snap_load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid magic"));
        cleanup(&path);
    }

    #[test]
    fn test_truncated_file() {
        let path = test_file("truncated");
        std::fs::write(&path, b"CJC").unwrap(); // too short
        let result = snap_load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));
        cleanup(&path);
    }

    #[test]
    fn test_bad_version() {
        let path = test_file("bad_version");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&99u32.to_le_bytes()); // bad version
        bytes.extend_from_slice(&[0u8; 40]); // hash + data_len
        std::fs::write(&path, &bytes).unwrap();
        let result = snap_load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported version"));
        cleanup(&path);
    }

    #[test]
    fn test_missing_file() {
        let result = snap_load("__nonexistent_file_12345.snap");
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_array() {
        let path = test_file("array");
        let val = Value::Array(Rc::new(vec![
            Value::Int(1),
            Value::Float(2.5),
            Value::Bool(true),
        ]));
        snap_save(&val, &path).unwrap();
        let loaded = snap_load(&path).unwrap();
        assert!(matches!(loaded, Value::Array(_)));
        cleanup(&path);
    }
}
