//! Minimal Parquet reader (v0.4, **structural skeleton only**).
//!
//! ## Honest scope statement
//!
//! A full Parquet reader (Thrift compact protocol decoder + Snappy
//! decompressor + Plain / Dictionary / RLE / BitPacked encodings +
//! page-level navigation + nested schema handling) is 1,500-2,500 LOC
//! of careful, well-tested code. Building it correctly in a single
//! session would compromise either scope or quality.
//!
//! What v0.4 ships here:
//!
//! - **Magic-byte recognition** so we can distinguish "this is a
//!   Parquet file but Locke doesn't decode it yet" from "this is not
//!   Parquet at all."
//! - **File-shape validation**: footer-length sanity, trailing magic.
//! - **Better error messages** with a clear roadmap pointer for users
//!   who hit the boundary.
//!
//! What v0.4 does NOT ship:
//!
//! - Thrift compact protocol decoder
//! - Snappy / Gzip / Zstd / LZ4 decompression
//! - Plain / Dictionary / RLE encoding parsers
//! - Multi-row-group navigation
//! - Nested / repeated types
//!
//! v0.5 will land the full reader as a dedicated effort. Until then,
//! convert Parquet to CSV or JSONL with an external tool; Locke
//! handles those at full fidelity.

use std::fs;
use std::path::Path;

/// Outcome of attempting to open a Parquet file.
#[derive(Debug, Clone, PartialEq)]
pub enum ParquetOpenError {
    /// The file is too small to be a valid Parquet file (< 8 bytes:
    /// the two magic strings + a footer-length).
    TooSmall { size: u64 },
    /// The file does not start with the Parquet magic `PAR1`.
    NotParquet,
    /// The file starts with `PAR1` but does not end with it. Likely
    /// truncated or corrupted.
    MissingTrailingMagic,
    /// The footer length encoded in the trailer doesn't fit in the file.
    InvalidFooterLength { footer_len: u64, file_size: u64 },
    /// The file is a valid Parquet shell but v0.4 doesn't decode the
    /// metadata yet. Carries diagnostic info so users see *what*
    /// Locke could parse.
    UnsupportedV04 {
        file_size: u64,
        footer_size: u64,
        diagnostic: String,
    },
    /// I/O error reading the file.
    Io(String),
}

impl std::fmt::Display for ParquetOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParquetOpenError::TooSmall { size } => write!(f, "file is too small to be Parquet ({} bytes)", size),
            ParquetOpenError::NotParquet => write!(f, "file does not start with `PAR1` magic — not a Parquet file"),
            ParquetOpenError::MissingTrailingMagic => write!(f, "file starts with `PAR1` but is missing trailing magic — likely truncated"),
            ParquetOpenError::InvalidFooterLength { footer_len, file_size } => write!(
                f,
                "Parquet footer length {} doesn't fit in {}-byte file",
                footer_len, file_size
            ),
            ParquetOpenError::UnsupportedV04 { file_size, footer_size, diagnostic } => write!(
                f,
                "valid Parquet file ({} bytes, footer {} bytes) but Locke v0.4 cannot decode it yet: {}. \
                 Workaround: convert to CSV / JSONL with `pyarrow.parquet.read_table('file.parquet').to_pandas().to_csv(...)` or similar. \
                 Full Parquet support is on the v0.5 roadmap.",
                file_size, footer_size, diagnostic
            ),
            ParquetOpenError::Io(msg) => write!(f, "Parquet I/O error: {}", msg),
        }
    }
}

impl std::error::Error for ParquetOpenError {}

/// Read just the structural framing of a Parquet file.
///
/// Returns the footer size on success — useful for sanity-checking that
/// the file is well-formed even when Locke can't decode the contents.
///
/// In v0.4 this always returns `Err(UnsupportedV04)` for actually-valid
/// Parquet files; that error includes the decoded footer length so the
/// caller knows the file is structurally sound.
pub fn inspect_parquet_file(path: &Path) -> Result<u64, ParquetOpenError> {
    let bytes = fs::read(path).map_err(|e| ParquetOpenError::Io(e.to_string()))?;
    let size = bytes.len() as u64;

    // Magic is 4 bytes "PAR1" + 4 bytes footer length (LE u32) + 4 bytes "PAR1" trailing.
    // Minimum file size: 8 bytes (magic + trailing magic) + 4 bytes (length) + N (footer).
    if size < 12 {
        return Err(ParquetOpenError::TooSmall { size });
    }

    if &bytes[0..4] != b"PAR1" {
        return Err(ParquetOpenError::NotParquet);
    }

    let trailer_start = bytes.len() - 4;
    if &bytes[trailer_start..] != b"PAR1" {
        return Err(ParquetOpenError::MissingTrailingMagic);
    }

    // Footer length: 4 bytes immediately before the trailing magic.
    let footer_len_start = bytes.len() - 8;
    let footer_len_bytes: [u8; 4] = bytes[footer_len_start..footer_len_start + 4]
        .try_into()
        .map_err(|_| ParquetOpenError::Io("footer length slice".into()))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as u64;

    if footer_len + 8 > size {
        return Err(ParquetOpenError::InvalidFooterLength {
            footer_len,
            file_size: size,
        });
    }

    // We've validated the file is structurally a Parquet file. In v0.4
    // we stop here — the Thrift metadata decoder is deferred to v0.5.
    Err(ParquetOpenError::UnsupportedV04 {
        file_size: size,
        footer_size: footer_len,
        diagnostic: format!(
            "structural framing OK (magic + footer length valid). Thrift metadata decoder, \
             Snappy / Gzip / Zstd decompression, and Plain / Dictionary / RLE encoding parsers \
             are deferred to v0.5"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn too_small_file_is_rejected() {
        let f = write_temp(b"a");
        let res = inspect_parquet_file(f.path());
        assert!(matches!(res, Err(ParquetOpenError::TooSmall { .. })));
    }

    #[test]
    fn non_parquet_file_is_rejected() {
        let f = write_temp(b"This is not a Parquet file, just plain text padding to make it long enough.");
        let res = inspect_parquet_file(f.path());
        assert_eq!(res, Err(ParquetOpenError::NotParquet));
    }

    #[test]
    fn missing_trailing_magic_is_detected() {
        // Starts with PAR1, doesn't end with it.
        let bytes = b"PAR1\x00\x00\x00\x00garbage_payload_data_here_to_make_it_long_enough";
        let f = write_temp(bytes);
        let res = inspect_parquet_file(f.path());
        assert_eq!(res, Err(ParquetOpenError::MissingTrailingMagic));
    }

    #[test]
    fn well_formed_shell_returns_unsupported_v04() {
        // PAR1 + 16 bytes of dummy payload + footer_len(16) LE + PAR1
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"PAR1");
        bytes.extend_from_slice(&[0u8; 16]); // dummy footer
        bytes.extend_from_slice(&16u32.to_le_bytes()); // footer length
        bytes.extend_from_slice(b"PAR1");
        let f = write_temp(&bytes);
        let res = inspect_parquet_file(f.path());
        match res {
            Err(ParquetOpenError::UnsupportedV04 { footer_size, .. }) => {
                assert_eq!(footer_size, 16);
            }
            other => panic!("expected UnsupportedV04, got {:?}", other),
        }
    }

    #[test]
    fn invalid_footer_length_is_detected() {
        // PAR1 + few bytes + huge declared footer + PAR1
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"PAR1");
        bytes.extend_from_slice(&[0u8; 4]);
        bytes.extend_from_slice(&999_999u32.to_le_bytes()); // declared footer is bigger than file
        bytes.extend_from_slice(b"PAR1");
        let f = write_temp(&bytes);
        let res = inspect_parquet_file(f.path());
        assert!(matches!(
            res,
            Err(ParquetOpenError::InvalidFooterLength { .. })
        ));
    }
}
