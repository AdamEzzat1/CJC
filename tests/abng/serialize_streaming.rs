//! Phase 0.8 Item B2 — streaming snapshot encode tests.
//!
//! Verifies that `serialize_into(g, w)` produces byte-identical output
//! to `serialize(g)` regardless of the writer's chunking strategy,
//! that the resulting bytes still round-trip through `replay`, and
//! that I/O errors from the underlying writer short-circuit cleanly.

use std::io;
use std::io::Write;

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::serialize::{replay, serialize, serialize_into};

/// Build a non-trivial graph: 4 leaves, codebook frozen, several
/// rounds of observations spread across leaves so the audit log
/// has enough events that the streaming path's per-event payload
/// scratch is meaningfully exercised.
fn build_workload(seed: u64, n_obs: usize) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    g.set_codebook(1, 4, &[-1.0, 0.0, 1.0]).unwrap();
    for byte in 1u8..5 {
        let _ = g.add_node(0, byte).unwrap();
    }
    for i in 0..n_obs {
        let leaf = ((i % 4) + 1) as u32;
        let v = (i as f64 * 0.0001) - 0.5;
        g.observe(leaf, v).unwrap();
    }
    g
}

#[test]
fn serialize_into_matches_serialize_bytewise() {
    let g = build_workload(42, 1_024);

    let blob_vec = serialize(&g);

    let mut blob_stream: Vec<u8> = Vec::new();
    serialize_into(&g, &mut blob_stream).expect("serialize_into(Vec<u8>)");

    assert_eq!(
        blob_vec.len(),
        blob_stream.len(),
        "serialize and serialize_into produced different byte counts",
    );
    assert_eq!(
        blob_vec, blob_stream,
        "serialize and serialize_into produced different bytes",
    );
}

#[test]
fn serialize_into_round_trips_through_replay() {
    let g = build_workload(123, 256);

    let mut blob: Vec<u8> = Vec::new();
    serialize_into(&g, &mut blob).expect("serialize_into");

    let g2 = replay(&blob).expect("replay");
    assert_eq!(g.chain_head, g2.chain_head);
    assert_eq!(g.seed, g2.seed);
    assert_eq!(g.epoch, g2.epoch);
    assert_eq!(g.audit_len(), g2.audit_len());
    // Re-serialize the replayed graph and confirm it produces the
    // same bytes as the original streamed blob.
    let mut blob2: Vec<u8> = Vec::new();
    serialize_into(&g2, &mut blob2).expect("re-serialize_into");
    assert_eq!(
        blob, blob2,
        "round-trip via replay produced a graph that re-serializes differently"
    );
}

/// Writer that splits every `write` call into 1-byte writes to its
/// underlying buffer. Catches any place where serialize_into accidentally
/// relies on `write_all`'s "all or nothing" guarantee at the wrong
/// granularity.
struct OneByteWriter<'a> {
    inner: &'a mut Vec<u8>,
}

impl<'a> Write for OneByteWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        self.inner.push(buf[0]);
        Ok(1)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn serialize_into_one_byte_writes_succeed() {
    let g = build_workload(7, 64);

    let reference = serialize(&g);

    let mut chunked: Vec<u8> = Vec::new();
    {
        let mut writer = OneByteWriter {
            inner: &mut chunked,
        };
        serialize_into(&g, &mut writer).expect("serialize_into(OneByteWriter)");
    }

    assert_eq!(
        reference, chunked,
        "one-byte-at-a-time writes diverged from `serialize` output"
    );
}

/// Writer that returns `ErrorKind::BrokenPipe` after `n_ok` bytes have
/// been written.
struct FailingWriter {
    written: usize,
    fail_after: usize,
    accepted: Vec<u8>,
}

impl Write for FailingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.written >= self.fail_after {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "FailingWriter: planned failure",
            ));
        }
        let remaining = self.fail_after - self.written;
        let n = buf.len().min(remaining);
        self.accepted.extend_from_slice(&buf[..n]);
        self.written += n;
        if n < buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "FailingWriter: partial-then-fail",
            ));
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn serialize_into_propagates_io_errors() {
    let g = build_workload(0, 32);

    // Fail after 16 bytes (just past the magic + seed prefix). The
    // exact cutoff doesn't matter; what matters is that `?` short-
    // circuits and the call returns Err.
    let mut writer = FailingWriter {
        written: 0,
        fail_after: 16,
        accepted: Vec::new(),
    };
    let result = serialize_into(&g, &mut writer);

    let err = result.expect_err("serialize_into must propagate the writer's error");
    assert_eq!(
        err.kind(),
        io::ErrorKind::BrokenPipe,
        "expected BrokenPipe, got {err:?}"
    );
    // The writer accepted exactly the bytes up to the failure boundary.
    // Crucially, no bytes were written *past* the failure — proving
    // that `?` correctly short-circuits the rest of `serialize_into`.
    assert_eq!(
        writer.accepted.len(),
        16,
        "expected exactly 16 bytes accepted before the failure"
    );
}
