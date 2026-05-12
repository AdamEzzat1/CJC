//! Binary snapshot format v2 for `AdaptiveBeliefGraph`.
//!
//! Phase 0.2 changes from v1:
//!
//! 1. Magic bumps from `ABNG\x01` to `ABNG\x02`. Phase 0.1 snapshots are
//!    rejected with `BadMagic`.
//! 2. Optional codebook section after the header.
//! 3. Per-node section now carries `parent` (i32 BE; `-1` for root) and
//!    `children_kind` + variant-specific child payload.
//! 4. Audit events now carry the additional Phase 0.2 kinds (`NodeAdded`,
//!    `ChildrenPromoted`, `CodebookFrozen`).
//!
//! Replay rebuilds the graph from events alone (deterministic), then
//! verifies four invariants:
//!
//! 1. Every event's recomputed `new_hash` equals the stored value.
//! 2. Every replayed node's `canonical_bytes` equals the stored value.
//! 3. Every replayed node's children layout (kind + key→child pairs)
//!    equals the stored layout.
//! 4. The final `chain_head` equals the stored `final_hash`.

use std::io::Write;

use cjc_ad::pinn::Activation;
use cjc_runtime::tensor::Tensor;

use crate::audit::{AuditEvent, AuditKind};
use crate::blr::{BlrPrior, BlrState};
use crate::calibration::CalibrationBins;
use crate::children::{AdaptiveChildren, ChildrenKind};
use crate::codebook::QuantileCodebook;
use crate::density::DensityTracker;
use crate::drift::DriftBaseline;
use crate::genesis_hash;
use crate::graph::{AdaptiveBeliefGraph, N_ACTION_KINDS};
use crate::leaf_head::{decode_activation_tag, encode_activation_tag, LeafHead};
use crate::node::{AdaptiveBeliefNode, NodeId};
use crate::policy::DecisionPolicy;
use crate::stats::NodeStats;

// Phase 0.6 Item 4 — wire format v13 (`\x0D`). Bumped from v12 to
// absorb the new `AuditKind::BeliefUpdateBatch` variant (tag `0x1D`)
// with its variable-length payload (count u32 + values f64×count +
// batch_hash [u8; 32]). All 15 Phase 0.5 SHA-256 canaries are
// re-locked simultaneously as part of this bump.
const MAGIC: &[u8; 5] = b"ABNG\x0D";

// Phase 0.8 Item B3 — zstd-compressed snapshot wrapper magic. The
// 6 bytes `ABNGZ\x01` are unambiguously distinguishable from any
// v13/v14/... uncompressed magic (which all start `ABNG\x??`, where
// `??` is the version byte and never `Z` = 0x5A). On detection,
// `replay_with_outcome` decompresses the remaining bytes via
// `zstd::decode_all` and recurses into the uncompressed v13 decoder.
// Wire format of the inner stream is unchanged from v13, so all 28
// SHA-256 canaries remain valid for compressed snapshots that wrap
// canary-locked workloads.
const COMPRESSED_MAGIC: &[u8; 6] = b"ABNGZ\x01";

/// Errors returned by snapshot decoding.
#[derive(Debug, PartialEq)]
pub enum DecodeError {
    /// The blob did not start with the expected `ABNG\x08` magic header.
    /// Older magic bytes (`\x01..\x07`) are explicitly rejected — there
    /// is no backward-compatibility path; each phase clean-breaks.
    BadMagic,
    /// The blob was truncated before reaching the expected length.
    UnexpectedEof,
    /// A kind tag byte did not correspond to a known [`AuditKind`].
    UnknownKindTag(u8),
    /// A children-kind byte did not correspond to a known
    /// [`ChildrenKind`].
    UnknownChildrenKind(u8),
    /// `n_nodes == 0`, which is impossible — every graph has a root.
    EmptyGraph,
    /// The codebook section was malformed or its boundaries were not
    /// strictly ascending.
    BadCodebook,
    /// The replayed chain did not match the stored hashes; carries the
    /// `seq` of the first divergence.
    ChainMismatch { at_seq: u64 },
    /// A replayed node's `canonical_bytes` did not match the stored value;
    /// carries the `node_id`.
    StatsMismatch { node_id: NodeId },
    /// A replayed node's children layout did not match the stored layout;
    /// carries the `node_id`.
    ChildrenMismatch { node_id: NodeId },
    /// A `CodebookFrozen` event's stored hash didn't match the codebook
    /// in the header.
    CodebookHashMismatch,
    /// A `LeafHeadConfigured` event's stored hash didn't match the head
    /// in the header.
    LeafHeadHashMismatch,
    /// A `LeafParamsInitialized` / `LeafParamsUpdated` event's stored
    /// hash didn't match the live node's params hash after replay.
    /// Carries the node id and event seq for diagnosis.
    LeafParamsHashMismatch { node_id: NodeId, at_seq: u64 },
    /// A leaf-head-related field had an invalid encoding (e.g. unknown
    /// activation tag, zero dim).
    BadLeafHead,
    /// A BLR prior in the header had an invalid encoding (non-positive
    /// param, or hash mismatch).
    BadBlrPrior,
    /// A `BlrPriorConfigured` event's stored hash didn't match the prior
    /// in the header.
    BlrPriorHashMismatch,
    /// A `BlrInitialized` / `BlrUpdated` event's stored hash didn't
    /// match the live node's BLR state hash after replay.
    BlrStateHashMismatch { node_id: NodeId, at_seq: u64 },
    /// The replayed final chain head did not match the stored
    /// `final_hash`.
    FinalHashMismatch,
    /// Phase 0.3d-3 — the recomputed `policy_hash` of the decoded
    /// `DecisionPolicy` did not match the stored value, or the
    /// thresholds failed validation.
    DecisionPolicyHashMismatch,
    /// Phase 0.4 Track C-2.3.3 — an audit event's `seq` was not the
    /// expected next monotonic value `expected_seq + 1`. Catches blobs
    /// where the chain hashes are internally consistent (an attacker
    /// recomputed them) but events are reordered, missing, or
    /// duplicated.
    NonMonotonicSeq { expected: u64, got: u64 },
    /// Phase 0.4 Track C-2.3.3 — an audit event's `epoch` did not match
    /// the snapshot header's `epoch`. Catches forged events injected
    /// from a different graph epoch.
    EpochMismatch { expected: u64, got: u64 },
    /// Phase 0.4 Track C-2.3.3 — an audit event's recorded
    /// `stats_version` did not match the live node's `stats_version`
    /// after the event was applied. Catches reordered or swapped
    /// `*Updated` events for the same node, where the chain hashes
    /// validate but the per-event sequence numbers no longer match the
    /// live state evolution.
    StatsVersionMismatch { node_id: NodeId, at_seq: u64 },
    /// Phase 0.4 Track C-2.3.3 — the audit log either contains no events
    /// at all, has a non-`Created` event in the first slot, or contains
    /// a `Created` event after the first slot. Every well-formed graph
    /// has exactly one `Created` event at `seq == 0`.
    CreatedMustBeFirst,
    /// Phase 0.5 Item 1 — the snapshot's stored per-node
    /// `provenance_stamp_hash` did not agree with the value reproduced
    /// by event replay. Indicates the snapshot was tampered with after
    /// the audit chain was finalized, or the per-node section was
    /// rewritten with a stamp that no `ProvenanceStamped` event
    /// authorizes.
    ProvenanceMismatch { node_id: NodeId },
    /// Phase 0.5 Item 2 — the recorded `stats_hash` field on a
    /// `StatsSnapshot` audit event did not equal the SHA-256 of the
    /// post-replay per-node stats canonical bytes for the named
    /// node. Surfaced only by `smart_replay`; the naive `replay`
    /// path leaves the snapshot's `stats_hash` field unchecked.
    /// Indicates either the audit-event payload or the per-node
    /// section was tampered with.
    StatsSnapshotMismatch { node_id: NodeId, at_seq: u64 },
    /// Phase 0.6 Item 4 — a `BeliefUpdateBatch` event decoded with
    /// `count == 0`. Empty batches are rejected at both the encode
    /// and decode boundary; an empty observation list is a no-op
    /// best expressed by simply not calling `observe_batch`.
    EmptyBatch,
    /// Phase 0.6 Item 4 — a `BeliefUpdateBatch` event's recorded
    /// `batch_hash` did not equal the recomputed
    /// `sha256(count_be ‖ values_be)`. Indicates payload tamper.
    BatchHashMismatch { at_seq: u64 },
    /// Phase 0.8 Item B1 — opening or memory-mapping a snapshot file
    /// failed (e.g. file not found, permission denied, mmap call
    /// rejected by the OS). Carries the `io::ErrorKind` so callers can
    /// branch programmatically, plus a human-readable message. This
    /// variant is surfaced **only** by the `replay_mmap*` entry points;
    /// `replay`/`replay_with_outcome` operate on byte slices and never
    /// touch the filesystem.
    Io {
        kind: std::io::ErrorKind,
        message: String,
    },
    /// Phase 0.8 Item B3 — the snapshot starts with the compressed
    /// `ABNGZ\x01` magic but `cjc-abng` was built without the
    /// `compression` Cargo feature, so the zstd decoder is not linked
    /// in. Callers should rebuild with `--features compression` or
    /// re-serialize the snapshot uncompressed.
    CompressionFeatureDisabled,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::BadMagic => write!(f, "abng: bad magic header"),
            DecodeError::UnexpectedEof => write!(f, "abng: unexpected end of snapshot"),
            DecodeError::UnknownKindTag(t) => write!(f, "abng: unknown kind tag {t:#04x}"),
            DecodeError::UnknownChildrenKind(t) => {
                write!(f, "abng: unknown children kind {t:#04x}")
            }
            DecodeError::EmptyGraph => write!(f, "abng: snapshot has zero nodes"),
            DecodeError::BadCodebook => write!(f, "abng: malformed codebook in snapshot header"),
            DecodeError::ChainMismatch { at_seq } => {
                write!(f, "abng: chain mismatch at seq {at_seq}")
            }
            DecodeError::StatsMismatch { node_id } => {
                write!(f, "abng: stats mismatch on node {node_id}")
            }
            DecodeError::ChildrenMismatch { node_id } => {
                write!(f, "abng: children mismatch on node {node_id}")
            }
            DecodeError::CodebookHashMismatch => write!(f, "abng: codebook hash mismatch"),
            DecodeError::LeafHeadHashMismatch => write!(f, "abng: leaf head hash mismatch"),
            DecodeError::LeafParamsHashMismatch { node_id, at_seq } => write!(
                f,
                "abng: leaf params hash mismatch on node {node_id} at seq {at_seq}"
            ),
            DecodeError::BadLeafHead => write!(f, "abng: malformed leaf head in snapshot header"),
            DecodeError::BadBlrPrior => write!(f, "abng: malformed BLR prior in snapshot header"),
            DecodeError::BlrPriorHashMismatch => write!(f, "abng: BLR prior hash mismatch"),
            DecodeError::BlrStateHashMismatch { node_id, at_seq } => write!(
                f,
                "abng: BLR state hash mismatch on node {node_id} at seq {at_seq}"
            ),
            DecodeError::FinalHashMismatch => write!(f, "abng: final chain head mismatch"),
            DecodeError::DecisionPolicyHashMismatch => {
                write!(f, "abng: decision policy hash mismatch")
            }
            DecodeError::NonMonotonicSeq { expected, got } => write!(
                f,
                "abng: audit seq must be monotonic — expected {expected}, got {got}"
            ),
            DecodeError::EpochMismatch { expected, got } => write!(
                f,
                "abng: event epoch {got} does not match header epoch {expected}"
            ),
            DecodeError::StatsVersionMismatch { node_id, at_seq } => write!(
                f,
                "abng: stats_version mismatch on node {node_id} at seq {at_seq}"
            ),
            DecodeError::CreatedMustBeFirst => write!(
                f,
                "abng: audit log must start with exactly one Created event at seq 0"
            ),
            DecodeError::ProvenanceMismatch { node_id } => write!(
                f,
                "abng: provenance stamp mismatch on node {node_id}"
            ),
            DecodeError::StatsSnapshotMismatch { node_id, at_seq } => write!(
                f,
                "abng: StatsSnapshot stats_hash mismatch on node {node_id} at seq {at_seq}"
            ),
            DecodeError::EmptyBatch => write!(
                f,
                "abng: BeliefUpdateBatch event has count=0; empty batches are rejected"
            ),
            DecodeError::BatchHashMismatch { at_seq } => write!(
                f,
                "abng: BeliefUpdateBatch batch_hash does not match sha256(count ‖ values) at seq {at_seq}"
            ),
            DecodeError::Io { kind, message } => write!(
                f,
                "abng: snapshot I/O error ({kind:?}): {message}"
            ),
            DecodeError::CompressionFeatureDisabled => write!(
                f,
                "abng: snapshot is zstd-compressed but cjc-abng was built without the `compression` feature"
            ),
        }
    }
}

// ─── Encoding ────────────────────────────────────────────────────────────

/// Serialize a graph into the canonical v2 binary snapshot.
pub fn serialize(graph: &AdaptiveBeliefGraph) -> Vec<u8> {
    let mut out = Vec::new();
    // `Vec<u8>` implements `std::io::Write` and its `write_all`
    // implementation never fails (it just `extend_from_slice`s),
    // so the only failure mode here is allocation OOM — which
    // panics in stable Rust regardless of error handling shape.
    serialize_into(graph, &mut out)
        .expect("serialize_into: writes to Vec<u8> are infallible");
    out
}

/// Phase 0.8 Item B2 — stream a snapshot's bytes into any
/// [`std::io::Write`] sink without materializing the full blob.
///
/// Equivalent in output to `w.write_all(&serialize(g))?` but never
/// allocates a `Vec<u8>` sized to the whole snapshot. Per-event audit
/// payloads share a single reused scratch buffer (Phase 0.7 (A)
/// pattern); peak memory is `O(per_event_payload + per_node_section)`
/// rather than `O(snapshot_size)`.
///
/// # Determinism
///
/// For any graph `g`, `serialize_into(&g, w)` writes the same byte
/// sequence as `serialize(&g)` does, regardless of how `w` chunks
/// its underlying `write` calls. The snapshot magic, audit chain
/// hashes, and per-node `canonical_bytes` are unchanged — only the
/// memory pressure during emission differs.
///
/// # Buffering
///
/// Callers writing to a raw `File` should wrap it in
/// [`std::io::BufWriter`]: each `f64` inside a per-node tensor or
/// Welford accumulator is written as an 8-byte chunk, so an
/// unbuffered destination pays one syscall per 8 bytes. The bench
/// in `bench/abng_micro::bench_serialize_streaming_vs_buffered`
/// measures the buffered-vs-Vec comparison directly.
///
/// # Error propagation
///
/// Any `Err(io::Error)` from `w.write_all` short-circuits the
/// remaining writes via `?`. Bytes already written to `w` before
/// the failure remain there; this function does not attempt
/// rollback (the underlying `Write` trait has no rollback
/// primitive).
pub fn serialize_into(
    graph: &AdaptiveBeliefGraph,
    w: &mut dyn Write,
) -> std::io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&graph.seed.to_be_bytes())?;
    w.write_all(&graph.epoch.to_be_bytes())?;
    w.write_all(&graph.chain_head)?;

    // Codebook section.
    match &graph.codebook {
        None => w.write_all(&[0x00])?,
        Some(cb) => {
            w.write_all(&[0x01])?;
            encode_codebook(cb, w)?;
        }
    }

    // Phase 0.3a: leaf head section.
    match &graph.head {
        None => w.write_all(&[0x00])?,
        Some(head) => {
            w.write_all(&[0x01])?;
            encode_head(head, w)?;
        }
    }

    // Phase 0.3b: BLR prior section.
    match &graph.blr_prior {
        None => w.write_all(&[0x00])?,
        Some(prior) => {
            w.write_all(&[0x01])?;
            encode_blr_prior(prior, w)?;
        }
    }

    // Phase 0.3c: density-tracker enable flag + calibration n_bins.
    w.write_all(&[if graph.density_enabled { 0x01 } else { 0x00 }])?;
    match graph.calibration_n_bins {
        None => w.write_all(&[0x00])?,
        Some(n_bins) => {
            w.write_all(&[0x01])?;
            w.write_all(&[n_bins])?;
        }
    }

    // Phase 0.3d-3: decision policy section + action counts.
    match &graph.decision_policy {
        None => w.write_all(&[0x00])?,
        Some(policy) => {
            w.write_all(&[0x01])?;
            w.write_all(&policy.canonical_bytes())?;
            w.write_all(&policy.policy_hash)?;
        }
    }
    for &c in &graph.action_counts {
        w.write_all(&c.to_be_bytes())?;
    }
    // Phase 0.4-extended (v11) — unfreeze_count observability.
    w.write_all(&graph.unfreeze_count.to_be_bytes())?;

    // Nodes.
    w.write_all(&(graph.nodes.len() as u32).to_be_bytes())?;
    for node in &graph.nodes {
        let parent_i32: i32 = match node.parent {
            None => -1,
            Some(p) => p as i32,
        };
        w.write_all(&parent_i32.to_be_bytes())?;
        encode_children(&node.children, w)?;
        w.write_all(&node.stats.canonical_bytes())?;
        w.write_all(&node.stats_version.to_be_bytes())?;
        w.write_all(&node.stats_chain_head)?;
        // Phase 0.3a: per-node params blob.
        w.write_all(&(node.params.len() as u32).to_be_bytes())?;
        for t in &node.params {
            encode_tensor(t, w)?;
        }
        // Phase 0.3b: per-node BLR state blob.
        match &node.blr {
            None => w.write_all(&[0x00])?,
            Some(s) => {
                w.write_all(&[0x01])?;
                encode_blr_state(s, w)?;
            }
        }
        // Phase 0.3c: per-node density / calibration / drift blobs.
        match &node.density {
            None => w.write_all(&[0x00])?,
            Some(d) => {
                w.write_all(&[0x01])?;
                w.write_all(&d.canonical_bytes())?;
            }
        }
        match &node.calibration {
            None => w.write_all(&[0x00])?,
            Some(c) => {
                w.write_all(&[0x01])?;
                w.write_all(&c.canonical_bytes())?;
            }
        }
        match &node.drift_baseline {
            None => w.write_all(&[0x00])?,
            Some(b) => {
                w.write_all(&[0x01])?;
                w.write_all(&b.canonical_bytes())?;
            }
        }
        // Phase 0.3d-2 — per-node expected_epistemic (one f64 if captured).
        match node.expected_epistemic {
            None => w.write_all(&[0x00])?,
            Some(value) => {
                w.write_all(&[0x01])?;
                w.write_all(&value.to_bits().to_be_bytes())?;
            }
        }
        // Phase 0.3d-3 — per-node frozen / active flags.
        w.write_all(&[if node.is_frozen { 0x01 } else { 0x00 }])?;
        w.write_all(&[if node.is_active { 0x01 } else { 0x00 }])?;
        // Phase 0.3d-4 — per-node signature stability state.
        match node.last_signature {
            None => w.write_all(&[0x00])?,
            Some(sig) => {
                w.write_all(&[0x01])?;
                w.write_all(&sig)?;
            }
        }
        w.write_all(&node.signature_stable_calls.to_be_bytes())?;
        // Phase 0.4 Track B-2.2.2 — 3-window stability buffers
        // (snapshot v10). 6 × f64 + 2 × u8 = 50 bytes per node.
        for v in node.ece_history.iter() {
            w.write_all(&v.to_bits().to_be_bytes())?;
        }
        w.write_all(&[node.ece_fill_count])?;
        for v in node.sigma_history.iter() {
            w.write_all(&v.to_bits().to_be_bytes())?;
        }
        w.write_all(&[node.sigma_fill_count])?;
        // Phase 0.4 Track B-2.2.1 — Welford signature accumulators
        // (4 × 24 = 96 bytes per node).
        for welford in [
            &node.welford_prediction,
            &node.welford_uncertainty,
            &node.welford_calibration,
            &node.welford_routing,
        ] {
            w.write_all(&welford.canonical_bytes())?;
        }
        // Phase 0.5 Item 1 (v12) — per-node provenance stamp. 32 bytes
        // appended at the end of the per-node section. `[0u8; 32]` for
        // any node that has not been stamped via `stamp_provenance`.
        w.write_all(&node.provenance_stamp_hash)?;
    }

    // Audit log. Phase 0.7 (A): single payload buffer reused across
    // all events; `write_payload` clears+writes into it on each
    // iteration. Pre-0.7 the loop allocated a fresh Vec<u8> per event
    // via `payload_bytes()`. The length-prefix encoding (u32 + payload)
    // requires knowing the payload size up front, so the scratch
    // buffer remains. Peak memory during this loop is O(largest
    // single event's payload) — typically ≤ 100 bytes.
    w.write_all(&(graph.audit.len() as u64).to_be_bytes())?;
    let mut payload_buf: Vec<u8> = Vec::with_capacity(96);
    for event in &graph.audit {
        event.write_payload(&mut payload_buf);
        w.write_all(&(payload_buf.len() as u32).to_be_bytes())?;
        w.write_all(&payload_buf)?;
        w.write_all(&event.previous_hash)?;
        w.write_all(&event.new_hash)?;
    }
    Ok(())
}

/// Phase 0.8 Item B3 — zstd-compressed counterpart to
/// [`serialize_into`]. Writes the 6-byte compressed magic
/// (`ABNGZ\x01`) followed by a zstd stream containing the same byte
/// sequence that [`serialize_into`] would produce on its own.
///
/// Internally constructs a [`zstd::Encoder`] wrapping `w` and hands
/// it to [`serialize_into`] as a [`Write`] sink. The encoder
/// compresses on the fly; no double-materialization of the
/// uncompressed bytes ever occurs.
///
/// `level` is the zstd compression level (1 = fastest, 22 = best
/// compression). The Phase 0.8 B3 bench at level 3 produces a
/// typical 2-5× shrink on snapshots dominated by audit-event
/// payloads.
///
/// # Determinism
///
/// zstd is deterministic in single-thread mode (the binding's
/// default). For a given graph + level, the compressed bytes are
/// bit-stable across runs. Decompression via [`replay`] /
/// [`replay_with_outcome`] is therefore byte-identical to the
/// uncompressed path on the *replayed graph state*; the
/// **compressed wire bytes** are themselves byte-stable per
/// `(g, level, zstd_version)`.
///
/// Available only under the `compression` Cargo feature.
#[cfg(feature = "compression")]
pub fn serialize_into_compressed(
    graph: &AdaptiveBeliefGraph,
    w: &mut dyn Write,
    level: i32,
) -> std::io::Result<()> {
    w.write_all(COMPRESSED_MAGIC)?;
    let mut encoder = zstd::Encoder::new(w, level)?;
    serialize_into(graph, &mut encoder)?;
    // `finish` flushes any pending compressed bytes and writes the
    // frame epilogue. Without this, the resulting blob is truncated
    // and `zstd::decode_all` returns an error.
    encoder.finish()?;
    Ok(())
}

/// Phase 0.8 Item B3 — zstd-compressed counterpart to [`serialize`].
///
/// Convenience wrapper around [`serialize_into_compressed`] that
/// builds a `Vec<u8>`. Use [`serialize_into_compressed`] directly
/// when piping to a file or socket.
///
/// Available only under the `compression` Cargo feature.
#[cfg(feature = "compression")]
pub fn serialize_compressed(graph: &AdaptiveBeliefGraph, level: i32) -> Vec<u8> {
    let mut out = Vec::new();
    serialize_into_compressed(graph, &mut out, level)
        .expect("serialize_into_compressed: writes to Vec<u8> are infallible");
    out
}

/// Encode a [`LeafHead`] into the snapshot header.
///
/// Layout:
/// ```text
///   input_dim    u32 BE
///   output_dim   u32 BE
///   activation   u8
///   n_hidden     u16 BE
///   hidden_dims  u32 BE × n_hidden
///   config_hash  [u8; 32]
/// ```
fn encode_head(head: &LeafHead, w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(&head.input_dim.to_be_bytes())?;
    w.write_all(&head.output_dim.to_be_bytes())?;
    w.write_all(&[encode_activation_tag(head.activation)])?;
    w.write_all(&(head.hidden_dims.len() as u16).to_be_bytes())?;
    for &h in &head.hidden_dims {
        w.write_all(&h.to_be_bytes())?;
    }
    w.write_all(&head.config_hash)?;
    Ok(())
}

/// Encode a single param `Tensor`. Layout:
/// ```text
///   ndim   u8
///   shape  u32 BE × ndim
///   data   f64 BE × numel
/// ```
fn encode_tensor(t: &Tensor, w: &mut dyn Write) -> std::io::Result<()> {
    let shape = t.shape();
    debug_assert!(shape.len() <= u8::MAX as usize, "tensor ndim overflow");
    w.write_all(&[shape.len() as u8])?;
    for &d in shape {
        w.write_all(&(d as u32).to_be_bytes())?;
    }
    for x in t.to_vec() {
        w.write_all(&x.to_bits().to_be_bytes())?;
    }
    Ok(())
}

/// Encode a [`BlrPrior`]: 24 bytes of canonical params + 32 bytes of
/// hash witness.
fn encode_blr_prior(p: &BlrPrior, w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(&p.canonical_bytes())?;
    w.write_all(&p.config_hash)?;
    Ok(())
}

/// Encode a [`BlrState`]: just the canonical bytes (which already
/// include `d`, mean, precision, a, b, n_seen).
fn encode_blr_state(s: &BlrState, w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(&s.canonical_bytes())?;
    Ok(())
}

fn encode_codebook(cb: &QuantileCodebook, w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(&[cb.n_dims])?;
    w.write_all(&cb.n_bins.to_be_bytes())?;
    for row in &cb.bins {
        w.write_all(&(row.len() as u16).to_be_bytes())?;
        for &b in row {
            w.write_all(&b.to_bits().to_be_bytes())?;
        }
    }
    w.write_all(&cb.frozen_hash)?;
    Ok(())
}

/// Children-kind tag + variant payload.
///
/// Layout per kind:
/// ```text
///   None    : tag=0
///   Node4   : tag=1 + keys[4] (4 bytes) + slots[4] (4 × i32 BE)
///   Node16  : tag=2 + keys[16] (16) + slots[16] (16 × i32 BE)
///   Node48  : tag=3 + index[256] (256) + n_slots u8 (1)
///                   + slots[n_slots] × i32 BE
///   Node256 : tag=4 + slots[256] (256 × i32 BE)
/// ```
fn encode_children(children: &AdaptiveChildren, w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(&[children.kind() as u8])?;
    match children {
        AdaptiveChildren::None => {}
        AdaptiveChildren::Node4 { keys, slots } => {
            w.write_all(keys)?;
            for slot in slots.iter() {
                let v: i32 = slot.map(|id| id as i32).unwrap_or(-1);
                w.write_all(&v.to_be_bytes())?;
            }
        }
        AdaptiveChildren::Node16 { keys, slots } => {
            w.write_all(keys)?;
            for slot in slots.iter() {
                let v: i32 = slot.map(|id| id as i32).unwrap_or(-1);
                w.write_all(&v.to_be_bytes())?;
            }
        }
        AdaptiveChildren::Node48 { index, slots } => {
            w.write_all(&index[..])?;
            w.write_all(&[slots.len() as u8])?;
            for slot in slots.iter() {
                let v: i32 = slot.map(|id| id as i32).unwrap_or(-1);
                w.write_all(&v.to_be_bytes())?;
            }
        }
        AdaptiveChildren::Node256 { slots } => {
            for slot in slots.iter() {
                let v: i32 = slot.map(|id| id as i32).unwrap_or(-1);
                w.write_all(&v.to_be_bytes())?;
            }
        }
        AdaptiveChildren::Dense { signature } => {
            w.write_all(signature)?;
        }
    }
    Ok(())
}

// ─── Decoding ────────────────────────────────────────────────────────────

/// Cursor over a byte slice, returning `UnexpectedEof` when reads exceed
/// the buffer.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.pos + n > self.data.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u16_be(&mut self) -> Result<u16, DecodeError> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }
    fn u32_be(&mut self) -> Result<u32, DecodeError> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    fn i32_be(&mut self) -> Result<i32, DecodeError> {
        let bytes = self.take(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    fn u64_be(&mut self) -> Result<u64, DecodeError> {
        let bytes = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(a))
    }
    fn f64_be(&mut self) -> Result<f64, DecodeError> {
        let bits = self.u64_be()?;
        Ok(f64::from_bits(bits))
    }
    fn hash32(&mut self) -> Result<[u8; 32], DecodeError> {
        let bytes = self.take(32)?;
        let mut a = [0u8; 32];
        a.copy_from_slice(bytes);
        Ok(a)
    }
}

fn decode_density_tracker(cur: &mut Cursor) -> Result<DensityTracker, DecodeError> {
    let d = cur.u32_be()?;
    let n = cur.u64_be()?;
    let dz = d as usize;
    let mut mean_data = Vec::with_capacity(dz);
    for _ in 0..dz {
        mean_data.push(cur.f64_be()?);
    }
    let mut m2_data = Vec::with_capacity(dz);
    for _ in 0..dz {
        m2_data.push(cur.f64_be()?);
    }
    let mean = Tensor::from_vec(mean_data, &[dz]).map_err(|_| DecodeError::UnexpectedEof)?;
    let m2 = Tensor::from_vec(m2_data, &[dz]).map_err(|_| DecodeError::UnexpectedEof)?;
    Ok(DensityTracker { d, n, mean, m2 })
}

fn decode_calibration_bins(cur: &mut Cursor) -> Result<CalibrationBins, DecodeError> {
    let n_bins = cur.u8()?;
    if !(2..=100).contains(&n_bins) {
        return Err(DecodeError::UnexpectedEof);
    }
    let nb = n_bins as usize;
    let mut counts = Vec::with_capacity(nb);
    for _ in 0..nb {
        counts.push(cur.u32_be()?);
    }
    let mut correct_counts = Vec::with_capacity(nb);
    for _ in 0..nb {
        correct_counts.push(cur.u32_be()?);
    }
    let mut conf_sum_bits = Vec::with_capacity(nb);
    for _ in 0..nb {
        conf_sum_bits.push(cur.u64_be()?);
    }
    Ok(CalibrationBins {
        n_bins,
        counts,
        correct_counts,
        conf_sum_bits,
    })
}

fn decode_drift_baseline(cur: &mut Cursor) -> Result<DriftBaseline, DecodeError> {
    let d = cur.u32_be()?;
    let n_at_freeze = cur.u64_be()?;
    let dz = d as usize;
    let mut mean_data = Vec::with_capacity(dz);
    for _ in 0..dz {
        mean_data.push(cur.f64_be()?);
    }
    let mut std_data = Vec::with_capacity(dz);
    for _ in 0..dz {
        std_data.push(cur.f64_be()?);
    }
    let frozen_hash = cur.hash32()?;
    let mean = Tensor::from_vec(mean_data, &[dz]).map_err(|_| DecodeError::UnexpectedEof)?;
    let std = Tensor::from_vec(std_data, &[dz]).map_err(|_| DecodeError::UnexpectedEof)?;
    Ok(DriftBaseline {
        d,
        mean,
        std,
        n_at_freeze,
        frozen_hash,
    })
}

/// Decode a [`DecisionPolicy`] from the v7 header. Layout:
///
/// ```text
///   thresholds   f64 BE × 11    (88 bytes total)
///   policy_hash  [u8; 32]
/// ```
///
/// The recomputed `policy_hash` is checked against the stored value;
/// a mismatch returns [`DecodeError::DecisionPolicyHashMismatch`].
fn decode_decision_policy(cur: &mut Cursor) -> Result<DecisionPolicy, DecodeError> {
    let mut thresholds = [0f64; crate::policy::N_THRESHOLDS];
    for slot in thresholds.iter_mut() {
        *slot = f64::from_bits(cur.u64_be()?);
    }
    let stored_hash = cur.hash32()?;
    let policy = DecisionPolicy::new(&thresholds)
        .map_err(|_| DecodeError::DecisionPolicyHashMismatch)?;
    if policy.policy_hash != stored_hash {
        return Err(DecodeError::DecisionPolicyHashMismatch);
    }
    Ok(policy)
}

fn decode_blr_prior(cur: &mut Cursor) -> Result<BlrPrior, DecodeError> {
    // 24 canonical bytes (precision, a, b) + 32 hash.
    let canon = cur.take(24)?;
    let mut precision_bits = [0u8; 8];
    let mut a_bits = [0u8; 8];
    let mut b_bits = [0u8; 8];
    precision_bits.copy_from_slice(&canon[0..8]);
    a_bits.copy_from_slice(&canon[8..16]);
    b_bits.copy_from_slice(&canon[16..24]);
    let precision = f64::from_bits(u64::from_be_bytes(precision_bits));
    let a = f64::from_bits(u64::from_be_bytes(a_bits));
    let b = f64::from_bits(u64::from_be_bytes(b_bits));
    let stored_hash = cur.hash32()?;
    let prior = BlrPrior::new(precision, a, b).map_err(|_| DecodeError::BadBlrPrior)?;
    if prior.config_hash != stored_hash {
        return Err(DecodeError::BlrPriorHashMismatch);
    }
    Ok(prior)
}

fn decode_blr_state(cur: &mut Cursor) -> Result<BlrState, DecodeError> {
    let d = cur.u32_be()?;
    let dz = d as usize;
    let mut mean_data = Vec::with_capacity(dz);
    for _ in 0..dz {
        mean_data.push(cur.f64_be()?);
    }
    let mut prec_data = Vec::with_capacity(dz * dz);
    for _ in 0..(dz * dz) {
        prec_data.push(cur.f64_be()?);
    }
    let a = cur.f64_be()?;
    let b = cur.f64_be()?;
    let n_seen = cur.u64_be()?;
    // Phase 0.4 Track C-2.3.5 (snapshot v9): feature_version_hash
    // appended to BlrState canonical bytes.
    let feature_version_hash = cur.hash32()?;
    let mean = Tensor::from_vec(mean_data, &[dz]).map_err(|_| DecodeError::UnexpectedEof)?;
    let precision =
        Tensor::from_vec(prec_data, &[dz, dz]).map_err(|_| DecodeError::UnexpectedEof)?;
    Ok(BlrState {
        d,
        mean,
        precision,
        a,
        b,
        n_seen,
        feature_version_hash,
    })
}

fn decode_head(cur: &mut Cursor) -> Result<LeafHead, DecodeError> {
    let input_dim = cur.u32_be()?;
    let output_dim = cur.u32_be()?;
    let activation_tag = cur.u8()?;
    let activation: Activation = decode_activation_tag(activation_tag).ok_or(DecodeError::BadLeafHead)?;
    let n_hidden = cur.u16_be()? as usize;
    let mut hidden_dims = Vec::with_capacity(n_hidden);
    for _ in 0..n_hidden {
        hidden_dims.push(cur.u32_be()?);
    }
    let stored_hash = cur.hash32()?;
    if input_dim == 0 || output_dim == 0 || hidden_dims.iter().any(|&h| h == 0) {
        return Err(DecodeError::BadLeafHead);
    }
    let head = LeafHead::new(input_dim, hidden_dims, output_dim, activation);
    if head.config_hash != stored_hash {
        return Err(DecodeError::LeafHeadHashMismatch);
    }
    Ok(head)
}

fn decode_tensor(cur: &mut Cursor) -> Result<Tensor, DecodeError> {
    let ndim = cur.u8()? as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut numel: usize = 1;
    for _ in 0..ndim {
        let d = cur.u32_be()? as usize;
        numel = numel.saturating_mul(d);
        shape.push(d);
    }
    // Defensive: untrusted shape can drive `numel` to absurd values
    // under fuzz inputs. The cursor's remaining bytes upper-bound
    // the legitimate `numel` (each f64 takes 8 bytes); reject any
    // claim larger than that before allocating.
    let max_numel = cur.data.len().saturating_sub(cur.pos) / 8;
    if numel > max_numel {
        return Err(DecodeError::UnexpectedEof);
    }
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        data.push(cur.f64_be()?);
    }
    Tensor::from_vec(data, &shape).map_err(|_| DecodeError::UnexpectedEof)
}

fn decode_codebook(cur: &mut Cursor) -> Result<QuantileCodebook, DecodeError> {
    let n_dims = cur.u8()?;
    let n_bins = cur.u16_be()?;
    let expected_per_dim = n_bins.saturating_sub(1);
    let mut flat = Vec::with_capacity(n_dims as usize * expected_per_dim as usize);
    for _ in 0..n_dims {
        let n_boundaries = cur.u16_be()?;
        if n_boundaries != expected_per_dim {
            return Err(DecodeError::BadCodebook);
        }
        for _ in 0..n_boundaries {
            flat.push(cur.f64_be()?);
        }
    }
    let stored_hash = cur.hash32()?;
    let cb = QuantileCodebook::from_flat(n_dims as usize, n_bins, &flat)
        .map_err(|_| DecodeError::BadCodebook)?;
    if cb.frozen_hash != stored_hash {
        return Err(DecodeError::CodebookHashMismatch);
    }
    Ok(cb)
}

/// Stored snapshot of a node — the layout/data we expect to reproduce
/// from event replay.
struct StoredNode {
    parent: Option<NodeId>,
    children_kind: ChildrenKind,
    children_pairs: Vec<(u8, NodeId)>,
    /// Phase 0.3d-3 — `Some(signature)` when `children_kind == Dense`,
    /// `None` for every other kind.
    children_dense_signature: Option<[u8; 32]>,
    canonical_bytes: [u8; 32],
    stats_chain_head: [u8; 32],
    /// Phase 0.3a — final per-node params blob. Empty if no head.
    params: Vec<Tensor>,
    /// Phase 0.3b — final per-node BLR state. None if no prior.
    blr: Option<BlrState>,
    /// Phase 0.3c.
    density: Option<DensityTracker>,
    calibration: Option<CalibrationBins>,
    drift_baseline: Option<DriftBaseline>,
    /// Phase 0.3d-2 — per-node training-time epistemic-σ reference.
    expected_epistemic: Option<f64>,
    /// Phase 0.3d-3 — per-node frozen / active flags.
    is_frozen: bool,
    is_active: bool,
    /// Phase 0.3d-4 — per-node signature stability state.
    last_signature: Option<[u8; 32]>,
    signature_stable_calls: u64,
    /// Phase 0.4 Track B-2.2.2 — 3-window stability buffers (v10).
    ece_history: [f64; 3],
    ece_fill_count: u8,
    sigma_history: [f64; 3],
    sigma_fill_count: u8,
    /// Phase 0.4 Track B-2.2.1 — Welford signature accumulators (v10).
    welford_prediction: crate::signature::SignatureWelford,
    welford_uncertainty: crate::signature::SignatureWelford,
    welford_calibration: crate::signature::SignatureWelford,
    welford_routing: crate::signature::SignatureWelford,
    /// Phase 0.5 Item 1 (v12) — per-node provenance stamp. `[0u8; 32]`
    /// for unstamped nodes; otherwise the caller-chosen SHA-256 from
    /// the most-recent `ProvenanceStamped` event for this node.
    provenance_stamp_hash: [u8; 32],
}

fn decode_children(
    cur: &mut Cursor,
) -> Result<(ChildrenKind, Vec<(u8, NodeId)>, Option<[u8; 32]>), DecodeError> {
    let tag = cur.u8()?;
    let kind = ChildrenKind::from_tag(tag).ok_or(DecodeError::UnknownChildrenKind(tag))?;
    let mut pairs: Vec<(u8, NodeId)> = Vec::new();
    let mut dense_sig: Option<[u8; 32]> = None;
    match kind {
        ChildrenKind::None => {}
        ChildrenKind::Node4 => {
            let keys = cur.take(4)?.to_vec();
            for i in 0..4 {
                let v = cur.i32_be()?;
                if v >= 0 {
                    pairs.push((keys[i], v as NodeId));
                }
            }
        }
        ChildrenKind::Node16 => {
            let keys = cur.take(16)?.to_vec();
            for i in 0..16 {
                let v = cur.i32_be()?;
                if v >= 0 {
                    pairs.push((keys[i], v as NodeId));
                }
            }
        }
        ChildrenKind::Node48 => {
            let index_bytes = cur.take(256)?.to_vec();
            let n_slots = cur.u8()? as usize;
            let mut slots: Vec<i32> = Vec::with_capacity(n_slots);
            for _ in 0..n_slots {
                slots.push(cur.i32_be()?);
            }
            for byte in 0u16..=255 {
                let slot_idx = index_bytes[byte as usize];
                if slot_idx != 0xFF {
                    let v = slots
                        .get(slot_idx as usize)
                        .copied()
                        .ok_or(DecodeError::UnexpectedEof)?;
                    if v >= 0 {
                        pairs.push((byte as u8, v as NodeId));
                    }
                }
            }
        }
        ChildrenKind::Node256 => {
            for byte in 0u16..=255 {
                let v = cur.i32_be()?;
                if v >= 0 {
                    pairs.push((byte as u8, v as NodeId));
                }
            }
        }
        ChildrenKind::Dense => {
            let sig = cur.take(32)?;
            let mut s = [0u8; 32];
            s.copy_from_slice(sig);
            dense_sig = Some(s);
        }
    }
    pairs.sort_by_key(|&(k, _)| k);
    Ok((kind, pairs, dense_sig))
}

fn decode_stored_node(cur: &mut Cursor) -> Result<StoredNode, DecodeError> {
    let parent_i32 = cur.i32_be()?;
    let parent = if parent_i32 < 0 {
        None
    } else {
        Some(parent_i32 as NodeId)
    };
    let (children_kind, children_pairs, children_dense_signature) = decode_children(cur)?;
    // Phase 0.5 Item 4 (v12) — canonical_bytes grew from 24 → 32 bytes.
    let canon_slice = cur.take(32)?;
    let mut canonical_bytes = [0u8; 32];
    canonical_bytes.copy_from_slice(canon_slice);
    let _stats_version = cur.u64_be()?;
    let stats_chain_head = cur.hash32()?;
    let n_params = cur.u32_be()? as usize;
    // Defensive: untrusted `n_params` cannot drive a giant allocation;
    // `decode_tensor` will fail fast on unexpected EOF if bogus.
    let mut params = Vec::new();
    for _ in 0..n_params {
        params.push(decode_tensor(cur)?);
    }
    let blr_present = cur.u8()?;
    let blr = match blr_present {
        0 => None,
        1 => Some(decode_blr_state(cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    // Phase 0.3c — three optional blobs.
    let density_present = cur.u8()?;
    let density = match density_present {
        0 => None,
        1 => Some(decode_density_tracker(cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    let calibration_present = cur.u8()?;
    let calibration = match calibration_present {
        0 => None,
        1 => Some(decode_calibration_bins(cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    let drift_present = cur.u8()?;
    let drift_baseline = match drift_present {
        0 => None,
        1 => Some(decode_drift_baseline(cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    // Phase 0.3d-2 — per-node expected_epistemic.
    let ee_present = cur.u8()?;
    let expected_epistemic = match ee_present {
        0 => None,
        1 => {
            let bits = cur.u64_be()?;
            Some(f64::from_bits(bits))
        }
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    // Phase 0.3d-3 — per-node frozen / active flags.
    let is_frozen = match cur.u8()? {
        0 => false,
        1 => true,
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    let is_active = match cur.u8()? {
        0 => false,
        1 => true,
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    // Phase 0.3d-4 — per-node signature stability state.
    let last_signature = match cur.u8()? {
        0 => None,
        1 => {
            let mut s = [0u8; 32];
            s.copy_from_slice(cur.take(32)?);
            Some(s)
        }
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    let signature_stable_calls = cur.u64_be()?;
    // Phase 0.4 Track B-2.2.2 — 3-window stability buffers (v10).
    let mut ece_history = [0.0f64; 3];
    for h in ece_history.iter_mut() {
        *h = cur.f64_be()?;
    }
    let ece_fill_count = cur.u8()?;
    let mut sigma_history = [0.0f64; 3];
    for h in sigma_history.iter_mut() {
        *h = cur.f64_be()?;
    }
    let sigma_fill_count = cur.u8()?;
    // Phase 0.4 Track B-2.2.1 — Welford signature accumulators (v10).
    let welford_prediction = decode_signature_welford(cur)?;
    let welford_uncertainty = decode_signature_welford(cur)?;
    let welford_calibration = decode_signature_welford(cur)?;
    let welford_routing = decode_signature_welford(cur)?;
    // Phase 0.5 Item 1 (v12) — per-node provenance stamp.
    let mut provenance_stamp_hash = [0u8; 32];
    provenance_stamp_hash.copy_from_slice(cur.take(32)?);
    Ok(StoredNode {
        parent,
        children_kind,
        children_pairs,
        children_dense_signature,
        canonical_bytes,
        stats_chain_head,
        params,
        blr,
        density,
        calibration,
        drift_baseline,
        expected_epistemic,
        is_frozen,
        is_active,
        last_signature,
        signature_stable_calls,
        ece_history,
        ece_fill_count,
        sigma_history,
        sigma_fill_count,
        welford_prediction,
        welford_uncertainty,
        welford_calibration,
        welford_routing,
        provenance_stamp_hash,
    })
}

fn decode_signature_welford(
    cur: &mut Cursor,
) -> Result<crate::signature::SignatureWelford, DecodeError> {
    let n_seen = cur.u64_be()?;
    let mean = cur.f64_be()?;
    let m2 = cur.f64_be()?;
    Ok(crate::signature::SignatureWelford { n_seen, mean, m2 })
}

/// Phase 0.5 Item 2 / Phase 0.6 Item 3 — replay options. Controls
/// whether [`replay_with_options`] / [`replay_with_outcome`] should
/// attempt to fast-forward past pre-snapshot `BeliefUpdate` events for
/// nodes that have a [`AuditKind::StatsSnapshot`] marker in the audit
/// log.
///
/// Default (off) reproduces the original naive-replay semantics
/// byte-identically; the `smart_replay` flag is opt-in.
///
/// Determinism contract: `smart_replay = true` MUST produce a
/// `chain_head` and per-node state byte-identical to
/// `smart_replay = false` for all valid blobs. See the property
/// `smart_replay_output_equals_naive_replay` in
/// `tests/prop_tests/abng_decision_props.rs`.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayOptions {
    /// Phase 0.6 Item 3 — fast-forward optimization. When true, a
    /// pre-pass over the audit log identifies nodes whose final state
    /// is fully captured by a `StatsSnapshot` event (no `BeliefUpdate`
    /// events for that node follow the snapshot). For each such node,
    /// pre-snapshot `BeliefUpdate` events skip their per-event
    /// `node.observe(value)` mutation and the per-event
    /// `stats_version` / `stats_hash` checks; the snapshot's recorded
    /// `stats_hash` becomes the consolidated tamper checkpoint for
    /// every skipped event. When the `StatsSnapshot` event itself
    /// applies, `node.stats` is installed directly from
    /// [`crate::stats::NodeStats::from_canonical_bytes`] over the
    /// per-node section's stored `canonical_bytes`.
    ///
    /// Determinism contract: smart-replay output MUST stay
    /// byte-identical to naive replay. The compensating tamper signal
    /// is the StatsSnapshot's own `stats_hash`, which is verified
    /// against the post-install live state by the existing per-event
    /// `stats_hash` check.
    pub smart_replay: bool,
}

/// Phase 0.6 Item 3 — outcome of a replay pass: the rebuilt graph plus
/// the count of `BeliefUpdate` events whose `observe()` mutation was
/// skipped because they sat under a covering `StatsSnapshot`. The
/// count is informational — purely for tests and benchmarks that want
/// to assert the fast-forward layer actually engaged. It is *not*
/// part of the determinism contract.
///
/// `fast_forwarded_events` is always 0 when `smart_replay = false`.
pub struct ReplayOutcome {
    pub graph: AdaptiveBeliefGraph,
    pub fast_forwarded_events: u64,
}

/// Replay a v2 snapshot blob back into a fresh [`AdaptiveBeliefGraph`].
///
/// The replay path does not trust the stored hashes — it recomputes
/// everything from the recorded events and asserts equality.
///
/// Equivalent to [`replay_with_outcome`] with default options;
/// discards the smart-replay instrumentation counter.
pub fn replay(bytes: &[u8]) -> Result<AdaptiveBeliefGraph, DecodeError> {
    Ok(replay_with_outcome(bytes, ReplayOptions::default())?.graph)
}

/// Phase 0.6 Item 3 — replay with the smart-replay fast-forward
/// optimization optionally engaged, returning both the rebuilt graph
/// and the number of fast-forwarded `BeliefUpdate` events.
///
/// When `opts.smart_replay = false`, behaves identically to [`replay`]
/// (and the returned `fast_forwarded_events` is 0). When
/// `opts.smart_replay = true`, performs a pre-pass over the audit log
/// to identify per-node "fast-forwardable" snapshots and skips
/// pre-snapshot `BeliefUpdate` mutations for those nodes. The
/// `StatsSnapshot`'s recorded `stats_hash` is the consolidated tamper
/// checkpoint (see [`ReplayOptions::smart_replay`]).
///
/// Determinism contract: for any *valid* blob,
/// `replay_with_outcome(bytes, opts).graph` produces the same graph
/// as `replay(bytes)` regardless of `opts.smart_replay` (verified by
/// the `smart_replay_output_equals_naive_replay` proptest in
/// `tests/prop_tests/abng_decision_props.rs`).
pub fn replay_with_outcome(
    bytes: &[u8],
    opts: ReplayOptions,
) -> Result<ReplayOutcome, DecodeError> {
    // Phase 0.8 Item B3 — compressed snapshot dispatch. The 6-byte
    // `ABNGZ\x01` magic is detected here; the inner stream is
    // zstd-decoded and fed back through this same function (the
    // decompressed bytes start with the uncompressed v13 magic, so
    // the recursive call falls through to the body below — no
    // infinite recursion risk because the inner blob never carries
    // the compressed magic by construction).
    if bytes.len() >= COMPRESSED_MAGIC.len()
        && &bytes[..COMPRESSED_MAGIC.len()] == COMPRESSED_MAGIC
    {
        #[cfg(feature = "compression")]
        {
            return decompress_and_replay(bytes, opts);
        }
        #[cfg(not(feature = "compression"))]
        {
            return Err(DecodeError::CompressionFeatureDisabled);
        }
    }
    let mut cur = Cursor::new(bytes);

    // Header.
    let magic = cur.take(MAGIC.len())?;
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let seed = cur.u64_be()?;
    let epoch = cur.u64_be()?;
    let stored_final_hash = cur.hash32()?;

    // Codebook section.
    let codebook_present = cur.u8()?;
    let stored_codebook = match codebook_present {
        0 => None,
        1 => Some(decode_codebook(&mut cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };

    // Phase 0.3a: leaf head section.
    let head_present = cur.u8()?;
    let stored_head = match head_present {
        0 => None,
        1 => Some(decode_head(&mut cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };

    // Phase 0.3b: BLR prior section.
    let blr_prior_present = cur.u8()?;
    let stored_blr_prior = match blr_prior_present {
        0 => None,
        1 => Some(decode_blr_prior(&mut cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };

    // Phase 0.3c: graph-wide density-enabled flag + calibration n_bins.
    let stored_density_enabled = match cur.u8()? {
        0 => false,
        1 => true,
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    let stored_calibration_n_bins = match cur.u8()? {
        0 => None,
        1 => Some(cur.u8()?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };

    // Phase 0.3d-3 — decision policy section + action counts.
    let stored_decision_policy = match cur.u8()? {
        0 => None,
        1 => Some(decode_decision_policy(&mut cur)?),
        other => return Err(DecodeError::UnknownChildrenKind(other)),
    };
    let mut stored_action_counts = [0u64; N_ACTION_KINDS];
    for slot in stored_action_counts.iter_mut() {
        *slot = cur.u64_be()?;
    }
    // Phase 0.4-extended (v11) — unfreeze_count observability.
    let stored_unfreeze_count = cur.u64_be()?;

    // Per-node stored layouts.
    let n_nodes = cur.u32_be()?;
    if n_nodes == 0 {
        return Err(DecodeError::EmptyGraph);
    }
    // Defensive: an attacker-controlled (or fuzz-corrupted) blob can
    // claim `n_nodes = u32::MAX`, which would explode `with_capacity`.
    // Use `Vec::new()` and let `decode_stored_node` fail fast on
    // unexpected EOF if the count is bogus.
    let mut stored_nodes: Vec<StoredNode> = Vec::new();
    for _ in 0..n_nodes {
        stored_nodes.push(decode_stored_node(&mut cur)?);
    }

    // Audit log.
    let n_events = cur.u64_be()?;
    // Phase 0.4 Track C-2.3.3 — every well-formed graph has at least
    // one event (the genesis Created). A blob with `n_events == 0` is
    // logically inconsistent: the live root state must come from
    // somewhere, and the only chain anchor for it is `Created`.
    if n_events == 0 {
        return Err(DecodeError::CreatedMustBeFirst);
    }

    // Seed live graph with the root only; replay drives all subsequent
    // mutations.
    let mut graph = AdaptiveBeliefGraph {
        seed,
        epoch,
        nodes: vec![AdaptiveBeliefNode {
            node_id: 0,
            parent: None,
            children: AdaptiveChildren::new(),
            stats: NodeStats::new(),
            stats_version: 0,
            stats_chain_head: genesis_hash(),
            params: Vec::new(),
            blr: None,
            density: None,
            calibration: None,
            drift_baseline: None,
            expected_epistemic: None,
            is_frozen: false,
            is_active: true,
            last_signature: None,
            signature_stable_calls: 0,
            ece_history: [0.0; 3],
            ece_fill_count: 0,
            sigma_history: [0.0; 3],
            sigma_fill_count: 0,
            welford_prediction: crate::signature::SignatureWelford::new(),
            welford_uncertainty: crate::signature::SignatureWelford::new(),
            welford_calibration: crate::signature::SignatureWelford::new(),
            welford_routing: crate::signature::SignatureWelford::new(),
            // Phase 0.5 Item 1 — unstamped at replay seed.
            provenance_stamp_hash: [0u8; 32],
        }],
        // Defensive: untrusted `n_events` cannot drive a giant
        // `with_capacity` allocation; let the vec grow naturally.
        audit: Vec::new(),
        chain_head: genesis_hash(),
        codebook: None,
        head: None,
        blr_prior: None,
        density_enabled: false,
        calibration_n_bins: None,
        decision_policy: None,
        action_counts: [0u64; N_ACTION_KINDS],
        unfreeze_count: 0,
    };

    // Phase 0.6 Item 3 — decode all events into a Vec first, then run
    // a pre-pass (only when `opts.smart_replay`) to identify
    // fast-forwardable nodes, then run the apply pass with skip
    // logic. For `opts.smart_replay = false`, pass 2 is the original
    // streaming loop's body unchanged. The memory overhead is ~150B
    // per event (the decoded `AuditEvent` + 32-byte hash); for
    // typical workloads (small graphs, short logs) this is
    // negligible, and the perf win on compacted logs at scale (the
    // Phase 0.6 motivating case) far dominates.
    //
    // Defensive: untrusted `n_events` cannot drive a giant
    // `with_capacity` allocation; let the vec grow naturally.
    let mut prev_hash = genesis_hash();
    let mut expected_seq: u64 = 0;
    let mut decoded: Vec<(AuditEvent, [u8; 32])> = Vec::new();
    for event_index in 0..n_events {
        let payload_len = cur.u32_be()? as usize;
        let payload = cur.take(payload_len)?;
        let stored_previous = cur.hash32()?;
        let stored_new = cur.hash32()?;

        let event = decode_payload(payload, stored_previous, stored_new)?;

        // Phase 0.4 Track C-2.3.3 — semantic invariants run BEFORE the
        // chain hash check so an adversarial blob whose hashes are
        // internally consistent (attacker recomputed) still surfaces
        // the specific malformation rather than a generic
        // ChainMismatch. A blob that's also chain-corrupt is malformed
        // either way; the more specific error wins.

        // First event must be Created; only the first.
        if event_index == 0 && !matches!(event.kind, AuditKind::Created) {
            return Err(DecodeError::CreatedMustBeFirst);
        }
        if event_index > 0 && matches!(event.kind, AuditKind::Created) {
            return Err(DecodeError::CreatedMustBeFirst);
        }

        // seq must be 0, 1, 2, …
        if event.seq != expected_seq {
            return Err(DecodeError::NonMonotonicSeq {
                expected: expected_seq,
                got: event.seq,
            });
        }

        // epoch must match the header.
        if event.epoch != epoch {
            return Err(DecodeError::EpochMismatch {
                expected: epoch,
                got: event.epoch,
            });
        }

        if event.previous_hash != prev_hash {
            return Err(DecodeError::ChainMismatch { at_seq: event.seq });
        }
        let recomputed_new = AuditEvent::compute_new_hash(&prev_hash, payload);
        if recomputed_new != stored_new {
            return Err(DecodeError::ChainMismatch { at_seq: event.seq });
        }

        decoded.push((event, recomputed_new));
        prev_hash = recomputed_new;
        expected_seq += 1;
    }

    // Phase 0.6 Item 3 — pre-pass: identify fast-forwardable nodes.
    // A node N is fast-forwardable up to seq S iff:
    //   - some `StatsSnapshot { node_id: N }` event exists at seq S, AND
    //   - no `BeliefUpdate` event for node N exists at seq > S.
    // For each such node, pre-snapshot `BeliefUpdate` mutations skip
    // their `observe()` call and the per-event stats checks. Empty
    // BTreeMap when smart_replay is off.
    let ff_until_seq: std::collections::BTreeMap<NodeId, u64> = if opts.smart_replay {
        let mut last_snap: std::collections::BTreeMap<NodeId, u64> =
            std::collections::BTreeMap::new();
        let mut latest_belief: std::collections::BTreeMap<NodeId, u64> =
            std::collections::BTreeMap::new();
        for (event, _) in &decoded {
            match &event.kind {
                AuditKind::StatsSnapshot { node_id, .. } => {
                    last_snap.insert(*node_id, event.seq);
                }
                AuditKind::BeliefUpdate { .. } => {
                    latest_belief.insert(event.node_id, event.seq);
                }
                _ => {}
            }
        }
        let mut ff = std::collections::BTreeMap::new();
        for (node_id, snap_seq) in &last_snap {
            if latest_belief
                .get(node_id)
                .map_or(true, |bu_seq| *bu_seq < *snap_seq)
            {
                ff.insert(*node_id, *snap_seq);
            }
        }
        ff
    } else {
        std::collections::BTreeMap::new()
    };

    // Phase 0.6 Item 3 — apply pass. For fast-forwardable
    // BeliefUpdate events at seq < snapshot_seq we skip the observe
    // mutation AND the per-event stats_version / stats_hash checks
    // (the StatsSnapshot's recorded stats_hash is the consolidated
    // tamper checkpoint). When the StatsSnapshot itself applies for a
    // fast-forwardable node, we install the per-node section's
    // canonical_bytes / stats_chain_head / event.stats_version BEFORE
    // the per-event stats_hash check fires — so the existing check
    // becomes the consolidated tamper detection automatically.
    let mut fast_forwarded_events: u64 = 0;
    for (event, recomputed_new) in decoded {
        let should_skip = opts.smart_replay
            && matches!(event.kind, AuditKind::BeliefUpdate { .. })
            && ff_until_seq
                .get(&event.node_id)
                .map_or(false, |snap_seq| event.seq < *snap_seq);

        if should_skip {
            fast_forwarded_events += 1;
        } else {
            // Apply the event's effect on graph state.
            apply_event(
                &mut graph,
                &event,
                stored_codebook.as_ref(),
                stored_head.as_ref(),
                stored_blr_prior.as_ref(),
                stored_density_enabled,
                stored_calibration_n_bins,
            )?;

            // Phase 0.6 Item 3 — for the StatsSnapshot of a
            // fast-forwardable node, install the consolidated state
            // (canonical_bytes -> stats; event.stats_version;
            // stored.stats_chain_head) BEFORE the per-event stats
            // checks. After install, the existing
            // `event.stats_hash != live_stats_hash` check effectively
            // becomes the consolidated tamper check covering all
            // skipped BeliefUpdate events under this snapshot.
            if opts.smart_replay {
                if let AuditKind::StatsSnapshot { node_id, .. } = &event.kind {
                    if ff_until_seq.contains_key(node_id) {
                        let stored = &stored_nodes[*node_id as usize];
                        graph.nodes[*node_id as usize].stats =
                            crate::stats::NodeStats::from_canonical_bytes(
                                &stored.canonical_bytes,
                            );
                        graph.nodes[*node_id as usize].stats_version =
                            event.stats_version;
                        graph.nodes[*node_id as usize].stats_chain_head =
                            stored.stats_chain_head;
                    }
                }
                // Phase 0.5 Item 2 — StatsSnapshot internal-payload
                // consistency check. The kind carries `stats_hash`
                // inside its payload AND in the event-level slot;
                // both come from the same graph state at compaction
                // time and MUST be equal. Catches a tampered blob
                // that flipped the payload hash without recomputing
                // the chain.
                if let AuditKind::StatsSnapshot {
                    node_id,
                    stats_hash: payload_hash,
                } = &event.kind
                {
                    if *node_id != event.node_id || payload_hash != &event.stats_hash {
                        return Err(DecodeError::StatsSnapshotMismatch {
                            node_id: *node_id,
                            at_seq: event.seq,
                        });
                    }
                }
            }

            // Phase 0.4 Track C-2.3.3 — the event's recorded
            // `stats_version` must match the live node's post-apply
            // `stats_version`. Catches reordered or swapped *Updated
            // events for the same node, where the chain hashes still
            // validate but the per-event sequence numbers no longer
            // match the live state evolution.
            let live_stats_version =
                graph.nodes[event.node_id as usize].stats_version;
            if event.stats_version != live_stats_version {
                return Err(DecodeError::StatsVersionMismatch {
                    node_id: event.node_id,
                    at_seq: event.seq,
                });
            }

            // Verify the per-node stats hash matches what the event recorded.
            let live_stats_hash = graph.nodes[event.node_id as usize].stats.stats_hash();
            if live_stats_hash != event.stats_hash {
                return Err(DecodeError::StatsMismatch {
                    node_id: event.node_id,
                });
            }
        }

        graph.chain_head = recomputed_new;
        graph.audit.push(event);
    }

    // Verify per-node canonical bytes, chain heads, and children layouts.
    // Then install stored params on each node and verify the params hash
    // matches the most-recent LeafParamsInitialized/Updated event for
    // that node (if any).
    for (i, expected) in stored_nodes.iter().enumerate() {
        let live = &graph.nodes[i];
        if live.stats.canonical_bytes() != expected.canonical_bytes {
            return Err(DecodeError::StatsMismatch { node_id: i as NodeId });
        }
        if live.parent != expected.parent {
            return Err(DecodeError::ChildrenMismatch { node_id: i as NodeId });
        }
        if live.children.kind() != expected.children_kind {
            return Err(DecodeError::ChildrenMismatch { node_id: i as NodeId });
        }
        if live.children.iter() != expected.children_pairs {
            return Err(DecodeError::ChildrenMismatch { node_id: i as NodeId });
        }
        if live.stats_chain_head != expected.stats_chain_head {
            return Err(DecodeError::ChildrenMismatch { node_id: i as NodeId });
        }
        // Install the stored params (if any) — these reflect the final
        // post-update state, including any LeafParamsUpdated writes that
        // happened during training.
        graph.nodes[i].params = expected.params.clone();
        if !expected.params.is_empty() {
            // Find the latest LeafParams* event for this node and verify
            // its witness hash matches the stored params.
            let stored_phash =
                crate::leaf_head::params_hash(&graph.nodes[i].params);
            let latest_event_hash = graph
                .audit
                .iter()
                .rev()
                .find_map(|e| {
                    if e.node_id == i as NodeId {
                        match &e.kind {
                            AuditKind::LeafParamsInitialized { params_hash } => {
                                Some((e.seq, *params_hash))
                            }
                            AuditKind::LeafParamsUpdated { params_hash } => {
                                Some((e.seq, *params_hash))
                            }
                            // Phase 0.4 Track C-2.3.6 — batch writeback
                            // is also a valid latest-hash source.
                            AuditKind::LeafParamsUpdatedBatch { params_hash } => {
                                Some((e.seq, *params_hash))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                });
            if let Some((seq, witness)) = latest_event_hash {
                if witness != stored_phash {
                    return Err(DecodeError::LeafParamsHashMismatch {
                        node_id: i as NodeId,
                        at_seq: seq,
                    });
                }
            }
        }
        // Phase 0.3b: install stored BLR state (if any) and verify
        // against most-recent BlrInitialized/BlrUpdated witness.
        graph.nodes[i].blr = expected.blr.clone();
        if let Some(blr) = &graph.nodes[i].blr {
            let stored_shash = blr.state_hash();
            let latest_event_hash = graph
                .audit
                .iter()
                .rev()
                .find_map(|e| {
                    if e.node_id == i as NodeId {
                        match &e.kind {
                            AuditKind::BlrInitialized { state_hash } => {
                                Some((e.seq, *state_hash))
                            }
                            AuditKind::BlrUpdated { state_hash } => {
                                Some((e.seq, *state_hash))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                });
            if let Some((seq, witness)) = latest_event_hash {
                if witness != stored_shash {
                    return Err(DecodeError::BlrStateHashMismatch {
                        node_id: i as NodeId,
                        at_seq: seq,
                    });
                }
            }
        }
        // Phase 0.3c — install stored density / calibration / drift_baseline
        // and verify each against the most-recent witness for that node.
        graph.nodes[i].density = expected.density.clone();
        if let Some(d) = &graph.nodes[i].density {
            let stored = d.state_hash();
            let latest = graph.audit.iter().rev().find_map(|e| {
                if e.node_id == i as NodeId {
                    match &e.kind {
                        AuditKind::DensityTrackerInstalled { state_hash }
                        | AuditKind::DensityUpdated { state_hash } => {
                            Some((e.seq, *state_hash))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            if let Some((seq, w)) = latest {
                if w != stored {
                    return Err(DecodeError::ChainMismatch { at_seq: seq });
                }
            }
        }
        graph.nodes[i].calibration = expected.calibration.clone();
        if let Some(c) = &graph.nodes[i].calibration {
            let stored = c.state_hash();
            let latest = graph.audit.iter().rev().find_map(|e| {
                if e.node_id == i as NodeId {
                    match &e.kind {
                        AuditKind::CalibrationInstalled { state_hash }
                        | AuditKind::CalibrationUpdated { state_hash } => {
                            Some((e.seq, *state_hash))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            if let Some((seq, w)) = latest {
                if w != stored {
                    return Err(DecodeError::ChainMismatch { at_seq: seq });
                }
            }
        }
        graph.nodes[i].drift_baseline = expected.drift_baseline.clone();
        if let Some(b) = &graph.nodes[i].drift_baseline {
            let stored = b.state_hash();
            let latest = graph.audit.iter().rev().find_map(|e| {
                if e.node_id == i as NodeId {
                    match &e.kind {
                        AuditKind::DriftBaselineFrozen { state_hash } => {
                            Some((e.seq, *state_hash))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            if let Some((seq, w)) = latest {
                if w != stored {
                    return Err(DecodeError::ChainMismatch { at_seq: seq });
                }
            }
        }
        // Phase 0.3d-2 — install stored expected_epistemic and verify
        // against the most-recent ExpectedEpistemicCaptured witness.
        graph.nodes[i].expected_epistemic = expected.expected_epistemic;
        if let Some(value) = graph.nodes[i].expected_epistemic {
            let stored = cjc_snap::hash::sha256(&value.to_bits().to_be_bytes());
            let latest = graph.audit.iter().rev().find_map(|e| {
                if e.node_id == i as NodeId {
                    match &e.kind {
                        AuditKind::ExpectedEpistemicCaptured { state_hash } => {
                            Some((e.seq, *state_hash))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            if let Some((seq, w)) = latest {
                if w != stored {
                    return Err(DecodeError::ChainMismatch { at_seq: seq });
                }
            }
        }
        // Phase 0.3d-3 — install per-node frozen / active flags
        // and verify children layout for Dense.
        graph.nodes[i].is_frozen = expected.is_frozen;
        graph.nodes[i].is_active = expected.is_active;
        if expected.children_kind == ChildrenKind::Dense {
            let live_sig = match &graph.nodes[i].children {
                AdaptiveChildren::Dense { signature } => Some(*signature),
                _ => None,
            };
            let stored_sig = expected.children_dense_signature;
            if live_sig != stored_sig {
                return Err(DecodeError::ChildrenMismatch {
                    node_id: i as NodeId,
                });
            }
        }
        // Phase 0.3d-4 — install per-node signature stability state.
        // No witness event for these — they're internal to decide_step
        // and re-derived from observe / structural-mutation events
        // during replay if `decide_step` is re-run after replay. The
        // round-trip preserves them verbatim from the snapshot.
        graph.nodes[i].last_signature = expected.last_signature;
        graph.nodes[i].signature_stable_calls = expected.signature_stable_calls;
        // Phase 0.4 Track B-2.2.2 — install 3-window stability buffers.
        // Same pattern: no witness event; the snapshot is authoritative
        // and `decide_step` calls after replay continue from where the
        // pre-snapshot decide_step calls left off.
        graph.nodes[i].ece_history = expected.ece_history;
        graph.nodes[i].ece_fill_count = expected.ece_fill_count;
        graph.nodes[i].sigma_history = expected.sigma_history;
        graph.nodes[i].sigma_fill_count = expected.sigma_fill_count;
        // Phase 0.4 Track B-2.2.1 — install Welford signature
        // accumulators. Same no-witness-event pattern; the snapshot
        // is authoritative for the running summaries.
        graph.nodes[i].welford_prediction = expected.welford_prediction;
        graph.nodes[i].welford_uncertainty = expected.welford_uncertainty;
        graph.nodes[i].welford_calibration = expected.welford_calibration;
        graph.nodes[i].welford_routing = expected.welford_routing;
        // Phase 0.5 Item 1 (v12) — install provenance stamp. The
        // ProvenanceStamped audit event applied during event replay
        // also writes this field; cross-check that the snapshot's
        // stored value matches what replay produced.
        if graph.nodes[i].provenance_stamp_hash != expected.provenance_stamp_hash {
            return Err(DecodeError::ProvenanceMismatch {
                node_id: i as NodeId,
            });
        }
        graph.nodes[i].provenance_stamp_hash = expected.provenance_stamp_hash;
    }

    if graph.chain_head != stored_final_hash {
        return Err(DecodeError::FinalHashMismatch);
    }

    // Codebook installed during `CodebookFrozen` replay; should now match
    // the snapshot header's codebook.
    match (&graph.codebook, &stored_codebook) {
        (Some(live), Some(stored)) if live.frozen_hash == stored.frozen_hash => {}
        (None, None) => {}
        _ => return Err(DecodeError::CodebookHashMismatch),
    }

    // Same check for the leaf head — installed during `LeafHeadConfigured`
    // replay; live and stored should agree on `config_hash`.
    match (&graph.head, &stored_head) {
        (Some(live), Some(stored)) if live.config_hash == stored.config_hash => {}
        (None, None) => {}
        _ => return Err(DecodeError::LeafHeadHashMismatch),
    }

    // Same check for the BLR prior.
    match (&graph.blr_prior, &stored_blr_prior) {
        (Some(live), Some(stored)) if live.config_hash == stored.config_hash => {}
        (None, None) => {}
        _ => return Err(DecodeError::BlrPriorHashMismatch),
    }

    // Phase 0.3c — graph-wide flags must agree.
    if graph.density_enabled != stored_density_enabled {
        return Err(DecodeError::ChainMismatch {
            at_seq: graph.audit.len() as u64,
        });
    }
    if graph.calibration_n_bins != stored_calibration_n_bins {
        return Err(DecodeError::ChainMismatch {
            at_seq: graph.audit.len() as u64,
        });
    }

    // Phase 0.3d-3 — install decision policy + verify action_counts
    // exactly match what apply_event accumulated during replay.
    graph.decision_policy = stored_decision_policy;
    if graph.action_counts != stored_action_counts {
        return Err(DecodeError::ChainMismatch {
            at_seq: graph.audit.len() as u64,
        });
    }
    // Phase 0.4-extended (v11) — verify unfreeze_count matches what
    // apply_event accumulated. The increment site mirrors the live
    // graph's `unfreeze` method (replay-side `apply_event` for
    // `Unfreeze` flips `is_frozen` AND bumps `unfreeze_count`).
    if graph.unfreeze_count != stored_unfreeze_count {
        return Err(DecodeError::ChainMismatch {
            at_seq: graph.audit.len() as u64,
        });
    }

    Ok(ReplayOutcome {
        graph,
        fast_forwarded_events,
    })
}

/// Phase 0.8 Item B3 — decompress a snapshot that begins with the
/// `ABNGZ\x01` magic and replay the inner uncompressed v13 stream.
///
/// Caller has already verified the magic; this function reads the
/// compressed payload from `bytes[COMPRESSED_MAGIC.len()..]`, hands
/// it to `zstd::decode_all`, and feeds the result back through
/// [`replay_with_outcome`]. Because the decompressed bytes start
/// with the uncompressed v13 magic (`ABNG\x0D`), the recursive call
/// falls through to the existing decoder body — there is no
/// infinite-recursion risk.
///
/// Determinism: zstd is deterministic in single-thread mode (the
/// Rust binding's default). For a given input + compression level,
/// the output bytes are bit-stable across runs and platforms. The
/// `decompress_round_trips_uncompressed` integration test asserts
/// the decompressed inner stream is byte-equal to the original
/// uncompressed snapshot.
#[cfg(feature = "compression")]
fn decompress_and_replay(
    bytes: &[u8],
    opts: ReplayOptions,
) -> Result<ReplayOutcome, DecodeError> {
    let inner = zstd::decode_all(&bytes[COMPRESSED_MAGIC.len()..])
        .map_err(|e| DecodeError::Io {
            kind: e.kind(),
            message: format!("zstd decode: {e}"),
        })?;
    replay_with_outcome(&inner, opts)
}

/// Phase 0.5 Item 2 / Phase 0.6 Item 3 — replay with explicit options.
/// See [`ReplayOptions`] for the flag set.
///
/// With `smart_replay = true`, runs the Phase 0.6 Item 3 fast-forward
/// optimization: pre-snapshot `BeliefUpdate` events on
/// fast-forwardable nodes skip their `observe()` mutation, the
/// StatsSnapshot's stored `stats_hash` is the consolidated tamper
/// checkpoint, and the StatsSnapshot's payload-internal vs
/// event-level `stats_hash` consistency cross-check fires inline.
///
/// Determinism contract: for any *valid* blob,
/// `replay_with_options(bytes, opts)` produces the same graph as
/// `replay(bytes)` regardless of `opts.smart_replay`. For *tampered*
/// blobs that flip a `StatsSnapshot.stats_hash` to a value
/// inconsistent with the per-node section, the smart path errors
/// where the naive path would silently accept.
pub fn replay_with_options(
    bytes: &[u8],
    opts: ReplayOptions,
) -> Result<AdaptiveBeliefGraph, DecodeError> {
    Ok(replay_with_outcome(bytes, opts)?.graph)
}

/// Phase 0.5 Item 2 — opt-in smart-replay entry point.
///
/// Equivalent to [`replay_with_options`] with
/// `ReplayOptions { smart_replay: true }`. The determinism contract
/// is that `smart_replay(bytes)` and `replay(bytes)` produce the
/// same `AdaptiveBeliefGraph` for any valid `bytes` (verified by
/// integration + property tests). For tampered blobs whose
/// `StatsSnapshot.stats_hash` no longer matches the per-node section,
/// smart-replay surfaces a specific
/// [`DecodeError::StatsSnapshotMismatch`].
pub fn smart_replay(bytes: &[u8]) -> Result<AdaptiveBeliefGraph, DecodeError> {
    replay_with_options(
        bytes,
        ReplayOptions {
            smart_replay: true,
        },
    )
}

/// Phase 0.8 Item B1 — memory-mapped snapshot replay.
///
/// Opens `path` read-only, memory-maps it via the OS page cache, and
/// feeds the resulting `&[u8]` into the same decoder used by
/// [`replay`]. The page cache faults pages in lazily as the decoder's
/// `Cursor` advances, so RAM peak during replay is `O(working_set)`
/// rather than `O(snapshot_size)` — typically two orders of magnitude
/// smaller for GB-scale snapshots.
///
/// Determinism contract: for any valid snapshot file `f`,
/// `replay_mmap(f)` returns a graph byte-identical to
/// `replay(&std::fs::read(f)?)`. The byte stream the decoder sees is
/// identical between the two paths; only the memory provenance
/// differs.
///
/// Errors:
/// - [`DecodeError::Io`] if the file cannot be opened or mapped.
/// - All the usual [`DecodeError`] variants if the bytes themselves
///   are malformed.
///
/// Concurrency: snapshots are produced write-once by
/// [`serialize`] and never mutated after publication. Mapping a file
/// that is being concurrently rewritten is undefined behaviour on
/// every supported platform and is the caller's responsibility to
/// avoid.
pub fn replay_mmap(path: &std::path::Path) -> Result<AdaptiveBeliefGraph, DecodeError> {
    Ok(replay_mmap_with_outcome(path, ReplayOptions::default())?.graph)
}

/// Phase 0.8 Item B1 — memory-mapped snapshot replay with smart-replay
/// instrumentation.
///
/// Same semantics as [`replay_with_outcome`] but reads the snapshot
/// bytes from a memory-mapped file rather than a caller-supplied
/// slice. See [`replay_mmap`] for the determinism contract and error
/// model.
pub fn replay_mmap_with_outcome(
    path: &std::path::Path,
    opts: ReplayOptions,
) -> Result<ReplayOutcome, DecodeError> {
    let file = std::fs::File::open(path).map_err(|e| DecodeError::Io {
        kind: e.kind(),
        message: format!("open {}: {e}", path.display()),
    })?;
    // SAFETY: `memmap2::Mmap::map` is `unsafe` because the OS may
    // permit other processes to mutate the underlying file while we
    // hold the mapping, which would race with our reads. ABNG
    // snapshots are produced write-once by `serialize()` and the
    // contract documented on `replay_mmap` requires callers not to
    // concurrently rewrite the file. Under that contract the mapping
    // is sound; if the contract is violated the behaviour is
    // undefined regardless of how we read the file. We treat any I/O
    // failure from the syscall as a fatal `DecodeError::Io`.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| DecodeError::Io {
        kind: e.kind(),
        message: format!("mmap {}: {e}", path.display()),
    })?;
    replay_with_outcome(&mmap[..], opts)
}

/// Apply a single event to a graph being rebuilt during replay.
#[allow(clippy::too_many_arguments)]
fn apply_event(
    graph: &mut AdaptiveBeliefGraph,
    event: &AuditEvent,
    stored_codebook: Option<&QuantileCodebook>,
    stored_head: Option<&LeafHead>,
    stored_blr_prior: Option<&BlrPrior>,
    stored_density_enabled: bool,
    stored_calibration_n_bins: Option<u8>,
) -> Result<(), DecodeError> {
    match &event.kind {
        AuditKind::Created => {
            // Phase 0.1 contract: emitted once for the root at construction
            // time. The seed graph already has the root, so this is a no-op.
        }
        AuditKind::BeliefUpdate { value } => {
            if event.node_id as usize >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[event.node_id as usize].observe(*value);
        }
        AuditKind::NodeAdded { parent, key_byte } => {
            if (*parent as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            // The new node id is the next free arena index, which by the
            // way live append_event sequenced them is *always*
            // graph.nodes.len() at this point.
            let new_id: NodeId = graph.nodes.len() as NodeId;
            if event.node_id != new_id {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*parent as usize]
                .children
                .add_child(*key_byte, new_id);
            let mut child = AdaptiveBeliefNode::new(new_id, Some(*parent), graph.chain_head);
            // If a head is configured (live state, set by an earlier
            // LeafHeadConfigured event), Xavier-init the child's params.
            if let Some(head) = &graph.head {
                child.params = crate::leaf_head::init_params(head, graph.seed, new_id);
            }
            // If a BLR prior is configured, init the child's BLR state.
            if let (Some(head), Some(prior)) = (&graph.head, &graph.blr_prior) {
                let d = crate::graph::blr_feature_dim(head);
                child.blr = Some(BlrState::from_prior(prior, d));
            }
            // Phase 0.3c — density / calibration init.
            if let (true, Some(head)) = (graph.density_enabled, &graph.head) {
                let d = crate::graph::blr_feature_dim(head);
                child.density = Some(DensityTracker::new(d));
            }
            if let Some(n_bins) = graph.calibration_n_bins {
                child.calibration = Some(
                    CalibrationBins::new(n_bins).expect("n_bins validated at install"),
                );
            }
            graph.nodes.push(child);
        }
        AuditKind::ChildrenPromoted { .. } => {
            // Promotion is a *consequence* of the immediately-following
            // NodeAdded event (which calls add_child, which auto-promotes).
            // We don't need to do anything here — verifying the chain
            // hash already proves the parent's children kind transitioned
            // exactly as recorded.
        }
        AuditKind::CodebookFrozen { codebook_hash } => {
            // The codebook bytes live in the header; install now and
            // verify the hash matches the event's witness.
            let cb = stored_codebook
                .cloned()
                .ok_or(DecodeError::CodebookHashMismatch)?;
            if cb.frozen_hash != *codebook_hash {
                return Err(DecodeError::CodebookHashMismatch);
            }
            graph.codebook = Some(cb);
        }
        AuditKind::LeafHeadConfigured { config_hash } => {
            // Install the head and Xavier-initialize the root's params,
            // matching the live `set_leaf_head` path. Subsequent
            // NodeAdded events will see `graph.head.is_some()` and
            // initialize children appropriately.
            let head = stored_head.cloned().ok_or(DecodeError::LeafHeadHashMismatch)?;
            if head.config_hash != *config_hash {
                return Err(DecodeError::LeafHeadHashMismatch);
            }
            graph.nodes[0].params = crate::leaf_head::init_params(&head, graph.seed, 0);
            graph.head = Some(head);
        }
        AuditKind::LeafParamsInitialized { params_hash } => {
            // Witness — verify the live node's params hash matches.
            // The live params were set in NodeAdded (or in LeafHeadConfigured
            // for the root); we just check.
            if (event.node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            let live = crate::leaf_head::params_hash(&graph.nodes[event.node_id as usize].params);
            if live != *params_hash {
                return Err(DecodeError::LeafParamsHashMismatch {
                    node_id: event.node_id,
                    at_seq: event.seq,
                });
            }
        }
        AuditKind::LeafParamsUpdated { .. } => {
            // The new params bytes live in the per-node section (installed
            // *after* event replay, before the final hash check). At this
            // point we can't reproduce the params from the event alone —
            // that's the trade-off for keeping events small. The end-of-
            // replay verifier asserts the stored params match the
            // most-recent LeafParams* event's witness for each node.
        }
        AuditKind::BlrPriorConfigured { config_hash } => {
            // Install the prior and initialize the root's BLR state.
            // Subsequent NodeAdded events will see graph.blr_prior.is_some()
            // and init children appropriately.
            let prior = stored_blr_prior
                .cloned()
                .ok_or(DecodeError::BlrPriorHashMismatch)?;
            if prior.config_hash != *config_hash {
                return Err(DecodeError::BlrPriorHashMismatch);
            }
            let head = graph
                .head
                .as_ref()
                .ok_or(DecodeError::BlrPriorHashMismatch)?;
            let d = crate::graph::blr_feature_dim(head);
            graph.nodes[0].blr = Some(BlrState::from_prior(&prior, d));
            graph.blr_prior = Some(prior);
        }
        AuditKind::BlrInitialized { .. } => {
            if (event.node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            // Phase 0.4 Track C-2.3.5 — `BlrInitialized` fires at install
            // time AND at `reset_blr` time. The pre-0.4 per-event
            // witness check used live state's `state_hash`, but during
            // apply-event-phase the live BLR state doesn't accurately
            // reflect mid-history mutations: `LeafParamsUpdated` is a
            // no-op here, so `params_hash(live params)` stays at the
            // Xavier-init hash even after subsequent leaf-param writes.
            // Computing `feature_version_hash` from those stale params
            // and verifying against a witness recorded at training time
            // (when params had drifted) would always fail for reset_blr
            // events. The pre-event chain hash and the end-of-replay
            // per-node verify (line ~1163) together still catch all
            // tampering: the chain hash covers every event's payload
            // (including state_hash), and the end-of-replay verify
            // checks the installed BLR state against the latest
            // witness. So this branch is now witness-only no-op.
        }
        AuditKind::BlrUpdated { .. } => {
            // Same trade-off as LeafParamsUpdated: the new state lives
            // in the per-node section, not the event payload. End-of-
            // replay verifier asserts most-recent witness matches.
        }
        AuditKind::DensityTrackerInstalled { state_hash } => {
            if (event.node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            // For seq 0 root install, set the graph-wide flag if not yet
            // and initialize the root's density tracker. For child nodes
            // the tracker was set in NodeAdded.
            if event.node_id == 0 && graph.nodes[0].density.is_none() {
                if !stored_density_enabled {
                    return Err(DecodeError::ChainMismatch { at_seq: event.seq });
                }
                let head = graph
                    .head
                    .as_ref()
                    .ok_or(DecodeError::ChainMismatch { at_seq: event.seq })?;
                let d = crate::graph::blr_feature_dim(head);
                graph.nodes[0].density = Some(DensityTracker::new(d));
                graph.density_enabled = true;
            }
            // Verify witness.
            let live = graph.nodes[event.node_id as usize]
                .density
                .as_ref()
                .ok_or(DecodeError::ChainMismatch { at_seq: event.seq })?
                .state_hash();
            if live != *state_hash {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
        }
        AuditKind::DensityUpdated { .. } => {
            // State lives in the per-node section; verifier checks witness.
        }
        AuditKind::CalibrationInstalled { state_hash } => {
            if (event.node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            if event.node_id == 0 && graph.nodes[0].calibration.is_none() {
                let n_bins = stored_calibration_n_bins
                    .ok_or(DecodeError::ChainMismatch { at_seq: event.seq })?;
                graph.nodes[0].calibration = Some(
                    CalibrationBins::new(n_bins)
                        .map_err(|_| DecodeError::ChainMismatch { at_seq: event.seq })?,
                );
                graph.calibration_n_bins = Some(n_bins);
            }
            let live = graph.nodes[event.node_id as usize]
                .calibration
                .as_ref()
                .ok_or(DecodeError::ChainMismatch { at_seq: event.seq })?
                .state_hash();
            if live != *state_hash {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
        }
        AuditKind::CalibrationUpdated { .. } => {
            // State lives in the per-node section.
        }
        AuditKind::DriftBaselineFrozen { .. } => {
            // State lives in the per-node section.
        }
        AuditKind::ExpectedEpistemicCaptured { .. } => {
            // Phase 0.3d-2: captured value lives in the per-node section,
            // not the event payload. End-of-replay verifier checks the
            // most-recent witness against the installed value.
        }
        // ── Phase 0.3d-3 structural-action events ────────────────
        AuditKind::Grow {
            parent,
            key_byte,
            child,
        } => {
            // Reconstruct the same structural mutation force_grow performs.
            // The new child must land at arena index `child` (== n_nodes).
            if (*parent as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            if *child != graph.nodes.len() as NodeId {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*parent as usize]
                .children
                .add_child(*key_byte, *child);
            let mut new_node =
                AdaptiveBeliefNode::new(*child, Some(*parent), graph.chain_head);
            // Mirror force_grow's per-subsystem init order exactly.
            if let Some(head) = stored_head {
                new_node.params = crate::leaf_head::init_params(head, graph.seed, *child);
            }
            if let (Some(head), Some(prior)) = (stored_head, stored_blr_prior) {
                let d = crate::graph::blr_feature_dim(head);
                new_node.blr = Some(BlrState::from_prior(prior, d));
            }
            if let (true, Some(head)) = (stored_density_enabled, stored_head) {
                let d = crate::graph::blr_feature_dim(head);
                new_node.density = Some(DensityTracker::new(d));
            }
            if let Some(n_bins) = stored_calibration_n_bins {
                new_node.calibration = Some(
                    CalibrationBins::new(n_bins)
                        .map_err(|_| DecodeError::ChainMismatch { at_seq: event.seq })?,
                );
            }
            graph.nodes.push(new_node);
            graph.action_counts[crate::graph::ActionKind::Grow as usize] = graph
                .action_counts[crate::graph::ActionKind::Grow as usize]
                .saturating_add(1);
        }
        AuditKind::Split {
            parent,
            child_a,
            child_b,
        } => {
            if (*parent as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            // Reconstruct the deterministic key bytes the same way
            // force_split does.
            let mix = graph
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(*parent as u64);
            let key_a = (mix as u8) & 0xFE;
            let key_b = key_a.wrapping_add(1);
            if *child_a != graph.nodes.len() as NodeId {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*parent as usize].children.add_child(key_a, *child_a);
            graph
                .nodes
                .push(fresh_replay_child(graph, *parent, *child_a, stored_head, stored_blr_prior, stored_density_enabled, stored_calibration_n_bins, event.seq)?);
            if *child_b != graph.nodes.len() as NodeId {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*parent as usize].children.add_child(key_b, *child_b);
            graph
                .nodes
                .push(fresh_replay_child(graph, *parent, *child_b, stored_head, stored_blr_prior, stored_density_enabled, stored_calibration_n_bins, event.seq)?);
            graph.action_counts[crate::graph::ActionKind::Split as usize] = graph
                .action_counts[crate::graph::ActionKind::Split as usize]
                .saturating_add(1);
        }
        AuditKind::Merge { absorbed, into } => {
            if (*absorbed as usize) >= graph.nodes.len()
                || (*into as usize) >= graph.nodes.len()
            {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            // Phase 0.4 Track B-2.2.6 — replay-side merge math. Mirror
            // graph.force_merge so live state matches training-time
            // state when subsequent events fire on `into`.
            //
            // 1. Combine NodeStats. The next event on `into` witnesses
            //    `into.stats.stats_hash()`, so this combine is required
            //    for the chain to verify.
            let absorbed_stats = graph.nodes[*absorbed as usize].stats.clone();
            graph.nodes[*into as usize].stats.combine(&absorbed_stats);
            graph.nodes[*into as usize].stats_version = graph.nodes[*into as usize]
                .stats_version
                .saturating_add(1);

            // 2. Combine BLR posteriors when both nodes carry one.
            if let (Some(prior), true, true) = (
                graph.blr_prior.clone(),
                graph.nodes[*into as usize].blr.is_some(),
                graph.nodes[*absorbed as usize].blr.is_some(),
            ) {
                let absorbed_blr = graph.nodes[*absorbed as usize]
                    .blr
                    .as_ref()
                    .expect("absorbed blr present")
                    .clone();
                let into_blr = graph.nodes[*into as usize]
                    .blr
                    .as_mut()
                    .expect("into blr present");
                if into_blr.combine(&absorbed_blr, &prior).is_err() {
                    return Err(DecodeError::ChainMismatch { at_seq: event.seq });
                }
            }

            // 3. Deactivate absorbed and bump action counter.
            graph.nodes[*absorbed as usize].is_active = false;
            graph.action_counts[crate::graph::ActionKind::Merge as usize] = graph
                .action_counts[crate::graph::ActionKind::Merge as usize]
                .saturating_add(1);
        }
        AuditKind::Prune { node_id } => {
            if (*node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*node_id as usize].is_active = false;
            graph.action_counts[crate::graph::ActionKind::Prune as usize] = graph
                .action_counts[crate::graph::ActionKind::Prune as usize]
                .saturating_add(1);
        }
        AuditKind::Compress { signature } => {
            if (event.node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[event.node_id as usize].children =
                AdaptiveChildren::Dense {
                    signature: *signature,
                };
            graph.action_counts[crate::graph::ActionKind::Compress as usize] = graph
                .action_counts[crate::graph::ActionKind::Compress as usize]
                .saturating_add(1);
        }
        AuditKind::Freeze { node_id } => {
            if (*node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*node_id as usize].is_frozen = true;
            graph.action_counts[crate::graph::ActionKind::Freeze as usize] = graph
                .action_counts[crate::graph::ActionKind::Freeze as usize]
                .saturating_add(1);
        }
        AuditKind::Unfreeze { node_id } => {
            // Phase 0.3d-4 — Unfreeze does not bump action_counts (the
            // counters track structural mutations, not state-flag flips).
            if (*node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*node_id as usize].is_frozen = false;
            // Phase 0.4-extended (v11) — observability counter. Mirrors
            // the live `unfreeze` method's bump path so replay produces
            // a graph whose `unfreeze_count` equals what was originally
            // serialized.
            graph.unfreeze_count = graph.unfreeze_count.saturating_add(1);
        }
        AuditKind::BlrNumericalRescue { .. } => {
            // Phase 0.4 Track C-2.3.4 — diagnostic-only event. The
            // state-changing event for the same update is the
            // immediately-preceding `BlrUpdated`, which already
            // advanced the BLR posterior. No state mutation here; the
            // rescue event is metadata about the prior update.
        }
        AuditKind::LeafParamsUpdatedBatch { .. } => {
            // Phase 0.4 Track C-2.3.6 — hash-witness event. The actual
            // params blob lives in the per-node section of the
            // snapshot; the post-replay verify loop matches the
            // params_hash carried here against the reconstructed live
            // params (same path that handles `LeafParamsUpdated`).
        }
        AuditKind::Routed { .. } => {
            // Phase 0.4 Track A — descend trace event. Read-only
            // origin; no graph state mutation. Replay advances the
            // chain (the canonical-payload bytes determine new_hash)
            // but doesn't touch graph topology, BLR, or stats.
        }
        AuditKind::StatsSnapshot { .. } => {
            // Phase 0.4 Track A — log-compaction marker. Phase 0.4
            // ships only the marker; smart-replay (fast-forward past
            // *Updated runs to the StatsSnapshot) is deferred to
            // Phase 0.5. Apply_event is a pure no-op for now — the
            // chain advances via the canonical-payload bytes, but no
            // graph state mutation happens.
        }
        AuditKind::ProvenanceStamped { node_id, hash } => {
            // Phase 0.5 Item 1 — write the stamp into the target node.
            // The cross-check against the snapshot's stored value runs
            // after event replay (in the per-node verification loop).
            if (*node_id as usize) >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            graph.nodes[*node_id as usize].provenance_stamp_hash = *hash;
        }
        AuditKind::BeliefUpdateBatch {
            count,
            batch_hash,
            values,
        } => {
            // Phase 0.6 Item 4 — apply the batched observations.
            //
            // Validation: count == values.len() (decoded), count >= 1
            // (decoder rejects 0), batch_hash matches the recomputed
            // sha256 of count_be ‖ values_be.
            //
            // State change: apply N Welford observes to node.stats in
            // row order (preserves Kahan determinism), bump
            // stats_version by 1, advance stats_chain ONCE. The
            // single chain advance is the wire-format contract — a
            // round-trip from an observe_batch-built graph back into
            // the audit log must produce one BeliefUpdateBatch event,
            // not N BeliefUpdate events.
            if event.node_id as usize >= graph.nodes.len() {
                return Err(DecodeError::ChainMismatch { at_seq: event.seq });
            }
            if *count == 0 || values.len() as u32 != *count {
                return Err(DecodeError::EmptyBatch);
            }
            // Recompute batch_hash and verify.
            let mut buf = Vec::with_capacity(4 + 8 * values.len());
            buf.extend_from_slice(&count.to_be_bytes());
            for v in values {
                buf.extend_from_slice(&v.to_bits().to_be_bytes());
            }
            let recomputed_batch_hash = cjc_snap::hash::sha256(&buf);
            if recomputed_batch_hash != *batch_hash {
                return Err(DecodeError::BatchHashMismatch {
                    at_seq: event.seq,
                });
            }
            let node = &mut graph.nodes[event.node_id as usize];
            // Single helper — applies N Welford observes in row order,
            // bumps stats_version by 1, advances stats_chain once.
            // Matches what `Graph::observe_batch` did at emit time.
            node.observe_batch_apply(values);
        }
    }
    Ok(())
}

/// Build a fresh child node during replay, mirroring
/// `AdaptiveBeliefGraph::fresh_child_for` exactly. Lifted here so
/// `apply_event` doesn't need a `&mut Graph` reference into the new
/// node's storage during construction.
fn fresh_replay_child(
    graph: &AdaptiveBeliefGraph,
    parent: NodeId,
    new_id: NodeId,
    stored_head: Option<&LeafHead>,
    stored_blr_prior: Option<&BlrPrior>,
    stored_density_enabled: bool,
    stored_calibration_n_bins: Option<u8>,
    seq: u64,
) -> Result<AdaptiveBeliefNode, DecodeError> {
    let mut child = AdaptiveBeliefNode::new(new_id, Some(parent), graph.chain_head);
    if let Some(head) = stored_head {
        child.params = crate::leaf_head::init_params(head, graph.seed, new_id);
    }
    if let (Some(head), Some(prior)) = (stored_head, stored_blr_prior) {
        let d = crate::graph::blr_feature_dim(head);
        child.blr = Some(BlrState::from_prior(prior, d));
    }
    if let (true, Some(head)) = (stored_density_enabled, stored_head) {
        let d = crate::graph::blr_feature_dim(head);
        child.density = Some(DensityTracker::new(d));
    }
    if let Some(n_bins) = stored_calibration_n_bins {
        child.calibration = Some(
            CalibrationBins::new(n_bins)
                .map_err(|_| DecodeError::ChainMismatch { at_seq: seq })?,
        );
    }
    Ok(child)
}

fn decode_payload(
    payload: &[u8],
    previous_hash: [u8; 32],
    new_hash: [u8; 32],
) -> Result<AuditEvent, DecodeError> {
    let mut cur = Cursor::new(payload);
    let seq = cur.u64_be()?;
    let epoch = cur.u64_be()?;
    let node_id = cur.u32_be()?;
    let tag = cur.u8()?;
    let kind = match tag {
        0x00 => AuditKind::Created,
        0x01 => {
            let value_bits = cur.u64_be()?;
            AuditKind::BeliefUpdate {
                value: f64::from_bits(value_bits),
            }
        }
        0x02 => {
            let parent = cur.u32_be()?;
            let key_byte = cur.u8()?;
            AuditKind::NodeAdded { parent, key_byte }
        }
        0x03 => {
            let from = cur.u8()?;
            let to = cur.u8()?;
            AuditKind::ChildrenPromoted { from, to }
        }
        0x04 => {
            let codebook_hash = cur.hash32()?;
            AuditKind::CodebookFrozen { codebook_hash }
        }
        0x05 => {
            let config_hash = cur.hash32()?;
            AuditKind::LeafHeadConfigured { config_hash }
        }
        0x06 => {
            let params_hash = cur.hash32()?;
            AuditKind::LeafParamsInitialized { params_hash }
        }
        0x07 => {
            let params_hash = cur.hash32()?;
            AuditKind::LeafParamsUpdated { params_hash }
        }
        0x08 => {
            let config_hash = cur.hash32()?;
            AuditKind::BlrPriorConfigured { config_hash }
        }
        0x09 => {
            let state_hash = cur.hash32()?;
            AuditKind::BlrInitialized { state_hash }
        }
        0x0A => {
            let state_hash = cur.hash32()?;
            AuditKind::BlrUpdated { state_hash }
        }
        0x0B => {
            let state_hash = cur.hash32()?;
            AuditKind::DensityTrackerInstalled { state_hash }
        }
        0x0C => {
            let state_hash = cur.hash32()?;
            AuditKind::DensityUpdated { state_hash }
        }
        0x0D => {
            let state_hash = cur.hash32()?;
            AuditKind::CalibrationInstalled { state_hash }
        }
        0x0E => {
            let state_hash = cur.hash32()?;
            AuditKind::CalibrationUpdated { state_hash }
        }
        0x0F => {
            let state_hash = cur.hash32()?;
            AuditKind::DriftBaselineFrozen { state_hash }
        }
        0x17 => {
            let state_hash = cur.hash32()?;
            AuditKind::ExpectedEpistemicCaptured { state_hash }
        }
        // Phase 0.3d-3 structural-action events.
        0x10 => {
            let parent = cur.u32_be()?;
            let key_byte = cur.u8()?;
            let child = cur.u32_be()?;
            AuditKind::Grow {
                parent,
                key_byte,
                child,
            }
        }
        0x11 => {
            let parent = cur.u32_be()?;
            let child_a = cur.u32_be()?;
            let child_b = cur.u32_be()?;
            AuditKind::Split {
                parent,
                child_a,
                child_b,
            }
        }
        0x12 => {
            let absorbed = cur.u32_be()?;
            let into = cur.u32_be()?;
            AuditKind::Merge { absorbed, into }
        }
        0x13 => {
            let node_id = cur.u32_be()?;
            AuditKind::Prune { node_id }
        }
        0x14 => {
            let signature = cur.hash32()?;
            AuditKind::Compress { signature }
        }
        0x15 => {
            let node_id = cur.u32_be()?;
            AuditKind::Freeze { node_id }
        }
        0x16 => {
            let node_id = cur.u32_be()?;
            AuditKind::Unfreeze { node_id }
        }
        0x18 => {
            let reason = cur.u8()?;
            let b_pre_clamp_bits = cur.u64_be()?;
            AuditKind::BlrNumericalRescue {
                reason,
                b_pre_clamp_bits,
            }
        }
        0x19 => {
            let params_hash = cur.hash32()?;
            AuditKind::LeafParamsUpdatedBatch { params_hash }
        }
        0x1B => {
            // Phase 0.4 Track A — Routed (opt-in trace event).
            let leaf = cur.u32_be()?;
            let matched_prefix = cur.u8()?;
            AuditKind::Routed { leaf, matched_prefix }
        }
        0x1A => {
            // Phase 0.4 Track A — StatsSnapshot (log-compaction marker).
            let node_id = cur.u32_be()?;
            let stats_hash = cur.hash32()?;
            AuditKind::StatsSnapshot { node_id, stats_hash }
        }
        0x1C => {
            // Phase 0.5 Item 1 — ProvenanceStamped. 36-byte body.
            let node_id = cur.u32_be()?;
            let hash = cur.hash32()?;
            AuditKind::ProvenanceStamped { node_id, hash }
        }
        0x1D => {
            // Phase 0.6 Item 4 — BeliefUpdateBatch. Variable body:
            //   count u32 BE (4)
            //   values f64×count (8 * count, each .to_bits().to_be_bytes())
            //   batch_hash [u8; 32] (32)
            let count = cur.u32_be()?;
            // Defensive: reject empty or absurdly large batches at
            // the boundary. An adversarial blob claiming
            // `count = u32::MAX` would otherwise drive `Vec::new`
            // growth past the cursor's remaining bytes; cur.f64_be()
            // would error first, but we add an explicit guard so the
            // surfaced error names the problem.
            if count == 0 {
                return Err(DecodeError::EmptyBatch);
            }
            // 8 * count must not overflow + must not exceed remaining
            // cursor space. Use `Vec::new()` (not `with_capacity`) so
            // an adversarial `count` cannot drive a giant pre-allocation.
            let mut values: Vec<f64> = Vec::new();
            for _ in 0..count {
                values.push(cur.f64_be()?);
            }
            let batch_hash = cur.hash32()?;
            AuditKind::BeliefUpdateBatch {
                count,
                batch_hash,
                values,
            }
        }
        other => return Err(DecodeError::UnknownKindTag(other)),
    };
    let stats_version = cur.u64_be()?;
    let stats_hash = cur.hash32()?;
    Ok(AuditEvent {
        seq,
        epoch,
        node_id,
        kind,
        stats_version,
        stats_hash,
        previous_hash,
        new_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_graph() -> AdaptiveBeliefGraph {
        let mut g = AdaptiveBeliefGraph::new(123);
        for i in 0..5 {
            g.observe(0, (i as f64) * 1.5).unwrap();
        }
        g
    }

    fn build_multinode() -> AdaptiveBeliefGraph {
        let mut g = AdaptiveBeliefGraph::new(7);
        let a = g.add_node(0, 1).unwrap();
        let b = g.add_node(0, 2).unwrap();
        let _c = g.add_node(a, 3).unwrap();
        for v in [1.0, 2.0, 3.0] {
            g.observe(a, v).unwrap();
        }
        for v in [10.0, 20.0] {
            g.observe(b, v).unwrap();
        }
        g
    }

    #[test]
    fn round_trip_simple() {
        let g = build_simple_graph();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
    }

    #[test]
    fn round_trip_byte_identical() {
        let g = build_simple_graph();
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        assert_eq!(blob1, blob2);
    }

    #[test]
    fn round_trip_multinode_byte_identical() {
        let g = build_multinode();
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        assert_eq!(blob1, blob2);
    }

    #[test]
    fn round_trip_with_codebook() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_codebook(2, 4, &[0.5, 1.5, 2.5, 0.5, 1.5, 2.5]).unwrap();
        g.observe(0, 1.0).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert!(g2.codebook.is_some());
        assert_eq!(
            g.codebook.as_ref().unwrap().frozen_hash,
            g2.codebook.as_ref().unwrap().frozen_hash
        );
    }

    #[test]
    fn bad_magic_rejected() {
        let g = build_simple_graph();
        let mut blob = serialize(&g);
        blob[0] = b'X';
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v1_magic_rejected() {
        let mut blob = serialize(&build_simple_graph());
        blob[4] = 0x01; // ABNG\x01 — Phase 0.1 magic
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn truncated_blob_rejected() {
        let g = build_multinode();
        let blob = serialize(&g);
        let truncated = &blob[..blob.len() / 2];
        assert!(matches!(replay(truncated), Err(DecodeError::UnexpectedEof)));
    }

    #[test]
    fn determinism_blob_equality() {
        let g1 = build_multinode();
        let g2 = build_multinode();
        assert_eq!(serialize(&g1), serialize(&g2));
    }

    #[test]
    fn promotion_round_trip() {
        // Force at least one promotion (Node4 -> Node16) by adding 5 children.
        let mut g = AdaptiveBeliefGraph::new(0);
        for k in 0u8..5 {
            g.add_node(0, k).unwrap();
        }
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.nodes[0].children.kind(), g2.nodes[0].children.kind());
        assert_eq!(g.chain_head, g2.chain_head);
    }

    // ── Phase 0.4-extended (v11) — snapshot v11 ────────────────────

    #[test]
    fn v13_magic_in_blob() {
        // Phase 0.6 Item 4 — wire format v13 (`\x0D`). Bumped from
        // v12 to absorb the new `BeliefUpdateBatch` audit kind
        // (tag 0x1D).
        let g = AdaptiveBeliefGraph::new(0);
        let blob = serialize(&g);
        assert_eq!(&blob[..5], b"ABNG\x0D");
    }

    #[test]
    fn v12_magic_rejected() {
        // After the v12 → v13 bump (Phase 0.6 Item 4:
        // BeliefUpdateBatch audit kind at tag 0x1D), v12 blobs are no
        // longer accepted.
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x0C;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v11_magic_rejected() {
        // After the v11 → v12 bump (Items A + B: DecisionPolicy gained
        // ece_stability_max_delta + sigma_stability_ratio thresholds;
        // graph header gained unfreeze_count u64), v11 blobs are no
        // longer accepted.
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x0B;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v10_magic_rejected() {
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x0A;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v9_magic_rejected() {
        // After the v9 → v10 bump (B-2.2.7 + B-2.2.{1,2}: DecisionPolicy
        // gained drift_unfreeze threshold and per-node sections gained
        // ring buffers + Welford signature state), Phase 0.4 Track C
        // blobs are no longer accepted.
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x09;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v8_magic_rejected() {
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x08;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v7_magic_rejected() {
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x07;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v6_magic_rejected() {
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x06;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    #[test]
    fn v5_magic_rejected() {
        let g = AdaptiveBeliefGraph::new(0);
        let mut blob = serialize(&g);
        blob[4] = 0x05;
        assert_eq!(replay(&blob).unwrap_err(), DecodeError::BadMagic);
    }

    fn build_graph_with_expected_epistemic() -> AdaptiveBeliefGraph {
        let mut g = AdaptiveBeliefGraph::new(7);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g.set_expected_epistemic(0, 0.42).unwrap();
        g
    }

    #[test]
    fn round_trip_preserves_expected_epistemic() {
        let g = build_graph_with_expected_epistemic();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert_eq!(g.nodes[0].expected_epistemic, g2.nodes[0].expected_epistemic);
        assert_eq!(g2.nodes[0].expected_epistemic, Some(0.42));
    }

    #[test]
    fn round_trip_byte_identical_with_expected_epistemic() {
        let g = build_graph_with_expected_epistemic();
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        assert_eq!(blob1, blob2);
    }

    #[test]
    fn round_trip_uncaptured_node_stays_none() {
        // Node 0 captures, child node does not. Replay preserves both.
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        g.set_expected_epistemic(0, 0.7).unwrap();
        let c = g.add_node(0, 9).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g2.nodes[0].expected_epistemic, Some(0.7));
        assert_eq!(g2.nodes[c as usize].expected_epistemic, None);
    }

    // ── Phase 0.3d-3 round-trips ─────────────────────────────────

    fn ok_thresholds() -> [f64; 14] {
        [
            0.5, 64.0, 128.0, 0.05, 0.02,
            4.0, 0.1, 32.0, 10.0, 8.0,
            20.0, f64::MAX, // drift_unfreeze disabled
            0.005, 1.05,    // ece_stability_max_delta + sigma_stability_ratio (v11)
        ]
    }

    #[test]
    fn round_trip_with_decision_policy() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert_eq!(
            g.decision_policy.as_ref().map(|p| p.policy_hash),
            g2.decision_policy.as_ref().map(|p| p.policy_hash)
        );
    }

    #[test]
    fn round_trip_with_force_grow_then_freeze() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let c = g.force_grow(0, 7).unwrap();
        g.force_freeze(c).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert!(g2.is_frozen(c).unwrap());
        assert_eq!(g2.action_counts, g.action_counts);
    }

    #[test]
    fn round_trip_with_force_split() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let (a, b) = g.force_split(0).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert_eq!(g2.node_count(), g.node_count());
        assert_eq!(g2.action_counts[crate::graph::ActionKind::Split as usize], 1);
        assert!(a < b);
    }

    #[test]
    fn round_trip_with_force_compress_dense_signature() {
        let mut g = AdaptiveBeliefGraph::new(0);
        // Force-compress an empty leaf — produces a Dense with a known
        // signature for the empty-children/empty-subsystem state.
        g.force_compress(0).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        match (&g.nodes[0].children, &g2.nodes[0].children) {
            (
                AdaptiveChildren::Dense { signature: a },
                AdaptiveChildren::Dense { signature: b },
            ) => assert_eq!(a, b),
            _ => panic!("expected Dense in both"),
        }
    }

    #[test]
    fn round_trip_with_force_prune() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let c = g.add_node(0, 5).unwrap();
        g.force_prune(c).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert!(!g2.is_active(c).unwrap());
    }

    #[test]
    fn round_trip_with_force_merge() {
        let mut g = AdaptiveBeliefGraph::new(0);
        let a = g.add_node(0, 5).unwrap();
        let b = g.add_node(0, 7).unwrap();
        g.force_merge(a, b).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert!(!g2.is_active(a).unwrap());
        assert!(g2.is_active(b).unwrap());
    }

    #[test]
    fn round_trip_byte_identical_with_full_p3d3_stack() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        let _c1 = g.force_grow(0, 1).unwrap();
        let _c2 = g.force_grow(0, 2).unwrap();
        g.force_freeze(0).unwrap();
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        assert_eq!(blob1, blob2);
    }

    // ── Phase 0.3d-4 round-trips ─────────────────────────────────

    #[test]
    fn round_trip_with_decide_step_stability_state() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        // Drive several decide_step passes to populate stability state.
        for _ in 0..3 {
            g.decide_step();
        }
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert_eq!(g2.nodes[0].last_signature, g.nodes[0].last_signature);
        assert_eq!(
            g2.nodes[0].signature_stable_calls,
            g.nodes[0].signature_stable_calls
        );
    }

    #[test]
    fn round_trip_with_unfreeze() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_freeze(0).unwrap();
        g.unfreeze(0).unwrap();
        let blob = serialize(&g);
        let g2 = replay(&blob).unwrap();
        assert_eq!(g.chain_head, g2.chain_head);
        assert!(!g2.is_frozen(0).unwrap());
        // Action counts unchanged by unfreeze.
        assert_eq!(g.action_counts, g2.action_counts);
    }

    #[test]
    fn round_trip_byte_identical_with_decide_step() {
        let mut g = AdaptiveBeliefGraph::new(7);
        g.set_decision_policy(&ok_thresholds()).unwrap();
        for _ in 0..70 {
            g.observe(0, 1.0).unwrap();
        }
        for _ in 0..3 {
            g.decide_step();
        }
        let blob1 = serialize(&g);
        let g2 = replay(&blob1).unwrap();
        let blob2 = serialize(&g2);
        assert_eq!(blob1, blob2);
    }

    #[test]
    fn replay_rejects_mismatched_action_counts() {
        // Synthesize a v7 blob, tamper with the action_counts bytes,
        // and expect replay to detect the mismatch via the
        // verifier's stored-vs-applied comparison.
        let mut g = AdaptiveBeliefGraph::new(0);
        g.force_grow(0, 1).unwrap();
        let mut blob = serialize(&g);
        // The action_counts table follows the policy section's
        // presence flag (= 0 here) at a known offset. We don't need
        // an exact offset to tamper: scan for the first occurrence of
        // a `01` byte (Grow count = 1) preceded by 7 zero bytes and
        // bump it. Easier: just flip the last byte of the first u64
        // by walking from the calibration n_bins flag. Simpler still:
        // search for the sequence that *must* contain action_counts
        // and corrupt it.
        //
        // Pragmatic approach: force_grow bumps Grow count to 1, so
        // the action_counts blob is `[0, 0, 0, 0, 0, 0, 0, 1, 0...x40]`.
        // Find that marker and flip it.
        let needle = [0u8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let pos = blob
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("action_counts marker not found");
        blob[pos + 7] = 99; // bump Grow count from 1 → 99
        let err = replay(&blob).unwrap_err();
        assert!(matches!(err, DecodeError::ChainMismatch { .. }));
    }
}
