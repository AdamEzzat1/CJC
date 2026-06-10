//! Compression plans: a deterministically-ordered collection of candidates
//! plus the executor that runs each through its kind-appropriate
//! compressor and validates the result.
//!
//! ## What a plan does
//!
//! A [`CompressionPlan`] is a `Vec<CompressionCandidate>` that has been
//! sorted by candidate ID at construction time. The sort guarantees
//! [`CompressionPlan::execute`] visits entries in the same order across
//! runs, which is the foundation for byte-identical
//! [`crate::CompressionReport::report_hash`] outputs.
//!
//! For each entry:
//!
//! - **`LosslessTrace`** → run [`crate::lossless_trace::lossless_compress_bytes`],
//!   then `decompress_bytes`, then assert the reconstructed bytes hash
//!   equals the input hash. Round-trip failure → `MalformedRoundTrip`.
//! - **`MotifDictionary`** → run
//!   [`crate::motif_dictionary::compress_motif_dictionary`] +
//!   `decompress_motif_dictionary`, same hash-equality check.
//! - **`LowRankAdvisory`** → decode `(rows, cols, max_rank, matrix)` from
//!   the payload via [`decode_low_rank_payload`], call
//!   [`crate::lowrank::compress_low_rank`], compare observed
//!   `frobenius_error` against the candidate's declared tolerance.
//! - **`TensorTrainAdvisory`** → decode `(shape, max_bond, tensor)`, call
//!   [`crate::tensor_train::compress_tensor_train`], same tolerance
//!   check.
//!
//! Lossy compressors' tolerance budget comes from
//! [`Criticality::AdvisoryOnly::tolerance_f`](crate::candidate::Criticality::AdvisoryOnly)
//! — the candidate already carries it, so the executor doesn't take a
//! separate budget parameter.
//!
//! ## Payload encoding for lossy candidates
//!
//! The opaque `Vec<u8>` in
//! [`CompressionCandidate::new`](crate::candidate::CompressionCandidate::new)
//! must, for lossy kinds, encode the matrix or tensor shape. The two
//! free functions [`encode_low_rank_payload`] / [`encode_tensor_train_payload`]
//! produce the canonical bytes, and the executor decodes them.

use cjc_cana::hash::hash_bytes;

use crate::candidate::{CompressionCandidate, CompressionError, CompressionKind};
use crate::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};
use crate::lowrank::{compress_low_rank, LowRankPayload};
use crate::motif_dictionary::{compress_motif_dictionary, decompress_motif_dictionary};
use crate::report::{CompressionReport, EntryStatus, ReportEntry};
use crate::tensor_train::{compress_tensor_train, TensorTrainPayload};

// ---------------------------------------------------------------------------
// PlanEntry
// ---------------------------------------------------------------------------

/// One candidate in a [`CompressionPlan`], plus its 0-based slot index
/// (assigned at plan-construction time, preserved across `execute()`).
#[derive(Debug, Clone)]
pub struct PlanEntry {
    /// Slot index in the plan, assigned in sorted-by-ID order. Stable
    /// across executor invocations; safe to use as a join key.
    pub slot: u32,
    /// The candidate itself.
    pub candidate: CompressionCandidate,
}

// ---------------------------------------------------------------------------
// CompressionPlan
// ---------------------------------------------------------------------------

/// A deterministically-ordered list of candidates ready for execution.
///
/// Construct via [`CompressionPlan::new`]. The constructor sorts by
/// candidate ID, so two plans built from the same set of candidates
/// (in any input order) produce byte-identical reports.
#[derive(Debug, Clone)]
pub struct CompressionPlan {
    entries: Vec<PlanEntry>,
    /// FNV-1a hash of the concatenation of every candidate's
    /// canonical hash, in slot order. Identifies the plan content.
    plan_hash: u64,
}

impl CompressionPlan {
    /// Build a plan. Candidates are sorted by [`CandidateId`] before
    /// being assigned slots. Duplicate IDs are allowed (different
    /// candidates can share an ID — the report's per-slot ordering still
    /// disambiguates), but the canonical ordering between them is by
    /// payload hash to maintain determinism.
    pub fn new(mut candidates: Vec<CompressionCandidate>) -> Self {
        // Sort by (id, canonical_hash) — id is the primary key, but on
        // duplicates we tie-break by content hash so the order is total
        // and stable.
        candidates.sort_by(|a, b| {
            a.id()
                .cmp(&b.id())
                .then_with(|| a.canonical_hash().cmp(&b.canonical_hash()))
        });
        let entries: Vec<PlanEntry> = candidates
            .into_iter()
            .enumerate()
            .map(|(i, c)| PlanEntry {
                slot: i as u32,
                candidate: c,
            })
            .collect();
        let plan_hash = compute_plan_hash(&entries);
        Self { entries, plan_hash }
    }

    /// Number of entries in the plan.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the plan has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate entries in slot order.
    pub fn entries(&self) -> impl Iterator<Item = &PlanEntry> {
        self.entries.iter()
    }

    /// Content-addressed plan hash. Two plans with byte-identical
    /// candidate sets produce the same hash; adding, removing, or
    /// mutating any candidate changes it.
    pub fn plan_hash(&self) -> u64 {
        self.plan_hash
    }

    /// Run every candidate through the kind-appropriate compressor and
    /// validate the result. Returns a [`CompressionReport`] whose
    /// `entries` are in slot order.
    ///
    /// **Never panics.** Decompressor/decoder errors are surfaced as
    /// `EntryStatus::MalformedRoundTrip` or `Decode` variants in the
    /// per-entry status field.
    pub fn execute(&self) -> CompressionReport {
        let mut report_entries: Vec<ReportEntry> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            report_entries.push(execute_entry(entry));
        }
        CompressionReport::new(self.plan_hash, report_entries)
    }
}

/// Hash a plan's entry list: FNV-1a over `(slot, candidate.canonical_hash())`
/// pairs in slot order. Stable for byte-identical inputs.
fn compute_plan_hash(entries: &[PlanEntry]) -> u64 {
    let mut bytes = Vec::with_capacity(entries.len() * 12);
    for e in entries {
        bytes.extend_from_slice(&e.slot.to_le_bytes());
        bytes.extend_from_slice(&e.candidate.canonical_hash().to_le_bytes());
    }
    hash_bytes(&bytes)
}

// ---------------------------------------------------------------------------
// Per-entry execution
// ---------------------------------------------------------------------------

fn execute_entry(entry: &PlanEntry) -> ReportEntry {
    let c = &entry.candidate;
    let kind = c.kind();
    let criticality = c.criticality();
    let original_len = c.payload_len();
    let input_hash = hash_bytes(c.payload());
    let mut header = ReportEntry {
        slot: entry.slot,
        candidate_id: c.id(),
        candidate_label: c.label().to_string(),
        kind,
        criticality_tag: criticality.tag(),
        declared_tolerance: criticality.tolerance(),
        original_len,
        compressed_len: 0,
        input_hash,
        summary_hash: 0,
        reconstructed_hash: 0,
        observed_error: 0.0,
        status: EntryStatus::Validated,
    };

    match kind {
        CompressionKind::LosslessTrace => execute_lossless_trace(&mut header, c.payload()),
        CompressionKind::MotifDictionary => execute_motif_dictionary(&mut header, c.payload()),
        CompressionKind::LowRankAdvisory => {
            execute_low_rank(&mut header, c.payload(), criticality.tolerance())
        }
        CompressionKind::TensorTrainAdvisory => {
            execute_tensor_train(&mut header, c.payload(), criticality.tolerance())
        }
    }
    header
}

fn execute_lossless_trace(header: &mut ReportEntry, payload: &[u8]) {
    let p = lossless_compress_bytes(payload);
    header.compressed_len = p.bytes.len();
    header.summary_hash = p.compressed_hash;
    match lossless_decompress_bytes(&p.bytes) {
        Ok(decoded) => {
            let reconstructed_hash = hash_bytes(&decoded);
            header.reconstructed_hash = reconstructed_hash;
            if reconstructed_hash != header.input_hash {
                header.status = EntryStatus::MalformedRoundTrip;
            }
        }
        Err(e) => {
            header.status = EntryStatus::DecodeFailed { error: e };
        }
    }
}

fn execute_motif_dictionary(header: &mut ReportEntry, payload: &[u8]) {
    let p = compress_motif_dictionary(payload);
    header.compressed_len = p.bytes.len();
    header.summary_hash = p.compressed_hash;
    match decompress_motif_dictionary(&p.bytes) {
        Ok(decoded) => {
            let reconstructed_hash = hash_bytes(&decoded);
            header.reconstructed_hash = reconstructed_hash;
            if reconstructed_hash != header.input_hash {
                header.status = EntryStatus::MalformedRoundTrip;
            }
        }
        Err(e) => {
            header.status = EntryStatus::DecodeFailed { error: e };
        }
    }
}

fn execute_low_rank(header: &mut ReportEntry, payload: &[u8], declared_tol: f64) {
    let (matrix, rows, cols, max_rank) = match decode_low_rank_payload(payload) {
        Ok(t) => t,
        Err(e) => {
            header.status = EntryStatus::DecodeFailed { error: e };
            return;
        }
    };
    match compress_low_rank(&matrix, rows, cols, max_rank) {
        Ok(p) => {
            header.compressed_len = p.canonical_bytes().len();
            header.summary_hash = p.summary_hash;
            // The "reconstructed hash" for a lossy compressor is the
            // hash of the reconstructed f64 matrix's bit pattern.
            let recon = p.reconstruct();
            let mut recon_bytes = Vec::with_capacity(recon.len() * 8);
            for x in &recon {
                recon_bytes.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            header.reconstructed_hash = hash_bytes(&recon_bytes);
            header.observed_error = p.frobenius_error;
            if p.frobenius_error > declared_tol {
                header.status = EntryStatus::ToleranceExceeded {
                    declared: declared_tol,
                    observed: p.frobenius_error,
                };
            }
        }
        Err(e) => {
            header.status = EntryStatus::DecodeFailed { error: e };
        }
    }
}

fn execute_tensor_train(header: &mut ReportEntry, payload: &[u8], declared_tol: f64) {
    let (tensor, shape, max_bond) = match decode_tensor_train_payload(payload) {
        Ok(t) => t,
        Err(e) => {
            header.status = EntryStatus::DecodeFailed { error: e };
            return;
        }
    };
    // Internal SVD tolerance: a tight cutoff so we don't accidentally drop
    // good singular values. The OBSERVED reconstruction error then has
    // to clear the candidate's declared tolerance.
    let internal_tol = 1e-12;
    match compress_tensor_train(&tensor, &shape, max_bond, internal_tol) {
        Ok(p) => {
            header.compressed_len = p.canonical_bytes().len();
            header.summary_hash = p.summary_hash;
            let recon = p.reconstruct();
            let mut recon_bytes = Vec::with_capacity(recon.len() * 8);
            for x in &recon {
                recon_bytes.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            header.reconstructed_hash = hash_bytes(&recon_bytes);
            header.observed_error = p.frobenius_error;
            if p.frobenius_error > declared_tol {
                header.status = EntryStatus::ToleranceExceeded {
                    declared: declared_tol,
                    observed: p.frobenius_error,
                };
            }
        }
        Err(e) => {
            header.status = EntryStatus::DecodeFailed { error: e };
        }
    }
}

// ---------------------------------------------------------------------------
// Payload codecs for lossy candidates
// ---------------------------------------------------------------------------

const LOW_RANK_MAGIC: &[u8; 4] = b"CLRP";
const TENSOR_TRAIN_MAGIC: &[u8; 4] = b"CTTP";

/// Encode `(matrix, rows, cols, max_rank)` into the canonical payload
/// bytes consumed by [`CompressionKind::LowRankAdvisory`].
pub fn encode_low_rank_payload(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    max_rank: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(LOW_RANK_MAGIC.len() + 12 + matrix.len() * 8);
    out.extend_from_slice(LOW_RANK_MAGIC);
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&(cols as u32).to_le_bytes());
    out.extend_from_slice(&(max_rank as u32).to_le_bytes());
    for x in matrix {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Decode a low-rank payload. Returns `(matrix, rows, cols, max_rank)`.
fn decode_low_rank_payload(
    bytes: &[u8],
) -> Result<(Vec<f64>, usize, usize, usize), CompressionError> {
    if bytes.len() < LOW_RANK_MAGIC.len() + 12 {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "low-rank payload header truncated",
        });
    }
    if &bytes[0..4] != LOW_RANK_MAGIC {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "bad low-rank magic",
        });
    }
    let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let max_rank = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let data_start = 16;
    let expected_data_len = rows * cols * 8;
    if bytes.len() != data_start + expected_data_len {
        return Err(CompressionError::MalformedPayload {
            at_byte: data_start,
            reason: "low-rank payload data length mismatches header shape",
        });
    }
    let mut matrix = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        let off = data_start + i * 8;
        let bits = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        matrix.push(f64::from_bits(bits));
    }
    Ok((matrix, rows, cols, max_rank))
}

/// Encode `(tensor, shape, max_bond)` into the canonical payload bytes
/// consumed by [`CompressionKind::TensorTrainAdvisory`].
pub fn encode_tensor_train_payload(tensor: &[f64], shape: &[usize], max_bond: usize) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(TENSOR_TRAIN_MAGIC.len() + 4 + 4 * shape.len() + 4 + tensor.len() * 8);
    out.extend_from_slice(TENSOR_TRAIN_MAGIC);
    out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &d in shape {
        out.extend_from_slice(&(d as u32).to_le_bytes());
    }
    out.extend_from_slice(&(max_bond as u32).to_le_bytes());
    for x in tensor {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn decode_tensor_train_payload(
    bytes: &[u8],
) -> Result<(Vec<f64>, Vec<usize>, usize), CompressionError> {
    if bytes.len() < TENSOR_TRAIN_MAGIC.len() + 4 {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "tensor-train payload header truncated",
        });
    }
    if &bytes[0..4] != TENSOR_TRAIN_MAGIC {
        return Err(CompressionError::MalformedPayload {
            at_byte: 0,
            reason: "bad tensor-train magic",
        });
    }
    let ndim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let header_len = 8 + ndim * 4 + 4;
    if bytes.len() < header_len {
        return Err(CompressionError::MalformedPayload {
            at_byte: 8,
            reason: "tensor-train shape header truncated",
        });
    }
    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let off = 8 + i * 4;
        shape.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize);
    }
    let max_bond_off = 8 + ndim * 4;
    let max_bond =
        u32::from_le_bytes(bytes[max_bond_off..max_bond_off + 4].try_into().unwrap()) as usize;
    let data_start = max_bond_off + 4;
    let total: usize = shape.iter().product();
    let expected_data_len = total * 8;
    if bytes.len() != data_start + expected_data_len {
        return Err(CompressionError::MalformedPayload {
            at_byte: data_start,
            reason: "tensor-train data length mismatches header shape",
        });
    }
    let mut tensor = Vec::with_capacity(total);
    for i in 0..total {
        let off = data_start + i * 8;
        let bits = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        tensor.push(f64::from_bits(bits));
    }
    Ok((tensor, shape, max_bond))
}

// ---------------------------------------------------------------------------
// Convenience type re-exports (not used directly by execute() — they're
// here so wiring tests can construct typed payloads without importing
// individual compressors).
// ---------------------------------------------------------------------------

/// Re-export of the low-rank payload type for convenience in tests that
/// construct plans and want to inspect the underlying compressed form.
pub type LowRankSummary = LowRankPayload;
/// Re-export of the tensor-train payload type for the same reason.
pub type TensorTrainSummary = TensorTrainPayload;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{CandidateId, CompressionCandidate, Criticality};

    fn lossless_cand(id: u64, label: &str, payload: Vec<u8>) -> CompressionCandidate {
        CompressionCandidate::new(
            CandidateId(id),
            CompressionKind::LosslessTrace,
            Criticality::SemanticCritical,
            payload,
            label,
        )
        .unwrap()
    }

    fn motif_cand(id: u64, label: &str, payload: Vec<u8>) -> CompressionCandidate {
        CompressionCandidate::new(
            CandidateId(id),
            CompressionKind::MotifDictionary,
            Criticality::SemanticCritical,
            payload,
            label,
        )
        .unwrap()
    }

    fn low_rank_cand(
        id: u64,
        label: &str,
        tol: f64,
        matrix: &[f64],
        rows: usize,
        cols: usize,
        max_rank: usize,
    ) -> CompressionCandidate {
        let payload = encode_low_rank_payload(matrix, rows, cols, max_rank);
        CompressionCandidate::new(
            CandidateId(id),
            CompressionKind::LowRankAdvisory,
            Criticality::AdvisoryOnly { tolerance_f: tol },
            payload,
            label,
        )
        .unwrap()
    }

    fn tt_cand(
        id: u64,
        label: &str,
        tol: f64,
        tensor: &[f64],
        shape: &[usize],
        max_bond: usize,
    ) -> CompressionCandidate {
        let payload = encode_tensor_train_payload(tensor, shape, max_bond);
        CompressionCandidate::new(
            CandidateId(id),
            CompressionKind::TensorTrainAdvisory,
            Criticality::AdvisoryOnly { tolerance_f: tol },
            payload,
            label,
        )
        .unwrap()
    }

    #[test]
    fn empty_plan_executes_to_empty_report() {
        let plan = CompressionPlan::new(vec![]);
        assert!(plan.is_empty());
        let report = plan.execute();
        assert_eq!(report.entries().count(), 0);
        assert!(report.all_validated());
    }

    #[test]
    fn plan_sorts_by_candidate_id() {
        let plan = CompressionPlan::new(vec![
            lossless_cand(3, "c", vec![1; 4]),
            lossless_cand(1, "a", vec![1; 4]),
            lossless_cand(2, "b", vec![1; 4]),
        ]);
        let ids: Vec<u64> = plan.entries().map(|e| e.candidate.id().0).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn lossless_entry_round_trips() {
        let plan = CompressionPlan::new(vec![lossless_cand(1, "x", b"abracadabra".to_vec())]);
        let report = plan.execute();
        assert!(report.all_validated());
        let entry = report.entries().next().unwrap();
        assert_eq!(entry.status, EntryStatus::Validated);
        assert_eq!(entry.input_hash, entry.reconstructed_hash);
    }

    #[test]
    fn motif_entry_round_trips() {
        let plan = CompressionPlan::new(vec![motif_cand(
            1,
            "x",
            b"abcabcabc-abcabcabc-abcabcabc".to_vec(),
        )]);
        let report = plan.execute();
        assert!(report.all_validated());
        let entry = report.entries().next().unwrap();
        assert_eq!(entry.input_hash, entry.reconstructed_hash);
    }

    #[test]
    fn low_rank_entry_validates_when_under_tol() {
        // A perfectly rank-1 matrix; rank-1 reconstruction should be near-exact.
        let u = [1.0f64, 2.0, 3.0];
        let v = [4.0f64, 5.0, 6.0];
        let mut m = vec![0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                m[r * 3 + c] = u[r] * v[c];
            }
        }
        let plan = CompressionPlan::new(vec![low_rank_cand(1, "m", 1e-6, &m, 3, 3, 1)]);
        let report = plan.execute();
        let entry = report.entries().next().unwrap();
        assert_eq!(
            entry.status,
            EntryStatus::Validated,
            "got {:?}",
            entry.status
        );
        assert!(entry.observed_error < 1e-6);
    }

    #[test]
    fn low_rank_entry_flags_tolerance_exceeded_when_undertruncated() {
        // 3x3 non-rank-1 matrix, rank-1 with tolerance 0 → fails.
        let m: Vec<f64> = (1..=9).map(|x| x as f64).collect();
        let plan = CompressionPlan::new(vec![low_rank_cand(1, "m", 0.0, &m, 3, 3, 1)]);
        let report = plan.execute();
        let entry = report.entries().next().unwrap();
        match &entry.status {
            EntryStatus::ToleranceExceeded { declared, observed } => {
                assert_eq!(*declared, 0.0);
                assert!(*observed > 0.0);
            }
            other => panic!("expected ToleranceExceeded, got {other:?}"),
        }
    }

    #[test]
    fn tensor_train_entry_validates_when_full_bond() {
        let t = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let plan = CompressionPlan::new(vec![tt_cand(1, "t", 1e-6, &t, &[2, 2, 2], 8)]);
        let report = plan.execute();
        let entry = report.entries().next().unwrap();
        assert_eq!(
            entry.status,
            EntryStatus::Validated,
            "got {:?}",
            entry.status
        );
        assert!(entry.observed_error < 1e-6);
    }

    #[test]
    fn tensor_train_entry_flags_tolerance_exceeded_when_bond_one() {
        let t = vec![1.0, 2.0, 3.0, 4.0]; // shape [2, 2], rank-2 by construction
        let plan = CompressionPlan::new(vec![tt_cand(1, "t", 0.0, &t, &[2, 2], 1)]);
        let report = plan.execute();
        let entry = report.entries().next().unwrap();
        assert!(matches!(
            entry.status,
            EntryStatus::ToleranceExceeded { .. }
        ));
    }

    #[test]
    fn malformed_lossy_payload_surfaces_decode_error() {
        // Build a "low rank" candidate whose payload bytes don't decode.
        let candidate = CompressionCandidate::new(
            CandidateId(1),
            CompressionKind::LowRankAdvisory,
            Criticality::AdvisoryOnly { tolerance_f: 0.1 },
            b"GARBAGE".to_vec(),
            "broken",
        )
        .unwrap();
        let plan = CompressionPlan::new(vec![candidate]);
        let report = plan.execute();
        let entry = report.entries().next().unwrap();
        assert!(matches!(entry.status, EntryStatus::DecodeFailed { .. }));
    }

    #[test]
    fn plan_hash_is_deterministic() {
        // Build the same plan twice; hash equal.
        let p1 = CompressionPlan::new(vec![
            lossless_cand(1, "a", vec![1, 2, 3]),
            lossless_cand(2, "b", vec![4, 5, 6]),
        ]);
        let p2 = CompressionPlan::new(vec![
            // Order intentionally different.
            lossless_cand(2, "b", vec![4, 5, 6]),
            lossless_cand(1, "a", vec![1, 2, 3]),
        ]);
        assert_eq!(p1.plan_hash(), p2.plan_hash());
    }

    #[test]
    fn plan_hash_differs_on_different_payload() {
        let p1 = CompressionPlan::new(vec![lossless_cand(1, "a", vec![1, 2, 3])]);
        let p2 = CompressionPlan::new(vec![lossless_cand(1, "a", vec![1, 2, 4])]);
        assert_ne!(p1.plan_hash(), p2.plan_hash());
    }

    #[test]
    fn encode_decode_low_rank_round_trips() {
        let m: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let bytes = encode_low_rank_payload(&m, 3, 4, 2);
        let (back_m, rows, cols, max_rank) = decode_low_rank_payload(&bytes).unwrap();
        assert_eq!(back_m, m);
        assert_eq!(rows, 3);
        assert_eq!(cols, 4);
        assert_eq!(max_rank, 2);
    }

    #[test]
    fn encode_decode_tensor_train_round_trips() {
        let t: Vec<f64> = (1..=12).map(|x| x as f64 * 0.5).collect();
        let bytes = encode_tensor_train_payload(&t, &[2, 2, 3], 4);
        let (back_t, shape, max_bond) = decode_tensor_train_payload(&bytes).unwrap();
        assert_eq!(back_t, t);
        assert_eq!(shape, vec![2, 2, 3]);
        assert_eq!(max_bond, 4);
    }

    #[test]
    fn decode_low_rank_rejects_bad_magic() {
        let r = decode_low_rank_payload(b"XXXX____________");
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "bad low-rank magic",
                ..
            })
        ));
    }

    #[test]
    fn decode_tensor_train_rejects_bad_magic() {
        let r = decode_tensor_train_payload(b"XXXX____________");
        assert!(matches!(
            r,
            Err(CompressionError::MalformedPayload {
                reason: "bad tensor-train magic",
                ..
            })
        ));
    }
}
