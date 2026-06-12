//! Phase E — compression prototypes, measured on real artifacts.
//!
//! Exit criterion (roadmap §6 row E): **before/after bytes at bounded
//! reconstruction error.** Three prototypes:
//!
//! 1. **Trace streams** — `MirTraceEvent` streams from instrumented
//!    runs of the hash-pinned Phase-D subjects, serialized two ways
//!    (canonical row-major; delta/XOR columnar) and compressed with
//!    the two committed lossless codecs (byte-RLE, motif dictionary).
//!    Reconstruction bound: ZERO — every measurement decodes back and
//!    compares bit-exactly, hard error otherwise.
//! 2. **Checkpoint low-rank** — the real chess-RL weight checkpoint
//!    (snap format, ~1.1 MB) walked for 2-D tensors, each compressed
//!    to the SMALLEST rank within an `AdvisoryOnly`-style relative
//!    Frobenius tolerance (binary search — truncation error is
//!    monotone in rank).
//! 3. **Disk artifacts** — committed files (profile DB, phases CSV)
//!    byte-level through both lossless codecs with roundtrip checks.
//!
//! Everything here is diagnostics: nothing feeds back into compile
//! decisions, hashes, or row stable fields. Prototype transforms live
//! in this bench crate — they graduate to `cjc-cana-compress` only
//! with measured numbers behind them.

use std::path::PathBuf;

use cana_diagnostics::{compile_subject, subjects, workspace_root, SEED};
use cjc_cana_compress::lossless_trace::{lossless_compress_bytes, lossless_decompress_bytes};
use cjc_cana_compress::motif_dictionary::{
    compress_motif_dictionary, decompress_motif_dictionary,
};
use cjc_cana_compress::lowrank::compress_low_rank;
use cjc_mir_exec::run_program_instrumented;
use cjc_nss::mir_adapter::MirTraceEvent;
use cjc_runtime::{Tensor, Value};

// =============================================================================
// Trace-stream serialization (prototype transforms; bit-exact invertible)
// =============================================================================

// Format v1: adds the Phase F `alloc_bytes_in_window` column.
const CANON_MAGIC: &[u8; 4] = b"CTE1";
const DELTA_MAGIC: &[u8; 4] = b"CTD1";
/// Bytes per event in the canonical encoding (see layout below).
const EVENT_BYTES: usize = 53;

fn event_flags(e: &MirTraceEvent) -> u8 {
    (e.branch_taken as u8) | ((e.io_event as u8) << 1) | ((e.gc_event as u8) << 2)
}

/// Canonical row-major encoding: `magic + count:u64` then per event
/// `tick:u64, block_id:u32, register_pressure:f64bits, heap:u64,
/// call_depth:u32, flags:u8, instruction_count:u32, thermal:f64bits,
/// alloc_bytes:u64` (53 bytes/event, all little-endian, floats by bit
/// pattern).
pub fn events_to_canonical_bytes(events: &[MirTraceEvent]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + events.len() * EVENT_BYTES);
    out.extend_from_slice(CANON_MAGIC);
    out.extend_from_slice(&(events.len() as u64).to_le_bytes());
    for e in events {
        out.extend_from_slice(&e.tick.to_le_bytes());
        out.extend_from_slice(&e.block_id.to_le_bytes());
        out.extend_from_slice(&e.register_pressure.to_bits().to_le_bytes());
        out.extend_from_slice(&e.heap_bytes_in_use.to_le_bytes());
        out.extend_from_slice(&e.call_depth.to_le_bytes());
        out.push(event_flags(e));
        out.extend_from_slice(&e.instruction_count.to_le_bytes());
        out.extend_from_slice(&e.thermal_intensity.to_bits().to_le_bytes());
        out.extend_from_slice(&e.alloc_bytes_in_window.to_le_bytes());
    }
    out
}

/// Delta/XOR columnar encoding: same header, then eight whole columns
/// in fixed order. Integer columns store `wrapping_sub` deltas against
/// the previous value (first entry raw); f64 columns store the XOR of
/// consecutive bit patterns (first raw); flags raw. Loop-dominated
/// traces turn into long zero runs, which is exactly what the byte-RLE
/// codec eats — this transform IS the "trace-stream RLE" idea from the
/// research doc §2, made concrete.
pub fn events_to_delta_bytes(events: &[MirTraceEvent]) -> Vec<u8> {
    let n = events.len();
    let mut out = Vec::with_capacity(12 + n * EVENT_BYTES);
    out.extend_from_slice(DELTA_MAGIC);
    out.extend_from_slice(&(n as u64).to_le_bytes());

    let prev_u64 = |sel: fn(&MirTraceEvent) -> u64, out: &mut Vec<u8>| {
        let mut prev = 0u64;
        for (i, e) in events.iter().enumerate() {
            let v = sel(e);
            let stored = if i == 0 { v } else { v.wrapping_sub(prev) };
            out.extend_from_slice(&stored.to_le_bytes());
            prev = v;
        }
    };
    let prev_u32 = |sel: fn(&MirTraceEvent) -> u32, out: &mut Vec<u8>| {
        let mut prev = 0u32;
        for (i, e) in events.iter().enumerate() {
            let v = sel(e);
            let stored = if i == 0 { v } else { v.wrapping_sub(prev) };
            out.extend_from_slice(&stored.to_le_bytes());
            prev = v;
        }
    };
    let xor_f64 = |sel: fn(&MirTraceEvent) -> f64, out: &mut Vec<u8>| {
        let mut prev = 0u64;
        for (i, e) in events.iter().enumerate() {
            let bits = sel(e).to_bits();
            let stored = if i == 0 { bits } else { bits ^ prev };
            out.extend_from_slice(&stored.to_le_bytes());
            prev = bits;
        }
    };

    prev_u64(|e| e.tick, &mut out);
    prev_u32(|e| e.block_id, &mut out);
    xor_f64(|e| e.register_pressure, &mut out);
    prev_u64(|e| e.heap_bytes_in_use, &mut out);
    prev_u32(|e| e.call_depth, &mut out);
    for e in events {
        out.push(event_flags(e));
    }
    prev_u32(|e| e.instruction_count, &mut out);
    xor_f64(|e| e.thermal_intensity, &mut out);
    prev_u64(|e| e.alloc_bytes_in_window, &mut out);
    out
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.pos.checked_add(N)?;
        if end > self.bytes.len() {
            return None;
        }
        let mut buf = [0u8; N];
        buf.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Some(buf)
    }
    fn u64(&mut self) -> Option<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }
    fn u32(&mut self) -> Option<u32> {
        self.take::<4>().map(u32::from_le_bytes)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|b| b[0])
    }
}

fn apply_flags(e: &mut MirTraceEvent, flags: u8) -> Option<()> {
    if flags & !0b111 != 0 {
        return None; // reserved bits must be zero
    }
    e.branch_taken = flags & 1 != 0;
    e.io_event = flags & 2 != 0;
    e.gc_event = flags & 4 != 0;
    Some(())
}

/// Decode the canonical encoding. `None` on any malformation — never
/// panics (bolero-fuzzed).
pub fn canonical_bytes_to_events(bytes: &[u8]) -> Option<Vec<MirTraceEvent>> {
    let mut r = ByteReader::new(bytes);
    if &r.take::<4>()? != CANON_MAGIC {
        return None;
    }
    let n = r.u64()?;
    // Reject counts that cannot fit in the remaining bytes (guards the
    // pre-allocation below against malicious headers).
    if (n as usize).checked_mul(EVENT_BYTES)? != bytes.len().checked_sub(12)? {
        return None;
    }
    let mut events = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut e = MirTraceEvent::minimal(0, 0);
        e.tick = r.u64()?;
        e.block_id = r.u32()?;
        e.register_pressure = f64::from_bits(r.u64()?);
        e.heap_bytes_in_use = r.u64()?;
        e.call_depth = r.u32()?;
        apply_flags(&mut e, r.u8()?)?;
        e.instruction_count = r.u32()?;
        e.thermal_intensity = f64::from_bits(r.u64()?);
        e.alloc_bytes_in_window = r.u64()?;
        events.push(e);
    }
    Some(events)
}

/// Decode the delta/XOR columnar encoding. `None` on any malformation.
pub fn delta_bytes_to_events(bytes: &[u8]) -> Option<Vec<MirTraceEvent>> {
    let mut r = ByteReader::new(bytes);
    if &r.take::<4>()? != DELTA_MAGIC {
        return None;
    }
    let n = r.u64()? as usize;
    if n.checked_mul(EVENT_BYTES)? != bytes.len().checked_sub(12)? {
        return None;
    }
    let mut events = vec![MirTraceEvent::minimal(0, 0); n];

    {
        let mut prev = 0u64;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u64()?;
            let v = if i == 0 { stored } else { stored.wrapping_add(prev) };
            e.tick = v;
            prev = v;
        }
    }
    {
        let mut prev = 0u32;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u32()?;
            let v = if i == 0 { stored } else { stored.wrapping_add(prev) };
            e.block_id = v;
            prev = v;
        }
    }
    {
        let mut prev = 0u64;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u64()?;
            let bits = if i == 0 { stored } else { stored ^ prev };
            e.register_pressure = f64::from_bits(bits);
            prev = bits;
        }
    }
    {
        let mut prev = 0u64;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u64()?;
            let v = if i == 0 { stored } else { stored.wrapping_add(prev) };
            e.heap_bytes_in_use = v;
            prev = v;
        }
    }
    {
        let mut prev = 0u32;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u32()?;
            let v = if i == 0 { stored } else { stored.wrapping_add(prev) };
            e.call_depth = v;
            prev = v;
        }
    }
    for e in events.iter_mut() {
        let flags = r.u8()?;
        apply_flags(e, flags)?;
    }
    {
        let mut prev = 0u32;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u32()?;
            let v = if i == 0 { stored } else { stored.wrapping_add(prev) };
            e.instruction_count = v;
            prev = v;
        }
    }
    {
        let mut prev = 0u64;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u64()?;
            let bits = if i == 0 { stored } else { stored ^ prev };
            e.thermal_intensity = f64::from_bits(bits);
            prev = bits;
        }
    }
    {
        let mut prev = 0u64;
        for (i, e) in events.iter_mut().enumerate() {
            let stored = r.u64()?;
            let v = if i == 0 { stored } else { stored.wrapping_add(prev) };
            e.alloc_bytes_in_window = v;
            prev = v;
        }
    }
    Some(events)
}

/// Bit-exact event equality (the derived `PartialEq` compares floats by
/// value, which would treat distinct NaN payloads as unequal and
/// `-0.0 == 0.0` — roundtrip verification needs the bit patterns).
pub fn events_bitwise_equal(a: &[MirTraceEvent], b: &[MirTraceEvent]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.tick == y.tick
                && x.block_id == y.block_id
                && x.register_pressure.to_bits() == y.register_pressure.to_bits()
                && x.heap_bytes_in_use == y.heap_bytes_in_use
                && x.call_depth == y.call_depth
                && x.branch_taken == y.branch_taken
                && x.io_event == y.io_event
                && x.gc_event == y.gc_event
                && x.instruction_count == y.instruction_count
                && x.thermal_intensity.to_bits() == y.thermal_intensity.to_bits()
                && x.alloc_bytes_in_window == y.alloc_bytes_in_window
        })
}

// =============================================================================
// Lossless byte measurement (both committed codecs + roundtrip proof)
// =============================================================================

/// One lossless measurement: original vs both codecs, with roundtrip
/// verdicts. A `false` roundtrip is reported, and the caller treats it
/// as a hard error — a lossless codec that doesn't roundtrip is a bug,
/// not a data point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteCompressionOutcome {
    pub original_bytes: usize,
    pub rle_bytes: usize,
    pub motif_bytes: usize,
    pub rle_roundtrip_ok: bool,
    pub motif_roundtrip_ok: bool,
}

impl ByteCompressionOutcome {
    pub fn rle_ratio(&self) -> f64 {
        self.original_bytes as f64 / self.rle_bytes.max(1) as f64
    }
    pub fn motif_ratio(&self) -> f64 {
        self.original_bytes as f64 / self.motif_bytes.max(1) as f64
    }
    pub fn roundtrips_ok(&self) -> bool {
        self.rle_roundtrip_ok && self.motif_roundtrip_ok
    }
}

/// Run both committed lossless codecs over `input` and verify both
/// roundtrips byte-exactly.
pub fn measure_lossless(input: &[u8]) -> ByteCompressionOutcome {
    let rle = lossless_compress_bytes(input);
    let rle_roundtrip_ok = lossless_decompress_bytes(&rle.bytes)
        .map(|back| back == input)
        .unwrap_or(false);
    let motif = compress_motif_dictionary(input);
    let motif_roundtrip_ok = decompress_motif_dictionary(&motif.bytes)
        .map(|back| back == input)
        .unwrap_or(false);
    ByteCompressionOutcome {
        original_bytes: input.len(),
        rle_bytes: rle.bytes.len(),
        motif_bytes: motif.bytes.len(),
        rle_roundtrip_ok,
        motif_roundtrip_ok,
    }
}

// =============================================================================
// Prototype 1 — trace streams from instrumented runs of Phase-D subjects
// =============================================================================

/// Exact-name list (membership, never substring — the config-name
/// trap rule) of subjects whose instrumented streams the probe
/// measures. Chosen for size spread: smallest churn loop → 65k-iteration
/// churn loop, scalar FP, nested-loop FP gradient, tensor element-wise,
/// frozen holdout.
pub const TRACE_SUBJECTS: &[&str] = &[
    "mem_grad_a1",
    "mem_grad_a5",
    "fp_hot",
    "grad_f90_d1_n1024",
    "tensor_ew_n32_i200",
    "holdout_alloc_pulse",
];

/// Cap on events per stream fed to the codecs. NOT a silent cap: the
/// outcome records both the full stream length and how many events
/// were measured, and the report prints the truncation.
pub const TRACE_EVENT_CAP: usize = 250_000;

/// Trace-stream measurement for one subject.
#[derive(Debug, Clone)]
pub struct TraceStreamOutcome {
    pub subject: String,
    /// Full instrumented stream length (events).
    pub total_events: usize,
    /// Events actually measured (`min(total, TRACE_EVENT_CAP)`).
    pub measured_events: usize,
    /// Canonical row-major representation through both codecs.
    pub canonical: ByteCompressionOutcome,
    /// Delta/XOR columnar representation through both codecs.
    pub delta: ByteCompressionOutcome,
    /// Both representations decoded back to the original events
    /// bit-exactly (this is the ZERO-error reconstruction proof; the
    /// codec roundtrips inside `canonical`/`delta` prove the byte
    /// layer separately).
    pub representations_bitexact: bool,
}

/// Instrument one subject and measure its trace stream. Errors are
/// hard: a subject that fails to compile/run or a representation that
/// fails to roundtrip invalidates the prototype claim.
pub fn measure_trace_subject(name: &str) -> Result<TraceStreamOutcome, String> {
    let subject = subjects()
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("unknown trace subject {name}"))?;
    let compiled = compile_subject(&subject.source)?;
    let (_val, _exec, events) = run_program_instrumented(&compiled.ast, SEED)
        .map_err(|e| format!("instrumented run of {name} failed: {e:?}"))?;
    let total_events = events.len();
    let measured: &[MirTraceEvent] = &events[..total_events.min(TRACE_EVENT_CAP)];

    let canon_bytes = events_to_canonical_bytes(measured);
    let delta_bytes = events_to_delta_bytes(measured);

    let canon_back = canonical_bytes_to_events(&canon_bytes)
        .ok_or_else(|| format!("{name}: canonical decode failed"))?;
    let delta_back = delta_bytes_to_events(&delta_bytes)
        .ok_or_else(|| format!("{name}: delta decode failed"))?;
    let representations_bitexact = events_bitwise_equal(measured, &canon_back)
        && events_bitwise_equal(measured, &delta_back);
    if !representations_bitexact {
        return Err(format!(
            "{name}: representation roundtrip is NOT bit-exact — prototype invalid"
        ));
    }

    let canonical = measure_lossless(&canon_bytes);
    let delta = measure_lossless(&delta_bytes);
    if !canonical.roundtrips_ok() || !delta.roundtrips_ok() {
        return Err(format!("{name}: lossless codec roundtrip failed"));
    }

    Ok(TraceStreamOutcome {
        subject: name.to_string(),
        total_events,
        measured_events: measured.len(),
        canonical,
        delta,
        representations_bitexact,
    })
}

// =============================================================================
// Prototype 2 — checkpoint low-rank under an advisory Frobenius tolerance
// =============================================================================

/// AdvisoryOnly-style relative Frobenius tolerance for checkpoint
/// tensors. 0.05 matches the research doc's "bounded Frobenius error"
/// framing for DIAGNOSTIC checkpoints — never training-resumption
/// paths.
pub const CHECKPOINT_TOLERANCE: f64 = 0.05;

/// One tensor's low-rank outcome.
#[derive(Debug, Clone)]
pub struct TensorLowRankOutcome {
    /// Path of the tensor inside the checkpoint value tree.
    pub path: String,
    pub rows: usize,
    pub cols: usize,
    /// `8 · rows · cols` — the raw f64 payload.
    pub original_bytes: usize,
    /// Smallest rank whose relative Frobenius error ≤ tolerance, if
    /// the search found one whose payload is also SMALLER than raw.
    pub chosen_rank: Option<usize>,
    pub compressed_bytes: Option<usize>,
    pub frobenius_error: Option<f64>,
}

/// Whole-checkpoint outcome.
#[derive(Debug, Clone)]
pub struct CheckpointOutcome {
    pub source: String,
    /// Bytes of the checkpoint file on disk.
    pub file_bytes: usize,
    /// 2-D tensors measured.
    pub matrices: Vec<TensorLowRankOutcome>,
    /// Raw bytes of tensors that are NOT 2-D (biases, scalars) —
    /// stored uncompressed; counted so the headline ratio is honest
    /// about the whole artifact, not just its compressible part.
    pub passthrough_bytes: usize,
}

impl CheckpointOutcome {
    /// Total tensor-payload bytes before/after, counting passthrough
    /// tensors at raw size and within-tolerance matrices at payload
    /// size (matrices that didn't compress stay raw).
    pub fn totals(&self) -> (usize, usize) {
        let mut before = self.passthrough_bytes;
        let mut after = self.passthrough_bytes;
        for m in &self.matrices {
            before += m.original_bytes;
            after += m.compressed_bytes.unwrap_or(m.original_bytes);
        }
        (before, after)
    }
}

/// Recursively collect named tensors from a snap-loaded value tree.
pub fn collect_tensors(value: &Value, path: String, out: &mut Vec<(String, Tensor)>) {
    match value {
        Value::Tensor(t) => out.push((path, t.clone())),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                collect_tensors(item, format!("{path}[{i}]"), out);
            }
        }
        Value::Map(m) => {
            for (k, v) in m.borrow().iter() {
                collect_tensors(v, format!("{path}.{k:?}"), out);
            }
        }
        _ => {}
    }
}

/// Binary-search the smallest rank meeting `tolerance` (relative
/// Frobenius). Valid because SVD truncation error is monotone
/// non-increasing in rank. Returns `None` if even full rank misses the
/// tolerance (shouldn't happen — full rank is exact up to iteration
/// tolerance) or if the winning payload isn't actually smaller than
/// the raw matrix.
pub fn smallest_rank_within(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    tolerance: f64,
) -> Result<Option<(usize, usize, f64)>, String> {
    let max_rank = rows.min(cols);
    let err_at = |rank: usize| -> Result<(usize, f64), String> {
        let payload = compress_low_rank(matrix, rows, cols, rank)
            .map_err(|e| format!("compress_low_rank rank {rank}: {e:?}"))?;
        Ok((payload.canonical_bytes().len(), payload.frobenius_error))
    };

    // Establish an upper bound that meets tolerance (doubling).
    let mut hi = 1usize;
    let mut hi_result = err_at(hi)?;
    while hi_result.1 > tolerance && hi < max_rank {
        hi = (hi * 2).min(max_rank);
        hi_result = err_at(hi)?;
    }
    if hi_result.1 > tolerance {
        return Ok(None); // even full rank misses (degenerate input)
    }
    // Binary search in (hi/2, hi] for the smallest passing rank.
    let mut lo = if hi == 1 { 1 } else { hi / 2 + 1 };
    let mut best = (hi, hi_result.0, hi_result.1);
    while lo < best.0 {
        let mid = lo + (best.0 - lo) / 2;
        let (bytes, err) = err_at(mid)?;
        if err <= tolerance {
            best = (mid, bytes, err);
        } else {
            lo = mid + 1;
        }
    }
    let raw = 8 * rows * cols;
    if best.1 >= raw {
        return Ok(None); // within tolerance but not actually smaller
    }
    Ok(Some(best))
}

/// Candidate locations for the real chess-RL checkpoint: this
/// worktree's `bench_results`, then the MAIN repo's (worktrees live
/// under `<main>/.claude/worktrees/<name>`, so three `..` reach it).
pub fn checkpoint_candidates() -> Vec<PathBuf> {
    let root = workspace_root();
    vec![
        root.join("bench_results/chess_rl_v2_1/checkpoint_ep60.bin"),
        root.join("../../../bench_results/chess_rl_v2_1/checkpoint_ep60.bin"),
    ]
}

/// Measure one checkpoint file. `Err` only for a present-but-broken
/// file; absence is the caller's graceful-skip case.
///
/// Two real formats exist (research-doc claim corrected on contact):
/// the chess-RL checkpoints are `cjc_runtime::tensor_snap` lists
/// (magic `CJCT` — a hash-footed flat tensor list), NOT `cjc-snap`
/// value files (magic `CJCS`). Both are handled; `CJCT` first because
/// that is what the named artifact actually is.
pub fn measure_checkpoint(path: &PathBuf) -> Result<CheckpointOutcome, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let file_bytes = raw.len();
    let mut tensors: Vec<(String, Tensor)> = Vec::new();
    match cjc_runtime::tensor_snap::decode_list(&raw) {
        Ok(list) => {
            for (i, t) in list.into_iter().enumerate() {
                tensors.push((format!("ckpt[{i}]"), t));
            }
        }
        Err(_) => {
            // Not a tensor-snap list — try the cjc-snap value format.
            let value = cjc_snap::persist::snap_load(&path.to_string_lossy())
                .map_err(|e| format!("neither tensor_snap nor snap format: {e}"))?;
            collect_tensors(&value, "ckpt".to_string(), &mut tensors);
        }
    }
    if tensors.is_empty() {
        return Err(format!("{}: no tensors found", path.display()));
    }

    let mut matrices = Vec::new();
    let mut passthrough_bytes = 0usize;
    for (tensor_path, tensor) in &tensors {
        let shape = tensor.shape().to_vec();
        if shape.len() == 2 && shape[0] >= 2 && shape[1] >= 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let data = tensor.to_vec();
            let original_bytes = 8 * rows * cols;
            match smallest_rank_within(&data, rows, cols, CHECKPOINT_TOLERANCE)? {
                Some((rank, bytes, err)) => matrices.push(TensorLowRankOutcome {
                    path: tensor_path.clone(),
                    rows,
                    cols,
                    original_bytes,
                    chosen_rank: Some(rank),
                    compressed_bytes: Some(bytes),
                    frobenius_error: Some(err),
                }),
                None => matrices.push(TensorLowRankOutcome {
                    path: tensor_path.clone(),
                    rows,
                    cols,
                    original_bytes,
                    chosen_rank: None,
                    compressed_bytes: None,
                    frobenius_error: None,
                }),
            }
        } else {
            // Scalars (empty shape) count as one element.
            let elems: usize = shape.iter().product::<usize>().max(1);
            passthrough_bytes += 8 * elems;
        }
    }

    Ok(CheckpointOutcome {
        source: path.display().to_string(),
        file_bytes,
        matrices,
        passthrough_bytes,
    })
}

// =============================================================================
// Prototype 3 — committed disk artifacts, byte-level
// =============================================================================

/// Committed artifacts measured byte-level. Relative to the workspace
/// root; all are in git, so this section of the report is reproducible
/// from a clean checkout.
pub const DISK_ARTIFACTS: &[&str] = &[
    "bench_results/cana_ablation/profiles.cpdb",
    "bench_results/cana_diagnostics/phases.csv",
];

/// One disk artifact's measurement.
#[derive(Debug, Clone)]
pub struct DiskArtifactOutcome {
    pub path: String,
    pub outcome: ByteCompressionOutcome,
}

pub fn measure_disk_artifact(rel_path: &str) -> Result<DiskArtifactOutcome, String> {
    let full = workspace_root().join(rel_path);
    let bytes =
        std::fs::read(&full).map_err(|e| format!("cannot read {}: {e}", full.display()))?;
    let outcome = measure_lossless(&bytes);
    if !outcome.roundtrips_ok() {
        return Err(format!("{rel_path}: lossless roundtrip failed"));
    }
    Ok(DiskArtifactOutcome {
        path: rel_path.to_string(),
        outcome,
    })
}

// =============================================================================
// Report
// =============================================================================

fn ratio_cell(o: &ByteCompressionOutcome) -> String {
    format!(
        "{} → RLE {} ({:.2}×) / motif {} ({:.2}×)",
        o.original_bytes,
        o.rle_bytes,
        o.rle_ratio(),
        o.motif_bytes,
        o.motif_ratio()
    )
}

/// Render the Phase E report. All numbers are measured by this run;
/// the checkpoint section is `None` when the artifact is absent on
/// this machine (it is not committed to git).
pub fn render_report(
    traces: &[TraceStreamOutcome],
    checkpoint: Option<&CheckpointOutcome>,
    disks: &[DiskArtifactOutcome],
) -> String {
    let mut md = String::new();
    md.push_str("# Phase E — compression prototypes: before/after bytes at bounded error\n\n");
    md.push_str(
        "All lossless measurements are roundtrip-verified byte-exactly (RLE and\n\
         motif codecs) and, for trace streams, decoded back to bit-exact events\n\
         — reconstruction error is ZERO by proof, not by assumption. The\n\
         checkpoint section is lossy-advisory: each matrix is compressed to the\n\
         smallest rank whose relative Frobenius error stays within the\n\
         tolerance, and matrices that cannot beat raw storage are kept raw.\n\n",
    );

    md.push_str("## Prototype 1 — instrumented trace streams (lossless, zero error)\n\n");
    md.push_str(&format!(
        "Streams from instrumented runs of {} hash-pinned Phase-D subjects \
         (seed {SEED}); events capped at {TRACE_EVENT_CAP} per stream — \
         truncation, when it happens, is printed per row.\n\n",
        TRACE_SUBJECTS.len()
    ));
    md.push_str("| subject | events (measured/total) | canonical bytes → codecs | delta-columnar bytes → codecs | best ratio |\n");
    md.push_str("|---|---|---|---|---|\n");
    for t in traces {
        let best = t
            .canonical
            .rle_ratio()
            .max(t.canonical.motif_ratio())
            .max(t.delta.rle_ratio())
            .max(t.delta.motif_ratio());
        md.push_str(&format!(
            "| {} | {}/{} | {} | {} | **{:.2}×** |\n",
            t.subject,
            t.measured_events,
            t.total_events,
            ratio_cell(&t.canonical),
            ratio_cell(&t.delta),
            best
        ));
    }
    md.push('\n');

    md.push_str("## Prototype 2 — checkpoint low-rank (advisory, rel-Frobenius ≤ ");
    md.push_str(&format!("{CHECKPOINT_TOLERANCE})\n\n"));
    match checkpoint {
        Some(c) => {
            md.push_str(&format!(
                "Source: `{}` ({} file bytes). Diagnostic checkpoints only — \
                 never training-resumption paths.\n\n",
                c.source, c.file_bytes
            ));
            md.push_str("| tensor | shape | raw bytes | chosen rank | payload bytes | rel-Frobenius |\n");
            md.push_str("|---|---|---|---|---|---|\n");
            for m in &c.matrices {
                match (m.chosen_rank, m.compressed_bytes, m.frobenius_error) {
                    (Some(rank), Some(bytes), Some(err)) => md.push_str(&format!(
                        "| {} | {}×{} | {} | {} | {} | {:.4} |\n",
                        m.path, m.rows, m.cols, m.original_bytes, rank, bytes, err
                    )),
                    _ => md.push_str(&format!(
                        "| {} | {}×{} | {} | — (kept raw) | {} | — |\n",
                        m.path, m.rows, m.cols, m.original_bytes, m.original_bytes
                    )),
                }
            }
            let (before, after) = c.totals();
            md.push_str(&format!(
                "\nTensor payload totals (incl. {} passthrough bytes for non-2-D \
                 tensors, kept raw): **{} → {} bytes ({:.2}×)**\n\n",
                c.passthrough_bytes,
                before,
                after,
                before as f64 / after.max(1) as f64
            ));
        }
        None => md.push_str(
            "SKIPPED — no checkpoint artifact found on this machine (the chess-RL \
             checkpoints are not committed to git; pass a path as the first CLI \
             argument to measure one).\n\n",
        ),
    }

    md.push_str("## Prototype 3 — committed disk artifacts (lossless)\n\n");
    md.push_str("| artifact | bytes → codecs |\n|---|---|\n");
    for d in disks {
        md.push_str(&format!("| {} | {} |\n", d.path, ratio_cell(&d.outcome)));
    }
    md.push_str(
        "\nHard wall unchanged: these are diagnostics artifacts; nothing here\n\
         feeds compile decisions, hashes, or profile-row stable fields.\n",
    );
    md
}
