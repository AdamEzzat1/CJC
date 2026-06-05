//! Phase 7a.3 — trace-based replay debugging.
//!
//! Gated behind the `trace` Cargo feature. Provides a write-side
//! [`TraceWriter`] that records per-training-step state hashes and
//! losses to a sidecar file, plus a read-side [`bisect_traces`] that
//! finds the first divergent step between two trace files. Designed
//! to be the first thing you reach for when the Phase 5b
//! cross-platform CI matrix reports a byte-identity divergence.
//!
//! # Workflow
//!
//! 1. CI fails: same `(config, seed)` produces different `sweep_hash`
//!    on macOS vs Linux.
//! 2. Developer enables `trace` and re-runs the experiment locally on
//!    each platform, producing `linux.trace` and `macos.trace`.
//! 3. `bisect_traces("linux.trace", "macos.trace")` reports the first
//!    step at which a field differs, and which field. That points the
//!    investigation at a specific subsystem (loss computation,
//!    gradient, optimizer step, …).
//!
//! # Determinism
//!
//! The trace file's bytes are deterministic for a given
//! `(config, seed)` — the writer never reads time, never calls
//! `Instant::now`, never iterates a `HashMap`. Two traces of the same
//! training run on the same platform are byte-identical (verified by
//! the test `trace_files_byte_identical_across_runs`).
//!
//! # Hash function choice
//!
//! Uses FNV-1a (64-bit). Self-contained, no `serde` / `cjc-locke`
//! dependency, byte-identical across platforms. The hash quality
//! requirement is "different inputs ⇒ different hashes with
//! overwhelming probability" — collision resistance isn't relevant
//! here because we're comparing matched pairs, not searching for
//! adversarial inputs.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::CronosGanError;
use crate::gan::{TemporalGan, TemporalGanConfig};
use crate::gan_trainer::{TemporalGanTrainStep, TemporalGanTrainer};
use crate::training::Trainable;

// ─── FNV-1a 64-bit hash ──────────────────────────────────────────────────

/// FNV-1a 64-bit hash of an `f64` slice, byte-identical across runs
/// and platforms. Hashes the slice via `f64::to_bits().to_le_bytes()`
/// — same hash for the same numeric values regardless of host
/// endianness (Cronos GAN targets little-endian platforms anyway, but
/// the LE serialisation makes the hash robust to a hypothetical
/// big-endian port).
pub fn fnv1a_hash_f64s(values: &[f64]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h: u64 = FNV_OFFSET;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

// ─── TraceEvent — one per training step ──────────────────────────────────

/// One per-step trace record. Captures enough state to bisect a
/// cross-platform divergence to its originating step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub step: u64,
    pub ssm_params_pre_hash: u64,
    pub liquid_params_pre_hash: u64,
    pub ssm_loss_bits: u64,
    pub liquid_loss_bits: u64,
    pub absolute_gap_bits: u64,
    pub regime_shift_score_bits: u64,
    pub ssm_params_post_hash: u64,
    pub liquid_params_post_hash: u64,
}

impl TraceEvent {
    /// Render this event as a single tab-separated `key=value` line,
    /// terminated by `\n`. Numeric fields use hex for hashes (16-char
    /// padded) and decimal for `step`; loss/gap/regime fields are
    /// hex-formatted f64 bits so a textual diff is byte-accurate.
    pub fn to_line(&self) -> String {
        format!(
            "step={}\tssm_params_pre=0x{:016x}\tliquid_params_pre=0x{:016x}\t\
             ssm_loss_bits=0x{:016x}\tliquid_loss_bits=0x{:016x}\t\
             absolute_gap_bits=0x{:016x}\tregime_shift_score_bits=0x{:016x}\t\
             ssm_params_post=0x{:016x}\tliquid_params_post=0x{:016x}\n",
            self.step,
            self.ssm_params_pre_hash,
            self.liquid_params_pre_hash,
            self.ssm_loss_bits,
            self.liquid_loss_bits,
            self.absolute_gap_bits,
            self.regime_shift_score_bits,
            self.ssm_params_post_hash,
            self.liquid_params_post_hash,
        )
    }

    /// Field names in the canonical key=value layout. Useful for
    /// `bisect_traces` reports.
    pub const FIELD_NAMES: [&'static str; 9] = [
        "step",
        "ssm_params_pre",
        "liquid_params_pre",
        "ssm_loss_bits",
        "liquid_loss_bits",
        "absolute_gap_bits",
        "regime_shift_score_bits",
        "ssm_params_post",
        "liquid_params_post",
    ];
}

// ─── TraceWriter ─────────────────────────────────────────────────────────

/// Owns a buffered handle to a trace file. Writes one
/// [`TraceEvent`] line per call. Flushed automatically on drop.
pub struct TraceWriter {
    inner: BufWriter<File>,
}

impl TraceWriter {
    /// Create a fresh trace file at `path`, truncating any prior
    /// contents.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            inner: BufWriter::new(file),
        })
    }

    /// Append one event line.
    pub fn write_event(&mut self, event: &TraceEvent) -> io::Result<()> {
        self.inner.write_all(event.to_line().as_bytes())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ─── Trainer integration: step_with_trace ────────────────────────────────

impl TemporalGanTrainer {
    /// Phase 7a.3: like [`Self::step`], but additionally writes one
    /// [`TraceEvent`] to `writer`. The training math is identical —
    /// trace recording only reads the trainer / gan state; it never
    /// mutates them. The default `step()` path is unaffected.
    pub fn step_with_trace(
        &mut self,
        gan: &mut TemporalGan,
        inputs: &[f64],
        targets: &[f64],
        writer: &mut TraceWriter,
    ) -> Result<TemporalGanTrainStep, CronosGanError> {
        // Capture pre-step state hashes. These read the canonical
        // flat-param representation each `Trainable` exposes.
        let step_idx = self.step_count();
        let ssm_pre = fnv1a_hash_f64s(&gan.ssm().params_flat());
        let liquid_pre = fnv1a_hash_f64s(&gan.liquid().params_flat());

        // Run the step normally. This is the only call that touches
        // the model; trace machinery only observes.
        let result = self.step(gan, inputs, targets)?;

        let ssm_post = fnv1a_hash_f64s(&gan.ssm().params_flat());
        let liquid_post = fnv1a_hash_f64s(&gan.liquid().params_flat());

        let event = TraceEvent {
            step: step_idx,
            ssm_params_pre_hash: ssm_pre,
            liquid_params_pre_hash: liquid_pre,
            ssm_loss_bits: result.ssm_loss.to_bits(),
            liquid_loss_bits: result.liquid_loss.to_bits(),
            absolute_gap_bits: result.disagreement.absolute_gap.to_bits(),
            regime_shift_score_bits: result.disagreement.regime_shift_score.to_bits(),
            ssm_params_post_hash: ssm_post,
            liquid_params_post_hash: liquid_post,
        };

        writer
            .write_event(&event)
            .map_err(|e| CronosGanError::Unsupported {
                detail: format!("TraceWriter failed at step {}: {}", step_idx, e),
            })?;

        Ok(result)
    }
}

// Silence unused-import warning when `Trainable`'s `params_flat` is
// only invoked through the impl above (some compilers flag this).
#[allow(dead_code)]
fn _force_trainable_import(_: &dyn Trainable) {
    // no-op; ensures the import isn't pruned by clippy/unused-imports.
}

// Suppress the unused-import note for `TemporalGanConfig` — kept in
// scope for downstream callers that read `gan.config()` after a trace
// step.
#[allow(dead_code)]
fn _config_helper(g: &TemporalGan) -> &TemporalGanConfig {
    g.config()
}

// ─── bisect_traces — diff two trace files ────────────────────────────────

/// Result of comparing two trace files line by line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BisectResult {
    /// The two files are byte-identical. `n_steps` is the number of
    /// events compared.
    Identical { n_steps: u64 },
    /// First divergence found: at line index `step`, field `field`
    /// differs between the two files.
    Divergent {
        step: u64,
        field: &'static str,
        value_a: String,
        value_b: String,
    },
    /// Both files were byte-identical for `prefix_steps` steps, then
    /// one file ended (the other still has more lines). `len_a` /
    /// `len_b` are the total event counts in each.
    LengthMismatch {
        prefix_steps: u64,
        len_a: u64,
        len_b: u64,
    },
}

/// Compare two trace files step by step. Returns the first divergence
/// (or `Identical` if every step matches).
pub fn bisect_traces(
    path_a: impl AsRef<Path>,
    path_b: impl AsRef<Path>,
) -> io::Result<BisectResult> {
    let a = BufReader::new(File::open(path_a)?);
    let b = BufReader::new(File::open(path_b)?);
    let mut lines_a = a.lines();
    let mut lines_b = b.lines();
    let mut step: u64 = 0;
    loop {
        match (lines_a.next(), lines_b.next()) {
            (None, None) => return Ok(BisectResult::Identical { n_steps: step }),
            (Some(la), Some(lb)) => {
                let la = la?;
                let lb = lb?;
                if la == lb {
                    step += 1;
                    continue;
                }
                // Lines differ. Find which field.
                let fields_a: Vec<&str> = la.split('\t').collect();
                let fields_b: Vec<&str> = lb.split('\t').collect();
                for (i, (fa, fb)) in fields_a.iter().zip(fields_b.iter()).enumerate() {
                    if fa != fb {
                        let field_name =
                            TraceEvent::FIELD_NAMES.get(i).copied().unwrap_or("<unknown>");
                        return Ok(BisectResult::Divergent {
                            step,
                            field: field_name,
                            value_a: (*fa).to_string(),
                            value_b: (*fb).to_string(),
                        });
                    }
                }
                // Same field count but different line content (rare —
                // would mean field count mismatched). Report a
                // catch-all.
                return Ok(BisectResult::Divergent {
                    step,
                    field: "<line>",
                    value_a: la,
                    value_b: lb,
                });
            }
            (some, none_) => {
                // One stream ended before the other. Count remainder.
                let mut len_a = step;
                let mut len_b = step;
                if some.is_some() {
                    len_a += 1;
                    for line in lines_a {
                        line?;
                        len_a += 1;
                    }
                }
                if none_.is_some() {
                    len_b += 1;
                    for line in lines_b {
                        line?;
                        len_b += 1;
                    }
                }
                return Ok(BisectResult::LengthMismatch {
                    prefix_steps: step,
                    len_a,
                    len_b,
                });
            }
        }
    }
}
