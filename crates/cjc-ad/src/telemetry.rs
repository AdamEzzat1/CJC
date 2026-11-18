//! Postulate Phase 1 — the per-step training telemetry stream.
//!
//! Opt-in, write-only, sampled. One `backward` call is one step. Every
//! `every`-th step writes one JSON line:
//!
//! ```text
//! {"step":120,"seed":42,"loss":0.0213,"grad_norms":{"n0":0.084},"non_finite":[]}
//! ```
//!
//! `grad_norms` is keyed by parameter node (`n<id>`); a client maps ids to
//! names and lines through the `GraphExport` spans. `non_finite` lists
//! `[node, count]` for every node whose forward value holds a NaN or Inf.
//!
//! Lines go into a fixed-size ring and are flushed to an append-only file when
//! the ring fills, on [`Telemetry::flush`], and on drop. Modelled on `horus`:
//! the probe reads the tape and never writes to it, so a run with telemetry on
//! hashes the same as one with it off (`tests/postulate_phase1.rs`).
//!
//! Collection is deterministic in content (same program and seed → same lines);
//! only the file I/O timing is not.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use crate::export::{l2_norm, push_f64};
use crate::{GradGraph, GradOp};

/// Default sampling interval, in steps.
pub const DEFAULT_EVERY: u64 = 10;
/// Default ring size, in records.
pub const DEFAULT_CAPACITY: usize = 64;

pub struct Telemetry {
    every: u64,
    capacity: usize,
    seed: u64,
    step: u64,
    ring: VecDeque<String>,
    path: Option<PathBuf>,
    file: Option<File>,
    /// Every line written, when the sink is in memory (tests, the LSP runner).
    pub memory: Option<Vec<String>>,
}

impl Telemetry {
    /// Telemetry that appends to `path` (created if missing).
    pub fn to_file(path: impl Into<PathBuf>, seed: u64, every: u64) -> Self {
        Self::new(Some(path.into()), seed, every)
    }

    /// Telemetry that keeps its lines in [`Self::memory`].
    pub fn in_memory(seed: u64, every: u64) -> Self {
        let mut t = Self::new(None, seed, every);
        t.memory = Some(Vec::new());
        t
    }

    fn new(path: Option<PathBuf>, seed: u64, every: u64) -> Self {
        Self {
            every: every.max(1),
            capacity: DEFAULT_CAPACITY,
            seed,
            step: 0,
            ring: VecDeque::with_capacity(DEFAULT_CAPACITY),
            path,
            file: None,
            memory: None,
        }
    }

    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Steps seen so far.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Called after `backward(loss_idx)` on `g`.
    pub fn on_backward(&mut self, g: &GradGraph, loss_idx: usize) {
        let step = self.step;
        self.step += 1;
        if step % self.every != 0 {
            return;
        }
        let line = record(g, loss_idx, step, self.seed);
        self.ring.push_back(line);
        if self.ring.len() >= self.capacity {
            self.flush();
        }
    }

    /// Write out everything in the ring.
    pub fn flush(&mut self) {
        if self.ring.is_empty() {
            return;
        }
        if let Some(mem) = &mut self.memory {
            mem.extend(self.ring.drain(..));
            return;
        }
        if self.file.is_none() {
            if let Some(p) = &self.path {
                self.file = OpenOptions::new().create(true).append(true).open(p).ok();
            }
        }
        if let Some(f) = &mut self.file {
            let mut buf = String::new();
            for l in self.ring.drain(..) {
                buf.push_str(&l);
                buf.push('\n');
            }
            // Telemetry is best-effort: an I/O error must never fail the run.
            let _ = f.write_all(buf.as_bytes());
        } else {
            self.ring.clear();
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.flush();
    }
}

/// One telemetry record for the tape as it stands after a backward pass.
pub fn record(g: &GradGraph, loss_idx: usize, step: u64, seed: u64) -> String {
    let mut o = String::with_capacity(128);
    o.push_str(&format!("{{\"step\":{step},\"seed\":{seed},\"loss\":"));
    let loss = g.tensor(loss_idx).to_vec();
    push_f64(&mut o, loss.first().copied().unwrap_or(f64::NAN));
    o.push_str(",\"grad_norms\":{");
    let mut first = true;
    for i in 0..g.len() {
        if !matches!(g.op(i), GradOp::Parameter) {
            continue;
        }
        if let Some(gr) = g.grad(i) {
            if !first {
                o.push(',');
            }
            first = false;
            o.push_str(&format!("\"n{i}\":"));
            push_f64(&mut o, l2_norm(&gr.to_vec()));
        }
    }
    o.push_str("},\"non_finite\":[");
    let mut first = true;
    for i in 0..g.len() {
        let bad = g.tensor(i).to_vec().iter().filter(|x| !x.is_finite()).count();
        if bad > 0 {
            if !first {
                o.push(',');
            }
            first = false;
            o.push_str(&format!("[{i},{bad}]"));
        }
    }
    o.push_str("]}");
    o
}
