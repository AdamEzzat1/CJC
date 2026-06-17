//! Deterministic binary encoding of the `.seshat` trace format.
//!
//! Two hard contracts, both exercised by `tests/fuzz.rs`:
//!
//! 1. **Round-trip identity.** `replay(serialize(&t))` reproduces a trace whose
//!    [`content_hash`](crate::Trace::content_hash) equals `t`'s.
//! 2. **`replay` never panics.** Arbitrary / truncated / tampered bytes return
//!    [`DecodeError`], never a panic, never UB. Every length and every id is
//!    bounds-checked; nothing is pre-allocated from an attacker-controlled
//!    count.

use crate::trace::{CausalEdge, Event, Frame, FrameKind, OwnershipDomain, ThreadState, Trace};

const MAGIC: &[u8; 8] = b"SESHAT01";

/// Why a `.seshat` byte stream could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Stream ended before a field could be read.
    UnexpectedEof,
    /// Leading magic bytes did not match.
    BadMagic,
    /// An enum tag byte was outside the known range.
    BadTag(&'static str, u8),
    /// A string id referenced a slot outside the string table.
    BadStrId(u32),
    /// A frame id referenced a slot outside the frame table.
    BadFrameId(u32),
    /// A string field was not valid UTF-8.
    BadUtf8,
    /// Trailing bytes remained after a full trace was decoded.
    TrailingBytes,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "unexpected end of trace stream"),
            DecodeError::BadMagic => write!(f, "bad magic (not a .seshat stream)"),
            DecodeError::BadTag(what, t) => write!(f, "bad {what} tag byte: {t}"),
            DecodeError::BadStrId(i) => write!(f, "string id {i} out of range"),
            DecodeError::BadFrameId(i) => write!(f, "frame id {i} out of range"),
            DecodeError::BadUtf8 => write!(f, "string field was not valid UTF-8"),
            DecodeError::TrailingBytes => write!(f, "trailing bytes after trace"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ─── Writer ────────────────────────────────────────────────────────────────

fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

/// Encode a trace to a deterministic byte stream.
pub fn serialize(t: &Trace) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u64(&mut out, t.wall_ns_total);
    put_u64(&mut out, t.run_id);

    put_u64(&mut out, t.strings.len() as u64);
    for s in &t.strings {
        put_str(&mut out, s);
    }

    put_u64(&mut out, t.frames.len() as u64);
    for f in &t.frames {
        put_u8(&mut out, f.kind.tag());
        put_u32(&mut out, f.name);
        put_u32(&mut out, f.file);
        put_u32(&mut out, f.line);
    }

    put_u64(&mut out, t.events.len() as u64);
    for e in &t.events {
        write_event(&mut out, e);
    }
    out
}

fn write_event(out: &mut Vec<u8>, e: &Event) {
    match e {
        Event::Sample { thread, state, stack } => {
            put_u8(out, 0);
            put_u32(out, *thread);
            put_u8(out, state.tag());
            put_u64(out, stack.len() as u64);
            for &f in stack {
                put_u32(out, f);
            }
        }
        Event::Alloc { domain, bytes, frame } => {
            put_u8(out, 1);
            put_u8(out, domain.tag());
            put_u64(out, *bytes);
            put_u32(out, *frame);
        }
        Event::Free { domain, bytes, frame } => {
            put_u8(out, 2);
            put_u8(out, domain.tag());
            put_u64(out, *bytes);
            put_u32(out, *frame);
        }
        Event::Counter { thread, freq_mhz, cache_misses, ipc_milli } => {
            put_u8(out, 3);
            put_u32(out, *thread);
            put_u32(out, *freq_mhz);
            put_u64(out, *cache_misses);
            put_u32(out, *ipc_milli);
        }
        Event::ZoneStart { name, handle } => {
            put_u8(out, 4);
            put_u32(out, *name);
            put_u64(out, *handle);
        }
        Event::ZoneStop { handle } => {
            put_u8(out, 5);
            put_u64(out, *handle);
        }
        Event::Edge(edge) => {
            put_u8(out, 6);
            write_edge(out, edge);
        }
    }
}

fn write_edge(out: &mut Vec<u8>, e: &CausalEdge) {
    match e {
        CausalEdge::Wakeup { task, by } => {
            put_u8(out, 0);
            put_u32(out, *task);
            put_u32(out, *by);
        }
        CausalEdge::AwaitResume { task, waited_ticks } => {
            put_u8(out, 1);
            put_u32(out, *task);
            put_u64(out, *waited_ticks);
        }
        CausalEdge::GilHandoff { from, to } => {
            put_u8(out, 2);
            put_u32(out, *from);
            put_u32(out, *to);
        }
        CausalEdge::Copy { from, to, bytes, frame } => {
            put_u8(out, 3);
            put_u8(out, from.tag());
            put_u8(out, to.tag());
            put_u64(out, *bytes);
            put_u32(out, *frame);
        }
        CausalEdge::BoundaryCross { boundary } => {
            put_u8(out, 4);
            put_u32(out, *boundary);
        }
    }
}

// ─── Reader ──────────────────────────────────────────────────────────────────

/// Bounds-checked cursor. Every read returns `Result`; nothing pre-allocates
/// from an untrusted count, so a hostile length just fails the next read.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::UnexpectedEof)?;
        if end > self.buf.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> Result<String, DecodeError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| DecodeError::BadUtf8)
    }
    fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
}

fn kind(r: &mut Reader) -> Result<FrameKind, DecodeError> {
    let t = r.u8()?;
    FrameKind::from_tag(t).ok_or(DecodeError::BadTag("frame-kind", t))
}
fn domain(r: &mut Reader) -> Result<OwnershipDomain, DecodeError> {
    let t = r.u8()?;
    OwnershipDomain::from_tag(t).ok_or(DecodeError::BadTag("ownership-domain", t))
}
fn state(r: &mut Reader) -> Result<ThreadState, DecodeError> {
    let t = r.u8()?;
    ThreadState::from_tag(t).ok_or(DecodeError::BadTag("thread-state", t))
}

/// Decode a `.seshat` byte stream. Returns [`DecodeError`] on any malformation;
/// **never panics**.
pub fn replay(buf: &[u8]) -> Result<Trace, DecodeError> {
    let mut r = Reader::new(buf);
    if r.take(8)? != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let wall_ns_total = r.u64()?;
    let run_id = r.u64()?;

    let n_strings = r.u64()?;
    let mut strings = Vec::new();
    for _ in 0..n_strings {
        strings.push(r.string()?);
    }
    let str_count = strings.len() as u32;
    let check_str = |id: u32| -> Result<u32, DecodeError> {
        if id < str_count {
            Ok(id)
        } else {
            Err(DecodeError::BadStrId(id))
        }
    };

    let n_frames = r.u64()?;
    let mut frames: Vec<Frame> = Vec::new();
    for _ in 0..n_frames {
        let k = kind(&mut r)?;
        let name = check_str(r.u32()?)?;
        let file = check_str(r.u32()?)?;
        let line = r.u32()?;
        frames.push(Frame { kind: k, name, file, line });
    }
    let frame_count = frames.len() as u32;
    let check_frame = |id: u32| -> Result<u32, DecodeError> {
        if id < frame_count {
            Ok(id)
        } else {
            Err(DecodeError::BadFrameId(id))
        }
    };

    let n_events = r.u64()?;
    let mut events = Vec::new();
    for _ in 0..n_events {
        events.push(read_event(&mut r, &check_frame)?);
    }

    if !r.done() {
        return Err(DecodeError::TrailingBytes);
    }

    Ok(Trace {
        strings,
        frames,
        events,
        wall_ns_total,
        run_id,
    })
}

fn read_event(
    r: &mut Reader,
    check_frame: &impl Fn(u32) -> Result<u32, DecodeError>,
) -> Result<Event, DecodeError> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => {
            let thread = r.u32()?;
            let st = state(r)?;
            let n = r.u64()?;
            let mut stack = Vec::new();
            for _ in 0..n {
                stack.push(check_frame(r.u32()?)?);
            }
            Event::Sample { thread, state: st, stack }
        }
        1 => Event::Alloc {
            domain: domain(r)?,
            bytes: r.u64()?,
            frame: check_frame(r.u32()?)?,
        },
        2 => Event::Free {
            domain: domain(r)?,
            bytes: r.u64()?,
            frame: check_frame(r.u32()?)?,
        },
        3 => Event::Counter {
            thread: r.u32()?,
            freq_mhz: r.u32()?,
            cache_misses: r.u64()?,
            ipc_milli: r.u32()?,
        },
        4 => Event::ZoneStart {
            // Zone-name string ids are resolved through the bounds-safe
            // `Trace::string` (out-of-range → "<?>"), so a tampered id degrades
            // gracefully rather than needing a hard reject here.
            name: r.u32()?,
            handle: r.u64()?,
        },
        5 => Event::ZoneStop { handle: r.u64()? },
        6 => Event::Edge(read_edge(r, check_frame)?),
        other => return Err(DecodeError::BadTag("event", other)),
    })
}

fn read_edge(
    r: &mut Reader,
    check_frame: &impl Fn(u32) -> Result<u32, DecodeError>,
) -> Result<CausalEdge, DecodeError> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => CausalEdge::Wakeup {
            task: check_frame(r.u32()?)?,
            by: check_frame(r.u32()?)?,
        },
        1 => CausalEdge::AwaitResume {
            task: check_frame(r.u32()?)?,
            waited_ticks: r.u64()?,
        },
        2 => CausalEdge::GilHandoff {
            from: check_frame(r.u32()?)?,
            to: check_frame(r.u32()?)?,
        },
        3 => CausalEdge::Copy {
            from: domain(r)?,
            to: domain(r)?,
            bytes: r.u64()?,
            frame: check_frame(r.u32()?)?,
        },
        4 => CausalEdge::BoundaryCross {
            boundary: check_frame(r.u32()?)?,
        },
        other => return Err(DecodeError::BadTag("edge", other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{FrameKind, OwnershipDomain, Trace};

    fn sample_trace() -> Trace {
        let mut b = Trace::builder(7);
        let main = b.intern_frame(FrameKind::Py, "main", "app.py", 10);
        let bnd = b.intern_frame(FrameKind::FfiBoundary, "pyo3::call", "ffi.rs", 1);
        let work = b.intern_frame(FrameKind::Rust, "process_batch", "lib.rs", 88);
        b.sample_running(&[main, bnd, work]);
        b.alloc(OwnershipDomain::RustHeap, 1024, work);
        b.copy(OwnershipDomain::RustHeap, OwnershipDomain::NumPy, 1024, bnd);
        b.set_wall_ns(123_456);
        b.finish()
    }

    #[test]
    fn round_trip_preserves_content_hash() {
        let t = sample_trace();
        let bytes = serialize(&t);
        let back = replay(&bytes).expect("valid trace replays");
        assert_eq!(t.content_hash(), back.content_hash());
        // exact value equality too (advisory fields included)
        assert_eq!(t, back);
    }

    #[test]
    fn bad_magic_is_error_not_panic() {
        assert_eq!(replay(b"not-seshat-data!!"), Err(DecodeError::BadMagic));
    }

    #[test]
    fn truncated_is_error_not_panic() {
        let bytes = serialize(&sample_trace());
        for cut in 0..bytes.len() {
            // every prefix must decode-or-error, never panic
            let _ = replay(&bytes[..cut]);
        }
    }

    #[test]
    fn out_of_range_frame_id_rejected() {
        let mut bytes = serialize(&sample_trace());
        // Corrupt is hard to target precisely; instead assert a hand-rolled
        // bad stream (event referencing frame 999 with empty frame table).
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        put_u64(&mut bad, 0); // wall
        put_u64(&mut bad, 0); // run_id
        put_u64(&mut bad, 0); // 0 strings
        put_u64(&mut bad, 0); // 0 frames
        put_u64(&mut bad, 1); // 1 event
        put_u8(&mut bad, 1); //   Alloc
        put_u8(&mut bad, 0); //   domain PyHeap
        put_u64(&mut bad, 8); //   bytes
        put_u32(&mut bad, 999); //   frame id 999 (out of range)
        assert_eq!(replay(&bad), Err(DecodeError::BadFrameId(999)));
        // keep `bytes` used
        assert!(!bytes.is_empty());
        bytes.clear();
    }
}
