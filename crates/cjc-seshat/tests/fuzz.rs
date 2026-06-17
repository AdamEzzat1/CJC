//! Bolero fuzz targets for Seshat. On Windows these run as proptest; on Linux
//! CI they can be promoted to libfuzzer/AFL coverage-guided fuzzing.
//!
//! Four target classes (per `SESHAT_IMPLEMENTATION_PROMPT.md` §Tests):
//!
//! 1. **Trace decode fuzz** — arbitrary bytes into `replay` must surface
//!    `DecodeError`, never panic.
//! 2. **Structural analysis fuzz** — random builder-op streams into
//!    `analyze_trace` must never panic, stay bounded, and re-hash identically.
//! 3. **Tamper fuzz** — byte flips in a valid trace must not panic.
//! 4. **Builtin-arg fuzz** — random args into `dispatch_seshat` must `Err`,
//!    never panic.

use std::rc::Rc;

use cjc_runtime::value::Value;
use cjc_seshat::{
    analyze_trace, dispatch_seshat, replay, serialize, FrameKind, OwnershipDomain, ThreadState,
    Trace,
};

const MAX_OPS: usize = 64;

/// Build a trace by interpreting a fuzz byte stream as a sequence of 4-byte
/// builder commands. Bounded so a hostile input can't explode the trace.
fn trace_from_bytes(data: &[u8]) -> Trace {
    let mut b = Trace::builder(0);
    // a fixed small frame pool keyed by a byte
    let frames: Vec<u32> = (0..8u32)
        .map(|i| b.intern_frame(kind_for(i as u8), &format!("f{i}"), "fuzz.rs", i))
        .collect();
    let mut open_zones: Vec<u64> = Vec::new();
    let mut i = 0usize;
    let mut ops = 0usize;
    while i + 3 < data.len() && ops < MAX_OPS {
        let (op, a, b2, c) = (data[i], data[i + 1], data[i + 2], data[i + 3]);
        i += 4;
        ops += 1;
        match op % 7 {
            0 => {
                // sample with a stack of up to 4 frames
                let depth = (a % 5) as usize;
                let stack: Vec<u32> = (0..depth)
                    .map(|k| frames[((b2 as usize + k) % frames.len())])
                    .collect();
                b.sample(c as u32, state_for(a), &stack);
            }
            1 => b.alloc(domain_for(a), b2 as u64 * 100, frames[(c as usize) % frames.len()]),
            2 => b.free(domain_for(a), b2 as u64 * 100, frames[(c as usize) % frames.len()]),
            3 => b.copy(
                domain_for(a),
                domain_for(b2),
                c as u64 * 100,
                frames[0],
            ),
            4 => {
                let h = b.zone_start("stage");
                open_zones.push(h);
            }
            5 => {
                if let Some(h) = open_zones.pop() {
                    b.zone_stop(h);
                }
            }
            _ => b.counter(0, 2000 + a as u32 * 10, b2 as u64, c as u32),
        }
    }
    b.finish()
}

fn kind_for(i: u8) -> FrameKind {
    match i % 5 {
        0 => FrameKind::Py,
        1 => FrameKind::Rust,
        2 => FrameKind::Native,
        3 => FrameKind::FfiBoundary,
        _ => FrameKind::AsyncTask,
    }
}
fn domain_for(i: u8) -> OwnershipDomain {
    match i % 8 {
        0 => OwnershipDomain::PyHeap,
        1 => OwnershipDomain::RustHeap,
        2 => OwnershipDomain::Mmap,
        3 => OwnershipDomain::NumPy,
        4 => OwnershipDomain::Arrow,
        5 => OwnershipDomain::Tensor,
        6 => OwnershipDomain::Gpu,
        _ => OwnershipDomain::NativeExt,
    }
}
fn state_for(i: u8) -> ThreadState {
    match i % 6 {
        0 => ThreadState::Running,
        1 => ThreadState::GilWait,
        2 => ThreadState::LockWait,
        3 => ThreadState::ChannelWait,
        4 => ThreadState::IoWait,
        _ => ThreadState::AsyncIdle,
    }
}

#[test]
fn fuzz_decode_never_panics() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|data: &Vec<u8>| {
            // arbitrary bytes: either a valid trace or a DecodeError, never panic
            let _ = replay(data);
        });
}

#[test]
fn fuzz_structural_analysis_bounded_and_stable() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|data: &Vec<u8>| {
            let t = trace_from_bytes(data);
            let r1 = analyze_trace(&t);
            // bounded: boundary share is a valid percentage
            let share = cjc_seshat::analyze::pct_milli(
                r1.boundary.boundary_samples,
                r1.boundary.total_samples,
            );
            assert!(share <= 100_000);
            // contention components sum to the total (no double counting)
            let c = &r1.contention;
            assert_eq!(
                c.running + c.gil_wait + c.lock_wait + c.channel_wait + c.io_wait + c.async_idle,
                c.total
            );
            // analysing again is byte-identical (determinism under fuzz)
            let r2 = analyze_trace(&t);
            assert_eq!(r1.content_hash(), r2.content_hash());
            // and it round-trips
            let back = replay(&serialize(&t)).expect("self-serialized trace replays");
            assert_eq!(t.content_hash(), back.content_hash());
        });
}

#[test]
fn fuzz_tamper_never_panics() {
    bolero::check!()
        .with_type::<(Vec<u8>, u16, u8)>()
        .for_each(|(seed, offset, mask): &(Vec<u8>, u16, u8)| {
            // Build a valid trace, serialize, flip a byte, and try to replay.
            let t = trace_from_bytes(seed);
            let mut bytes = serialize(&t);
            if !bytes.is_empty() {
                let idx = (*offset as usize) % bytes.len();
                bytes[idx] ^= *mask;
            }
            let _ = replay(&bytes); // must not panic
        });
}

#[test]
fn fuzz_builtin_args_never_panic() {
    bolero::check!()
        .with_type::<(u8, i64, i64)>()
        .for_each(|&(sel, x, y): &(u8, i64, i64)| {
            // NB: `seshat_dump_trace` is intentionally excluded — it performs
            // filesystem I/O, which has no place in a fuzz loop. Its arg
            // handling is covered by the unit tests.
            let names = [
                "seshat_reset",
                "seshat_zone_start",
                "seshat_zone_stop",
                "seshat_mark_boundary",
                "seshat_mark_copy",
                "seshat_alloc_tag",
                "seshat_event_count",
                "not_a_seshat_builtin",
            ];
            let name = names[(sel as usize) % names.len()];
            // a grab-bag of arg shapes the parser might hand us
            let argsets: Vec<Vec<Value>> = vec![
                vec![],
                vec![Value::Int(x)],
                vec![Value::Int(x), Value::Int(y)],
                vec![Value::String(Rc::new(format!("{x}")))],
                vec![
                    Value::String(Rc::new("rust".to_string())),
                    Value::String(Rc::new("numpy".to_string())),
                    Value::Int(y),
                ],
                vec![Value::Bool(true), Value::Float(1.5)],
            ];
            for args in &argsets {
                // never panic — Ok(Some)/Ok(None)/Err are all fine.
                let _ = dispatch_seshat(name, args);
            }
        });
}
