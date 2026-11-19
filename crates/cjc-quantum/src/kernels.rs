//! Dense-statevector gate kernels: strided, branch-free, optionally threaded.
//!
//! # Why the results are bit-identical to the reference kernels
//!
//! A gate acts on disjoint groups of amplitudes (pairs for single-qubit gates,
//! CNOT, and SWAP; single entries for CZ). Every group is updated with exactly
//! the same `mul_fixed` / `add` / `swap` / `neg` sequence as the reference
//! loop in `gates.rs`, so the order in which groups are visited, and which
//! thread visits them, cannot change a single bit. The kernels only change
//! *which indices are enumerated*: the reference loops scan all 2ⁿ indices
//! and skip the ones a gate does not start from; these enumerate the starting
//! indices directly. `tests` compare against the reference kernels bit for
//! bit, for every gate, qubit position, and thread count 1–8.
//!
//! # Threads
//!
//! The work is split into contiguous ranges of groups, one per thread
//! (`std::thread::scope`, no dependencies). The thread count comes from the
//! CJC runtime policy (`cjc_runtime::runtime_policy::current_effective_threads`,
//! ADR-0025), so `--threads` and thermal modes apply. Small states stay on one
//! thread, because spawning costs more than the gate.

use cjc_runtime::complex::ComplexF64;

/// Minimum groups per thread before a gate is split across threads.
const MIN_GROUPS_PER_THREAD: usize = 1 << 15;

/// A raw pointer that may cross into scoped threads. Safety: every thread
/// writes a disjoint set of indices (see each kernel).
#[derive(Clone, Copy)]
struct Shared(*mut ComplexF64);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// Threads to use for `groups` independent work items.
fn thread_count(groups: usize) -> usize {
    let cap = cjc_runtime::runtime_policy::current_effective_threads().max(1);
    cap.min(groups / MIN_GROUPS_PER_THREAD).max(1)
}

/// Run `body(start, end)` over `0..groups`, split into contiguous ranges.
fn for_ranges(groups: usize, threads: usize, body: impl Fn(usize, usize) + Sync) {
    if threads <= 1 {
        body(0, groups);
        return;
    }
    let chunk = (groups + threads - 1) / threads;
    std::thread::scope(|s| {
        for t in 1..threads {
            let (start, end) = (t * chunk, ((t + 1) * chunk).min(groups));
            if start < end {
                let body = &body;
                s.spawn(move || body(start, end));
            }
        }
        body(0, chunk.min(groups));
    });
}

/// Insert a zero bit at position `pos` of `x`.
#[inline(always)]
fn insert_zero(x: usize, pos: usize) -> usize {
    ((x >> pos) << (pos + 1)) | (x & ((1usize << pos) - 1))
}

/// Single-qubit gate `u` on qubit `q`.
pub fn apply_single_qubit(amps: &mut [ComplexF64], q: usize, u: [[ComplexF64; 2]; 2]) {
    apply_single_qubit_threads(amps, q, u, thread_count(amps.len() / 2));
}

/// [`apply_single_qubit`] with an explicit thread count (tests, benchmarks).
pub fn apply_single_qubit_threads(
    amps: &mut [ComplexF64],
    q: usize,
    u: [[ComplexF64; 2]; 2],
    threads: usize,
) {
    let bit = 1usize << q;
    let pairs = amps.len() / 2;
    let p = Shared(amps.as_mut_ptr());
    for_ranges(pairs, threads, move |start, end| {
        let base_ptr = p;
        // Pair k (0 ≤ k < 2ⁿ⁻¹) is (i, i + bit) with i = insert_zero(k, q).
        // Walk k in runs that stay inside one block, so the inner loop is
        // contiguous (and auto-vectorisable).
        let mut k = start;
        while k < end {
            let within = k & (bit - 1);
            let run = (bit - within).min(end - k);
            let i0 = insert_zero(k, q);
            for off in 0..run {
                let i = i0 + off;
                let j = i + bit;
                // SAFETY: i and j < amps.len(); pair (i, j) belongs to this
                // range only, so no other thread touches these indices.
                unsafe {
                    let a0 = *base_ptr.0.add(i);
                    let a1 = *base_ptr.0.add(j);
                    *base_ptr.0.add(i) = u[0][0].mul_fixed(a0).add(u[0][1].mul_fixed(a1));
                    *base_ptr.0.add(j) = u[1][0].mul_fixed(a0).add(u[1][1].mul_fixed(a1));
                }
            }
            k += run;
        }
    });
}

/// Swap `i(k)` with `i(k) | flip` for every k: the shared shape of CNOT,
/// SWAP, and Toffoli. `zero_bits` (ascending) are the positions inserted as 0;
/// `set` is ORed into the first index and `flip` distinguishes the partner.
fn swap_pairs(amps: &mut [ComplexF64], zero_bits: &[usize], set: usize, partner: usize, threads: usize) {
    let groups = amps.len() >> zero_bits.len();
    let p = Shared(amps.as_mut_ptr());
    let zb: Vec<usize> = zero_bits.to_vec();
    for_ranges(groups, threads, move |start, end| {
        let base_ptr = p;
        for k in start..end {
            let mut i = k;
            for &b in &zb {
                i = insert_zero(i, b);
            }
            // SAFETY: the (i | set, i | partner) pairs are disjoint across k.
            unsafe {
                std::ptr::swap(base_ptr.0.add(i | set), base_ptr.0.add(i | partner));
            }
        }
    });
}

fn sorted2(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

/// CNOT(ctrl, tgt): swap (ctrl=1, tgt=0) with (ctrl=1, tgt=1).
pub fn apply_cnot(amps: &mut [ComplexF64], ctrl: usize, tgt: usize) {
    apply_cnot_threads(amps, ctrl, tgt, thread_count(amps.len() / 4));
}

pub fn apply_cnot_threads(amps: &mut [ComplexF64], ctrl: usize, tgt: usize, threads: usize) {
    let (c, t) = (1usize << ctrl, 1usize << tgt);
    swap_pairs(amps, &sorted2(ctrl, tgt), c, c | t, threads);
}

/// SWAP(a, b): swap (a=0, b=1) with (a=1, b=0).
pub fn apply_swap(amps: &mut [ComplexF64], a: usize, b: usize) {
    apply_swap_threads(amps, a, b, thread_count(amps.len() / 4));
}

pub fn apply_swap_threads(amps: &mut [ComplexF64], a: usize, b: usize, threads: usize) {
    swap_pairs(amps, &sorted2(a, b), 1usize << b, 1usize << a, threads);
}

/// Toffoli(c1, c2, tgt): swap (c1=1, c2=1, tgt=0) with tgt=1.
pub fn apply_toffoli(amps: &mut [ComplexF64], c1: usize, c2: usize, tgt: usize) {
    apply_toffoli_threads(amps, c1, c2, tgt, thread_count(amps.len() / 8));
}

pub fn apply_toffoli_threads(amps: &mut [ComplexF64], c1: usize, c2: usize, tgt: usize, threads: usize) {
    let mut bits = [c1, c2, tgt];
    bits.sort_unstable();
    let set = (1usize << c1) | (1usize << c2);
    swap_pairs(amps, &bits, set, set | (1usize << tgt), threads);
}

/// CZ(a, b): negate amplitudes with a=1 and b=1.
pub fn apply_cz(amps: &mut [ComplexF64], a: usize, b: usize) {
    apply_cz_threads(amps, a, b, thread_count(amps.len() / 4));
}

pub fn apply_cz_threads(amps: &mut [ComplexF64], a: usize, b: usize, threads: usize) {
    let bits = sorted2(a, b);
    let set = (1usize << a) | (1usize << b);
    let groups = amps.len() / 4;
    let p = Shared(amps.as_mut_ptr());
    for_ranges(groups, threads, move |start, end| {
        let base_ptr = p;
        for k in start..end {
            let i = insert_zero(insert_zero(k, bits[0]), bits[1]) | set;
            // SAFETY: each i is produced by exactly one k.
            unsafe {
                *base_ptr.0.add(i) = (*base_ptr.0.add(i)).neg();
            }
        }
    });
}

/// |aᵢ|² for every amplitude (same `norm_sq` as the reference), threaded.
pub fn probabilities(amps: &[ComplexF64]) -> Vec<f64> {
    let n = amps.len();
    let threads = thread_count(n / 2);
    if threads <= 1 {
        return amps.iter().map(|a| a.norm_sq()).collect();
    }
    let mut out = vec![0.0f64; n];
    let chunk = (n + threads - 1) / threads;
    std::thread::scope(|s| {
        for (dst, src) in out.chunks_mut(chunk).zip(amps.chunks(chunk)) {
            s.spawn(move || {
                for (d, a) in dst.iter_mut().zip(src) {
                    *d = a.norm_sq();
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;
    use crate::statevector::Statevector;

    fn random_state(n: usize, seed: u64) -> Vec<ComplexF64> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        (0..1usize << n)
            .map(|k| {
                // Include exact zeros and negative zeros, where sign handling shows.
                match k % 7 {
                    0 => ComplexF64::new(0.0, -0.0),
                    _ => ComplexF64::new(
                        (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                        (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                    ),
                }
            })
            .collect()
    }

    fn bits(v: &[ComplexF64]) -> Vec<(u64, u64)> {
        v.iter().map(|a| (a.re.to_bits(), a.im.to_bits())).collect()
    }

    fn reference(gate: &Gate, amps: &[ComplexF64]) -> Vec<ComplexF64> {
        let mut sv = Statevector::from_amplitudes(amps.to_vec()).unwrap();
        crate::gates::apply_reference(gate, &mut sv);
        sv.amplitudes
    }

    #[test]
    fn kernels_match_reference_for_every_gate_position_and_thread_count() {
        for n in [1usize, 2, 3, 5, 7] {
            let base = random_state(n, n as u64 * 31);
            let mut gates = Vec::new();
            for q in 0..n {
                gates.extend([
                    Gate::H(q), Gate::X(q), Gate::Y(q), Gate::Z(q), Gate::S(q), Gate::T(q),
                    Gate::Rx(q, 0.3), Gate::Ry(q, -1.7), Gate::Rz(q, 2.9),
                ]);
                for r in 0..n {
                    if r != q {
                        gates.extend([Gate::CNOT(q, r), Gate::CZ(q, r), Gate::SWAP(q, r)]);
                        for t in 0..n {
                            if t != q && t != r {
                                gates.push(Gate::Toffoli(q, r, t));
                            }
                        }
                    }
                }
            }
            for g in &gates {
                let want = bits(&reference(g, &base));
                for threads in [1usize, 2, 3, 8] {
                    let mut amps = base.clone();
                    match *g {
                        Gate::CNOT(c, t) => apply_cnot_threads(&mut amps, c, t, threads),
                        Gate::CZ(a, b) => apply_cz_threads(&mut amps, a, b, threads),
                        Gate::SWAP(a, b) => apply_swap_threads(&mut amps, a, b, threads),
                        Gate::Toffoli(a, b, c) => apply_toffoli_threads(&mut amps, a, b, c, threads),
                        _ => {
                            let (q, u) = crate::gates::single_qubit_matrix(g).unwrap();
                            apply_single_qubit_threads(&mut amps, q, u, threads);
                        }
                    }
                    assert_eq!(bits(&amps), want, "{:?} n={} threads={}", g, n, threads);
                }
            }
        }
    }

    #[test]
    fn threaded_probabilities_match_serial() {
        let amps = random_state(17, 5);
        let serial: Vec<u64> = amps.iter().map(|a| a.norm_sq().to_bits()).collect();
        let par: Vec<u64> = probabilities(&amps).iter().map(|p| p.to_bits()).collect();
        assert_eq!(serial, par);
    }

    #[test]
    fn large_state_threads_agree() {
        // 2^18 amplitudes: large enough that the policy path really splits.
        let base = random_state(18, 9);
        for g in [Gate::H(0), Gate::Ry(17, 0.4), Gate::CNOT(17, 0), Gate::Toffoli(3, 16, 9)] {
            let want = bits(&reference(&g, &base));
            for threads in [1usize, 4, 7] {
                let mut amps = base.clone();
                match g {
                    Gate::CNOT(c, t) => apply_cnot_threads(&mut amps, c, t, threads),
                    Gate::Toffoli(a, b, c) => apply_toffoli_threads(&mut amps, a, b, c, threads),
                    _ => {
                        let (q, u) = crate::gates::single_qubit_matrix(&g).unwrap();
                        apply_single_qubit_threads(&mut amps, q, u, threads);
                    }
                }
                assert_eq!(bits(&amps), want, "{:?} threads={}", g, threads);
            }
        }
    }
}
