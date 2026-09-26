// Emits the exact f64 bit patterns cjc-quantum's rotation gates compute:
// gates.rs rx/ry/rz_matrix use (theta / 2.0).cos() and (theta / 2.0).sin().
// Build: rustc -O libm_bits.rs && ./libm_bits > libm_bits_<platform>.txt
fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn main() {
    use std::f64::consts::PI;
    let mut angles: Vec<f64> = vec![
        0.0, PI / 7.0, PI / 4.0, PI / 3.0, PI / 2.0, 2.0 * PI / 3.0, 3.0 * PI / 4.0, PI,
        -PI, 2.0 * PI, 1.0, 1.57, 0.5, 0.1, 0.3, 1e-300, 1e-8, 1e6, 1e15, 1e300,
    ];
    let mut s = 42u64;
    for _ in 0..200_000 {
        let u = (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        angles.push((u * 2.0 - 1.0) * 4.0 * PI); // gate-realistic range [-4pi, 4pi]
    }
    for _ in 0..20_000 {
        let u = (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        angles.push((u * 2.0 - 1.0) * 1e6); // large-argument stress
    }
    for t in angles {
        let h = t / 2.0;
        println!("{:016x} {:016x} {:016x}", t.to_bits(), h.cos().to_bits(), h.sin().to_bits());
    }
}
