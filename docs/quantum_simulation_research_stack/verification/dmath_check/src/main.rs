//! Dump `cjc_repro::dmath` outputs as hex for accuracy checking against
//! mpmath (`dmath_check.py`) and for cross-platform hash comparison.
//!
//!     cargo run --release > dmath_out.txt
//!
//! Each line: `fn x_bits y_bits` (hex).
use cjc_repro::dmath;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn main() {
    let mut s = 0x5eed_u64;
    let mut xs: Vec<f64> = Vec::new();
    for _ in 0..40_000 {
        let u = (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        xs.push((u * 2.0 - 1.0) * 8.0 * std::f64::consts::PI); // gate range
    }
    for _ in 0..20_000 {
        let u = (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        xs.push((u * 2.0 - 1.0) * std::f64::consts::FRAC_PI_4); // kernel range
    }
    for _ in 0..40_000 {
        let r = splitmix(&mut s);
        let b = (r & !(0x7ffu64 << 52)) | ((splitmix(&mut s) % 0x7ff) << 52);
        xs.push(f64::from_bits(b)); // every exponent
    }
    for k in -2000i64..=2000 {
        xs.push(k as f64 * std::f64::consts::FRAC_PI_2);
    }
    xs.push(6381956970095103.0 * 2f64.powi(797));
    for &x in &xs {
        println!("sin {:016x} {:016x}", x.to_bits(), dmath::sin(x).to_bits());
        println!("cos {:016x} {:016x}", x.to_bits(), dmath::cos(x).to_bits());
    }
    for _ in 0..60_000 {
        let u = (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let x = (u * 2.0 - 1.0) * 745.0;
        println!("exp {:016x} {:016x}", x.to_bits(), dmath::exp(x).to_bits());
    }
    for _ in 0..60_000 {
        let r = splitmix(&mut s) & !(1u64 << 63); // positive, any exponent incl. subnormal
        let x = f64::from_bits(r % 0x7ff0_0000_0000_0000);
        println!("ln {:016x} {:016x}", x.to_bits(), dmath::ln(x).to_bits());
    }
    for _ in 0..20_000 {
        let u = (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let x = 0.5 + u; // ln near 1 (cancellation region)
        println!("ln {:016x} {:016x}", x.to_bits(), dmath::ln(x).to_bits());
    }
}
