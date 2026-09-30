use crate::guard::unreachable;

pub const ALPHA: f64 = -1.586134342059924;
pub const BETA: f64 = -0.052980118572961;
pub const GAMMA: f64 = 0.882911075530934;
pub const DELTA: f64 = 0.443506852043971;
pub const K: f64 = 1.230174104914001;

pub fn reflect(i: isize, n: usize) -> usize {
    if n < 2 {
        unreachable("fit::transform::cdf97::reflect", &format!("length {n} has no interior"));
    }
    let span = 2 * (n as isize) - 2;
    let folded = i.rem_euclid(span);
    if folded >= n as isize {
        (span - folded) as usize
    } else {
        folded as usize
    }
}

fn lift_odd(buf: &mut [f64], coefficient: f64) {
    let n = buf.len();
    for i in (1..n).step_by(2) {
        let left = buf[reflect(i as isize - 1, n)];
        let right = buf[reflect(i as isize + 1, n)];
        buf[i] += coefficient * (left + right);
    }
}

fn lift_even(buf: &mut [f64], coefficient: f64) {
    let n = buf.len();
    for i in (0..n).step_by(2) {
        let left = buf[reflect(i as isize - 1, n)];
        let right = buf[reflect(i as isize + 1, n)];
        buf[i] += coefficient * (left + right);
    }
}

pub fn forward(buf: &mut [f64]) {
    let half = checked_half(buf.len(), "fit::transform::cdf97::forward");
    lift_odd(buf, ALPHA);
    lift_even(buf, BETA);
    lift_odd(buf, GAMMA);
    lift_even(buf, DELTA);
    let mut out = vec![0.0f64; buf.len()];
    for i in 0..half {
        out[i] = buf[2 * i] / K;
        out[half + i] = buf[2 * i + 1] * K;
    }
    buf.copy_from_slice(&out);
}

pub fn inverse(buf: &mut [f64]) {
    let half = checked_half(buf.len(), "fit::transform::cdf97::inverse");
    let mut out = vec![0.0f64; buf.len()];
    for i in 0..half {
        out[2 * i] = buf[i] * K;
        out[2 * i + 1] = buf[half + i] / K;
    }
    buf.copy_from_slice(&out);
    lift_even(buf, -DELTA);
    lift_odd(buf, -GAMMA);
    lift_even(buf, -BETA);
    lift_odd(buf, -ALPHA);
}

fn checked_half(n: usize, site: &'static str) -> usize {
    if n < 4 || n % 2 != 0 {
        unreachable(site, &format!("length {n} is not an even length of at least four"));
    }
    n / 2
}
