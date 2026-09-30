use crate::guard::unreachable;

pub const HAAR_SCALE: f64 = 1.4142135623730951;

pub fn forward(buf: &mut [f64]) {
    let half = checked_half(buf.len(), "fit::transform::haar::forward");
    let mut out = vec![0.0f64; buf.len()];
    for i in 0..half {
        let even = buf[2 * i];
        let odd = buf[2 * i + 1];
        let detail = odd - even;
        let approx = even + detail * 0.5;
        out[i] = approx * HAAR_SCALE;
        out[half + i] = detail / HAAR_SCALE;
    }
    buf.copy_from_slice(&out);
}

pub fn inverse(buf: &mut [f64]) {
    let half = checked_half(buf.len(), "fit::transform::haar::inverse");
    let mut out = vec![0.0f64; buf.len()];
    for i in 0..half {
        let approx = buf[i] / HAAR_SCALE;
        let detail = buf[half + i] * HAAR_SCALE;
        let even = approx - detail * 0.5;
        out[2 * i] = even;
        out[2 * i + 1] = even + detail;
    }
    buf.copy_from_slice(&out);
}

fn checked_half(n: usize, site: &'static str) -> usize {
    if n < 2 || n % 2 != 0 {
        unreachable(site, &format!("length {n} is not an even length of at least two"));
    }
    n / 2
}
