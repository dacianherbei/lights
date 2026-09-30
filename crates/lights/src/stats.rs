
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stat {
    pub mean: f64,
    pub ci95: f64,
    pub min: f64,
}

pub const CI95_Z: f64 = 1.96;

pub fn stat(samples: &[f64]) -> Stat {
    if samples.is_empty() {
        return Stat::default();
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let min = samples.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    if samples.len() < 2 {
        return Stat {
            mean,
            ci95: 0.0,
            min,
        };
    }
    let variance = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / (n - 1.0);
    Stat {
        mean,
        ci95: CI95_Z * variance.sqrt() / n.sqrt(),
        min,
    }
}

pub fn intervals_overlap(a: Stat, b: Stat) -> bool {
    a.mean - a.ci95 <= b.mean + b.ci95 && b.mean - b.ci95 <= a.mean + a.ci95
}
