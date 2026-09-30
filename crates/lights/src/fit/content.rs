use crate::error::Result;
use crate::fit::analytic::AnalyticArm;
use crate::fit::coverage::{coverage_image_to_colour, exact_coverage};
use crate::fit::target::Target;
use crate::guard::Budget;
use crate::image::FloatImage;
use crate::numerics::vector::{Rgb, Vec2};
use crate::scene::demo::Lcg;

pub const RASTER_WIDTH: u32 = 64;
pub const RASTER_HEIGHT: u32 = 64;
pub const CONTROL_ABSENT: &str = "ABSENT";

pub const N1_NAME: &str = "n1-colour-field";
pub const N2_NAME: &str = "n2-soft-shadow";
pub const N3_NAME: &str = "n3-band-limited-noise";
pub const N4_NAME: &str = "n4-broadband-patch";

pub const N1_STOPS: [(f64, f64, f64, f64, f64); 5] = [
    (11.37, 9.13, 0.92, 0.21, 0.17),
    (52.71, 14.29, 0.14, 0.47, 0.88),
    (30.11, 33.53, 0.96, 0.83, 0.19),
    (8.93, 51.67, 0.11, 0.72, 0.38),
    (55.31, 57.19, 0.58, 0.24, 0.79),
];
pub const N1_DISTANCE_FLOOR: f64 = 1.0e-6;

pub const N2_CENTRE: (f64, f64) = (30.83, 27.41);
pub const N2_HALF_EXTENT: (f64, f64) = (17.83, 11.41);
pub const N2_CORNER_RADIUS: f64 = 6.0;
pub const N2_ROTATION_DEGREES: f64 = 23.0;
pub const N2_SHADOW_OFFSET: (f64, f64) = (5.37, 3.11);
pub const N2_BLUR_SIGMA: f64 = 3.0;
pub const N2_BLUR_SIGMAS: f64 = 4.0;
pub const N2_SHADOW_STRENGTH: f64 = 0.55;
pub const N2_PAPER: (f64, f64, f64) = (0.93, 0.92, 0.89);
pub const N2_SHADOW: (f64, f64, f64) = (0.08, 0.07, 0.10);
pub const N2_INK: (f64, f64, f64) = (0.22, 0.35, 0.62);

pub const N3_SEED: u64 = 30_003;
pub const N3_LATTICES: [u32; 2] = [16, 8];
pub const N3_AMPLITUDES: [f64; 2] = [0.65, 0.35];
pub const N3_FLOOR: f64 = 0.08;
pub const N3_SPAN: f64 = 0.84;

pub const N4_SEED_NEAR: u64 = 40_004;
pub const N4_SEED_FAR: u64 = 40_005;
pub const N4_LATTICES: [u32; 3] = [16, 8, 4];
pub const N4_AMPLITUDES: [f64; 3] = [0.55, 0.30, 0.15];
pub const N4_FLOOR: f64 = 0.05;
pub const N4_SPAN: f64 = 0.90;
pub const N4_EDGES: [(f64, f64, f64); 4] = [
    (17.0, 21.43, 18.77),
    (63.0, 41.09, 26.31),
    (104.0, 28.61, 44.13),
    (149.0, 37.77, 37.29),
];

pub fn pixel_sample(px: u32, py: u32) -> Vec2<f64> {
    Vec2::new(px as f64 + 0.5, py as f64 + 0.5)
}

pub fn rgb(triple: (f64, f64, f64)) -> Rgb<f64> {
    Rgb::new(triple.0, triple.1, triple.2)
}

pub fn n1_sample(p: Vec2<f64>) -> Rgb<f64> {
    let mut weight = 0.0f64;
    let mut acc = Rgb::zero();
    for (x, y, r, g, b) in N1_STOPS {
        let d = Vec2::new(p.x - x, p.y - y);
        let w = 1.0 / d.length_squared().max(N1_DISTANCE_FLOOR);
        weight += w;
        acc = acc + Rgb::new(r, g, b) * w;
    }
    acc * (1.0 / weight)
}

pub fn n1_field() -> FloatImage<f64> {
    let mut image = FloatImage::<f64>::new(RASTER_WIDTH, RASTER_HEIGHT);
    for py in 0..RASTER_HEIGHT {
        for px in 0..RASTER_WIDTH {
            image.set(px, py, n1_sample(pixel_sample(px, py)));
        }
    }
    image
}

pub fn n2_rounded_rect_sdf(p: Vec2<f64>, centre: Vec2<f64>) -> f64 {
    let (s, c) = N2_ROTATION_DEGREES.to_radians().sin_cos();
    let d = Vec2::new(p.x - centre.x, p.y - centre.y);
    let local = Vec2::new(d.x * c + d.y * s, -d.x * s + d.y * c);
    let qx = local.x.abs() - (N2_HALF_EXTENT.0 - N2_CORNER_RADIUS);
    let qy = local.y.abs() - (N2_HALF_EXTENT.1 - N2_CORNER_RADIUS);
    let outside = Vec2::new(qx.max(0.0), qy.max(0.0)).length();
    outside + qx.max(qy).min(0.0) - N2_CORNER_RADIUS
}

pub const N2_ERF_MAX_ERROR: f64 = 1.5e-7;

pub fn normal_cdf(x: f64) -> f64 {
    let t = x * std::f64::consts::FRAC_1_SQRT_2;
    0.5 * (1.0 + erf(t))
}

pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-a * a).exp())
}

pub fn n2_soft_shadow_sample(p: Vec2<f64>) -> Rgb<f64> {
    let offset = Vec2::new(
        N2_CENTRE.0 + N2_SHADOW_OFFSET.0,
        N2_CENTRE.1 + N2_SHADOW_OFFSET.1,
    );
    let shadow = normal_cdf(-n2_rounded_rect_sdf(p, offset) / N2_BLUR_SIGMA);
    let a = (shadow * N2_SHADOW_STRENGTH).clamp(0.0, 1.0);
    let under = rgb(N2_PAPER) * (1.0 - a) + rgb(N2_SHADOW) * a;
    let inside = n2_rounded_rect_sdf(p, Vec2::new(N2_CENTRE.0, N2_CENTRE.1));
    if inside < 0.0 { rgb(N2_INK) } else { under }
}

pub fn n2_coverage(centre: Vec2<f64>) -> Vec<f64> {
    let mut plane = vec![0.0f64; (RASTER_WIDTH as usize) * (RASTER_HEIGHT as usize)];
    for py in 0..RASTER_HEIGHT {
        for px in 0..RASTER_WIDTH {
            let d = n2_rounded_rect_sdf(pixel_sample(px, py), centre);
            plane[(py as usize) * (RASTER_WIDTH as usize) + (px as usize)] =
                (0.5 - d).clamp(0.0, 1.0);
        }
    }
    plane
}

pub fn gaussian_taps(sigma: f64, sigmas: f64) -> Vec<f64> {
    let radius = (sigma * sigmas).ceil() as i64;
    let mut taps = Vec::with_capacity((2 * radius + 1) as usize);
    let mut total = 0.0f64;
    for i in -radius..=radius {
        let t = (i as f64) / sigma;
        let w = (-0.5 * t * t).exp();
        total += w;
        taps.push(w);
    }
    for t in taps.iter_mut() {
        *t /= total;
    }
    taps
}

pub fn blur_separable(plane: &[f64], width: u32, height: u32, taps: &[f64]) -> Vec<f64> {
    let radius = (taps.len() as i64 - 1) / 2;
    let stride = width as usize;
    let mut horizontal = vec![0.0f64; plane.len()];
    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let mut acc = 0.0f64;
            for (index, tap) in taps.iter().enumerate() {
                let sx = x + (index as i64) - radius;
                if sx < 0 || sx >= width as i64 {
                    continue;
                }
                acc += tap * plane[(y as usize) * stride + (sx as usize)];
            }
            horizontal[(y as usize) * stride + (x as usize)] = acc;
        }
    }
    let mut vertical = vec![0.0f64; plane.len()];
    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let mut acc = 0.0f64;
            for (index, tap) in taps.iter().enumerate() {
                let sy = y + (index as i64) - radius;
                if sy < 0 || sy >= height as i64 {
                    continue;
                }
                acc += tap * horizontal[(sy as usize) * stride + (x as usize)];
            }
            vertical[(y as usize) * stride + (x as usize)] = acc;
        }
    }
    vertical
}

pub fn n2_shadow_plane() -> Vec<f64> {
    let offset = Vec2::new(
        N2_CENTRE.0 + N2_SHADOW_OFFSET.0,
        N2_CENTRE.1 + N2_SHADOW_OFFSET.1,
    );
    let taps = gaussian_taps(N2_BLUR_SIGMA, N2_BLUR_SIGMAS);
    blur_separable(
        &n2_coverage(offset),
        RASTER_WIDTH,
        RASTER_HEIGHT,
        &taps,
    )
}

pub fn n2_field() -> FloatImage<f64> {
    let shadow = n2_shadow_plane();
    let rect = n2_coverage(Vec2::new(N2_CENTRE.0, N2_CENTRE.1));
    let paper = rgb(N2_PAPER);
    let dark = rgb(N2_SHADOW);
    let ink = rgb(N2_INK);
    let mut image = FloatImage::<f64>::new(RASTER_WIDTH, RASTER_HEIGHT);
    for py in 0..RASTER_HEIGHT {
        for px in 0..RASTER_WIDTH {
            let o = (py as usize) * (RASTER_WIDTH as usize) + (px as usize);
            let a = (shadow[o] * N2_SHADOW_STRENGTH).clamp(0.0, 1.0);
            let under = paper * (1.0 - a) + dark * a;
            let r = rect[o].clamp(0.0, 1.0);
            image.set(px, py, under * (1.0 - r) + ink * r);
        }
    }
    image
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueLattice {
    pub period: u32,
    pub nodes_x: u32,
    pub nodes_y: u32,
    pub values: Vec<f64>,
}

pub fn smoothstep(t: f64) -> f64 {
    let c = t.clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

impl ValueLattice {
    pub fn draw(period: u32, width: u32, height: u32, rng: &mut Lcg) -> Self {
        let nodes_x = width / period + 1;
        let nodes_y = height / period + 1;
        let count = (nodes_x as usize) * (nodes_y as usize);
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(rng.unit());
        }
        Self {
            period,
            nodes_x,
            nodes_y,
            values,
        }
    }

    pub fn node(&self, ix: u32, iy: u32) -> f64 {
        let cx = ix.min(self.nodes_x - 1);
        let cy = iy.min(self.nodes_y - 1);
        self.values[(cy as usize) * (self.nodes_x as usize) + (cx as usize)]
    }

    pub fn sample(&self, p: Vec2<f64>) -> f64 {
        let period = self.period as f64;
        let gx = p.x.max(0.0) / period;
        let gy = p.y.max(0.0) / period;
        let cell_x = gx.floor();
        let cell_y = gy.floor();
        let tx = smoothstep(gx - cell_x);
        let ty = smoothstep(gy - cell_y);
        let ix = cell_x as u32;
        let iy = cell_y as u32;
        let n00 = self.node(ix, iy);
        let n10 = self.node(ix + 1, iy);
        let n01 = self.node(ix, iy + 1);
        let n11 = self.node(ix + 1, iy + 1);
        let top = n00 * (1.0 - tx) + n10 * tx;
        let bottom = n01 * (1.0 - tx) + n11 * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

pub fn colour_noise(seed: u64, periods: &[u32], width: u32, height: u32) -> Vec<Vec<ValueLattice>> {
    let mut rng = Lcg::new(seed);
    let mut channels = Vec::with_capacity(3);
    for _ in 0..3 {
        let mut octaves = Vec::with_capacity(periods.len());
        for period in periods {
            octaves.push(ValueLattice::draw(*period, width, height, &mut rng));
        }
        channels.push(octaves);
    }
    channels
}

pub fn sample_colour_noise(
    channels: &[Vec<ValueLattice>],
    amplitudes: &[f64],
    floor: f64,
    span: f64,
    p: Vec2<f64>,
) -> Rgb<f64> {
    let mut out = [0.0f64; 3];
    for (index, slot) in out.iter_mut().enumerate() {
        let octaves = match channels.get(index) {
            Some(o) => o,
            None => crate::guard::unreachable(
                "fit::content::sample_colour_noise",
                "a colour noise field must carry one octave stack per channel",
            ),
        };
        let mut acc = 0.0f64;
        for (octave, amplitude) in octaves.iter().zip(amplitudes.iter()) {
            acc += amplitude * octave.sample(p);
        }
        *slot = floor + span * acc.clamp(0.0, 1.0);
    }
    Rgb::new(out[0], out[1], out[2])
}

pub fn n3_field() -> FloatImage<f64> {
    let channels = colour_noise(N3_SEED, &N3_LATTICES, RASTER_WIDTH, RASTER_HEIGHT);
    let mut image = FloatImage::<f64>::new(RASTER_WIDTH, RASTER_HEIGHT);
    for py in 0..RASTER_HEIGHT {
        for px in 0..RASTER_WIDTH {
            image.set(
                px,
                py,
                sample_colour_noise(
                    &channels,
                    &N3_AMPLITUDES,
                    N3_FLOOR,
                    N3_SPAN,
                    pixel_sample(px, py),
                ),
            );
        }
    }
    image
}

pub fn n4_edge_frame(degrees: f64) -> (Vec2<f64>, Vec2<f64>) {
    let (s, c) = degrees.to_radians().sin_cos();
    (Vec2::new(c, s), Vec2::new(-s, c))
}

pub fn n4_region(p: Vec2<f64>) -> bool {
    let mut near = false;
    for (degrees, ax, ay) in N4_EDGES {
        let (_, normal) = n4_edge_frame(degrees);
        let d = Vec2::new(p.x - ax, p.y - ay);
        if d.dot(normal) > 0.0 {
            near = !near;
        }
    }
    near
}

pub fn n4_field() -> FloatImage<f64> {
    let near = colour_noise(N4_SEED_NEAR, &N4_LATTICES, RASTER_WIDTH, RASTER_HEIGHT);
    let far = colour_noise(N4_SEED_FAR, &N4_LATTICES, RASTER_WIDTH, RASTER_HEIGHT);
    let mut image = FloatImage::<f64>::new(RASTER_WIDTH, RASTER_HEIGHT);
    for py in 0..RASTER_HEIGHT {
        for px in 0..RASTER_WIDTH {
            let p = pixel_sample(px, py);
            let source = if n4_region(p) { &near } else { &far };
            image.set(
                px,
                py,
                sample_colour_noise(source, &N4_AMPLITUDES, N4_FLOOR, N4_SPAN, p),
            );
        }
    }
    image
}

#[derive(Clone, Debug, PartialEq)]
pub struct RasterContent {
    pub name: &'static str,
    pub viewport: (u32, u32),
    pub image: FloatImage<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Analytic(Target<f64>),
    Raster(RasterContent),
}

impl Content {
    pub fn n1_colour_field() -> Self {
        Content::Raster(RasterContent {
            name: N1_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            image: n1_field(),
        })
    }

    pub fn n2_soft_shadow() -> Self {
        Content::Raster(RasterContent {
            name: N2_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            image: n2_field(),
        })
    }

    pub fn n3_band_limited_noise() -> Self {
        Content::Raster(RasterContent {
            name: N3_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            image: n3_field(),
        })
    }

    pub fn n4_broadband_patch() -> Self {
        Content::Raster(RasterContent {
            name: N4_NAME,
            viewport: (RASTER_WIDTH, RASTER_HEIGHT),
            image: n4_field(),
        })
    }

    pub fn carried_forward() -> Vec<Self> {
        vec![
            Content::Analytic(Target::t2_straight_edge()),
            Content::Analytic(Target::t3_right_angle_corner()),
            Content::Analytic(Target::t5_thin_stroke()),
            Content::Analytic(Target::t6_glyph_like()),
        ]
    }

    pub fn newly_generated() -> Vec<Self> {
        vec![
            Self::n1_colour_field(),
            Self::n2_soft_shadow(),
            Self::n3_band_limited_noise(),
            Self::n4_broadband_patch(),
        ]
    }

    pub fn all() -> Vec<Self> {
        let mut out = Self::carried_forward();
        out.extend(Self::newly_generated());
        out
    }

    pub fn name(&self) -> &'static str {
        match self {
            Content::Analytic(t) => t.name,
            Content::Raster(r) => r.name,
        }
    }

    pub fn viewport(&self) -> (u32, u32) {
        match self {
            Content::Analytic(t) => t.viewport,
            Content::Raster(r) => r.viewport,
        }
    }

    pub fn analytic_control(&self) -> Option<AnalyticArm> {
        match self {
            Content::Analytic(_) => Some(AnalyticArm::Exact),
            Content::Raster(_) => None,
        }
    }

    pub fn control_label(&self) -> &'static str {
        match self.analytic_control() {
            Some(arm) => arm.name(),
            None => CONTROL_ABSENT,
        }
    }

    pub fn fit_scaffold(&self) -> Target<f64> {
        match self {
            Content::Analytic(t) => t.clone(),
            Content::Raster(r) => {
                Target::ink_on_paper(r.name, r.viewport, Vec::new(), Vec::new(), Vec::new())
            }
        }
    }

    pub fn truth(&self, max_depth: u32, budget: &mut Budget) -> Result<FloatImage<f64>> {
        match self {
            Content::Analytic(t) => {
                let (coverage, _) = exact_coverage(t, max_depth, budget)?;
                Ok(coverage_image_to_colour(t, &coverage))
            }
            Content::Raster(r) => Ok(r.image.clone()),
        }
    }
}
