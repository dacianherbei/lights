
use lights::blend::OverAccum;
use lights::{
    Additive, BlendMode, Camera, FloatImage, FloatingPoint, Framebuffer, Gaussian,
    GaussianNormalised, Kernel, KernelFamily, Lcg, LightsError, Normalised, NormalisedAccum, Over,
    RenderMethod, Renderer, Rgb, Scene, SharedSettings, ordering_probe_scene, sphere_camera,
    spheres,
};
use std::time::Instant;

const W: u32 = 1920;
const H: u32 = 1080;
const REPS: usize = 5;
const SENTINEL: u8 = 0xA5;
const PADDING: [usize; 3] = [0, 12, 3];
const SHOWN: usize = 6;

fn frame() -> FloatImage<f32> {
    let mut img = FloatImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let u = x as f32 / W as f32;
            let v = y as f32 / H as f32;
            img.set(x, y, Rgb::new(u, v, 1.0 - u * v));
        }
    }
    img
}

fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

#[test]
fn the_cpu_side_pack_term_is_measured_at_1080p() {
    let img = frame();
    let mut packed = Vec::new();
    let mut times = Vec::new();
    for _ in 0..REPS {
        let start = Instant::now();
        let out = img.to_clamped_u8();
        times.push(start.elapsed().as_secs_f64() * 1e3);
        packed.push(out);
    }
    let (lo, mid, hi) = spread(times);

    let out = packed.pop().expect("REPS is not zero");
    assert_eq!(out.pixels.len(), (W * H) as usize);
    assert_eq!(out.width, W);
    assert_eq!(out.height, H);

    for &(x, y) in &[(0u32, 0u32), (W / 2, H / 2), (W - 1, H - 1), (7, H - 3)] {
        let u = x as f32 / W as f32;
        let v = y as f32 / H as f32;
        let want = [
            (u.clamp(0.0, 1.0) * 255.0).round() as u8,
            (v.clamp(0.0, 1.0) * 255.0).round() as u8,
            ((1.0 - u * v).clamp(0.0, 1.0) * 255.0).round() as u8,
        ];
        let got = out.pixels[(y * W + x) as usize];
        for c in 0..3 {
            let delta = got[c].abs_diff(want[c]);
            assert!(
                delta <= 1,
                "pack at ({x},{y}) channel {c}: got {}, expected about {} \
                 (delta {delta}). The timing above is measuring work that is \
                 not the work this test names.",
                got[c],
                want[c]
            );
        }
    }

    let rgb_bytes = out.pixels.len() * 3;
    let rgba_bytes = out.pixels.len() * 4;
    println!(
        "pack {W}x{H} f32 -> Image(Vec<[u8;3]>): \
         min {lo:.3} ms, median {mid:.3} ms, max {hi:.3} ms over {REPS} reps \
         (spread {:.2}x); payload {rgb_bytes} B RGB, {rgba_bytes} B if RGBA; \
         at 60 Hz that is {:.0} MB/s RGB and {:.0} MB/s RGBA; \
         one frame budget is 16.67 ms",
        hi / lo.max(f64::MIN_POSITIVE),
        rgb_bytes as f64 * 60.0 / 1e6,
        rgba_bytes as f64 * 60.0 / 1e6,
    );

    assert!(
        out.pixels.len() > 2_000_000,
        "the pack produced {} pixels, so the timing above measured almost nothing",
        out.pixels.len()
    );
    assert!(
        mid > 0.0,
        "median pack time was not positive, so the clock was not read"
    );
}

#[derive(Default)]
struct Agreement {
    departures: Vec<String>,
    pixels: usize,
    lit: usize,
    red_unlike_blue: usize,
    rounding_would_differ: usize,
    below_zero: usize,
    above_one: usize,
    rows_unlike: usize,
}

impl Agreement {
    fn check<T: FloatingPoint, B: BlendMode<T>>(
        &mut self,
        label: &str,
        fb: &Framebuffer<T, B>,
        background: Rgb<T>,
    ) {
        let (w, h) = (fb.width() as usize, fb.height() as usize);
        let want: Vec<[u8; 4]> = fb
            .resolve(background)
            .pixels
            .iter()
            .map(|p| [p[2], p[1], p[0], u8::MAX])
            .collect();
        for pad in PADDING {
            let stride = 4 * w + pad;
            let mut got = vec![SENTINEL; stride * h];
            if let Err(e) = fb.resolve_bgra8(background, &mut got, stride) {
                panic!("{label}: a target of exactly stride x height bytes was refused: {e}");
            }
            for (y, row) in got.chunks_exact(stride).enumerate() {
                for (x, px) in row[..4 * w].chunks_exact(4).enumerate() {
                    if px != want[y * w + x] {
                        self.departures.push(format!(
                            "{label}, stride {stride}: pixel ({x},{y}) is {px:?} and the three \
                             passes give {:?}",
                            want[y * w + x]
                        ));
                    }
                }
                if row[4 * w..].iter().any(|b| *b != SENTINEL) {
                    self.departures.push(format!(
                        "{label}, stride {stride}: row {y} wrote into its {pad} padding bytes"
                    ));
                }
            }
        }
        let float = fb.resolve_float(background);
        let floor = [background.x, background.y, background.z].map(|c| c.to_f64());
        for p in float.pixels.iter() {
            let v = [p.x, p.y, p.z].map(|c| c.to_f64());
            self.pixels += 1;
            self.lit += usize::from(v != floor);
            for c in v {
                self.below_zero += usize::from(c < 0.0);
                self.above_one += usize::from(c > 1.0);
                self.rounding_would_differ +=
                    usize::from((0.0..=1.0).contains(&c) && (c * 255.0).fract() >= 0.5);
            }
        }
        self.red_unlike_blue += want.iter().filter(|p| p[0] != p[2]).count();
        self.rows_unlike += usize::from(h > 1 && want[..w] != want[(h - 1) * w..]);
    }

    fn assert_agrees(&self, what: &str) {
        assert!(
            self.departures.is_empty(),
            "{what}: the fused BGRA8 resolve departs from resolve, to_clamped_u8 and the \
             B G R A expansion at {} places; first {:?}",
            self.departures.len(),
            &self.departures[..self.departures.len().min(SHOWN)]
        );
    }
}

fn filled<T: FloatingPoint, B: BlendMode<T>>(
    width: u32,
    height: u32,
    tile: u32,
    seed: u64,
    cell: fn(&mut Lcg) -> B::Accum,
) -> Framebuffer<T, B> {
    let mut fb = match Framebuffer::<T, B>::new(width, height, tile) {
        Ok(fb) => fb,
        Err(e) => panic!("a {width}x{height} framebuffer at tile {tile}: {e}"),
    };
    let grid = fb.grid();
    let mut rng = Lcg::new(seed);
    let tiles = match fb.split_tiles_mut(&grid) {
        Ok(t) => t,
        Err(e) => panic!("the framebuffer refused its own grid: {e}"),
    };
    for tile in tiles {
        for c in tile.cells.iter_mut() {
            *c = cell(&mut rng);
        }
    }
    fb
}

fn colour<T: FloatingPoint>(rng: &mut Lcg, lo: f64, hi: f64) -> Rgb<T> {
    let mut one = || T::from_f64(lo + (hi - lo) * rng.unit());
    Rgb::new(one(), one(), one())
}

fn over_cell<T: FloatingPoint>(rng: &mut Lcg) -> OverAccum<T> {
    OverAccum {
        rgb: colour(rng, -0.2, 1.2),
        transmittance: T::from_f64(rng.unit()),
    }
}

fn normalised_cell<T: FloatingPoint>(rng: &mut Lcg) -> NormalisedAccum<T> {
    let empty = rng.unit() < 0.2;
    NormalisedAccum {
        weighted: colour(rng, 0.0, 2.0),
        weight: if empty { T::ZERO } else { T::from_f64(2.0 * rng.unit()) },
    }
}

fn additive_cell<T: FloatingPoint>(rng: &mut Lcg) -> Rgb<T> {
    colour(rng, -0.5, 1.5)
}

fn background<T: FloatingPoint>() -> Rgb<T> {
    Rgb::new(T::from_f64(0.31), T::from_f64(0.17), T::from_f64(0.44))
}

#[test]
fn the_fused_bgra8_resolve_is_the_three_pass_handoff_byte_for_byte_in_every_blend_mode() {
    let sizes = [(37u32, 23u32, 16u32), (48, 32, 16), (7, 5, 4), (1, 1, 16)];
    let mut seen = Agreement::default();
    for (i, &(w, h, tile)) in sizes.iter().enumerate() {
        let seed = 17 * i as u64;
        let at = format!("{w}x{h} at tile {tile}");
        seen.check(&format!("over f32 {at}"), &filled::<f32, Over>(w, h, tile, seed, over_cell), background());
        seen.check(&format!("over f64 {at}"), &filled::<f64, Over>(w, h, tile, seed, over_cell), background());
        seen.check(
            &format!("normalised f32 {at}"),
            &filled::<f32, Normalised>(w, h, tile, seed + 1, normalised_cell),
            background(),
        );
        seen.check(
            &format!("normalised f64 {at}"),
            &filled::<f64, Normalised>(w, h, tile, seed + 1, normalised_cell),
            background(),
        );
        seen.check(
            &format!("additive f32 {at}"),
            &filled::<f32, Additive>(w, h, tile, seed + 2, additive_cell),
            background(),
        );
        seen.check(
            &format!("additive f64 {at}"),
            &filled::<f64, Additive>(w, h, tile, seed + 2, additive_cell),
            background(),
        );
    }
    seen.assert_agrees("synthetic accumulators");
    assert!(
        seen.red_unlike_blue > 0,
        "no pixel has red unlike blue, so a B G R A channel swap would pass unseen"
    );
    assert!(
        seen.rounding_would_differ > 0,
        "no channel lands in the upper half of a level, so a fused path that rounds where the \
         pack truncates would pass unseen"
    );
    assert!(
        seen.below_zero > 0 && seen.above_one > 0,
        "the accumulators never resolve outside [0,1] ({} below, {} above), so a fused path \
         that skips the clamp would pass unseen",
        seen.below_zero,
        seen.above_one
    );
    assert!(seen.rows_unlike > 0, "every frame's first row equals its last, so a flip would pass unseen");
    println!(
        "fused BGRA8 == three passes on {} synthetic pixels, three strides each; {} with red \
         unlike blue, {} channels where rounding would differ, {} below 0 and {} above 1",
        seen.pixels, seen.red_unlike_blue, seen.rounding_would_differ, seen.below_zero, seen.above_one
    );
}

fn real<T: FloatingPoint, K: Kernel<T>>(
    seen: &mut Agreement,
    method: &RenderMethod,
    family: KernelFamily,
    scene: &Scene<T>,
    label: &str,
) {
    let (w, h) = (101u32, 67u32);
    let mut base = SharedSettings::<f64>::contract();
    base.background = Rgb::new(0.04, 0.05, 0.07);
    let settings = SharedSettings::<T>::from_f64(&base);
    let mut renderer =
        match Renderer::<T, K>::new(method, &settings, &family.shape(), w, h, scene) {
            Ok(r) => r,
            Err(e) => panic!("{label}: the renderer could not be built: {e}"),
        };
    let camera = match Camera::<T>::from_spec(&sphere_camera(w, h), settings.near_plane) {
        Ok(c) => c,
        Err(e) => panic!("{label}: the sphere camera is valid: {e}"),
    };
    if let Err(e) = renderer.render_frame(&camera) {
        panic!("{label}: the render failed: {e}");
    }
    let lit = seen.lit;
    seen.check(label, renderer.framebuffer(), settings.background);
    assert!(seen.lit > lit, "{label}: the render left every pixel at the background, so it compared nothing");
}

#[test]
fn the_fused_resolve_of_a_real_render_is_its_three_pass_handoff_for_every_built_method() {
    let methods = RenderMethod::implemented();
    assert!(!methods.is_empty(), "no method is built, so this case would pass by measuring nothing");
    let mut seen = Agreement::default();
    let mut covered = 0usize;
    for method in methods.iter() {
        let name = method.id().name();
        real::<f32, Gaussian>(&mut seen, method, KernelFamily::Gaussian, &spheres(), &format!("{name} gaussian f32 spheres"));
        real::<f64, Gaussian>(&mut seen, method, KernelFamily::Gaussian, &spheres(), &format!("{name} gaussian f64 spheres"));
        real::<f64, GaussianNormalised>(
            &mut seen,
            method,
            KernelFamily::GaussianNormalised,
            &spheres(),
            &format!("{name} normalised f64 spheres"),
        );
        real::<f32, Gaussian>(
            &mut seen,
            method,
            KernelFamily::Gaussian,
            &ordering_probe_scene(),
            &format!("{name} gaussian f32 ordering probe"),
        );
        covered += 1;
    }
    assert_eq!(covered, methods.len(), "every built method must be walked");
    seen.assert_agrees("real renders");
    assert!(seen.red_unlike_blue > 0, "no rendered pixel has red unlike blue, so a channel swap would pass unseen");
    println!(
        "fused BGRA8 == three passes on {} rendered pixels over {covered} method(s), {} of them lit",
        seen.pixels, seen.lit
    );
}

#[test]
fn the_fused_target_refuses_a_stride_or_a_length_that_cannot_hold_the_frame() {
    let (w, h) = (37u32, 23u32);
    let fb = filled::<f64, Over>(w, h, 16, 5, over_cell);
    let row = 4 * w as usize;
    let rows = h as usize;
    let refused = |stride: usize, len: usize| {
        let mut bytes = vec![SENTINEL; len];
        match fb.resolve_bgra8(background(), &mut bytes, stride) {
            Err(LightsError::ByteTarget { width, height, stride: s, len: l }) => {
                assert_eq!((width, height, s, l), (w, h, stride, len));
            }
            other => panic!(
                "a {w}x{h} frame at a stride of {stride} over {len} bytes must be refused as a \
                 ByteTarget, got {other:?}"
            ),
        }
        assert!(bytes.iter().all(|b| *b == SENTINEL), "a refused target was written into");
    };
    refused(row - 1, (row - 1) * rows + row);
    refused(row, row * rows - 1);
    refused(row + 12, (row + 12) * (rows - 1) + row - 1);
    refused(0, row * rows);

    let stride = row + 12;
    let mut exact = vec![SENTINEL; stride * (rows - 1) + row];
    if let Err(e) = fb.resolve_bgra8(background(), &mut exact, stride) {
        panic!("a target whose last row carries no padding is exactly enough and was refused: {e}");
    }
    let image = fb.resolve(background());
    let last = image.pixels[image.pixels.len() - 1];
    assert_eq!(
        exact[exact.len() - 4..],
        [last[2], last[1], last[0], u8::MAX],
        "the last pixel of an exactly sized target is not the frame's last pixel"
    );
}

fn millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

#[test]
fn the_fused_handoff_is_timed_beside_the_three_passes_it_replaces_at_1080p() {
    let fb = filled::<f32, Over>(W, H, 16, 29, over_cell);
    let bg = Rgb::new(0.04f32, 0.05, 0.07);
    let stride = 4 * W as usize;
    let mut expanded = vec![0u8; stride * H as usize];
    let mut fused = vec![0u8; stride * H as usize];
    let (mut three, mut one) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        let start = Instant::now();
        let image = fb.resolve(bg);
        for (out, pixel) in expanded.chunks_exact_mut(4).zip(&image.pixels) {
            out.copy_from_slice(&[pixel[2], pixel[1], pixel[0], u8::MAX]);
        }
        three.push(millis(start));
        let start = Instant::now();
        if let Err(e) = fb.resolve_bgra8(bg, &mut fused, stride) {
            panic!("the 1080p target is exactly the frame and was refused: {e}");
        }
        one.push(millis(start));
    }
    let first = expanded.iter().zip(&fused).position(|(a, b)| a != b);
    assert!(
        first.is_none(),
        "the two timed arms wrote different bytes (first at {first:?}), so the timings below \
         would compare different work"
    );
    assert!(
        expanded.chunks_exact(4).any(|p| p != &expanded[..4]),
        "the timed frame is one flat colour, so the timing measured almost nothing"
    );
    let (lo3, mid3, hi3) = spread(three);
    let (lo1, mid1, hi1) = spread(one);
    println!(
        "handoff {W}x{H} f32 over, into a reused {} B BGRA8 target, {REPS} reps: \
         three passes (resolve_float, to_clamped_u8, B G R A expansion) min {lo3:.3} median {mid3:.3} max {hi3:.3} ms; \
         fused resolve_bgra8 min {lo1:.3} median {mid1:.3} max {hi1:.3} ms; median ratio {:.2}x",
        stride * H as usize,
        mid3 / mid1.max(f64::MIN_POSITIVE)
    );
    assert!(mid1 > 0.0 && mid3 > 0.0, "a median was not positive, so the clock was not read");
}
