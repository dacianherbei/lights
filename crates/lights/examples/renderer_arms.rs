use lights::fit::{CELL_DIAGONAL, Criterion, DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, Mapping, Target, TargetMetric};
use lights::fit::{coverage_budget, coverage_image_to_colour, exact_coverage, measure_target_mapped};
use lights::{Capture, DrawList, RasterLayer, Result, Similarity2, Stand, Vec2};
use std::hint::black_box;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[path = "support/tiny_arm.rs"]
mod tiny_arm;
#[path = "../src/bin/support/invocation.rs"]
mod invocation;

use tiny_arm::{QualityArm, TinyList};

const USAGE: &str = "\
renderer_arms --ops <DIR> --out <DIR> [options]

Replays every optical frame captured by `unified_showcase --frame-bench --ops-out <DIR>`
through two CPU renderers, lights' DrawList and tiny-skia, serially and in parallel row
bands, and writes arms.tsv (one row per scene, scale, list, renderer and mode) and
quality.tsv (the registered edge criterion re-measured on each renderer's own render of
the six canonical targets, with the exact-area coverage as the calibration row).

  --ops <DIR>         REQUIRED. The ops_<scene>_<ppp>.tsv captures.
  --out <DIR>         REQUIRED. Where arms.tsv and quality.tsv are written.
  --reps <N>          timed repetitions per job after one discarded warm-up (default 7, floor 5)
  --band <ROWS>       rows per parallel band (default 16)
  --scenes <A,B>      only these scenes
  --scales <1,2>      only these pixels per point
  --points <WxH>      the frame in points the captures were drawn at (default 1920x1080)

Exit 0 clean; 2 a flag this binary does not know or a missing value; 3 a gate refused
the timing; 4 an error.";

const POINTS: [f64; 2] = [1920.0, 1080.0];
const REPEAT: usize = 32;
const LISTS: [(&str, &[Stand]); 2] = [("native", &[Stand::Native]), ("framed", &[Stand::Native, Stand::Proxy])];
const BAND_PAINT_LIMIT: f64 = 0.01;
const QUALITY_SCALES: [f64; 2] = [1.0, 2.0];

struct Args {
    ops: PathBuf,
    out: PathBuf,
    reps: usize,
    points: [f64; 2],
    band: NonZeroU32,
    scenes: Option<Vec<String>>,
    scales: Option<Vec<String>>,
}

fn parse(argv: &[String]) -> std::result::Result<Args, String> {
    let mut ops: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut reps = 7usize;
    let mut points = POINTS;
    let mut band = 16u32;
    let mut scenes = None;
    let mut scales = None;
    let mut index = 0;
    for _ in 0..argv.len() {
        let Some(flag) = argv.get(index) else {
            break;
        };
        let Some(value) = argv.get(index + 1) else {
            return Err(format!("{flag} needs a value\n\n{USAGE}"));
        };
        let number = |name: &str| value.parse::<usize>().map_err(|_| format!("{name} needs a number, got {value}"));
        let list = || Some(value.split(',').map(str::to_string).collect::<Vec<String>>());
        match flag.as_str() {
            "--ops" => ops = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--reps" => reps = number("--reps")?,
            "--band" => band = number("--band")? as u32,
            "--scenes" => scenes = list(),
            "--scales" => scales = list(),
            "--points" => {
                let parsed = value
                    .split_once('x')
                    .and_then(|(w, h)| Some([w.parse::<f64>().ok()?, h.parse::<f64>().ok()?]))
                    .filter(|[w, h]| *w >= 1.0 && *h >= 1.0);
                match parsed {
                    Some(p) => points = p,
                    None => return Err(format!("--points needs WxH in points, got {value}")),
                }
            }
            other => {
                return Err(format!(
                    "unknown flag {other}\n\nA flag this binary does not understand is REFUSED rather than ignored.\n\n{USAGE}"
                ));
            }
        }
        index += 2;
    }
    let (Some(ops), Some(out)) = (ops, out) else {
        return Err(format!("--ops and --out are REQUIRED\n\n{USAGE}"));
    };
    let Some(band) = NonZeroU32::new(band) else {
        return Err("--band must be at least one row".to_string());
    };
    Ok(Args {
        ops,
        out,
        reps: reps.max(5),
        points,
        band,
        scenes,
        scales,
    })
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Renderer {
    Lights,
    Tiny,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Serial,
    Parallel,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Arm {
    Fixed,
    Touch,
    Ops,
    Full,
    Pack,
}

const RENDERERS: [Renderer; 2] = [Renderer::Lights, Renderer::Tiny];
const MODES: [Mode; 2] = [Mode::Serial, Mode::Parallel];
const ARMS: [Arm; 5] = [Arm::Fixed, Arm::Touch, Arm::Ops, Arm::Full, Arm::Pack];

impl Renderer {
    fn name(self) -> &'static str {
        match self {
            Renderer::Lights => "lights",
            Renderer::Tiny => "tiny-skia",
        }
    }
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Serial => "serial",
            Mode::Parallel => "parallel",
        }
    }

    fn bands(self, band: NonZeroU32) -> Option<NonZeroU32> {
        match self {
            Mode::Serial => None,
            Mode::Parallel => Some(band),
        }
    }
}

struct Lists {
    name: &'static str,
    lights: DrawList<f32>,
    lights_repeated: DrawList<f32>,
    tiny: TinyList,
    tiny_repeated: TinyList,
    layer: RasterLayer<f32>,
    lights_hash: u64,
    tiny_hash: u64,
    lights_ink: f64,
    tiny_ink: f64,
    lights_paint: f64,
    tiny_paint: f64,
    band_paint: f64,
    versus: (f64, u64),
    band_diff: (u8, u64),
}

struct Frame {
    scene: String,
    ppp: f64,
    width: u32,
    height: u32,
    lists: Vec<Lists>,
    empty_lights: DrawList<f32>,
    empty_tiny: TinyList,
}

fn fnv(hash: u64, word: u64) -> u64 {
    (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3)
}

fn layer_hash(layer: &RasterLayer<f32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for plane in [&layer.premultiplied, &layer.coverage] {
        for p in plane.pixels.iter() {
            for v in [p.x, p.y, p.z] {
                h = fnv(h, f64::from(v).to_bits());
            }
        }
    }
    h
}

fn bytes_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| fnv(h, u64::from(*b)))
}

fn offscreen(ppp: f64, width: u32, height: u32) -> Result<Similarity2<f32>> {
    let away = -4.0 * f64::from(width.max(height));
    Similarity2::new(ppp as f32, 0.0, Vec2::splat(away as f32))
}

fn lights_render(list: &DrawList<f32>, view: Similarity2<f32>, w: u32, h: u32, mode: Mode, band: NonZeroU32) -> Result<RasterLayer<f32>> {
    match mode {
        Mode::Serial => list.render_through(view, w, h),
        Mode::Parallel => list.render_through_parallel(view, w, h, band),
    }
}

fn refused(why: String) -> Result<std::result::Result<Frame, String>> {
    Ok(Err(why))
}

fn frame(text: &str, scene: &str, ppp: f64, args: &Args) -> Result<std::result::Result<Frame, String>> {
    let capture = Capture::<f32>::parse(text)?;
    let width = (args.points[0] * ppp).round() as u32;
    let height = (args.points[1] * ppp).round() as u32;
    let view = Similarity2::uniform(ppp as f32);
    let away = offscreen(ppp, width, height)?;
    let mut lists = Vec::new();
    for (name, admit) in LISTS {
        let (list, refusals) = capture.draw_list(admit);
        for r in refusals.iter() {
            println!("{scene} @{ppp}x {name}: lights refused {r}");
        }
        let mut repeated = DrawList::new();
        for _ in 0..REPEAT {
            for op in list.ops().iter() {
                repeated.push(op.clone())?;
            }
        }
        let (Some(tiny), Some(tiny_repeated)) = (TinyList::from_list(&list, view), TinyList::from_list(&repeated, away)) else {
            return refused(format!("{scene} @{ppp}x {name}: the view is not axis-aligned"));
        };
        let serial = list.render_through(view, width, height)?;
        let parallel = list.render_through_parallel(view, width, height, args.band)?;
        let (hs, hp) = (layer_hash(&serial), layer_hash(&parallel));
        if hs != hp {
            return refused(format!("{scene} @{ppp}x {name}: lights parallel renders {hp:016x} and serial {hs:016x}"));
        }
        let (Some(ts), Some(tp)) = (
            tiny_arm::render(&tiny, width, height, None),
            tiny_arm::render(&tiny, width, height, Some(args.band)),
        ) else {
            return refused(format!("{scene} @{ppp}x {name}: tiny-skia could not allocate a {width}x{height} pixmap"));
        };
        let band_diff = tiny_arm::pixmap_difference(&ts, &tp);
        let (tiny_paint, band_paint) = (tiny_arm::paint_sum(ts.data()), tiny_arm::paint_sum(tp.data()));
        if (band_paint - tiny_paint).abs() > BAND_PAINT_LIMIT * tiny_paint {
            return refused(format!(
                "{scene} @{ppp}x {name}: tiny-skia's banded render paints {band_paint:.1} px against {tiny_paint:.1} serially, beyond {BAND_PAINT_LIMIT}"
            ));
        }
        let packed = tiny_arm::pack_rgba(&serial);
        let lights_paint = tiny_arm::paint_sum(&packed);
        let versus = tiny_arm::departure(&packed, ts.data());
        let tiny_ink = tiny_arm::pixmap_ink(&ts);
        lists.push(Lists {
            name,
            lights_ink: tiny_arm::layer_ink(&serial),
            tiny_ink,
            lights_paint,
            tiny_paint,
            band_paint,
            versus,
            lights_hash: hs,
            tiny_hash: bytes_hash(ts.data()),
            band_diff,
            layer: serial,
            lights: list,
            lights_repeated: repeated,
            tiny,
            tiny_repeated,
        });
    }
    Ok(Ok(Frame {
        scene: scene.to_string(),
        ppp,
        width,
        height,
        lists,
        empty_lights: DrawList::new(),
        empty_tiny: TinyList::default(),
    }))
}

fn timed(f: &Frame, list: usize, renderer: Renderer, mode: Mode, arm: Arm, band: NonZeroU32) -> Result<f64> {
    let l = &f.lists[list];
    let view = Similarity2::uniform(f.ppp as f32);
    let (w, h) = (f.width, f.height);
    let bands = mode.bands(band);
    let start = Instant::now();
    match (renderer, arm) {
        (Renderer::Lights, Arm::Fixed) => {
            black_box(lights_render(&f.empty_lights, view, w, h, mode, band)?);
        }
        (Renderer::Lights, Arm::Touch) => {
            let mut layer = lights_render(&f.empty_lights, view, w, h, mode, band)?;
            match mode {
                Mode::Serial => {
                    layer.premultiplied.pixels.fill(lights::Rgb::splat(1.0));
                    layer.coverage.pixels.fill(lights::Rgb::splat(1.0));
                }
                Mode::Parallel => {
                    use rayon::prelude::*;
                    let rows = band.get() as usize * w as usize;
                    layer
                        .premultiplied
                        .pixels
                        .par_chunks_mut(rows)
                        .zip(layer.coverage.pixels.par_chunks_mut(rows))
                        .for_each(|(p, c)| {
                            p.fill(lights::Rgb::splat(1.0));
                            c.fill(lights::Rgb::splat(1.0));
                        });
                }
            }
            black_box(&layer);
        }
        (Renderer::Lights, Arm::Ops) => {
            black_box(lights_render(&l.lights_repeated, offscreen(f.ppp, w, h)?, w, h, mode, band)?);
        }
        (Renderer::Lights, Arm::Full) => {
            black_box(lights_render(&l.lights, view, w, h, mode, band)?);
        }
        (Renderer::Lights, Arm::Pack) => {
            let packed = match bands {
                None => tiny_arm::pack_rgba(&l.layer),
                Some(rows) => tiny_arm::pack_rgba_parallel(&l.layer, rows),
            };
            black_box(packed);
        }
        (Renderer::Tiny, Arm::Fixed) => {
            black_box(tiny_arm::render(&f.empty_tiny, w, h, bands));
        }
        (Renderer::Tiny, Arm::Touch) => {
            black_box(tiny_arm::touch(w, h, bands));
        }
        (Renderer::Tiny, Arm::Ops) => {
            black_box(tiny_arm::render(&l.tiny_repeated, w, h, bands));
        }
        (Renderer::Tiny, Arm::Full) => {
            black_box(tiny_arm::render(&l.tiny, w, h, bands));
        }
        (Renderer::Tiny, Arm::Pack) => {}
    }
    Ok(start.elapsed().as_secs_f64() * 1e3)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) * 0.5 }
}

fn shell(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

const COLUMNS: [&str; 43] = [
    "scene", "ppp", "width_px", "height_px", "list", "renderer", "mode", "threads", "band_rows", "ops", "drawn",
    "masked", "fixed_ms", "fixed_min", "fixed_max", "touch_ms", "touch_min", "touch_max", "opsx_ms", "opsx_min",
    "opsx_max", "full_ms", "full_min", "full_max", "full_spread", "page_ms", "ops_ms", "px_ms", "per_op_ns",
    "pack_ms", "pack_min", "pack_max", "frame_bytes", "handoff_bytes", "ink_px", "paint_px", "band_paint_px",
    "band_diff_max", "band_diff_px", "vs_lights_mean", "vs_lights_far_px", "frame_hash", "reps",
];

const QUALITY_COLUMNS: [&str; 14] = [
    "arm", "target", "ppp", "width_px", "height_px", "judged", "meets", "transition_10_90", "contrast_frac",
    "peak_frac", "fwhm", "fwhm_error", "apex_displacement", "interior_ripple",
];

fn captures(args: &Args) -> Result<Vec<(String, f64, String)>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(&args.ops)? {
        let path = entry?.path();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let Some(stem) = name.strip_prefix("ops_").and_then(|n| n.strip_suffix(".tsv")) else {
            continue;
        };
        let Some((scene, ppp)) = stem.rsplit_once('_') else {
            continue;
        };
        let keep = args.scenes.as_ref().is_none_or(|s| s.iter().any(|x| x == scene))
            && args.scales.as_ref().is_none_or(|s| s.iter().any(|x| x == ppp));
        let Ok(scale) = ppp.parse::<f64>() else {
            continue;
        };
        if keep {
            found.push((scene.to_string(), scale, std::fs::read_to_string(&path)?));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    Ok(found)
}

fn metric_cells(m: &TargetMetric) -> Vec<String> {
    let f = |v: f64| if v.is_finite() { format!("{v:.4}") } else { "nan".to_string() };
    vec![
        m.judged.to_string(),
        if m.meets { "yes" } else { "no" }.to_string(),
        f(m.transition_10_90),
        f(m.contrast_frac),
        f(m.peak_frac),
        f(m.fwhm),
        f(m.fwhm_error),
        f(m.apex_displacement),
        f(m.interior_ripple),
    ]
}

fn quality_rows() -> Result<std::result::Result<Vec<Vec<String>>, String>> {
    let criterion = Criterion::registered();
    let mut rows = Vec::new();
    for target in Target::<f64>::all() {
        for scale in QUALITY_SCALES {
            let (w, h) = Mapping::at_scale(scale).device_viewport(target.viewport);
            for arm in [QualityArm::Lights, QualityArm::Tiny] {
                let Some(m) = tiny_arm::quality(arm, &target, scale) else {
                    return Ok(Err(format!("{} could not draw or measure {} at {scale}x", arm.name(), target.name)));
                };
                let mut row = vec![arm.name().to_string(), target.name.to_string(), format!("{scale}"), w.to_string(), h.to_string()];
                row.extend(metric_cells(&m));
                rows.push(row);
            }
        }
        let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
        let (cover, _) = exact_coverage(&target, DEFAULT_MAX_DEPTH, &mut budget)?;
        let image = coverage_image_to_colour(&target, &cover);
        let m = measure_target_mapped(&image, &target, &criterion, Mapping::identity())?;
        let mut row = vec!["exact-area".to_string(), target.name.to_string(), "1".to_string(), target.viewport.0.to_string(), target.viewport.1.to_string()];
        row.extend(metric_cells(&m));
        rows.push(row);
    }
    Ok(Ok(rows))
}

fn run(args: &Args) -> Result<ExitCode> {
    let mut frames: Vec<Frame> = Vec::new();
    for (scene, ppp, text) in captures(args)? {
        match frame(&text, &scene, ppp, args)? {
            Ok(f) => frames.push(f),
            Err(why) => {
                println!("REFUSED: {why}");
                return Ok(ExitCode::from(3));
            }
        }
    }
    if frames.is_empty() {
        println!("no ops_<scene>_<ppp>.tsv capture under {}", args.ops.display());
        return Ok(ExitCode::from(2));
    }
    let mut jobs: Vec<(usize, usize, Renderer, Mode, Arm)> = Vec::new();
    for (fi, f) in frames.iter().enumerate() {
        for li in 0..f.lists.len() {
            for renderer in RENDERERS {
                for mode in MODES {
                    for arm in ARMS {
                        jobs.push((fi, li, renderer, mode, arm));
                    }
                }
            }
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); jobs.len()];
    for rep in 0..=args.reps {
        let started = Instant::now();
        for (j, (fi, li, renderer, mode, arm)) in jobs.iter().enumerate() {
            let ms = timed(&frames[*fi], *li, *renderer, *mode, *arm, args.band)?;
            if rep > 0 {
                samples[j].push(ms);
            }
        }
        println!(
            "rep {rep} of {} {} in {:.1} s",
            args.reps,
            if rep == 0 { "(warm-up, discarded)" } else { "" },
            started.elapsed().as_secs_f64()
        );
    }
    let threads = rayon::current_num_threads();
    let mut rows = Vec::new();
    for (fi, f) in frames.iter().enumerate() {
        for (li, l) in f.lists.iter().enumerate() {
            for renderer in RENDERERS {
                for mode in MODES {
                    let arm = |a: Arm| -> Vec<f64> {
                        jobs.iter()
                            .position(|j| j.0 == fi && j.1 == li && j.2 == renderer && j.3 == mode && j.4 == a)
                            .map(|k| samples[k].clone())
                            .unwrap_or_default()
                    };
                    let stats = |v: &Vec<f64>| {
                        (
                            median(v.clone()),
                            v.iter().copied().fold(f64::INFINITY, f64::min),
                            v.iter().copied().fold(0.0, f64::max),
                        )
                    };
                    let (fixed, fixed_min, fixed_max) = stats(&arm(Arm::Fixed));
                    let (touched, touch_min, touch_max) = stats(&arm(Arm::Touch));
                    let (opsx, opsx_min, opsx_max) = stats(&arm(Arm::Ops));
                    let (full, full_min, full_max) = stats(&arm(Arm::Full));
                    let (pack, pack_min, pack_max) = match renderer {
                        Renderer::Lights => stats(&arm(Arm::Pack)),
                        Renderer::Tiny => (0.0, 0.0, 0.0),
                    };
                    let ops = l.lights.len();
                    let per_op_ns = (opsx - fixed) * 1e6 / (REPEAT * ops.max(1)) as f64;
                    let ops_ms = per_op_ns * ops as f64 / 1e6;
                    let px_ms = full - fixed - ops_ms;
                    let pixels = u64::from(f.width) * u64::from(f.height);
                    let (drawn, masked, frame_bytes, handoff_bytes, ink, paint, band_paint, diff, versus, hash) = match renderer {
                        Renderer::Lights => (ops, 0usize, pixels * 24, pixels * 4, l.lights_ink, l.lights_paint, l.lights_paint, (0u8, 0u64), (0.0, 0u64), l.lights_hash),
                        Renderer::Tiny => (l.tiny.len(), l.tiny.masked, pixels * 4, 0, l.tiny_ink, l.tiny_paint, l.band_paint, l.band_diff, l.versus, l.tiny_hash),
                    };
                    rows.push(vec![
                        f.scene.clone(),
                        format!("{}", f.ppp),
                        f.width.to_string(),
                        f.height.to_string(),
                        l.name.to_string(),
                        renderer.name().to_string(),
                        mode.name().to_string(),
                        match mode {
                            Mode::Parallel => threads.to_string(),
                            Mode::Serial => "1".to_string(),
                        },
                        match mode {
                            Mode::Parallel => args.band.get().to_string(),
                            Mode::Serial => f.height.to_string(),
                        },
                        ops.to_string(),
                        drawn.to_string(),
                        masked.to_string(),
                        format!("{fixed:.4}"),
                        format!("{fixed_min:.4}"),
                        format!("{fixed_max:.4}"),
                        format!("{touched:.4}"),
                        format!("{touch_min:.4}"),
                        format!("{touch_max:.4}"),
                        format!("{opsx:.4}"),
                        format!("{opsx_min:.4}"),
                        format!("{opsx_max:.4}"),
                        format!("{full:.4}"),
                        format!("{full_min:.4}"),
                        format!("{full_max:.4}"),
                        format!("{:.2}", full_max / full_min),
                        format!("{:.4}", touched - fixed),
                        format!("{ops_ms:.4}"),
                        format!("{px_ms:.4}"),
                        format!("{per_op_ns:.1}"),
                        format!("{pack:.4}"),
                        format!("{pack_min:.4}"),
                        format!("{pack_max:.4}"),
                        frame_bytes.to_string(),
                        handoff_bytes.to_string(),
                        format!("{ink:.1}"),
                        format!("{paint:.1}"),
                        format!("{band_paint:.1}"),
                        diff.0.to_string(),
                        diff.1.to_string(),
                        format!("{:.4}", versus.0),
                        versus.1.to_string(),
                        format!("{hash:016x}"),
                        args.reps.to_string(),
                    ]);
                }
            }
        }
    }
    let header = vec![
        format!("build\t{}", if cfg!(debug_assertions) { "debug" } else { "release" }),
        format!(
            "hardware\t{}, {} logical cores, macOS {}",
            shell("sysctl", &["-n", "machdep.cpu.brand_string"]),
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
            shell("sw_vers", &["-productVersion"])
        ),
        format!("tree\t{}", shell("git", &["describe", "--always", "--dirty"])),
        format!("frame\t{}x{} points at each capture's pixels per point; the view is Similarity2::uniform(ppp); precision f32 for lights, 8-bit premultiplied RGBA for tiny-skia", args.points[0], args.points[1]),
        "lists\tnative and framed exactly as L-175 builds them (Capture::draw_list); tiny-skia draws the SAME DrawOps, each converted once, outside the clock, to a device-space tiny-skia figure: a rect without radius to fill_rect, a rounded rect to a path of four cubic quarter circles, an outline or ring to a stroke with round joins, a segment or polyline to a stroke with round caps and joins, a polygon to an even-odd fill".to_string(),
        "clips\ta rect that crosses its clip is cut to the clip; any other figure that crosses its clip is drawn through a Mask built inside the render, once per clip per band, zero except the clip rect's pixel fractions; an op wholly outside its clip is not drawn by either renderer".to_string(),
        format!(
            "arms\tfixed = an empty list at the same viewport (the frame buffer allocated zeroed and freed, never touched); touch = the same buffer allocated, every byte written once and freed, in the mode's own threading; opsx = the list repeated {REPEAT} times through a view that moves every op off the frame; full = the list through the real view; pack = lights' handoff from its two float planes to 8-bit RGBA (serial: FloatImage::to_clamped_u8 then an RGBA expansion, the one path the crate offers plus the pass it lacks; parallel: one fused pass over the same bands), timed alone on the list's finished render; tiny-skia's buffer is already 8-bit premultiplied RGBA, so its pack is 0. page_ms = touch - fixed, per_op_ns = (opsx - fixed) / ({REPEAT} x ops), ops_ms = per_op_ns x ops, px_ms = full - fixed - ops_ms. For tiny-skia an off-frame op is dropped by the band cull before tiny-skia sees it, so its per_op_ns is the cull alone and its per-op work (stroking, edge building) is inside px_ms"
        ),
        format!(
            "modes\tserial = one band the height of the frame on the calling thread; parallel = {}-row bands on rayon's global pool of {threads} threads, each band a tiny-skia PixmapMut over its own rows",
            args.band
        ),
        format!(
            "order\trep-major: every job once per rep, {} reps after one discarded warm-up; each figure is the median over reps with min and max beside it; the render call is timed with its result dropped inside the clock",
            args.reps
        ),
        format!("gates\tbefore any time is taken: lights' serial and parallel renders must hash alike; tiny-skia's banded render must paint within {BAND_PAINT_LIMIT} of its serial render (band_paint_px against paint_px), and its per-pixel departure is recorded, not refused (band_diff_max levels at band_diff_px pixels), because tiny-skia is not exact under a band cut: a hairline, a stroke 1 px or thinner, loses pixels at band edges; either failure refuses the run with exit 3"),
        format!("memory\tframe_bytes = the render's own output buffers (lights: two f32 RGB planes, 24 B per pixel; tiny-skia: one RGBA8 pixmap, 4 B per pixel); handoff_bytes = the extra RGBA8 buffer lights' pack writes; ink_px = the sum of alpha over the frame, in pixels (every optical frame is opaque, so it is the frame's area and says nothing about content); paint_px = the sum of (r + g + b) / 3 over the 8-bit premultiplied frame, in pixels, lights' after its serial pack; vs_lights_mean = tiny-skia's mean absolute departure from lights' packed frame in levels per channel, vs_lights_far_px = the pixels where some channel departs by more than {} levels", tiny_arm::FAR_LEVELS),
    ];
    std::fs::create_dir_all(&args.out)?;
    let arms_path = args.out.join("arms.tsv");
    std::fs::write(&arms_path, invocation::table(&header, &COLUMNS, &rows))?;
    println!("wrote {} rows to {}", rows.len(), arms_path.display());
    let c = Criterion::registered();
    let quality_header = vec![
        format!("build\t{}", if cfg!(debug_assertions) { "debug" } else { "release" }),
        format!(
            "criterion\tCriterion::registered: max_transition {}, min_contrast_frac {}, min_peak_frac {}, target_fwhm {}, max_fwhm_error {}, max_interior_ripple {:.6}",
            c.max_transition, c.min_contrast_frac, c.min_peak_frac, c.target_fwhm, c.max_fwhm_error, c.max_interior_ripple
        ),
        format!(
            "arms\tlights-drawlist = DrawList::render_through over the paper at f32; tiny-skia = the same DrawList converted by the arm's own converter and drawn serially into a pixmap; exact-area = fit::exact_coverage (max depth {DEFAULT_MAX_DEPTH}, cells of diagonal {CELL_DIAGONAL}), the box-filtered area, at 1x only. Each target is drawn as DrawOps: a half plane as the polygon it cuts from the viewport grown by {} px, cut again by any half-plane hole; an arc as a {}-point polyline; a disc hole as a paper disc drawn last",
            tiny_arm::HALF_PLANE_MARGIN,
            tiny_arm::ARC_POINTS
        ),
        "measure\tmeasure_target_mapped on each arm's own fresh render with Mapping::at_scale(ppp); nothing is read from a renderer's bookkeeping".to_string(),
    ];
    let quality_path = args.out.join("quality.tsv");
    let qrows = match quality_rows()? {
        Ok(r) => r,
        Err(why) => {
            println!("REFUSED: {why}");
            return Ok(ExitCode::from(3));
        }
    };
    std::fs::write(&quality_path, invocation::table(&quality_header, &QUALITY_COLUMNS, &qrows))?;
    println!("wrote {} rows to {}", qrows.len(), quality_path.display());
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(message) => {
            println!("{message}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            println!("renderer_arms failed: {e}");
            ExitCode::from(4)
        }
    }
}
