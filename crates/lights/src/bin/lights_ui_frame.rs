use lights::capture::{Capture, Stand};
use lights::draw::DrawList;
use lights::image::{FloatImage, RasterLayer, write_png};
use lights::numerics::float::FloatingPoint;
use lights::numerics::similarity::Similarity2;
use lights::numerics::vector::{Rgb, Vec2};
use lights::{LightsError, Result};
use rayon::prelude::*;
use std::hint::black_box;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[path = "support/invocation.rs"]
mod invocation;

const USAGE: &str = "\
lights-ui-frame --ops <DIR> --out <FILE> [options]

Replays every optical frame captured by `unified_showcase --frame-bench --ops-out <DIR>`
through DrawList, serially and in parallel row bands, at f32 and f64, and writes one
row per scene, scale, precision, list and mode.

  --ops <DIR>         REQUIRED. The ops_<scene>_<ppp>.tsv captures.
  --out <FILE>        REQUIRED. The table this run writes.
  --reps <N>          timed repetitions per job after one discarded warm-up (default 7, floor 5)
  --band <ROWS>       rows per parallel band (default 16, the crate's tile edge)
  --scenes <A,B>      only these scenes
  --scales <1,2>      only these pixels per point
  --points <WxH>      the frame in points the captures were drawn at (default 1920x1080)
  --png <DIR>         also write the serial f64 framed render of each capture, over white

Exit 0 clean; 2 a flag this binary does not know or a missing value; 3 the serial and
parallel renders of one list differ in any bit, which refuses the timing; 4 an error.";

const POINTS: [f64; 2] = [1920.0, 1080.0];
const REPEAT: usize = 32;
const LISTS: [(&str, &[Stand]); 2] = [("native", &[Stand::Native]), ("framed", &[Stand::Native, Stand::Proxy])];

struct Args {
    ops: PathBuf,
    out: PathBuf,
    reps: usize,
    points: [f64; 2],
    band: NonZeroU32,
    scenes: Option<Vec<String>>,
    scales: Option<Vec<String>>,
    png: Option<PathBuf>,
}

fn parse(argv: &[String]) -> std::result::Result<Args, String> {
    let mut ops: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut reps = 7usize;
    let mut points = POINTS;
    let mut band = 16u32;
    let mut scenes = None;
    let mut scales = None;
    let mut png = None;
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
            "--png" => png = Some(PathBuf::from(value)),
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
        png,
    })
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Serial,
    Parallel,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Serial => "serial",
            Mode::Parallel => "parallel",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Arm {
    Fixed,
    Touch,
    Ops,
    Full,
}

const MODES: [Mode; 2] = [Mode::Serial, Mode::Parallel];
const ARMS: [Arm; 4] = [Arm::Fixed, Arm::Touch, Arm::Ops, Arm::Full];

struct Census {
    ops: usize,
    native: usize,
    proxy: usize,
    absent: usize,
    refused: usize,
    visited: u64,
    inked: u64,
    hash: u64,
}

struct Frame<T> {
    scene: String,
    ppp: f64,
    width: u32,
    height: u32,
    lists: Vec<(&'static str, DrawList<T>, DrawList<T>, Census)>,
    empty: DrawList<T>,
}

enum Replay {
    Single(Frame<f32>),
    Double(Frame<f64>),
}

fn fnv(hash: u64, word: u64) -> u64 {
    (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3)
}

fn layer_hash<T: FloatingPoint>(layer: &RasterLayer<T>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for plane in [&layer.premultiplied, &layer.coverage] {
        for p in plane.pixels.iter() {
            for v in [p.x, p.y, p.z] {
                h = fnv(h, v.to_f64().to_bits());
            }
        }
    }
    h
}

fn view<T: FloatingPoint>(ppp: f64) -> Similarity2<T> {
    Similarity2::uniform(T::from_f64(ppp))
}

fn offscreen<T: FloatingPoint>(ppp: f64, width: u32, height: u32) -> Result<Similarity2<T>> {
    let away = -4.0 * f64::from(width.max(height));
    Similarity2::new(T::from_f64(ppp), T::ZERO, Vec2::splat(T::from_f64(away)))
}

fn render<T: FloatingPoint>(
    list: &DrawList<T>,
    through: Similarity2<T>,
    width: u32,
    height: u32,
    mode: Mode,
    band: NonZeroU32,
) -> Result<RasterLayer<T>> {
    match mode {
        Mode::Serial => list.render_through(through, width, height),
        Mode::Parallel => list.render_through_parallel(through, width, height, band),
    }
}

fn census<T: FloatingPoint>(
    capture: &Capture<T>,
    list: &DrawList<T>,
    admit: &[Stand],
    refused: usize,
    ppp: f64,
    width: u32,
    height: u32,
    band: NonZeroU32,
) -> Result<std::result::Result<Census, String>> {
    let count = |s: Stand| capture.ops.iter().filter(|c| c.stand == s && admit.contains(&s)).count();
    let absent = capture.ops.iter().filter(|c| c.stand == Stand::Absent).count();
    let through = view::<T>(ppp);
    let mut visited = 0u64;
    let mut inked = 0u64;
    for op in list.ops().iter() {
        if let Some([x0, y0, x1, y1]) = op.footprint(through, width, height) {
            visited += u64::from(x1 - x0) * u64::from(y1 - y0);
            for py in y0..y1 {
                for px in x0..x1 {
                    if op.coverage_through(through, px, py) > T::ZERO {
                        inked += 1;
                    }
                }
            }
        }
    }
    let mut hashes = Vec::new();
    for mode in MODES {
        hashes.push((mode, layer_hash(&render(list, through, width, height, mode, band)?)));
    }
    if let Some((mode, h)) = hashes.iter().find(|(_, h)| *h != hashes[0].1) {
        return Ok(Err(format!(
            "{} renders {h:016x} and serial renders {:016x}: the modes drew different pixels, so their times are not comparable",
            mode.name(),
            hashes[0].1
        )));
    }
    Ok(Ok(Census {
        ops: list.len(),
        native: count(Stand::Native),
        proxy: count(Stand::Proxy),
        absent,
        refused,
        visited,
        inked,
        hash: hashes[0].1,
    }))
}

fn frame<T: FloatingPoint>(
    text: &str,
    scene: &str,
    ppp: f64,
    args: &Args,
    png: Option<&Path>,
) -> Result<std::result::Result<Frame<T>, String>> {
    let band = args.band;
    let capture = Capture::<T>::parse(text)?;
    let width = (args.points[0] * ppp).round() as u32;
    let height = (args.points[1] * ppp).round() as u32;
    let mut lists = Vec::new();
    for (name, admit) in LISTS {
        let (list, refused) = capture.draw_list(admit);
        for r in refused.iter() {
            println!("{scene} @{ppp}x {name}: refused {r}");
        }
        let mut repeated = DrawList::new();
        for _ in 0..REPEAT {
            for op in list.ops().iter() {
                repeated.push(op.clone())?;
            }
        }
        match census(&capture, &list, admit, refused.len(), ppp, width, height, band)? {
            Ok(c) => lists.push((name, list, repeated, c)),
            Err(why) => return Ok(Err(format!("{scene} @{ppp}x {name}: {why}"))),
        }
    }
    if let (Some(dir), Some((_, list, _, _))) = (png, lists.last()) {
        let layer = list.render_through(view::<T>(ppp), width, height)?;
        let white = FloatImage::filled(width, height, Rgb::splat(T::ONE));
        let path = dir.join(format!("{scene}_{ppp}.png"));
        write_png(&path, &layer.over(&white)?.to_clamped_u8())?;
    }
    Ok(Ok(Frame {
        scene: scene.to_string(),
        ppp,
        width,
        height,
        lists,
        empty: DrawList::new(),
    }))
}

fn touch<T: FloatingPoint>(width: u32, height: u32, mode: Mode, band: NonZeroU32) {
    let mut premultiplied = FloatImage::<T>::new(width, height);
    let mut coverage = FloatImage::<T>::new(width, height);
    let one = Rgb::splat(T::ONE);
    match mode {
        Mode::Serial => {
            premultiplied.pixels.fill(one);
            coverage.pixels.fill(one);
        }
        Mode::Parallel => {
            let rows = band.get() as usize * width as usize;
            premultiplied
                .pixels
                .par_chunks_mut(rows)
                .zip(coverage.pixels.par_chunks_mut(rows))
                .for_each(|(p, c)| {
                    p.fill(one);
                    c.fill(one);
                });
        }
    }
    black_box(&premultiplied);
    black_box(&coverage);
}

fn timed<T: FloatingPoint>(f: &Frame<T>, list: usize, mode: Mode, arm: Arm, band: NonZeroU32) -> Result<f64> {
    let start = Instant::now();
    let layer = match arm {
        Arm::Fixed => Some(render(&f.empty, view::<T>(f.ppp), f.width, f.height, mode, band)?),
        Arm::Touch => {
            touch::<T>(f.width, f.height, mode, band);
            None
        }
        Arm::Ops => Some(render(&f.lists[list].2, offscreen::<T>(f.ppp, f.width, f.height)?, f.width, f.height, mode, band)?),
        Arm::Full => Some(render(&f.lists[list].1, view::<T>(f.ppp), f.width, f.height, mode, band)?),
    };
    black_box(&layer);
    drop(layer);
    Ok(start.elapsed().as_secs_f64() * 1e3)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
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

const COLUMNS: [&str; 37] = [
    "scene", "ppp", "width_px", "height_px", "precision", "list", "mode", "threads", "band_rows", "ops",
    "native", "proxy", "absent", "refused", "visited_px", "inked_px", "fixed_ms", "fixed_min", "fixed_max",
    "touch_ms", "touch_min", "touch_max", "opsx_ms", "opsx_min", "opsx_max", "full_ms", "full_min", "full_max",
    "full_spread", "page_ms", "ops_ms", "px_ms", "per_op_ns", "per_px_ns", "px_share", "frame_hash", "reps",
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

fn run(args: &Args) -> Result<ExitCode> {
    let mut replays: Vec<Replay> = Vec::new();
    for (scene, ppp, text) in captures(args)? {
        match frame::<f32>(&text, &scene, ppp, args, None)? {
            Ok(f) => replays.push(Replay::Single(f)),
            Err(why) => {
                println!("REFUSED: {why}");
                return Ok(ExitCode::from(3));
            }
        }
        match frame::<f64>(&text, &scene, ppp, args, args.png.as_deref())? {
            Ok(f) => replays.push(Replay::Double(f)),
            Err(why) => {
                println!("REFUSED: {why}");
                return Ok(ExitCode::from(3));
            }
        }
    }
    if replays.is_empty() {
        println!("no ops_<scene>_<ppp>.tsv capture under {}", args.ops.display());
        return Ok(ExitCode::from(2));
    }
    let mut jobs: Vec<(usize, usize, Mode, Arm)> = Vec::new();
    for (r, replay) in replays.iter().enumerate() {
        let lists = match replay {
            Replay::Single(f) => f.lists.len(),
            Replay::Double(f) => f.lists.len(),
        };
        for list in 0..lists {
            for mode in MODES {
                for arm in ARMS {
                    jobs.push((r, list, mode, arm));
                }
            }
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); jobs.len()];
    for rep in 0..=args.reps {
        let started = Instant::now();
        for (j, (r, list, mode, arm)) in jobs.iter().enumerate() {
            let ms = match &replays[*r] {
                Replay::Single(f) => timed(f, *list, *mode, *arm, args.band)?,
                Replay::Double(f) => timed(f, *list, *mode, *arm, args.band)?,
            };
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
    for (r, replay) in replays.iter().enumerate() {
        let (scene, ppp, width, height, precision, lists): (&str, f64, u32, u32, &str, Vec<(&str, &Census)>) = match replay {
            Replay::Single(f) => (&f.scene, f.ppp, f.width, f.height, "f32", f.lists.iter().map(|l| (l.0, &l.3)).collect()),
            Replay::Double(f) => (&f.scene, f.ppp, f.width, f.height, "f64", f.lists.iter().map(|l| (l.0, &l.3)).collect()),
        };
        for (li, (list_name, c)) in lists.iter().enumerate() {
            for mode in MODES {
                let arm = |a: Arm| -> Vec<f64> {
                    jobs.iter()
                        .position(|j| j.0 == r && j.1 == li && j.2 == mode && j.3 == a)
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
                let per_op_ns = (opsx - fixed) * 1e6 / (REPEAT * c.ops.max(1)) as f64;
                let ops_ms = per_op_ns * c.ops as f64 / 1e6;
                let px_ms = full - fixed - ops_ms;
                rows.push(vec![
                    scene.to_string(),
                    format!("{ppp}"),
                    width.to_string(),
                    height.to_string(),
                    precision.to_string(),
                    list_name.to_string(),
                    mode.name().to_string(),
                    match mode {
                        Mode::Parallel => threads.to_string(),
                        Mode::Serial => "1".to_string(),
                    },
                    match mode {
                        Mode::Parallel => args.band.get().to_string(),
                        Mode::Serial => height.to_string(),
                    },
                    c.ops.to_string(),
                    c.native.to_string(),
                    c.proxy.to_string(),
                    c.absent.to_string(),
                    c.refused.to_string(),
                    c.visited.to_string(),
                    c.inked.to_string(),
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
                    format!("{:.3}", px_ms * 1e6 / c.visited.max(1) as f64),
                    format!("{:.3}", px_ms / full),
                    format!("{:016x}", c.hash),
                    args.reps.to_string(),
                ]);
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
        format!("frame\t{}x{} points at each capture's pixels per point; the view is Similarity2::uniform(ppp)", args.points[0], args.points[1]),
        format!(
            "lists\tnative = the ops lights has a shape for, drawn at the captured geometry; framed = native plus the proxies (a glyph as its quad, a mesh or text triangle as a flat polygon, a blurred or textured rect as a flat rect of its reach); absent = kinds lights cannot draw at all"
        ),
        format!(
            "arms\tfixed = an empty DrawList at the same viewport (two planes allocated zeroed and freed, never touched); touch = the same two planes allocated, every pixel written once and freed, in the mode's own threading; opsx = the list repeated {REPEAT} times through a view that moves every footprint off the frame (every per-op step, no pixel); full = the list through the real view. page_ms = touch - fixed, per_op_ns = (opsx - fixed) / ({REPEAT} x ops), ops_ms = per_op_ns x ops, px_ms = full - fixed - ops_ms, per_px_ns = px_ms / visited_px, px_share = px_ms / full"
        ),
        format!(
            "modes\tserial = render_through on the calling thread, one band the height of the frame; parallel = render_through_parallel over {}-row bands on rayon's global pool of {threads} threads",
            args.band
        ),
        format!(
            "order\trep-major: every job once per rep, {} reps after one discarded warm-up; each figure is the median over reps with min and max beside it; the render call is timed with its result dropped inside the clock",
            args.reps
        ),
        "gate\tbefore any time is taken, every list is rendered in every mode and the bits of both planes hashed; one differing hash refuses the run with exit 3".to_string(),
    ];
    let text = invocation::table(&header, &COLUMNS, &rows);
    std::fs::write(&args.out, text)?;
    println!("wrote {} rows to {}", rows.len(), args.out.display());
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
            println!("lights-ui-frame failed: {e}");
            let _ = LightsError::EmptyScene;
            ExitCode::from(4)
        }
    }
}
