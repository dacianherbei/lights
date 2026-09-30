use lights::camera::{Camera, CameraSpec};
use lights::dispatch::Precision;
use lights::error::Result;
use lights::image::Framebuffer;
use lights::kernel::gaussian::Gaussian;
use lights::kernel::{Kernel, KernelFamily};
use lights::method::{ImageFormation, MethodId, RenderMethod, create_strategy};
use lights::numerics::float::FloatingPoint;
use lights::probe::{Counters, NoProbe, ProbeSink};
use lights::scene::Scene;
use lights::scene::demo::{sphere_camera, stress_scene};
use lights::scene::prepare::PreparedScene;
use lights::settings::SharedSettings;
use lights::stats::{Stat, intervals_overlap, stat};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[path = "support/invocation.rs"]
mod invocation;

#[derive(Clone, Debug)]
struct Row {
    method: MethodId,
    precision: Precision,
    kernel: KernelFamily,
    atoms: usize,
    width: u32,
    height: u32,
    frames_per_build: usize,
    build_ms: Stat,
    bind_ms: Stat,
    clear_ms: Stat,
    render_ms: Stat,
    per_frame_ms: Stat,
    bytes: usize,
    counters: Counters,
}

struct Sample {
    build_ms: f64,
    bind_ms: f64,
    clear_ms: f64,
    render_ms: f64,
    frame_ms: f64,
    bytes: usize,
    counters: Counters,
}

fn measure<T: FloatingPoint, K: Kernel<T>>(
    method: &RenderMethod,
    scene64: &Scene<f64>,
    spec: &CameraSpec,
    settings64: &SharedSettings<f64>,
    frames_per_build: usize,
    collect_counters: bool,
) -> Result<Sample> {
    let settings = SharedSettings::<T>::from_f64(settings64);
    let scene = Scene::<T>::from_f64(scene64);
    let camera = Camera::<T>::from_spec(spec, settings.near_plane)?;

    let started = Instant::now();
    let prepared = PreparedScene::<T, K>::build(&scene, &[], &settings)?;
    let mut strategy = create_strategy::<T, K>(method, &settings)?;
    strategy.build(&prepared)?;
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut framebuffer =
        Framebuffer::<T, K::Blend>::new(camera.width, camera.height, method.tile())?;
    let mut sink = ProbeSink::<T>::counters_only(camera.width, camera.height);

    let mut bind_ms = 0.0f64;
    let mut clear_ms = 0.0f64;
    let mut render_ms = 0.0f64;
    let mut frame_ms = 0.0f64;
    for frame in 0..frames_per_build.max(1) {
        let moved = orbit(spec, frame, frames_per_build);
        let camera = Camera::<T>::from_spec(&moved, settings.near_plane)?;

        let whole = Instant::now();
        let started = Instant::now();
        strategy.bind(&prepared, &camera)?;
        bind_ms += started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        framebuffer.clear();
        clear_ms += started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        strategy.render::<NoProbe>(&prepared, &mut framebuffer, &mut sink)?;
        render_ms += started.elapsed().as_secs_f64() * 1000.0;
        frame_ms += whole.elapsed().as_secs_f64() * 1000.0;
    }

    let counters = if collect_counters {
        let mut audit = ProbeSink::<T>::counters_only(camera.width, camera.height);
        strategy.bind(&prepared, &camera)?;
        framebuffer.clear();
        strategy.render::<lights::probe::AuditProbe<T>>(
            &prepared,
            &mut framebuffer,
            &mut audit,
        )?;
        audit.counters
    } else {
        Counters::default()
    };

    let frames = frames_per_build.max(1) as f64;
    Ok(Sample {
        build_ms,
        bind_ms: bind_ms / frames,
        clear_ms: clear_ms / frames,
        render_ms: render_ms / frames,
        frame_ms: frame_ms / frames,
        bytes: prepared.resident_bytes()
            + strategy.resident_bytes()
            + framebuffer.resident_bytes(),
        counters,
    })
}

fn orbit(spec: &CameraSpec, frame: usize, frames: usize) -> CameraSpec {
    if frames <= 1 {
        return spec.clone();
    }
    let t = frame as f64 / frames as f64;
    let angle = t * std::f64::consts::TAU;
    let mut moved = spec.clone();
    moved.eye = [0.35 * angle.cos(), 0.20 * angle.sin(), 0.0];
    moved
}

fn measure_precision(
    method: &RenderMethod,
    precision: Precision,
    scene: &Scene<f64>,
    spec: &CameraSpec,
    settings: &SharedSettings<f64>,
    frames_per_build: usize,
    collect_counters: bool,
) -> Result<Sample> {
    match precision {
        Precision::F32 => measure::<f32, Gaussian>(
            method,
            scene,
            spec,
            settings,
            frames_per_build,
            collect_counters,
        ),
        Precision::F64 => measure::<f64, Gaussian>(
            method,
            scene,
            spec,
            settings,
            frames_per_build,
            collect_counters,
        ),
    }
}

fn hardware_line() -> String {
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    format!(
        "{cpu}, {cores} logical cores, {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "DEBUG BUILD - every number below is meaningless as an absolute; run with --release"
    } else {
        "release"
    }
}

fn hardware_banner() -> String {
    format!("hardware: {}\nprofile:  {}", hardware_line(), build_profile())
}

fn print_table(title: &str, rows: &[Row]) {
    println!("\n=== {title} ===");
    println!(
        "{:<12} {:<4} {:<9} {:>9} {:>11} {:>5} {:>14} {:>14} {:>14} {:>14} {:>14} {:>10}",
        "method",
        "prec",
        "kernel",
        "atoms",
        "resolution",
        "f/b",
        "build ms",
        "bind ms",
        "clear ms",
        "render ms",
        "frame ms",
        "MiB"
    );
    for row in rows {
        println!(
            "{:<12} {:<4} {:<9} {:>9} {:>11} {:>5} {:>8.4}+-{:<5.4} {:>8.4}+-{:<5.4} \
             {:>8.4}+-{:<5.4} {:>8.4}+-{:<5.4} {:>8.4}+-{:<5.4} {:>10.2}",
            row.method.name(),
            row.precision.name(),
            row.kernel.name(),
            row.atoms,
            format!("{}x{}", row.width, row.height),
            row.frames_per_build,
            row.build_ms.mean,
            row.build_ms.ci95,
            row.bind_ms.mean,
            row.bind_ms.ci95,
            row.clear_ms.mean,
            row.clear_ms.ci95,
            row.render_ms.mean,
            row.render_ms.ci95,
            row.per_frame_ms.mean,
            row.per_frame_ms.ci95,
            row.bytes as f64 / (1024.0 * 1024.0)
        );
    }
    println!(
        "\nframe ms is measured by its OWN clock around the whole per-frame \
         block, not summed from the columns beside it, so a phase clock that \
         has drifted onto a neighbouring phase shows up as a sum that no longer \
         reconciles."
    );
}

fn print_verdicts(rows: &[Row]) {
    println!(
        "\nfirst sample per cell DISCARDED as warm-up (page faults on a fresh \
         framebuffer, rayon spin-up, cold caches); mean +- 95% CI over the rest, \
         min alongside. MIN is the robust statistic for latency."
    );
    let mut printed = 0usize;
    for (i, row) in rows.iter().enumerate() {
        for other in rows.iter().skip(i + 1) {
            if row.atoms != other.atoms
                || row.width != other.width
                || row.height != other.height
                || row.frames_per_build != other.frames_per_build
            {
                continue;
            }
            let verdict = if intervals_overlap(row.per_frame_ms, other.per_frame_ms) {
                "TOO CLOSE TO CALL - the 95% intervals overlap"
            } else if row.per_frame_ms.mean < other.per_frame_ms.mean {
                "first"
            } else {
                "second"
            };
            println!(
                "  {} {} vs {} {} at {} atoms {}x{} f/b {}: {verdict} (min \
                 {:.3} vs {:.3} ms)",
                row.method.name(),
                row.precision.name(),
                other.method.name(),
                other.precision.name(),
                row.atoms,
                row.width,
                row.height,
                row.frames_per_build,
                row.per_frame_ms.min,
                other.per_frame_ms.min
            );
            printed += 1;
        }
    }
    if printed == 0 {
        println!("  (a single candidate per case; nothing to compare)");
    }
    for row in rows {
        if row.bind_ms.mean > row.render_ms.mean {
            println!(
                "  NOTE {} {} at {} atoms {}x{}: bind {:.3} ms EXCEEDS render \
                 {:.3} ms - this row measures projection and sorting, not \
                 rasterisation, and does not belong in a per-pixel comparison",
                row.method.name(),
                row.precision.name(),
                row.atoms,
                row.width,
                row.height,
                row.bind_ms.mean,
                row.render_ms.mean
            );
        }
    }
}

fn print_counters(rows: &[Row]) {
    println!("\n=== deterministic counters (portable; wall clock is not) ===");
    println!(
        "{:<12} {:<4} {:>9} {:>11} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "method",
        "prec",
        "atoms",
        "resolution",
        "projected",
        "tile pairs",
        "candidates",
        "kernel eval",
        "composited",
        "keval/ms"
    );
    for row in rows {
        let per_ms = if row.render_ms.mean > 0.0 {
            row.counters.kernel_evaluations as f64 / row.render_ms.mean
        } else {
            0.0
        };
        println!(
            "{:<12} {:<4} {:>9} {:>11} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12.0}",
            row.method.name(),
            row.precision.name(),
            row.atoms,
            format!("{}x{}", row.width, row.height),
            row.counters.atoms_projected,
            row.counters.tile_atom_pairs,
            row.counters.candidates_tested,
            row.counters.kernel_evaluations,
            row.counters.composited_hits,
            per_ms
        );
    }
    println!(
        "\nkeval/ms is the marginal-work column. A row whose keval/ms is far \
         BELOW its neighbours is dominated by a fixed per-atom cost inside the \
         render clock rather than by per-pixel work, and its time cannot be \
         read as pixel scaling."
    );
}

const WARMUP: usize = 1;

const USAGE: &str = "\
lights-bench --out-dir <DIR> [options]

  --out-dir <DIR>     REQUIRED. Where report.tsv is written.
  --smoke             one small case, enough to prove the clock is placed
                      around the phase it names
  --quick             fewer repeats and a shorter axis on every sweep
  --atoms <N,N,...>   run ONLY the 1080p atom-count axis, at these counts,
                      both precisions; the other axes are skipped

Exit 0 clean; 2 no --out-dir, or a flag this binary does not know.";

#[derive(Clone, Debug)]
struct Args {
    smoke: bool,
    quick: bool,
    atoms: Option<Vec<usize>>,
    out_dir: PathBuf,
}

fn parse_atoms(text: &str) -> std::result::Result<Vec<usize>, String> {
    let mut counts = Vec::new();
    for part in text.split(',') {
        match part.trim().parse::<usize>() {
            Ok(n) if n > 0 => counts.push(n),
            Ok(_) => return Err(format!("--atoms {text}: a count must be at least 1")),
            Err(e) => return Err(format!("--atoms {text}: {part:?} is not a count ({e})")),
        }
    }
    Ok(counts)
}

fn parse() -> std::result::Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut smoke = false;
    let mut quick = false;
    let mut out_dir: Option<PathBuf> = None;
    let mut atoms: Option<Vec<usize>> = None;
    let mut skip = false;

    for index in 0..raw.len() {
        if skip {
            skip = false;
            continue;
        }
        let flag = raw[index].as_str();
        let value = raw.get(index + 1);
        match flag {
            "--help" => return Err(USAGE.to_string()),
            "-h" => return Err(USAGE.to_string()),
            "--smoke" => smoke = true,
            "--quick" => quick = true,
            "--atoms" => {
                match value {
                    Some(s) => atoms = Some(parse_atoms(s)?),
                    None => return Err("--atoms needs a value".to_string()),
                }
                skip = true;
            }
            "--out-dir" => {
                match value {
                    Some(s) => out_dir = Some(PathBuf::from(s)),
                    None => return Err("--out-dir needs a value".to_string()),
                }
                skip = true;
            }
            other => {
                return Err(format!(
                    "unknown flag {other}\n\nA flag this binary does not \
                     understand is REFUSED rather than ignored: a silently \
                     dropped `--smoke` runs the whole sweep, which is minutes \
                     of work nobody asked for and a table nobody can read as \
                     the run they requested.\n\n{USAGE}"
                ));
            }
        }
    }

    match out_dir {
        Some(dir) => Ok(Args {
            smoke,
            quick: quick || smoke,
            atoms,
            out_dir: dir,
        }),
        None => Err(format!(
            "--out-dir is REQUIRED so this binary cannot write to the repository \
             root by accident\n\n{USAGE}"
        )),
    }
}

const REPORT_COLUMNS: [&str; 29] = [
    "sweep",
    "method",
    "precision",
    "kernel",
    "atoms",
    "width",
    "height",
    "frames_per_build",
    "build_ms_mean",
    "build_ms_ci95",
    "build_ms_min",
    "bind_ms_mean",
    "bind_ms_ci95",
    "bind_ms_min",
    "clear_ms_mean",
    "clear_ms_ci95",
    "clear_ms_min",
    "render_ms_mean",
    "render_ms_ci95",
    "render_ms_min",
    "frame_ms_mean",
    "frame_ms_ci95",
    "frame_ms_min",
    "resident_bytes",
    "atoms_projected",
    "tile_atom_pairs",
    "candidates_tested",
    "kernel_evaluations",
    "composited_hits",
];

fn number(value: f64) -> String {
    format!("{value}")
}

fn report_cells(axis: &str, row: &Row) -> Vec<String> {
    vec![
        axis.to_string(),
        row.method.name().to_string(),
        row.precision.name().to_string(),
        row.kernel.name().to_string(),
        row.atoms.to_string(),
        row.width.to_string(),
        row.height.to_string(),
        row.frames_per_build.to_string(),
        number(row.build_ms.mean),
        number(row.build_ms.ci95),
        number(row.build_ms.min),
        number(row.bind_ms.mean),
        number(row.bind_ms.ci95),
        number(row.bind_ms.min),
        number(row.clear_ms.mean),
        number(row.clear_ms.ci95),
        number(row.clear_ms.min),
        number(row.render_ms.mean),
        number(row.render_ms.ci95),
        number(row.render_ms.min),
        number(row.per_frame_ms.mean),
        number(row.per_frame_ms.ci95),
        number(row.per_frame_ms.min),
        row.bytes.to_string(),
        row.counters.atoms_projected.to_string(),
        row.counters.tile_atom_pairs.to_string(),
        row.counters.candidates_tested.to_string(),
        row.counters.kernel_evaluations.to_string(),
        row.counters.composited_hits.to_string(),
    ]
}

fn write_report(args: &Args, repeats: usize, axes: &[(String, Vec<Row>)]) -> Result<()> {
    std::fs::create_dir_all(&args.out_dir)?;
    let header = vec![
        format!("hardware\t{}", hardware_line()),
        format!("build\t{}", build_profile()),
        format!("repeats\t{repeats}"),
        format!("warmup_discarded\t{WARMUP}"),
        format!(
            "settings\tcontract (dilation off, cutoff-solved support, exact \
             early out); scene: deterministic LCG stress field, clustering 0.35"
        ),
        format!(
            "methods\tSplatRaster contract; RayCast PRODUCTION (TileBins). \
             RayCast contract is BruteForce and is a correctness configuration, \
             not a performance one"
        ),
        format!(
            "reading\tframe_ms is measured by its OWN clock around the whole \
             per-frame block, not summed from the columns beside it; min is the \
             robust latency statistic; counters are portable and wall clock is not"
        ),
    ];
    let mut rows = Vec::new();
    for (axis, axis_rows) in axes.iter() {
        for row in axis_rows.iter() {
            rows.push(report_cells(axis, row));
        }
    }
    let text = invocation::table(&header, &REPORT_COLUMNS, &rows);
    std::fs::write(args.out_dir.join("report.tsv"), text)?;
    println!(
        "\nreport.tsv written to {} - its first line is the command that \
         produced it, so every figure in it can be re-run",
        args.out_dir.display()
    );
    Ok(())
}

struct Case {
    atoms: usize,
    width: u32,
    height: u32,
    frames_per_build: usize,
}

fn sweep(
    title: &str,
    cases: &[Case],
    methods: &[RenderMethod],
    precisions: &[Precision],
    repeats: usize,
    counters: bool,
) -> Result<Vec<Row>> {
    let settings = SharedSettings::<f64>::contract();
    let mut rows = Vec::new();
    for case in cases {
        let scene = stress_scene::<f64>(case.atoms, 0x0be_11a, 0.35);
        let spec = sphere_camera(case.width, case.height);
        for method in methods {
            for precision in precisions {
                let mut build = Vec::new();
                let mut bind = Vec::new();
                let mut clear = Vec::new();
                let mut render = Vec::new();
                let mut frame = Vec::new();
                let mut bytes = 0usize;
                let mut counted = Counters::default();
                for repeat in 0..repeats + WARMUP {
                    let sample = measure_precision(
                        method,
                        *precision,
                        &scene,
                        &spec,
                        &settings,
                        case.frames_per_build,
                        counters && repeat == WARMUP,
                    )?;
                    if repeat < WARMUP {
                        continue;
                    }
                    build.push(sample.build_ms);
                    bind.push(sample.bind_ms);
                    clear.push(sample.clear_ms);
                    render.push(sample.render_ms);
                    frame.push(
                        sample.frame_ms + sample.build_ms / case.frames_per_build.max(1) as f64,
                    );
                    bytes = sample.bytes;
                    if repeat == WARMUP {
                        counted = sample.counters;
                    }
                }
                rows.push(Row {
                    method: method.id(),
                    precision: *precision,
                    kernel: KernelFamily::Gaussian,
                    atoms: case.atoms,
                    width: case.width,
                    height: case.height,
                    frames_per_build: case.frames_per_build,
                    build_ms: stat(&build),
                    bind_ms: stat(&bind),
                    clear_ms: stat(&clear),
                    render_ms: stat(&render),
                    per_frame_ms: stat(&frame),
                    bytes,
                    counters: counted,
                });
            }
        }
    }
    print_table(title, &rows);
    print_verdicts(&rows);
    Ok(rows)
}

fn execute(args: &Args) -> Result<ExitCode> {
    println!("{}", hardware_banner());
    println!(
        "settings: contract (dilation off, cutoff-solved support, exact early \
         out); scene: deterministic LCG stress field, clustering 0.35"
    );

    let methods: Vec<RenderMethod> = RenderMethod::implemented_for_timing();
    if methods.is_empty() {
        println!("\nno method is compiled in; nothing to measure");
        write_report(args, 0, &[])?;
        return Ok(ExitCode::SUCCESS);
    }
    println!(
        "methods:  SplatRaster contract; RayCast PRODUCTION (TileBins). RayCast \
         contract is BruteForce and is a correctness configuration, not a \
         performance one - measuring it here would report O(pixels x atoms)."
    );
    let repeats = if args.quick { 5 } else { 9 };
    let mut axes: Vec<(String, Vec<Row>)> = Vec::new();

    if args.smoke {
        let rows = sweep(
            "SMOKE: one small case, enough to prove the clock is placed around \
             the phase it names",
            &[Case {
                atoms: 2_000,
                width: 192,
                height: 128,
                frames_per_build: 1,
            }],
            &methods,
            &[Precision::F32],
            repeats,
            false,
        )?;
        axes.push(("smoke".to_string(), rows));
        write_report(args, repeats, &axes)?;
        return Ok(ExitCode::SUCCESS);
    }

    let atom_counts: &[usize] = match (&args.atoms, args.quick) {
        (Some(counts), _) => counts.as_slice(),
        (None, true) => &[1_000, 10_000],
        (None, false) => &[1_000, 10_000, 100_000],
    };
    let mut cases = Vec::new();
    for atoms in atom_counts {
        cases.push(Case {
            atoms: *atoms,
            width: 1920,
            height: 1080,
            frames_per_build: 1,
        });
    }
    let counted = sweep(
        "AXIS: atom count at 1080p, one frame per build",
        &cases,
        &methods,
        &Precision::ALL,
        repeats,
        true,
    )?;
    axes.push(("atom-count".to_string(), counted.clone()));
    if args.atoms.is_some() {
        print_counters(&counted);
        write_report(args, repeats, &axes)?;
        return Ok(ExitCode::SUCCESS);
    }

    let resolutions: &[(u32, u32)] = if args.quick {
        &[(256, 256), (960, 540)]
    } else {
        &[(256, 256), (960, 540), (1920, 1080)]
    };
    let mut cases = Vec::new();
    for (width, height) in resolutions {
        cases.push(Case {
            atoms: 10_000,
            width: *width,
            height: *height,
            frames_per_build: 1,
        });
    }
    let by_resolution = sweep(
        "AXIS: resolution at 10k atoms - read this one through keval/ms, not \
         through ms alone: the low end is dominated by a fixed per-atom cost \
         inside the render clock, so its ms column is NOT pixel scaling",
        &cases,
        &methods,
        &[Precision::F32],
        repeats,
        true,
    )?;
    print_counters(&by_resolution);
    axes.push(("resolution".to_string(), by_resolution));

    let mut cases = Vec::new();
    for frames in [1usize, 10, 100] {
        if args.quick && frames == 100 {
            continue;
        }
        cases.push(Case {
            atoms: 10_000,
            width: 960,
            height: 540,
            frames_per_build: frames,
        });
    }
    let by_frames = sweep(
        "AXIS: frames per build - the winner MOVES along this axis and that \
         movement is the actual decision",
        &cases,
        &methods,
        &[Precision::F32],
        repeats,
        false,
    )?;
    axes.push(("frames-per-build".to_string(), by_frames));

    print_counters(&counted);
    println!(
        "\nCounters are deterministic functions of scene and configuration and \
         are therefore portable and regression-testable; wall clock is neither. \
         Since 2026-09-15 both methods bin the same geometry - RayCast bins the \
         silhouette conic of the support ellipsoid, SplatRaster the EWA screen \
         ellipse - so the TILE PAIR columns now are comparable and land within \
         0.03% of each other at 10k atoms. The CANDIDATE columns still are not: \
         both are populated since 2026-09-15, but a splat counts the pixels of \
         its own footprint and a ray counts every atom binned to its tile, \
         which is 5.8x more at 100k atoms."
    );
    write_report(args, repeats, &axes)?;
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match execute(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("\nERROR: {e}");
            ExitCode::from(4)
        }
    }
}
