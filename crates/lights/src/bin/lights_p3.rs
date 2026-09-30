use lights::error::{LightsError, Result};
use lights::fit::analytic::AnalyticArm;
use lights::fit::edge::Criterion;
use lights::fit::harness::{ProbeRequest, ProfileChoice, run};
use lights::fit::report::{
    ATOM_ARM, Report, never_met, overfit_flagged, print_report, write_report,
};
use lights::fit::strip::{DEFAULT_GAP, DEFAULT_ZOOM, horizontal_strip, upscale};
use lights::fit::sweep::Concurrency;
use lights::fit::target::Target;
use lights::image::write_png;
use lights::kernel::{Beta, KernelFamily};
use lights::numerics::vector::Rgb;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "support/invocation.rs"]
mod invocation;

const USAGE: &str = "\
lights-p3 --out-dir <DIR> [options]

Settles ONE aspect: the ATOM CURVE. Every other axis is held at the P2
baseline on purpose, so a difference in the table is a difference in the
kernel and in nothing else.

  --out-dir <DIR>     REQUIRED. report.tsv, sweep.tsv, probes.tsv, bins.tsv,
                      every per-cell PNG and the per-target strips land here.
  --targets <LIST>    comma-separated 1..6, or `all` (default)
  --kernels <LIST>    sweep (default) | b0.5 | b1 | b2 | b4 | b8 | compact |
                      gaussian. `sweep` is the whole curve axis with the plain
                      gaussian carried along as the end-to-end control.
  --profile <NAME>    contract (default) | prefiltered. The default is NOT
                      prefiltered: the dilation is a gaussian identity, so
                      switching it on would apply a gaussian assumption to
                      non-gaussian atoms and confound the axis under test.
  --max-atoms <N>     default 8
  --tile <N>          default 16
  --repeats <N>       timing repeats after one discarded warm-up, default 5
  --depth <N>         exact-coverage subdivision depth, default 10
  --max-width <PX>    criterion: 10-90 transition width, default 1.0
  --min-contrast <F>  criterion: contrast fraction, default 0.9
  --sensitivity       also run the dilation and cutoff sweeps (off by default:
                      those knobs belong to a later aspect)
  --sweep <MODE>      parallel (default) fits every (target, blend, profile)
                      cell at once, one whole fit per job; serial fits them
                      one after another. Render timing runs serially after
                      every fit has returned, and the tables are
                      bit-identical either way except the wall-clock columns.
  --gap <N>           strip separator width in pixels, default 3
  --zoom <N>          zoom factor for the second glyph strip, default 4
  --no-strips         skip the strip PNGs

Exit 0 clean; 1 a cell never met its criterion inside --max-atoms (a RESULT,
flagged in the table, and expected here: P2 established the FITTER stops these
cells, not the budget); 2 no --out-dir; 3 the analytic control arm failed its
own criterion, so the criterion is invalid and no atom count is printed; 4 an
error.";

#[derive(Clone, Debug)]
struct Args {
    request: ProbeRequest,
    out_dir: PathBuf,
    gap: u32,
    zoom: u32,
    strips: bool,
}

fn parse_targets(text: &str) -> std::result::Result<Vec<usize>, String> {
    let count = Target::<f64>::all().len();
    if text == "all" {
        return Ok((0..count).collect());
    }
    let mut out = Vec::new();
    for piece in text.split(',') {
        let n = match piece.trim().trim_start_matches('t').parse::<usize>() {
            Ok(v) => v,
            Err(e) => return Err(format!("target {piece}: {e}")),
        };
        if n < 1 || n > count {
            return Err(format!("target {n} is outside 1..={count}"));
        }
        out.push(n - 1);
    }
    Ok(out)
}

fn parse_kernels(text: &str) -> std::result::Result<Vec<KernelFamily>, String> {
    if text == "sweep" {
        return Ok(KernelFamily::CURVE_SWEEP.to_vec());
    }
    let mut out = Vec::new();
    for piece in text.split(',') {
        match piece.trim() {
            "b0.5" => out.push(KernelFamily::SuperGaussian(Beta::Half)),
            "b1" => out.push(KernelFamily::SuperGaussian(Beta::One)),
            "b2" => out.push(KernelFamily::SuperGaussian(Beta::Two)),
            "b4" => out.push(KernelFamily::SuperGaussian(Beta::Four)),
            "b8" => out.push(KernelFamily::SuperGaussian(Beta::Eight)),
            "compact" => out.push(KernelFamily::CompactBump),
            "gaussian" => out.push(KernelFamily::Gaussian),
            other => return Err(format!("unknown kernel {other}")),
        }
    }
    Ok(out)
}

fn parse_profile(text: &str) -> std::result::Result<Vec<ProfileChoice>, String> {
    match text.trim() {
        "contract" => Ok(vec![ProfileChoice::Contract]),
        "prefiltered" => Ok(vec![ProfileChoice::Prefiltered]),
        other => Err(format!(
            "unknown profile {other}: this run holds the profile FIXED, so it \
             takes one name and not a list"
        )),
    }
}

fn parse() -> std::result::Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut request = ProbeRequest::all();
    request.blends = KernelFamily::CURVE_SWEEP.to_vec();
    request.profiles = vec![ProfileChoice::Contract];
    request.sensitivity = false;
    let mut criterion = Criterion::registered();
    let mut out_dir: Option<PathBuf> = None;
    let mut gap = DEFAULT_GAP;
    let mut zoom = DEFAULT_ZOOM;
    let mut strips = true;
    let mut skip = false;

    for index in 0..raw.len() {
        if skip {
            skip = false;
            continue;
        }
        let flag = raw[index].as_str();
        let value = raw.get(index + 1);
        let need = |v: Option<&String>| -> std::result::Result<String, String> {
            match v {
                Some(s) => Ok(s.clone()),
                None => Err(format!("{flag} needs a value")),
            }
        };
        match flag {
            "--help" => return Err(USAGE.to_string()),
            "-h" => return Err(USAGE.to_string()),
            "--out-dir" => {
                out_dir = Some(PathBuf::from(need(value)?));
                skip = true;
            }
            "--targets" => {
                request.targets = parse_targets(&need(value)?)?;
                skip = true;
            }
            "--kernels" => {
                request.blends = parse_kernels(&need(value)?)?;
                skip = true;
            }
            "--profile" => {
                request.profiles = parse_profile(&need(value)?)?;
                skip = true;
            }
            "--max-atoms" => {
                request.max_atoms = need(value)?
                    .parse::<usize>()
                    .map_err(|e| format!("max-atoms: {e}"))?;
                skip = true;
            }
            "--tile" => {
                request.tile = need(value)?
                    .parse::<u32>()
                    .map_err(|e| format!("tile: {e}"))?;
                skip = true;
            }
            "--repeats" => {
                request.repeats = need(value)?
                    .parse::<usize>()
                    .map_err(|e| format!("repeats: {e}"))?;
                skip = true;
            }
            "--depth" => {
                request.quadrature_depth = need(value)?
                    .parse::<u32>()
                    .map_err(|e| format!("depth: {e}"))?;
                skip = true;
            }
            "--max-width" => {
                criterion.max_transition = need(value)?
                    .parse::<f64>()
                    .map_err(|e| format!("max-width: {e}"))?;
                skip = true;
            }
            "--min-contrast" => {
                criterion.min_contrast_frac = need(value)?
                    .parse::<f64>()
                    .map_err(|e| format!("min-contrast: {e}"))?;
                skip = true;
            }
            "--gap" => {
                gap = need(value)?.parse::<u32>().map_err(|e| format!("gap: {e}"))?;
                skip = true;
            }
            "--zoom" => {
                zoom = need(value)?
                    .parse::<u32>()
                    .map_err(|e| format!("zoom: {e}"))?;
                skip = true;
            }
            "--sensitivity" => request.sensitivity = true,
            "--sweep" => {
                request.concurrency = Concurrency::parse(&need(value)?)?;
                skip = true;
            }
            "--no-strips" => strips = false,
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }

    request.criterion = criterion;
    if request.blends.is_empty() {
        return Err("--kernels selected nothing to compare".to_string());
    }
    for kernel in request.blends.iter() {
        if !kernel.blend_is_over() {
            return Err(format!(
                "{} does not composite with Over. This run settles the ATOM \
                 CURVE and holds the blend rule fixed; a kernel that moves both \
                 at once cannot be read",
                kernel.name()
            ));
        }
    }
    match out_dir {
        Some(dir) => Ok(Args {
            request,
            out_dir: dir,
            gap,
            zoom,
            strips,
        }),
        None => Err(format!(
            "--out-dir is REQUIRED so this binary cannot write to the repository \
             root by accident\n\n{USAGE}"
        )),
    }
}

fn panel_names(args: &Args, target: &str) -> Vec<(String, String)> {
    let profile = match args.request.profiles.first() {
        Some(p) => p.name(),
        None => "contract",
    };
    let mut out = Vec::new();
    for kernel in args.request.blends.iter() {
        out.push((
            format!("{target}-{}-{profile}-atoms", kernel.name()),
            kernel.name().to_string(),
        ));
    }
    out.push((
        format!("{target}-{}", AnalyticArm::Exact.name()),
        AnalyticArm::Exact.name().to_string(),
    ));
    out
}

fn atoms_of(report: &Report, target: &str, kernel: &str) -> usize {
    for row in report.rows.iter() {
        if row.target == target && row.blend == kernel && row.arm == ATOM_ARM {
            return row.atoms;
        }
    }
    0
}

fn emit_strips(args: &Args, report: &Report) -> Result<Vec<(PathBuf, String)>> {
    let mut manifest = Vec::new();
    let all = Target::<f64>::all();
    for index in args.request.targets.iter() {
        let target = match all.get(*index) {
            Some(t) => t.name,
            None => {
                return Err(LightsError::InvalidCamera(
                    "lights-p3 was given a target index that does not exist",
                ));
            }
        };
        let wanted = panel_names(args, target);
        let mut panels = Vec::new();
        let mut labels = Vec::new();
        for (sheet_name, label) in wanted.iter() {
            let found = report.sheets.iter().find(|s| &s.name == sheet_name);
            match found {
                Some(sheet) => {
                    panels.push(sheet.image.clone());
                    labels.push(match label.as_str() {
                        "analytic-exact" => label.clone(),
                        other => format!("{other} ({} atoms)", atoms_of(report, target, other)),
                    });
                }
                None => {
                    return Err(LightsError::InvalidCamera(
                        "a sheet the strip needs is missing from the report",
                    ));
                }
            }
        }
        let strip = horizontal_strip(&panels, args.gap, Rgb::new(0.25, 0.0, 0.35))?;
        let path = args.out_dir.join(format!("strip-{target}.png"));
        write_png(&path, &strip.to_clamped_u8())?;
        manifest.push((
            path,
            format!(
                "{target} at 1x, left to right: {}. Separator bars are purple so \
                 a panel boundary can never be read as image content.",
                labels.join(" | ")
            ),
        ));
        if args.zoom > 1 {
            let zoomed = upscale(&strip, args.zoom)?;
            let path = args
                .out_dir
                .join(format!("strip-{target}-{}x.png", args.zoom));
            write_png(&path, &zoomed.to_clamped_u8())?;
            manifest.push((
                path,
                format!(
                    "{target} at {}x, nearest-neighbour so every pixel the atoms \
                     actually produced stays a flat square, same order.",
                    args.zoom
                ),
            ));
        }
    }
    Ok(manifest)
}

fn absolute(path: &Path) -> String {
    match std::fs::canonicalize(path) {
        Ok(p) => p.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn execute(args: &Args) -> Result<ExitCode> {
    let report = run(&args.request)?;
    let tables = write_report(&args.out_dir, &report)?;
    invocation::stamp(&tables)?;

    if !report.header.control_arm_valid {
        lights::fit::report::print_header(&report.header);
        eprintln!(
            "\nCONTROL ARM FAILED ITS OWN CRITERION. The analytic-exact arm is \
             the quality ceiling: nothing can beat exact box coverage of the \
             same shape. A criterion it cannot meet is a verdict on the \
             threshold, not on the atoms, so NO ATOM COUNT IS PRINTED."
        );
        return Ok(ExitCode::from(3));
    }

    print_report(&report);

    println!(
        "\n=== READ THIS BEFORE READING ANY NUMBER ABOVE ===\n\
         P2 established that the FITTER, not the atom budget, stopped 20 of 24 \
         cells, so every ABSOLUTE atom count here is an UPPER BOUND and a \
         statement about greedy matching pursuit. What this run compares is \
         kernels against each other under the SAME fitter, and \
         super-gaussian-b1 reproduces the gaussian BIT FOR BIT, so the RELATIVE \
         ranking is fair. One caveat measured in tests/fit_greedy.rs: the \
         search itself is gaussian-tuned, and on a known single-atom target it \
         reaches the optimum to within 1/255 under the gaussian, beta = 1 and \
         the compact bump, but NOT under beta = 0.5, 2, 4 or 8. Part of any gap \
         those columns show is the search, not the curve."
    );

    if args.strips {
        let manifest = emit_strips(args, &report)?;
        println!("\n=== IMAGE MANIFEST ===");
        for (path, caption) in manifest.iter() {
            println!("{}\n    {caption}", absolute(path));
        }
    }

    let flagged = overfit_flagged(&report.rows);
    if !flagged.is_empty() {
        println!(
            "\nSAMPLE-GRID OVERFIT FLAGGED on {} cell(s): the width measured on \
             the grid the atoms were fitted to differs from the width through a \
             half-pixel-shifted camera by more than {} px, so those atoms are \
             cancelling aliasing AT THE SAMPLED PIXEL POSITIONS rather than \
             representing the shape:",
            flagged.len(),
            lights::fit::report::OVERFIT_FLAG_PX
        );
        for row in flagged {
            println!(
                "  {} {} fitted {:.4} px, shifted {:.4} px, gap {:.4} px",
                row.target, row.blend, row.transition_10_90_px,
                row.shifted_grid_transition_10_90_px, row.overfit_gap_px
            );
        }
    }

    let missed = never_met(&report.rows);
    if missed.is_empty() {
        println!("\nEVERY CELL MET ITS CRITERION INSIDE THE ATOM BUDGET.");
        return Ok(ExitCode::SUCCESS);
    }
    println!(
        "\n{} CELL(S) NEVER MET THE CRITERION INSIDE --max-atoms {}. That is a \
         RESULT, not a failure of the run, and the budget was NOT extended \
         until something passed:",
        missed.len(),
        args.request.max_atoms
    );
    for row in missed {
        println!(
            "  {} {} at {} atoms: 10-90 {:.4} px, contrast {:.4}, peak {:.4}, \
             fwhm {:.4} px",
            row.target,
            row.blend,
            row.atoms_available,
            row.transition_10_90_px,
            row.contrast_frac,
            row.peak_frac,
            row.fwhm_px
        );
    }
    Ok(ExitCode::FAILURE)
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
        Err(LightsError::BudgetExhausted { site, limit }) => {
            eprintln!(
                "\nBUDGET EXHAUSTED at {site} (limit {limit}). A truncated fit \
                 would report an atom count that measures the budget rather \
                 than the representation, so the run fails instead."
            );
            ExitCode::from(4)
        }
        Err(e) => {
            eprintln!("\nERROR: {e}");
            ExitCode::from(4)
        }
    }
}
