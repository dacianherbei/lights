use lights::error::{LightsError, Result};
use lights::fit::edge::Criterion;
use lights::fit::harness::{ProbeRequest, ProfileChoice, run};
use lights::fit::report::{never_met, overfit_flagged, print_report, write_report};
use lights::fit::sweep::Concurrency;
use lights::fit::target::Target;
use lights::kernel::KernelFamily;
use std::path::PathBuf;
use std::process::ExitCode;

#[path = "support/invocation.rs"]
mod invocation;

const USAGE: &str = "\
lights-p2 --out-dir <DIR> [options]

  --out-dir <DIR>     REQUIRED. Where report.tsv, sweep.tsv, sensitivity.tsv
                      and the per-target PNGs are written.
  --targets <LIST>    comma-separated 1..6, or `all` (default)
  --blends <LIST>     over | normalised | all (default); `all` is the BLEND
                      PAIR, not every KernelFamily: the atom-curve families
                      belong to lights-p3, which sweeps that axis instead
  --profiles <LIST>   prefiltered | contract | all (default)
  --max-atoms <N>     default 8
  --tile <N>          default 16
  --repeats <N>       timing repeats after one discarded warm-up, default 5
  --tiled-field       also measure each cell tiled 40x into 512x512
  --no-sensitivity    skip the dilation and cutoff sweeps
  --sweep <MODE>      parallel (default) fits every (target, blend, profile)
                      cell at once, one whole fit per job; serial fits them
                      one after another. Render timing runs serially after
                      every fit has returned, and the tables are
                      bit-identical either way except the wall-clock columns.
  --depth <N>         exact-coverage subdivision depth, default 10
  --max-width <PX>    criterion: 10-90 transition width, default 1.0
  --min-contrast <F>  criterion: contrast fraction, default 0.9

Exit 0 clean; 1 a target never met its criterion inside --max-atoms (a
RESULT, flagged in the table); 2 no --out-dir; 3 the analytic control arm
failed its own criterion, so the criterion is invalid and no atom count is
printed; 4 an error.";

#[derive(Clone, Debug)]
struct Args {
    request: ProbeRequest,
    out_dir: PathBuf,
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

fn parse_blends(text: &str) -> std::result::Result<Vec<KernelFamily>, String> {
    if text == "all" {
        return Ok(KernelFamily::BLEND_PAIR.to_vec());
    }
    let mut out = Vec::new();
    for piece in text.split(',') {
        match piece.trim() {
            "over" => out.push(KernelFamily::Gaussian),
            "gaussian" => out.push(KernelFamily::Gaussian),
            "normalised" => out.push(KernelFamily::GaussianNormalised),
            other => return Err(format!("unknown blend {other}")),
        }
    }
    Ok(out)
}

fn parse_profiles(text: &str) -> std::result::Result<Vec<ProfileChoice>, String> {
    if text == "all" {
        return Ok(ProfileChoice::ALL.to_vec());
    }
    let mut out = Vec::new();
    for piece in text.split(',') {
        match piece.trim() {
            "prefiltered" => out.push(ProfileChoice::Prefiltered),
            "contract" => out.push(ProfileChoice::Contract),
            other => return Err(format!("unknown profile {other}")),
        }
    }
    Ok(out)
}

fn parse() -> std::result::Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut request = ProbeRequest::all();
    let mut criterion = Criterion::registered();
    let mut out_dir: Option<PathBuf> = None;
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
            "--blends" => {
                request.blends = parse_blends(&need(value)?)?;
                skip = true;
            }
            "--profiles" => {
                request.profiles = parse_profiles(&need(value)?)?;
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
            "--tiled-field" => request.tiled_field = true,
            "--no-sensitivity" => request.sensitivity = false,
            "--sweep" => {
                request.concurrency = Concurrency::parse(&need(value)?)?;
                skip = true;
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }

    request.criterion = criterion;
    match out_dir {
        Some(dir) => Ok(Args { request, out_dir: dir }),
        None => Err(format!(
            "--out-dir is REQUIRED so this binary cannot write to the repository \
             root by accident\n\n{USAGE}"
        )),
    }
}

fn threads_the_render_used(request: &ProbeRequest) -> usize {
    let resolved = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut most = 0usize;
    for profile in request.profiles.iter() {
        let configured = profile.settings().threads;
        let actual = if configured == 0 { resolved } else { configured };
        most = most.max(actual);
    }
    most.max(1)
}

fn execute(args: &Args) -> Result<ExitCode> {
    let mut report = run(&args.request)?;
    report.header.threads = threads_the_render_used(&args.request);
    let tables = write_report(&args.out_dir, &report)?;
    invocation::stamp(&tables)?;

    if !report.header.control_arm_valid {
        lights::fit::report::print_header(&report.header);
        eprintln!(
            "\nCONTROL ARM FAILED ITS OWN CRITERION. The analytic-exact arm is \
             the quality ceiling: nothing can beat exact box coverage of the \
             same shape. A criterion it cannot meet is a verdict on the \
             threshold, not on the atoms, so NO ATOM COUNT IS PRINTED. \
             report.tsv carries the control rows so the failing measurement can \
             be inspected."
        );
        return Ok(ExitCode::from(3));
    }

    print_report(&report);

    let flagged = overfit_flagged(&report.rows);
    if !flagged.is_empty() {
        println!(
            "\nSAMPLE-GRID OVERFIT FLAGGED on {} cell(s). The width measured on \
             the grid the atoms were fitted to differs from the width measured \
             through a half-pixel-shifted camera by more than {} px, which \
             means those atoms are cancelling aliasing AT THE SAMPLED PIXEL \
             POSITIONS rather than representing the shape:",
            flagged.len(),
            lights::fit::report::OVERFIT_FLAG_PX
        );
        for row in flagged {
            println!(
                "  {} {} {} fitted {:.4} px, shifted {:.4} px, gap {:.4} px",
                row.target,
                row.blend,
                row.profile,
                row.transition_10_90_px,
                row.shifted_grid_transition_10_90_px,
                row.overfit_gap_px
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
            "  {} {} {} at {} atoms: 10-90 {:.4} px, contrast {:.4}, peak \
             {:.4}, fwhm {:.4} px",
            row.target,
            row.blend,
            row.profile,
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
