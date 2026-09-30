use lights::error::{LightsError, Result};
use lights::fit::aspect2::{ClassPanels, TermChoice, ZOOMS, caption, class_sweep, strip_for_class};
use lights::fit::content::Content;
use lights::fit::nterm::{
    ATOM_SCALARS_NATIVE, Arm, ByteRule, ByteVerdict, MAX_ATOMS, NTermCurve, NTermRequest,
    REACHABLE_ARMS, VALUE_BYTES, byte_crossings, nterm_curve,
};
use lights::fit::strip::{DEFAULT_GAP, upscale};
use lights::fit::sweep::Concurrency;
use lights::image::write_png;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "support/invocation.rs"]
mod invocation;

const USAGE: &str = "\
lights-p4 --out-dir <DIR> [options]

Draws the aspect-2 image strips: every arm on every content class, at 1x and
at 4x, one strip per class per zoom, and prints a manifest of absolute paths
with a one-line caption each.

  --out-dir <DIR>     REQUIRED. Every PNG lands here and nowhere else.
  --classes <LIST>    comma-separated names or 1..8, or `all` (default).
                      1..4 are the carried-forward t2, t3, t5, t6; 5..8 are
                      the new n1, n2, n3, n4, which have NO analytic control.
  --terms <SPEC>      `shared` (default) renders every column at the smallest
                      ceiling in the strip, which is the atom ceiling. An
                      integer renders every column at that count; a column
                      whose ceiling is lower renders its ceiling and the
                      caption marks it CEILING.
  --max-atoms <N>     atom budget handed to the greedy fitter, default 8
  --tile <N>          default 16
  --depth <N>         exact-coverage subdivision depth, default 10
  --gap <N>           separator width between panels in pixels, default 3
  --zoom <N>          the second strip's zoom factor, default 4
  --sweep <MODE>      parallel (default) fits every (class, arm) pair at once,
                      one whole fit per job; serial fits them one after
                      another. The strips are bit-identical either way.
  --nterm             draw no strips. Instead fit every (class, arm) pair
                      once and write nterm.tsv (one row per ladder rung) and
                      crossings.tsv (per class, per atom rung, per transform
                      arm: the bytes that transform needs to reach the atom
                      rung's relative L2, bracketed, against the atoms' own
                      nine-scalar f32 bytes).

Exit 0 clean; 2 no --out-dir or a bad flag; 4 an error.";

#[derive(Clone, Debug)]
struct Args {
    classes: Vec<usize>,
    arms: Vec<Arm>,
    choice: TermChoice,
    request: NTermRequest,
    out_dir: PathBuf,
    gap: u32,
    zoom: u32,
    concurrency: Concurrency,
    nterm: bool,
}

fn class_names() -> Vec<&'static str> {
    Content::all().iter().map(|c| c.name()).collect()
}

fn parse_classes(text: &str) -> std::result::Result<Vec<usize>, String> {
    let names = class_names();
    if text == "all" {
        return Ok((0..names.len()).collect());
    }
    let mut out = Vec::new();
    for piece in text.split(',') {
        let token = piece.trim();
        let found = names.iter().position(|n| *n == token || n.starts_with(token));
        match found {
            Some(index) => out.push(index),
            None => match token.parse::<usize>() {
                Ok(n) if n >= 1 && n <= names.len() => out.push(n - 1),
                Ok(n) => return Err(format!("class {n} is outside 1..={}", names.len())),
                Err(e) => return Err(format!("class {token}: {e}; known classes are {}", names.join(", "))),
            },
        }
    }
    Ok(out)
}

fn parse_terms(text: &str) -> std::result::Result<TermChoice, String> {
    let token = text.trim();
    if token == "shared" {
        return Ok(TermChoice::SharedCeiling);
    }
    match token.parse::<usize>() {
        Ok(0) => Err("--terms 0 draws nothing but the background".to_string()),
        Ok(n) => Ok(TermChoice::Explicit(n)),
        Err(e) => Err(format!("--terms {token}: {e}")),
    }
}

fn parse() -> std::result::Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut classes: Vec<usize> = (0..class_names().len()).collect();
    let mut choice = TermChoice::SharedCeiling;
    let mut request = NTermRequest::new(REACHABLE_ARMS[0]);
    request.max_atoms = MAX_ATOMS;
    let mut out_dir: Option<PathBuf> = None;
    let mut gap = DEFAULT_GAP;
    let mut zoom = ZOOMS[1];
    let mut concurrency = Concurrency::Parallel;
    let mut nterm = false;
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
            "--classes" => {
                classes = parse_classes(&need(value)?)?;
                skip = true;
            }
            "--terms" => {
                choice = parse_terms(&need(value)?)?;
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
            "--depth" => {
                request.quadrature_depth = need(value)?
                    .parse::<u32>()
                    .map_err(|e| format!("depth: {e}"))?;
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
            "--sweep" => {
                concurrency = Concurrency::parse(&need(value)?)?;
                skip = true;
            }
            "--nterm" => {
                nterm = true;
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }

    if classes.is_empty() {
        return Err("--classes selected nothing to draw".to_string());
    }
    match out_dir {
        Some(dir) => Ok(Args {
            classes,
            arms: REACHABLE_ARMS.to_vec(),
            choice,
            request,
            out_dir: dir,
            gap,
            zoom,
            concurrency,
            nterm,
        }),
        None => Err(format!(
            "--out-dir is REQUIRED so this binary cannot write to the repository \
             root by accident\n\n{USAGE}"
        )),
    }
}

fn absolute(path: &Path) -> String {
    match std::fs::canonicalize(path) {
        Ok(p) => p.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn banner(args: &Args) {
    println!(
        "=== READ THIS BEFORE READING ANY STRIP BELOW ===\n\
         THE COMPARISON IS STRUCTURALLY ASYMMETRIC. The two transform columns \
         have NO fitter: keeping the N largest coefficients under the \
         subband-weighted ranking is their optimum in that basis, obtained \
         exactly and for free. The two atom columns got greedy matching \
         pursuit, which the earlier run measured as a LOWER BOUND on atom \
         quality in 20 of 24 cells, hand-built atoms having beaten it in both \
         directions. A strip in which a transform column looks better is \
         therefore ambiguous between `the transform represents this content \
         better` and `the transform got its optimum and the atoms did not`, \
         and no caption here resolves that.\n\
         THE ANALYTIC COLUMN IS ABSENT ON n1..n4, drawn as a magenta hatch \
         that no content class can produce. It was NOT replaced by a weak \
         invented arm: no SDF describes a Shepard colour field, a blurred \
         shadow, value noise or a broadband patch.\n\
         TWO OF THE SIX ARMS ARE MISSING AND THAT IS A FITTER LIMIT, NOT A \
         RESULT. The signed arms (Gabor, Gauss-Hermite) are KernelFamily \
         variants and the greedy fitter accepts them, but each at a \
         fixed carrier or lobe chosen by the crate: the ladder carries no \
         frequency, orientation or phase knob. A column fitted that way would \
         measure the carrier this crate happened to fix, not the atom, and \
         drawing it would be the invented arm this run refuses everywhere \
         else.\n\
         TERMS: every column in one strip is rendered at the same requested \
         count ({}). At the shared ceiling the transforms have too few \
         coefficients to place even a constant colour, which flatters the \
         atoms; at a high count the atom columns are pinned at their ceiling, \
         which flatters the transforms. Read both sets or neither.",
        args.choice.name()
    );
}

fn emit(args: &Args, class: &ClassPanels) -> Result<Vec<(PathBuf, String, u32)>> {
    let strip = strip_for_class(class, args.gap)?;
    let mut out = Vec::new();
    let base = format!("strip-{}-n{}", class.content, class.requested);
    let path = args.out_dir.join(format!("{base}.png"));
    write_png(&path, &strip.to_clamped_u8())?;
    out.push((path, caption(class, 1), 1));
    if args.zoom > 1 {
        let zoomed = upscale(&strip, args.zoom)?;
        let path = args.out_dir.join(format!("{base}-{}x.png", args.zoom));
        write_png(&path, &zoomed.to_clamped_u8())?;
        out.push((path, caption(class, args.zoom), args.zoom));
    }
    Ok(out)
}

const REPORT_COLUMNS: [&str; 9] = [
    "content",
    "viewport_width",
    "viewport_height",
    "control",
    "shared_ceiling",
    "requested",
    "zoom",
    "columns",
    "png",
];

fn write_report(args: &Args, rows: &[Vec<String>]) -> Result<()> {
    let header = vec![
        format!("terms\t{}", args.choice.name()),
        format!("max_atoms\t{}", args.request.max_atoms),
        format!("tile\t{}", args.request.tile),
        format!("quadrature_depth\t{}", args.request.quadrature_depth),
        format!("gap_px\t{}", args.gap),
        format!(
            "reading\tthe comparison is structurally asymmetric: the transform \
             columns get their exact optimum for free and the atom columns get \
             greedy matching pursuit, which is a LOWER BOUND on atom quality"
        ),
        format!(
            "absent\tthe analytic column is absent on n1..n4 and is drawn as a \
             magenta hatch; it was NOT replaced by an invented arm"
        ),
    ];
    let text = invocation::table(&header, &REPORT_COLUMNS, rows);
    std::fs::write(args.out_dir.join("report.tsv"), text)?;
    Ok(())
}

fn report_row(class: &ClassPanels, zoom: u32, path: &Path) -> Vec<String> {
    vec![
        class.content.clone(),
        class.viewport.0.to_string(),
        class.viewport.1.to_string(),
        class.control.clone(),
        class.shared_ceiling.to_string(),
        class.requested.to_string(),
        zoom.to_string(),
        class.column_labels().join(","),
        match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => absolute(path),
        },
    ]
}

const NTERM_COLUMNS: [&str; 12] = [
    "content",
    "viewport_width",
    "viewport_height",
    "arm",
    "optimality",
    "byte_rule",
    "terms",
    "reading",
    "relative_l2",
    "max_abs_error",
    "bytes_explicit",
    "bytes_combinatorial",
];

const CROSSING_COLUMNS: [&str; 12] = [
    "content",
    "atom_arm",
    "transform_arm",
    "transform_optimality",
    "atoms",
    "atom_relative_l2",
    "atom_bytes_native",
    "transform_terms_above",
    "transform_terms",
    "transform_bytes_low",
    "transform_bytes_high",
    "verdict",
];

fn cell(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.6}"),
        None => "-".to_string(),
    }
}

fn nterm_header(args: &Args) -> Vec<String> {
    vec![
        format!("max_atoms\t{}", args.request.max_atoms),
        format!("tile\t{}", args.request.tile),
        format!("quadrature_depth\t{}", args.request.quadrature_depth),
        format!("value_bytes\t{VALUE_BYTES}"),
        format!("atom_scalars_native\t{ATOM_SCALARS_NATIVE}"),
        format!(
            "reading\tatom bytes are nine f32 scalars an atom; a transform's low bracket is \
             the combinatorial cost one term past the last ladder rung that missed, its high \
             bracket the explicit cost at the first rung that reached. The low bracket assumes \
             the error does not rise with more terms, which holds for an orthonormal basis and \
             is not proven for a biorthogonal one"
        ),
        format!(
            "asymmetry\tthe atom curve is a greedy LOWER BOUND on atom quality, and a \
             transform gets its ranked optimum for free"
        ),
    ]
}

fn curve_rows(curve: &NTermCurve) -> Vec<Vec<String>> {
    curve
        .points
        .iter()
        .map(|p| {
            vec![
                curve.content.clone(),
                curve.viewport.0.to_string(),
                curve.viewport.1.to_string(),
                curve.arm.clone(),
                curve.optimality.name().to_string(),
                curve.byte_rule.name().to_string(),
                p.terms.to_string(),
                p.reading.name().to_string(),
                format!("{:.6}", p.relative_l2),
                format!("{:.6}", p.max_abs_error),
                format!("{:.6}", p.bytes_explicit),
                format!("{:.6}", p.bytes_combinatorial),
            ]
        })
        .collect()
}

fn execute_nterm(args: &Args, chosen: &[Content]) -> Result<ExitCode> {
    let mut jobs: Vec<(usize, Arm)> = Vec::with_capacity(chosen.len() * args.arms.len());
    for position in 0..chosen.len() {
        for arm in args.arms.iter() {
            jobs.push((position, *arm));
        }
    }
    let curves = lights::fit::sweep::map(args.concurrency, &jobs, |(position, arm)| {
        let mut request = args.request.clone();
        request.arm = *arm;
        nterm_curve(&chosen[*position], &request)
    })?;

    let mut points = Vec::new();
    for curve in curves.iter() {
        points.extend(curve_rows(curve));
    }
    let mut crossings = Vec::new();
    let mut tally = [0usize; 4];
    let arms = args.arms.len();
    for position in 0..chosen.len() {
        let own = &curves[position * arms..(position + 1) * arms];
        for atoms in own.iter().filter(|c| c.byte_rule == ByteRule::AtomParameterVector) {
            for transform in own.iter().filter(|c| c.byte_rule == ByteRule::RetainedCoefficient) {
                for crossing in byte_crossings(atoms, transform)? {
                    let slot = match crossing.verdict {
                        ByteVerdict::TransformCheaper => 0,
                        ByteVerdict::AtomsCheaper => 1,
                        ByteVerdict::Bracketed => 2,
                        ByteVerdict::NeverReached => 3,
                    };
                    tally[slot] += 1;
                    crossings.push(vec![
                        atoms.content.clone(),
                        atoms.arm.clone(),
                        transform.arm.clone(),
                        transform.optimality.name().to_string(),
                        crossing.atoms.to_string(),
                        format!("{:.6}", crossing.atom_relative_l2),
                        format!("{:.6}", crossing.atom_bytes),
                        crossing.transform_terms_above.to_string(),
                        match crossing.transform_terms {
                            Some(t) => t.to_string(),
                            None => "-".to_string(),
                        },
                        cell(crossing.transform_bytes_low),
                        cell(crossing.transform_bytes_high),
                        crossing.verdict.name().to_string(),
                    ]);
                }
            }
        }
    }

    let header = nterm_header(args);
    std::fs::write(
        args.out_dir.join("nterm.tsv"),
        invocation::table(&header, &NTERM_COLUMNS, &points),
    )?;
    std::fs::write(
        args.out_dir.join("crossings.tsv"),
        invocation::table(&header, &CROSSING_COLUMNS, &crossings),
    )?;
    println!(
        "\n{} curve(s), {} rung row(s), {} crossing(s) written to {}",
        curves.len(),
        points.len(),
        crossings.len(),
        absolute(&args.out_dir)
    );
    for (verdict, count) in ByteVerdict::ALL.iter().zip(tally.iter()) {
        println!("  {:<18} {count}", verdict.name());
    }
    Ok(ExitCode::SUCCESS)
}

fn execute(args: &Args) -> Result<ExitCode> {
    std::fs::create_dir_all(&args.out_dir)?;

    let all = Content::all();
    let mut chosen = Vec::with_capacity(args.classes.len());
    for index in args.classes.iter() {
        match all.get(*index) {
            Some(c) => chosen.push(c.clone()),
            None => {
                return Err(LightsError::InvalidCamera(
                    "lights-p4 was given a content class index that does not exist",
                ));
            }
        }
    }
    if args.nterm {
        return execute_nterm(args, &chosen);
    }
    banner(args);
    let classes = class_sweep(&chosen, &args.arms, args.choice, &args.request, args.concurrency)?;
    let mut manifest = Vec::new();
    let mut table = Vec::new();
    for class in classes.iter() {
        println!(
            "\n{} {}x{}: control {}, shared ceiling {} term(s), rendered at {} term(s), columns {}",
            class.content,
            class.viewport.0,
            class.viewport.1,
            class.control,
            class.shared_ceiling,
            class.requested,
            class.column_labels().join(", ")
        );
        for (path, line, zoom) in emit(args, class)? {
            table.push(report_row(class, zoom, &path));
            manifest.push((path, line));
        }
    }

    write_report(args, &table)?;

    println!("\n=== IMAGE MANIFEST ===");
    for (path, line) in manifest.iter() {
        println!("{}\n    {line}", absolute(path));
    }
    println!(
        "\n{} image(s) written, and report.tsv beside them opens with the \
         command that produced them.",
        manifest.len()
    );
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
        Err(LightsError::BudgetExhausted { site, limit }) => {
            eprintln!(
                "\nBUDGET EXHAUSTED at {site} (limit {limit}). A truncated fit \
                 would put a picture of the budget on the page instead of a \
                 picture of the representation, so the run fails instead."
            );
            ExitCode::from(4)
        }
        Err(e) => {
            eprintln!("\nERROR: {e}");
            ExitCode::from(4)
        }
    }
}
