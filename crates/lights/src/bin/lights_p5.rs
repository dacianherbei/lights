use lights::error::{LightsError, Result};
use lights::fit::aspect2::{absent_panel, separator};
use lights::fit::atoms_io;
use lights::fit::aspect3::{
    Arm3, Aspect3Request, ClassLadder, Direction, Independence, Resample, arms, fit_at,
    header_lines, render_atoms_at, write_table,
};
use lights::fit::harness::ProfileChoice;
use lights::fit::nterm::relative_l2;
use lights::fit::scale::{Convention, Scale, ScaleClass};
use lights::fit::sweep::Concurrency;
use lights::fit::strip::{DEFAULT_GAP, horizontal_strip, upscale};
use lights::fit::target::Target;
use lights::image::{FloatImage, write_png};
use lights::kernel::KernelFamily;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "support/invocation.rs"]
mod invocation;

const USAGE_HEAD: &str = "\
lights-p5 --out-dir <DIR> [options]

Settles ONE aspect: RESOLUTION INDEPENDENCE. Fit once, freeze the description,
render it at a ladder of scales, and measure how quality decays away from the
fitting scale. Every other axis is held at the aspect-1 and aspect-2 baseline.

  --out-dir <DIR>     REQUIRED. Both TSVs, every PNG and the manifest land here.";

const USAGE_TAIL: &str = "  --direction <NAME>  magnify | minify | both (default). magnify fits at 1x and
                      renders out to 8x; minify fits at 8x and renders in to a
                      quarter.
  --profile <NAME>    contract | prefiltered | both (default). Minification is
                      run under BOTH on purpose: contract() ships NO pixel
                      prefilter, so a minification loss under it would be the
                      missing prefilter rather than the representation.
  --max-atoms <N>     atom budget handed to the greedy fitter, default 8
  --terms <N>         coefficients kept by each wavelet arm, default 64
  --tile <N>          default 16
  --depth <N>         exact-coverage subdivision depth, default 10
  --gap <N>           strip separator width in pixels, default 3
  --pixel-search      hold the fitter's search in DEVICE pixels instead of
                      scene units. Off by default: the fitter's centre step,
                      seed sigma and log-scale clamps are all pixel constants,
                      so leaving them alone makes the refitted arm a different
                      searcher at every rung.
  --no-search-probe   skip the side probe that sizes exactly that confound
  --no-strips         skip the PNGs
  --sweep <MODE>      parallel (default) runs every whole fit of the run at once,
                      one fit per job; serial runs them one after another. The
                      tables and PNGs are bit-identical either way except the
                      render_ns column, which times the render alone.

Exit 0 clean; 2 no --out-dir or a bad flag; 3 the frozen and refitted arms
disagreed at the scale they were both fitted at, which means the ladder is not
deterministic and NO decay ratio on it is printed; 4 an error.";

const DEFAULT_CLASS_COUNT: usize = 4;

const PAD: &str = "                      ";

fn usage() -> String {
    let names = class_names();
    let default: Vec<&str> = names.iter().take(DEFAULT_CLASS_COUNT).copied().collect();
    let count = names.len();
    let lines = [
        USAGE_HEAD.to_string(),
        format!("  --classes <LIST>    comma-separated names or 1..{count}, or `all`, or `every`."),
        format!("{PAD}The catalogue, in index order:"),
        format!("{PAD}  {}", names.join(", ")),
        format!(
            "{PAD}`all` is the DEFAULT {} and NOT the whole catalogue:",
            default.len()
        ),
        format!("{PAD}  {}", default.join(", ")),
        format!("{PAD}-- one crisp class the analytic arm wins outright, the corner,"),
        format!("{PAD}one class NO signed distance can express, and the 12px glyph."),
        format!("{PAD}`every` is the only spelling that runs all {count}."),
        USAGE_TAIL.to_string(),
    ];
    lines.join("\n")
}

const DECAY_REFERENCE_TOLERANCE: f64 = 0.05;

#[derive(Clone, Debug)]
struct Args {
    classes: Vec<usize>,
    directions: Vec<Direction>,
    profiles: Vec<ProfileChoice>,
    max_atoms: usize,
    terms: usize,
    tile: u32,
    depth: u32,
    gap: u32,
    scene_invariant: bool,
    search_probe: bool,
    strips: bool,
    concurrency: Concurrency,
    out_dir: PathBuf,
}

fn catalogue() -> Vec<ScaleClass> {
    vec![
        ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
        ScaleClass::Analytic(Target::<f64>::t3_right_angle_corner()),
        ScaleClass::colour_field(),
        ScaleClass::Analytic(Target::<f64>::t6_glyph_like()),
        ScaleClass::Analytic(Target::<f64>::t5_thin_stroke()),
        ScaleClass::band_limited_noise(),
        ScaleClass::soft_shadow(),
    ]
}

fn class_names() -> Vec<&'static str> {
    catalogue().iter().map(|c| c.name()).collect()
}

fn parse_classes(text: &str) -> std::result::Result<Vec<usize>, String> {
    let names = class_names();
    if text == "all" {
        return Ok((0..DEFAULT_CLASS_COUNT).collect());
    }
    if text == "every" {
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
                Err(e) => {
                    return Err(format!(
                        "class {token}: {e}; known classes are {}",
                        names.join(", ")
                    ));
                }
            },
        }
    }
    Ok(out)
}

fn parse_direction(text: &str) -> std::result::Result<Vec<Direction>, String> {
    match text.trim() {
        "magnify" => Ok(vec![Direction::Magnify]),
        "minify" => Ok(vec![Direction::Minify]),
        "both" => Ok(Direction::ALL.to_vec()),
        other => Err(format!("unknown direction {other}")),
    }
}

fn parse_profile(text: &str) -> std::result::Result<Vec<ProfileChoice>, String> {
    match text.trim() {
        "contract" => Ok(vec![ProfileChoice::Contract]),
        "prefiltered" => Ok(vec![ProfileChoice::Prefiltered]),
        "both" => Ok(vec![ProfileChoice::Contract, ProfileChoice::Prefiltered]),
        other => Err(format!("unknown profile {other}")),
    }
}

fn parse() -> std::result::Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args {
        classes: (0..DEFAULT_CLASS_COUNT).collect(),
        directions: Direction::ALL.to_vec(),
        profiles: vec![ProfileChoice::Contract, ProfileChoice::Prefiltered],
        max_atoms: 8,
        terms: lights::fit::aspect3::DEFAULT_WAVELET_TERMS,
        tile: 16,
        depth: 10,
        gap: DEFAULT_GAP,
        scene_invariant: true,
        search_probe: true,
        strips: true,
        concurrency: Concurrency::Parallel,
        out_dir: PathBuf::new(),
    };
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
            "--help" => return Err(usage()),
            "-h" => return Err(usage()),
            "--out-dir" => {
                out_dir = Some(PathBuf::from(need(value)?));
                skip = true;
            }
            "--classes" => {
                args.classes = parse_classes(&need(value)?)?;
                skip = true;
            }
            "--direction" => {
                args.directions = parse_direction(&need(value)?)?;
                skip = true;
            }
            "--profile" => {
                args.profiles = parse_profile(&need(value)?)?;
                skip = true;
            }
            "--max-atoms" => {
                args.max_atoms = need(value)?
                    .parse::<usize>()
                    .map_err(|e| format!("max-atoms: {e}"))?;
                skip = true;
            }
            "--terms" => {
                args.terms = need(value)?
                    .parse::<usize>()
                    .map_err(|e| format!("terms: {e}"))?;
                skip = true;
            }
            "--tile" => {
                args.tile = need(value)?.parse::<u32>().map_err(|e| format!("tile: {e}"))?;
                skip = true;
            }
            "--depth" => {
                args.depth = need(value)?.parse::<u32>().map_err(|e| format!("depth: {e}"))?;
                skip = true;
            }
            "--gap" => {
                args.gap = need(value)?.parse::<u32>().map_err(|e| format!("gap: {e}"))?;
                skip = true;
            }
            "--pixel-search" => args.scene_invariant = false,
            "--no-search-probe" => args.search_probe = false,
            "--no-strips" => args.strips = false,
            "--sweep" => {
                args.concurrency = Concurrency::parse(&need(value)?)?;
                skip = true;
            }
            other => return Err(format!("unknown flag {other}\n\n{}", usage())),
        }
    }

    if args.classes.is_empty() {
        return Err("--classes selected nothing to measure".to_string());
    }
    if args.profiles.is_empty() {
        return Err("--profile selected nothing".to_string());
    }
    match out_dir {
        Some(dir) => {
            args.out_dir = dir;
            Ok(args)
        }
        None => Err(format!(
            "--out-dir is REQUIRED so this binary cannot write to the repository \
             root by accident\n\n{}",
            usage()
        )),
    }
}

fn request_for(args: &Args, direction: Direction) -> Result<Aspect3Request> {
    let mut request = Aspect3Request::new(direction)?;
    request.profiles = args.profiles.clone();
    request.max_atoms = args.max_atoms;
    request.wavelet_terms = args.terms;
    request.tile = args.tile;
    request.quadrature_depth = args.depth;
    request.scene_invariant_search = args.scene_invariant;
    request.concurrency = args.concurrency;
    Ok(request)
}

fn absolute(path: &Path) -> String {
    match std::fs::canonicalize(path) {
        Ok(p) => p.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn banner(args: &Args) {
    println!(
        "=== THE CRITERION CONVENTION, RULED BEFORE ANY NUMBER BELOW ===\n\
         PRIMARY: SCENE UNITS. {}\n\
         SECONDARY: DEVICE PIXELS. {}\n\
         The two columns are ONE multiplication apart: device = scene * s. The \
         ruling is scene-primary because this experiment holds the DESCRIPTION \
         fixed and varies only the sampling rate; the scene does not change when \
         the camera zooms, so a scene criterion asks one question at every rung \
         and a difference along the ladder is attributable to the description. \
         A pixel criterion re-asks a different, four-times-stricter question at \
         4x, and a decay curve measured that way cannot be separated from the \
         goalposts moving. The pixel column is kept because the registered \
         thresholds are DISPLAY criteria and `does it look sharp on the monitor` \
         is a genuine pixel question. The analytic bar is the proof the \
         convention matters: in pixels it is FLAT across the whole ladder by \
         construction, in scene units it is 1/s. Both are the same measurement.\n\
         \n\
         === THE FOUR ARMS ARE NOT SYMMETRIC ===\n\
         FROZEN ATOMS ({}) is the arm on trial. REFITTED ATOMS ({}) is the atom \
         quality CEILING and the honest control for it; the gap between them IS \
         the cost of resolution independence. ANALYTIC SDF ({}) cannot lose: it \
         is a one-device-pixel ramp at every scale by construction, so it is the \
         BAR and not a competitor, and on content no signed distance describes \
         it is drawn as a magenta hatch marked ABSENT rather than replaced by a \
         weak invented arm. THE WAVELET ARMS ({}) are lattice-bound by \
         definition: a coefficient set lives on a fixed dyadic lattice and does \
         NOT resample for free. A loss there is STRUCTURAL, not a defect of Haar \
         or CDF 9/7 — it is the difference between a continuous description and \
         a sampled one, which is the thing under test. To keep that loss a floor \
         rather than a strawman each wavelet arm is handed the textbook-correct \
         resampler in its direction: {} and {}.\n\
         \n\
         === EVERY WAY A NUMBER BELOW COULD BE AN ARTEFACT ===\n\
         1. CRITERION SCALING. Ruled above. Both columns printed, neither alone.\n\
         2. THE MISSING PREFILTER UNDER MINIFICATION. SharedSettings::contract() \
            ships screen_dilation = 0, meaning NO pixel prefilter at all, so a \
            minification loss under contract() would be aliasing and not the \
            representation. Minification is therefore run under BOTH contract() \
            and prefiltered(), and the two are printed side by side. A loss that \
            is present under contract() and gone under prefiltered() is the \
            prefilter. Only the atom arms honour the profile; the analytic and \
            wavelet arms carry their own and are marked profile-free.\n\
         3. THE FITTER FLOOR DIFFERS BY SCALE. Four fitter constants are \
            dimensioned in PIXELS: the centre step, the seed sigma and both \
            log-scale clamps. The search convention here is {}. What NO setting \
            can equalise is the seed lattice: seeds are chosen at pixel centres \
            from the residual peak, so a refit at 8x picks among 64 times more \
            distinct positions than a refit at 1x. That makes the refitted \
            CEILING genuinely stronger at large scales and WIDENS the frozen-to- \
            refitted gap at the far end of the magnification ladder for a reason \
            about the search, not about resolution independence.\n\
         4. THE BASE SCALE PRIVILEGES THREE OF THE FOUR ARMS. At the base rung \
            the frozen arm IS the refitted arm, the wavelet lattice IS the truth \
            lattice, and only the analytic arm is doing anything different, so \
            the base row is the minimum of every decay curve by construction. \
            That is why BOTH ladders are run from BOTH ends, and why the \
            headline is each arm against ITS OWN base. The two ladders are not \
            one experiment run backwards: 1x is the pixel-scale regime where \
            aspect 1 found the kernel ranking REVERSES, and 8x is the regime \
            where every feature is eight pixels wide.\n\
         5. THE TRUTH ITSELF. The truth at each rung is the pixel-box integral \
            of ONE fixed continuous field, sampled on a scene-invariant grid, so \
            no rung gets a better-resolved truth than another. But the truth at \
            1x is the box INTEGRAL and aspect 2 used the POINT SAMPLE, so \
            absolute relative-L2 values are NOT comparable between the two runs. \
            Ratios along a ladder are.\n\
         6. A FLAT DECAY FROM A BAD BASE IS NOT RESOLUTION INDEPENDENCE. decay \
            is each arm against ITS OWN base row, so an arm that is equally bad \
            at every rung scores 1.00 exactly like an arm that carries its \
            quality perfectly. Read the BASE ROW's relL2 first; the decay column \
            is only meaningful once you know what it is a decay from.\n\
         7. THE DECAY COLUMN HAS A DENOMINATOR AND THE DENOMINATOR MOVES. \
            relative L2 divides by the norm of the truth AT THAT RUNG, so where \
            an arm's error lives decides how its decay reads even when the error \
            itself never changes. An error spread over the whole field decays at \
            1.000, the FULL-FIELD reference. An error confined to a boundary of \
            fixed DEVICE-pixel width covers a shrinking share of a growing image, \
            so it decays at sqrt(base_side/side), the BOUNDARY reference printed \
            in the `bdry ref` column beside decay. The analytic bar is the proof \
            this is real and not pedantry: its decay sits ON the boundary \
            reference to within two percent at every rung of both ladders and \
            its max|e| never moves at all, so ALL of its apparent improvement under \
            magnification and ALL of its apparent collapse under minification are \
            the denominator and none of it is the description. A decay is only a \
            statement about a description once it is read against whichever of \
            the two references its arm belongs to.\n\
         8. CARRIED CAVEATS. C1: greedy matching pursuit is a FLOOR, so every \
            atom row is a lower bound on atom quality; a ratio along a ladder \
            divides much of that out because the same fitter runs at every rung. \
            C2: the byte comparison is NOT established; this run bills every arm \
            in one symmetric f32 currency so the columns are at least in the \
            same money, and still claims no byte ranking.",
        Convention::SceneUnits.legend(),
        Convention::DevicePixels.legend(),
        Independence::ContinuousFrozen.name(),
        Independence::ContinuousRefit.name(),
        Independence::ByConstruction.name(),
        Independence::LatticeBound.name(),
        Resample::Bilinear.name(),
        Resample::PixelBoxArea.name(),
        if args.scene_invariant {
            "SCENE-INVARIANT: those four constants move with the scale, so the \
             refitted arm searches the same scene at every rung"
        } else {
            "DEVICE-PIXEL: those four constants are held fixed, so the refitted \
             arm searches a finer scene grid and is capped by a smaller scene \
             atom at large scales. This is the confound, switched ON deliberately"
        }
    );
}

fn print_ladder(ladder: &ClassLadder) {
    println!(
        "\n=== {} {} : base {} at {}x{}, analytic bar {} ===",
        ladder.class,
        ladder.direction,
        ladder.base_scale.label(),
        ladder.base_viewport.0,
        ladder.base_viewport.1,
        if ladder.analytic_present { "present" } else { "ABSENT" }
    );
    println!(
        "{:<28} {:<12} {:>6} {:>9} {:>8} {:>7} {:>8} {:>8} {:>9} {:>8} {:>4} {:>4} {:>9} {:>8} {:>4} {:>4}",
        "arm",
        "profile",
        "scale",
        "viewport",
        "relL2",
        "decay",
        "bdry ref",
        "max|e|",
        "10-90 scn",
        "10-90 px",
        "scn",
        "px",
        "fwhm scn",
        "fwhm px",
        "scn",
        "px"
    );
    for row in ladder.rows.iter() {
        println!(
            "{:<28} {:<12} {:>6} {:>9} {:>8} {:>7} {:>8} {:>8} {:>9} {:>8} {:>4} {:>4} {:>9} {:>8} {:>4} {:>4}",
            row.arm,
            row.profile,
            row.scale.label(),
            format!("{}x{}", row.viewport.0, row.viewport.1),
            if row.absent { "ABSENT".to_string() } else { fixed(row.relative_l2, 5) },
            if row.absent { "-".to_string() } else { fixed(row.decay, 4) },
            if row.absent { "-".to_string() } else { fixed(row.boundary_reference, 4) },
            if row.absent { "-".to_string() } else { fixed(row.max_abs, 5) },
            fixed(row.transition.scene, 4),
            fixed(row.transition.device, 3),
            met(row.quality_measured, row.transition.scene, row.transition.meets_scene),
            met(row.quality_measured, row.transition.device, row.transition.meets_device),
            fixed(row.fwhm.scene, 4),
            fixed(row.fwhm.device, 3),
            met(row.quality_measured, row.fwhm.scene, row.fwhm.meets_scene),
            met(row.quality_measured, row.fwhm.device, row.fwhm.meets_device)
        );
    }
    println!(
        "  A dash is a probe this target does not carry: the edge classes are judged on the \
         10-90 transition and the stroke and glyph classes on the FWHM, and neither is \
         invented for a target that has no probe of that kind."
    );
}

fn fixed(v: f64, places: usize) -> String {
    if v.is_finite() {
        return format!("{v:.places$}");
    }
    "-".to_string()
}

fn met(measured: bool, value: f64, verdict: bool) -> &'static str {
    if !measured || !value.is_finite() {
        return "-";
    }
    if verdict { "yes" } else { "no" }
}

fn display_factor(viewport: (u32, u32), display: (u32, u32)) -> u32 {
    if viewport.0 == 0 {
        return 1;
    }
    (display.0 / viewport.0).max(1)
}

fn display_size(ladder: &ClassLadder) -> (u32, u32) {
    let mut best = (1u32, 1u32);
    for row in ladder.rows.iter() {
        if row.viewport.0 > best.0 {
            best = row.viewport;
        }
    }
    best
}

fn panel(image: &FloatImage<f64>, display: (u32, u32)) -> Result<FloatImage<f64>> {
    upscale(image, display_factor((image.width, image.height), display))
}

fn frozen_ladder_strip(
    args: &Args,
    ladder: &ClassLadder,
    kernel: KernelFamily,
    profile: &str,
) -> Result<Option<(PathBuf, String)>> {
    let display = display_size(ladder);
    let name = Arm3::Frozen(kernel).name();
    let mut panels = Vec::new();
    let mut labels = Vec::new();
    for row in ladder.rows.iter() {
        if row.arm != name || row.profile != profile {
            continue;
        }
        match ladder.render_of(&name, row.scale, profile) {
            Some(image) => {
                panels.push(panel(image, display)?);
                labels.push(format!(
                    "{} relL2={:.4} decay={:.2}",
                    row.scale.label(),
                    row.relative_l2,
                    row.decay
                ));
            }
            None => (),
        }
    }
    if panels.is_empty() {
        return Ok(None);
    }
    let strip = horizontal_strip(&panels, args.gap, separator())?;
    let path = args.out_dir.join(format!(
        "ladder-{}-{}-{profile}-frozen.png",
        ladder.class, ladder.direction
    ));
    write_png(&path, &strip.to_clamped_u8())?;
    Ok(Some((
        path,
        format!(
            "{} {} under {profile}: ONE frozen atom set, fitted at {} and re-sampled across \
             the whole ladder, left to right {}. Every panel is nearest-neighbour upscaled to \
             {}x{} so no value is invented and a 16x16 render stays visibly 16x16; that makes \
             the small rungs look worse to an eye used to smoothing, which is the honest \
             depiction and not a rendering fault. relL2 is against the pixel-box truth at that \
             rung; decay is against this arm's OWN base row, which is 1.00 by construction.",
            ladder.class,
            ladder.direction,
            ladder.base_scale.label(),
            labels.join(" | "),
            display.0,
            display.1
        ),
    )))
}

fn extremes_strip(
    args: &Args,
    ladder: &ClassLadder,
    kernel: KernelFamily,
    profile: &str,
    scale: Scale,
) -> Result<Option<(PathBuf, String)>> {
    let display = display_size(ladder);
    let truth = match ladder.truth_of(scale) {
        Some(t) => t,
        None => return Ok(None),
    };
    let mut panels = vec![panel(truth, display)?];
    let mut labels = vec!["truth".to_string()];
    for arm in arms(kernel) {
        let name = arm.name();
        let wanted = if arm.honours_profile() { profile } else { "profile-free" };
        let row = ladder
            .rows
            .iter()
            .find(|r| r.arm == name && r.scale == scale && r.profile == wanted);
        match ladder.render_of(&name, scale, wanted) {
            Some(image) => {
                panels.push(panel(image, display)?);
                labels.push(match row {
                    Some(r) => format!("{name} relL2={:.4} decay={:.2}", r.relative_l2, r.decay),
                    None => name.clone(),
                });
            }
            None => {
                let viewport = match ladder.rows.iter().find(|r| r.scale == scale) {
                    Some(r) => r.viewport,
                    None => (display.0, display.1),
                };
                panels.push(panel(&absent_panel(viewport.0, viewport.1), display)?);
                labels.push(format!("{name} ABSENT"));
            }
        }
    }
    let strip = horizontal_strip(&panels, args.gap, separator())?;
    let path = args.out_dir.join(format!(
        "extremes-{}-{}-{profile}-{}.png",
        ladder.class,
        ladder.direction,
        scale.slug()
    ));
    write_png(&path, &strip.to_clamped_u8())?;
    Ok(Some((
        path,
        format!(
            "{} {} at {} under {profile}, left to right: {}. Purple bars are separators, never \
             image content; a magenta hatch is an arm declared ABSENT and carries no image. \
             The columns are NOT four contenders: the analytic column is the BAR and cannot \
             lose, the two wavelet columns are lattice-bound so their loss here is structural, \
             and the frozen and refitted atom columns are the pair whose GAP is the measurement.\
             {}",
            ladder.class,
            ladder.direction,
            scale.label(),
            labels.join(" | "),
            if scale == ladder.base_scale {
                " This is the BASE rung, where the frozen and refitted columns are the same fit \
                 and must be the same picture bit for bit."
            } else {
                " This is the FAR rung, the whole point of the ladder."
            }
        ),
    )))
}

fn search_probe(args: &Args, direction: Direction, class: &ScaleClass) -> Result<String> {
    let mut request = request_for(args, direction)?;
    let far = match request.ladder.last() {
        Some(s) => *s,
        None => {
            return Err(LightsError::InvalidCamera("an empty ladder has no far rung"));
        }
    };
    let settings = match args.profiles.first() {
        Some(p) => p.settings(),
        None => return Err(LightsError::InvalidCamera("no profile to probe under")),
    };
    let mut cells = lights::fit::coverage::coverage_budget(request.cell_limit);
    let truth = class.truth(far, request.quadrature_depth, &mut cells)?;
    let viewport = class.viewport_at(far)?;
    let mut out = Vec::new();
    for scene_invariant in [true, false] {
        request.scene_invariant_search = scene_invariant;
        let fitted = fit_at(class, &truth, far, settings, &request)?;
        let image =
            render_atoms_at(&fitted.atoms, viewport, request.kernel, &settings, request.tile)?;
        out.push((request.search_convention(), relative_l2(&truth, &image)?));
    }
    let scene = out.first().map(|(_, e)| *e).unwrap_or(f64::NAN);
    let device = out.last().map(|(_, e)| *e).unwrap_or(f64::NAN);
    Ok(format!(
        "  {} {} at {}: refit relL2 {:.5} with a scene-invariant search against {:.5} with the \
         pixel search, a difference of {:.5}. That difference is the size of the fitter-floor artefact in the \
         refitted CEILING at the far rung, measured rather than argued away.",
        class.name(),
        direction.name(),
        far.label(),
        scene,
        device,
        (scene - device).abs()
    ))
}

fn execute(args: &Args) -> Result<ExitCode> {
    std::fs::create_dir_all(&args.out_dir)?;
    banner(args);

    let all = catalogue();
    let mut selected = Vec::new();
    for index in args.classes.iter() {
        match all.get(*index) {
            Some(c) => selected.push(c.clone()),
            None => {
                return Err(LightsError::InvalidCamera(
                    "lights-p5 was given a content class index that does not exist",
                ));
            }
        }
    }

    let mut manifest: Vec<(PathBuf, String)> = Vec::new();
    let mut checkpoints: Vec<(PathBuf, String)> = Vec::new();
    let mut failures = 0usize;
    let mut checks = 0usize;
    let mut probes: Vec<String> = Vec::new();

    for direction in args.directions.iter() {
        let request = request_for(args, *direction)?;
        let report = lights::fit::aspect3::run(&selected, &request)?;
        for line in header_lines(&report) {
            let mut parts = line.splitn(2, '\t');
            let name = parts.next().unwrap_or("");
            let value = parts.next().unwrap_or("");
            println!("{name:<34} {value}");
        }
        for ladder in report.ladders.iter() {
            print_ladder(ladder);
        }
        let table_path = args.out_dir.join(format!("ladder-{}.tsv", direction.name()));
        write_table(&table_path, &report)?;
        for ladder in report.ladders.iter() {
            for (profile, atoms) in ladder.frozen.iter() {
                let path = args.out_dir.join(format!(
                    "frozen-{}-{}-{profile}.fit",
                    ladder.class,
                    ladder.direction
                ));
                atoms_io::write_atoms(&path, atoms, ladder.base_viewport)?;
                checkpoints.push((
                    path,
                    format!(
                        "{} {} under {profile}: the {} frozen atoms, as fitted at {}x{}. \
                         This is the checkpoint the frozen strips are drawn from; read it \
                         with fit::atoms_io::read_atoms and re-draw without refitting.",
                        ladder.class,
                        ladder.direction,
                        atoms.len(),
                        ladder.base_viewport.0,
                        ladder.base_viewport.1
                    ),
                ));
            }
        }
        invocation::stamp(std::slice::from_ref(&table_path))?;
        let breached = report.ceiling_violations();
        if !breached.is_empty() {
            println!(
                "\n=== THE REFITTED CEILING IS NOT A CEILING ON {} CELL(S) ===\n\
                 Caveat C1 says greedy matching pursuit is a FLOOR, not an optimum. Where the \
                 FROZEN atom set beats a FRESH fit at the same rung, the refitted arm is not \
                 the atom quality ceiling it is meant to be: it is one greedy trajectory that \
                 landed worse than the one carried up from the base scale. The frozen-to- \
                 refitted gap at those rungs is a MINIMUM statement about the cost of \
                 resolution independence and possibly zero, and it must not be read as a \
                 measurement of the representation:",
                breached.len()
            );
            for (class, profile, scale, frozen, refit) in breached.iter() {
                println!(
                    "  {class} {profile} at {}: frozen {frozen:.5} beats refitted {refit:.5} by \
                     {:.5}",
                    scale.label(),
                    refit - frozen
                );
            }
        }
        let metric_only = report.decay_that_is_only_the_metric(DECAY_REFERENCE_TOLERANCE);
        if !metric_only.is_empty() {
            println!(
                "\n=== THE DECAY COLUMN HAS A DENOMINATOR (ARTEFACT 7 ABOVE), SIZED RATHER \
                 THAN ASSERTED: {} row(s) on the {} ladder ===\n\
                 On each row below the decay is within {:.0}% of sqrt(base_side/side), the \
                 reference for an error confined to a boundary of fixed DEVICE-pixel width. \
                 Such a decay is the shrinking SHARE that a fixed error takes of a growing \
                 image, not a change in the description. Read these rows against max|e|, \
                 which does not divide by the truth, before reading them as quality:\n\
                 {:<28} {:<22} {:<12} {:>7} {:>9} {:>9}",
                metric_only.len(),
                direction.name(),
                DECAY_REFERENCE_TOLERANCE * 100.0,
                "arm",
                "class",
                "profile",
                "scale",
                "decay",
                "bdry ref"
            );
            for (class, arm, profile, scale, decay, reference) in metric_only.iter() {
                println!(
                    "{:<28} {:<22} {:<12} {:>7} {:>9.4} {:>9.4}",
                    arm,
                    class,
                    profile,
                    scale.label(),
                    decay,
                    reference
                );
            }
        }
        let gaps = report.prefilter_gap();
        if !gaps.is_empty() {
            println!(
                "\n=== THE MISSING PREFILTER UNDER MINIFICATION (ARTEFACT 2 ABOVE, PROJECT \
                 CAVEAT C3), SIZED RATHER THAN ASSERTED: {} under both profiles \
                 ===\n\
                 contract() ships screen_dilation = 0, so it applies NO pixel prefilter. A row \
                 whose error falls when the prefilter is switched on was losing to ALIASING, \
                 not to the representation. Only the atom arms honour the profile.\n\
                 {:<28} {:<22} {:>7} {:>11} {:>13} {:>11}",
                direction.name(),
                "arm",
                "class",
                "scale",
                "contract",
                "prefiltered",
                "prefilter win"
            );
            for (class, arm, scale, contract, prefiltered) in gaps.iter() {
                println!(
                    "{:<28} {:<22} {:>7} {:>11.5} {:>13.5} {:>11.5}",
                    arm,
                    class,
                    scale.label(),
                    contract,
                    prefiltered,
                    contract - prefiltered
                );
            }
        }
        checks += report.base_agreement_checked();
        failures += report.base_agreement_failures().len();

        if args.strips {
            for ladder in report.ladders.iter() {
                for profile in args.profiles.iter() {
                    match frozen_ladder_strip(args, ladder, request.kernel, profile.name())? {
                        Some(entry) => manifest.push(entry),
                        None => (),
                    }
                    let ends = [request.base()?, match request.ladder.last() {
                        Some(s) => *s,
                        None => request.base()?,
                    }];
                    for scale in ends {
                        match extremes_strip(args, ladder, request.kernel, profile.name(), scale)? {
                            Some(entry) => manifest.push(entry),
                            None => (),
                        }
                    }
                }
            }
        }

        if args.search_probe {
            match selected.first() {
                Some(class) => probes.push(search_probe(args, *direction, class)?),
                None => (),
            }
        }
    }

    if !probes.is_empty() {
        println!(
            "\n=== THE FITTER FLOOR DIFFERS BY SCALE (ARTEFACT 3 ABOVE, PROJECT CAVEAT \
             C1), SIZED RATHER THAN ASSERTED ===\n\
             The fitter's centre step, seed sigma and both log-scale clamps are DEVICE PIXEL \
             constants. Held fixed, the refitted arm searches a finer scene grid at large \
             scales and is forbidden a scene atom larger than exp(MAX_LOG_SCALE)/s. Below, the \
             same class refitted at the far rung under both search conventions:"
        );
        for line in probes.iter() {
            println!("{line}");
        }
    }

    if args.strips {
        println!("\n=== IMAGE MANIFEST ===");
        for (path, caption) in manifest.iter() {
            println!("{}\n    {caption}", absolute(path));
        }
        println!("\n{} image(s) written.", manifest.len());
    }

    println!("\n=== FROZEN CHECKPOINT MANIFEST ===");
    for (path, caption) in checkpoints.iter() {
        println!("{}\n    {caption}", absolute(path));
    }
    println!("\n{} checkpoint(s) written.", checkpoints.len());

    println!(
        "\nTSV: {}",
        args.directions
            .iter()
            .map(|d| absolute(&args.out_dir.join(format!("ladder-{}.tsv", d.name()))))
            .collect::<Vec<String>>()
            .join(", ")
    );

    if failures > 0 {
        eprintln!(
            "\n{failures} OF {checks} BASE-SCALE AGREEMENT CHECKS FAILED. At the scale both \
             atom arms were fitted at they are the SAME fit on the SAME truth and must be the \
             same picture bit for bit. A difference means the ladder is not deterministic, so \
             every decay ratio on it is measuring noise and none of them should be read."
        );
        return Ok(ExitCode::from(3));
    }
    println!(
        "\nALL {checks} BASE-SCALE AGREEMENT CHECKS PASSED: the frozen and refitted arms are \
         the same picture at the rung they were both fitted at, which is what makes every \
         other rung's gap readable."
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
                "\nBUDGET EXHAUSTED at {site} (limit {limit}). A truncated fit would put a \
                 picture of the budget on the ladder instead of a picture of the \
                 representation, so the run fails instead."
            );
            ExitCode::from(4)
        }
        Err(e) => {
            eprintln!("\nERROR: {e}");
            ExitCode::from(4)
        }
    }
}
