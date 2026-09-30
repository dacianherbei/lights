#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::aspect2::{ClassPanels, TermChoice, class_panels, class_sweep};
use lights::fit::aspect3::{
    Aspect3Report, Aspect3Request, ClassLadder, Direction, class_ladder, header_lines,
};
use lights::fit::atom2d::{Atom2d, camera};
use lights::fit::content::Content;
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget, coverage_image_to_colour, exact_coverage};
use lights::fit::greedy::{FitRequest, Initialisation, fit, fit_budget, render_or_background};
use lights::fit::harness::{ProbeRequest, ProfileChoice, run};
use lights::fit::nterm::{NTermRequest, REACHABLE_ARMS};
use lights::fit::report::{Report, Row};
use lights::fit::scale::{Scale, ScaleClass};
use lights::fit::sweep::{Concurrency, map};
use lights::fit::target::Target;
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::settings::SharedSettings;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn atom_bits(a: &Atom2d) -> [u64; 9] {
    [
        a.centre.x.to_bits(),
        a.centre.y.to_bits(),
        a.log_scale.x.to_bits(),
        a.log_scale.y.to_bits(),
        a.angle.to_bits(),
        a.opacity.to_bits(),
        a.colour.x.to_bits(),
        a.colour.y.to_bits(),
        a.colour.z.to_bits(),
    ]
}

fn same_atoms(a: &[Atom2d], b: &[Atom2d]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(p, q)| atom_bits(p) == atom_bits(q))
}

fn same_image(a: &FloatImage<f64>, b: &FloatImage<f64>) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.pixels.iter().zip(b.pixels.iter()).all(|(p, q)| {
            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
                == (q.x.to_bits(), q.y.to_bits(), q.z.to_bits())
        })
}

fn truth_of(target: &Target<f64>) -> FloatImage<f64> {
    let mut cells = coverage_budget(DEFAULT_CELL_LIMIT);
    match exact_coverage(target, DEFAULT_MAX_DEPTH, &mut cells) {
        Ok((coverage, _)) => coverage_image_to_colour(target, &coverage),
        Err(e) => panic!("{} has no truth: {e}", target.name),
    }
}

fn fit_jobs() -> Vec<(usize, KernelFamily)> {
    vec![
        (0, KernelFamily::GaussianNormalised),
        (1, KernelFamily::GaussianNormalised),
        (0, KernelFamily::Gaussian),
        (1, KernelFamily::Gaussian),
    ]
}

fn fit_all(concurrency: Concurrency, threads: usize) -> Vec<Vec<Atom2d>> {
    let all = Target::<f64>::all();
    let truths: Vec<FloatImage<f64>> = all.iter().take(2).map(truth_of).collect();
    let jobs = fit_jobs();
    let outcome = map(concurrency, &jobs, |(index, blend)| {
        let mut settings = SharedSettings::<f64>::prefiltered();
        settings.threads = threads;
        let mut request = FitRequest::new(*blend, settings);
        request.max_atoms = 2;
        let mut budget = fit_budget(request.max_atoms as u64 * 1_000_000);
        fit(&all[*index], &truths[*index], &request, &mut budget).map(|o| o.atoms)
    });
    match outcome {
        Ok(atoms) => atoms,
        Err(e) => panic!("the fit sweep failed: {e}"),
    }
}

#[test]
fn a_parallel_sweep_fits_bit_identical_atoms_to_a_serial_one() {
    let serial = fit_all(Concurrency::Serial, 1);
    let parallel = fit_all(Concurrency::Parallel, 1);
    assert_eq!(serial.len(), fit_jobs().len());
    assert!(serial.iter().all(|atoms| !atoms.is_empty()), "a fit placed no atom, so the comparison is vacuous");
    for (job, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
        assert!(same_atoms(s, p), "job {job}: the parallel fit differs from the serial one");
    }
}

#[test]
fn the_fitter_ignores_the_callers_thread_count() {
    let one = fit_all(Concurrency::Serial, 1);
    for threads in [0usize, 4] {
        let many = fit_all(Concurrency::Serial, threads);
        for (job, (a, b)) in one.iter().zip(many.iter()).enumerate() {
            assert!(same_atoms(a, b), "job {job}: threads {threads} moved the fit");
        }
    }
}

#[test]
fn the_atom_comparator_fails_on_a_single_flipped_bit_of_a_single_atom() {
    let fitted = fit_all(Concurrency::Serial, 1);
    let atoms = &fitted[0];
    assert!(same_atoms(atoms, atoms));
    for field in 0..9 {
        let mut flipped = atoms.clone();
        let last = flipped.len() - 1;
        let a = &mut flipped[last];
        let slot = match field {
            0 => &mut a.centre.x,
            1 => &mut a.centre.y,
            2 => &mut a.log_scale.x,
            3 => &mut a.log_scale.y,
            4 => &mut a.angle,
            5 => &mut a.opacity,
            6 => &mut a.colour.x,
            7 => &mut a.colour.y,
            _ => &mut a.colour.z,
        };
        *slot = f64::from_bits(slot.to_bits() ^ 1);
        assert!(!same_atoms(atoms, &flipped), "field {field}: one flipped bit went unseen");
    }
    let shorter = &atoms[..atoms.len() - 1];
    assert!(!same_atoms(atoms, shorter), "a dropped atom went unseen");
}

#[test]
fn outcomes_come_back_in_job_order_not_completion_order() {
    let jobs: Vec<u64> = (0..8).collect();
    let out = match map(Concurrency::Parallel, &jobs, |job| {
        std::thread::sleep(Duration::from_millis(10 * (8 - job)));
        Ok(*job)
    }) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(out, jobs, "the slowest job finishes last and must still come back first");
}

#[test]
fn the_surviving_error_is_the_one_the_serial_run_would_raise() {
    let jobs: Vec<u64> = (0..6).collect();
    for concurrency in Concurrency::ALL {
        let outcome = map(concurrency, &jobs, |job| {
            std::thread::sleep(Duration::from_millis(10 * (6 - job)));
            match job {
                1 => Err(LightsError::InvalidCamera("job one")),
                4 => Err(LightsError::InvalidCamera("job four")),
                other => Ok(*other),
            }
        });
        match outcome {
            Err(LightsError::InvalidCamera(which)) => {
                assert_eq!(which, "job one", "{}: the later job's error survived", concurrency.name())
            }
            Err(e) => panic!("unexpected error {e}"),
            Ok(_) => panic!("{}: two failing jobs returned Ok", concurrency.name()),
        }
    }
}

#[test]
fn a_tile_render_is_bit_exact_whatever_the_thread_count() {
    let fitted = fit_all(Concurrency::Serial, 1);
    let all = Target::<f64>::all();
    let spec = camera(all[0].viewport.0, all[0].viewport.1);
    let mut one = SharedSettings::<f64>::prefiltered();
    one.threads = 1;
    let reference = match render_or_background(&fitted[0], &spec, KernelFamily::GaussianNormalised, &one, 4) {
        Ok(i) => i,
        Err(e) => panic!("{e}"),
    };
    for threads in [0usize, 2, 8] {
        let mut many = one;
        many.threads = threads;
        let image = match render_or_background(&fitted[0], &spec, KernelFamily::GaussianNormalised, &many, 4) {
            Ok(i) => i,
            Err(e) => panic!("{e}"),
        };
        assert!(same_image(&reference, &image), "threads {threads} moved a pixel");
    }
}

fn small_p2(concurrency: Concurrency) -> Report {
    let mut request = ProbeRequest::all();
    request.targets = vec![0, 1, 2];
    request.blends = vec![KernelFamily::GaussianNormalised, KernelFamily::Gaussian];
    request.profiles = vec![ProfileChoice::Prefiltered];
    request.max_atoms = 2;
    request.repeats = 1;
    request.sensitivity = true;
    request.tiled_field = true;
    request.concurrency = concurrency;
    match run(&request) {
        Ok(r) => r,
        Err(e) => panic!("the P2 harness failed: {e}"),
    }
}

fn untimed(row: &Row) -> String {
    let mut r = row.clone();
    r.render_ns_mean = 0.0;
    r.render_ns_ci95 = 0.0;
    format!("{r:?}")
}

#[test]
fn a_parallel_p2_report_is_bit_identical_to_a_serial_one() {
    let serial = small_p2(Concurrency::Serial);
    let parallel = small_p2(Concurrency::Parallel);
    assert!(serial.rows.len() > 3, "too few rows to compare");
    let a: Vec<String> = serial.rows.iter().map(untimed).collect();
    let b: Vec<String> = parallel.rows.iter().map(untimed).collect();
    assert_eq!(a, b, "the rows differ, or come back in another order");
    assert_eq!(format!("{:?}", serial.sweep), format!("{:?}", parallel.sweep));
    assert_eq!(format!("{:?}", serial.sensitivity), format!("{:?}", parallel.sensitivity));
    assert_eq!(format!("{:?}", serial.probes), format!("{:?}", parallel.probes));
    assert_eq!(format!("{:?}", serial.bins), format!("{:?}", parallel.bins));
    assert_eq!(serial.header.undecided_area.to_bits(), parallel.header.undecided_area.to_bits());
    assert_eq!(serial.sheets.len(), parallel.sheets.len());
    for (s, p) in serial.sheets.iter().zip(parallel.sheets.iter()) {
        assert_eq!(s.name, p.name);
        assert!(same_image(&s.image, &p.image), "sheet {} differs", s.name);
    }
}

#[test]
fn every_timed_row_is_timed_by_the_serial_phase() {
    let report = small_p2(Concurrency::Parallel);
    for row in report.rows.iter() {
        assert!(
            row.render_ns_mean.is_finite() && row.render_ns_mean > 0.0,
            "{} {} {} {} carries no render time: {}",
            row.target,
            row.blend,
            row.arm,
            row.field,
            row.render_ns_mean
        );
    }
}

#[test]
fn the_undecided_area_is_the_fold_over_every_target_in_request_order() {
    let report = small_p2(Concurrency::Parallel);
    let all = Target::<f64>::all();
    let mut expected = 0.0f64;
    let mut nonzero = 0;
    for index in [0usize, 1, 2] {
        let mut cells = coverage_budget(DEFAULT_CELL_LIMIT);
        let (_, residual) = match exact_coverage(&all[index], DEFAULT_MAX_DEPTH, &mut cells) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        if residual.undecided_area != 0.0 {
            nonzero += 1;
        }
        expected += residual.undecided_area;
    }
    assert!(nonzero >= 2, "fewer than two targets leave undecided area, so a dropped term is invisible");
    assert_eq!(report.header.undecided_area.to_bits(), expected.to_bits());
}

fn classes(concurrency: Concurrency) -> Vec<ClassPanels> {
    let all = Content::all();
    let chosen = vec![all[0].clone(), all[4].clone()];
    let mut request = NTermRequest::new(REACHABLE_ARMS[0]);
    request.max_atoms = 2;
    match class_sweep(&chosen, &REACHABLE_ARMS, TermChoice::SharedCeiling, &request, concurrency) {
        Ok(c) => c,
        Err(e) => panic!("the aspect-2 sweep failed: {e}"),
    }
}

#[test]
fn a_parallel_aspect2_sweep_draws_bit_identical_strips() {
    let serial = classes(Concurrency::Serial);
    let parallel = classes(Concurrency::Parallel);
    assert_eq!(serial.len(), 2);
    assert_eq!(parallel.len(), 2);
    for (s, p) in serial.iter().zip(parallel.iter()) {
        assert_eq!(s.content, p.content, "classes came back in another order");
        assert_eq!(s.shared_ceiling, p.shared_ceiling);
        assert_eq!(s.column_labels(), p.column_labels());
        assert_eq!(s.panels.len(), REACHABLE_ARMS.len() + 2);
        for (a, b) in s.panels.iter().zip(p.panels.iter()) {
            assert!(same_image(&a.image, &b.image), "{} {} differs", s.content, a.label);
            assert_eq!(a.relative_l2.map(f64::to_bits), b.relative_l2.map(f64::to_bits));
        }
    }
}

fn widest_overlap(concurrency: Concurrency) -> usize {
    let running = AtomicUsize::new(0);
    let widest = AtomicUsize::new(0);
    let jobs: Vec<u64> = (0..8).collect();
    let outcome = map(concurrency, &jobs, |job| {
        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
        widest.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(20));
        running.fetch_sub(1, Ordering::SeqCst);
        Ok(*job)
    });
    match outcome {
        Ok(v) => assert_eq!(v, jobs),
        Err(e) => panic!("{e}"),
    }
    widest.load(Ordering::SeqCst)
}

#[test]
fn a_serial_sweep_runs_one_job_at_a_time_and_a_parallel_one_overlaps_them() {
    assert_eq!(widest_overlap(Concurrency::Serial), 1, "the serial arm ran jobs side by side");
    if rayon::current_num_threads() > 1 {
        assert!(widest_overlap(Concurrency::Parallel) > 1, "the parallel arm never overlapped two jobs");
    }
}

#[test]
fn each_class_of_a_sweep_is_the_class_drawn_alone() {
    let swept = classes(Concurrency::Parallel);
    let all = Content::all();
    let mut request = NTermRequest::new(REACHABLE_ARMS[0]);
    request.max_atoms = 2;
    for (index, class) in [0usize, 4].iter().zip(swept.iter()) {
        let alone = match class_panels(&all[*index], &REACHABLE_ARMS, TermChoice::SharedCeiling, &request) {
            Ok(c) => c,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(alone.content, class.content);
        assert_eq!(alone.shared_ceiling, class.shared_ceiling);
        for (a, b) in alone.panels.iter().zip(class.panels.iter()) {
            assert!(same_image(&a.image, &b.image), "{} {}: the swept panel is another class's", class.content, a.label);
        }
    }
}

#[test]
fn every_sweep_binary_refuses_an_unknown_sweep_mode() {
    let binaries = [
        env!("CARGO_BIN_EXE_lights-p2"),
        env!("CARGO_BIN_EXE_lights-p3"),
        env!("CARGO_BIN_EXE_lights-p4"),
        env!("CARGO_BIN_EXE_lights-p5"),
    ];
    for binary in binaries {
        let output = match Command::new(binary).args(["--out-dir", "/nonexistent", "--sweep", "sideways"]).output() {
            Ok(o) => o,
            Err(e) => panic!("{binary}: {e}"),
        };
        let text = String::from_utf8_lossy(&output.stderr).to_string();
        assert_eq!(output.status.code(), Some(2), "{binary}: {text}");
        assert!(text.contains("serial or parallel"), "{binary}: {text}");
    }
}

fn aspect3_request(concurrency: Concurrency) -> Aspect3Request {
    let mut request = match Aspect3Request::new(Direction::Magnify) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    request.ladder = [(1u32, 1u32), (2, 1)]
        .iter()
        .map(|(n, d)| match Scale::new(*n, *d) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        })
        .collect();
    request.max_atoms = 2;
    request.refine_iterations = 1;
    request.refine_tail = 0;
    request.initialisation = Initialisation::Isotropic;
    request.wavelet_terms = 32;
    request.profiles = vec![ProfileChoice::Contract, ProfileChoice::Prefiltered];
    request.concurrency = concurrency;
    request
}

fn aspect3_classes() -> Vec<ScaleClass> {
    vec![
        ScaleClass::Analytic(Target::<f64>::t4_curve()),
        ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
    ]
}

fn aspect3_report(concurrency: Concurrency) -> Aspect3Report {
    match lights::fit::aspect3::run(&aspect3_classes(), &aspect3_request(concurrency)) {
        Ok(r) => r,
        Err(e) => panic!("the aspect-3 sweep failed: {e}"),
    }
}

fn untimed_ladder(ladder: &ClassLadder) -> Vec<String> {
    ladder
        .rows
        .iter()
        .map(|row| {
            let mut r = row.clone();
            r.render_nanos = 0;
            format!("{r:?}")
        })
        .collect()
}

fn same_ladder(a: &ClassLadder, b: &ClassLadder, context: &str) {
    assert_eq!(a.class, b.class, "{context}: classes came back in another order");
    assert_eq!(untimed_ladder(a), untimed_ladder(b), "{context}: the rows differ");
    assert_eq!(a.truths.len(), b.truths.len());
    for ((sa, ta), (sb, tb)) in a.truths.iter().zip(b.truths.iter()) {
        assert_eq!(sa, sb);
        assert!(same_image(ta, tb), "{context}: truth at {} differs", sa.label());
    }
    assert_eq!(a.renders.len(), b.renders.len());
    for ((arm, scale, profile, ia), (_, _, _, ib)) in a.renders.iter().zip(b.renders.iter()) {
        assert!(same_image(ia, ib), "{context}: {arm} {profile} at {} differs", scale.label());
    }
    assert_eq!(a.frozen.len(), b.frozen.len());
    for ((pa, fa), (pb, fb)) in a.frozen.iter().zip(b.frozen.iter()) {
        assert_eq!(pa, pb);
        let bits_a: Vec<[u64; 9]> = fa.iter().map(atom_bits).collect();
        let bits_b: Vec<[u64; 9]> = fb.iter().map(atom_bits).collect();
        assert_eq!(bits_a, bits_b, "{context}: the frozen atoms under {pa} differ");
    }
}

#[test]
fn a_parallel_aspect3_ladder_is_bit_identical_to_a_serial_one() {
    let serial = aspect3_report(Concurrency::Serial);
    let parallel = aspect3_report(Concurrency::Parallel);
    assert_eq!(serial.ladders.len(), 2);
    assert_eq!(parallel.ladders.len(), 2);
    for (s, p) in serial.ladders.iter().zip(parallel.ladders.iter()) {
        assert!(s.rows.len() > 8, "too few rows to compare");
        same_ladder(s, p, "serial against parallel");
    }
    assert_eq!(header_lines(&serial), header_lines(&parallel));
    assert!(parallel.base_agreement_checked() > 0);
    assert!(parallel.base_agreement_failures().is_empty());
}

#[test]
fn each_class_of_an_aspect3_sweep_is_the_class_run_alone() {
    let swept = aspect3_report(Concurrency::Parallel);
    let request = aspect3_request(Concurrency::Serial);
    for (class, ladder) in aspect3_classes().iter().zip(swept.ladders.iter()) {
        let alone = match class_ladder(class, &request) {
            Ok(l) => l,
            Err(e) => panic!("{e}"),
        };
        same_ladder(&alone, ladder, "swept against alone");
    }
}

#[test]
fn every_aspect3_fit_is_one_job_in_one_flat_list() {
    let request = aspect3_request(Concurrency::Parallel);
    let jobs = lights::fit::aspect3::fit_jobs(3, &request);
    let per_profile = 1 + request.ladder.len();
    assert_eq!(jobs.len(), 3 * request.profiles.len() * per_profile);
    let mut expected = Vec::new();
    for class in 0..3 {
        for profile in 0..request.profiles.len() {
            expected.push((class, profile, None));
            for rung in 0..request.ladder.len() {
                expected.push((class, profile, Some(rung)));
            }
        }
    }
    let got: Vec<(usize, usize, Option<usize>)> =
        jobs.iter().map(|j| (j.class, j.profile, j.rung)).collect();
    assert_eq!(got, expected);
}

#[test]
fn an_aspect3_render_ns_excludes_the_fit_and_says_so() {
    let report = aspect3_report(Concurrency::Parallel);
    let header = header_lines(&report).join("\n");
    assert!(header.contains("render_ns_convention\trender-only-fit-excluded"), "{header}");
    let mut refit_rows = 0usize;
    for row in report.rows() {
        assert!(row.absent || row.render_nanos > 0, "{} {} carries no render time", row.arm, row.scale.label());
        if row.arm.starts_with("refitted-atoms") {
            refit_rows += 1;
        }
    }
    assert!(refit_rows > 0, "no refitted row to check the convention against");
}
