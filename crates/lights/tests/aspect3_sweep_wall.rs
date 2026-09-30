#![cfg(feature = "fit")]
use lights::fit::aspect3::{Aspect3Request, Direction, run};
use lights::fit::greedy::Initialisation;
use lights::fit::harness::ProfileChoice;
use lights::fit::scale::{Scale, ScaleClass};
use lights::fit::sweep::Concurrency;
use lights::fit::target::Target;
use std::time::Instant;

const REPS: usize = 3;
const MIN_SPEEDUP: f64 = 1.5;

fn fastest(concurrency: Concurrency) -> f64 {
    let classes = vec![
        ScaleClass::Analytic(Target::<f64>::t4_curve()),
        ScaleClass::Analytic(Target::<f64>::t6_glyph_like()),
        ScaleClass::Analytic(Target::<f64>::t3_right_angle_corner()),
        ScaleClass::Analytic(Target::<f64>::t2_straight_edge()),
        ScaleClass::Analytic(Target::<f64>::t5_thin_stroke()),
    ];
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
    request.quadrature_depth = 8;
    request.max_atoms = 4;
    request.refine_iterations = 2;
    request.refine_tail = 1;
    request.initialisation = Initialisation::Isotropic;
    request.profiles = vec![ProfileChoice::Contract];
    request.concurrency = concurrency;
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let started = Instant::now();
        match run(&classes, &request) {
            Ok(report) => assert_eq!(report.ladders.len(), classes.len()),
            Err(e) => panic!("{e}"),
        }
        best = best.min(started.elapsed().as_secs_f64());
    }
    best
}

#[test]
fn an_aspect3_parallel_sweep_takes_less_wall_than_a_serial_one() {
    let threads = rayon::current_num_threads();
    if threads < 4 {
        return;
    }
    let serial = fastest(Concurrency::Serial);
    let parallel = fastest(Concurrency::Parallel);
    eprintln!("serial {serial:.3} s, parallel {parallel:.3} s, {:.2}x on {threads} threads", serial / parallel);
    assert!(
        parallel * MIN_SPEEDUP < serial,
        "on {threads} threads the parallel sweep took {parallel:.3} s against serial {serial:.3} s, \
         under {MIN_SPEEDUP}x: the request's concurrency is not reaching the fits"
    );
}
