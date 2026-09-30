#![cfg(feature = "method-splat")]
use lights::blend::{BlendMode, Over};
use lights::camera::{Camera, CameraSpec};
use lights::image::TileAccum;
use lights::kernel::frame::ScreenFrame;
use lights::kernel::gaussian::Gaussian;
use lights::method::splat::SortKind;
use lights::method::splat::bin::bin_and_sort;
use lights::method::splat::execute::{gather_batch, render_tile, render_tile_atom_outer};
use lights::method::splat::project::ScreenAtom;
use lights::numerics::symmetric::Sym2;
use lights::numerics::vector::{Rgb, Vec2};
use lights::probe::{Counter, CounterProbe, Counters, NoProbe, Probe};
use lights::settings::{PixelConvention, SharedSettings};
use lights::tiling::{TileGrid, TileRect};

const TILE: u32 = 16;
const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;

fn synthetic_atom(
    centre: Vec2<f64>,
    half: Vec2<f64>,
    depth: f64,
    index: u32,
    colour: Rgb<f64>,
    opacity: f64,
) -> ScreenAtom<f64, Gaussian> {
    let cov = Sym2::new(half.x * half.x / 9.0, 0.0, half.y * half.y / 9.0);
    let frame = match ScreenFrame::from_covariance(cov) {
        Some(f) => f,
        None => panic!("synthetic covariance is not SPD"),
    };
    ScreenAtom {
        centre,
        depth,
        frame,
        prepared: (),
        half_extent: half,
        support_sq: 9.0,
        opacity,
        colour,
        bound: 0.0,
        index,
    }
}

fn crowd() -> Vec<ScreenAtom<f64, Gaussian>> {
    let mut atoms = Vec::new();
    let anchors = [
        Vec2::new(8.0, 8.0),
        Vec2::new(23.0, 9.0),
        Vec2::new(9.0, 23.0),
        Vec2::new(24.0, 24.0),
    ];
    for i in 0..32u32 {
        let f = i as f64;
        let anchor = anchors[(i % 4) as usize];
        let ring = (i / 4) as f64;
        atoms.push(synthetic_atom(
            Vec2::new(
                anchor.x + (ring - 3.5) * 0.3,
                anchor.y + (3.5 - ring) * 0.25,
            ),
            Vec2::new(5.0 + (f % 3.0) * 0.4, 5.0 + (f % 4.0) * 0.3),
            1.0 + f * 0.31,
            i,
            Rgb::new(
                0.2 + 0.03 * (i % 9) as f64,
                0.9 - 0.02 * (i % 11) as f64,
                0.3 + 0.025 * (i % 7) as f64,
            ),
            0.95,
        ));
    }
    atoms
}

fn camera() -> Camera<f64> {
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        WIDTH,
        HEIGHT,
    )
    .orthographic(HEIGHT as f64 * 0.5)
    .with_convention(PixelConvention::CentreSample);
    Camera::<f64>::from_spec(&spec, SharedSettings::<f64>::reference().near_plane)
        .expect("camera")
}

struct Run {
    cells: Vec<<Over as BlendMode<f64>>::Accum>,
    counters: Vec<(Counter, u64)>,
}

fn run(atom_outer: bool) -> Run {
    let grid = TileGrid::new(WIDTH, HEIGHT, TILE).expect("grid");
    let screen = crowd();
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    let settings = SharedSettings::<f64>::reference();
    let camera = camera();
    let per_tile = (TILE as usize) * (TILE as usize);
    let tiles: Vec<(usize, TileRect)> = grid
        .tiles()
        .enumerate()
        .map(|(i, (tx, ty))| (i, grid.rect(tx, ty)))
        .collect();
    let mut cells: Vec<<Over as BlendMode<f64>>::Accum> =
        vec![<Over as BlendMode<f64>>::init(); per_tile * tiles.len()];
    let mut total = Counters::default();
    let mut batch = Vec::new();

    for (index, rect) in tiles {
        gather_batch(&screen, &keys, ranges[index], &mut batch);
        if batch.is_empty() {
            continue;
        }
        let slice = &mut cells[index * per_tile..(index + 1) * per_tile];
        let mut acc = TileAccum::<f64, Over> {
            rect,
            stride: TILE as i32,
            cells: slice,
        };
        let mut probe = <CounterProbe<f64> as Probe<f64>>::for_tile(rect);
        if atom_outer {
            render_tile_atom_outer::<f64, Gaussian, CounterProbe<f64>>(
                &batch, rect, &camera, &settings, &mut acc, &mut probe,
            );
        } else {
            render_tile::<f64, Gaussian, CounterProbe<f64>>(
                &batch, rect, &camera, &settings, &mut acc, &mut probe,
            );
        }
        total.merge(&probe.counters());
    }

    let counters = vec![
        (Counter::PixelsVisited, total.pixels_visited),
        (Counter::CandidatesTested, total.candidates_tested),
        (Counter::KernelEvaluations, total.kernel_evaluations),
        (Counter::CompositedHits, total.composited_hits),
        (Counter::EarlyOuts, total.early_outs),
    ];

    Run { cells, counters }
}

#[test]
fn the_atom_outer_tile_loop_is_bit_identical_to_the_shipped_pixel_outer_one() {
    let reference = run(false);
    let variant = run(true);

    assert_eq!(
        reference.cells.len(),
        variant.cells.len(),
        "the two loops must cover the same accumulator"
    );

    let mut touched = 0usize;
    for (i, (a, b)) in reference.cells.iter().zip(variant.cells.iter()).enumerate() {
        if a.transmittance != 1.0 {
            touched += 1;
        }
        assert_eq!(
            a.rgb.x.to_bits(),
            b.rgb.x.to_bits(),
            "cell {i} red differs: {} vs {}",
            a.rgb.x,
            b.rgb.x
        );
        assert_eq!(a.rgb.y.to_bits(), b.rgb.y.to_bits(), "cell {i} green differs");
        assert_eq!(a.rgb.z.to_bits(), b.rgb.z.to_bits(), "cell {i} blue differs");
        assert_eq!(
            a.transmittance.to_bits(),
            b.transmittance.to_bits(),
            "cell {i} transmittance differs: {} vs {}",
            a.transmittance,
            b.transmittance
        );
    }

    assert!(
        touched > 200,
        "only {touched} accumulator cells were written, so this comparison is \
         mostly comparing untouched background against untouched background and \
         would pass with either loop replaced by nothing"
    );

    let early_outs = reference.counters[4].1;
    assert!(
        early_outs > 0,
        "the blend never told a pixel to stop, so the two loops were never \
         compared on the ONE thing the inversion has to get right by hand: the \
         shipped loop RETURNS out of a pixel, and the variant has to retire it \
         with a live flag instead. A scene that never early-outs makes that \
         difference invisible, and a mutation removing the retirement ESCAPED \
         against exactly such a scene on 2026-09-16"
    );
    assert_eq!(
        early_outs, variant.counters[4].1,
        "the two loops must retire the same number of pixels"
    );
}

#[test]
fn the_atom_outer_loop_reports_the_same_counters() {
    let reference = run(false);
    let variant = run(true);
    for ((c, a), (_, b)) in reference.counters.iter().zip(variant.counters.iter()) {
        assert_eq!(
            a, b,
            "{c:?} differs between the pixel-outer and atom-outer loops: {a} vs {b}. \
             The early-out is the part that is easy to get wrong -- the shipped loop \
             RETURNS out of the pixel, and the variant has to carry a per-pixel live \
             flag or it keeps testing candidates the reference never tests"
        );
    }
    let tested = reference.counters[1].1;
    assert!(
        tested > 0,
        "no candidate was tested, so the counter comparison proves nothing"
    );
}

const BENCH_W: u32 = 480;
const BENCH_H: u32 = 270;
const BENCH_ATOMS: u32 = 600;
const REPS: usize = 7;

fn bench_crowd() -> Vec<ScreenAtom<f64, Gaussian>> {
    let mut atoms = Vec::new();
    for i in 0..BENCH_ATOMS {
        let f = i as f64;
        atoms.push(synthetic_atom(
            Vec2::new(
                (f * 7.31) % (BENCH_W as f64 - 8.0) + 4.0,
                (f * 11.17) % (BENCH_H as f64 - 8.0) + 4.0,
            ),
            Vec2::new(5.0 + (f % 7.0), 5.0 + (f % 5.0)),
            1.0 + f * 0.013,
            i,
            Rgb::new(0.6, 0.5, 0.4),
            0.05,
        ));
    }
    atoms
}

fn bench_camera() -> Camera<f64> {
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        BENCH_W,
        BENCH_H,
    )
    .orthographic(BENCH_H as f64 * 0.5)
    .with_convention(PixelConvention::CentreSample);
    Camera::<f64>::from_spec(&spec, SharedSettings::<f64>::reference().near_plane)
        .expect("camera")
}

fn timed(atom_outer: bool) -> Vec<f64> {
    let grid = TileGrid::new(BENCH_W, BENCH_H, TILE).expect("grid");
    let screen = bench_crowd();
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    let settings = SharedSettings::<f64>::reference();
    let camera = bench_camera();
    let per_tile = (TILE as usize) * (TILE as usize);
    let tiles: Vec<(usize, TileRect)> = grid
        .tiles()
        .enumerate()
        .map(|(i, (tx, ty))| (i, grid.rect(tx, ty)))
        .collect();
    let mut cells: Vec<<Over as BlendMode<f64>>::Accum> =
        vec![<Over as BlendMode<f64>>::init(); per_tile * tiles.len()];
    let mut batch = Vec::new();
    let mut out = Vec::with_capacity(REPS);

    for _ in 0..REPS + 1 {
        for c in cells.iter_mut() {
            *c = <Over as BlendMode<f64>>::init();
        }
        let start = std::time::Instant::now();
        for (index, rect) in &tiles {
            gather_batch(&screen, &keys, ranges[*index], &mut batch);
            if batch.is_empty() {
                continue;
            }
            let slice = &mut cells[index * per_tile..(index + 1) * per_tile];
            let mut acc = TileAccum::<f64, Over> {
                rect: *rect,
                stride: TILE as i32,
                cells: slice,
            };
            if atom_outer {
                render_tile_atom_outer::<f64, Gaussian, NoProbe>(
                    &batch, *rect, &camera, &settings, &mut acc, &mut NoProbe,
                );
            } else {
                render_tile::<f64, Gaussian, NoProbe>(
                    &batch, *rect, &camera, &settings, &mut acc, &mut NoProbe,
                );
            }
        }
        out.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&cells);
    }
    out.remove(0);
    out.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    out
}

#[test]
fn time_the_atom_outer_loop_against_the_shipped_one() {
    let reference = timed(false);
    let variant = timed(true);
    let pick = |v: &Vec<f64>| (v[0], v[v.len() / 2], v[v.len() - 1]);
    let (r_min, r_med, r_max) = pick(&reference);
    let (v_min, v_med, v_max) = pick(&variant);
    println!(
        "L-149 {BENCH_W}x{BENCH_H} {BENCH_ATOMS} atoms, {REPS} reps, first discarded"
    );
    println!("  pixel-outer (shipped) min {r_min:.3} median {r_med:.3} max {r_max:.3} ms, spread {:.2}x", r_max / r_min);
    println!("  atom-outer  (variant) min {v_min:.3} median {v_med:.3} max {v_max:.3} ms, spread {:.2}x", v_max / v_min);
    println!(
        "  variant / shipped: {:.3}x on min, {:.3}x on median. Above 1.0 means the \
         restructuring is SLOWER.",
        v_min / r_min,
        v_med / r_med
    );
    assert!(
        r_min > 0.0 && v_min > 0.0,
        "a zero time means the workload was optimised away and nothing was measured"
    );
}

#[test]
fn the_variant_is_reachable_with_the_null_probe_too() {
    let grid = TileGrid::new(WIDTH, HEIGHT, TILE).expect("grid");
    let screen = crowd();
    let mut keys = Vec::new();
    let mut ranges = Vec::new();
    bin_and_sort::<f64, Gaussian>(
        &screen,
        &grid,
        true,
        SortKind::GlobalSort,
        &mut keys,
        &mut ranges,
    );
    let settings = SharedSettings::<f64>::reference();
    let camera = camera();
    let per_tile = (TILE as usize) * (TILE as usize);
    let rect = grid.rect(0, 0);
    let mut batch = Vec::new();
    gather_batch(&screen, &keys, ranges[0], &mut batch);
    let mut cells = vec![<Over as BlendMode<f64>>::init(); per_tile];
    let mut acc = TileAccum::<f64, Over> {
        rect,
        stride: TILE as i32,
        cells: &mut cells,
    };
    render_tile_atom_outer::<f64, Gaussian, NoProbe>(
        &batch,
        rect,
        &camera,
        &settings,
        &mut acc,
        &mut NoProbe,
    );
    assert!(
        cells.iter().any(|c| c.transmittance != 1.0),
        "the first tile must have been written, or the timing variant is measuring \
         an empty loop"
    );
}
