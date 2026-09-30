#![cfg(feature = "fit")]
use lights::fit::greedy::residual_energy;
use lights::image::FloatImage;
use lights::numerics::vector::Rgb;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Watcher;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static LARGE: Cell<usize> = const { Cell::new(0) };
    static LARGE_BYTES: Cell<usize> = const { Cell::new(0) };
}

const LARGE_ALLOCATION: usize = 4096;

unsafe impl GlobalAlloc for Watcher {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.with(Cell::get) {
            CALLS.with(|c| c.set(c.get() + 1));
            BYTES.with(|b| b.set(b.get() + layout.size()));
            if layout.size() >= LARGE_ALLOCATION {
                LARGE.with(|c| c.set(c.get() + 1));
                LARGE_BYTES.with(|b| b.set(b.get() + layout.size()));
            }
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static WATCHER: Watcher = Watcher;

fn allocations_during<R>(body: impl FnOnce() -> R) -> (R, usize, usize) {
    CALLS.with(|c| c.set(0));
    BYTES.with(|b| b.set(0));
    LARGE.with(|c| c.set(0));
    LARGE_BYTES.with(|b| b.set(0));
    ARMED.with(|a| a.set(true));
    let out = body();
    ARMED.with(|a| a.set(false));
    (out, CALLS.with(Cell::get), BYTES.with(Cell::get))
}

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

fn pattern(width: u32, height: u32, phase: f64) -> FloatImage<f64> {
    let mut image = FloatImage::filled(width, height, Rgb::new(0.0, 0.0, 0.0));
    for (i, p) in image.pixels.iter_mut().enumerate() {
        let t = i as f64 * 0.013 + phase;
        *p = Rgb::new(
            t.sin() * 0.7,
            (t * 1.7).cos() * 0.4,
            (t * 0.3).sin() * 1.3 + 0.1,
        );
    }
    image
}

fn reference_energy(truth: &FloatImage<f64>, rendered: &FloatImage<f64>) -> f64 {
    let difference = truth.difference(rendered).expect("same size");
    let mut acc = 0.0f64;
    for p in difference.pixels.iter() {
        acc += p.x * p.x + p.y * p.y + p.z * p.z;
    }
    acc
}

#[test]
fn residual_energy_allocates_nothing() {
    let truth = pattern(WIDTH, HEIGHT, 0.0);
    let rendered = pattern(WIDTH, HEIGHT, 0.37);
    let (energy, calls, bytes) = allocations_during(|| residual_energy(&truth, &rendered));
    let energy = energy.expect("same size");
    assert!(
        energy > 0.0,
        "the two patterns differ, so the energy is positive: {energy}"
    );
    assert_eq!(
        (calls, bytes),
        (0, 0),
        "residual_energy made {calls} allocations totalling {bytes} bytes; a {WIDTH}x{HEIGHT} difference image is {} bytes",
        WIDTH as usize * HEIGHT as usize * std::mem::size_of::<Rgb<f64>>()
    );
}

#[test]
fn residual_energy_is_bit_identical_to_difference_then_sum() {
    for (phase, width, height) in [
        (0.37, WIDTH, HEIGHT),
        (2.1, 7, 3),
        (0.0, 1, 1),
        (5.5, 33, 17),
    ] {
        let truth = pattern(width, height, 0.0);
        let rendered = pattern(width, height, phase);
        let fused = residual_energy(&truth, &rendered).expect("same size");
        let reference = reference_energy(&truth, &rendered);
        assert_eq!(
            fused.to_bits(),
            reference.to_bits(),
            "{width}x{height} phase {phase}: fused {fused:e} against reference {reference:e}"
        );
    }
}

#[test]
fn residual_energy_still_refuses_a_size_mismatch() {
    let truth = pattern(WIDTH, HEIGHT, 0.0);
    for (width, height) in [(WIDTH, HEIGHT - 1), (WIDTH - 1, HEIGHT), (HEIGHT, WIDTH)] {
        let rendered = pattern(width, height, 0.0);
        assert!(
            residual_energy(&truth, &rendered).is_err(),
            "{width}x{height} against {WIDTH}x{HEIGHT} must be refused"
        );
    }
}

fn median_and_spread(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(|a, b| a.total_cmp(b));
    (
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1],
    )
}

fn nanos_per_call(reps: usize, calls: u32, mut body: impl FnMut()) -> (f64, f64, f64) {
    let samples = (0..reps)
        .map(|_| {
            let started = std::time::Instant::now();
            for _ in 0..calls {
                body();
            }
            started.elapsed().as_nanos() as f64 / calls as f64
        })
        .collect();
    median_and_spread(samples)
}

#[test]
fn candidate_cost_census() {
    use lights::fit::atom2d::{Atom2d, camera, pixel_centre};
    use lights::fit::greedy::render_or_background;
    use lights::kernel::KernelFamily;
    use lights::numerics::vector::Vec2;
    use lights::settings::SharedSettings;
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let atoms: Vec<Atom2d> = (0..8)
        .map(|i| {
            Atom2d::from_sigma(
                pixel_centre(6 + 7 * i, 5 + 5 * i),
                Vec2::new(2.0, 1.2),
                0.3 * i as f64,
                0.8,
                Rgb::new(0.9, 0.4, 0.2),
            )
        })
        .collect();
    let spec = camera(WIDTH, HEIGHT);
    let truth = pattern(WIDTH, HEIGHT, 0.0);
    let frame = WIDTH as usize * HEIGHT as usize * std::mem::size_of::<Rgb<f64>>();
    let rendered = render_or_background(&atoms, &spec, KernelFamily::Gaussian, &settings, 16)
        .expect("renders");
    let (_, render_calls, render_bytes) = allocations_during(|| {
        render_or_background(&atoms, &spec, KernelFamily::Gaussian, &settings, 16)
    });
    let (_, residual_calls, residual_bytes) =
        allocations_during(|| residual_energy(&truth, &rendered));
    let (_, reference_calls, reference_bytes) =
        allocations_during(|| reference_energy(&truth, &rendered));
    println!("viewport\t{WIDTH}x{HEIGHT}\tatoms\t8\tframe_bytes\t{frame}");
    println!("allocs\trender\t{render_calls}\t{render_bytes}");
    let _ = allocations_during(|| {
        render_or_background(&atoms, &spec, KernelFamily::Gaussian, &settings, 16)
    });
    let (large, large_bytes) = (LARGE.with(Cell::get), LARGE_BYTES.with(Cell::get));
    println!("allocs\trender_at_least_{LARGE_ALLOCATION}_bytes\t{large}\t{large_bytes}");
    println!("allocs\tresidual_fused\t{residual_calls}\t{residual_bytes}");
    println!("allocs\tresidual_reference\t{reference_calls}\t{reference_bytes}");
    let reps = 9;
    let rows: [(&str, u32, Box<dyn FnMut()>); 6] = [
        (
            "render",
            400,
            Box::new(|| {
                std::hint::black_box(
                    render_or_background(&atoms, &spec, KernelFamily::Gaussian, &settings, 16).ok(),
                );
            }),
        ),
        (
            "residual_fused",
            4000,
            Box::new(|| {
                std::hint::black_box(residual_energy(&truth, &rendered).ok());
            }),
        ),
        (
            "residual_reference",
            4000,
            Box::new(|| {
                std::hint::black_box(reference_energy(&truth, &rendered));
            }),
        ),
        (
            "frame_alloc_free",
            4000,
            Box::new(|| {
                std::hint::black_box(FloatImage::filled(WIDTH, HEIGHT, Rgb::new(0.0, 0.0, 0.0)));
            }),
        ),
        (
            "frame_alloc_only",
            4000,
            Box::new(|| {
                std::hint::black_box(Vec::<Rgb<f64>>::with_capacity(
                    WIDTH as usize * HEIGHT as usize,
                ));
            }),
        ),
        (
            "timer_pair",
            400_000,
            Box::new(|| {
                let started = std::time::Instant::now();
                std::hint::black_box(started.elapsed().as_nanos());
            }),
        ),
    ];
    for (name, calls, mut body) in rows {
        let (median, low, high) = nanos_per_call(reps, calls, &mut body);
        println!("ns\t{name}\tmedian\t{median:.1}\tlow\t{low:.1}\thigh\t{high:.1}\treps\t{reps}");
    }
}
