use lights::blend::{Additive, BlendMode, BlendSettings, Over};
#[cfg(feature = "method-splat")]
use lights::camera::CameraSpec;
#[cfg(feature = "method-splat")]
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::image::{FloatImage, Framebuffer, Image, read_png, write_png, write_png_normalized};
#[cfg(feature = "method-splat")]
use lights::kernel::KernelFamily;
#[cfg(feature = "method-splat")]
use lights::method::RenderMethod;
#[cfg(feature = "method-splat")]
use lights::method::splat::SplatConfig;
use lights::numerics::vector::Rgb;
#[cfg(feature = "method-splat")]
use lights::numerics::vector::Vec3;
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, SharedSettings};
use lights::tiling::TileGrid;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct FrameWatcher;

static FRAME_SIZE: AtomicUsize = AtomicUsize::new(0);
static FRAME_HITS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);
static FRAME_ARMED: AtomicBool = AtomicBool::new(false);
static FRAME_GATE: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for FrameWatcher {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FRAME_ARMED.load(Ordering::Relaxed) {
            TOTAL_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            if layout.size() == FRAME_SIZE.load(Ordering::Relaxed) {
                FRAME_HITS.fetch_add(1, Ordering::Relaxed);
            }
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static FRAMES: FrameWatcher = FrameWatcher;

fn frames_of_size(size: usize, body: impl FnOnce()) -> (usize, usize) {
    let gate = match FRAME_GATE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    FRAME_SIZE.store(size, Ordering::Relaxed);
    FRAME_HITS.store(0, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    FRAME_ARMED.store(true, Ordering::Relaxed);
    body();
    FRAME_ARMED.store(false, Ordering::Relaxed);
    let measured = (
        FRAME_HITS.load(Ordering::Relaxed),
        TOTAL_BYTES.load(Ordering::Relaxed),
    );
    drop(gate);
    measured
}

const WIDEN_WIDTH: u32 = 97;
const WIDEN_HEIGHT: u32 = 53;

#[cfg(feature = "method-splat")]
fn widen_request(precision: Precision, collect: Collect) -> RenderRequest {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let mut scene = Scene::<f64>::new();
    for (x, z) in [(-0.05f64, 4.0f64), (0.05, 5.0)] {
        scene
            .push(
                Atom::isotropic(Vec3::new(x, 0.0, z), 0.08, 0.7, Rgb::new(1.0, 0.5, 0.25)),
                MIN_SCALE,
            )
            .expect("valid atom");
    }
    RenderRequest {
        precision,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::SplatRaster(SplatConfig::contract()),
        settings,
        camera: CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            WIDEN_WIDTH,
            WIDEN_HEIGHT,
        ),
        scene,
        collect,
        shapes: None,
    }
}

#[test]
fn framebuffer_cells_round_trip_through_the_tile_major_layout() {
    let mut fb = Framebuffer::<f64, Over>::new(35, 20, 16).expect("framebuffer");
    assert_eq!(fb.width(), 35);
    assert_eq!(fb.height(), 20);
    let grid = fb.grid();

    let settings = BlendSettings::<f64>::contract();
    {
        let tiles = fb.split_tiles_mut(&grid).expect("matching grid");
        assert_eq!(tiles.len(), grid.tile_count());
        for tile in tiles {
            for row in 0..tile.rect.h {
                for col in 0..tile.rect.w {
                    let x = tile.rect.x0 + col;
                    let y = tile.rect.y0 + row;
                    let value = (x as f64) / 100.0 + (y as f64) / 10000.0;
                    let local = (row * tile.stride + col) as usize;
                    <Over as BlendMode<f64>>::step(
                        &mut tile.cells[local],
                        1.0,
                        Rgb::splat(value),
                        1.0,
                        &settings,
                    );
                }
            }
        }
    }

    for y in 0..20u32 {
        for x in 0..35u32 {
            let expected = ((x as f64) / 100.0 + (y as f64) / 10000.0) * 0.99;
            let cell = fb.cell(x, y);
            assert!(
                (cell.rgb.x - expected).abs() < 1e-15,
                "pixel ({x},{y}) mapped to the wrong tile-major slot"
            );
        }
    }

    let resolved = fb.resolve_float(Rgb::zero());
    assert_eq!(resolved.width, 35);
    assert!((resolved.get(34, 19).x - fb.cell(34, 19).rgb.x).abs() < 1e-15);
    let coverage = fb.coverage();
    assert!((coverage.get(3, 4).x - 0.99).abs() < 1e-15);

    let wrong = TileGrid::new(35, 20, 8).expect("grid");
    assert!(fb.split_tiles_mut(&wrong).is_err(), "a grid mismatch is an error");

    fb.clear();
    assert_eq!(fb.cell(0, 0).transmittance, 1.0);
}

#[test]
fn framebuffer_is_parameterised_on_the_blend_not_the_method() {
    let over = Framebuffer::<f64, Over>::new(4, 4, 4).expect("framebuffer");
    let additive = Framebuffer::<f64, Additive>::new(4, 4, 4).expect("framebuffer");
    assert_eq!(over.cell(0, 0).transmittance, 1.0);
    assert_eq!(additive.cell(0, 0), Rgb::zero());
    let resolved = additive.resolve_float(Rgb::new(0.25, 0.0, 0.0));
    assert_eq!(
        resolved.get(0, 0),
        Rgb::new(0.25, 0.0, 0.0),
        "an additive buffer has no transmittance to weight the background by"
    );
    assert!(Framebuffer::<f64, Over>::new(0, 4, 4).is_err());
}

#[test]
fn float_image_arithmetic() {
    let mut a = FloatImage::<f64>::filled(3, 2, Rgb::splat(1.0));
    let b = FloatImage::<f64>::filled(3, 2, Rgb::splat(0.25));
    a.set(2, 1, Rgb::new(1.5, -0.5, 0.0));

    let d = a.difference(&b).expect("same size");
    assert_eq!(d.get(0, 0), Rgb::splat(0.75));
    let (peak, x, y) = d.max_abs();
    assert!((peak - 1.25).abs() < 1e-15);
    assert_eq!((x, y), (2, 1));

    let (lo, hi) = a.min_max_channel();
    assert!((lo + 0.5).abs() < 1e-15 && (hi - 1.5).abs() < 1e-15);
    assert!(a.rms() > 0.0);
    assert!(FloatImage::<f64>::new(1, 1).rms() == 0.0);

    let mismatched = FloatImage::<f64>::new(4, 2);
    assert!(a.difference(&mismatched).is_err());

    let narrow = FloatImage::<f32>::from_f64(&a);
    assert!((narrow.get(0, 0).x as f64 - 1.0).abs() < 1e-6);
    assert_eq!(narrow.to_f64().width, 3);
}

#[test]
fn quantisation_clamps_without_gamma() {
    let mut f = FloatImage::<f64>::new(4, 1);
    f.set(0, 0, Rgb::splat(-1.0));
    f.set(1, 0, Rgb::splat(0.0));
    f.set(2, 0, Rgb::splat(0.5));
    f.set(3, 0, Rgb::splat(2.0));
    let q = f.to_clamped_u8();
    assert_eq!(q.pixels[0], [0, 0, 0]);
    assert_eq!(q.pixels[1], [0, 0, 0]);
    assert_eq!(
        q.pixels[2],
        [127, 127, 127],
        "no gamma and no sRGB transfer: clamp, multiply by 255, truncate"
    );
    assert_eq!(q.pixels[3], [255, 255, 255]);
    assert_eq!(q.as_bytes().len(), 12);

    let mut hostile = FloatImage::<f64>::new(3, 1);
    hostile.set(0, 0, Rgb::splat(f64::NAN));
    hostile.set(1, 0, Rgb::splat(f64::INFINITY));
    hostile.set(2, 0, Rgb::splat(f64::NEG_INFINITY));
    let hq = hostile.to_clamped_u8();
    assert_eq!(hq.pixels[0], [0, 0, 0], "a NaN pixel must not become garbage");
    assert_eq!(hq.pixels[1], [255, 255, 255]);
    assert_eq!(hq.pixels[2], [0, 0, 0]);
    assert_eq!(
        (510.0f64) as u8,
        255,
        "RECORDED: Rust's float-to-int cast SATURATES, so it is a second net \
         behind the clamp. Removing the clamp is therefore not observable at u8 \
         output - the clamp states the intent and guards the f32 path, it is not \
         the only thing preventing wraparound."
    );
    assert_eq!((-255.0f64) as u8, 0);
}

#[test]
fn png_round_trips_through_the_codec() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("round_trip.png");

    let mut image = Image::new(5, 3);
    for (i, p) in image.pixels.iter_mut().enumerate() {
        *p = [(i * 7) as u8, (i * 11) as u8, (i * 3) as u8];
    }
    write_png(&path, &image).expect("write");
    let back = read_png(&path).expect("read");
    assert_eq!(back, image, "the codec round trip must be lossless");

    let mut float = FloatImage::<f64>::new(5, 3);
    float.set(1, 1, Rgb::splat(0.004));
    let normalized_path = dir.path().join("normalized.png");
    let scale = write_png_normalized(&normalized_path, &float).expect("write");
    assert!((scale - 250.0).abs() < 1e-9, "scale {scale}");
    let normalized = read_png(&normalized_path).expect("read");
    assert_eq!(normalized.pixels[1 * 5 + 1], [255, 255, 255]);

    let flat = FloatImage::<f64>::new(2, 2);
    let flat_path = dir.path().join("flat.png");
    assert_eq!(write_png_normalized(&flat_path, &flat).expect("write"), 1.0);

    assert!(write_png(&dir.path().join("no/such/dir/x.png"), &image).is_err());
    assert!(read_png(&dir.path().join("absent.png")).is_err());
}

#[cfg(feature = "method-splat")]
#[test]
fn the_public_render_path_pays_for_exactly_one_f64_frame() {
    let pixels = (WIDEN_WIDTH as usize) * (WIDEN_HEIGHT as usize);
    let wide_frame = pixels * core::mem::size_of::<Rgb<f64>>();
    let narrow_frame = pixels * core::mem::size_of::<Rgb<f32>>();
    assert_eq!(wide_frame, 123384);
    assert_eq!(narrow_frame, 61692);

    let (narrow_hits, _) = frames_of_size(narrow_frame, || {
        let outcome = render(&widen_request(Precision::F32, Collect::Pixels)).expect("render");
        assert_eq!(outcome.image.pixels.len(), pixels);
    });
    assert_eq!(
        narrow_hits, 1,
        "the instrument must see the f32 frame the renderer really resolves,          otherwise the counts below measure nothing"
    );

    let (wide_from_f32, _) = frames_of_size(wide_frame, || {
        render(&widen_request(Precision::F32, Collect::Pixels)).expect("render");
    });
    assert_eq!(
        wide_from_f32, 1,
        "a f32 render owes exactly one widening: RenderOutcome::image is f64"
    );

    let (wide_from_f64, total_f64) = frames_of_size(wide_frame, || {
        let outcome = render(&widen_request(Precision::F64, Collect::Pixels)).expect("render");
        assert_eq!(outcome.image.pixels.len(), pixels);
    });
    assert_eq!(
        wide_from_f64, 1,
        "a f64 render allocated {wide_from_f64} whole {wide_frame}-byte frames out of          {total_f64} bytes in the whole call: resolve_float already built the f64 frame          the caller receives, so widening it again is a pure copy"
    );

    let (wide_with_probes, _) = frames_of_size(wide_frame, || {
        let outcome =
            render(&widen_request(Precision::F64, Collect::PixelsAndTolerance)).expect("render");
        assert!(outcome.tolerance.is_some(), "the tolerance mode must produce its image");
    });
    assert_eq!(
        wide_with_probes, 3,
        "image, tolerance and truncation: one frame each, never two"
    );
}

#[test]
#[cfg(feature = "method-splat")]
fn a_render_hands_out_premultiplied_colour_and_coverage_that_recompose_to_the_image() {
    let background = Rgb::new(0.31, 0.17, 0.44);
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    settings.background = background;

    let mut scene = Scene::<f64>::new();
    for (index, colour) in [
        Rgb::new(0.9, 0.2, 0.1),
        Rgb::new(0.1, 0.8, 0.3),
        Rgb::new(0.2, 0.3, 0.9),
    ]
    .iter()
    .enumerate()
    {
        let atom = Atom::isotropic(
            Vec3::new(-0.25 + 0.25 * index as f64, 0.1 * index as f64, 4.0),
            0.16,
            0.75,
            *colour,
        );
        scene.push(atom, MIN_SCALE).expect("scene rejected an atom");
    }

    let methods = RenderMethod::implemented();
    assert!(
        !methods.is_empty(),
        "no method is built, so this case would pass by measuring nothing"
    );
    let blends = [KernelFamily::Gaussian, KernelFamily::GaussianNormalised];
    let mut checked = 0usize;
    for method in methods.iter() {
        for kernel in blends.iter() {
            let request = |collect| RenderRequest {
                precision: Precision::F64,
                kernel: *kernel,
                method: method.clone(),
                settings,
                camera: CameraSpec::look_at(
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    40.0,
                    64,
                    48,
                ),
                scene: scene.clone(),
                collect,
                shapes: None,
            };
            let flat = render(&request(Collect::Pixels)).expect("flat render");
            let split = render(&request(Collect::PixelsAndCoverage)).expect("split render");

            let premultiplied = match split.premultiplied {
                Some(p) => p,
                None => panic!(
                    "Collect::PixelsAndCoverage returned no premultiplied colour, so a caller \
                     cannot composite this render over anything"
                ),
            };
            let alpha = match split.coverage {
                Some(a) => a,
                None => panic!("Collect::PixelsAndCoverage returned no coverage"),
            };
            assert!(
                render(&request(Collect::Pixels))
                    .expect("flat render")
                    .coverage
                    .is_none(),
                "Collect::Pixels must not pay for a coverage plane nobody asked for"
            );

            let plate = FloatImage::<f64>::filled(64, 48, background);
            let recomposed = premultiplied
                .composite_over(&alpha, &plate)
                .expect("composite_over");
            let gap = recomposed
                .difference(&flat.image)
                .expect("difference")
                .max_abs();
            assert!(
                gap.0 < 1.0e-12,
                "{} + {}: premultiplied colour composited over the same background must \
                 reproduce resolve_float's own image. Worst pixel ({},{}) is off by {:e}. \
                 This is the identity resolve(acc, bg) == resolve(acc, 0) + bg*(1 - coverage), \
                 and it is what lets a caller draw this render over a canvas it already has. \
                 It is not bit-identical because 1 - (1 - t) is not t in floating point",
                method.id().name(),
                kernel.name(),
                gap.1,
                gap.2,
                gap.0
            );

            let (lo, hi) = alpha.min_max_channel();
            assert!(
                lo >= 0.0 && hi <= 1.0,
                "{} + {}: coverage must stay in [0,1]: got {lo} to {hi}",
                method.id().name(),
                kernel.name()
            );
            assert!(
                hi > 0.5,
                "{} + {}: three atoms at opacity 0.75 must cover something; peak coverage \
                 {hi} means the plane is empty and this case is measuring nothing",
                method.id().name(),
                kernel.name()
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        methods.len() * blends.len(),
        "every built method must be crossed with both blend classes"
    );
}

#[test]
fn the_framebuffer_states_its_own_size_and_it_is_the_padded_tile_grid() {
    let over = Framebuffer::<f64, Over>::new(1920, 1080, 16).expect("framebuffer");
    assert_eq!(
        over.resident_bytes(),
        66_846_720,
        "1920x1080 at tile 16 is 120x68 = 8160 TILES of 256 cells — the PADDED grid, not \
         2,073,600 pixels — and OverAccum<f64> is Rgb + transmittance = 32 bytes. 63.75 MiB \
         for one frame, which is larger than the 47.57 MiB lights-bench reported as the whole \
         cost of a 100,000-atom scene. A framebuffer that does not state its size is the \
         largest unmeasured allocation in the renderer"
    );
    assert_eq!(
        over.resident_bytes(),
        8160 * 256 * core::mem::size_of::<<Over as BlendMode<f64>>::Accum>(),
        "the number must be tiles x cells x accumulator, not a literal that drifts"
    );

    let small = Framebuffer::<f64, Over>::new(1921, 1081, 16).expect("framebuffer");
    assert!(
        small.resident_bytes() > over.resident_bytes(),
        "one pixel past a tile boundary buys a whole row AND a whole column of tiles: \
         {} against {}. That rounding is the reason the number is worth reporting",
        small.resident_bytes(),
        over.resident_bytes()
    );

    let f32_buffer = Framebuffer::<f32, Over>::new(1920, 1080, 16).expect("framebuffer");
    assert_eq!(
        f32_buffer.resident_bytes() * 2,
        over.resident_bytes(),
        "precision halves it exactly, so the framebuffer is the one cost in this crate that \
         the precision axis moves by construction"
    );
}
