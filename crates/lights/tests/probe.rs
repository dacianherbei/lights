#[cfg(feature = "method-splat")]
use lights::camera::CameraSpec;
#[cfg(feature = "method-splat")]
use lights::dispatch::{Collect, Precision, RenderRequest, render};
#[cfg(feature = "method-splat")]
use lights::kernel::KernelFamily;
#[cfg(feature = "method-splat")]
use lights::method::RenderMethod;
#[cfg(feature = "method-splat")]
use lights::method::splat::{SortKind, SplatConfig};
use lights::numerics::vector::Rgb;
#[cfg(feature = "method-splat")]
use lights::numerics::vector::Vec3;
use lights::probe::{AuditProbe, Counter, Counters, NoProbe, Probe, ProbeSink};
#[cfg(feature = "method-splat")]
use lights::scene::{Atom, Scene};
#[cfg(feature = "method-splat")]
use lights::settings::{MIN_SCALE, SharedSettings};
use lights::tiling::TileRect;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct SizeWatcher;

static WATCH_SIZE: AtomicUsize = AtomicUsize::new(0);
static WATCH_HITS: AtomicUsize = AtomicUsize::new(0);
static WATCH_ARMED: AtomicBool = AtomicBool::new(false);
static WATCH_GATE: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for SizeWatcher {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCH_ARMED.load(Ordering::Relaxed)
            && layout.size() == WATCH_SIZE.load(Ordering::Relaxed)
        {
            WATCH_HITS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static WATCHER: SizeWatcher = SizeWatcher;

fn allocations_of_size(size: usize, body: impl FnOnce()) -> usize {
    let gate = match WATCH_GATE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    WATCH_SIZE.store(size, Ordering::Relaxed);
    WATCH_HITS.store(0, Ordering::Relaxed);
    WATCH_ARMED.store(true, Ordering::Relaxed);
    body();
    WATCH_ARMED.store(false, Ordering::Relaxed);
    let hits = WATCH_HITS.load(Ordering::Relaxed);
    drop(gate);
    hits
}

const PROBE_TILE: u32 = 23;

#[cfg(feature = "method-splat")]
fn probe_request(collect: Collect) -> RenderRequest {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    let mut scene = Scene::<f64>::new();
    for (offset, colour) in [
        (0.0f64, Rgb::new(1.0, 0.0, 0.0)),
        (0.06, Rgb::new(0.0, 1.0, 0.0)),
        (-0.06, Rgb::new(0.0, 0.0, 1.0)),
    ] {
        scene
            .push(
                Atom::isotropic(Vec3::new(offset, offset, 5.0), 0.08, 0.7, colour),
                MIN_SCALE,
            )
            .expect("valid atom");
    }
    RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::SplatRaster(SplatConfig {
            tile: PROBE_TILE,
            sort: SortKind::GlobalSort,
        }),
        settings,
        camera: CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            2 * PROBE_TILE,
            2 * PROBE_TILE,
        ),
        scene,
        collect,
        shapes: None,
    }
}

fn rect() -> TileRect {
    TileRect {
        x0: 16,
        y0: 32,
        w: 4,
        h: 3,
    }
}

#[test]
fn no_probe_is_zero_sized_and_records_nothing() {
    assert_eq!(core::mem::size_of::<NoProbe>(), 0);
    let mut sink = ProbeSink::<f64>::with_tolerance(64, 64);
    let mut probe = <NoProbe as Probe<f64>>::for_tile(rect());
    <NoProbe as Probe<f64>>::count(&mut probe, Counter::KernelEvaluations, 999);
    <NoProbe as Probe<f64>>::tolerance(&mut probe, 0, 1.0, 1.0);
    <NoProbe as Probe<f64>>::merge_into(probe, &mut sink);
    assert_eq!(sink.counters, Counters::default());
    assert!(sink.tolerance.is_none());
}

#[test]
fn counters_merge_and_read_back_by_name() {
    let mut a = Counters::default();
    a.add(Counter::AtomsPrepared, 3);
    a.add(Counter::EarlyOuts, 5);
    let mut b = Counters::default();
    b.add(Counter::AtomsPrepared, 4);
    b.add(Counter::PixelsVisited, 7);
    a.merge(&b);
    assert_eq!(a.get(Counter::AtomsPrepared), 7);
    assert_eq!(a.get(Counter::EarlyOuts), 5);
    assert_eq!(a.get(Counter::PixelsVisited), 7);
    assert_eq!(a.get(Counter::SortKeys), 0);

    let every = [
        Counter::AtomsPrepared,
        Counter::AtomsProjected,
        Counter::AtomsCulledNear,
        Counter::AtomsCulledDet,
        Counter::AtomsCulledSupport,
        Counter::AtomsClamped,
        Counter::TileAtomPairs,
        Counter::SortKeys,
        Counter::CandidatesTested,
        Counter::KernelEvaluations,
        Counter::CompositedHits,
        Counter::EarlyOuts,
        Counter::PixelsVisited,
    ];
    for (i, c) in every.iter().enumerate() {
        let mut single = Counters::default();
        single.add(*c, (i + 1) as u64);
        assert_eq!(
            single.get(*c),
            (i + 1) as u64,
            "add and get must address the same field"
        );
        let total: u64 = every.iter().map(|other| single.get(*other)).sum();
        assert_eq!(total, (i + 1) as u64, "add must touch exactly one field");
    }
}

#[test]
fn audit_probe_without_tolerance_is_the_counters_only_probe() {
    let mut sink = ProbeSink::<f64>::counters_only(64, 64);
    assert!(!sink.collect_tolerance);
    let mut probe = <AuditProbe<f64> as Probe<f64>>::for_tile(rect());
    assert!(
        probe.tolerance_slice().is_none(),
        "no allocation until the render loop asks for one"
    );
    probe.count(Counter::CandidatesTested, 11);
    assert_eq!(probe.counters().get(Counter::CandidatesTested), 11);
    assert_eq!(probe.rect(), rect());
    probe.merge_into(&mut sink);
    assert_eq!(sink.counters.get(Counter::CandidatesTested), 11);
    assert!(sink.tolerance.is_none());
}

#[test]
fn audit_probe_accumulates_tolerance_at_the_right_pixels() {
    let mut sink = ProbeSink::<f64>::with_tolerance(64, 64);
    assert!(sink.collect_tolerance);
    let mut probe = <AuditProbe<f64> as Probe<f64>>::for_tile(rect());
    probe.tolerance(0, 0.5, 4.0);
    probe.tolerance(0, 0.25, 4.0);
    probe.tolerance(7, 1.0, 3.0);
    probe.tolerance(999999, 1.0, 1.0);
    assert!(probe.tolerance_slice().is_some());
    probe.merge_into(&mut sink);

    let image = sink.tolerance.as_ref().expect("tolerance image");
    assert!(
        (image.get(16, 32).x - 3.0).abs() < 1e-15,
        "T_i * B_atom accumulates in the SAME pass as the render"
    );
    assert!((image.get(19, 33).x - 3.0).abs() < 1e-15, "local index 7 is (col 3, row 1)");
    assert_eq!(image.get(0, 0).x, 0.0);
    assert_eq!(image.get(20, 32).x, 0.0);
}

#[test]
fn probe_sinks_from_two_tiles_do_not_collide() {
    let mut sink = ProbeSink::<f64>::with_tolerance(32, 32);
    for x0 in [0i32, 16] {
        let mut probe = <AuditProbe<f64> as Probe<f64>>::for_tile(TileRect {
            x0,
            y0: 0,
            w: 16,
            h: 16,
        });
        probe.count(Counter::PixelsVisited, 256);
        probe.tolerance(0, 1.0, (x0 + 1) as f64);
        probe.merge_into(&mut sink);
    }
    assert_eq!(sink.counters.get(Counter::PixelsVisited), 512);
    let image = sink.tolerance.as_ref().expect("tolerance image");
    assert!((image.get(0, 0).x - 1.0).abs() < 1e-15);
    assert!((image.get(16, 0).x - 17.0).abs() < 1e-15);
}

#[cfg(feature = "method-splat")]
#[test]
fn the_counters_only_mode_allocates_no_tolerance_buffer() {
    let tile_bytes = (PROBE_TILE as usize) * (PROBE_TILE as usize) * core::mem::size_of::<f64>();

    let mut saw_tolerance_image = false;
    let while_collecting = allocations_of_size(tile_bytes, || {
        let outcome = render(&probe_request(Collect::PixelsAndTolerance)).expect("render");
        saw_tolerance_image = outcome.tolerance.is_some();
    });
    assert!(saw_tolerance_image, "the tolerance mode must still produce an image");
    assert!(
        while_collecting > 0,
        "the instrument must be able to see a {tile_bytes}-byte tile buffer when one          is really allocated, otherwise the zero below measures nothing"
    );

    let mut counters = Counters::default();
    let while_counting = allocations_of_size(tile_bytes, || {
        let outcome = render(&probe_request(Collect::PixelsAndCounters)).expect("render");
        counters = outcome.counters;
    });
    assert!(
        counters.get(Counter::CompositedHits) > 0,
        "the cheap mode must still count what it composites"
    );
    assert_eq!(
        while_counting, 0,
        "Collect::PixelsAndCounters paid {while_counting} tile tolerance buffers of          {tile_bytes} bytes each, which merge_into then discards because the sink          does not collect tolerance"
    );
}
