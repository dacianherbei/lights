#![cfg(feature = "fit")]
use lights::fit::greedy::{FitRequest, fit, fit_budget};
use lights::fit::metric::{ExactReference, psnr};
use lights::fit::target::{PrimitiveSet, Target};
use lights::image::{FloatImage, Image, read_png};
use lights::kernel::KernelFamily;
use lights::numerics::vector::Rgb;
use lights::settings::SharedSettings;
use std::path::PathBuf;
use std::time::Instant;

const CROPS: [&str; 6] = [
    "alberto-montalesi-176097.png",
    "alejandro-escamilla-6.png",
    "ales-krivec-15949.png",
    "alexander-shustov-73.png",
    "allef-vinicius-109434.png",
    "amy-zhang-15940.png",
];
const SIDE: u32 = 64;
const TILE: u32 = 16;
const RENDER_LIMIT: u64 = 40_000_000;

const IMAGEGS_SCALARS: u64 = 8;
const LIGHTS_SCALARS: u64 = 9;
const HALF_BITS: u64 = 16;
const IMAGEGS_SIDE: u64 = 2048;
const IMAGEGS_DEFAULT_GAUSSIANS: u64 = 10_000;
const IMAGEGS_DEFAULT_BPP: f64 = 0.305;

const PUBLISHED: [(f64, f64, f64); 2] = [(0.122, 29.20, 4.57), (0.366, 32.99, 4.49)];
const MEASURED_GREEDY: [f64; 2] = [21.2448, 24.4766];
const MEASURED_FLAT: f64 = 17.2273;
const BAND: f64 = 0.01;

fn bits_per_gaussian(scalars: u64) -> u64 {
    scalars * HALF_BITS
}

fn primitives_at(bpp: f64, pixels: u64, scalars: u64) -> u64 {
    (bpp * pixels as f64 / bits_per_gaussian(scalars) as f64).floor() as u64
}

fn fixture(name: &str) -> Image {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/clic2020")
        .join(name);
    match read_png(&path) {
        Ok(i) => i,
        Err(e) => panic!("the CLIC2020 crop {name} did not decode: {e}"),
    }
}

fn frame(name: &'static str) -> Target<f64> {
    Target {
        name,
        viewport: (SIDE, SIDE),
        set: PrimitiveSet {
            ink: Rgb::new(1.0, 1.0, 1.0),
            parts: Vec::new(),
            regions: Vec::new(),
            holes: Vec::new(),
        },
        paper: Rgb::new(0.0, 0.0, 0.0),
        probes: Vec::new(),
    }
}

fn mean_colour(image: &FloatImage<f64>) -> FloatImage<f64> {
    let n = image.pixels.len() as f64;
    let mut sum = Rgb::new(0.0, 0.0, 0.0);
    for p in image.pixels.iter() {
        sum = sum + *p;
    }
    let mean = sum * (1.0 / n);
    let mut out = FloatImage::new(image.width, image.height);
    for p in out.pixels.iter_mut() {
        *p = mean;
    }
    out
}

struct Row {
    crop: &'static str,
    atoms_requested: usize,
    atoms_placed: usize,
    exhausted: bool,
    renders: u64,
    seconds: f64,
    decibels: f64,
    flat_decibels: f64,
}

fn fitted_row(crop: &'static str, atoms: usize) -> Row {
    let raster = fixture(crop);
    let reference = match ExactReference::<f64>::from_raster(&raster) {
        Ok(r) => r,
        Err(e) => panic!("{crop} is not a usable raster reference: {e}"),
    };
    let truth = reference.image().clone();
    let target = frame(crop);
    let mut request = FitRequest::new(KernelFamily::Gaussian, SharedSettings::<f64>::prefiltered());
    request.max_atoms = atoms;
    request.tile = TILE;
    let mut budget = fit_budget(RENDER_LIMIT);
    let started = Instant::now();
    let outcome = match fit(&target, &truth, &request, &mut budget) {
        Ok(o) => o,
        Err(e) => panic!("the fit of {crop} at {atoms} atoms failed: {e}"),
    };
    let seconds = started.elapsed().as_secs_f64();
    let rendered = match lights::fit::greedy::render_or_background(
        &outcome.atoms,
        &lights::fit::atom2d::camera(SIDE, SIDE),
        KernelFamily::Gaussian,
        &request.settings,
        TILE,
    ) {
        Ok(i) => i,
        Err(e) => panic!("re-rendering the fit of {crop} failed: {e}"),
    };
    let decibels = match psnr(&rendered, &reference) {
        Ok(v) => v,
        Err(e) => panic!("{crop} at {atoms} atoms has no decibel figure: {e}"),
    };
    let flat_decibels = match psnr(&mean_colour(&truth), &reference) {
        Ok(v) => v,
        Err(e) => panic!("{crop}'s mean colour has no decibel figure: {e}"),
    };
    Row {
        crop,
        atoms_requested: atoms,
        atoms_placed: outcome.atoms.len(),
        exhausted: outcome.improvement_exhausted,
        renders: outcome.renders,
        seconds,
        decibels,
        flat_decibels,
    }
}

#[test]
fn the_imagegs_bitrate_convention_reproduces_its_own_default() {
    let pixels = IMAGEGS_SIDE * IMAGEGS_SIDE;
    assert_eq!(bits_per_gaussian(IMAGEGS_SCALARS), 128);
    assert_eq!(bits_per_gaussian(LIGHTS_SCALARS), 144);
    let at_default = primitives_at(IMAGEGS_DEFAULT_BPP, pixels, IMAGEGS_SCALARS);
    let slack = IMAGEGS_DEFAULT_GAUSSIANS / 100;
    assert!(
        at_default.abs_diff(IMAGEGS_DEFAULT_GAUSSIANS) <= slack,
        "0.305 bpp at 2048x2048 under 8 float16 scalars gives {at_default} gaussians, \
         not the 10,000 image-gs ships; the convention is not the paper's"
    );
    assert_eq!(primitives_at(0.366, pixels, IMAGEGS_SCALARS), 11_993);
    assert_eq!(primitives_at(0.122, pixels, IMAGEGS_SCALARS), 3_997);
    let crop = u64::from(SIDE) * u64::from(SIDE);
    assert_eq!(primitives_at(0.366, crop, LIGHTS_SCALARS), 10);
    assert_eq!(primitives_at(0.122, crop, LIGHTS_SCALARS), 3);
}

#[test]
fn greedy_at_imagegs_bitrates_on_clic_crops() {
    let crop_pixels = u64::from(SIDE) * u64::from(SIDE);
    let mut table: Vec<(f64, f64, f64, Vec<Row>)> = Vec::new();
    for (bpp, published, spread) in PUBLISHED {
        let atoms = primitives_at(bpp, crop_pixels, LIGHTS_SCALARS) as usize;
        let rows: Vec<Row> = CROPS.iter().map(|c| fitted_row(c, atoms)).collect();
        table.push((bpp, published, spread, rows));
    }
    println!("bpp\tcrop\trequested\tplaced\texhausted\trenders\tseconds\tdB\tflat_dB");
    for (bpp, _, _, rows) in table.iter() {
        for r in rows.iter() {
            println!(
                "{bpp}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.4}\t{:.4}",
                r.crop,
                r.atoms_requested,
                r.atoms_placed,
                r.exhausted,
                r.renders,
                r.seconds,
                r.decibels,
                r.flat_decibels
            );
        }
    }
    for ((bpp, published, spread, rows), expected) in table.iter().zip(MEASURED_GREEDY) {
        let n = rows.len() as f64;
        let mean = rows.iter().map(|r| r.decibels).sum::<f64>() / n;
        let flat = rows.iter().map(|r| r.flat_decibels).sum::<f64>() / n;
        let renders = rows.iter().map(|r| r.renders).sum::<u64>() as f64 / n;
        println!(
            "summary bpp {bpp}: greedy {mean:.4} dB, mean colour {flat:.4} dB, \
             image-gs published {published} +- {spread} dB, gap {:.4} dB, mean renders {renders:.0}",
            published - mean
        );
        assert!(
            rows.iter().all(|r| r.atoms_placed == r.atoms_requested && !r.exhausted),
            "at {bpp} bpp a fit placed fewer atoms than image-gs's bitrate buys: the \
             comparison would not be at equal bits per pixel"
        );
        assert!(
            (flat - MEASURED_FLAT).abs() < BAND,
            "the mean-colour baseline moved to {flat:.4} dB from {MEASURED_FLAT}: the \
             crops are not decoding as the recorded rasters"
        );
        assert!(
            (mean - expected).abs() < BAND,
            "greedy at {bpp} bpp moved to {mean:.4} dB from the recorded {expected}: \
             re-measure and rewrite the record before quoting the gap"
        );
        assert!(
            rows.iter().all(|r| r.decibels > r.flat_decibels),
            "at {bpp} bpp a fit scored at or below its own mean colour: the \
             instrument is scoring something other than the fit"
        );
    }
}
