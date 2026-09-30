#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::coverage::{DEFAULT_MAX_DEPTH, coverage_budget};
use lights::fit::metric::{
    DECIBEL_GAIN, EXACTNESS_LIMIT, ExactReference, LN_10, PEAK, Provenance, RASTER_LEVELS, mse,
    psnr,
};
use lights::fit::target::Target;
use lights::image::{FloatImage, Image, read_png, write_png};
use lights::numerics::vector::Rgb;

fn reference(max_depth: u32) -> Result<ExactReference<f64>, LightsError> {
    let target = Target::<f64>::t2_straight_edge();
    let mut budget = coverage_budget(64_000_000);
    ExactReference::from_target(&target, max_depth, &mut budget)
}

fn offset_by(reference: &ExactReference<f64>, delta: f64) -> FloatImage<f64> {
    let base = reference.image();
    let mut out = FloatImage::<f64>::new(base.width, base.height);
    for y in 0..base.height {
        for x in 0..base.width {
            out.set(x, y, base.get(x, y) + Rgb::splat(delta));
        }
    }
    out
}

#[test]
fn the_reference_is_accepted_at_full_depth_and_refused_when_the_quadrature_is_coarse() {
    let fine = match reference(DEFAULT_MAX_DEPTH) {
        Ok(r) => r,
        Err(e) => panic!("depth {DEFAULT_MAX_DEPTH} must be exact enough, got {e}"),
    };
    let fraction = fine.undecided_fraction();
    assert!(
        fraction > 0.0,
        "a straight edge crossing a 64x64 raster cannot leave ZERO undecided area \
         at finite depth: {fraction} means the residual is not being accumulated"
    );
    assert!(
        fraction <= EXACTNESS_LIMIT,
        "undecided fraction {fraction} above the stated limit {EXACTNESS_LIMIT}"
    );
    assert_eq!(fine.residual().max_depth_reached, DEFAULT_MAX_DEPTH);

    match reference(0) {
        Ok(r) => panic!(
            "a single-cell quadrature left {} undecided and was accepted",
            r.undecided_fraction()
        ),
        Err(LightsError::ReferenceNotExact { fraction, limit }) => {
            assert_eq!(limit, EXACTNESS_LIMIT);
            assert!(
                fraction > 100.0 * EXACTNESS_LIMIT,
                "the coarse case must be refused by a wide margin, got {fraction}"
            );
            println!("coarse undecided fraction {fraction} against limit {limit}");
        }
        Err(e) => panic!("the coarse case must be refused by name, got {e}"),
    }
    println!("fine undecided fraction {fraction}");
}

#[test]
fn the_mean_squared_error_averages_over_three_channels_per_pixel() {
    let r = match reference(DEFAULT_MAX_DEPTH) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let delta = 0.25;
    let shifted = offset_by(&r, delta);
    let measured = match mse(&shifted, &r) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    let expected = delta * delta;
    assert!(
        (measured - expected).abs() < 1.0e-12,
        "a constant offset of {delta} on every channel must give exactly \
         {expected}, got {measured}: the divisor is pixels times three channels"
    );
}

#[test]
fn the_decibel_figure_falls_by_six_when_the_error_doubles() {
    let r = match reference(DEFAULT_MAX_DEPTH) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let near = match psnr(&offset_by(&r, 0.05), &r) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    let far = match psnr(&offset_by(&r, 0.10), &r) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    };
    let drop = near - far;
    let six = 20.0 * (2.0f64).ln() / LN_10;
    assert!(
        (drop - six).abs() < 1.0e-9,
        "doubling the error must cost exactly {six} dB, got {drop}: that is the \
         one number a wrong gain, a wrong logarithm base or a root-mean-square \
         in place of a mean square all move"
    );
    let hand = DECIBEL_GAIN * (PEAK * PEAK / (0.05 * 0.05)).ln() / LN_10;
    assert!(
        (near - hand).abs() < 1.0e-9,
        "the absolute figure must match 10*log10(peak^2/mse) computed by hand: \
         {near} against {hand}"
    );
    println!("psnr {near} dB at 0.05, {far} dB at 0.10, drop {drop} dB");
}

#[test]
fn an_identical_candidate_is_refused_rather_than_scored_infinite() {
    let r = match reference(DEFAULT_MAX_DEPTH) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let same = r.image().clone();
    match mse(&same, &r) {
        Ok(v) => assert_eq!(v, 0.0, "an image against itself has zero error"),
        Err(e) => panic!("{e}"),
    }
    match psnr(&same, &r) {
        Ok(v) => panic!("an identical candidate must be refused, got {v}"),
        Err(LightsError::MetricUndefined { what }) => {
            assert!(what.contains("identical"), "{what}")
        }
        Err(e) => panic!("wrong refusal: {e}"),
    }
}

#[test]
fn a_candidate_of_the_wrong_size_is_refused() {
    let r = match reference(DEFAULT_MAX_DEPTH) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    let wrong = FloatImage::<f64>::new(8, 8);
    match mse(&wrong, &r) {
        Ok(v) => panic!("an 8x8 candidate against a 64x64 reference scored {v}"),
        Err(LightsError::SizeMismatch { a, b }) => {
            assert_eq!(a, (8, 8));
            assert_eq!(b, (64, 64));
        }
        Err(e) => panic!("wrong refusal: {e}"),
    }
}

fn gradient_raster() -> Image {
    let mut raster = Image::new(17, 5);
    for (i, p) in raster.pixels.iter_mut().enumerate() {
        let v = (i * 3 % 256) as u8;
        *p = [v, 255 - v, (i % 2) as u8 * 255];
    }
    raster
}

fn raster_reference() -> ExactReference<f64> {
    match ExactReference::<f64>::from_raster(&gradient_raster()) {
        Ok(r) => r,
        Err(e) => panic!("a well-formed 8-bit raster must be accepted, got {e}"),
    }
}

#[test]
fn a_raster_reference_decodes_each_level_exactly_and_states_its_quantisation_floor() {
    let raster = gradient_raster();
    let r = raster_reference();
    assert_eq!(r.provenance(), Provenance::Raster { levels: 256 });
    assert_eq!(RASTER_LEVELS, 256);
    assert_eq!(r.undecided_fraction(), 0.0);
    for (decoded, bytes) in r.image().pixels.iter().zip(raster.pixels.iter()) {
        let want = [bytes[0], bytes[1], bytes[2]].map(|c| f64::from(c) / 255.0);
        assert_eq!([decoded.x, decoded.y, decoded.z], want);
    }
    let floor = r.quantisation_floor();
    let step = 1.0 / 255.0;
    assert!(
        (floor - step * step / 12.0).abs() < 1.0e-18,
        "the floor is the mean squared error of a rounding quantiser with step \
         1/255, step^2/12, and got {floor}"
    );
    let ceiling = DECIBEL_GAIN * (PEAK * PEAK / floor).ln() / LN_10;
    assert!(
        (ceiling - 58.922_616).abs() < 1.0e-5,
        "the decibel ceiling of an 8-bit raster is 10*log10(12*255^2), got {ceiling}"
    );
    let quadrature = match reference(DEFAULT_MAX_DEPTH) {
        Ok(q) => q,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(quadrature.provenance(), Provenance::Quadrature);
    assert_eq!(quadrature.quantisation_floor(), 0.0);
    println!("raster floor {floor} mse, ceiling {ceiling} dB");
}

#[test]
fn a_candidate_closer_than_the_rasters_quantisation_floor_is_refused() {
    let r = raster_reference();
    let floor = r.quantisation_floor();
    let inside = offset_by(&r, 0.5 * floor.sqrt());
    match psnr(&inside, &r) {
        Ok(v) => panic!(
            "a candidate a quarter of the floor away scored {v} dB, above the \
             raster's own resolution of the image it was quantised from"
        ),
        Err(LightsError::BeyondQuantisation { mean_square, floor: f }) => {
            assert_eq!(f, floor);
            assert!(mean_square < floor, "{mean_square} against {floor}");
        }
        Err(e) => panic!("wrong refusal: {e}"),
    }
    let far = match psnr(&offset_by(&r, 0.05), &r) {
        Ok(v) => v,
        Err(e) => panic!("a candidate well outside the floor must be scored, got {e}"),
    };
    let hand = DECIBEL_GAIN * (PEAK * PEAK / (0.05 * 0.05)).ln() / LN_10;
    assert!((far - hand).abs() < 1.0e-9, "{far} against {hand}");

    let quadrature = match reference(DEFAULT_MAX_DEPTH) {
        Ok(q) => q,
        Err(e) => panic!("{e}"),
    };
    match psnr(&offset_by(&quadrature, 0.5 * floor.sqrt()), &quadrature) {
        Ok(v) => assert!(v > 58.922_616, "{v}"),
        Err(e) => panic!(
            "the quadrature reference has no quantisation floor and must score the \
             same small offset, got {e}"
        ),
    }
}

#[test]
fn an_empty_or_malformed_raster_is_refused() {
    match ExactReference::<f64>::from_raster(&Image::new(0, 4)) {
        Ok(_) => panic!("an empty raster was accepted as a reference"),
        Err(LightsError::ViewportZero { width, height }) => assert_eq!((width, height), (0, 4)),
        Err(e) => panic!("wrong refusal: {e}"),
    }
    let mut short = gradient_raster();
    short.pixels.pop();
    match ExactReference::<f64>::from_raster(&short) {
        Ok(_) => panic!("a raster one pixel short of width*height was accepted"),
        Err(LightsError::MetricUndefined { what }) => assert!(what.contains("width times height")),
        Err(e) => panic!("wrong refusal: {e}"),
    }
    let mut long = gradient_raster();
    long.pixels.push([0, 0, 0]);
    match ExactReference::<f64>::from_raster(&long) {
        Ok(_) => panic!(
            "a raster one pixel past width*height was accepted, and the decode would \
             have dropped the extra pixel without a word"
        ),
        Err(LightsError::MetricUndefined { what }) => assert!(what.contains("width times height")),
        Err(e) => panic!("wrong refusal: {e}"),
    }
}

#[test]
fn a_raster_read_back_from_png_is_the_same_reference() {
    let raster = gradient_raster();
    let dir = std::env::temp_dir().join(format!("lights_fit_metric_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        panic!("{e}");
    }
    let path = dir.join("raster.png");
    if let Err(e) = write_png(&path, &raster) {
        panic!("{e}");
    }
    let back = match read_png(&path) {
        Ok(b) => b,
        Err(e) => panic!("{e}"),
    };
    let _ = std::fs::remove_dir_all(&dir);
    let decoded = match ExactReference::<f64>::from_raster(&back) {
        Ok(r) => r,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(decoded.image().pixels, raster_reference().image().pixels);
}
