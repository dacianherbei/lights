#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::strip::{DEFAULT_GAP, DEFAULT_ZOOM, horizontal_strip, upscale};
use lights::image::FloatImage;
use lights::numerics::vector::Rgb;

fn panel(width: u32, height: u32, shade: f64) -> FloatImage<f64> {
    let mut out = FloatImage::<f64>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            out.set(
                x,
                y,
                Rgb::new(shade, shade + 0.01 * (x as f64), shade + 0.02 * (y as f64)),
            );
        }
    }
    out
}

#[test]
fn a_strip_places_every_panel_at_its_own_offset() {
    let panels = [panel(5, 4, 0.1), panel(7, 4, 0.4), panel(3, 4, 0.8)];
    let gap = 2u32;
    let separator = Rgb::new(0.25, 0.0, 0.35);
    let strip = match horizontal_strip(&panels, gap, separator) {
        Ok(s) => s,
        Err(e) => panic!("the strip must compose: {e}"),
    };
    assert_eq!(strip.width, 5 + 7 + 3 + 2 * gap);
    assert_eq!(strip.height, 4);

    let mut cursor = 0u32;
    for source in panels.iter() {
        for y in 0..source.height {
            for x in 0..source.width {
                let there = strip.get(cursor + x, y);
                let here = source.get(x, y);
                assert_eq!(
                    (there.x, there.y, there.z),
                    (here.x, here.y, here.z),
                    "panel pixel ({x},{y}) landed wrong at column {}. A strip \
                     that shifts a panel makes two kernels look different when \
                     only the composition moved",
                    cursor + x
                );
            }
        }
        cursor += source.width;
        if cursor < strip.width {
            for column in 0..gap {
                let bar = strip.get(cursor + column, 0);
                assert_eq!(
                    (bar.x, bar.y, bar.z),
                    (separator.x, separator.y, separator.z),
                    "the separator bar must be painted, or a panel boundary \
                     reads as image content"
                );
            }
        }
        cursor += gap;
    }
}

#[test]
fn a_strip_with_no_gap_still_covers_every_column() {
    let panels = [panel(4, 3, 0.2), panel(4, 3, 0.6)];
    let strip = match horizontal_strip(&panels, 0, Rgb::zero()) {
        Ok(s) => s,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(strip.width, 8);
    assert_eq!(strip.get(0, 0).x, 0.2);
    assert_eq!(strip.get(4, 0).x, 0.6);
}

#[test]
fn a_strip_refuses_panels_of_different_heights() {
    let panels = [panel(4, 3, 0.2), panel(4, 5, 0.6)];
    match horizontal_strip(&panels, DEFAULT_GAP, Rgb::zero()) {
        Err(LightsError::SizeMismatch { a, b }) => {
            assert_eq!(a, (4, 5));
            assert_eq!(b, (4, 3));
        }
        Err(e) => panic!("the wrong refusal: {e}"),
        Ok(s) => panic!(
            "a {}x{} strip was built from panels of different heights, so one of \
             them was silently cropped or padded and the comparison it exists \
             for is no longer like for like",
            s.width, s.height
        ),
    }
    match horizontal_strip(&[], DEFAULT_GAP, Rgb::zero()) {
        Err(LightsError::ViewportZero { .. }) => (),
        Err(e) => panic!("the wrong refusal for an empty strip: {e}"),
        Ok(_) => panic!("an empty strip is not an image"),
    }
}

#[test]
fn upscaling_repeats_pixels_and_invents_none() {
    let source = panel(3, 2, 0.3);
    let zoomed = match upscale(&source, DEFAULT_ZOOM) {
        Ok(z) => z,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(zoomed.width, 3 * DEFAULT_ZOOM);
    assert_eq!(zoomed.height, 2 * DEFAULT_ZOOM);
    for y in 0..zoomed.height {
        for x in 0..zoomed.width {
            let there = zoomed.get(x, y);
            let here = source.get(x / DEFAULT_ZOOM, y / DEFAULT_ZOOM);
            assert_eq!(
                (there.x, there.y, there.z),
                (here.x, here.y, here.z),
                "({x},{y}) is not a repeat of its source pixel. The zoomed glyph \
                 exists to show what the atoms put in each pixel, so any \
                 interpolation would put the resampler's opinion into a picture \
                 about pixel grids"
            );
        }
    }
    match upscale(&source, 1) {
        Ok(same) => {
            assert_eq!((same.width, same.height), (source.width, source.height));
            assert_eq!(same.get(2, 1).x, source.get(2, 1).x);
        }
        Err(e) => panic!("a zoom of 1 is the identity: {e}"),
    }
    match upscale(&source, 0) {
        Err(LightsError::ViewportZero { .. }) => (),
        Err(e) => panic!("the wrong refusal for a zero zoom: {e}"),
        Ok(_) => panic!("a zoom of zero is not an image"),
    }
}
