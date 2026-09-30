use lights::image::{FloatImage, srgb_decode, srgb_encode};
use lights::numerics::vector::Rgb;

const EDGE_COVERAGE: f64 = 0.5;

fn one_pixel(value: f64) -> FloatImage<f64> {
    FloatImage::filled(1, 1, Rgb::splat(value))
}

fn edge_byte(ink: f64, background: f64, transfer_owned: bool) -> u8 {
    let (ink, background) = if transfer_owned {
        (srgb_decode(ink), srgb_decode(background))
    } else {
        (ink, background)
    };
    let premultiplied = one_pixel(ink * EDGE_COVERAGE);
    let coverage = one_pixel(EDGE_COVERAGE);
    let mut out = premultiplied
        .composite_over(&coverage, &one_pixel(background))
        .expect("same size");
    if transfer_owned {
        out = out.encode_srgb();
    }
    out.to_clamped_u8().pixels[0][0]
}

#[test]
fn the_srgb_transfer_is_the_iec_61966_2_1_pair() {
    assert!((srgb_encode(0.5_f64) - 0.735_356_983_052_449_5).abs() < 1e-12);
    assert!((srgb_decode(0.5_f64) - 0.214_041_140_482_232_55).abs() < 1e-12);
    assert!((srgb_encode(0.002_f64) - 0.002 * 12.92).abs() < 1e-15);
    assert!((srgb_decode(0.03_f64) - 0.03 / 12.92).abs() < 1e-15);
    assert_eq!(srgb_encode(0.0_f64), 0.0);
    assert!((srgb_encode(1.0_f64) - 1.0).abs() < 1e-12);
    assert!((srgb_encode(0.5_f32) - 0.735_357).abs() < 1e-6);
    for byte in 0..=255_u8 {
        let v = f64::from(byte) / 255.0;
        let back = (srgb_encode(srgb_decode(v)) * 255.0).round();
        assert_eq!(back, f64::from(byte), "byte {byte} does not survive decode then encode");
    }
}

#[test]
fn a_half_covered_edge_differs_by_sixty_levels_between_the_two_interpretations() {
    let naive_light_on_dark = edge_byte(1.0, 0.0, false);
    let owned_light_on_dark = edge_byte(1.0, 0.0, true);
    let naive_dark_on_light = edge_byte(0.0, 1.0, false);
    let owned_dark_on_light = edge_byte(0.0, 1.0, true);
    assert_eq!(naive_light_on_dark, 127, "lights applies no transfer of its own");
    assert_eq!(naive_dark_on_light, 127, "lights applies no transfer of its own");
    assert_eq!(owned_light_on_dark, 187);
    assert_eq!(owned_dark_on_light, 187);
    assert_eq!(owned_light_on_dark - naive_light_on_dark, 60);
}

#[test]
fn image_transfer_maps_every_channel_and_keeps_the_size() {
    let mut image = FloatImage::<f32>::new(2, 1);
    image.set(0, 0, Rgb::new(0.0, 0.5, 1.0));
    image.set(1, 0, Rgb::new(0.25, 0.75, 0.1));
    let encoded = image.encode_srgb();
    assert_eq!((encoded.width, encoded.height), (2, 1));
    let round = encoded.decode_srgb();
    for (a, b) in image.pixels.iter().zip(round.pixels.iter()) {
        for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }
    assert!((encoded.get(0, 0).y - srgb_encode(0.5_f32)).abs() < 1e-7);
    assert!((encoded.get(1, 0).z - srgb_encode(0.1_f32)).abs() < 1e-7);
}
