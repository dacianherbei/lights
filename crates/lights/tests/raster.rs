
use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::LightsError;
use lights::image::{FloatImage, Image, RasterLayer};
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::{Atom, Scene};
use lights::settings::SharedSettings;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const EXACT: f64 = 1.0e-15;
const INK: Rgb<f64> = Rgb {
    x: 0.95,
    y: 0.85,
    z: 0.10,
};

fn byte(c: u8) -> f64 {
    f64::from(c) / 255.0
}

fn close(a: Rgb<f64>, b: Rgb<f64>, tol: f64) -> bool {
    (a.x - b.x).abs() <= tol && (a.y - b.y).abs() <= tol && (a.z - b.z).abs() <= tol
}

fn glyph() -> (u32, u32, Vec<u8>) {
    let (w, h) = (7u32, 5u32);
    let mut bytes = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = if x == 0 || x == w - 1 {
                255
            } else if y == h / 2 {
                128
            } else {
                (x * 30 + y) as u8
            };
            bytes[(y * w + x) as usize] = v;
        }
    }
    (w, h, bytes)
}

fn atom_render(method: RenderMethod) -> FloatImage<f64> {
    let mut scene = Scene::new();
    for (i, colour) in [Rgb::new(0.9, 0.2, 0.2), Rgb::new(0.2, 0.3, 0.9)].iter().enumerate() {
        let atom = Atom::isotropic(Vec3::new(-0.6 + 1.2 * i as f64, 0.0, 0.0), 0.5, 0.9, *colour);
        if let Err(e) = scene.push(atom, 1.0e-6) {
            panic!("a valid atom was refused: {e:?}");
        }
    }
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings: SharedSettings {
            threads: 1,
            ..SharedSettings::contract()
        },
        camera: CameraSpec::look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 45.0, WIDTH, HEIGHT),
        scene,
        collect: Collect::PixelsAndCoverage,
        shapes: None,
    };
    let outcome = match render(&request) {
        Ok(o) => o,
        Err(e) => panic!("the atom pass failed: {e:?}"),
    };
    let (Some(colour), Some(coverage)) = (outcome.premultiplied, outcome.coverage) else {
        panic!("PixelsAndCoverage returned no coverage");
    };
    let plate = FloatImage::filled(WIDTH, HEIGHT, Rgb::new(0.1, 0.2, 0.3));
    match colour.composite_over(&coverage, &plate) {
        Ok(frame) => frame,
        Err(e) => panic!("{e:?}"),
    }
}

#[test]
fn a_glyph_arrives_as_coverage_and_lands_over_an_atom_render_with_its_ink() {
    let methods = RenderMethod::implemented();
    assert!(!methods.is_empty(), "no method is built, so this test would test nothing");
    let (gw, gh, bytes) = glyph();
    let layer = match RasterLayer::<f64>::from_coverage8(gw, gh, &bytes, INK) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    let (x0, y0) = (29i32, 21i32);
    let placed = match layer.placed(WIDTH, HEIGHT, x0, y0) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    let mut covered = 0usize;
    for method in methods.iter() {
        let frame = atom_render(method.clone());
        let centre = frame.get(32, 23);
        assert!(
            (centre.x - 0.1).abs() > 0.05 || (centre.z - 0.3).abs() > 0.05,
            "{}: the atom pass drew nothing under the glyph, so the composite proves nothing: {centre:?}",
            method.id().name()
        );
        let out = match placed.over(&frame) {
            Ok(o) => o,
            Err(e) => panic!("{e:?}"),
        };
        for py in 0..HEIGHT {
            for px in 0..WIDTH {
                let (gx, gy) = (px as i32 - x0, py as i32 - y0);
                let under = frame.get(px, py);
                let want = if gx >= 0 && gy >= 0 && (gx as u32) < gw && (gy as u32) < gh {
                    let a = byte(bytes[(gy as u32 * gw + gx as u32) as usize]);
                    INK * a + under * (1.0 - a)
                } else {
                    under
                };
                let got = out.get(px, py);
                assert!(
                    close(got, want, EXACT),
                    "{}: pixel ({px},{py}) is {got:?}, want {want:?}",
                    method.id().name()
                );
            }
        }
        let solid = out.get(x0 as u32, y0 as u32);
        assert!(close(solid, INK, EXACT), "full coverage must be the ink alone: {solid:?}");
        covered += 1;
    }
    assert_eq!(covered, methods.len());
}

#[test]
fn straight_rgba_is_premultiplied_once_and_its_alpha_is_the_coverage() {
    let bytes = [200u8, 100, 0, 128, 10, 20, 30, 255, 255, 255, 255, 0];
    let layer = match RasterLayer::<f64>::from_rgba8(3, 1, &bytes) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    let half = byte(128);
    assert!(close(layer.premultiplied.get(0, 0), Rgb::new(byte(200) * half, byte(100) * half, 0.0), EXACT));
    assert!(close(layer.coverage.get(0, 0), Rgb::splat(half), EXACT));
    assert!(close(layer.premultiplied.get(1, 0), Rgb::new(byte(10), byte(20), byte(30)), EXACT));
    assert!(close(layer.coverage.get(1, 0), Rgb::splat(1.0), EXACT));
    assert!(close(layer.premultiplied.get(2, 0), Rgb::zero(), EXACT), "zero alpha carries no colour");
    let bg = FloatImage::filled(3, 1, Rgb::new(0.5, 0.5, 0.5));
    let out = match layer.over(&bg) {
        Ok(o) => o,
        Err(e) => panic!("{e:?}"),
    };
    assert!(close(out.get(2, 0), Rgb::new(0.5, 0.5, 0.5), EXACT), "a transparent texel leaves the canvas alone");
    assert!(
        close(out.get(0, 0), Rgb::new(byte(200) * half + 0.5 * (1.0 - half), byte(100) * half + 0.5 * (1.0 - half), 0.5 * (1.0 - half)), EXACT),
        "{:?}",
        out.get(0, 0)
    );
}

#[test]
fn an_rgb_png_is_an_opaque_layer_that_replaces_what_it_covers() {
    let image = Image {
        width: 2,
        height: 1,
        pixels: vec![[255, 0, 51], [0, 102, 255]],
    };
    let layer = match RasterLayer::<f64>::opaque(&image) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    let bg = FloatImage::filled(2, 1, Rgb::new(0.7, 0.7, 0.7));
    let out = match layer.over(&bg) {
        Ok(o) => o,
        Err(e) => panic!("{e:?}"),
    };
    assert!(close(out.get(0, 0), Rgb::new(1.0, 0.0, 0.2), EXACT), "{:?}", out.get(0, 0));
    assert!(close(out.get(1, 0), Rgb::new(0.0, 0.4, 1.0), EXACT), "{:?}", out.get(1, 0));
}

#[test]
fn a_layer_placed_across_every_edge_keeps_only_the_texels_that_land() {
    let (w, h) = (4u32, 3u32);
    let bytes: Vec<u8> = (0..w * h).map(|i| (i * 20 + 15) as u8).collect();
    let layer = match RasterLayer::<f64>::from_coverage8(w, h, &bytes, Rgb::splat(1.0)) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    for (fw, fh, x0, y0) in [(5u32, 4u32, -2i32, -1i32), (5, 4, 3, 2), (5, 4, -10, 0), (2, 2, -1, -1)] {
        let placed = match layer.placed(fw, fh, x0, y0) {
            Ok(l) => l,
            Err(e) => panic!("{e:?}"),
        };
        assert_eq!((placed.width(), placed.height()), (fw, fh));
        for py in 0..fh {
            for px in 0..fw {
                let (sx, sy) = (px as i32 - x0, py as i32 - y0);
                let want = if sx >= 0 && sy >= 0 && (sx as u32) < w && (sy as u32) < h {
                    byte(bytes[(sy as u32 * w + sx as u32) as usize])
                } else {
                    0.0
                };
                let got = placed.coverage.get(px, py).x;
                assert!((got - want).abs() <= EXACT, "frame {fw}x{fh} at ({x0},{y0}): ({px},{py}) {got} vs {want}");
            }
        }
    }
}

#[test]
fn a_raster_whose_byte_count_is_not_its_size_is_refused() {
    match RasterLayer::<f64>::from_coverage8(3, 2, &[0u8; 5], INK) {
        Err(LightsError::RasterLength { width: 3, height: 2, channels: 1, got: 5 }) => {}
        other => panic!("a short coverage raster was not refused: {other:?}"),
    }
    match RasterLayer::<f64>::from_rgba8(2, 2, &[0u8; 12]) {
        Err(LightsError::RasterLength { channels: 4, got: 12, .. }) => {}
        other => panic!("an RGB buffer passed as RGBA was not refused: {other:?}"),
    }
    match RasterLayer::<f64>::from_coverage8(2, 2, &[0u8; 5], INK) {
        Err(LightsError::RasterLength { got: 5, .. }) => {}
        other => panic!("a coverage raster with a byte too many was not refused: {other:?}"),
    }
    match RasterLayer::<f64>::from_rgba8(0, 2, &[]) {
        Err(LightsError::ViewportZero { .. }) => {}
        other => panic!("an empty raster was not refused: {other:?}"),
    }
}

fn write_rgba_png(path: &std::path::Path, width: u32, height: u32, bytes: &[u8]) {
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => panic!("{e:?}"),
    };
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = match encoder.write_header() {
        Ok(w) => w,
        Err(e) => panic!("{e:?}"),
    };
    if let Err(e) = writer.write_image_data(bytes) {
        panic!("{e:?}");
    }
}

fn scratch() -> tempfile::TempDir {
    match tempfile::Builder::new().prefix("lights-raster-").tempdir() {
        Ok(d) => d,
        Err(e) => panic!("{e:?}"),
    }
}

#[test]
fn an_rgba_png_on_disk_is_a_layer_whose_alpha_is_its_coverage() {
    let dir = scratch();
    let path = dir.path().join("icon.png");
    let bytes = [255u8, 0, 0, 128, 0, 255, 0, 0, 0, 0, 255, 255];
    write_rgba_png(&path, 3, 1, &bytes);
    let layer = match RasterLayer::<f64>::read_png(&path) {
        Ok(l) => l,
        Err(e) => panic!("an 8-bit RGBA png was refused: {e:?}"),
    };
    let from_bytes = match RasterLayer::<f64>::from_rgba8(3, 1, &bytes) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    assert_eq!(layer, from_bytes, "a png file and its own bytes must be the same layer");
    let bg = FloatImage::filled(3, 1, Rgb::new(0.5, 0.5, 0.5));
    let out = match layer.over(&bg) {
        Ok(o) => o,
        Err(e) => panic!("{e:?}"),
    };
    let a = byte(128);
    let half = Rgb::new(a + 0.5 * (1.0 - a), 0.5 * (1.0 - a), 0.5 * (1.0 - a));
    assert!(close(out.get(0, 0), half, EXACT), "{:?}", out.get(0, 0));
    assert!(close(out.get(1, 0), Rgb::splat(0.5), EXACT), "a transparent texel must leave the background: {:?}", out.get(1, 0));
    assert!(close(out.get(2, 0), Rgb::new(0.0, 0.0, 1.0), EXACT), "{:?}", out.get(2, 0));
}

#[test]
fn a_png_lights_wrote_reads_back_as_a_layer_drawn_over_every_method() {
    let methods = RenderMethod::implemented();
    assert!(!methods.is_empty(), "no method is built, so this test would test nothing");
    let dir = scratch();
    let path = dir.path().join("badge.png");
    let (bw, bh) = (5u32, 4u32);
    let badge = Image {
        width: bw,
        height: bh,
        pixels: (0..bw * bh).map(|i| [(i * 12) as u8, 200, (255 - i * 9) as u8]).collect(),
    };
    if let Err(e) = lights::image::write_png(&path, &badge) {
        panic!("{e:?}");
    }
    let layer = match RasterLayer::<f64>::read_png(&path) {
        Ok(l) => l,
        Err(e) => panic!("the png lights itself wrote was refused: {e:?}"),
    };
    let (x0, y0) = (30i32, 22i32);
    let placed = match layer.placed(WIDTH, HEIGHT, x0, y0) {
        Ok(l) => l,
        Err(e) => panic!("{e:?}"),
    };
    let mut covered = 0usize;
    for method in methods.iter() {
        let frame = atom_render(method.clone());
        let out = match placed.over(&frame) {
            Ok(o) => o,
            Err(e) => panic!("{e:?}"),
        };
        for py in 0..HEIGHT {
            for px in 0..WIDTH {
                let (gx, gy) = (px as i32 - x0, py as i32 - y0);
                let want = if gx >= 0 && gy >= 0 && (gx as u32) < bw && (gy as u32) < bh {
                    let p = badge.pixels[(gy as u32 * bw + gx as u32) as usize];
                    Rgb::new(byte(p[0]), byte(p[1]), byte(p[2]))
                } else {
                    frame.get(px, py)
                };
                let got = out.get(px, py);
                assert!(
                    close(got, want, EXACT),
                    "{}: pixel ({px},{py}) is {got:?}, want {want:?}",
                    method.id().name()
                );
            }
        }
        covered += 1;
    }
    assert_eq!(covered, methods.len());
}

#[test]
fn a_png_that_is_neither_rgb_nor_rgba_is_refused_rather_than_misread() {
    let dir = scratch();
    let path = dir.path().join("grey.png");
    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => panic!("{e:?}"),
    };
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 2, 2);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    match encoder.write_header() {
        Ok(mut w) => {
            if let Err(e) = w.write_image_data(&[0, 64, 128, 255]) {
                panic!("{e:?}");
            }
        }
        Err(e) => panic!("{e:?}"),
    }
    match RasterLayer::<f64>::read_png(&path) {
        Err(LightsError::PngEncode(_)) => {}
        other => panic!("a greyscale png must be refused, got {other:?}"),
    }
}
