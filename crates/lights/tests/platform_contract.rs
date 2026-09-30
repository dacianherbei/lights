use lights::blend::OverAccum;
use lights::{Camera, CameraSpec, FloatImage, Framebuffer, Image, Over, Rgb, Vec3};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

const W: u32 = 5;
const H: u32 = 3;
const TILE: u32 = 4;
const SENTINEL: u8 = 0xA5;
const PADDING: usize = 12;
const ROOMY: usize = 16;

const DEFERRED: [&str; 3] = ["crate_split", "porting", "shared_part"];
const WITHDRAWN: [&str; 2] = ["dependency_audit", "optical_contract"];
const RULING: &str = "PM, 2026-09-29";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn contract() -> String {
    let path = crate_root().join("../../docs/lights/PLATFORM_CONTRACT.md");
    fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the platform contract (L-187) must exist at {}: {e}. It is what a host \
             implementation of lights is built against, and this test keeps its byte \
             table equal to what the crate actually hands out",
            path.display()
        )
    })
}

fn pinned(text: &str, prefix: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(prefix))
        .map(|rest| match rest.split_once(':') {
            Some((k, v)) => (k.trim().to_string(), v.trim().to_string()),
            None => panic!("`{prefix}{rest}` is pinned without a `key: value` shape"),
        })
        .collect()
}

fn keys(pins: &[(String, String)]) -> BTreeSet<String> {
    let set: BTreeSet<String> = pins.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(set.len(), pins.len(), "a key is pinned more than once: {pins:?}");
    set
}

fn pack(lit: &[(u32, u32, Rgb<f64>)]) -> Image {
    let mut image = FloatImage::<f64>::new(W, H);
    for &(x, y, v) in lit {
        image.set(x, y, v);
    }
    image.to_clamped_u8()
}

fn level(v: f64) -> u8 {
    pack(&[(0, 0, Rgb::new(v, v, v))]).pixels[0][0]
}

fn bytes_per_pixel() -> usize {
    let image = pack(&[]);
    let bytes = image.pixels.as_flattened();
    let pixels = (W * H) as usize;
    assert_eq!(bytes.len() % pixels, 0, "{} bytes do not divide into {pixels} pixels", bytes.len());
    bytes.len() / pixels
}

fn channel_order() -> String {
    let image = pack(&[(0, 0, Rgb::new(1.0, 0.5, 0.0))]);
    image.pixels.as_flattened()[..3]
        .iter()
        .map(|b| match b {
            255 => "R",
            0 => "B",
            _ => "G",
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn alpha() -> String {
    match bytes_per_pixel() {
        3 => "none, resolve composites over the background it is given".to_string(),
        n => format!("{n} bytes per pixel, so a channel beyond R G B exists"),
    }
}

fn row_stride() -> String {
    let white = Rgb::new(1.0, 1.0, 1.0);
    let image = pack(&[(0, 1, white), (W - 1, H - 1, white)]);
    let bytes = image.pixels.as_flattened();
    let bpp = bytes_per_pixel();
    let stride = match bytes.iter().position(|b| *b == 255) {
        Some(at) => at,
        None => panic!("no lit byte in a frame with two white pixels"),
    };
    let tail_lit = bytes[bytes.len() - bpp..].iter().all(|b| *b == 255);
    if stride == bpp * W as usize && bytes.len() == stride * H as usize && tail_lit {
        format!("{bpp} x width, no padding")
    } else {
        format!(
            "{stride} bytes from pixel (0,0) to pixel (0,1) at width {W}, {} bytes in all, last pixel lit {tail_lit}",
            bytes.len()
        )
    }
}

fn row_zero() -> String {
    let spec = CameraSpec::look_at([0.0, 0.0, -4.0], [0.0; 3], [0.0, 1.0, 0.0], 60.0, W * 20, H * 20);
    let camera = match Camera::<f64>::from_spec(&spec, 0.01) {
        Ok(c) => c,
        Err(e) => panic!("the probe camera is valid: {e}"),
    };
    let on_screen = |p: Vec3<f64>| match camera.project(camera.to_camera(p)) {
        Some(q) => q.y,
        None => panic!("the probe point {p:?} is in front of the camera"),
    };
    let centre = on_screen(Vec3::new(0.0, 0.0, 0.0));
    let above = on_screen(Vec3::new(0.0, 1.0, 0.0));
    if above < centre {
        "the edge CameraSpec::up points at".to_string()
    } else {
        "the edge opposite CameraSpec::up".to_string()
    }
}

fn transfer() -> String {
    match level(0.5) {
        127 | 128 => "none, the bytes are linear-coded".to_string(),
        187 | 188 => "sRGB-encoded".to_string(),
        other => format!("unrecognised, 0.5 packs to {other}"),
    }
}

fn quantisation() -> String {
    let probes = [-0.25, 0.5, 254.9 / 255.0, 1.75];
    let got: Vec<u8> = probes.iter().map(|v| level(*v)).collect();
    match got.as_slice() {
        [0, 127, 254, 255] => "truncation, floor(255 x clamp(v, 0, 1))".to_string(),
        [0, 128, 255, 255] => "rounding, round(255 x clamp(v, 0, 1))".to_string(),
        other => format!("unrecognised, {probes:?} pack to {other:?}"),
    }
}

fn float_layout() -> String {
    let source = match fs::read_to_string(crate_root().join("src/numerics/vector.rs")) {
        Ok(s) => s,
        Err(e) => panic!("src/numerics/vector.rs declares Vec3 and could not be read: {e}"),
    };
    let lines: Vec<&str> = source.lines().collect();
    let at = match lines.iter().position(|l| l.trim_start().starts_with("pub struct Vec3<")) {
        Some(n) => n,
        None => panic!("Vec3 is no longer declared in src/numerics/vector.rs, so this scan reads nothing"),
    };
    let attributes: Vec<&str> = lines[..at]
        .iter()
        .rev()
        .take_while(|l| l.trim_start().starts_with("#["))
        .copied()
        .collect();
    assert!(!attributes.is_empty(), "the attribute scan above Vec3 found not even its derive");
    if attributes.iter().any(|a| a.contains("repr(")) {
        "Vec3 carries a repr, so a FloatImage may have a byte layout".to_string()
    } else {
        "none, Vec3 is repr(Rust)".to_string()
    }
}

fn observed() -> Vec<(&'static str, String)> {
    vec![
        ("bytes_per_pixel", bytes_per_pixel().to_string()),
        ("channel_order", channel_order()),
        ("alpha", alpha()),
        ("row_stride", row_stride()),
        ("row_0", row_zero()),
        ("transfer", transfer()),
        ("quantisation", quantisation()),
        ("float_layout", float_layout()),
    ]
}

fn target_frame(lit: &[(u32, u32, Rgb<f64>)]) -> Framebuffer<f64, Over> {
    let mut fb = match Framebuffer::<f64, Over>::new(W, H, TILE) {
        Ok(fb) => fb,
        Err(e) => panic!("a {W}x{H} framebuffer at tile {TILE}: {e}"),
    };
    let grid = fb.grid();
    let tiles = match fb.split_tiles_mut(&grid) {
        Ok(t) => t,
        Err(e) => panic!("the framebuffer refused its own grid: {e}"),
    };
    for tile in tiles {
        let r = tile.rect;
        for &(x, y, v) in lit {
            let (x, y) = (x as i32, y as i32);
            if x >= r.x0 && x < r.x0 + r.w && y >= r.y0 && y < r.y0 + r.h {
                let at = ((y - r.y0) * tile.stride + (x - r.x0)) as usize;
                tile.cells[at] = OverAccum { rgb: v, transmittance: 0.0 };
            }
        }
    }
    fb
}

fn fused(fb: &Framebuffer<f64, Over>, stride: usize) -> Option<Vec<u8>> {
    let mut bytes = vec![SENTINEL; stride * H as usize];
    fb.resolve_bgra8(Rgb::new(0.0, 0.0, 0.0), &mut bytes, stride).ok().map(|()| bytes)
}

fn target_bytes_per_pixel() -> usize {
    let stride = ROOMY * W as usize;
    let bytes = match fused(&target_frame(&[]), stride) {
        Some(b) => b,
        None => panic!("a stride of {ROOMY} bytes a pixel was refused"),
    };
    let written = bytes[..stride].iter().position(|b| *b == SENTINEL).unwrap_or(stride);
    assert!(written > 0, "the fused resolve wrote nothing into row 0, so there is no pixel to measure");
    assert_eq!(written % W as usize, 0, "{written} bytes written into a row of {W} pixels");
    written / W as usize
}

fn target_channel_order() -> String {
    let bpp = target_bytes_per_pixel();
    let fb = target_frame(&[(0, 0, Rgb::new(0.8, 0.5, 0.2))]);
    match fused(&fb, bpp * W as usize) {
        Some(bytes) => bytes[..bpp]
            .iter()
            .map(|b| match b {
                204 => "R",
                127 => "G",
                51 => "B",
                255 => "A",
                _ => "?",
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => "refused at a stride of bytes per pixel x width".to_string(),
    }
}

fn target_alpha() -> String {
    let bpp = target_bytes_per_pixel();
    let fb = target_frame(&[(1, 1, Rgb::new(0.3, 0.6, 0.9)), (4, 2, Rgb::new(1.0, 1.0, 1.0))]);
    let bytes = match fused(&fb, bpp * W as usize) {
        Some(b) => b,
        None => return "refused at a stride of bytes per pixel x width".to_string(),
    };
    let alphas: Vec<u8> = bytes.chunks_exact(bpp).filter_map(|p| p.get(3).copied()).collect();
    if bpp == 4 && alphas.iter().all(|a| *a == u8::MAX) {
        "255, the frame is opaque over the background resolve is given".to_string()
    } else {
        format!("{bpp} bytes per pixel with fourth bytes {alphas:?}")
    }
}

fn target_row_stride() -> String {
    let bpp = target_bytes_per_pixel();
    let row = bpp * W as usize;
    let white = Rgb::new(1.0, 1.0, 1.0);
    let fb = target_frame(&[(0, 1, white)]);
    for stride in [row, row + PADDING] {
        let bytes = match fused(&fb, stride) {
            Some(b) => b,
            None => return format!("a stride of {stride} bytes was refused at width {W}"),
        };
        if bytes[stride..stride + 3] != [255, 255, 255] || bytes[stride - bpp..stride - bpp + 3] == [255, 255, 255] {
            return format!("pixel (0,1) is not at byte {stride} under a stride of {stride}");
        }
    }
    if fused(&fb, row - 1).is_some() {
        return format!("a stride of {} bytes, one short of a row, was accepted", row - 1);
    }
    format!("the caller's, at least {bpp} x width")
}

fn target_padding() -> String {
    let bpp = target_bytes_per_pixel();
    let stride = bpp * W as usize + PADDING;
    let fb = target_frame(&[(W - 1, 0, Rgb::new(1.0, 1.0, 1.0)), (W - 1, H - 1, Rgb::new(0.5, 0.5, 0.5))]);
    let bytes = match fused(&fb, stride) {
        Some(b) => b,
        None => return format!("a padded stride of {stride} bytes was refused"),
    };
    let touched = bytes
        .chunks_exact(stride)
        .filter(|row| row[bpp * W as usize..].iter().any(|b| *b != SENTINEL))
        .count();
    if touched == 0 {
        format!("bytes past {bpp} x width in a row are left as they were")
    } else {
        format!("{touched} of {H} rows had their padding written")
    }
}

fn target_colour() -> String {
    let bpp = target_bytes_per_pixel();
    let lit = [
        (0, 0, Rgb::new(0.8, 0.5, 0.2)),
        (2, 0, Rgb::new(-0.25, 1.75, 254.9 / 255.0)),
        (1, 1, Rgb::new(0.3, 0.6, 0.9)),
        (4, 2, Rgb::new(0.5, 0.25, 0.125)),
        (3, 2, Rgb::new(0.999, 0.001, 0.49)),
    ];
    let fb = target_frame(&lit);
    let frame = fb.resolve(Rgb::new(0.0, 0.0, 0.0));
    let bytes = match fused(&fb, bpp * W as usize) {
        Some(b) => b,
        None => return "refused at a stride of bytes per pixel x width".to_string(),
    };
    let differ = bytes
        .chunks_exact(bpp)
        .zip(&frame.pixels)
        .position(|(got, p)| got.len() < 3 || [got[2], got[1], got[0]] != *p);
    match differ {
        None => "the frame's R G B bytes, byte for byte".to_string(),
        Some(at) => format!("differs from the frame's R G B at pixel {at}"),
    }
}

fn target_observed() -> Vec<(&'static str, String)> {
    vec![
        ("bytes_per_pixel", target_bytes_per_pixel().to_string()),
        ("channel_order", target_channel_order()),
        ("alpha", target_alpha()),
        ("row_stride", target_row_stride()),
        ("padding", target_padding()),
        ("colour", target_colour()),
    ]
}

fn assert_table(doc: &str, prefix: &str, observed: &[(&'static str, String)]) {
    let stated = pinned(doc, prefix);
    let want: BTreeSet<String> = observed.iter().map(|(k, _)| k.to_string()).collect();
    assert_eq!(
        keys(&stated),
        want,
        "docs/lights/PLATFORM_CONTRACT.md must pin exactly the {prefix} facts this test can \
         observe, no more and no fewer: an unobserved line is a claim nothing checks"
    );
    for (key, value) in observed {
        let said = match stated.iter().find(|(k, _)| k == key) {
            Some((_, v)) => v,
            None => panic!("{prefix}{key} is not pinned"),
        };
        assert_eq!(
            said, value,
            "{prefix}{key}: the contract says `{said}` and the crate does `{value}`. A host is \
             written against the contract, so either the code regressed or the contract must \
             be amended in the same commit"
        );
    }
}

#[test]
fn the_contracts_byte_table_is_what_lights_hands_a_host() {
    assert_table(&contract(), "frame.", &observed());
}

#[test]
fn the_macos_byte_target_is_what_the_fused_resolve_writes() {
    assert_table(&contract(), "bgra8.", &target_observed());
}

#[test]
fn the_deferral_and_the_withdrawn_assumptions_stay_written_down() {
    let doc = contract();
    let deferred = pinned(&doc, "deferred.");
    let want: BTreeSet<String> = DEFERRED.iter().map(|k| k.to_string()).collect();
    assert_eq!(
        keys(&deferred),
        want,
        "L-187 records the PM's deferral so that no context re-opens it; removing a line is \
         re-opening it without a ruling"
    );
    for (key, value) in &deferred {
        assert!(value.contains(RULING), "deferred.{key} must cite its ruling ({RULING}): `{value}`");
    }
    let withdrawn = pinned(&doc, "withdrawn.");
    let want: BTreeSet<String> = WITHDRAWN.iter().map(|k| k.to_string()).collect();
    assert_eq!(
        keys(&withdrawn),
        want,
        "L-187's close bar is that the assumptions it used to carry are withdrawn in writing"
    );
    for (key, value) in &withdrawn {
        assert!(value.len() > 40, "withdrawn.{key} must say what was withdrawn and why: `{value}`");
    }
}
