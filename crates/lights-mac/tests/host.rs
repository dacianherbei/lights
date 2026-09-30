use std::collections::BTreeMap;
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use lights_mac::{BYTES_PER_PRESENTED_PIXEL, HostEvent};

const OPEN_POINTS: (f64, f64) = (320.0, 200.0);
const RESIZED_POINTS: (f64, f64) = (400.0, 260.0);
const PRESS_POINT: (f64, f64) = (40.0, 50.0);
const DRAG_POINT: (f64, f64) = (120.0, 90.0);
const FRAMES: [&str; 2] = ["open", "resized"];
const TIMED_PIXELS: &str = "1920x1080";
const HANDOFF_REPS: usize = 20;
const HANDOFF_TERMS: [&str; 6] = [
    "handoff_render_ms",
    "handoff_resolve_float_ms",
    "handoff_to_clamped_u8_ms",
    "handoff_expand_ms",
    "handoff_fused_ms",
    "handoff_three_pass_ms",
];
const RUN_FRAMES: usize = 20;
const RUNS_PER_ARM: usize = 2;
const FRAME_TERMS: [&str; 4] = [
    "frame_fused_cpu_ms",
    "frame_fused_interval_ms",
    "frame_three_pass_cpu_ms",
    "frame_three_pass_interval_ms",
];

struct Probe {
    report: BTreeMap<String, String>,
    frames: BTreeMap<String, (Vec<u8>, Vec<u8>, Vec<u8>)>,
}

impl Probe {
    fn get(&self, key: &str) -> &str {
        self.report.get(key).map(String::as_str).unwrap_or_else(|| {
            panic!(
                "the probe's report has no {key} line, only {:?}",
                self.report.keys().collect::<Vec<_>>()
            )
        })
    }

    fn screen_scale(&self) -> f64 {
        let text = self.get("screen_scale");
        text.parse()
            .unwrap_or_else(|_| panic!("screen_scale is {text:?}, so the window reported no screen"))
    }

    fn events(&self, key: &str) -> Vec<&str> {
        let text = self.get(key);
        if text.is_empty() {
            Vec::new()
        } else {
            text.split(" | ").collect()
        }
    }

    fn frame_width(&self, name: &str) -> usize {
        let size = self.get(&format!("{name}_frame"));
        size.split_once('x')
            .and_then(|(w, _)| w.parse().ok())
            .unwrap_or_else(|| panic!("{name}_frame is {size:?}, not WIDTHxHEIGHT"))
    }
}

fn probe() -> &'static Probe {
    static PROBE: OnceLock<Probe> = OnceLock::new();
    PROBE.get_or_init(run_probe)
}

fn run_probe() -> Probe {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!(
        "lights-mac-probe-{}-{stamp}",
        std::process::id()
    ));
    let out = Command::new(env!("CARGO_BIN_EXE_lights-mac-probe"))
        .arg(&dir)
        .output()
        .expect("the probe binary could not be started");
    assert!(
        out.status.success(),
        "the probe failed ({}): {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(dir.join("report.tsv")).expect("the probe wrote no report");
    let report = text
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let mut frames = BTreeMap::new();
    for name in FRAMES {
        let read = |ext: &str| {
            std::fs::read(dir.join(format!("{name}.{ext}")))
                .unwrap_or_else(|e| panic!("the probe wrote no {name}.{ext}: {e}"))
        };
        frames.insert(name.to_string(), (read("rgb"), read("bgra"), read("fused")));
    }
    let _ = std::fs::remove_dir_all(&dir);
    Probe { report, frames }
}

fn pixels(points: (f64, f64), scale: f64) -> (u32, u32) {
    ((points.0 * scale).round() as u32, (points.1 * scale).round() as u32)
}

fn size(points: (f64, f64), scale: f64) -> String {
    let (w, h) = pixels(points, scale);
    format!("{w}x{h}")
}

fn in_order(delivered: &[&str], expected: &[HostEvent]) -> bool {
    let mut rest = delivered.iter();
    expected.iter().all(|e| {
        let want = format!("{e:?}");
        rest.any(|got| *got == want)
    })
}

#[test]
fn the_window_reports_its_screens_scale_factor_and_its_size_in_pixels() {
    let p = probe();
    assert_eq!(p.get("open_points"), format!("{}x{}", OPEN_POINTS.0, OPEN_POINTS.1));
    let screen = p.screen_scale();
    let host: f64 = p.get("scale").parse().expect("scale is not a number");
    assert_eq!(
        host, screen,
        "the host reports a scale factor of {host} and the window's own screen says {screen}"
    );
    assert_eq!(p.get("open_pixels"), size(OPEN_POINTS, screen));
    assert_eq!(p.get("open_frame"), size(OPEN_POINTS, screen));
    println!("scale factor {screen}: {} pt is {} px", p.get("open_points"), p.get("open_pixels"));
}

#[test]
fn the_drawable_carries_the_lights_frame_byte_for_byte_as_bgra() {
    let p = probe();
    for name in FRAMES {
        let (rgb, bgra, _) = &p.frames[name];
        assert!(!rgb.is_empty() && rgb.len() % 3 == 0, "{name}.rgb holds {} bytes", rgb.len());
        let expected: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|px| [px[2], px[1], px[0], u8::MAX])
            .collect();
        assert_eq!(
            bgra.len(),
            rgb.len() / 3 * BYTES_PER_PRESENTED_PIXEL,
            "{name}: the drawable holds {} bytes for {} pixels",
            bgra.len(),
            rgb.len() / 3
        );
        let first = expected.iter().zip(bgra).position(|(want, got)| want != got);
        assert!(
            first.is_none(),
            "{name}: the drawable differs from the lights frame, first at byte {first:?}"
        );
        let row = p.frame_width(name) * 3;
        let rows: Vec<&[u8]> = rgb.chunks_exact(row).collect();
        assert_ne!(
            rows.first(),
            rows.last(),
            "{name}: the first and last rows are equal, so a vertical flip would pass unseen"
        );
        assert!(
            rgb.chunks_exact(3).any(|px| px[0] != px[2]),
            "{name}: no pixel has red unlike blue, so a channel swap would pass unseen"
        );
    }
}

#[test]
fn a_resize_arrives_as_an_event_with_the_new_pixel_size_and_the_next_frame_has_it() {
    let p = probe();
    let scale = p.screen_scale();
    assert_eq!(p.get("resized_points"), format!("{}x{}", RESIZED_POINTS.0, RESIZED_POINTS.1));
    let (width, height) = pixels(RESIZED_POINTS, scale);
    let delivered = p.events("resize_events");
    assert!(
        in_order(&delivered, &[HostEvent::Resized { width, height, scale }]),
        "a resize to {RESIZED_POINTS:?} pt delivered {delivered:?}"
    );
    assert_eq!(p.get("resized_pixels"), format!("{width}x{height}"));
    assert_eq!(p.get("resized_frame"), format!("{width}x{height}"));
    let (_, bgra, _) = &p.frames["resized"];
    assert_eq!(bgra.len(), (width * height) as usize * BYTES_PER_PRESENTED_PIXEL);
}

#[test]
fn pointer_press_motion_and_release_arrive_in_pixels_from_the_top_left() {
    let p = probe();
    let s = p.screen_scale();
    let expected = [
        HostEvent::PointerPressed { x: PRESS_POINT.0 * s, y: PRESS_POINT.1 * s },
        HostEvent::PointerMoved { x: DRAG_POINT.0 * s, y: DRAG_POINT.1 * s },
        HostEvent::PointerReleased { x: DRAG_POINT.0 * s, y: DRAG_POINT.1 * s },
    ];
    let delivered = p.events("input_events");
    assert!(
        in_order(&delivered, &expected),
        "pressing at {PRESS_POINT:?} pt and dragging to {DRAG_POINT:?} pt from the top left \
         delivered {delivered:?}, not {expected:?}"
    );
}

#[test]
fn a_key_press_names_its_key_by_code_and_by_character() {
    let p = probe();
    let expected = [
        HostEvent::KeyPressed { code: 0, characters: "a".to_string() },
        HostEvent::KeyPressed { code: 53, characters: "\u{1b}".to_string() },
    ];
    let delivered = p.events("input_events");
    assert!(in_order(&delivered, &expected), "two key presses delivered {delivered:?}");
}

#[test]
fn closing_the_window_is_a_request_the_caller_answers() {
    let p = probe();
    let delivered = p.events("close_events");
    assert!(
        in_order(&delivered, &[HostEvent::CloseRequested]),
        "the close button delivered {delivered:?}"
    );
    assert_eq!(
        p.get("visible_after_close_request"),
        "true",
        "the window closed itself, so the caller never got to answer the request"
    );
}

#[test]
fn a_frame_lights_resolves_into_the_hosts_buffer_shows_the_bytes_of_the_expanded_image() {
    let p = probe();
    for name in FRAMES {
        let (rgb, bgra, fused) = &p.frames[name];
        let expected: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|px| [px[2], px[1], px[0], u8::MAX])
            .collect();
        assert_eq!(
            fused.len(),
            expected.len(),
            "{name}: the fused present left {} bytes in the drawable for {} pixels",
            fused.len(),
            rgb.len() / 3
        );
        let first = expected.iter().zip(fused).position(|(want, got)| want != got);
        assert!(
            first.is_none(),
            "{name}: the drawable after lights resolved straight into the host's buffer differs \
             from the lights frame, first at byte {first:?}. The buffer held a flat poison \
             frame before the fill, so a fill that writes nothing cannot pass"
        );
        assert_eq!(fused, bgra, "{name}: the fused and the expanded present show different bytes");
    }
}

fn samples(p: &Probe, key: &str) -> (f64, f64, f64, usize) {
    let text = p.get(key);
    let words: Vec<&str> = text.split(' ').collect();
    let number = |label: &str| -> f64 {
        let at = words.iter().position(|w| *w == label).unwrap_or_else(|| {
            panic!("{key} is {text:?}, with no {label}")
        });
        words[at + 1]
            .parse()
            .unwrap_or_else(|_| panic!("{key}: {label} is {:?}, not a number", words[at + 1]))
    };
    let n = number("over") as usize;
    (number("min"), number("median"), number("max"), n)
}

#[test]
fn the_handoff_is_timed_in_both_arms_at_1080p_on_the_frame_the_host_shows() {
    let p = probe();
    assert_eq!(p.get("handoff_pixels"), TIMED_PIXELS);
    assert_eq!(
        p.get("handoff_arms_agree"),
        "true",
        "the probe must read back both timed arms and find them equal before it quotes a time"
    );
    let counted = HANDOFF_TERMS
        .iter()
        .map(|k| (*k, HANDOFF_REPS))
        .chain(FRAME_TERMS.iter().map(|k| (*k, RUN_FRAMES * RUNS_PER_ARM)));
    for (key, want) in counted {
        let (min, median, max, n) = samples(p, key);
        assert_eq!(n, want, "{key}: {n} samples, not {want}");
        assert!(
            min > 0.0 && min <= median && median <= max,
            "{key}: min {min} median {median} max {max} is not an ordered positive spread, so the \
             clock was not read around the work"
        );
        println!("{key}: {}", p.get(key));
    }
}
