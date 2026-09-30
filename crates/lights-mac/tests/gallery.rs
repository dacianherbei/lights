use std::collections::{BTreeMap, BTreeSet};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lights_mac::HostEvent;

const STEPS: [&str; 8] = ["open", "dragged", "methods", "oracle", "surfels", "lit", "resized", "back"];
const PROBE_DEADLINE: Duration = Duration::from_secs(120);
const POLL: Duration = Duration::from_millis(50);

struct Probe {
    report: BTreeMap<String, String>,
    frames: BTreeMap<String, Vec<u8>>,
}

impl Probe {
    fn get(&self, key: &str) -> &str {
        self.report.get(key).map(String::as_str).unwrap_or_else(|| {
            panic!(
                "the gallery probe's report has no {key} line, only {:?}",
                self.report.keys().collect::<Vec<_>>()
            )
        })
    }

    fn number(&self, key: &str) -> f64 {
        let text = self.get(key);
        text.parse()
            .unwrap_or_else(|_| panic!("{key} is {text:?}, not a number"))
    }

    fn pair(&self, key: &str, separator: char) -> (f64, f64) {
        let text = self.get(key);
        text.split_once(separator)
            .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
            .unwrap_or_else(|| panic!("{key} is {text:?}, not two numbers split by {separator:?}"))
    }

    fn pixels(&self, key: &str) -> (u32, u32) {
        let (w, h) = self.pair(key, 'x');
        (w as u32, h as u32)
    }

    fn rect(&self, key: &str) -> [f64; 4] {
        let text = self.get(key);
        let parts: Vec<f64> = text.split(' ').filter_map(|p| p.parse().ok()).collect();
        <[f64; 4]>::try_from(parts).unwrap_or_else(|_| panic!("{key} is {text:?}, not x y w h"))
    }

    fn list(&self, key: &str) -> Vec<&str> {
        let text = self.get(key);
        if text.is_empty() { Vec::new() } else { text.split(" | ").collect() }
    }

    fn frame(&self, step: &str) -> &[u8] {
        self.frames
            .get(step)
            .map(Vec::as_slice)
            .unwrap_or_else(|| panic!("step {step} presented no frame the probe could read back"))
    }

    fn at(&self, key: &str) -> (f64, f64) {
        let scale = self.number("scale");
        let (x, y) = self.pair(key, ' ');
        (x * scale, y * scale)
    }
}

fn probe() -> &'static Probe {
    static PROBE: OnceLock<Result<Probe, String>> = OnceLock::new();
    match PROBE.get_or_init(run_probe) {
        Ok(probe) => probe,
        Err(why) => panic!("{why}"),
    }
}

fn run_probe() -> Result<Probe, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("lights-gallery-probe-{}-{stamp}", std::process::id()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_lights-gallery"))
        .arg("--probe")
        .arg(&dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("the gallery binary could not be started: {e}"))?;
    let polls = PROBE_DEADLINE.as_millis() / POLL.as_millis();
    let exited = (0..polls).any(|_| {
        std::thread::sleep(POLL);
        matches!(child.try_wait(), Ok(Some(_)))
    });
    if !exited {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "the gallery probe did not finish within {PROBE_DEADLINE:?} and was killed: \
             something in the window is waiting for an event that never comes"
        ));
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("the gallery probe's output could not be read: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "the gallery probe failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = std::fs::read_to_string(dir.join("report.tsv"))
        .map_err(|e| format!("the gallery probe wrote no report: {e}"))?;
    let report: BTreeMap<String, String> = text
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let mut frames = BTreeMap::new();
    for step in STEPS {
        if let Ok(bytes) = std::fs::read(dir.join(format!("{step}.bgra"))) {
            frames.insert(step.to_string(), bytes);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(Probe { report, frames })
}

fn in_order(delivered: &[&str], expected: &[HostEvent]) -> bool {
    let mut rest = delivered.iter();
    expected.iter().all(|e| {
        let want = format!("{e:?}");
        rest.any(|got| *got == want)
    })
}

fn scene_of(p: &Probe, step: &str) -> String {
    p.get(&format!("{step}.scene")).to_string()
}

#[test]
fn the_gallery_opens_a_window_whose_selector_lists_three_scenes_and_every_change_is_a_new_frame() {
    let p = probe();
    let scenes = p.list("scenes");
    assert!(scenes.len() >= 3, "the gallery lists {scenes:?}; the close bar asks for three scenes");
    let distinct: BTreeSet<&&str> = scenes.iter().collect();
    assert_eq!(distinct.len(), scenes.len(), "two scenes share a name: {scenes:?}");
    assert_eq!(p.list("segments"), scenes, "the selector's segments are not the scenes, in order");

    for step in STEPS {
        assert_eq!(
            p.get(&format!("{step}.presented")),
            "true",
            "step {step} changed what is shown and presented no new frame"
        );
        let (w, h) = p.pixels(&format!("{step}.frame_pixels"));
        let frame = p.frame(step);
        assert_eq!(frame.len(), (w * h * 4) as usize, "step {step}: the drawable is not {w}x{h} BGRA");
        let first = &frame[..4];
        assert!(
            frame.chunks_exact(4).any(|px| px != first),
            "step {step}: the drawable holds one colour, so nothing was drawn"
        );
    }
    let shown: BTreeSet<String> = STEPS.iter().map(|s| scene_of(p, s)).collect();
    assert_eq!(shown.len(), scenes.len(), "the run showed {shown:?} of {scenes:?}");

    assert!(
        p.list("close.events").contains(&format!("{:?}", HostEvent::CloseRequested).as_str()),
        "the close button raised no request"
    );
    assert_eq!(p.get("close.open"), "false", "the gallery did not answer the close request by ending");
}

#[test]
fn each_scene_shows_its_own_caption_above_the_render_area() {
    let p = probe();
    let mut captions = BTreeMap::new();
    for step in STEPS {
        let scene = scene_of(p, step);
        let caption = p.get(&format!("{step}.caption"));
        assert!(!caption.is_empty(), "step {step}: the caption above {scene} is empty");
        assert_eq!(
            caption,
            p.get(&format!("caption_of.{scene}")),
            "step {step}: the caption on screen is not the one {scene} carries"
        );
        captions.insert(scene, caption.to_string());

        let [_, caption_y, caption_w, caption_h] = p.rect(&format!("{step}.caption_rect"));
        let [_, render_y, render_w, render_h] = p.rect(&format!("{step}.render_rect"));
        assert!(caption_w > 0.0 && caption_h > 0.0, "step {step}: the caption has no area");
        assert!(render_w > 0.0 && render_h > 0.0, "step {step}: the render area has no area");
        assert!(
            caption_y >= render_y + render_h,
            "step {step}: the caption (bottom at {caption_y}) is not above the render area (top at {})",
            render_y + render_h
        );
    }
    let distinct: BTreeSet<&String> = captions.values().collect();
    assert_eq!(distinct.len(), captions.len(), "two scenes show the same caption: {captions:?}");
}

#[test]
fn the_scene_follows_the_keys_and_the_selectors_own_choice() {
    let p = probe();
    let scenes = p.list("scenes");
    for (step, index, how) in [
        ("open", 0, "on opening"),
        ("methods", 1, "after the key 2"),
        ("surfels", 2, "after the selector's third segment was chosen"),
        ("back", 0, "after the key 1"),
    ] {
        assert_eq!(scene_of(p, step), scenes[index], "{how}, the gallery shows the wrong scene");
        assert_eq!(p.get(&format!("{step}.selected")), index.to_string(), "{how}, the wrong scene is selected");
        assert_eq!(
            p.get(&format!("{step}.segment")),
            index.to_string(),
            "{how}, the selector does not show the scene that is shown"
        );
    }
    assert_ne!(p.frame("open"), p.frame("methods"), "the key 2 did not change the frame");
    assert_ne!(p.frame("oracle"), p.frame("surfels"), "choosing the third segment did not change the frame");
}

#[test]
fn a_drag_turns_the_orbit_by_one_turn_per_frame_width_and_the_next_frame_shows_it() {
    let p = probe();
    let (x0, y0) = p.at("drag_from_points");
    let by = p.number("drag_by_points") * p.number("scale");
    let expected = [
        HostEvent::PointerPressed { x: x0, y: y0 },
        HostEvent::PointerMoved { x: x0 + by, y: y0 },
        HostEvent::PointerReleased { x: x0 + by, y: y0 },
    ];
    assert!(
        in_order(&p.list("dragged.events"), &expected),
        "the drag arrived as {:?}, not {expected:?}",
        p.list("dragged.events")
    );
    let (width, _) = p.pixels("open.render_pixels");
    let turned = p.number("dragged.state.turn") - p.number("open.state.turn");
    let wanted = by / f64::from(width) * p.number("turns_per_width");
    assert!(
        (turned - wanted).abs() < 1.0e-12,
        "a drag of {by} px across a {width} px frame turned {turned}, not {wanted}"
    );
    assert_ne!(p.frame("open"), p.frame("dragged"), "the orbit turned and the frame did not change");
    assert_eq!(
        p.get("back.state.turn"),
        p.get("dragged.state.turn"),
        "the orbit lost its turn while another scene was shown"
    );
}

#[test]
fn the_ray_caster_is_off_until_asked_for_and_the_status_says_which_method_is_shown() {
    let p = probe();
    assert_eq!(p.get("methods.state.method"), "splat-raster", "the method scene does not open on splat");
    assert_eq!(p.get("methods.state.rendered_by"), "splat-raster", "splat is selected and something else drew");
    assert_eq!(p.get("oracle.state.method"), "ray-cast", "R did not select the ray caster");
    assert_eq!(p.get("oracle.state.rendered_by"), "ray-cast", "R selected the ray caster and it did not draw");
    assert!(
        p.get("methods.status").starts_with("splat-raster, the interactive default"),
        "the status line does not say splat is the interactive default: {:?}",
        p.get("methods.status")
    );
    assert!(
        p.get("oracle.status").starts_with("ray-cast, the oracle and not an interactive mode"),
        "the status line does not say the ray caster is the oracle: {:?}",
        p.get("oracle.status")
    );
    assert!(
        p.get("caption_of.methods").contains("off until asked for"),
        "the method scene's caption does not say the ray caster is off until asked for"
    );
    assert_eq!(p.get("methods.frame_pixels"), p.get("oracle.frame_pixels"), "the two methods drew different sizes");
    assert_ne!(p.frame("methods"), p.frame("oracle"), "R drew the same bytes as splat");
}

#[test]
fn every_frame_shows_its_cpu_time_beside_the_time_it_was_shown_after() {
    let p = probe();
    for step in STEPS {
        let cpu = p.number(&format!("{step}.cpu_ms"));
        let shown = p.number(&format!("{step}.shown_ms"));
        let render = p.number(&format!("{step}.render_ms"));
        let fill = p.number(&format!("{step}.fill_ms"));
        assert!(render > 0.0 && fill > 0.0, "step {step}: render {render} ms and fill {fill} ms");
        assert!(
            (cpu - (render + fill)).abs() < 1.0e-9,
            "step {step}: cpu {cpu} ms is not the render {render} ms and the fill {fill} ms"
        );
        assert!(cpu <= shown, "step {step}: cpu {cpu} ms and shown after {shown} ms");
        let status = p.get(&format!("{step}.status"));
        for part in [format!("cpu {cpu:.2} ms"), format!("shown after {shown:.2} ms")] {
            assert!(status.contains(&part), "step {step}: the status line {status:?} does not show {part:?}");
        }
    }
}

#[test]
fn the_surfel_light_goes_where_the_pointer_is() {
    let p = probe();
    let (x, y) = p.at("light_at_points");
    assert!(
        p.list("lit.events").contains(&format!("{:?}", HostEvent::PointerPressed { x, y }).as_str()),
        "the press that places the light never arrived: {:?}",
        p.list("lit.events")
    );
    let (w, h) = p.pixels("lit.render_pixels");
    let (w, h) = (f64::from(w), f64::from(h));
    let reach = 0.5 * w.min(h);
    let (u, v) = ((x - 0.5 * w) / reach, (y - 0.5 * h) / reach);
    let length = (u * u + v * v).sqrt().max(1.0);
    let (u, v) = (u / length, v / length);
    let (tu, tv) = p.pair("lit.state.toward", ' ');
    assert!(
        (tu - u).abs() < 1.0e-12 && (tv - v).abs() < 1.0e-12,
        "the pointer at ({x}, {y}) px put the light at ({tu}, {tv}) on the disc, not ({u}, {v})"
    );
    assert!(u < 0.0 && v < 0.0, "the probe's own geometry is off: ({u}, {v}) is not up and to the left");
    assert_ne!(p.frame("surfels"), p.frame("lit"), "the light moved and the frame did not change");

    let rows = p.number("surfel_rows") as u32;
    let (fw, fh) = p.pixels("lit.frame_pixels");
    assert_eq!(fh, rows, "the surfel frame is {fh} rows, not {rows}");
    assert_eq!(fw, (w * f64::from(rows) / h).round() as u32, "the surfel frame's width does not keep the aspect");
}

#[test]
fn a_resize_reaches_the_render_area_below_the_chrome() {
    let p = probe();
    let scale = p.number("scale");
    let chrome = p.number("chrome_points");
    for (step, size) in [("open", "window_points"), ("resized", "resized_points")] {
        let (w, h) = p.pair(size, 'x');
        let expected = ((w * scale).round() as u32, ((h - chrome) * scale).round() as u32);
        assert_eq!(
            p.pixels(&format!("{step}.render_pixels")),
            expected,
            "step {step}: a {w}x{h} pt window less {chrome} pt of chrome is not the render area"
        );
    }
    let (w, h) = p.pixels("resized.render_pixels");
    let resized = HostEvent::Resized {
        width: w,
        height: h,
        scale,
    };
    assert!(
        p.list("resized.events").contains(&format!("{resized:?}").as_str()),
        "the resize arrived as {:?}, not as {resized:?}",
        p.list("resized.events")
    );
    let rows = p.number("surfel_rows") as u32;
    let expected = ((f64::from(w) * f64::from(rows) / f64::from(h)).round() as u32, rows);
    assert_eq!(p.pixels("resized.frame_pixels"), expected, "the frame after the resize is not at the new size");
}
