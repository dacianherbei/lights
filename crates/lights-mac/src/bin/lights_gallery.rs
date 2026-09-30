mod gallery;

use std::fmt::Display;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use gallery::Scene;
use gallery::chrome::{CHROME_POINTS, Chrome};
use gallery::methods::ORACLE_KEY;
use gallery::orbit::TURNS_PER_WIDTH;
use gallery::surfels::SURFEL_ROWS;
use lights_mac::{Host, HostEvent};
use objc2_app_kit::{NSApplicationActivationPolicy, NSEvent, NSEventModifierFlags, NSEventType};
use objc2_foundation::{MainThreadMarker, NSPoint, NSSize, NSString};

const USAGE: &str = "\
lights-gallery                          a window with a scene selector, a caption and a frame time
lights-gallery --probe <DIR> [WxH]      drive the same window by script and write DIR/report.tsv";

const WINDOW_POINTS: (f64, f64) = (960.0, 600.0);
const PROBE_POINTS: (f64, f64) = (480.0, 320.0);
const RESIZE_BY: (f64, f64) = (80.0, 40.0);
const ESCAPE: u16 = 53;
const IDLE_WAIT_SECONDS: f64 = 0.5;
const SETTLE_SECONDS: f64 = 0.05;
const DRAG_FROM: (f64, f64) = (100.0, 80.0);
const DRAG_BY: f64 = 120.0;
const LIGHT_AT: (f64, f64) = (60.0, 50.0);

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

struct Shown {
    frame_pixels: (u32, u32),
    render_ms: f64,
    fill_ms: f64,
    cpu_ms: f64,
    shown_ms: f64,
    bytes: Vec<u8>,
}

struct App {
    scenes: Vec<Box<dyn Scene>>,
    selected: usize,
    chrome: Chrome,
    stale: bool,
    open: bool,
}

impl App {
    fn new(host: &Host, mtm: MainThreadMarker) -> Outcome<Self> {
        let scenes = gallery::scenes()?;
        let names: Vec<&str> = scenes.iter().map(|s| s.name()).collect();
        let chrome = Chrome::install(host, mtm, &names)?;
        chrome.show(0, scenes[0].caption());
        Ok(Self {
            scenes,
            selected: 0,
            chrome,
            stale: true,
            open: true,
        })
    }

    fn select(&mut self, index: usize) {
        if index >= self.scenes.len() || index == self.selected {
            return;
        }
        self.selected = index;
        self.chrome.show(index, self.scenes[index].caption());
        self.stale = true;
    }

    fn apply(&mut self, event: &HostEvent) {
        match event {
            HostEvent::CloseRequested | HostEvent::KeyPressed { code: ESCAPE, .. } => self.open = false,
            HostEvent::KeyPressed { characters, .. } => match characters.parse::<usize>() {
                Ok(number) if number >= 1 => self.select(number - 1),
                Ok(_) | Err(_) => self.stale |= self.scenes[self.selected].key(characters),
            },
            HostEvent::Resized { .. } => self.stale = true,
            HostEvent::PointerPressed { .. } | HostEvent::PointerMoved { .. } | HostEvent::PointerReleased { .. } => {
                self.stale |= self.scenes[self.selected].pointer(event)
            }
        }
    }

    fn step(&mut self, host: &mut Host, wait: f64, read_back: bool) -> Outcome<(Vec<HostEvent>, Option<Shown>)> {
        let events = host.pump(wait);
        for event in &events {
            self.apply(event);
        }
        if let Some(clicked) = self.chrome.selected() {
            self.select(clicked);
        }
        Ok((events, self.frame(host, read_back)?))
    }

    fn frame(&mut self, host: &mut Host, read_back: bool) -> Outcome<Option<Shown>> {
        let (width, height) = host.pixel_size();
        if !self.stale || !self.open || width == 0 || height == 0 {
            return Ok(None);
        }
        let begun = Instant::now();
        let (w, h) = self.scenes[self.selected].render(width, height)?;
        let rendered = begun.elapsed();
        let scene = &self.scenes[self.selected];
        let mut filled = Duration::ZERO;
        let fill = |bytes: &mut [u8], stride: usize| {
            let started = Instant::now();
            let outcome = scene.fill(bytes, stride);
            filled = started.elapsed();
            outcome
        };
        let bytes = if read_back {
            host.present_and_read_back(w, h, fill)?
        } else {
            host.present_filled(w, h, fill)?;
            Vec::new()
        };
        let shown_ms = millis(begun.elapsed());
        let cpu_ms = millis(rendered + filled);
        self.chrome.set_status(&format!(
            "{} · cpu {cpu_ms:.2} ms · shown after {shown_ms:.2} ms",
            scene.status()
        ));
        self.stale = false;
        Ok(Some(Shown {
            frame_pixels: (w, h),
            render_ms: millis(rendered),
            fill_ms: millis(filled),
            cpu_ms,
            shown_ms,
            bytes,
        }))
    }
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1.0e3
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match args.as_slice() {
        [] => run(),
        [flag, dir] if flag == "--probe" => probe(Path::new(dir), PROBE_POINTS),
        [flag, dir, size] if flag == "--probe" => match parse_points(size) {
            Some(points) => probe(Path::new(dir), points),
            None => Err(format!("{size} is not WIDTHxHEIGHT in points").into()),
        },
        [..] => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lights-gallery: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_points(text: &str) -> Option<(f64, f64)> {
    let (w, h) = text.split_once('x')?;
    let (w, h): (f64, f64) = (w.parse().ok()?, h.parse().ok()?);
    (w > 0.0 && h > CHROME_POINTS).then_some((w, h))
}

fn run() -> Outcome<()> {
    let mtm = MainThreadMarker::new().ok_or("lights-gallery must start on the main thread")?;
    let mut host = Host::open(
        mtm,
        "lights gallery - keys 1 to 3 choose a scene",
        WINDOW_POINTS.0,
        WINDOW_POINTS.1,
        NSApplicationActivationPolicy::Regular,
    )?;
    unsafe { host.app().activate() };
    let mut app = App::new(&host, mtm)?;
    app.frame(&mut host, false)?;
    while app.open {
        app.step(&mut host, IDLE_WAIT_SECONDS, false)?;
    }
    Ok(())
}

struct Probe<'a> {
    dir: &'a Path,
    report: String,
}

impl Probe<'_> {
    fn line(&mut self, key: &str, value: impl Display) {
        self.report.push_str(&format!("{key}\t{value}\n"));
    }

    fn record(&mut self, name: &str, app: &App, host: &Host, stepped: (Vec<HostEvent>, Option<Shown>)) -> Outcome<()> {
        let (events, shown) = stepped;
        let scene = &app.scenes[app.selected];
        let events: Vec<String> = events.iter().map(|e| format!("{e:?}")).collect();
        self.line(&format!("{name}.events"), events.join(" | "));
        self.line(&format!("{name}.scene"), scene.name());
        self.line(&format!("{name}.selected"), app.selected);
        self.line(
            &format!("{name}.segment"),
            app.chrome.selected().map_or("none".to_string(), |s| s.to_string()),
        );
        self.line(&format!("{name}.caption"), app.chrome.caption_text());
        self.line(&format!("{name}.status"), app.chrome.status_text());
        self.line(&format!("{name}.caption_rect"), rect(app.chrome.caption_rect()));
        self.line(&format!("{name}.render_rect"), rect(app.chrome.render_rect()));
        let (w, h) = host.pixel_size();
        self.line(&format!("{name}.render_pixels"), format!("{w}x{h}"));
        for (key, value) in scene.state() {
            self.line(&format!("{name}.state.{key}"), value);
        }
        self.line(&format!("{name}.presented"), shown.is_some());
        if let Some(shown) = shown {
            let (fw, fh) = shown.frame_pixels;
            self.line(&format!("{name}.frame_pixels"), format!("{fw}x{fh}"));
            self.line(&format!("{name}.render_ms"), shown.render_ms);
            self.line(&format!("{name}.fill_ms"), shown.fill_ms);
            self.line(&format!("{name}.cpu_ms"), shown.cpu_ms);
            self.line(&format!("{name}.shown_ms"), shown.shown_ms);
            std::fs::write(self.dir.join(format!("{name}.bgra")), &shown.bytes)?;
        }
        Ok(())
    }
}

fn rect(r: objc2_foundation::NSRect) -> String {
    format!("{} {} {} {}", r.origin.x, r.origin.y, r.size.width, r.size.height)
}

fn press(host: &Host, chrome: &Chrome, kind: NSEventType, at: NSPoint) -> Outcome<()> {
    if !chrome.lands_on_render(at) {
        return Err(format!("a press at {at:?} in the window would not land on the render area").into());
    }
    let number = unsafe { host.window().windowNumber() };
    let event = unsafe {
        NSEvent::mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure(
            kind,
            at,
            NSEventModifierFlags::empty(),
            0.0,
            number,
            None,
            0,
            1,
            1.0,
        )
    }
    .ok_or("AppKit made no mouse event")?;
    unsafe { host.app().sendEvent(&event) };
    Ok(())
}

fn key(host: &Host, text: &str) -> Outcome<()> {
    let number = unsafe { host.window().windowNumber() };
    let characters = NSString::from_str(text);
    let event = unsafe {
        NSEvent::keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode(
            NSEventType::KeyDown,
            NSPoint::new(0.0, 0.0),
            NSEventModifierFlags::empty(),
            0.0,
            number,
            None,
            &characters,
            &characters,
            false,
            0,
        )
    }
    .ok_or("AppKit made no key event")?;
    host.app().postEvent_atStart(&event, false);
    Ok(())
}

fn probe(dir: &Path, points: (f64, f64)) -> Outcome<()> {
    let mtm = MainThreadMarker::new().ok_or("the probe must start on the main thread")?;
    std::fs::create_dir_all(dir)?;
    let mut host = Host::open(
        mtm,
        "lights gallery probe",
        points.0,
        points.1,
        NSApplicationActivationPolicy::Accessory,
    )?;
    let mut app = App::new(&host, mtm)?;
    let mut probe = Probe {
        dir,
        report: String::new(),
    };
    let names: Vec<&str> = app.scenes.iter().map(|s| s.name()).collect();
    probe.line("scale", host.scale_factor());
    probe.line("window_points", format!("{}x{}", points.0, points.1));
    probe.line("chrome_points", CHROME_POINTS);
    probe.line("scenes", names.join(" | "));
    probe.line("segments", app.chrome.labels().join(" | "));
    probe.line("turns_per_width", TURNS_PER_WIDTH);
    probe.line("surfel_rows", SURFEL_ROWS);
    probe.line("drag_from_points", format!("{} {}", DRAG_FROM.0, DRAG_FROM.1));
    probe.line("drag_by_points", DRAG_BY);
    probe.line("light_at_points", format!("{} {}", LIGHT_AT.0, LIGHT_AT.1));
    for scene in &app.scenes {
        probe.line(&format!("caption_of.{}", scene.name()), scene.caption());
    }

    let first = app.frame(&mut host, true)?;
    probe.record("open", &app, &host, (Vec::new(), first))?;

    let area = app.chrome.render_rect();
    let at = |x: f64, y: f64| NSPoint::new(area.origin.x + x, area.origin.y + area.size.height - y);
    let (x0, y0) = DRAG_FROM;
    for (kind, point) in [
        (NSEventType::LeftMouseDown, at(x0, y0)),
        (NSEventType::LeftMouseDragged, at(x0 + DRAG_BY, y0)),
        (NSEventType::LeftMouseUp, at(x0 + DRAG_BY, y0)),
    ] {
        press(&host, &app.chrome, kind, point)?;
    }
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("dragged", &app, &host, stepped)?;

    key(&host, "2")?;
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("methods", &app, &host, stepped)?;

    key(&host, ORACLE_KEY)?;
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("oracle", &app, &host, stepped)?;

    app.chrome.choose(2);
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("surfels", &app, &host, stepped)?;

    for kind in [NSEventType::LeftMouseDown, NSEventType::LeftMouseUp] {
        press(&host, &app.chrome, kind, at(LIGHT_AT.0, LIGHT_AT.1))?;
    }
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("lit", &app, &host, stepped)?;

    let resized = (points.0 + RESIZE_BY.0, points.1 + RESIZE_BY.1);
    host.window().setContentSize(NSSize::new(resized.0, resized.1));
    probe.line("resized_points", format!("{}x{}", resized.0, resized.1));
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("resized", &app, &host, stepped)?;

    key(&host, "1")?;
    let stepped = app.step(&mut host, SETTLE_SECONDS, true)?;
    probe.record("back", &app, &host, stepped)?;

    unsafe { host.window().performClose(None) };
    let (events, _) = app.step(&mut host, SETTLE_SECONDS, true)?;
    let events: Vec<String> = events.iter().map(|e| format!("{e:?}")).collect();
    probe.line("close.events", events.join(" | "));
    probe.line("close.open", app.open);

    std::fs::write(dir.join("report.tsv"), probe.report)?;
    Ok(())
}
