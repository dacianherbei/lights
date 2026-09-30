mod common;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use common::Stage;
use lights_mac::{Host, HostError, HostEvent, write_bgra};
use objc2_app_kit::{NSApplicationActivationPolicy, NSEvent, NSEventModifierFlags, NSEventType};
use objc2_foundation::{MainThreadMarker, NSPoint, NSSize, NSString};

const OPEN_POINTS: (f64, f64) = (320.0, 200.0);
const RESIZED_POINTS: (f64, f64) = (400.0, 260.0);
const PRESS_POINT: (f64, f64) = (40.0, 50.0);
const DRAG_POINT: (f64, f64) = (120.0, 90.0);
const KEYS: [(u16, &str); 2] = [(0, "a"), (53, "\u{1b}")];
const SETTLE_SECONDS: f64 = 0.05;
const TIMED_PRESENTS: usize = 20;
const SPACING_SECONDS: f64 = 0.04;
const TIMED_PIXELS: (u32, u32) = (1920, 1080);
const HANDOFF_REPS: usize = 20;
const RUN_FRAMES: usize = 20;
const RUN_ORDER: [bool; 4] = [false, true, true, false];
const POISON: u8 = 0x5A;

type Outcome = Result<(), Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    let Some(dir) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: lights-mac-probe <output directory>");
        return ExitCode::from(2);
    };
    match run(&dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lights-mac-probe: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(dir: &Path) -> Outcome {
    let mtm = MainThreadMarker::new().ok_or("the probe must start on the main thread")?;
    std::fs::create_dir_all(dir)?;
    let mut report = String::new();
    let mut host = Host::open(
        mtm,
        "lights-mac probe",
        OPEN_POINTS.0,
        OPEN_POINTS.1,
        NSApplicationActivationPolicy::Accessory,
    )?;
    let opened = host.pump(SETTLE_SECONDS);
    let screen = host.window().screen().map(|s| s.backingScaleFactor());
    line(&mut report, "open_points", format!("{}x{}", OPEN_POINTS.0, OPEN_POINTS.1));
    line(&mut report, "scale", host.scale_factor());
    line(&mut report, "screen_scale", screen.map_or("none".to_string(), |s| s.to_string()));
    line(&mut report, "open_events", events(&opened));
    shown(&mut host, dir, "open", &mut report)?;

    host.window().setContentSize(NSSize::new(RESIZED_POINTS.0, RESIZED_POINTS.1));
    let resized = host.pump(SETTLE_SECONDS);
    line(&mut report, "resized_points", format!("{}x{}", RESIZED_POINTS.0, RESIZED_POINTS.1));
    line(&mut report, "resize_events", events(&resized));
    shown(&mut host, dir, "resized", &mut report)?;

    let number = unsafe { host.window().windowNumber() };
    let mouse = [
        (NSEventType::LeftMouseDown, PRESS_POINT),
        (NSEventType::LeftMouseDragged, DRAG_POINT),
        (NSEventType::LeftMouseUp, DRAG_POINT),
    ];
    for (kind, (x, y)) in mouse {
        let at = NSPoint::new(x, RESIZED_POINTS.1 - y);
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
    }
    for (code, text) in KEYS {
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
                code,
            )
        }
        .ok_or("AppKit made no key event")?;
        host.app().postEvent_atStart(&event, false);
    }
    let input = host.pump(SETTLE_SECONDS);
    line(
        &mut report,
        "input_points",
        format!("press {PRESS_POINT:?} drag {DRAG_POINT:?} release {DRAG_POINT:?}"),
    );
    line(&mut report, "input_events", events(&input));

    unsafe { host.window().performClose(None) };
    let close = host.pump(SETTLE_SECONDS);
    line(&mut report, "close_events", events(&close));
    line(&mut report, "visible_after_close_request", host.window().isVisible());

    timed(&mut host, &mut report)?;
    handoff(&mut host, &mut report)?;
    std::fs::write(dir.join("report.tsv"), report)?;
    Ok(())
}

fn shown(host: &mut Host, dir: &Path, name: &str, report: &mut String) -> Outcome {
    let (width, height) = host.pixel_size();
    let mut stage = Stage::new(width, height)?;
    stage.render()?;
    let frame = stage.resolve().to_clamped_u8();
    let (w, h) = stage.size();
    host.present_filled(w, h, |bytes, _| {
        bytes.fill(POISON);
        Ok(())
    })?;
    let fused = host.present_and_read_back(w, h, |bytes, stride| Ok(stage.fill(bytes, stride)?))?;
    let read = host.present_and_read_back(w, h, |bytes, stride| write_bgra(&frame, bytes, stride))?;
    line(report, &format!("{name}_pixels"), format!("{width}x{height}"));
    line(report, &format!("{name}_frame"), format!("{w}x{h}"));
    std::fs::write(dir.join(format!("{name}.rgb")), frame.pixels.as_flattened())?;
    std::fs::write(dir.join(format!("{name}.bgra")), read)?;
    std::fs::write(dir.join(format!("{name}.fused")), fused)?;
    Ok(())
}

fn timed(host: &mut Host, report: &mut String) -> Outcome {
    let scale = host.scale_factor();
    host.window().setContentSize(NSSize::new(
        f64::from(TIMED_PIXELS.0) / scale,
        f64::from(TIMED_PIXELS.1) / scale,
    ));
    let _ = host.pump(SETTLE_SECONDS);
    let (width, height) = host.pixel_size();
    let mut stage = Stage::new(width, height)?;
    let started = Instant::now();
    stage.render()?;
    let frame = stage.resolve().to_clamped_u8();
    let render_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let mut paced = Vec::with_capacity(TIMED_PRESENTS);
    let mut spaced = Vec::with_capacity(TIMED_PRESENTS);
    for _ in 0..TIMED_PRESENTS {
        let started = Instant::now();
        host.present(&frame)?;
        paced.push(started.elapsed().as_secs_f64() * 1.0e3);
    }
    for _ in 0..TIMED_PRESENTS {
        let _ = host.pump(SPACING_SECONDS);
        let started = Instant::now();
        host.present(&frame)?;
        spaced.push(started.elapsed().as_secs_f64() * 1.0e3);
    }
    line(report, "timed_pixels", format!("{width}x{height}"));
    line(report, "timed_render_ms", format!("{render_ms:.3}"));
    line(report, "timed_present_back_to_back_ms", spread(&mut paced));
    line(report, "timed_present_spaced_ms", spread(&mut spaced));
    Ok(())
}

fn handoff(host: &mut Host, report: &mut String) -> Outcome {
    let (width, height) = host.pixel_size();
    let mut stage = Stage::new(width, height)?;
    let mut terms: [Vec<f64>; 5] = Default::default();
    for rep in 0..HANDOFF_REPS {
        let started = Instant::now();
        stage.render()?;
        terms[0].push(millis(started));
        let three = |host: &mut Host, terms: &mut [Vec<f64>; 5]| -> Result<(), HostError> {
            let started = Instant::now();
            let float = stage.resolve();
            terms[1].push(millis(started));
            let started = Instant::now();
            let image = float.to_clamped_u8();
            terms[2].push(millis(started));
            host.present_filled(width, height, |bytes, stride| {
                let begun = Instant::now();
                let written = write_bgra(&image, bytes, stride);
                terms[3].push(millis(begun));
                written
            })
        };
        let fused = |host: &mut Host, terms: &mut [Vec<f64>; 5]| -> Result<(), HostError> {
            host.present_filled(width, height, |bytes, stride| {
                let begun = Instant::now();
                let written = stage.fill(bytes, stride);
                terms[4].push(millis(begun));
                Ok(written?)
            })
        };
        if rep % 2 == 0 {
            three(host, &mut terms)?;
            fused(host, &mut terms)?;
        } else {
            fused(host, &mut terms)?;
            three(host, &mut terms)?;
        }
    }
    let frame = stage.resolve().to_clamped_u8();
    let expanded = host.present_and_read_back(width, height, |bytes, stride| write_bgra(&frame, bytes, stride))?;
    let filled = host.present_and_read_back(width, height, |bytes, stride| Ok(stage.fill(bytes, stride)?))?;
    if expanded != filled {
        return Err(format!(
            "the two timed arms show different bytes at {width}x{height}, first at {:?}",
            expanded.iter().zip(&filled).position(|(a, b)| a != b)
        )
        .into());
    }
    let mut three: Vec<f64> = (0..HANDOFF_REPS).map(|i| terms[1][i] + terms[2][i] + terms[3][i]).collect();
    let names = [
        "handoff_render_ms",
        "handoff_resolve_float_ms",
        "handoff_to_clamped_u8_ms",
        "handoff_expand_ms",
        "handoff_fused_ms",
    ];
    line(report, "handoff_pixels", format!("{width}x{height}"));
    for (name, samples) in names.iter().zip(terms.iter_mut()) {
        line(report, name, spread(samples));
    }
    line(report, "handoff_three_pass_ms", spread(&mut three));
    line(report, "handoff_arms_agree", "true");

    let mut runs: [Vec<f64>; 4] = Default::default();
    let [fused_cpu, fused_shown, three_cpu, three_shown] = &mut runs;
    for fused in RUN_ORDER {
        if fused {
            free_running(host, &mut stage, true, fused_cpu, fused_shown)?;
        } else {
            free_running(host, &mut stage, false, three_cpu, three_shown)?;
        }
    }
    let names = [
        "frame_fused_cpu_ms",
        "frame_fused_interval_ms",
        "frame_three_pass_cpu_ms",
        "frame_three_pass_interval_ms",
    ];
    for (name, samples) in names.iter().zip(runs.iter_mut()) {
        line(report, name, spread(samples));
    }
    Ok(())
}

fn free_running(
    host: &mut Host,
    stage: &mut Stage,
    fused: bool,
    cpu: &mut Vec<f64>,
    shown: &mut Vec<f64>,
) -> Outcome {
    let (width, height) = stage.size();
    let mut last: Option<Instant> = None;
    for _ in 0..=RUN_FRAMES {
        let started = Instant::now();
        stage.render()?;
        let mut work = 0.0;
        if fused {
            host.present_filled(width, height, |bytes, stride| {
                let written = stage.fill(bytes, stride);
                work = millis(started);
                Ok(written?)
            })?;
        } else {
            let image = stage.resolve().to_clamped_u8();
            host.present_filled(width, height, |bytes, stride| {
                let written = write_bgra(&image, bytes, stride);
                work = millis(started);
                written
            })?;
        }
        let now = Instant::now();
        if let Some(previous) = last {
            cpu.push(work);
            shown.push(now.duration_since(previous).as_secs_f64() * 1.0e3);
        }
        last = Some(now);
    }
    Ok(())
}

fn millis(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1.0e3
}

fn spread(ms: &mut [f64]) -> String {
    ms.sort_by(f64::total_cmp);
    format!(
        "median {:.3} min {:.3} max {:.3} over {}",
        ms[ms.len() / 2],
        ms[0],
        ms[ms.len() - 1],
        ms.len()
    )
}

fn events(list: &[HostEvent]) -> String {
    list.iter().map(|e| format!("{e:?}")).collect::<Vec<_>>().join(" | ")
}

fn line(report: &mut String, key: &str, value: impl std::fmt::Display) {
    let _ = writeln!(report, "{key}\t{value}");
}
