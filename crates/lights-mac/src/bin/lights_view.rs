mod common;

use std::process::ExitCode;
use std::time::Instant;

use common::Stage;
use lights_mac::{Host, HostEvent};
use objc2_app_kit::NSApplicationActivationPolicy;
use objc2_foundation::MainThreadMarker;

const ESCAPE: u16 = 53;
const IDLE_WAIT_SECONDS: f64 = 0.5;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lights-view: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mtm = MainThreadMarker::new().ok_or("lights-view must start on the main thread")?;
    let mut host = Host::open(mtm, "lights", 960.0, 540.0, NSApplicationActivationPolicy::Regular)?;
    unsafe { host.app().activate() };
    let (width, height) = host.pixel_size();
    let mut stage = Stage::new(width, height)?;
    let mut stale = true;
    let mut open = true;
    while open {
        if stale {
            let started = Instant::now();
            stage.render()?;
            let rendered = started.elapsed().as_secs_f64() * 1.0e3;
            let (w, h) = stage.size();
            host.present_filled(w, h, |bytes, stride| Ok(stage.fill(bytes, stride)?))?;
            println!(
                "frame {w}x{h} px at scale {}: rendered in {rendered:.3} ms, shown after {:.3} ms",
                host.scale_factor(),
                started.elapsed().as_secs_f64() * 1.0e3
            );
            stale = false;
        }
        for event in host.pump(IDLE_WAIT_SECONDS) {
            println!("{event:?}");
            match event {
                HostEvent::CloseRequested => open = false,
                HostEvent::KeyPressed { code: ESCAPE, .. } => open = false,
                HostEvent::Resized { width, height, .. } => {
                    if width > 0 && height > 0 && (width, height) != stage.size() {
                        stage = Stage::new(width, height)?;
                        stale = true;
                    }
                }
                HostEvent::KeyPressed { .. }
                | HostEvent::PointerPressed { .. }
                | HostEvent::PointerMoved { .. }
                | HostEvent::PointerReleased { .. } => {}
            }
        }
    }
    Ok(())
}
