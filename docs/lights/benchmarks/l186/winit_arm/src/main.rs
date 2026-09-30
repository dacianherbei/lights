use std::time::Duration;

use objc2::rc::Retained;
use objc2_app_kit::{NSApplication, NSEvent, NSEventModifierFlags, NSEventType, NSView, NSWindow};
use objc2_foundation::{MainThreadMarker, NSPoint, NSString};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const RESIZED_POINTS: (f64, f64) = (400.0, 260.0);
const PRESS_POINT: (f64, f64) = (40.0, 50.0);
const DRAG_POINT: (f64, f64) = (120.0, 90.0);
const KEYS: [(u16, &str); 2] = [(0, "a"), (53, "\u{1b}")];
const SETTLE: Duration = Duration::from_millis(50);

#[derive(Debug)]
enum Seen {
    Pressed { x: f64, y: f64 },
    Moved { x: f64, y: f64 },
    Released { x: f64, y: f64 },
    Key { physical: String, logical: String },
    Resized { width: u32, height: u32 },
    Scale(f64),
    Close,
}

struct Arm {
    made: bool,
    inject: Vec<Retained<NSEvent>>,
    close: Option<Retained<NSWindow>>,
    window: Option<Window>,
    cursor: (f64, f64),
    seen: Vec<Seen>,
}

impl ApplicationHandler for Arm {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.made {
            self.made = true;
            let attributes = Window::default_attributes()
                .with_title("l186 winit arm")
                .with_inner_size(LogicalSize::new(RESIZED_POINTS.0, RESIZED_POINTS.1));
            self.window = event_loop.create_window(attributes).ok();
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(mtm) = MainThreadMarker::new() {
            let app = NSApplication::sharedApplication(mtm);
            for event in self.inject.drain(..) {
                unsafe { app.sendEvent(&event) };
            }
        }
        if let Some(window) = self.close.take() {
            unsafe { window.performClose(None) };
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x, position.y);
                self.seen.push(Seen::Moved { x: position.x, y: position.y });
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                let (x, y) = self.cursor;
                self.seen.push(match state {
                    ElementState::Pressed => Seen::Pressed { x, y },
                    ElementState::Released => Seen::Released { x, y },
                });
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.seen.push(Seen::Key {
                    physical: format!("{:?}", event.physical_key),
                    logical: format!("{:?}", event.logical_key),
                });
            }
            WindowEvent::Resized(size) => self.seen.push(Seen::Resized { width: size.width, height: size.height }),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => self.seen.push(Seen::Scale(scale_factor)),
            WindowEvent::CloseRequested => self.seen.push(Seen::Close),
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mtm = MainThreadMarker::new().ok_or("main thread only")?;
    let between = std::env::var("L186_INJECT").as_deref() == Ok("between");
    println!("inject\t{}", if between { "between pumps" } else { "inside about_to_wait" });
    let mut event_loop = EventLoop::new()?;
    let mut arm = Arm { made: false, inject: Vec::new(), close: None, window: None, cursor: (0.0, 0.0), seen: Vec::new() };
    let _ = event_loop.pump_app_events(Some(SETTLE), &mut arm);
    let window = arm.window.take().ok_or("winit made no window")?;
    println!("scale\t{}", window.scale_factor());
    println!("inner_size\t{:?}", window.inner_size());
    let RawWindowHandle::AppKit(handle) = window.window_handle()?.as_raw() else {
        return Err("not an AppKit window".into());
    };
    let view = unsafe { Retained::retain(handle.ns_view.as_ptr().cast::<NSView>()) }.ok_or("no view")?;
    let ns_window = view.window().ok_or("the view has no window")?;
    let number = unsafe { ns_window.windowNumber() };
    let height = view.bounds().size.height;
    let app = NSApplication::sharedApplication(mtm);
    arm.seen.clear();
    for (kind, (x, y)) in [
        (NSEventType::LeftMouseDown, PRESS_POINT),
        (NSEventType::LeftMouseDragged, DRAG_POINT),
        (NSEventType::LeftMouseUp, DRAG_POINT),
    ] {
        let event = unsafe {
            NSEvent::mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure(
                kind, NSPoint::new(x, height - y), NSEventModifierFlags::empty(), 0.0, number, None, 0, 1, 1.0,
            )
        }
        .ok_or("no mouse event")?;
        if between {
            unsafe { app.sendEvent(&event) };
        } else {
            arm.inject.push(event);
        }
    }
    for (code, text) in KEYS {
        let characters = NSString::from_str(text);
        let event = unsafe {
            NSEvent::keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode(
                NSEventType::KeyDown, NSPoint::new(0.0, 0.0), NSEventModifierFlags::empty(), 0.0, number, None,
                &characters, &characters, false, code,
            )
        }
        .ok_or("no key event")?;
        app.postEvent_atStart(&event, false);
    }
    let _ = event_loop.pump_app_events(Some(SETTLE), &mut arm);
    println!("input_events\t{:?}", arm.seen);
    arm.seen.clear();
    let _ = window.request_inner_size(LogicalSize::new(RESIZED_POINTS.0 + 80.0, RESIZED_POINTS.1 + 40.0));
    let _ = event_loop.pump_app_events(Some(SETTLE), &mut arm);
    println!("resize_events\t{:?}", arm.seen);
    arm.seen.clear();
    if between {
        unsafe { ns_window.performClose(None) };
    } else {
        arm.close = Some(ns_window.clone());
    }
    let _ = event_loop.pump_app_events(Some(SETTLE), &mut arm);
    println!("close_events\t{:?}", arm.seen);
    Ok(())
}
