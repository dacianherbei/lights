use lights::scene::showcase::{demo_object, orbit_camera};
use lights::{Camera, RenderMethod, Scene as Atoms, SharedSettings};
use lights_mac::{HostError, HostEvent};

use super::{Held, Scene};

pub const TURNS_PER_WIDTH: f64 = 1.0;

pub struct Orbit {
    atoms: Atoms<f32>,
    settings: SharedSettings<f32>,
    held: Held,
    width: u32,
    turn: f64,
    grab: Option<f64>,
}

impl Orbit {
    pub fn new() -> Self {
        Self {
            atoms: Atoms::from_f64(&demo_object()),
            settings: super::settings(),
            held: Held::default(),
            width: 0,
            turn: 0.0,
            grab: None,
        }
    }

    fn drag_to(&mut self, x: f64) -> bool {
        let Some(from) = self.grab.replace(x) else {
            return false;
        };
        if self.width == 0 || x == from {
            return false;
        }
        self.turn += (x - from) / f64::from(self.width) * TURNS_PER_WIDTH;
        true
    }
}

impl Scene for Orbit {
    fn name(&self) -> &'static str {
        "orbit"
    }

    fn caption(&self) -> &'static str {
        "Drag across the frame to turn the camera around the object: one full turn per width of \
         the frame. The ring is 36 atoms, the core one, and each of the four plates one flattened \
         atom, drawn by splat in f32 - the object and the camera path of the demo page's orbit."
    }

    fn status(&self) -> String {
        format!("turn {:.3}", self.turn)
    }

    fn state(&self) -> Vec<(&'static str, String)> {
        vec![("turn", format!("{}", self.turn))]
    }

    fn render(&mut self, width: u32, height: u32) -> lights::Result<(u32, u32)> {
        self.width = width;
        let method = RenderMethod::splat_contract();
        let renderer = self.held.at((width, height, 0), &method, &self.settings, &self.atoms)?;
        let spec = orbit_camera(width, height, self.turn);
        renderer.render_frame(&Camera::<f32>::from_spec(&spec, self.settings.near_plane)?)?;
        Ok((width, height))
    }

    fn fill(&self, bytes: &mut [u8], stride: usize) -> Result<(), HostError> {
        self.held.fill(self.settings.background, bytes, stride)
    }

    fn pointer(&mut self, event: &HostEvent) -> bool {
        match *event {
            HostEvent::PointerPressed { x, .. } => {
                self.grab = Some(x);
                false
            }
            HostEvent::PointerMoved { x, .. } => self.drag_to(x),
            HostEvent::PointerReleased { x, .. } => {
                let turned = self.drag_to(x);
                self.grab = None;
                turned
            }
            HostEvent::KeyPressed { .. } | HostEvent::Resized { .. } | HostEvent::CloseRequested => false,
        }
    }

    fn key(&mut self, _characters: &str) -> bool {
        false
    }
}
