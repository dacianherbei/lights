use lights::scene::showcase::{demo_object, still_camera};
use lights::{Camera, LightsError, MethodId, RenderMethod, Scene as Atoms, SharedSettings};
use lights_mac::{HostError, HostEvent};

use super::{Held, Scene};

pub const ORACLE_KEY: &str = "r";

pub struct Methods {
    methods: Vec<RenderMethod>,
    shown: usize,
    atoms: Atoms<f32>,
    settings: SharedSettings<f32>,
    held: Held,
}

impl Methods {
    pub fn new() -> lights::Result<Self> {
        let methods = RenderMethod::implemented();
        if methods.is_empty() {
            return Err(LightsError::MethodNotImplemented {
                method: MethodId::SplatRaster,
            });
        }
        let shown = methods
            .iter()
            .position(|m| m.id() == MethodId::SplatRaster)
            .unwrap_or(0);
        Ok(Self {
            methods,
            shown,
            atoms: Atoms::from_f64(&demo_object()),
            settings: super::settings(),
            held: Held::default(),
        })
    }

    fn id(&self) -> MethodId {
        self.methods[self.shown].id()
    }
}

impl Scene for Methods {
    fn name(&self) -> &'static str {
        "methods"
    }

    fn caption(&self) -> &'static str {
        "One view rendered two ways. Splat is shown and is the interactive default; press R for \
         the ray caster, the correctness oracle, and R again to come back. The ray caster is off \
         until asked for: L-123 measured it at 15.590 ms per 1080p frame at 10k atoms against \
         splat's 4.858 ms. The agreement harness holds the two within a per-atom bound at every \
         pixel; here the agreement is something to look at."
    }

    fn status(&self) -> String {
        match self.id() {
            MethodId::SplatRaster => format!("{}, the interactive default", self.id().name()),
            MethodId::RayCast => format!("{}, the oracle and not an interactive mode", self.id().name()),
        }
    }

    fn state(&self) -> Vec<(&'static str, String)> {
        let rendered = self.held.method().map_or("nothing", MethodId::name);
        vec![("method", self.id().name().to_string()), ("rendered_by", rendered.to_string())]
    }

    fn render(&mut self, width: u32, height: u32) -> lights::Result<(u32, u32)> {
        let method = &self.methods[self.shown];
        let renderer = self.held.at((width, height, self.shown), method, &self.settings, &self.atoms)?;
        let spec = still_camera(width, height);
        renderer.render_frame(&Camera::<f32>::from_spec(&spec, self.settings.near_plane)?)?;
        Ok((width, height))
    }

    fn fill(&self, bytes: &mut [u8], stride: usize) -> Result<(), HostError> {
        self.held.fill(self.settings.background, bytes, stride)
    }

    fn pointer(&mut self, _event: &HostEvent) -> bool {
        false
    }

    fn key(&mut self, characters: &str) -> bool {
        if !characters.eq_ignore_ascii_case(ORACLE_KEY) || self.methods.len() < 2 {
            return false;
        }
        self.shown = (self.shown + 1) % self.methods.len();
        true
    }
}
