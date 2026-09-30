pub mod chrome;
pub mod methods;
pub mod orbit;
pub mod surfels;

use lights::{Gaussian, KernelFamily, MethodId, RenderMethod, Renderer, Rgb, Scene as Atoms, SharedSettings};
use lights_mac::{HostError, HostEvent};

pub trait Scene {
    fn name(&self) -> &'static str;
    fn caption(&self) -> &'static str;
    fn status(&self) -> String;
    fn state(&self) -> Vec<(&'static str, String)>;
    fn render(&mut self, width: u32, height: u32) -> lights::Result<(u32, u32)>;
    fn fill(&self, bytes: &mut [u8], stride: usize) -> Result<(), HostError>;
    fn pointer(&mut self, event: &HostEvent) -> bool;
    fn key(&mut self, characters: &str) -> bool;
}

pub fn scenes() -> lights::Result<Vec<Box<dyn Scene>>> {
    Ok(vec![
        Box::new(orbit::Orbit::new()),
        Box::new(methods::Methods::new()?),
        Box::new(surfels::Surfels::new()),
    ])
}

pub fn settings() -> SharedSettings<f32> {
    let mut base = SharedSettings::<f64>::contract();
    base.background = Rgb::new(0.04, 0.05, 0.07);
    SharedSettings::<f32>::from_f64(&base)
}

pub fn no_frame() -> HostError {
    HostError::FrameLength {
        width: 0,
        height: 0,
        pixels: 0,
    }
}

type Key = (u32, u32, usize);

#[derive(Default)]
pub struct Held {
    renderer: Option<(Key, Renderer<f32, Gaussian>)>,
}

impl Held {
    pub fn at(
        &mut self,
        key: Key,
        method: &RenderMethod,
        settings: &SharedSettings<f32>,
        atoms: &Atoms<f32>,
    ) -> lights::Result<&mut Renderer<f32, Gaussian>> {
        let held = match self.renderer.take() {
            Some((kept, renderer)) if kept == key => self.renderer.insert((kept, renderer)),
            Some(_) | None => {
                let shape = KernelFamily::Gaussian.shape();
                let renderer = Renderer::new(method, settings, &shape, key.0, key.1, atoms)?;
                self.renderer.insert((key, renderer))
            }
        };
        Ok(&mut held.1)
    }

    pub fn method(&self) -> Option<MethodId> {
        self.renderer.as_ref().map(|(_, renderer)| renderer.method())
    }

    pub fn fill(&self, background: Rgb<f32>, bytes: &mut [u8], stride: usize) -> Result<(), HostError> {
        let Some((_, renderer)) = &self.renderer else {
            return Err(no_frame());
        };
        Ok(renderer.framebuffer().resolve_bgra8(background, bytes, stride)?)
    }
}
