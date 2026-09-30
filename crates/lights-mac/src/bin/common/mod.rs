#![allow(dead_code)]
use lights::{
    Camera, FloatImage, Gaussian, KernelFamily, RenderMethod, Renderer, Result, Rgb,
    SharedSettings, sphere_camera, spheres,
};

pub struct Stage {
    renderer: Renderer<f32, Gaussian>,
    settings: SharedSettings<f32>,
    width: u32,
    height: u32,
}

impl Stage {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let mut base = SharedSettings::<f64>::contract();
        base.background = Rgb::new(0.04, 0.05, 0.07);
        let settings = SharedSettings::<f32>::from_f64(&base);
        let renderer = Renderer::<f32, Gaussian>::new(
            &RenderMethod::splat_contract(),
            &settings,
            &KernelFamily::Gaussian.shape(),
            width,
            height,
            &spheres::<f32>(),
        )?;
        Ok(Self {
            renderer,
            settings,
            width,
            height,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn render(&mut self) -> Result<()> {
        let spec = sphere_camera(self.width, self.height);
        let camera = Camera::<f32>::from_spec(&spec, self.settings.near_plane)?;
        self.renderer.render_frame(&camera)
    }

    pub fn resolve(&self) -> FloatImage<f32> {
        self.renderer.resolve(self.settings.background)
    }

    pub fn fill(&self, bytes: &mut [u8], stride: usize) -> Result<()> {
        self.renderer.framebuffer().resolve_bgra8(self.settings.background, bytes, stride)
    }
}
