use lights::scene::showcase::{SURFEL_SHELL_COUNT, material_shell, shell_camera, surfel_shell};
use lights::{Camera, DirectionalLight, Image, LightsError, SurfelScene, Vec3, render_surfels};
use lights_mac::{HostError, HostEvent, write_bgra};

use super::Scene;

pub const SURFEL_ROWS: u32 = 135;
pub const FIRST_TOWARD: [f64; 2] = [0.7, -0.3];
const AMBIENT: f64 = 0.2;
const DIFFUSE: f64 = 0.8;
const NEAR: f64 = 1.0e-3;

pub struct Surfels {
    shell: SurfelScene<f64>,
    toward: [f64; 2],
    held: bool,
    area: (u32, u32),
    light: Vec3<f64>,
    frame: Option<Image>,
}

impl Surfels {
    pub fn new() -> Self {
        Self {
            shell: material_shell(&surfel_shell(SURFEL_SHELL_COUNT)),
            toward: FIRST_TOWARD,
            held: false,
            area: (0, 0),
            light: Vec3::new(0.0, 0.0, 0.0),
            frame: None,
        }
    }

    fn place(&mut self, x: f64, y: f64) -> bool {
        let (w, h) = (f64::from(self.area.0), f64::from(self.area.1));
        if w <= 0.0 || h <= 0.0 {
            return false;
        }
        let reach = 0.5 * w.min(h);
        let (u, v) = ((x - 0.5 * w) / reach, (y - 0.5 * h) / reach);
        let length = (u * u + v * v).sqrt();
        self.toward = if length > 1.0 { [u / length, v / length] } else { [u, v] };
        true
    }
}

impl Scene for Surfels {
    fn name(&self) -> &'static str {
        "surfels"
    }

    fn caption(&self) -> &'static str {
        "The 900-surfel shell. Each disc stores its normal and is lit by it; the body is glossy \
         and the cap emits, so the cap ignores the light. Press or drag to put the light where \
         the pointer is: the centre is the camera, the rim is side-on. The surfel pass tests \
         every surfel at every pixel, so this scene renders 135 rows and the window scales them."
    }

    fn status(&self) -> String {
        let [u, v] = self.toward;
        let (w, h) = self.frame.as_ref().map_or((0, 0), |f| (f.width, f.height));
        format!("light at ({u:+.2}, {v:+.2}) on the disc, rendered at {w}x{h}")
    }

    fn state(&self) -> Vec<(&'static str, String)> {
        let [u, v] = self.toward;
        let l = self.light;
        vec![
            ("toward", format!("{u} {v}")),
            ("light", format!("{} {} {}", l.x, l.y, l.z)),
        ]
    }

    fn render(&mut self, width: u32, height: u32) -> lights::Result<(u32, u32)> {
        self.area = (width, height);
        let rows = SURFEL_ROWS.min(height).max(1);
        let columns = ((f64::from(width) * f64::from(rows) / f64::from(height.max(1))).round() as u32).max(1);
        let camera = Camera::<f64>::from_spec(&shell_camera(columns, rows), NEAR)?;
        let [u, v] = self.toward;
        let depth = (1.0 - u * u - v * v).max(0.0).sqrt();
        let toward = camera.rotation.transpose().mul_vec(Vec3::new(u, v, -depth));
        let light = DirectionalLight::new(toward, AMBIENT, DIFFUSE).ok_or(LightsError::NonFinite {
            what: "the direction toward the light",
        })?;
        self.light = light.direction;
        let (colour, _) = render_surfels(&self.shell, &camera, Some(&light))?;
        self.frame = Some(colour.to_clamped_u8());
        Ok((columns, rows))
    }

    fn fill(&self, bytes: &mut [u8], stride: usize) -> Result<(), HostError> {
        let Some(frame) = &self.frame else {
            return Err(super::no_frame());
        };
        write_bgra(frame, bytes, stride)
    }

    fn pointer(&mut self, event: &HostEvent) -> bool {
        match *event {
            HostEvent::PointerPressed { x, y } => {
                self.held = true;
                self.place(x, y)
            }
            HostEvent::PointerMoved { x, y } => self.held && self.place(x, y),
            HostEvent::PointerReleased { x, y } => {
                let moved = self.held && self.place(x, y);
                self.held = false;
                moved
            }
            HostEvent::KeyPressed { .. } | HostEvent::Resized { .. } | HostEvent::CloseRequested => false,
        }
    }

    fn key(&mut self, _characters: &str) -> bool {
        false
    }
}
