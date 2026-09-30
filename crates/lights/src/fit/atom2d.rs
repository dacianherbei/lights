use crate::camera::CameraSpec;
use crate::error::Result;
use crate::kernel::CurveShape;
use crate::numerics::quaternion::Quat;
use crate::numerics::vector::{Rgb, Vec2, Vec3};
use crate::scene::{Atom, Scene};
use crate::settings::PixelConvention;

pub const Z0: f64 = 64.0;
pub const DZ: f64 = 1.0 / 64.0;
pub const ORTHOGRAPHIC_FOV_DEGREES: f64 = 45.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Atom2d {
    pub centre: Vec2<f64>,
    pub log_scale: Vec2<f64>,
    pub angle: f64,
    pub opacity: f64,
    pub colour: Rgb<f64>,
}

impl Atom2d {
    pub fn from_sigma(
        centre: Vec2<f64>,
        sigma: Vec2<f64>,
        angle: f64,
        opacity: f64,
        colour: Rgb<f64>,
    ) -> Self {
        Self {
            centre,
            log_scale: Vec2::new(sigma.x.ln(), sigma.y.ln()),
            angle,
            opacity,
            colour,
        }
    }

    pub fn sigma(&self) -> Vec2<f64> {
        Vec2::new(self.log_scale.x.exp(), self.log_scale.y.exp())
    }

    pub fn is_finite(&self) -> bool {
        self.centre.is_finite()
            && self.log_scale.is_finite()
            && self.angle.is_finite()
            && self.opacity.is_finite()
            && self.colour.is_finite()
    }

    pub fn long_axis(&self) -> Vec2<f64> {
        let (s, c) = self.angle.sin_cos();
        if self.log_scale.x >= self.log_scale.y {
            Vec2::new(c, s)
        } else {
            Vec2::new(-s, c)
        }
    }

    pub fn anisotropy(&self) -> f64 {
        (self.log_scale.x - self.log_scale.y).abs().exp()
    }

    pub fn bits(&self) -> [u64; 9] {
        [
            self.centre.x.to_bits(),
            self.centre.y.to_bits(),
            self.log_scale.x.to_bits(),
            self.log_scale.y.to_bits(),
            self.angle.to_bits(),
            self.opacity.to_bits(),
            self.colour.x.to_bits(),
            self.colour.y.to_bits(),
            self.colour.z.to_bits(),
        ]
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShapedAtom2d {
    pub atom: Atom2d,
    pub shape: CurveShape,
}

impl ShapedAtom2d {
    pub fn new(atom: Atom2d, shape: CurveShape) -> Self {
        Self { atom, shape }
    }
}

pub fn pixel_centre(px: u32, py: u32) -> Vec2<f64> {
    Vec2::new(px as f64 + 0.5, py as f64 + 0.5)
}

pub fn image_to_world(p: Vec2<f64>, viewport: (u32, u32)) -> Vec2<f64> {
    Vec2::new(
        p.x - viewport.0 as f64 * 0.5,
        p.y - viewport.1 as f64 * 0.5,
    )
}

pub fn world_to_image(p: Vec2<f64>, viewport: (u32, u32)) -> Vec2<f64> {
    Vec2::new(
        p.x + viewport.0 as f64 * 0.5,
        p.y + viewport.1 as f64 * 0.5,
    )
}

pub fn camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        ORTHOGRAPHIC_FOV_DEGREES,
        width,
        height,
    )
    .orthographic(height as f64 * 0.5)
    .with_convention(PixelConvention::CentreSample)
}

pub fn shifted_camera(width: u32, height: u32, shift: Vec2<f64>) -> CameraSpec {
    camera(width, height).with_intrinsics(
        [1.0, 1.0],
        [
            width as f64 * 0.5 + shift.x,
            height as f64 * 0.5 + shift.y,
        ],
    )
}

pub fn depth_of(index: usize) -> f64 {
    Z0 - index as f64 * DZ
}

pub fn to_atom(a: &Atom2d, index: usize, viewport: (u32, u32)) -> Atom<f64> {
    let world = image_to_world(a.centre, viewport);
    let sigma = a.sigma();
    let sz = sigma.x.min(sigma.y);
    let rotation = match Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), a.angle) {
        Some(q) => q,
        None => Quat::identity(),
    };
    Atom::new(
        Vec3::new(world.x, world.y, depth_of(index)),
        rotation,
        Vec3::new(sigma.x, sigma.y, sz),
        a.opacity,
        a.colour,
    )
}

pub fn to_scene(atoms: &[Atom2d], viewport: (u32, u32), scale_floor: f64) -> Result<Scene<f64>> {
    let mut scene = Scene::<f64>::new();
    for (index, a) in atoms.iter().enumerate() {
        scene.push(to_atom(a, index, viewport), scale_floor)?;
    }
    Ok(scene)
}

pub fn to_shaped_scene(
    atoms: &[ShapedAtom2d],
    viewport: (u32, u32),
    scale_floor: f64,
) -> Result<(Scene<f64>, Vec<CurveShape>)> {
    let mut scene = Scene::<f64>::new();
    let mut shapes = Vec::with_capacity(atoms.len());
    for (index, a) in atoms.iter().enumerate() {
        scene.push(to_atom(&a.atom, index, viewport), scale_floor)?;
        shapes.push(a.shape);
    }
    Ok((scene, shapes))
}
