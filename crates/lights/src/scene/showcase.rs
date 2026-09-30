use crate::camera::CameraSpec;
use crate::numerics::quaternion::Quat;
use crate::numerics::vector::{Rgb, Vec2, Vec3};
use crate::scene::{Atom, Scene};
use crate::settings::{MIN_SCALE, PixelConvention};
use crate::surfel::{Material, Surfel, SurfelScene};

pub const RING_ATOMS: usize = 36;
const OBJECT_CENTRE: [f64; 3] = [0.0, 0.0, 0.0];
pub const SURFEL_SHELL_COUNT: usize = 900;
const SURFEL_LAMP_CAP: f64 = 0.85;

fn hue(t: f64) -> Rgb<f64> {
    let x = t - t.floor();
    let r = ((6.0 * x - 3.0).abs() - 1.0).clamp(0.0, 1.0);
    let g = (2.0 - (6.0 * x - 2.0).abs()).clamp(0.0, 1.0);
    let b = (2.0 - (6.0 * x - 4.0).abs()).clamp(0.0, 1.0);
    Rgb::new(0.10 + 0.90 * r, 0.10 + 0.90 * g, 0.10 + 0.90 * b)
}

pub fn demo_object() -> Scene<f64> {
    let mut atoms: Vec<Atom<f64>> = Vec::new();
    for i in 0..RING_ATOMS {
        let t = i as f64 / RING_ATOMS as f64;
        let angle = t * std::f64::consts::TAU;
        let lift = 0.34 * (3.0 * angle).sin();
        atoms.push(Atom::isotropic(
            Vec3::new(0.62 * angle.cos(), lift, 0.62 * angle.sin()),
            0.052,
            0.98,
            hue(t),
        ));
    }
    atoms.push(Atom::isotropic(
        Vec3::new(0.0, 0.0, 0.0),
        0.13,
        0.92,
        Rgb::new(0.99, 0.92, 0.62),
    ));
    for i in 0..4 {
        let angle = (i as f64 / 4.0) * std::f64::consts::TAU;
        let half = angle * 0.5;
        atoms.push(Atom::new(
            Vec3::new(0.55 * angle.cos(), 0.62, 0.55 * angle.sin()),
            Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
            Vec3::new(0.62, 0.018, 0.20),
            0.85,
            Rgb::new(0.22, 0.28, 0.40),
        ));
    }
    let mut scene = Scene::new();
    for atom in &atoms {
        if let Err(e) = scene.push(*atom, MIN_SCALE) {
            crate::guard::unreachable("scene::showcase::demo_object", &format!("{e}"));
        }
    }
    scene
}

pub fn orbit_camera(width: u32, height: u32, turn: f64) -> CameraSpec {
    let radius = 1.85;
    let angle = turn * std::f64::consts::TAU;
    let eye = [
        OBJECT_CENTRE[0] + radius * angle.sin(),
        OBJECT_CENTRE[1] - 0.62,
        OBJECT_CENTRE[2] - radius * angle.cos(),
    ];
    CameraSpec::look_at(eye, OBJECT_CENTRE, [0.0, -1.0, 0.0], 55.0, width, height)
        .with_convention(PixelConvention::CornerSample)
}

pub fn still_camera(width: u32, height: u32) -> CameraSpec {
    orbit_camera(width, height, 0.045)
}

pub fn shell_camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, -5.0],
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        30.0,
        width,
        height,
    )
}

pub fn surfel_shell(count: usize) -> SurfelScene<f64> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let spacing = (4.0 / count as f64).sqrt();
    let mut shell = SurfelScene::new();
    for i in 0..count {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
        let ring = (1.0 - y * y).sqrt();
        let (sin, cos) = (golden * i as f64).sin_cos();
        let n = Vec3::new(ring * cos, y, ring * sin);
        let helper = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        match Surfel::new(n, n, helper, Vec2::new(spacing, spacing), 0.99, hue(i as f64 / count as f64 * 0.15 + 0.55)) {
            Ok(surfel) => shell.push(surfel),
            Err(e) => crate::guard::unreachable("scene::showcase::surfel_shell", &format!("{e}")),
        }
    }
    shell
}

pub fn material_shell(shell: &SurfelScene<f64>) -> SurfelScene<f64> {
    let glossy = match Material::new(Rgb::zero(), 0.6, 48.0) {
        Ok(m) => m,
        Err(e) => crate::guard::unreachable("scene::showcase::material_shell", &format!("{e}")),
    };
    let lamp = match Material::new(Rgb::new(1.0, 0.85, 0.5), 0.0, 1.0) {
        Ok(m) => m,
        Err(e) => crate::guard::unreachable("scene::showcase::material_shell", &format!("{e}")),
    };
    let mut out = SurfelScene::new();
    for s in shell.surfels() {
        let m = if -s.normal.y > SURFEL_LAMP_CAP { lamp } else { glossy };
        out.push(s.with_material(m));
    }
    out
}
