use crate::camera::CameraSpec;
use crate::error::Result;
use crate::numerics::float::FloatingPoint;
use crate::numerics::quaternion::Quat;
use crate::numerics::vector::{Rgb, Vec3};
use crate::scene::{Atom, Scene};
use crate::settings::{MIN_SCALE, PixelConvention};

pub const SPHERE_SCENE: [([f64; 3], f64, [f64; 3]); 6] = [
    ([0.0, 0.0, -3.0], 0.5, [0.85, 0.25, 0.20]),
    ([0.0, -100.5, -1.0], 100.0, [0.45, 0.45, 0.48]),
    ([0.2, 0.5, -1.0], 0.5, [0.20, 0.70, 0.35]),
    ([0.5, 0.7, -1.0], 0.5, [0.25, 0.40, 0.90]),
    ([0.7, 0.8, -1.0], 0.5, [0.90, 0.75, 0.20]),
    ([0.9, 0.2, -1.0], 0.5, [0.70, 0.30, 0.85]),
];

pub const GROUND_INDEX: usize = 1;
pub const DEMO_OPACITY: f64 = 0.9;
pub const SIGMA_PER_RADIUS: f64 = 1.0 / 3.0;
pub const DEMO_WIDTH: u32 = 1800;
pub const DEMO_HEIGHT: u32 = 1012;
pub const DEMO_VIEWPORT_HEIGHT: f64 = 2.0;

pub fn to_lights_frame(p: [f64; 3]) -> Vec3<f64> {
    Vec3::new(p[0], -p[1], -p[2])
}

fn push_all<T: FloatingPoint>(atoms: &[Atom<f64>]) -> Scene<T> {
    let mut scene = Scene::<T>::new();
    let floor = T::from_f64(MIN_SCALE);
    for atom in atoms {
        match scene.push(Atom::<T>::from_f64(atom), floor) {
            Ok(()) => (),
            Err(e) => crate::guard::unreachable(
                "scene::demo::push_all",
                &format!("a demo atom failed validation: {e:?}"),
            ),
        }
    }
    scene
}

fn isotropic_atoms(skip_ground: bool) -> Vec<Atom<f64>> {
    let mut out = Vec::new();
    for (index, (centre, radius, colour)) in SPHERE_SCENE.iter().enumerate() {
        if skip_ground && index == GROUND_INDEX {
            continue;
        }
        out.push(Atom::isotropic(
            to_lights_frame(*centre),
            radius * SIGMA_PER_RADIUS,
            DEMO_OPACITY,
            Rgb::new(colour[0], colour[1], colour[2]),
        ));
    }
    out
}

pub fn spheres<T: FloatingPoint>() -> Scene<T> {
    push_all(&isotropic_atoms(false))
}

pub fn spheres_no_ground<T: FloatingPoint>() -> Scene<T> {
    push_all(&isotropic_atoms(true))
}

pub fn spheres_anisotropic<T: FloatingPoint>() -> Scene<T> {
    let shapes = [
        ([1.6, 0.7, 0.9], 0.37),
        ([1.0, 1.0, 1.0], 0.0),
        ([0.6, 1.7, 0.8], 1.11),
        ([1.3, 0.5, 1.2], 2.05),
        ([0.8, 1.4, 0.6], 0.74),
        ([1.5, 0.9, 0.55], 2.61),
    ];
    let mut out = Vec::new();
    for (index, (centre, radius, colour)) in SPHERE_SCENE.iter().enumerate() {
        let (shape, angle) = shapes[index];
        let sigma = radius * SIGMA_PER_RADIUS;
        let half: f64 = angle * 0.5;
        let axis = Vec3::new(0.36, 0.48, 0.8);
        let sin = half.sin();
        out.push(Atom::new(
            to_lights_frame(*centre),
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
            Vec3::new(sigma * shape[0], sigma * shape[1], sigma * shape[2]),
            DEMO_OPACITY,
            Rgb::new(colour[0], colour[1], colour[2]),
        ));
    }
    push_all(&out)
}

pub fn spheres_anisotropic_no_ground<T: FloatingPoint>() -> Scene<T> {
    let full = spheres_anisotropic::<f64>();
    let mut out = Vec::new();
    for (index, atom) in full.atoms().iter().enumerate() {
        if index == GROUND_INDEX {
            continue;
        }
        out.push(*atom);
    }
    push_all(&out)
}

pub fn sphere_camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        90.0,
        width,
        height,
    )
    .with_convention(PixelConvention::CornerSample)
}

pub fn sphere_intrinsics(width: u32, height: u32) -> ([f64; 2], [f64; 2]) {
    let w = width as f64;
    let h = height as f64;
    let viewport_height = DEMO_VIEWPORT_HEIGHT;
    let viewport_width = (w / h) * viewport_height;
    (
        [(w - 1.0) / viewport_width, (h - 1.0) / viewport_height],
        [(w - 1.0) * 0.5, (h - 1.0) * 0.5],
    )
}

pub fn sphere_exact_intrinsics_camera(width: u32, height: u32) -> CameraSpec {
    let (focal, principal) = sphere_intrinsics(width, height);
    sphere_camera(width, height).with_intrinsics(focal, principal)
}

pub fn flip_row(lights_row: f64, height: u32) -> f64 {
    (height as f64 - 1.0) - lights_row
}

pub fn narrow_camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        24.0,
        width,
        height,
    )
}

pub fn orthographic_camera(width: u32, height: u32, extent_y: f64) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        width,
        height,
    )
    .orthographic(extent_y)
}

pub fn single_atom_scene<T: FloatingPoint>(atom: Atom<T>) -> Result<Scene<T>> {
    let mut scene = Scene::<T>::new();
    scene.push(atom, T::from_f64(MIN_SCALE))?;
    Ok(scene)
}

pub fn ordering_probe_scene<T: FloatingPoint>() -> Scene<T> {
    let palette = [
        Rgb::new(1.0, 0.0, 0.0),
        Rgb::new(0.0, 0.0, 1.0),
        Rgb::new(0.0, 1.0, 0.0),
        Rgb::new(1.0, 1.0, 0.0),
    ];
    let mut atoms = Vec::new();
    for i in 0..8u32 {
        let t = i as f64;
        let half: f64 = 0.35 + 0.29 * t;
        atoms.push(Atom::new(
            Vec3::new(0.05 * (1.7 * t).sin(), 0.05 * (2.3 * t).cos(), 4.0),
            Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
            Vec3::new(0.30, 0.010, 0.010),
            0.7,
            palette[(i % 4) as usize],
        ));
    }
    push_all(&atoms)
}

pub fn demo_out_dir(argument: Option<String>) -> std::path::PathBuf {
    match argument {
        Some(dir) => std::path::PathBuf::from(dir),
        None => std::env::temp_dir().join("lights-render-demo"),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DemoImageLine {
    pub method: String,
    pub width: u32,
    pub height: u32,
    pub file: String,
    pub build_nanos: u128,
    pub bind_nanos: u128,
    pub render_nanos: u128,
    pub clamped_atoms: u64,
    pub atoms_projected: u64,
    pub kernel_evaluations: u64,
    pub composited_hits: u64,
    pub early_outs: u64,
}

impl DemoImageLine {
    pub fn lines(&self) -> Vec<String> {
        vec![
            format!(
                "{:<12} {}x{} -> {}",
                self.method, self.width, self.height, self.file
            ),
            format!(
                "             build {:.3} ms, bind {:.3} ms, render {:.3} ms, \
                 clamped atoms {}",
                self.build_nanos as f64 / 1.0e6,
                self.bind_nanos as f64 / 1.0e6,
                self.render_nanos as f64 / 1.0e6,
                self.clamped_atoms
            ),
            format!(
                "             atoms projected {}, kernel evaluations {}, \
                 composited {}, early outs {}",
                self.atoms_projected,
                self.kernel_evaluations,
                self.composited_hits,
                self.early_outs
            ),
        ]
    }
}

pub fn demo_closing_line(written: usize) -> String {
    format!(
        "{written} methods rendered the SAME Scene through the SAME CameraSpec \
         with the SAME KernelFamily at the SAME Precision, returning the same \
         FloatImage<f64> type. The only thing that changed between them is the \
         RenderMethod field."
    )
}

pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = self.state;
        (x >> 33) ^ x
    }

    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    pub fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

pub const STRESS_CLUSTERS: usize = 16;

pub fn stress_scene<T: FloatingPoint>(
    atom_count: usize,
    seed: u64,
    clustering: f64,
) -> Scene<T> {
    let mix = clustering.clamp(0.0, 1.0);
    let mut rng = Lcg::new(seed);
    let mut centres = Vec::with_capacity(STRESS_CLUSTERS);
    for _ in 0..STRESS_CLUSTERS {
        centres.push(Vec3::new(
            rng.signed() * 1.6,
            rng.signed() * 1.0,
            4.0 + rng.unit() * 6.0,
        ));
    }
    let mut atoms = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        let uniform = Vec3::new(
            rng.signed() * 2.0,
            rng.signed() * 1.2,
            3.0 + rng.unit() * 8.0,
        );
        let anchor = centres[index % STRESS_CLUSTERS];
        let jitter = Vec3::new(rng.signed() * 0.18, rng.signed() * 0.18, rng.signed() * 0.18);
        let clustered = anchor + jitter;
        let mean = uniform * (1.0 - mix) + clustered * mix;
        let sigma = 0.012 + rng.unit() * 0.05;
        let half = rng.unit() * std::f64::consts::PI;
        let axis = Vec3::new(rng.signed(), rng.signed(), rng.signed());
        let axis = match axis.normalize() {
            Some(v) => v,
            None => Vec3::new(0.0, 0.0, 1.0),
        };
        let sin = half.sin();
        atoms.push(Atom::new(
            mean,
            Quat::new_wxyz(half.cos(), axis.x * sin, axis.y * sin, axis.z * sin),
            Vec3::new(
                sigma * (0.5 + rng.unit()),
                sigma * (0.5 + rng.unit()),
                sigma * (0.5 + rng.unit()),
            ),
            0.15 + rng.unit() * 0.6,
            Rgb::new(rng.unit(), rng.unit(), rng.unit()),
        ));
    }
    push_all(&atoms)
}
