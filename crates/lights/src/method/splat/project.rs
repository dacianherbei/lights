use crate::bound::atom_bound_with_lambda;
use crate::camera::Camera;
use crate::kernel::Kernel;
use crate::kernel::frame::ScreenFrame;
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Rgb, Vec2, Vec3};
use crate::probe::{Counter, Counters};
use crate::scene::prepare::{AtomView, PreparedScene};
use crate::settings::{SharedSettings, SupportMode};

pub const DETERMINANT_EPSILON: f64 = 1.0e-9;

pub struct ScreenAtom<T: FloatingPoint, K: Kernel<T>> {
    pub centre: Vec2<T>,
    pub depth: T,
    pub frame: ScreenFrame<T>,
    pub prepared: K::Screen,
    pub half_extent: Vec2<T>,
    pub support_sq: T,
    pub opacity: T,
    pub colour: Rgb<T>,
    pub bound: T,
    pub index: u32,
}

impl<T: FloatingPoint, K: Kernel<T>> Clone for ScreenAtom<T, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: FloatingPoint, K: Kernel<T>> Copy for ScreenAtom<T, K> {}

impl<T: FloatingPoint, K: Kernel<T>> core::fmt::Debug for ScreenAtom<T, K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ScreenAtom")
            .field("centre", &self.centre)
            .field("depth", &self.depth)
            .field("cov", &self.frame.cov)
            .field("conic", &self.frame.conic)
            .field("half_extent", &self.half_extent)
            .field("support_sq", &self.support_sq)
            .field("opacity", &self.opacity)
            .field("colour", &self.colour)
            .field("bound", &self.bound)
            .field("index", &self.index)
            .finish()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CullReason {
    Near,
    Determinant,
    Support,
    Opacity,
    NonFinite,
}

pub enum ProjectOutcome<T: FloatingPoint, K: Kernel<T>> {
    Visible(ScreenAtom<T, K>),
    Culled(CullReason),
}

impl<T: FloatingPoint, K: Kernel<T>> ProjectOutcome<T, K> {
    pub fn visible(&self) -> Option<&ScreenAtom<T, K>> {
        match self {
            ProjectOutcome::Visible(a) => Some(a),
            ProjectOutcome::Culled(_) => None,
        }
    }

    pub fn cull_reason(&self) -> Option<CullReason> {
        match self {
            ProjectOutcome::Visible(_) => None,
            ProjectOutcome::Culled(r) => Some(*r),
        }
    }
}

#[inline]
pub fn determinant_guard<T: FloatingPoint>(cov: Sym2<T>) -> bool {
    let det = cov.determinant();
    let scale = (cov.xx * cov.xx + cov.yy * cov.yy).max(T::ONE);
    det > T::from_f64(DETERMINANT_EPSILON) * scale
}

#[inline]
pub fn tight_half_extent<T: FloatingPoint>(cov: Sym2<T>, radius: T) -> Vec2<T> {
    Vec2::new(radius * cov.xx.sqrt(), radius * cov.yy.sqrt())
}

pub fn screen_covariance<T: FloatingPoint>(
    sigma_camera: &crate::numerics::symmetric::Sym3<T>,
    camera: &Camera<T>,
    cam_point: Vec3<T>,
) -> (Sym2<T>, bool) {
    let (clamped, fired) = camera.clamp_for_jacobian(cam_point);
    let j = camera.jacobian(clamped);
    (sigma_camera.project_2x3(&j), fired)
}

pub fn support_radius<T: FloatingPoint, K: Kernel<T>>(
    prepared: &K::Screen,
    peak: T,
    settings: &SharedSettings<T>,
) -> T {
    match settings.support {
        SupportMode::FixedSigmas(n) => T::from_f64(n),
        SupportMode::CutoffSolved => {
            K::support_screen(prepared, peak, settings.blend.alpha_cutoff)
        }
    }
}

pub fn project_atom<T: FloatingPoint, K: Kernel<T>>(
    view: AtomView<'_, T, K>,
    camera: &Camera<T>,
    settings: &SharedSettings<T>,
) -> (ProjectOutcome<T, K>, bool) {
    let cam = camera.to_camera(view.mean);
    if !cam.is_finite() {
        return (ProjectOutcome::Culled(CullReason::NonFinite), false);
    }
    if !(cam.z > settings.near_plane) {
        return (ProjectOutcome::Culled(CullReason::Near), false);
    }
    let centre = match camera.project(cam) {
        Some(c) => c,
        None => return (ProjectOutcome::Culled(CullReason::Near), false),
    };
    if !centre.is_finite() {
        return (ProjectOutcome::Culled(CullReason::NonFinite), false);
    }

    let sigma_cam = camera.covariance_to_camera(view.sigma);
    let (raw, fired) = screen_covariance(&sigma_cam, camera, cam);
    let (params, cov, gain) = K::prefilter(view.params, raw, settings.screen_dilation);

    if !determinant_guard(cov) {
        return (ProjectOutcome::Culled(CullReason::Determinant), fired);
    }
    let frame = match ScreenFrame::from_covariance(cov) {
        Some(f) => f,
        None => return (ProjectOutcome::Culled(CullReason::Determinant), fired),
    };
    let prepared = K::prepare_screen(&params, &frame);

    let peak = if settings.energy_compensation {
        view.opacity * gain
    } else {
        view.opacity
    };
    if !(peak > settings.blend.alpha_cutoff) {
        return (ProjectOutcome::Culled(CullReason::Opacity), fired);
    }

    let radius = support_radius::<T, K>(&prepared, peak, settings);
    if !(radius > T::ZERO) {
        return (ProjectOutcome::Culled(CullReason::Support), fired);
    }
    let half_extent = tight_half_extent(cov, radius);
    if !half_extent.is_finite() {
        return (ProjectOutcome::Culled(CullReason::NonFinite), fired);
    }

    let lambda_cam = view.lambda.congruence(&camera.rotation);
    let bound = atom_bound_with_lambda(cam, sigma_cam.zz, &lambda_cam, peak);

    (
        ProjectOutcome::Visible(ScreenAtom {
            centre,
            depth: cam.z,
            frame,
            prepared,
            half_extent,
            support_sq: radius * radius,
            opacity: peak,
            colour: view.colour,
            bound,
            index: view.index as u32,
        }),
        fired,
    )
}

pub fn project_scene<T: FloatingPoint, K: Kernel<T>>(
    scene: &PreparedScene<T, K>,
    camera: &Camera<T>,
    settings: &SharedSettings<T>,
    out: &mut Vec<ScreenAtom<T, K>>,
) -> (Counters, u64, u64) {
    out.clear();
    out.reserve(scene.len());
    let mut counters = Counters::default();
    counters.add(Counter::AtomsPrepared, scene.len() as u64);
    let mut clamped = 0u64;
    let mut determinant_culled = 0u64;
    for i in 0..scene.len() {
        let (outcome, fired) = project_atom::<T, K>(scene.atom(i), camera, settings);
        if fired {
            clamped += 1;
            counters.add(Counter::AtomsClamped, 1);
        }
        match outcome {
            ProjectOutcome::Visible(a) => {
                counters.add(Counter::AtomsProjected, 1);
                out.push(a);
            }
            ProjectOutcome::Culled(CullReason::Near) => {
                counters.add(Counter::AtomsCulledNear, 1)
            }
            ProjectOutcome::Culled(CullReason::Determinant) => {
                determinant_culled += 1;
                counters.add(Counter::AtomsCulledDet, 1)
            }
            ProjectOutcome::Culled(CullReason::NonFinite) => {
                counters.add(Counter::AtomsCulledDet, 1)
            }
            ProjectOutcome::Culled(CullReason::Support) => {
                counters.add(Counter::AtomsCulledSupport, 1)
            }
            ProjectOutcome::Culled(CullReason::Opacity) => {
                counters.add(Counter::AtomsCulledSupport, 1)
            }
        }
    }
    (counters, clamped, determinant_culled)
}
