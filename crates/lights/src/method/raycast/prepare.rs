use crate::bound::atom_bound_with_lambda;
use crate::camera::{Camera, Ray};
use crate::extent::ellipsoid_screen_extent;
use crate::kernel::Kernel;
use crate::kernel::frame::RayFrame;
use crate::numerics::float::FloatingPoint;
use crate::numerics::matrix::Mat3;
use crate::numerics::symmetric::{LowerTri3, Sym3};
use crate::numerics::vector::{Rgb, Vec3};
use crate::probe::{Counter, Counters};
use crate::scene::prepare::PreparedScene;
use crate::settings::{SharedSettings, SupportMode};
use crate::tiling::TileGrid;

pub struct RayAtom<T: FloatingPoint, K: Kernel<T>> {
    pub mean: Vec3<T>,
    pub mean_cam: Vec3<T>,
    pub lambda: Sym3<T>,
    pub lambda_cam: Sym3<T>,
    pub whiten: LowerTri3<T>,
    pub params: K::Params,
    pub prepared: K::Ray,
    pub opacity: T,
    pub colour: Rgb<T>,
    pub support: T,
    pub support_sq: T,
    pub scale_max: T,
    pub scale_min: T,
    pub sigma_zz: T,
    pub rho_e_sq: T,
    pub depth: T,
    pub bound: T,
    pub index: u32,
}

impl<T: FloatingPoint, K: Kernel<T>> Clone for RayAtom<T, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: FloatingPoint, K: Kernel<T>> Copy for RayAtom<T, K> {}

impl<T: FloatingPoint, K: Kernel<T>> core::fmt::Debug for RayAtom<T, K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RayAtom")
            .field("mean", &self.mean)
            .field("mean_cam", &self.mean_cam)
            .field("lambda", &self.lambda)
            .field("opacity", &self.opacity)
            .field("colour", &self.colour)
            .field("support", &self.support)
            .field("scale_max", &self.scale_max)
            .field("scale_min", &self.scale_min)
            .field("sigma_zz", &self.sigma_zz)
            .field("rho_e_sq", &self.rho_e_sq)
            .field("depth", &self.depth)
            .field("bound", &self.bound)
            .field("index", &self.index)
            .finish()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RayQuadratic<T> {
    pub a: T,
    pub b: T,
    pub c: T,
}

#[inline(always)]
pub fn quadratic<T: FloatingPoint>(
    lambda: &Sym3<T>,
    delta: Vec3<T>,
    direction: Vec3<T>,
) -> RayQuadratic<T> {
    let ld = lambda.apply(direction);
    RayQuadratic {
        a: direction.dot(ld),
        b: delta.dot(ld),
        c: lambda.quad_form(delta),
    }
}

#[inline(always)]
pub fn peak_parameter<T: FloatingPoint>(q: &RayQuadratic<T>) -> T {
    -q.b / q.a
}

#[inline(always)]
pub fn whitened_residual<T: FloatingPoint>(
    whiten: &LowerTri3<T>,
    delta: Vec3<T>,
    direction: Vec3<T>,
    t_star: T,
) -> Vec3<T> {
    whiten.apply(delta + direction * t_star)
}

#[inline(always)]
pub fn residual_r2<T: FloatingPoint>(
    lambda: &Sym3<T>,
    delta: Vec3<T>,
    direction: Vec3<T>,
    t_star: T,
) -> T {
    let residual = delta + direction * t_star;
    lambda.quad_form(residual).max(T::ZERO)
}

#[inline(always)]
pub fn axial_sigma<T: FloatingPoint>(q: &RayQuadratic<T>) -> T {
    q.a.sqrt().recip()
}

pub fn line_integral<T: FloatingPoint>(q: &RayQuadratic<T>, r2: T) -> T {
    if !(q.a > T::ZERO) {
        return T::ZERO;
    }
    (T::TAU / q.a).sqrt() * (-T::HALF * r2).exp()
}

pub fn extinction_for_opacity<T: FloatingPoint>(opacity: T, axial_sigma: T) -> Option<T> {
    let residual = T::ONE - opacity;
    if !(residual > T::ZERO) || !(axial_sigma > T::ZERO) {
        return None;
    }
    let value = -residual.ln() / (T::TAU.sqrt() * axial_sigma);
    if value.is_finite() { Some(value) } else { None }
}

pub fn beer_lambert_alpha<T: FloatingPoint>(kappa: T, arclength_integral: T) -> T {
    T::ONE - (-kappa * arclength_integral).exp()
}

#[inline(always)]
pub fn cheap_reject<T: FloatingPoint>(
    mean_relative: Vec3<T>,
    direction: Vec3<T>,
    support: T,
    scale_max: T,
) -> bool {
    let h = mean_relative.cross(direction).length();
    h > support * scale_max
}

#[inline(always)]
pub fn cheap_accept<T: FloatingPoint>(
    mean_relative: Vec3<T>,
    direction: Vec3<T>,
    support: T,
    scale_min: T,
) -> bool {
    let h = mean_relative.cross(direction).length();
    h <= support * scale_min
}

pub fn to_world_ray<T: FloatingPoint>(
    to_world: &Mat3<T>,
    eye: Vec3<T>,
    ray: &Ray<T>,
) -> Ray<T> {
    Ray {
        origin: to_world.mul_vec(ray.origin) + eye,
        direction: to_world.mul_vec(ray.direction),
        t_min: ray.t_min,
        t_max: ray.t_max,
    }
}

pub fn support_radius<T: FloatingPoint, K: Kernel<T>>(
    prepared: &K::Ray,
    peak: T,
    settings: &SharedSettings<T>,
) -> T {
    match settings.support {
        SupportMode::FixedSigmas(n) => T::from_f64(n),
        SupportMode::CutoffSolved => K::support_ray(prepared, peak, settings.blend.alpha_cutoff),
    }
}

pub fn prepare_atoms<T: FloatingPoint, K: Kernel<T>>(
    scene: &PreparedScene<T, K>,
    camera: &Camera<T>,
    settings: &SharedSettings<T>,
    out: &mut Vec<RayAtom<T, K>>,
) -> (Counters, u64) {
    out.clear();
    out.reserve(scene.len());
    let mut counters = Counters::default();
    counters.add(Counter::AtomsPrepared, scene.len() as u64);
    let view_axis = camera.rotation.row(2);

    for i in 0..scene.len() {
        let view = scene.atom(i);
        let cam = camera.to_camera(view.mean);
        if !cam.is_finite() {
            counters.add(Counter::AtomsCulledDet, 1);
            continue;
        }
        if !(cam.z > settings.near_plane) {
            counters.add(Counter::AtomsCulledNear, 1);
            continue;
        }
        let peak = view.opacity;
        if !(peak > settings.blend.alpha_cutoff) {
            counters.add(Counter::AtomsCulledSupport, 1);
            continue;
        }
        let centre_direction = match (view.mean - camera.eye).normalize() {
            Some(d) => d,
            None => {
                counters.add(Counter::AtomsCulledDet, 1);
                continue;
            }
        };
        let frame = RayFrame::new(*view.lambda, *view.whiten, centre_direction);
        let prepared = K::prepare_ray(view.params, &frame);
        let support = support_radius::<T, K>(&prepared, peak, settings);
        if !(support > T::ZERO) {
            counters.add(Counter::AtomsCulledSupport, 1);
            continue;
        }
        let sigma_zz = view.sigma.quad_form(view_axis);
        let lambda_cam = view.lambda.congruence(&camera.rotation);
        let rho_e_sq = lambda_cam.quad_form(cam);
        let bound = atom_bound_with_lambda(cam, sigma_zz, &lambda_cam, peak);
        counters.add(Counter::AtomsProjected, 1);
        out.push(RayAtom {
            mean: view.mean,
            mean_cam: cam,
            lambda: *view.lambda,
            lambda_cam,
            whiten: *view.whiten,
            params: *view.params,
            prepared,
            opacity: peak,
            colour: view.colour,
            support,
            support_sq: support * support,
            scale_max: view.scale_max,
            scale_min: view.scale_min,
            sigma_zz,
            rho_e_sq,
            depth: cam.z,
            bound,
            index: view.index as u32,
        });
    }
    (counters, 0)
}

pub struct TileBinIndex {
    grid: TileGrid,
    values: Vec<u32>,
    ranges: Vec<(u32, u32)>,
    unbounded: Vec<u32>,
}

pub const UNBOUNDED_REPLICATION_SITE: &str = "raycast::tile_bin_unbounded_replication";
pub const UNBOUNDED_REPLICATION_LIMIT: u64 = 1 << 24;

impl TileBinIndex {
    pub fn build<T: FloatingPoint, K: Kernel<T>>(
        atoms: &[RayAtom<T, K>],
        camera: &Camera<T>,
        grid: &TileGrid,
    ) -> crate::error::Result<Self> {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut unbounded: Vec<u32> = Vec::new();
        let camera64 = camera.to_f64();
        for (slot, atom) in atoms.iter().enumerate() {
            let extent = ellipsoid_screen_extent(
                &camera64,
                atom.mean_cam.to_f64(),
                &atom.lambda_cam.to_f64(),
                atom.support_sq.to_f64(),
            );
            let extent = match extent {
                Some(e) => e,
                None => {
                    unbounded.push(slot as u32);
                    continue;
                }
            };
            let span = match grid.span_of_box(extent.min_corner(), extent.max_corner()) {
                Some(s) => s,
                None => continue,
            };
            for ty in span.ty0..span.ty1 {
                for tx in span.tx0..span.tx1 {
                    let rect = grid.rect(tx, ty);
                    if extent.misses_rect(&rect) {
                        continue;
                    }
                    pairs.push((grid.tile_id(tx, ty), slot as u32));
                }
            }
        }
        pairs.sort_unstable();

        let tile_count = grid.tile_count();
        let replicated = (unbounded.len() as u64).saturating_mul(tile_count as u64);
        if replicated > UNBOUNDED_REPLICATION_LIMIT {
            return Err(crate::error::LightsError::BudgetExhausted {
                site: UNBOUNDED_REPLICATION_SITE,
                limit: UNBOUNDED_REPLICATION_LIMIT,
            });
        }
        let mut values = Vec::with_capacity(pairs.len() + unbounded.len() * tile_count);
        let mut ranges = vec![(0u32, 0u32); tile_count];
        let mut cursor = 0usize;
        for tile in 0..tile_count {
            let start = values.len() as u32;
            for slot in unbounded.iter() {
                values.push(*slot);
            }
            let mut taken = cursor;
            for pair in pairs[cursor..].iter() {
                if pair.0 as usize != tile {
                    break;
                }
                values.push(pair.1);
                taken += 1;
            }
            cursor = taken;
            let end = values.len() as u32;
            if end > start {
                values[start as usize..end as usize].sort_unstable();
            }
            ranges[tile] = (start, end);
        }

        Ok(Self {
            grid: *grid,
            values,
            ranges,
            unbounded,
        })
    }

    pub fn replicated_pairs(&self) -> u64 {
        (self.unbounded.len() as u64).saturating_mul(self.ranges.len() as u64)
    }

    pub fn pair_count(&self) -> u64 {
        self.values.len() as u64
    }

    pub fn grid(&self) -> TileGrid {
        self.grid
    }

    pub fn unbounded_atoms(&self) -> &[u32] {
        &self.unbounded
    }

    pub fn tile_candidates(&self, tile_id: usize) -> &[u32] {
        match self.ranges.get(tile_id) {
            Some((start, end)) => &self.values[*start as usize..*end as usize],
            None => &[],
        }
    }

    pub fn candidates(&self, px: u32, py: u32) -> &[u32] {
        if px >= self.grid.width || py >= self.grid.height {
            return &[];
        }
        let tx = (px / self.grid.tile) as i32;
        let ty = (py / self.grid.tile) as i32;
        self.tile_candidates(self.grid.tile_id(tx, ty) as usize)
    }

    pub fn resident_bytes(&self) -> usize {
        self.values.len() * core::mem::size_of::<u32>()
            + self.ranges.len() * core::mem::size_of::<(u32, u32)>()
            + self.unbounded.len() * core::mem::size_of::<u32>()
    }
}
