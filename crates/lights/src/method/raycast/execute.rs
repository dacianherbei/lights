use crate::blend::{BlendMode, Step};
use crate::camera::{Camera, Ray};
use crate::error::Result;
use crate::guard::Budget;
use crate::image::{DepthImage, TileAccum};
use crate::kernel::Kernel;
use crate::method::raycast::RayCastConfig;
use crate::method::raycast::prepare::{
    RayAtom, TileBinIndex, axial_sigma, cheap_reject, peak_parameter, quadratic, to_world_ray,
    whitened_residual,
};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::{Rgb, Vec3};
use crate::probe::{Counter, Probe};
use crate::settings::{DirectionalLight, OrderingModel, SharedSettings};
use crate::tiling::TileRect;

pub const HIT_BUDGET_SITE: &str = "raycast::max_hits_per_pixel";

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hit<T> {
    pub key: T,
    pub weight: T,
    pub opacity: T,
    pub colour: Rgb<T>,
    pub bound: T,
    pub index: u32,
}

#[inline(always)]
pub fn order_key<T: FloatingPoint>(v: T) -> u64 {
    let bits = v.to_f64().to_bits();
    let sign = 1u64 << 63;
    if bits & sign != 0 { !bits } else { bits | sign }
}

pub fn sort_hits<T: FloatingPoint>(hits: &mut [Hit<T>]) {
    hits.sort_unstable_by_key(|h| (order_key(h.key), h.index));
}

pub fn lambert<T: FloatingPoint, K: Kernel<T>>(
    atom: &RayAtom<T, K>,
    delta: Vec3<T>,
    direction: Vec3<T>,
    t_star: T,
    light: &DirectionalLight<T>,
) -> T {
    let offset = delta + direction * t_star;
    let gradient = atom.lambda.apply(offset);
    match (gradient.normalize(), light.direction.normalize()) {
        (Some(n), Some(l)) => light.ambient + light.diffuse * n.dot(l).max(T::ZERO),
        (None, Some(_)) => light.ambient,
        (Some(_), None) => light.ambient,
        (None, None) => light.ambient,
    }
}

pub fn intersect<T: FloatingPoint, K: Kernel<T>>(
    atom: &RayAtom<T, K>,
    ray: &Ray<T>,
    ordering: OrderingModel,
    near_plane: T,
) -> Option<Hit<T>> {
    intersect_lit(atom, ray, ordering, near_plane, None)
}

pub fn intersect_lit<T: FloatingPoint, K: Kernel<T>>(
    atom: &RayAtom<T, K>,
    ray: &Ray<T>,
    ordering: OrderingModel,
    near_plane: T,
    light: Option<&DirectionalLight<T>>,
) -> Option<Hit<T>> {
    let delta = ray.origin - atom.mean;
    let q = quadratic(&atom.lambda, delta, ray.direction);
    if !(q.a > T::ZERO) {
        return None;
    }
    let t_star = peak_parameter(&q);
    if !t_star.is_finite() {
        return None;
    }
    if t_star > ray.t_max {
        return None;
    }
    let residual = whitened_residual(&atom.whiten, delta, ray.direction, t_star);
    let r2 = residual.length_squared();
    if !(r2 <= atom.support_sq) {
        return None;
    }
    let sigma_t = axial_sigma(&q);
    if t_star + atom.support * sigma_t < near_plane {
        return None;
    }
    let weight = K::eval_ray(&atom.prepared, residual);
    if !weight.is_finite() {
        return None;
    }
    let colour = if K::MODULATED {
        atom.colour * K::modulate_ray(&atom.prepared, residual)
    } else {
        atom.colour
    };
    let colour = match light {
        Some(l) => colour * lambert(atom, delta, ray.direction, t_star, l),
        None => colour,
    };
    let key = match ordering {
        OrderingModel::CentreDepth => atom.depth,
        OrderingModel::HitDepth => t_star,
    };
    Some(Hit {
        key,
        weight,
        opacity: atom.opacity,
        colour,
        bound: atom.bound,
        index: atom.index,
    })
}

pub fn collect_hits<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    atoms: &[RayAtom<T, K>],
    candidates: Option<&[u32]>,
    ray: &Ray<T>,
    settings: &SharedSettings<T>,
    max_hits_per_pixel: u64,
    probe: &mut P,
    hits: &mut Vec<Hit<T>>,
) -> Result<()> {
    hits.clear();
    let mut budget = Budget::new(HIT_BUDGET_SITE, max_hits_per_pixel);
    match candidates {
        Some(list) => {
            for slot in list.iter() {
                let index = *slot as usize;
                if index >= atoms.len() {
                    continue;
                }
                gather_one::<T, K, P>(&atoms[index], ray, settings, probe, hits, &mut budget)?;
            }
        }
        None => {
            for atom in atoms.iter() {
                gather_one::<T, K, P>(atom, ray, settings, probe, hits, &mut budget)?;
            }
        }
    }
    if <K::Blend as BlendMode<T>>::ORDER_DEPENDENT {
        sort_hits(hits);
    }
    Ok(())
}

#[inline]
fn gather_one<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    atom: &RayAtom<T, K>,
    ray: &Ray<T>,
    settings: &SharedSettings<T>,
    probe: &mut P,
    hits: &mut Vec<Hit<T>>,
    budget: &mut Budget,
) -> Result<()> {
    probe.count(Counter::CandidatesTested, 1);
    if cheap_reject(
        atom.mean - ray.origin,
        ray.direction,
        atom.support,
        atom.scale_max,
    ) {
        return Ok(());
    }
    match intersect_lit(
        atom,
        ray,
        settings.ordering,
        settings.near_plane,
        settings.light.as_ref(),
    ) {
        Some(hit) => {
            budget.tick()?;
            probe.count(Counter::KernelEvaluations, 1);
            hits.push(hit);
            Ok(())
        }
        None => Ok(()),
    }
}

pub fn composite_hits<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    hits: &[Hit<T>],
    local_pixel: usize,
    settings: &SharedSettings<T>,
    cell: &mut <K::Blend as BlendMode<T>>::Accum,
    probe: &mut P,
) {
    for hit in hits.iter() {
        let transmittance = T::ONE - <K::Blend as BlendMode<T>>::coverage(cell);
        probe.tolerance(local_pixel, transmittance, hit.bound);
        if hit.opacity * hit.weight <= settings.blend.alpha_cutoff + hit.bound {
            probe.truncation(local_pixel, transmittance);
        }
        match <K::Blend as BlendMode<T>>::step(
            cell,
            hit.weight,
            hit.colour,
            hit.opacity,
            &settings.blend,
        ) {
            Step::Continue => probe.count(Counter::CompositedHits, 1),
            Step::Stop => {
                probe.count(Counter::EarlyOuts, 1);
                return;
            }
        }
    }
}

pub fn shade_pixel<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    atoms: &[RayAtom<T, K>],
    candidates: Option<&[u32]>,
    ray: &Ray<T>,
    local_pixel: usize,
    settings: &SharedSettings<T>,
    max_hits_per_pixel: u64,
    cell: &mut <K::Blend as BlendMode<T>>::Accum,
    probe: &mut P,
    hits: &mut Vec<Hit<T>>,
) -> Result<()> {
    probe.count(Counter::PixelsVisited, 1);
    collect_hits::<T, K, P>(
        atoms,
        candidates,
        ray,
        settings,
        max_hits_per_pixel,
        probe,
        hits,
    )?;
    composite_hits::<T, K, P>(hits, local_pixel, settings, cell, probe);
    Ok(())
}

pub fn render_tile<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    atoms: &[RayAtom<T, K>],
    index: Option<&TileBinIndex>,
    camera: &Camera<T>,
    rect: TileRect,
    settings: &SharedSettings<T>,
    config: &RayCastConfig,
    limit: Option<&DepthImage<T>>,
    acc: &mut TileAccum<'_, T, K::Blend>,
    probe: &mut P,
    hits: &mut Vec<Hit<T>>,
) -> Result<()> {
    let candidates: Option<&[u32]> = match index {
        Some(bins) => Some(bins.candidates(rect.x0.max(0) as u32, rect.y0.max(0) as u32)),
        None => None,
    };
    let empty = match candidates {
        Some(list) => list.is_empty(),
        None => atoms.is_empty(),
    };
    if empty {
        return Ok(());
    }
    let to_world = camera.rotation.transpose();
    for row in 0..rect.h {
        for col in 0..rect.w {
            let px = (rect.x0 + col) as u32;
            let py = (rect.y0 + row) as u32;
            let slot = (row * acc.stride + col) as usize;
            if slot >= acc.cells.len() {
                continue;
            }
            let ray = limited(to_world_ray(&to_world, camera.eye, &camera.ray(px, py)), limit, px, py);
            let local = (row * rect.w + col) as usize;
            let cell = &mut acc.cells[slot];
            shade_pixel::<T, K, P>(
                atoms,
                candidates,
                &ray,
                local,
                settings,
                config.max_hits_per_pixel,
                cell,
                probe,
                hits,
            )?;
        }
    }
    Ok(())
}

pub fn limited<T: FloatingPoint>(ray: Ray<T>, limit: Option<&DepthImage<T>>, px: u32, py: u32) -> Ray<T> {
    match limit {
        Some(depth) => Ray {
            t_max: ray.t_max.min(depth.get(px, py)),
            ..ray
        },
        None => ray,
    }
}
