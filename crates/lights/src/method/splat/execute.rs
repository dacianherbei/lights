use crate::blend::{BlendMode, Step};
use crate::camera::Camera;
use crate::image::TileAccum;
use crate::kernel::Kernel;
use crate::method::splat::bin::SortKey;
use crate::method::splat::project::ScreenAtom;
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::LowerTri2;
use crate::numerics::vector::{Rgb, Vec2};
use crate::probe::{Counter, Probe};
use crate::settings::SharedSettings;
use crate::tiling::TileRect;

pub struct BatchEntry<T: FloatingPoint, K: Kernel<T>> {
    pub centre: Vec2<T>,
    pub l_inv: LowerTri2<T>,
    pub support_sq: T,
    pub prepared: K::Screen,
    pub opacity: T,
    pub colour: Rgb<T>,
    pub bound: T,
    pub index: u32,
}

impl<T: FloatingPoint, K: Kernel<T>> Clone for BatchEntry<T, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: FloatingPoint, K: Kernel<T>> Copy for BatchEntry<T, K> {}

#[inline]
pub fn entry_of<T: FloatingPoint, K: Kernel<T>>(atom: &ScreenAtom<T, K>) -> BatchEntry<T, K> {
    BatchEntry {
        centre: atom.centre,
        l_inv: atom.frame.l_inv,
        support_sq: atom.support_sq,
        prepared: atom.prepared,
        opacity: atom.opacity,
        colour: atom.colour,
        bound: atom.bound,
        index: atom.index,
    }
}

pub fn gather_batch<T: FloatingPoint, K: Kernel<T>>(
    screen: &[ScreenAtom<T, K>],
    keys: &[SortKey],
    range: (u32, u32),
    out: &mut Vec<BatchEntry<T, K>>,
) {
    out.clear();
    let start = range.0 as usize;
    let end = (range.1 as usize).min(keys.len());
    if start >= end {
        return;
    }
    out.reserve(end - start);
    for key in &keys[start..end] {
        let slot = key.2 as usize;
        if slot < screen.len() {
            out.push(entry_of(&screen[slot]));
        }
    }
}

#[inline]
pub fn shade_pixel<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    batch: &[BatchEntry<T, K>],
    sample: Vec2<T>,
    local_pixel: usize,
    settings: &SharedSettings<T>,
    cell: &mut <K::Blend as BlendMode<T>>::Accum,
    probe: &mut P,
) {
    probe.count(Counter::PixelsVisited, 1);
    for entry in batch {
        probe.count(Counter::CandidatesTested, 1);
        let offset = entry.centre - sample;
        let whitened = entry.l_inv.apply(offset);
        let r2 = whitened.length_squared();
        if !(r2 <= entry.support_sq) {
            continue;
        }
        probe.count(Counter::KernelEvaluations, 1);
        let weight = K::eval_screen(&entry.prepared, whitened);
        let colour = if K::MODULATED {
            entry.colour * K::modulate_screen(&entry.prepared, whitened)
        } else {
            entry.colour
        };
        let transmittance = T::ONE - <K::Blend as BlendMode<T>>::coverage(cell);
        probe.tolerance(local_pixel, transmittance, entry.bound);
        if entry.opacity * weight <= settings.blend.alpha_cutoff + entry.bound {
            probe.truncation(local_pixel, transmittance);
        }
        match <K::Blend as BlendMode<T>>::step(
            cell,
            weight,
            colour,
            entry.opacity,
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

pub fn render_tile_atom_outer<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    batch: &[BatchEntry<T, K>],
    rect: TileRect,
    camera: &Camera<T>,
    settings: &SharedSettings<T>,
    acc: &mut TileAccum<'_, T, K::Blend>,
    probe: &mut P,
) {
    let width = rect.w as usize;
    let count = width * rect.h as usize;
    let mut samples: Vec<Vec2<T>> = Vec::with_capacity(count);
    let mut slots: Vec<usize> = Vec::with_capacity(count);
    let mut live: Vec<bool> = Vec::with_capacity(count);
    for row in 0..rect.h {
        for col in 0..rect.w {
            let slot = (row * acc.stride + col) as usize;
            let present = slot < acc.cells.len();
            samples.push(camera.sample_point((rect.x0 + col) as u32, (rect.y0 + row) as u32));
            slots.push(slot);
            live.push(present);
            if present {
                probe.count(Counter::PixelsVisited, 1);
            }
        }
    }

    for entry in batch {
        for local in 0..count {
            if !live[local] {
                continue;
            }
            probe.count(Counter::CandidatesTested, 1);
            let offset = entry.centre - samples[local];
            let whitened = entry.l_inv.apply(offset);
            let r2 = whitened.length_squared();
            if !(r2 <= entry.support_sq) {
                continue;
            }
            probe.count(Counter::KernelEvaluations, 1);
            let weight = K::eval_screen(&entry.prepared, whitened);
            let colour = if K::MODULATED {
                entry.colour * K::modulate_screen(&entry.prepared, whitened)
            } else {
                entry.colour
            };
            let cell = &mut acc.cells[slots[local]];
            let transmittance = T::ONE - <K::Blend as BlendMode<T>>::coverage(cell);
            probe.tolerance(local, transmittance, entry.bound);
            if entry.opacity * weight <= settings.blend.alpha_cutoff + entry.bound {
                probe.truncation(local, transmittance);
            }
            match <K::Blend as BlendMode<T>>::step(
                cell,
                weight,
                colour,
                entry.opacity,
                &settings.blend,
            ) {
                Step::Continue => probe.count(Counter::CompositedHits, 1),
                Step::Stop => {
                    probe.count(Counter::EarlyOuts, 1);
                    live[local] = false;
                }
            }
        }
    }
}

pub fn render_tile<T: FloatingPoint, K: Kernel<T>, P: Probe<T>>(
    batch: &[BatchEntry<T, K>],
    rect: TileRect,
    camera: &Camera<T>,
    settings: &SharedSettings<T>,
    acc: &mut TileAccum<'_, T, K::Blend>,
    probe: &mut P,
) {
    for row in 0..rect.h {
        for col in 0..rect.w {
            let px = (rect.x0 + col) as u32;
            let py = (rect.y0 + row) as u32;
            let sample = camera.sample_point(px, py);
            let slot = (row * acc.stride + col) as usize;
            if slot >= acc.cells.len() {
                continue;
            }
            let local = (row * rect.w + col) as usize;
            let cell = &mut acc.cells[slot];
            shade_pixel::<T, K, P>(batch, sample, local, settings, cell, probe);
        }
    }
}
