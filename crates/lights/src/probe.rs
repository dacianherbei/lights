use crate::image::FloatImage;
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::Rgb;
use crate::tiling::TileRect;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub atoms_prepared: u64,
    pub atoms_projected: u64,
    pub atoms_culled_near: u64,
    pub atoms_culled_det: u64,
    pub atoms_culled_support: u64,
    pub atoms_clamped: u64,
    pub tile_atom_pairs: u64,
    pub sort_keys: u64,
    pub candidates_tested: u64,
    pub kernel_evaluations: u64,
    pub composited_hits: u64,
    pub early_outs: u64,
    pub pixels_visited: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Counter {
    AtomsPrepared,
    AtomsProjected,
    AtomsCulledNear,
    AtomsCulledDet,
    AtomsCulledSupport,
    AtomsClamped,
    TileAtomPairs,
    SortKeys,
    CandidatesTested,
    KernelEvaluations,
    CompositedHits,
    EarlyOuts,
    PixelsVisited,
}

impl Counters {
    pub fn merge(&mut self, o: &Counters) {
        self.atoms_prepared += o.atoms_prepared;
        self.atoms_projected += o.atoms_projected;
        self.atoms_culled_near += o.atoms_culled_near;
        self.atoms_culled_det += o.atoms_culled_det;
        self.atoms_culled_support += o.atoms_culled_support;
        self.atoms_clamped += o.atoms_clamped;
        self.tile_atom_pairs += o.tile_atom_pairs;
        self.sort_keys += o.sort_keys;
        self.candidates_tested += o.candidates_tested;
        self.kernel_evaluations += o.kernel_evaluations;
        self.composited_hits += o.composited_hits;
        self.early_outs += o.early_outs;
        self.pixels_visited += o.pixels_visited;
    }

    pub fn get(&self, c: Counter) -> u64 {
        match c {
            Counter::AtomsPrepared => self.atoms_prepared,
            Counter::AtomsProjected => self.atoms_projected,
            Counter::AtomsCulledNear => self.atoms_culled_near,
            Counter::AtomsCulledDet => self.atoms_culled_det,
            Counter::AtomsCulledSupport => self.atoms_culled_support,
            Counter::AtomsClamped => self.atoms_clamped,
            Counter::TileAtomPairs => self.tile_atom_pairs,
            Counter::SortKeys => self.sort_keys,
            Counter::CandidatesTested => self.candidates_tested,
            Counter::KernelEvaluations => self.kernel_evaluations,
            Counter::CompositedHits => self.composited_hits,
            Counter::EarlyOuts => self.early_outs,
            Counter::PixelsVisited => self.pixels_visited,
        }
    }

    pub fn add(&mut self, c: Counter, n: u64) {
        match c {
            Counter::AtomsPrepared => self.atoms_prepared += n,
            Counter::AtomsProjected => self.atoms_projected += n,
            Counter::AtomsCulledNear => self.atoms_culled_near += n,
            Counter::AtomsCulledDet => self.atoms_culled_det += n,
            Counter::AtomsCulledSupport => self.atoms_culled_support += n,
            Counter::AtomsClamped => self.atoms_clamped += n,
            Counter::TileAtomPairs => self.tile_atom_pairs += n,
            Counter::SortKeys => self.sort_keys += n,
            Counter::CandidatesTested => self.candidates_tested += n,
            Counter::KernelEvaluations => self.kernel_evaluations += n,
            Counter::CompositedHits => self.composited_hits += n,
            Counter::EarlyOuts => self.early_outs += n,
            Counter::PixelsVisited => self.pixels_visited += n,
        }
    }
}

pub trait Probe<T: FloatingPoint>: Sized + Send {
    fn for_tile(rect: TileRect) -> Self;
    fn count(&mut self, c: Counter, n: u64);
    fn tolerance(&mut self, local_pixel: usize, transmittance: T, bound: T);
    fn truncation(&mut self, local_pixel: usize, transmittance: T);
    fn merge_into(self, sink: &mut ProbeSink<T>);
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NoProbe;

impl<T: FloatingPoint> Probe<T> for NoProbe {
    #[inline(always)]
    fn for_tile(_rect: TileRect) -> Self {
        NoProbe
    }
    #[inline(always)]
    fn count(&mut self, _c: Counter, _n: u64) {}
    #[inline(always)]
    fn tolerance(&mut self, _local_pixel: usize, _transmittance: T, _bound: T) {}
    #[inline(always)]
    fn truncation(&mut self, _local_pixel: usize, _transmittance: T) {}
    #[inline(always)]
    fn merge_into(self, _sink: &mut ProbeSink<T>) {}
}

#[derive(Copy, Clone, Debug, Default)]
pub struct CounterProbe<T> {
    counters: Counters,
    element: core::marker::PhantomData<T>,
}

impl<T: FloatingPoint> CounterProbe<T> {
    pub fn counters(&self) -> Counters {
        self.counters
    }
}

impl<T: FloatingPoint> Probe<T> for CounterProbe<T> {
    #[inline(always)]
    fn for_tile(_rect: TileRect) -> Self {
        Self {
            counters: Counters::default(),
            element: core::marker::PhantomData,
        }
    }
    #[inline(always)]
    fn count(&mut self, c: Counter, n: u64) {
        self.counters.add(c, n);
    }
    #[inline(always)]
    fn tolerance(&mut self, _local_pixel: usize, _transmittance: T, _bound: T) {}
    #[inline(always)]
    fn truncation(&mut self, _local_pixel: usize, _transmittance: T) {}
    fn merge_into(self, sink: &mut ProbeSink<T>) {
        sink.counters.merge(&self.counters);
    }
}

#[derive(Clone, Debug)]
pub struct AuditProbe<T> {
    rect: TileRect,
    counters: Counters,
    tolerance: Option<Vec<T>>,
    truncation: Option<Vec<T>>,
}

impl<T: FloatingPoint> AuditProbe<T> {
    pub fn counters(&self) -> Counters {
        self.counters
    }

    pub fn rect(&self) -> TileRect {
        self.rect
    }

    pub fn tolerance_slice(&self) -> Option<&[T]> {
        self.tolerance.as_deref()
    }

    pub fn truncation_slice(&self) -> Option<&[T]> {
        self.truncation.as_deref()
    }
}

impl<T: FloatingPoint> Probe<T> for AuditProbe<T> {
    fn for_tile(rect: TileRect) -> Self {
        Self {
            rect,
            counters: Counters::default(),
            tolerance: None,
            truncation: None,
        }
    }

    #[inline(always)]
    fn count(&mut self, c: Counter, n: u64) {
        self.counters.add(c, n);
    }

    fn tolerance(&mut self, local_pixel: usize, transmittance: T, bound: T) {
        let cells = (self.rect.w.max(0) as usize) * (self.rect.h.max(0) as usize);
        let buffer = self.tolerance.get_or_insert_with(|| vec![T::ZERO; cells]);
        if local_pixel < buffer.len() {
            buffer[local_pixel] += transmittance * bound;
        }
    }

    fn truncation(&mut self, local_pixel: usize, transmittance: T) {
        let cells = (self.rect.w.max(0) as usize) * (self.rect.h.max(0) as usize);
        let buffer = self.truncation.get_or_insert_with(|| vec![T::ZERO; cells]);
        if local_pixel < buffer.len() {
            buffer[local_pixel] += transmittance;
        }
    }

    fn merge_into(self, sink: &mut ProbeSink<T>) {
        sink.counters.merge(&self.counters);
        if !sink.collect_tolerance {
            return;
        }
        let rect = self.rect;
        let width = sink.width;
        let height = sink.height;
        if let Some(buffer) = self.tolerance {
            let image = sink
                .tolerance
                .get_or_insert_with(|| FloatImage::new(width, height));
            scatter_tile(image, rect, &buffer);
        }
        if let Some(buffer) = self.truncation {
            let image = sink
                .truncation
                .get_or_insert_with(|| FloatImage::new(width, height));
            scatter_tile(image, rect, &buffer);
        }
    }
}

fn scatter_tile<T: FloatingPoint>(image: &mut FloatImage<T>, rect: TileRect, buffer: &[T]) {
    for row in 0..rect.h.max(0) {
        for col in 0..rect.w.max(0) {
            let local = (row * rect.w + col) as usize;
            let x = (rect.x0 + col) as u32;
            let y = (rect.y0 + row) as u32;
            if x < image.width && y < image.height && local < buffer.len() {
                let previous = image.get(x, y);
                image.set(x, y, previous + Rgb::splat(buffer[local]));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProbeSink<T> {
    pub counters: Counters,
    pub tolerance: Option<FloatImage<T>>,
    pub truncation: Option<FloatImage<T>>,
    pub width: u32,
    pub height: u32,
    pub collect_tolerance: bool,
}

impl<T: FloatingPoint> ProbeSink<T> {
    pub fn counters_only(width: u32, height: u32) -> Self {
        Self {
            counters: Counters::default(),
            tolerance: None,
            truncation: None,
            width,
            height,
            collect_tolerance: false,
        }
    }

    pub fn with_tolerance(width: u32, height: u32) -> Self {
        Self {
            counters: Counters::default(),
            tolerance: None,
            truncation: None,
            width,
            height,
            collect_tolerance: true,
        }
    }
}
