pub mod raycast;
pub mod splat;

use crate::blend::BlendMode;
use crate::camera::Camera;
use crate::error::Result;
use crate::image::{DepthImage, Framebuffer};
use crate::kernel::Kernel;
use crate::numerics::float::FloatingPoint;
use crate::probe::{Counters, Probe, ProbeSink};
use crate::scene::prepare::PreparedScene;
use crate::settings::SharedSettings;
use std::sync::atomic::{AtomicUsize, Ordering};

static THREAD_POOL_CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

pub fn thread_pool_constructions() -> usize {
    THREAD_POOL_CONSTRUCTIONS.load(Ordering::Relaxed)
}

pub fn build_thread_pool(threads: usize) -> Option<rayon::ThreadPool> {
    if threads <= 1 {
        return None;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();
    if pool.is_some() {
        THREAD_POOL_CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
    }
    pool
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MethodId {
    SplatRaster,
    RayCast,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TraversalOrder {
    ObjectOrder,
    ImageOrder,
}

impl MethodId {
    pub const ALL: [MethodId; 2] = [
        MethodId::SplatRaster,
        MethodId::RayCast,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MethodId::SplatRaster => "splat-raster",
            MethodId::RayCast => "ray-cast",
        }
    }

    pub fn implemented(self) -> bool {
        match self {
            MethodId::SplatRaster => cfg!(feature = "method-splat"),
            MethodId::RayCast => cfg!(feature = "method-raycast"),
        }
    }

    pub fn order(self) -> TraversalOrder {
        match self {
            MethodId::SplatRaster => TraversalOrder::ObjectOrder,
            MethodId::RayCast => TraversalOrder::ImageOrder,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RenderMethod {
    SplatRaster(splat::SplatConfig),
    RayCast(raycast::RayCastConfig),
}

impl RenderMethod {
    pub fn id(&self) -> MethodId {
        match self {
            RenderMethod::SplatRaster(_) => MethodId::SplatRaster,
            RenderMethod::RayCast(_) => MethodId::RayCast,
        }
    }

    pub fn tile(&self) -> u32 {
        match self {
            RenderMethod::SplatRaster(c) => c.tile,
            RenderMethod::RayCast(c) => c.tile,
        }
    }

    pub fn implemented() -> Vec<RenderMethod> {
        let all = [
            RenderMethod::splat_contract(),
            RenderMethod::raycast_contract(),
        ];
        all.into_iter().filter(|m| m.id().implemented()).collect()
    }

    pub fn implemented_for_timing() -> Vec<RenderMethod> {
        Self::implemented()
            .into_iter()
            .map(|m| match m.id() {
                MethodId::RayCast => RenderMethod::raycast_production(),
                MethodId::SplatRaster => m,
            })
            .collect()
    }

    pub fn splat_reference() -> Self {
        RenderMethod::SplatRaster(splat::SplatConfig::reference())
    }

    pub fn splat_contract() -> Self {
        RenderMethod::SplatRaster(splat::SplatConfig::contract())
    }

    pub fn raycast_production() -> Self {
        RenderMethod::RayCast(raycast::RayCastConfig::production())
    }

    pub fn raycast_contract() -> Self {
        RenderMethod::RayCast(raycast::RayCastConfig::contract())
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PhaseReport {
    pub wall_nanos: u128,
    pub resident_bytes: usize,
    pub counters: Counters,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RenderReport {
    pub build: PhaseReport,
    pub bind: PhaseReport,
    pub render: PhaseReport,
    pub framebuffer_bytes: usize,
    pub clamped_atoms: u64,
    pub determinant_culled_atoms: u64,
}

pub trait ImageFormation<T: FloatingPoint, K: Kernel<T>>: Sized + Send + Sync {
    fn id(&self) -> MethodId;

    fn build(&mut self, scene: &PreparedScene<T, K>) -> Result<PhaseReport>;

    fn bind(
        &mut self,
        scene: &PreparedScene<T, K>,
        camera: &Camera<T>,
    ) -> Result<PhaseReport>;

    fn render<P: Probe<T>>(
        &self,
        scene: &PreparedScene<T, K>,
        fb: &mut Framebuffer<T, K::Blend>,
        sink: &mut ProbeSink<T>,
    ) -> Result<PhaseReport>;

    fn sample_pixel(
        &self,
        scene: &PreparedScene<T, K>,
        px: u32,
        py: u32,
    ) -> Result<<K::Blend as BlendMode<T>>::Accum>;

    fn clamped_atoms(&self) -> u64;

    fn determinant_culled_atoms(&self) -> u64;

    fn limit_depth(&mut self, limit: Option<&DepthImage<T>>) -> Result<()>;

    fn resident_bytes(&self) -> usize;
}

pub enum RenderStrategy<T: FloatingPoint, K: Kernel<T>> {
    SplatRaster(splat::SplatRaster<T, K>),
    RayCast(raycast::RayCast<T, K>),
}

impl<T: FloatingPoint, K: Kernel<T>> ImageFormation<T, K> for RenderStrategy<T, K> {
    fn id(&self) -> MethodId {
        match self {
            RenderStrategy::SplatRaster(s) => s.id(),
            RenderStrategy::RayCast(s) => s.id(),
        }
    }

    fn build(&mut self, scene: &PreparedScene<T, K>) -> Result<PhaseReport> {
        match self {
            RenderStrategy::SplatRaster(s) => s.build(scene),
            RenderStrategy::RayCast(s) => s.build(scene),
        }
    }

    fn bind(
        &mut self,
        scene: &PreparedScene<T, K>,
        camera: &Camera<T>,
    ) -> Result<PhaseReport> {
        match self {
            RenderStrategy::SplatRaster(s) => s.bind(scene, camera),
            RenderStrategy::RayCast(s) => s.bind(scene, camera),
        }
    }

    fn render<P: Probe<T>>(
        &self,
        scene: &PreparedScene<T, K>,
        fb: &mut Framebuffer<T, K::Blend>,
        sink: &mut ProbeSink<T>,
    ) -> Result<PhaseReport> {
        match self {
            RenderStrategy::SplatRaster(s) => s.render::<P>(scene, fb, sink),
            RenderStrategy::RayCast(s) => s.render::<P>(scene, fb, sink),
        }
    }

    fn sample_pixel(
        &self,
        scene: &PreparedScene<T, K>,
        px: u32,
        py: u32,
    ) -> Result<<K::Blend as BlendMode<T>>::Accum> {
        match self {
            RenderStrategy::SplatRaster(s) => s.sample_pixel(scene, px, py),
            RenderStrategy::RayCast(s) => s.sample_pixel(scene, px, py),
        }
    }

    fn determinant_culled_atoms(&self) -> u64 {
        match self {
            RenderStrategy::SplatRaster(s) => s.determinant_culled_atoms(),
            RenderStrategy::RayCast(s) => s.determinant_culled_atoms(),
        }
    }

    fn clamped_atoms(&self) -> u64 {
        match self {
            RenderStrategy::SplatRaster(s) => s.clamped_atoms(),
            RenderStrategy::RayCast(s) => s.clamped_atoms(),
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            RenderStrategy::SplatRaster(s) => s.resident_bytes(),
            RenderStrategy::RayCast(s) => s.resident_bytes(),
        }
    }

    fn limit_depth(&mut self, limit: Option<&DepthImage<T>>) -> Result<()> {
        match self {
            RenderStrategy::SplatRaster(s) => s.limit_depth(limit),
            RenderStrategy::RayCast(s) => s.limit_depth(limit),
        }
    }
}

pub fn create_strategy<T: FloatingPoint, K: Kernel<T>>(
    method: &RenderMethod,
    settings: &SharedSettings<T>,
) -> Result<RenderStrategy<T, K>> {
    match method {
        RenderMethod::SplatRaster(c) => create_splat_raster(*c, settings),
        RenderMethod::RayCast(c) => create_ray_cast(*c, settings),
    }
}

pub fn create_splat_raster<T: FloatingPoint, K: Kernel<T>>(
    config: splat::SplatConfig,
    settings: &SharedSettings<T>,
) -> Result<RenderStrategy<T, K>> {
    Ok(RenderStrategy::SplatRaster(splat::SplatRaster::new(
        config, settings,
    )?))
}

pub fn create_ray_cast<T: FloatingPoint, K: Kernel<T>>(
    config: raycast::RayCastConfig,
    settings: &SharedSettings<T>,
) -> Result<RenderStrategy<T, K>> {
    Ok(RenderStrategy::RayCast(raycast::RayCast::new(
        config, settings,
    )?))
}
