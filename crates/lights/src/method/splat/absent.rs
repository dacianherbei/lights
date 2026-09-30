use crate::blend::BlendMode;
use crate::camera::Camera;
use crate::error::{LightsError, Result};
use crate::image::{DepthImage, Framebuffer};
use crate::kernel::Kernel;
use crate::method::splat::SplatConfig;
use crate::method::{ImageFormation, MethodId, PhaseReport};
use crate::numerics::float::FloatingPoint;
use crate::probe::{Probe, ProbeSink};
use crate::scene::prepare::PreparedScene;
use crate::settings::SharedSettings;

pub struct SplatRaster<T: FloatingPoint, K: Kernel<T>> {
    _marker: core::marker::PhantomData<(T, K)>,
}

impl<T: FloatingPoint, K: Kernel<T>> SplatRaster<T, K> {
    pub fn new(_config: SplatConfig, _settings: &SharedSettings<T>) -> Result<Self> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::SplatRaster,
        })
    }
}

impl<T: FloatingPoint, K: Kernel<T>> ImageFormation<T, K> for SplatRaster<T, K> {
    fn id(&self) -> MethodId {
        MethodId::SplatRaster
    }

    fn build(&mut self, _scene: &PreparedScene<T, K>) -> Result<PhaseReport> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::SplatRaster,
        })
    }

    fn bind(&mut self, _scene: &PreparedScene<T, K>, _camera: &Camera<T>) -> Result<PhaseReport> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::SplatRaster,
        })
    }

    fn render<P: Probe<T>>(
        &self,
        _scene: &PreparedScene<T, K>,
        _fb: &mut Framebuffer<T, K::Blend>,
        _sink: &mut ProbeSink<T>,
    ) -> Result<PhaseReport> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::SplatRaster,
        })
    }

    fn sample_pixel(
        &self,
        _scene: &PreparedScene<T, K>,
        _px: u32,
        _py: u32,
    ) -> Result<<K::Blend as BlendMode<T>>::Accum> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::SplatRaster,
        })
    }

    fn determinant_culled_atoms(&self) -> u64 {
        0
    }

    fn clamped_atoms(&self) -> u64 {
        0
    }

    fn resident_bytes(&self) -> usize {
        0
    }

    fn limit_depth(&mut self, _limit: Option<&DepthImage<T>>) -> Result<()> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::SplatRaster,
        })
    }
}
