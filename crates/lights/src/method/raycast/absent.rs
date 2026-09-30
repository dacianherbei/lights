use crate::blend::BlendMode;
use crate::camera::Camera;
use crate::error::{LightsError, Result};
use crate::image::{DepthImage, Framebuffer};
use crate::kernel::Kernel;
use crate::method::raycast::RayCastConfig;
use crate::method::{ImageFormation, MethodId, PhaseReport};
use crate::numerics::float::FloatingPoint;
use crate::probe::{Probe, ProbeSink};
use crate::scene::prepare::PreparedScene;
use crate::settings::SharedSettings;

pub struct RayCast<T: FloatingPoint, K: Kernel<T>> {
    _marker: core::marker::PhantomData<(T, K)>,
}

impl<T: FloatingPoint, K: Kernel<T>> RayCast<T, K> {
    pub fn new(_config: RayCastConfig, _settings: &SharedSettings<T>) -> Result<Self> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::RayCast,
        })
    }
}

impl<T: FloatingPoint, K: Kernel<T>> ImageFormation<T, K> for RayCast<T, K> {
    fn id(&self) -> MethodId {
        MethodId::RayCast
    }

    fn build(&mut self, _scene: &PreparedScene<T, K>) -> Result<PhaseReport> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::RayCast,
        })
    }

    fn bind(&mut self, _scene: &PreparedScene<T, K>, _camera: &Camera<T>) -> Result<PhaseReport> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::RayCast,
        })
    }

    fn render<P: Probe<T>>(
        &self,
        _scene: &PreparedScene<T, K>,
        _fb: &mut Framebuffer<T, K::Blend>,
        _sink: &mut ProbeSink<T>,
    ) -> Result<PhaseReport> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::RayCast,
        })
    }

    fn sample_pixel(
        &self,
        _scene: &PreparedScene<T, K>,
        _px: u32,
        _py: u32,
    ) -> Result<<K::Blend as BlendMode<T>>::Accum> {
        Err(LightsError::MethodNotImplemented {
            method: MethodId::RayCast,
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
            method: MethodId::RayCast,
        })
    }
}
