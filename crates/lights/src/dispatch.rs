use crate::camera::{Camera, CameraSpec};
use crate::error::{LightsError, Result};
use crate::image::{DepthImage, FloatImage, Framebuffer};
use crate::kernel::compact::CompactBump;
use crate::kernel::gabor::{GaborColour, GaborDensity};
use crate::kernel::gaussian::{Gaussian, GaussianNormalised};
use crate::kernel::hermite::{GaussHermiteColour, GaussHermiteDensity};
use crate::kernel::supergaussian::SuperGaussian;
use crate::kernel::{CurveShape, Kernel, KernelDescription, KernelFamily, describe};
use crate::method::{ImageFormation, MethodId, RenderMethod, RenderReport, create_strategy};
use crate::numerics::float::FloatingPoint;
use crate::probe::{AuditProbe, CounterProbe, Counters, NoProbe, ProbeSink};
use crate::scene::Scene;
use crate::scene::prepare::PreparedScene;
use crate::settings::SharedSettings;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    pub const ALL: [Precision; 2] = [Precision::F32, Precision::F64];

    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Collect {
    Pixels,
    PixelsAndCounters,
    PixelsAndTolerance,
    PixelsAndCoverage,
}

#[derive(Clone, Debug)]
pub struct RenderRequest {
    pub precision: Precision,
    pub kernel: KernelFamily,
    pub method: RenderMethod,
    pub settings: SharedSettings<f64>,
    pub camera: CameraSpec,
    pub scene: Scene<f64>,
    pub collect: Collect,
    pub shapes: Option<Vec<CurveShape>>,
}

#[derive(Clone, Debug)]
pub struct RenderOutcome {
    pub image: FloatImage<f64>,
    pub report: RenderReport,
    pub counters: Counters,
    pub tolerance: Option<FloatImage<f64>>,
    pub truncation: Option<FloatImage<f64>>,
    pub premultiplied: Option<FloatImage<f64>>,
    pub coverage: Option<FloatImage<f64>>,
    pub kernel: KernelDescription,
    pub method: MethodId,
    pub precision: Precision,
}

pub fn render(request: &RenderRequest) -> Result<RenderOutcome> {
    dispatch_precision(request, None)
}

pub fn render_depth_limited(request: &RenderRequest, limit: &DepthImage<f64>) -> Result<RenderOutcome> {
    dispatch_precision(request, Some(limit))
}

fn dispatch_precision(request: &RenderRequest, limit: Option<&DepthImage<f64>>) -> Result<RenderOutcome> {
    match request.precision {
        Precision::F32 => dispatch_kernel::<f32>(request, limit),
        Precision::F64 => dispatch_kernel::<f64>(request, limit),
    }
}

fn shapes_for(request: &RenderRequest, uniform: &CurveShape) -> Result<Vec<CurveShape>> {
    match request.shapes.as_ref() {
        None => Ok(vec![*uniform; request.scene.len()]),
        Some(list) => {
            if list.len() != request.scene.len() {
                return Err(LightsError::ShapeCountMismatch {
                    shapes: list.len(),
                    atoms: request.scene.len(),
                });
            }
            Ok(list.clone())
        }
    }
}

fn dispatch_kernel<T: FloatingPoint>(request: &RenderRequest, limit: Option<&DepthImage<f64>>) -> Result<RenderOutcome> {
    let shape = request.kernel.shape();
    match request.kernel {
        KernelFamily::Gaussian => dispatch_method::<T, Gaussian>(request, &shape, limit),
        KernelFamily::GaussianNormalised => {
            dispatch_method::<T, GaussianNormalised>(request, &shape, limit)
        }
        KernelFamily::SuperGaussian(_) => dispatch_method::<T, SuperGaussian>(request, &shape, limit),
        KernelFamily::CompactBump => dispatch_method::<T, CompactBump>(request, &shape, limit),
        KernelFamily::GaborColour(_) => dispatch_method::<T, GaborColour>(request, &shape, limit),
        KernelFamily::GaborDensity(_) => dispatch_method::<T, GaborDensity>(request, &shape, limit),
        KernelFamily::GaussHermiteColour(_) => {
            dispatch_method::<T, GaussHermiteColour>(request, &shape, limit)
        }
        KernelFamily::GaussHermiteDensity(_) => {
            dispatch_method::<T, GaussHermiteDensity>(request, &shape, limit)
        }
    }
}

fn dispatch_method<T: FloatingPoint, K: Kernel<T>>(
    request: &RenderRequest,
    shape: &CurveShape,
    limit: Option<&DepthImage<f64>>,
) -> Result<RenderOutcome> {
    let settings = SharedSettings::<T>::from_f64(&request.settings);
    let camera = Camera::<T>::from_spec(&request.camera, settings.near_plane)?;
    let scene = Scene::<T>::from_f64(&request.scene);
    let shapes = shapes_for(request, shape)?;
    let params: Vec<K::Params> = shapes.iter().map(|s| K::params_from(s)).collect();
    let prepared = PreparedScene::<T, K>::build(&scene, &params, &settings)?;

    let mut strategy = create_strategy::<T, K>(&request.method, &settings)?;
    let build = strategy.build(&prepared)?;
    let bind = strategy.bind(&prepared, &camera)?;
    match limit {
        Some(depth) => strategy.limit_depth(Some(&depth.convert::<T>()))?,
        None => {}
    }

    let mut framebuffer =
        Framebuffer::<T, K::Blend>::new(camera.width, camera.height, request.method.tile())?;

    let mut sink = match request.collect {
        Collect::Pixels => ProbeSink::<T>::counters_only(camera.width, camera.height),
        Collect::PixelsAndCounters => ProbeSink::<T>::counters_only(camera.width, camera.height),
        Collect::PixelsAndTolerance => ProbeSink::<T>::with_tolerance(camera.width, camera.height),
        Collect::PixelsAndCoverage => ProbeSink::<T>::counters_only(camera.width, camera.height),
    };

    let render = match request.collect {
        Collect::Pixels => strategy.render::<NoProbe>(&prepared, &mut framebuffer, &mut sink),
        Collect::PixelsAndCounters => {
            strategy.render::<CounterProbe<T>>(&prepared, &mut framebuffer, &mut sink)
        }
        Collect::PixelsAndTolerance => {
            strategy.render::<AuditProbe<T>>(&prepared, &mut framebuffer, &mut sink)
        }
        Collect::PixelsAndCoverage => {
            strategy.render::<CounterProbe<T>>(&prepared, &mut framebuffer, &mut sink)
        }
    }?;

    let image = framebuffer.resolve_float(settings.background).into_f64();
    let (premultiplied, coverage) = match request.collect {
        Collect::Pixels => (None, None),
        Collect::PixelsAndCounters => (None, None),
        Collect::PixelsAndTolerance => (None, None),
        Collect::PixelsAndCoverage => {
            let (colour, alpha) = framebuffer.resolve_premultiplied();
            (Some(colour.into_f64()), Some(alpha.into_f64()))
        }
    };

    Ok(RenderOutcome {
        image,
        report: RenderReport {
            build,
            bind,
            render,
            framebuffer_bytes: framebuffer.resident_bytes(),
            clamped_atoms: strategy.clamped_atoms(),
            determinant_culled_atoms: strategy.determinant_culled_atoms(),
        },
        counters: sink.counters,
        tolerance: sink.tolerance.take().map(|t| t.into_f64()),
        truncation: sink.truncation.take().map(|t| t.into_f64()),
        premultiplied,
        coverage,
        kernel: describe::<T, K>(),
        method: request.method.id(),
        precision: request.precision,
    })
}
