pub mod agreement;
pub mod bound;
pub mod blend;
pub mod camera;
pub mod capture;
pub mod dispatch;
pub mod draw;
pub mod error;
pub mod extent;
#[cfg(feature = "fit")]
pub mod fit;
pub mod guard;
pub mod image;
pub mod kernel;
pub mod method;
pub mod numerics;
pub mod outline;
pub mod primitive;
pub mod probe;
pub mod renderer;
pub mod scene;
pub mod settings;
pub mod stats;
pub mod surfel;
pub mod tiling;

pub use error::{DegenerateReason, LightsError, Result};
pub use extent::{ScreenExtent, ellipsoid_screen_extent, min_quadratic_over_rect};
pub use renderer::Renderer;
pub use guard::{Budget, unreachable};

pub use numerics::float::FloatingPoint;
pub use numerics::matrix::{Mat2x3, Mat3};
pub use numerics::quaternion::Quat;
pub use numerics::similarity::Similarity2;
pub use numerics::symmetric::{LowerTri2, LowerTri3, Sym2, Sym3};
pub use numerics::vector::{Rgb, Vec2, Vec3};

pub use blend::{Additive, BlendMode, BlendSettings, CompositeClass, EarlyOut, Normalised,
    NormalisedAccum, Over, Step};

pub use kernel::compact::{CompactBump, CompactParams, ZERO_RADIUS, ZERO_RADIUS_SQUARED,
    compact_radius, compact_response};
pub use kernel::frame::{RayFrame, ScreenFrame};
pub use kernel::gabor::{GaborColour, GaborDensity, GaborParams, GaborRay, GaborScreen,
    gabor_carrier, gabor_envelope};
pub use kernel::gaussian::{Gaussian, GaussianNormalised, REFERENCE_THREE_SIGMA_CUTOFF,
    radius_for_cutoff};
pub use kernel::hermite::{GaussHermiteColour, GaussHermiteDensity, HermiteParams, HermiteRay,
    HermiteScreen};
pub use kernel::supergaussian::{SuperGaussian, SuperGaussianParams, super_gaussian_radius,
    super_gaussian_response};
pub use kernel::{Beta, CurveShape, FIRST_HERMITE_ORDER, GAUSSIAN_BETA, Kernel, KernelDescription,
    KernelFamily, KernelId, SECOND_HERMITE_ORDER, ZERO_FREQUENCY, ZERO_ORIENTATION, ZERO_PHASE,
    describe, moment_prefilter};

pub use settings::{
    DirectionalLight, OrderingModel, PixelConvention, SettingsFingerprint,
    SharedSettings, SupportMode,
};

pub use camera::{Camera, CameraSpec, Projection, Ray};

pub use scene::demo::{Lcg, SPHERE_SCENE, narrow_camera, ordering_probe_scene,
    orthographic_camera, sphere_camera, sphere_exact_intrinsics_camera, sphere_intrinsics,
    spheres, spheres_anisotropic, spheres_anisotropic_no_ground,
    spheres_no_ground, single_atom_scene, stress_scene, to_lights_frame};
pub use scene::prepare::{AtomView, PreparedScene};
pub use scene::{Atom, Scene};
pub use surfel::{Material, OPAQUE_TRANSMITTANCE, SURFEL_SUPPORT_SIGMAS, Surfel, SurfelHit, SurfelLayer,
    SurfelScene, intersect_surfel, render_surfel_layer, render_surfels, surfel_shade, surfel_specular,
    to_camera_frame};

pub use image::{DepthImage, FloatImage, Framebuffer, Image, RasterLayer, TileAccum, read_png, srgb_decode,
    srgb_encode, write_png, write_png_normalized};

pub use tiling::{DEFAULT_TILE, TileGrid, TileRect, TileSpan, floor_div, for_each_tile};

pub use probe::{AuditProbe, Counter, Counters, NoProbe, Probe, ProbeSink};

pub use method::raycast::{AccelKind, RayCastConfig};
pub use method::splat::{SortKind, SplatConfig};
pub use method::{
    ImageFormation, MethodId, PhaseReport, RenderMethod, RenderReport, RenderStrategy,
    TraversalOrder, create_ray_cast, create_splat_raster, create_strategy,
};

pub use draw::{Clip, DrawList, DrawOp, DrawShape, Ink};
pub use capture::{CAPTURE_COLUMNS, Capture, CapturedOp, Stand};
pub use dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};

pub use agreement::{AgreementReport, BoundTerms, ContractViolation, FIRST_ORDER_COEFFICIENT,
    HEADROOM, SECOND_ORDER_COEFFICIENT, assert_contract, atom_bound, atom_bound_terms,
    atom_bound_terms_with_lambda, atom_bound_with_lambda, compare, compose_tolerance,
    compose_tolerance_with_truncation, convention_divergence, envelope, run_pair};
