use crate::agreement::ContractViolation;
use crate::method::MethodId;
use crate::settings::OrderingModel;

pub type Result<T> = core::result::Result<T, LightsError>;

#[derive(Debug, thiserror::Error)]
pub enum LightsError {
    #[error("method not implemented: {method:?}")]
    MethodNotImplemented { method: MethodId },

    #[error("{shapes} per-atom kernel shapes for a scene of {atoms} atoms")]
    ShapeCountMismatch { shapes: usize, atoms: usize },

    #[error("kernel {kernel} has no screen marginal; {method:?} cannot project it")]
    KernelNotProjectable {
        kernel: &'static str,
        method: MethodId,
    },

    #[error(
        "{method:?} implements no pixel prefilter, so it cannot honour {setting}; \
         passing it would silently render an unprefiltered image under a \
         prefiltered configuration"
    )]
    PrefilterUnsupported {
        setting: &'static str,
        method: MethodId,
    },

    #[error("ordering model {ordering:?} unsupported by {method:?}")]
    OrderingUnsupported {
        ordering: OrderingModel,
        method: MethodId,
    },

    #[error(
        "{method:?} has no per-pixel normal, so it cannot honour a directional \
         light; rendering flat under a lit configuration would be a silent lie"
    )]
    LightingUnsupported { method: MethodId },

    #[error(
        "{method:?} cannot reject a hit beyond a per-pixel depth; compositing \
         its layer over an opaque surface would show what the surface hides"
    )]
    DepthLimitUnsupported { method: MethodId },

    #[error("atom {index} degenerate: {reason:?}")]
    DegenerateAtom {
        index: usize,
        reason: DegenerateReason,
    },

    #[error("surfel degenerate: {reason}")]
    DegenerateSurfel { reason: &'static str },

    #[error("non-finite value in {what}")]
    NonFinite { what: &'static str },

    #[error("empty scene")]
    EmptyScene,

    #[error("viewport must be non-zero, got {width}x{height}")]
    ViewportZero { width: u32, height: u32 },

    #[error("invalid camera: {0}")]
    InvalidCamera(&'static str),

    #[error("invalid outline: {0}")]
    InvalidOutline(&'static str),

    #[error(
        "the prepared scene bound to {method:?} is epoch {bound} and the one handed to the \
         render is epoch {given}: bind() and render() were given two different scenes"
    )]
    SceneNotBound {
        method: MethodId,
        bound: u64,
        given: u64,
    },

    #[error("image size mismatch: {a:?} vs {b:?}")]
    SizeMismatch { a: (u32, u32), b: (u32, u32) },

    #[error("a {width}x{height} raster of {channels} channel(s) cannot hold {got} values")]
    RasterLength {
        width: u32,
        height: u32,
        channels: usize,
        got: usize,
    },

    #[error(
        "a {width}x{height} frame of four-byte pixels does not fit a target of {len} bytes at a \
         row stride of {stride} bytes"
    )]
    ByteTarget {
        width: u32,
        height: u32,
        stride: usize,
        len: usize,
    },

    #[error("draw op {index} refused: {reason}")]
    InvalidDrawOp { index: usize, reason: &'static str },

    #[error("atom {index} is outside a scene of {len}")]
    AtomOutOfRange { index: usize, len: usize },

    #[error("sub-rect {rect:?} does not fit inside frame {frame:?}")]
    SubRectOutsideFrame {
        rect: (u32, u32, u32, u32),
        frame: (u32, u32),
    },

    #[error("budget exhausted at {site}: limit {limit}")]
    BudgetExhausted { site: &'static str, limit: u64 },

    #[error("cross-method contract violated: {0:?}")]
    ContractViolation(ContractViolation),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("png encode: {0}")]
    PngEncode(String),

    #[error(
        "reference is not exact: {fraction} of its area is undecided quadrature, \
         above the stated limit {limit}. A distortion number against it would \
         report the quadrature rather than the representation"
    )]
    ReferenceNotExact { fraction: f64, limit: f64 },

    #[error(
        "candidate is {mean_square} from a raster reference, below that raster's own \
         quantisation floor {floor}: the figure would score agreement with the \
         rounding, not with the image the raster was quantised from"
    )]
    BeyondQuantisation { mean_square: f64, floor: f64 },

    #[error("distortion metric undefined: {what}")]
    MetricUndefined { what: &'static str },

    #[error(
        "kernel {kernel} has no analytic gradient of the fit loss; only the gaussian \
         under the over blend is carried through the reverse pass"
    )]
    GradientUnsupported { kernel: &'static str },

    #[error("scene file: {0}")]
    SceneFormat(String),

    #[error("frame capture line {line}: {reason}")]
    CaptureFormat { line: usize, reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegenerateReason {
    NonPositiveScale,
    NonSpdCovariance,
    UnnormalisableRotation,
    OpacityBelowCutoff,
    NonFiniteParameter,
}
