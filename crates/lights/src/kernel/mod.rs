pub mod compact;
pub mod frame;
pub mod gabor;
pub mod gaussian;
pub mod hermite;
pub mod supergaussian;

use crate::blend::{BlendMode, CompositeClass};
use crate::kernel::frame::{RayFrame, ScreenFrame};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym2;
use crate::numerics::vector::{Vec2, Vec3};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelId(pub u16);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CurveShape {
    pub beta: f64,
    pub frequency: f64,
    pub orientation: f64,
    pub elevation: f64,
    pub phase: f64,
    pub order: u8,
}

impl Default for CurveShape {
    fn default() -> Self {
        Self {
            beta: GAUSSIAN_BETA,
            frequency: ZERO_FREQUENCY,
            orientation: ZERO_ORIENTATION,
            elevation: ZERO_ELEVATION,
            phase: ZERO_PHASE,
            order: FIRST_HERMITE_ORDER,
        }
    }
}

impl CurveShape {
    pub fn gaussian() -> Self {
        Self::default()
    }

    pub fn with_beta(beta: f64) -> Self {
        Self {
            beta,
            ..Self::default()
        }
    }

    pub fn gabor(frequency: f64, orientation: f64, phase: f64) -> Self {
        Self {
            frequency,
            orientation,
            phase,
            ..Self::default()
        }
    }

    pub fn hermite(order: u8, orientation: f64) -> Self {
        Self {
            order,
            orientation,
            ..Self::default()
        }
    }

    pub fn with_elevation(self, elevation: f64) -> Self {
        Self { elevation, ..self }
    }

    pub fn is_in_plane(&self) -> bool {
        self.elevation == ZERO_ELEVATION
    }

    pub fn is_gaussian(&self) -> bool {
        self.beta == GAUSSIAN_BETA
    }

    pub fn is_zero_frequency(&self) -> bool {
        self.frequency == ZERO_FREQUENCY && self.phase == ZERO_PHASE
    }
}

pub const GAUSSIAN_BETA: f64 = 1.0;
pub const ZERO_FREQUENCY: f64 = 0.0;
pub const ZERO_ORIENTATION: f64 = 0.0;
pub const ZERO_ELEVATION: f64 = 0.0;
pub const ZERO_PHASE: f64 = 0.0;
pub const FIRST_HERMITE_ORDER: u8 = 1;
pub const SECOND_HERMITE_ORDER: u8 = 2;
pub const QUARTER_CYCLE_FREQUENCY: f64 = 0.25;
pub const DIAGONAL_ORIENTATION: f64 = core::f64::consts::FRAC_PI_4;
pub const QUADRATURE_PHASE: f64 = core::f64::consts::FRAC_PI_2;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Beta {
    Half,
    One,
    Two,
    Four,
    Eight,
}

impl Beta {
    pub const SWEEP: [Beta; 5] = [Beta::Half, Beta::One, Beta::Two, Beta::Four, Beta::Eight];

    pub fn value(self) -> f64 {
        match self {
            Beta::Half => 0.5,
            Beta::One => 1.0,
            Beta::Two => 2.0,
            Beta::Four => 4.0,
            Beta::Eight => 8.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Beta::Half => "0.5",
            Beta::One => "1",
            Beta::Two => "2",
            Beta::Four => "4",
            Beta::Eight => "8",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Carrier {
    Axis,
    Diagonal,
    Quadrature,
}

impl Carrier {
    pub const SWEEP: [Carrier; 3] = [Carrier::Axis, Carrier::Diagonal, Carrier::Quadrature];

    pub fn frequency(self) -> f64 {
        match self {
            Carrier::Axis => QUARTER_CYCLE_FREQUENCY,
            Carrier::Diagonal => QUARTER_CYCLE_FREQUENCY,
            Carrier::Quadrature => QUARTER_CYCLE_FREQUENCY,
        }
    }

    pub fn orientation(self) -> f64 {
        match self {
            Carrier::Axis => ZERO_ORIENTATION,
            Carrier::Diagonal => DIAGONAL_ORIENTATION,
            Carrier::Quadrature => ZERO_ORIENTATION,
        }
    }

    pub fn phase(self) -> f64 {
        match self {
            Carrier::Axis => ZERO_PHASE,
            Carrier::Diagonal => ZERO_PHASE,
            Carrier::Quadrature => QUADRATURE_PHASE,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Carrier::Axis => "axis",
            Carrier::Diagonal => "diagonal",
            Carrier::Quadrature => "quadrature",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Lobe {
    First,
    Second,
}

impl Lobe {
    pub const SWEEP: [Lobe; 2] = [Lobe::First, Lobe::Second];

    pub fn order(self) -> u8 {
        match self {
            Lobe::First => FIRST_HERMITE_ORDER,
            Lobe::Second => SECOND_HERMITE_ORDER,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Lobe::First => "h1",
            Lobe::Second => "h2",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelFamily {
    Gaussian,
    GaussianNormalised,
    SuperGaussian(Beta),
    CompactBump,
    GaborColour(Carrier),
    GaborDensity(Carrier),
    GaussHermiteColour(Lobe),
    GaussHermiteDensity(Lobe),
}

impl KernelFamily {
    pub const FIT_SWEEP: [KernelFamily; 8] = [
        KernelFamily::Gaussian,
        KernelFamily::GaussianNormalised,
        KernelFamily::SuperGaussian(Beta::Half),
        KernelFamily::SuperGaussian(Beta::One),
        KernelFamily::SuperGaussian(Beta::Two),
        KernelFamily::SuperGaussian(Beta::Four),
        KernelFamily::SuperGaussian(Beta::Eight),
        KernelFamily::CompactBump,
    ];

    pub const MODULATED_SWEEP: [KernelFamily; 10] = [
        KernelFamily::GaborColour(Carrier::Axis),
        KernelFamily::GaborColour(Carrier::Diagonal),
        KernelFamily::GaborColour(Carrier::Quadrature),
        KernelFamily::GaborDensity(Carrier::Axis),
        KernelFamily::GaborDensity(Carrier::Diagonal),
        KernelFamily::GaborDensity(Carrier::Quadrature),
        KernelFamily::GaussHermiteColour(Lobe::First),
        KernelFamily::GaussHermiteColour(Lobe::Second),
        KernelFamily::GaussHermiteDensity(Lobe::First),
        KernelFamily::GaussHermiteDensity(Lobe::Second),
    ];

    pub const ALL: [KernelFamily; 18] = [
        KernelFamily::Gaussian,
        KernelFamily::GaussianNormalised,
        KernelFamily::SuperGaussian(Beta::Half),
        KernelFamily::SuperGaussian(Beta::One),
        KernelFamily::SuperGaussian(Beta::Two),
        KernelFamily::SuperGaussian(Beta::Four),
        KernelFamily::SuperGaussian(Beta::Eight),
        KernelFamily::CompactBump,
        KernelFamily::GaborColour(Carrier::Axis),
        KernelFamily::GaborColour(Carrier::Diagonal),
        KernelFamily::GaborColour(Carrier::Quadrature),
        KernelFamily::GaborDensity(Carrier::Axis),
        KernelFamily::GaborDensity(Carrier::Diagonal),
        KernelFamily::GaborDensity(Carrier::Quadrature),
        KernelFamily::GaussHermiteColour(Lobe::First),
        KernelFamily::GaussHermiteColour(Lobe::Second),
        KernelFamily::GaussHermiteDensity(Lobe::First),
        KernelFamily::GaussHermiteDensity(Lobe::Second),
    ];

    pub const BLEND_PAIR: [KernelFamily; 2] =
        [KernelFamily::Gaussian, KernelFamily::GaussianNormalised];

    pub const CURVE_SWEEP: [KernelFamily; 7] = [
        KernelFamily::SuperGaussian(Beta::Half),
        KernelFamily::SuperGaussian(Beta::One),
        KernelFamily::SuperGaussian(Beta::Two),
        KernelFamily::SuperGaussian(Beta::Four),
        KernelFamily::SuperGaussian(Beta::Eight),
        KernelFamily::CompactBump,
        KernelFamily::Gaussian,
    ];

    pub fn name(self) -> &'static str {
        match self {
            KernelFamily::Gaussian => "gaussian",
            KernelFamily::GaussianNormalised => "gaussian-normalised",
            KernelFamily::SuperGaussian(Beta::Half) => "super-gaussian-b0.5",
            KernelFamily::SuperGaussian(Beta::One) => "super-gaussian-b1",
            KernelFamily::SuperGaussian(Beta::Two) => "super-gaussian-b2",
            KernelFamily::SuperGaussian(Beta::Four) => "super-gaussian-b4",
            KernelFamily::SuperGaussian(Beta::Eight) => "super-gaussian-b8",
            KernelFamily::GaborColour(c) => match c {
                Carrier::Axis => "gabor-colour-axis",
                Carrier::Diagonal => "gabor-colour-diagonal",
                Carrier::Quadrature => "gabor-colour-quadrature",
            },
            KernelFamily::GaborDensity(c) => match c {
                Carrier::Axis => "gabor-density-axis",
                Carrier::Diagonal => "gabor-density-diagonal",
                Carrier::Quadrature => "gabor-density-quadrature",
            },
            KernelFamily::GaussHermiteColour(l) => match l {
                Lobe::First => "gauss-hermite-colour-h1",
                Lobe::Second => "gauss-hermite-colour-h2",
            },
            KernelFamily::GaussHermiteDensity(l) => match l {
                Lobe::First => "gauss-hermite-density-h1",
                Lobe::Second => "gauss-hermite-density-h2",
            },
            KernelFamily::CompactBump => "compact-bump",
        }
    }

    pub fn kernel_name(self) -> &'static str {
        match self {
            KernelFamily::Gaussian => "gaussian",
            KernelFamily::GaussianNormalised => "gaussian-normalised",
            KernelFamily::SuperGaussian(_) => "super-gaussian",
            KernelFamily::GaborColour(_) => "gabor-colour",
            KernelFamily::GaborDensity(_) => "gabor-density",
            KernelFamily::GaussHermiteColour(_) => "gauss-hermite-colour",
            KernelFamily::GaussHermiteDensity(_) => "gauss-hermite-density",
            KernelFamily::CompactBump => "compact-bump",
        }
    }

    pub fn id(self) -> KernelId {
        match self {
            KernelFamily::Gaussian => KernelId(0),
            KernelFamily::GaussianNormalised => KernelId(1),
            KernelFamily::SuperGaussian(_) => KernelId(2),
            KernelFamily::GaborColour(_) => KernelId(4),
            KernelFamily::GaborDensity(_) => KernelId(5),
            KernelFamily::GaussHermiteColour(_) => KernelId(6),
            KernelFamily::GaussHermiteDensity(_) => KernelId(7),
            KernelFamily::CompactBump => KernelId(3),
        }
    }

    pub fn is_signed(self) -> bool {
        match self {
            KernelFamily::Gaussian => false,
            KernelFamily::GaussianNormalised => false,
            KernelFamily::GaborColour(_) => false,
            KernelFamily::GaborDensity(_) => true,
            KernelFamily::GaussHermiteColour(_) => false,
            KernelFamily::GaussHermiteDensity(_) => true,
            KernelFamily::SuperGaussian(_) => false,
            KernelFamily::CompactBump => false,
        }
    }

    pub fn shape(self) -> CurveShape {
        match self {
            KernelFamily::Gaussian => CurveShape::gaussian(),
            KernelFamily::GaussianNormalised => CurveShape::gaussian(),
            KernelFamily::SuperGaussian(b) => CurveShape::with_beta(b.value()),
            KernelFamily::GaborColour(c) => {
                CurveShape::gabor(c.frequency(), c.orientation(), c.phase())
            }
            KernelFamily::GaborDensity(c) => {
                CurveShape::gabor(c.frequency(), c.orientation(), c.phase())
            }
            KernelFamily::GaussHermiteColour(l) => {
                CurveShape::hermite(l.order(), ZERO_ORIENTATION)
            }
            KernelFamily::GaussHermiteDensity(l) => {
                CurveShape::hermite(l.order(), ZERO_ORIENTATION)
            }
            KernelFamily::CompactBump => CurveShape::gaussian(),
        }
    }

    pub fn curve_is_gaussian(self) -> bool {
        match self {
            KernelFamily::Gaussian => true,
            KernelFamily::GaussianNormalised => true,
            KernelFamily::SuperGaussian(b) => b.value() == GAUSSIAN_BETA,
            KernelFamily::GaborColour(_) => false,
            KernelFamily::GaborDensity(_) => false,
            KernelFamily::GaussHermiteColour(_) => false,
            KernelFamily::GaussHermiteDensity(_) => false,
            KernelFamily::CompactBump => false,
        }
    }

    pub fn blend_is_over(self) -> bool {
        match self {
            KernelFamily::Gaussian => true,
            KernelFamily::GaussianNormalised => false,
            KernelFamily::SuperGaussian(_) => true,
            KernelFamily::GaborColour(_) => true,
            KernelFamily::GaborDensity(_) => false,
            KernelFamily::GaussHermiteColour(_) => true,
            KernelFamily::GaussHermiteDensity(_) => false,
            KernelFamily::CompactBump => true,
        }
    }

    pub fn eval_is_peak_normalised(self) -> bool {
        match self {
            KernelFamily::Gaussian => true,
            KernelFamily::GaussianNormalised => true,
            KernelFamily::SuperGaussian(_) => true,
            KernelFamily::CompactBump => true,
            KernelFamily::GaborColour(_) => true,
            KernelFamily::GaborDensity(c) => c.phase() == ZERO_PHASE,
            KernelFamily::GaussHermiteColour(_) => true,
            KernelFamily::GaussHermiteDensity(_) => false,
        }
    }

    pub fn is_modulated(self) -> bool {
        match self {
            KernelFamily::Gaussian => false,
            KernelFamily::GaussianNormalised => false,
            KernelFamily::SuperGaussian(_) => false,
            KernelFamily::CompactBump => false,
            KernelFamily::GaborColour(_) => true,
            KernelFamily::GaborDensity(_) => false,
            KernelFamily::GaussHermiteColour(_) => true,
            KernelFamily::GaussHermiteDensity(_) => false,
        }
    }
}

pub fn moment_prefilter<T: FloatingPoint>(cov: Sym2<T>, pixel_variance: T) -> (Sym2<T>, T) {
    if pixel_variance <= T::ZERO {
        return (cov, T::ONE);
    }
    let dilated = cov.add_diagonal(pixel_variance);
    let raw_det = cov.determinant();
    let new_det = dilated.determinant();
    let gain = if raw_det > T::ZERO && new_det > T::ZERO {
        (raw_det / new_det).sqrt()
    } else {
        T::ONE
    };
    (dilated, gain)
}

pub trait Kernel<T: FloatingPoint>: Copy + Clone + core::fmt::Debug + Send + Sync + 'static {
    type Params: Copy + Clone + Send + Sync + Default + core::fmt::Debug;
    type Screen: Copy + Clone + Send + Sync + core::fmt::Debug;
    type Ray: Copy + Clone + Send + Sync + core::fmt::Debug;
    type Blend: BlendMode<T>;

    const ID: KernelId;
    const NAME: &'static str;
    const RADIAL: bool;
    const HAS_SCREEN_MARGINAL: bool;
    const MODULATED: bool = false;

    fn params_from(shape: &CurveShape) -> Self::Params;

    fn prepare_screen(p: &Self::Params, f: &ScreenFrame<T>) -> Self::Screen;
    fn prepare_ray(p: &Self::Params, f: &RayFrame<T>) -> Self::Ray;

    fn eval_screen(s: &Self::Screen, u: Vec2<T>) -> T;
    fn eval_ray(r: &Self::Ray, u: Vec3<T>) -> T;

    #[inline(always)]
    fn d_eval_screen(_s: &Self::Screen, _u: Vec2<T>) -> Option<Vec2<T>> {
        None
    }

    #[inline(always)]
    fn modulate_screen(_s: &Self::Screen, _u: Vec2<T>) -> T {
        T::ONE
    }

    #[inline(always)]
    fn modulate_ray(_r: &Self::Ray, _u: Vec3<T>) -> T {
        T::ONE
    }

    fn support_screen(s: &Self::Screen, peak: T, cutoff: T) -> T;
    fn support_ray(r: &Self::Ray, peak: T, cutoff: T) -> T;

    fn prefilter(
        p: &Self::Params,
        cov: Sym2<T>,
        pixel_variance: T,
    ) -> (Self::Params, Sym2<T>, T);
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KernelDescription {
    pub id: KernelId,
    pub name: &'static str,
    pub class: CompositeClass,
    pub signed: bool,
    pub order_dependent: bool,
    pub radial: bool,
    pub projectable: bool,
    pub modulated: bool,
}

pub fn describe<T: FloatingPoint, K: Kernel<T>>() -> KernelDescription {
    KernelDescription {
        id: K::ID,
        name: K::NAME,
        class: <K::Blend as BlendMode<T>>::CLASS,
        signed: <K::Blend as BlendMode<T>>::SIGNED,
        order_dependent: <K::Blend as BlendMode<T>>::ORDER_DEPENDENT,
        radial: K::RADIAL,
        projectable: K::HAS_SCREEN_MARGINAL,
        modulated: K::MODULATED,
    }
}
