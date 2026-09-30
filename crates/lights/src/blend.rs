use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::Rgb;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompositeClass {
    Over,
    Additive,
    Normalised,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EarlyOut {
    Reference,
    Exact,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Continue,
    Stop,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlendSettings<T> {
    pub alpha_cap: T,
    pub alpha_cutoff: T,
    pub transmittance_cutoff: T,
    pub early_out: EarlyOut,
}

impl<T: FloatingPoint> BlendSettings<T> {
    pub fn reference() -> Self {
        Self {
            alpha_cap: T::from_f64(crate::settings::REFERENCE_ALPHA_CAP),
            alpha_cutoff: T::from_f64(crate::settings::REFERENCE_ALPHA_CUTOFF),
            transmittance_cutoff: T::from_f64(crate::settings::REFERENCE_TRANSMITTANCE_CUTOFF),
            early_out: EarlyOut::Reference,
        }
    }

    pub fn contract() -> Self {
        Self {
            early_out: EarlyOut::Exact,
            ..Self::reference()
        }
    }

    pub fn error_ceiling(&self) -> T {
        match self.early_out {
            EarlyOut::Exact => self.transmittance_cutoff,
            EarlyOut::Reference => self.transmittance_cutoff / (T::ONE - self.alpha_cap),
        }
    }

    pub fn from_f64(s: &BlendSettings<f64>) -> Self {
        Self {
            alpha_cap: T::from_f64(s.alpha_cap),
            alpha_cutoff: T::from_f64(s.alpha_cutoff),
            transmittance_cutoff: T::from_f64(s.transmittance_cutoff),
            early_out: s.early_out,
        }
    }

    pub fn to_f64(&self) -> BlendSettings<f64> {
        BlendSettings {
            alpha_cap: self.alpha_cap.to_f64(),
            alpha_cutoff: self.alpha_cutoff.to_f64(),
            transmittance_cutoff: self.transmittance_cutoff.to_f64(),
            early_out: self.early_out,
        }
    }
}

pub trait BlendMode<T: FloatingPoint>:
    Copy + Clone + core::fmt::Debug + Send + Sync + 'static
{
    type Accum: Copy + Clone + Send + Sync + core::fmt::Debug;

    const CLASS: CompositeClass;
    const SIGNED: bool;
    const ORDER_DEPENDENT: bool;
    const NAME: &'static str;

    fn init() -> Self::Accum;

    fn step(
        acc: &mut Self::Accum,
        weight: T,
        colour: Rgb<T>,
        opacity: T,
        s: &BlendSettings<T>,
    ) -> Step;

    fn saturated(acc: &Self::Accum, s: &BlendSettings<T>) -> bool;

    fn resolve(acc: &Self::Accum, background: Rgb<T>) -> Rgb<T>;

    fn coverage(acc: &Self::Accum) -> T;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OverAccum<T> {
    pub rgb: Rgb<T>,
    pub transmittance: T,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Over;

impl<T: FloatingPoint> BlendMode<T> for Over {
    type Accum = OverAccum<T>;

    const CLASS: CompositeClass = CompositeClass::Over;
    const SIGNED: bool = false;
    const ORDER_DEPENDENT: bool = true;
    const NAME: &'static str = "over";

    #[inline(always)]
    fn init() -> Self::Accum {
        OverAccum {
            rgb: Rgb::zero(),
            transmittance: T::ONE,
        }
    }

    #[inline(always)]
    fn step(
        acc: &mut Self::Accum,
        weight: T,
        colour: Rgb<T>,
        opacity: T,
        s: &BlendSettings<T>,
    ) -> Step {
        let alpha = (opacity * weight).min(s.alpha_cap);
        if alpha < s.alpha_cutoff {
            return Step::Continue;
        }
        match s.early_out {
            EarlyOut::Reference => {
                let next = acc.transmittance * (T::ONE - alpha);
                if next < s.transmittance_cutoff {
                    return Step::Stop;
                }
                acc.rgb += colour * (alpha * acc.transmittance);
                acc.transmittance = next;
                Step::Continue
            }
            EarlyOut::Exact => {
                acc.rgb += colour * (alpha * acc.transmittance);
                acc.transmittance = acc.transmittance * (T::ONE - alpha);
                if acc.transmittance < s.transmittance_cutoff {
                    Step::Stop
                } else {
                    Step::Continue
                }
            }
        }
    }

    #[inline(always)]
    fn saturated(acc: &Self::Accum, s: &BlendSettings<T>) -> bool {
        acc.transmittance < s.transmittance_cutoff
    }

    #[inline(always)]
    fn resolve(acc: &Self::Accum, background: Rgb<T>) -> Rgb<T> {
        acc.rgb + background * acc.transmittance
    }

    #[inline(always)]
    fn coverage(acc: &Self::Accum) -> T {
        T::ONE - acc.transmittance
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Additive;

impl<T: FloatingPoint> BlendMode<T> for Additive {
    type Accum = Rgb<T>;

    const CLASS: CompositeClass = CompositeClass::Additive;
    const SIGNED: bool = true;
    const ORDER_DEPENDENT: bool = false;
    const NAME: &'static str = "additive";

    #[inline(always)]
    fn init() -> Self::Accum {
        Rgb::zero()
    }

    #[inline(always)]
    fn step(
        acc: &mut Self::Accum,
        weight: T,
        colour: Rgb<T>,
        opacity: T,
        _s: &BlendSettings<T>,
    ) -> Step {
        *acc += colour * (opacity * weight);
        Step::Continue
    }

    #[inline(always)]
    fn saturated(_acc: &Self::Accum, _s: &BlendSettings<T>) -> bool {
        false
    }

    #[inline(always)]
    fn resolve(acc: &Self::Accum, background: Rgb<T>) -> Rgb<T> {
        *acc + background
    }

    #[inline(always)]
    fn coverage(_acc: &Self::Accum) -> T {
        T::ZERO
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NormalisedAccum<T> {
    pub weighted: Rgb<T>,
    pub weight: T,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Normalised;

impl<T: FloatingPoint> BlendMode<T> for Normalised {
    type Accum = NormalisedAccum<T>;

    const CLASS: CompositeClass = CompositeClass::Normalised;
    const SIGNED: bool = false;
    const ORDER_DEPENDENT: bool = false;
    const NAME: &'static str = "normalised";

    #[inline(always)]
    fn init() -> Self::Accum {
        NormalisedAccum {
            weighted: Rgb::zero(),
            weight: T::ZERO,
        }
    }

    #[inline(always)]
    fn step(
        acc: &mut Self::Accum,
        weight: T,
        colour: Rgb<T>,
        opacity: T,
        s: &BlendSettings<T>,
    ) -> Step {
        let w = opacity * weight;
        if w < s.alpha_cutoff {
            return Step::Continue;
        }
        acc.weighted += colour * w;
        acc.weight += w;
        Step::Continue
    }

    #[inline(always)]
    fn saturated(_acc: &Self::Accum, _s: &BlendSettings<T>) -> bool {
        false
    }

    #[inline(always)]
    fn resolve(acc: &Self::Accum, background: Rgb<T>) -> Rgb<T> {
        if !(acc.weight > T::ZERO) {
            return background;
        }
        let c = <Self as BlendMode<T>>::coverage(acc);
        let mean = acc.weighted * acc.weight.recip();
        background * (T::ONE - c) + mean * c
    }

    #[inline(always)]
    fn coverage(acc: &Self::Accum) -> T {
        acc.weight.min(T::ONE)
    }
}
