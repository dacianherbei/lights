use crate::numerics::vector::Rgb;

pub trait FloatingPoint:
    Copy
    + Clone
    + Default
    + PartialEq
    + PartialOrd
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
{
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    const TWO: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const PI: Self;
    const TAU: Self;
    const EPSILON: Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn recip(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, e: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn is_finite(self) -> bool;
    fn to_bits_ordered(self) -> u64;
    fn widen_rgb(pixels: Vec<Rgb<Self>>) -> Vec<Rgb<f64>>;
}

fn widen_rgb_f32(pixels: Vec<Rgb<f32>>) -> Vec<Rgb<f64>> {
    pixels.iter().map(|p| p.to_f64()).collect()
}

fn widen_rgb_f64(pixels: Vec<Rgb<f64>>) -> Vec<Rgb<f64>> {
    pixels
}

macro_rules! impl_floating_point {
    ($t:ty, $widen:expr) => {
        impl FloatingPoint for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const HALF: Self = 0.5;
            const TWO: Self = 2.0;
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const PI: Self = core::f64::consts::PI as $t;
            const TAU: Self = core::f64::consts::TAU as $t;
            const EPSILON: Self = <$t>::EPSILON;

            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            #[inline(always)]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            #[inline(always)]
            fn recip(self) -> Self {
                <$t>::recip(self)
            }
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                <$t>::min(self, other)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }
            #[inline(always)]
            fn clamp(self, lo: Self, hi: Self) -> Self {
                <$t>::clamp(self, lo, hi)
            }
            #[inline(always)]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$t>::mul_add(self, a, b)
            }
            #[inline(always)]
            fn powi(self, n: i32) -> Self {
                <$t>::powi(self, n)
            }
            #[inline(always)]
            fn powf(self, e: Self) -> Self {
                <$t>::powf(self, e)
            }
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                <$t>::sin_cos(self)
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            #[inline(always)]
            fn to_bits_ordered(self) -> u64 {
                <$t>::to_bits(self) as u64
            }
            #[inline(always)]
            fn widen_rgb(pixels: Vec<Rgb<Self>>) -> Vec<Rgb<f64>> {
                $widen(pixels)
            }
        }
    };
}

impl_floating_point!(f32, widen_rgb_f32);
impl_floating_point!(f64, widen_rgb_f64);
