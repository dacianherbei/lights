use crate::numerics::float::FloatingPoint;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type Rgb<T> = Vec3<T>;

impl<T: FloatingPoint> Vec2<T> {
    #[inline(always)]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
    #[inline(always)]
    pub fn splat(v: T) -> Self {
        Self { x: v, y: v }
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(T::ZERO)
    }
    #[inline(always)]
    pub fn dot(self, o: Self) -> T {
        self.x * o.x + self.y * o.y
    }
    #[inline(always)]
    pub fn length_squared(self) -> T {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }
    pub fn normalize(self) -> Option<Self> {
        let n = self.length();
        if n > T::ZERO && n.is_finite() {
            Some(self * n.recip())
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn min_component(self) -> T {
        self.x.min(self.y)
    }
    #[inline(always)]
    pub fn max_component(self) -> T {
        self.x.max(self.y)
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
    pub fn to_f64(self) -> Vec2<f64> {
        Vec2 {
            x: self.x.to_f64(),
            y: self.y.to_f64(),
        }
    }
    pub fn from_f64(v: Vec2<f64>) -> Self {
        Self {
            x: T::from_f64(v.x),
            y: T::from_f64(v.y),
        }
    }
}

impl<T: FloatingPoint> Vec3<T> {
    #[inline(always)]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
    #[inline(always)]
    pub fn splat(v: T) -> Self {
        Self { x: v, y: v, z: v }
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(T::ZERO)
    }
    #[inline(always)]
    pub fn dot(self, o: Self) -> T {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline(always)]
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    #[inline(always)]
    pub fn length_squared(self) -> T {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }
    pub fn normalize(self) -> Option<Self> {
        let n = self.length();
        if n > T::ZERO && n.is_finite() {
            Some(self * n.recip())
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn xy(self) -> Vec2<T> {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
    #[inline(always)]
    pub fn min_component(self) -> T {
        self.x.min(self.y).min(self.z)
    }
    #[inline(always)]
    pub fn max_component(self) -> T {
        self.x.max(self.y).max(self.z)
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    pub fn to_f64(self) -> Vec3<f64> {
        Vec3 {
            x: self.x.to_f64(),
            y: self.y.to_f64(),
            z: self.z.to_f64(),
        }
    }
    pub fn from_f64(v: Vec3<f64>) -> Self {
        Self {
            x: T::from_f64(v.x),
            y: T::from_f64(v.y),
            z: T::from_f64(v.z),
        }
    }
}

macro_rules! vec_ops {
    ($name:ident, $($field:ident),+) => {
        impl<T: FloatingPoint> core::ops::Add for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self { Self { $($field: self.$field + o.$field),+ } }
        }
        impl<T: FloatingPoint> core::ops::Sub for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self { Self { $($field: self.$field - o.$field),+ } }
        }
        impl<T: FloatingPoint> core::ops::Neg for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self { Self { $($field: -self.$field),+ } }
        }
        impl<T: FloatingPoint> core::ops::Mul<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, s: T) -> Self { Self { $($field: self.$field * s),+ } }
        }
        impl<T: FloatingPoint> core::ops::Div<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn div(self, s: T) -> Self { Self { $($field: self.$field / s),+ } }
        }
        impl<T: FloatingPoint> core::ops::AddAssign for $name<T> {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) { $(self.$field += o.$field;)+ }
        }
        impl<T: FloatingPoint> core::ops::SubAssign for $name<T> {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) { $(self.$field -= o.$field;)+ }
        }
    };
}

vec_ops!(Vec2, x, y);
vec_ops!(Vec3, x, y, z);

impl<T: FloatingPoint> core::ops::Index<usize> for Vec2<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            other => crate::guard::unreachable("Vec2::index", &format!("index {other} out of range")),
        }
    }
}

impl<T: FloatingPoint> core::ops::Index<usize> for Vec3<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            other => crate::guard::unreachable("Vec3::index", &format!("index {other} out of range")),
        }
    }
}
