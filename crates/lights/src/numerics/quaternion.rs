use crate::numerics::float::FloatingPoint;
use crate::numerics::matrix::Mat3;
use crate::numerics::vector::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: FloatingPoint> Quat<T> {
    pub fn identity() -> Self {
        Self {
            w: T::ONE,
            x: T::ZERO,
            y: T::ZERO,
            z: T::ZERO,
        }
    }

    pub fn new_wxyz(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    pub fn from_axis_angle(axis: Vec3<T>, radians: T) -> Option<Self> {
        let unit = axis.normalize()?;
        let (s, c) = (radians * T::HALF).sin_cos();
        Some(Self {
            w: c,
            x: unit.x * s,
            y: unit.y * s,
            z: unit.z * s,
        })
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn inverse(self) -> Option<Self> {
        let square = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        if !(square > T::ZERO) || !square.is_finite() {
            return None;
        }
        let inv = square.recip();
        let c = self.conjugate();
        Some(Self {
            w: c.w * inv,
            x: c.x * inv,
            y: c.y * inv,
            z: c.z * inv,
        })
    }

    pub fn rotate(self, v: Vec3<T>) -> Vec3<T> {
        self.to_mat3().mul_vec(v)
    }

    pub fn norm(self) -> T {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(self) -> Option<Self> {
        let n = self.norm();
        if n > T::ZERO && n.is_finite() {
            let inv = n.recip();
            Some(Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            })
        } else {
            None
        }
    }

    pub fn to_mat3(self) -> Mat3<T> {
        let two = T::TWO;
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        Mat3 {
            m: [
                [
                    T::ONE - two * (y * y + z * z),
                    two * (x * y - w * z),
                    two * (x * z + w * y),
                ],
                [
                    two * (x * y + w * z),
                    T::ONE - two * (x * x + z * z),
                    two * (y * z - w * x),
                ],
                [
                    two * (x * z - w * y),
                    two * (y * z + w * x),
                    T::ONE - two * (x * x + y * y),
                ],
            ],
        }
    }

    pub fn from_rotation_matrix(r: &Mat3<T>) -> Option<Self> {
        let m = &r.m;
        let four = T::TWO * T::TWO;
        let trace = m[0][0] + m[1][1] + m[2][2];
        let q = if trace > T::ZERO {
            let s = (trace + T::ONE).sqrt() * T::TWO;
            Self {
                w: s / four,
                x: (m[2][1] - m[1][2]) / s,
                y: (m[0][2] - m[2][0]) / s,
                z: (m[1][0] - m[0][1]) / s,
            }
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (T::ONE + m[0][0] - m[1][1] - m[2][2]).sqrt() * T::TWO;
            Self {
                w: (m[2][1] - m[1][2]) / s,
                x: s / four,
                y: (m[0][1] + m[1][0]) / s,
                z: (m[0][2] + m[2][0]) / s,
            }
        } else if m[1][1] > m[2][2] {
            let s = (T::ONE + m[1][1] - m[0][0] - m[2][2]).sqrt() * T::TWO;
            Self {
                w: (m[0][2] - m[2][0]) / s,
                x: (m[0][1] + m[1][0]) / s,
                y: s / four,
                z: (m[1][2] + m[2][1]) / s,
            }
        } else {
            let s = (T::ONE + m[2][2] - m[0][0] - m[1][1]).sqrt() * T::TWO;
            Self {
                w: (m[1][0] - m[0][1]) / s,
                x: (m[0][2] + m[2][0]) / s,
                y: (m[1][2] + m[2][1]) / s,
                z: s / four,
            }
        };
        q.normalize()
    }

    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn to_f64(self) -> Quat<f64> {
        Quat {
            w: self.w.to_f64(),
            x: self.x.to_f64(),
            y: self.y.to_f64(),
            z: self.z.to_f64(),
        }
    }

    pub fn from_f64(q: Quat<f64>) -> Self {
        Self {
            w: T::from_f64(q.w),
            x: T::from_f64(q.x),
            y: T::from_f64(q.y),
            z: T::from_f64(q.z),
        }
    }
}
