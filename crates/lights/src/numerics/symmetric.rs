use crate::numerics::float::FloatingPoint;
use crate::numerics::matrix::{Mat2x3, Mat3};
use crate::numerics::vector::{Vec2, Vec3};

pub const JACOBI_SWEEPS: usize = 32;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Sym2<T> {
    pub xx: T,
    pub xy: T,
    pub yy: T,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Sym3<T> {
    pub xx: T,
    pub xy: T,
    pub xz: T,
    pub yy: T,
    pub yz: T,
    pub zz: T,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LowerTri2<T> {
    pub a: T,
    pub b: T,
    pub c: T,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LowerTri3<T> {
    pub m: [T; 6],
}

impl<T: FloatingPoint> Sym2<T> {
    pub fn new(xx: T, xy: T, yy: T) -> Self {
        Self { xx, xy, yy }
    }

    pub fn zero() -> Self {
        Self {
            xx: T::ZERO,
            xy: T::ZERO,
            yy: T::ZERO,
        }
    }

    #[inline(always)]
    pub fn determinant(&self) -> T {
        self.xx * self.yy - self.xy * self.xy
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if !(det > T::ZERO) || !det.is_finite() {
            return None;
        }
        let inv = det.recip();
        Some(Self {
            xx: self.yy * inv,
            xy: -self.xy * inv,
            yy: self.xx * inv,
        })
    }

    pub fn add_diagonal(&self, d: T) -> Self {
        Self {
            xx: self.xx + d,
            xy: self.xy,
            yy: self.yy + d,
        }
    }

    #[inline(always)]
    pub fn quad_form(&self, v: Vec2<T>) -> T {
        self.xx * v.x * v.x + T::TWO * self.xy * v.x * v.y + self.yy * v.y * v.y
    }

    pub fn cholesky(&self) -> Option<LowerTri2<T>> {
        if !(self.xx > T::ZERO) {
            return None;
        }
        let a = self.xx.sqrt();
        let b = self.xy / a;
        let cc = self.yy - b * b;
        if !(cc > T::ZERO) {
            return None;
        }
        let c = cc.sqrt();
        if !(a.is_finite() && b.is_finite() && c.is_finite()) {
            return None;
        }
        Some(LowerTri2 { a, b, c })
    }

    pub fn eigenvalues(&self) -> (T, T) {
        let mid = (self.xx + self.yy) * T::HALF;
        let disc = (mid * mid - self.determinant()).max(T::ZERO);
        let root = disc.sqrt();
        (mid + root, mid - root)
    }

    pub fn diagonal(&self) -> Vec2<T> {
        Vec2::new(self.xx, self.yy)
    }

    pub fn is_finite(&self) -> bool {
        self.xx.is_finite() && self.xy.is_finite() && self.yy.is_finite()
    }

    pub fn to_f64(&self) -> Sym2<f64> {
        Sym2 {
            xx: self.xx.to_f64(),
            xy: self.xy.to_f64(),
            yy: self.yy.to_f64(),
        }
    }
}

impl<T: FloatingPoint> Sym3<T> {
    pub fn new(xx: T, xy: T, xz: T, yy: T, yz: T, zz: T) -> Self {
        Self {
            xx,
            xy,
            xz,
            yy,
            yz,
            zz,
        }
    }

    pub fn zero() -> Self {
        Self::new(
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
        )
    }

    pub fn from_factor(m: &Mat3<T>) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        Self {
            xx: r0.dot(r0),
            xy: r0.dot(r1),
            xz: r0.dot(r2),
            yy: r1.dot(r1),
            yz: r1.dot(r2),
            zz: r2.dot(r2),
        }
    }

    pub fn congruence(&self, w: &Mat3<T>) -> Self {
        let r0 = w.row(0);
        let r1 = w.row(1);
        let r2 = w.row(2);
        let s0 = self.apply(r0);
        let s1 = self.apply(r1);
        let s2 = self.apply(r2);
        Self {
            xx: r0.dot(s0),
            xy: r0.dot(s1),
            xz: r0.dot(s2),
            yy: r1.dot(s1),
            yz: r1.dot(s2),
            zz: r2.dot(s2),
        }
    }

    pub fn project_2x3(&self, j: &Mat2x3<T>) -> Sym2<T> {
        let r0 = j.row(0);
        let r1 = j.row(1);
        let s0 = self.apply(r0);
        let s1 = self.apply(r1);
        Sym2 {
            xx: r0.dot(s0),
            xy: r0.dot(s1),
            yy: r1.dot(s1),
        }
    }

    pub fn marginal_xy(&self) -> Sym2<T> {
        Sym2 {
            xx: self.xx,
            xy: self.xy,
            yy: self.yy,
        }
    }

    pub fn cholesky(&self) -> Option<LowerTri3<T>> {
        if !(self.xx > T::ZERO) {
            return None;
        }
        let l00 = self.xx.sqrt();
        let l10 = self.xy / l00;
        let l20 = self.xz / l00;
        let d11 = self.yy - l10 * l10;
        if !(d11 > T::ZERO) {
            return None;
        }
        let l11 = d11.sqrt();
        let l21 = (self.yz - l20 * l10) / l11;
        let d22 = self.zz - l20 * l20 - l21 * l21;
        if !(d22 > T::ZERO) {
            return None;
        }
        let l22 = d22.sqrt();
        let out = LowerTri3 {
            m: [l00, l10, l11, l20, l21, l22],
        };
        if out.is_finite() { Some(out) } else { None }
    }

    pub fn from_lower_transpose(li: &LowerTri3<T>) -> Self {
        let i00 = li.m[0];
        let i10 = li.m[1];
        let i11 = li.m[2];
        let i20 = li.m[3];
        let i21 = li.m[4];
        let i22 = li.m[5];
        Self {
            xx: i00 * i00 + i10 * i10 + i20 * i20,
            xy: i10 * i11 + i20 * i21,
            xz: i20 * i22,
            yy: i11 * i11 + i21 * i21,
            yz: i21 * i22,
            zz: i22 * i22,
        }
    }

    pub fn inverse_spd(&self) -> Option<Self> {
        let li = self.cholesky()?.inverse();
        let i00 = li.m[0];
        let i10 = li.m[1];
        let i11 = li.m[2];
        let i20 = li.m[3];
        let i21 = li.m[4];
        let i22 = li.m[5];
        let out = Self {
            xx: i00 * i00 + i10 * i10 + i20 * i20,
            xy: i10 * i11 + i20 * i21,
            xz: i20 * i22,
            yy: i11 * i11 + i21 * i21,
            yz: i21 * i22,
            zz: i22 * i22,
        };
        if out.is_finite() { Some(out) } else { None }
    }

    #[inline(always)]
    pub fn apply(&self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(
            self.xx * v.x + self.xy * v.y + self.xz * v.z,
            self.xy * v.x + self.yy * v.y + self.yz * v.z,
            self.xz * v.x + self.yz * v.y + self.zz * v.z,
        )
    }

    #[inline(always)]
    pub fn quad_form(&self, v: Vec3<T>) -> T {
        v.dot(self.apply(v))
    }

    pub fn diagonal(&self) -> Vec3<T> {
        Vec3::new(self.xx, self.yy, self.zz)
    }

    pub fn max_eigenvalue_bound(&self) -> T {
        let a = self.xx + self.xy.abs() + self.xz.abs();
        let b = self.yy + self.xy.abs() + self.yz.abs();
        let c = self.zz + self.xz.abs() + self.yz.abs();
        a.max(b).max(c)
    }

    pub fn min_eigenvalue_bound(&self) -> T {
        let a = self.xx - self.xy.abs() - self.xz.abs();
        let b = self.yy - self.xy.abs() - self.yz.abs();
        let c = self.zz - self.xz.abs() - self.yz.abs();
        a.min(b).min(c)
    }

    pub fn eigen(&self) -> Option<(Vec3<T>, Mat3<T>)> {
        if !self.is_finite() {
            return None;
        }
        let mut a = [
            [self.xx, self.xy, self.xz],
            [self.xy, self.yy, self.yz],
            [self.xz, self.yz, self.zz],
        ];
        let mut v = Mat3::<T>::identity().m;
        let scale = self.xx.abs() + self.yy.abs() + self.zz.abs() + self.xy.abs() + self.xz.abs()
            + self.yz.abs();
        if scale == T::ZERO {
            return Some((Vec3::new(T::ZERO, T::ZERO, T::ZERO), Mat3 { m: v }));
        }
        let tiny = T::EPSILON * T::EPSILON * scale * scale;
        for _sweep in 0..JACOBI_SWEEPS {
            let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
            if off <= tiny {
                break;
            }
            for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
                let apq = a[p][q];
                if apq == T::ZERO {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (T::TWO * apq);
                let sign = if theta >= T::ZERO { T::ONE } else { -T::ONE };
                let t = sign / (theta.abs() + (theta * theta + T::ONE).sqrt());
                let c = (t * t + T::ONE).sqrt().recip();
                let s = t * c;
                for row in a.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
                for k in 0..3 {
                    let (pk, qk) = (a[p][k], a[q][k]);
                    a[p][k] = c * pk - s * qk;
                    a[q][k] = s * pk + c * qk;
                }
                for row in v.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
            }
        }
        let vectors = Mat3 { m: v };
        let values = Vec3::new(a[0][0], a[1][1], a[2][2]);
        if !(values.is_finite() && vectors.is_finite()) {
            return None;
        }
        Some((values, vectors))
    }

    pub fn is_finite(&self) -> bool {
        self.xx.is_finite()
            && self.xy.is_finite()
            && self.xz.is_finite()
            && self.yy.is_finite()
            && self.yz.is_finite()
            && self.zz.is_finite()
    }

    pub fn to_f64(&self) -> Sym3<f64> {
        Sym3 {
            xx: self.xx.to_f64(),
            xy: self.xy.to_f64(),
            xz: self.xz.to_f64(),
            yy: self.yy.to_f64(),
            yz: self.yz.to_f64(),
            zz: self.zz.to_f64(),
        }
    }

    pub fn from_f64(s: &Sym3<f64>) -> Self {
        Self {
            xx: T::from_f64(s.xx),
            xy: T::from_f64(s.xy),
            xz: T::from_f64(s.xz),
            yy: T::from_f64(s.yy),
            yz: T::from_f64(s.yz),
            zz: T::from_f64(s.zz),
        }
    }
}

impl<T: FloatingPoint> LowerTri2<T> {
    #[inline(always)]
    pub fn apply(&self, v: Vec2<T>) -> Vec2<T> {
        Vec2::new(self.a * v.x, self.b * v.x + self.c * v.y)
    }

    #[inline(always)]
    pub fn apply_transpose(&self, v: Vec2<T>) -> Vec2<T> {
        Vec2::new(self.a * v.x + self.b * v.y, self.c * v.y)
    }

    pub fn inverse(&self) -> Self {
        let ia = self.a.recip();
        let ic = self.c.recip();
        Self {
            a: ia,
            b: -self.b * ia * ic,
            c: ic,
        }
    }

    pub fn is_finite(&self) -> bool {
        self.a.is_finite() && self.b.is_finite() && self.c.is_finite()
    }
}

impl<T: FloatingPoint> LowerTri3<T> {
    pub fn identity() -> Self {
        Self {
            m: [T::ONE, T::ZERO, T::ONE, T::ZERO, T::ZERO, T::ONE],
        }
    }

    #[inline(always)]
    pub fn apply(&self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(
            self.m[0] * v.x,
            self.m[1] * v.x + self.m[2] * v.y,
            self.m[3] * v.x + self.m[4] * v.y + self.m[5] * v.z,
        )
    }

    #[inline(always)]
    pub fn solve(&self, v: Vec3<T>) -> Vec3<T> {
        let x0 = v.x / self.m[0];
        let x1 = (v.y - self.m[1] * x0) / self.m[2];
        let x2 = (v.z - self.m[3] * x0 - self.m[4] * x1) / self.m[5];
        Vec3::new(x0, x1, x2)
    }

    pub fn inverse(&self) -> Self {
        let a = self.m[0];
        let b = self.m[1];
        let c = self.m[2];
        let d = self.m[3];
        let e = self.m[4];
        let f = self.m[5];
        let ia = a.recip();
        let ic = c.recip();
        let if_ = f.recip();
        Self {
            m: [
                ia,
                -b * ia * ic,
                ic,
                (b * e - c * d) * ia * ic * if_,
                -e * ic * if_,
                if_,
            ],
        }
    }

    pub fn determinant(&self) -> T {
        self.m[0] * self.m[2] * self.m[5]
    }

    pub fn is_finite(&self) -> bool {
        self.m.iter().all(|v| v.is_finite())
    }
}
