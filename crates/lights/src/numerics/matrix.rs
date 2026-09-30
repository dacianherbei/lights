use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::{Sym2, Sym3};
use crate::numerics::vector::{Vec2, Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat3<T> {
    pub m: [[T; 3]; 3],
}

impl<T: FloatingPoint> Mat3<T> {
    pub fn identity() -> Self {
        let z = T::ZERO;
        let o = T::ONE;
        Self {
            m: [[o, z, z], [z, o, z], [z, z, o]],
        }
    }

    pub fn from_rows(r0: Vec3<T>, r1: Vec3<T>, r2: Vec3<T>) -> Self {
        Self {
            m: [
                [r0.x, r0.y, r0.z],
                [r1.x, r1.y, r1.z],
                [r2.x, r2.y, r2.z],
            ],
        }
    }

    pub fn from_columns(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self {
        Self {
            m: [
                [c0.x, c1.x, c2.x],
                [c0.y, c1.y, c2.y],
                [c0.z, c1.z, c2.z],
            ],
        }
    }

    pub fn row(&self, i: usize) -> Vec3<T> {
        if i > 2 {
            crate::guard::unreachable("Mat3::row", &format!("row {i} out of range"));
        }
        Vec3::new(self.m[i][0], self.m[i][1], self.m[i][2])
    }

    pub fn column(&self, j: usize) -> Vec3<T> {
        if j > 2 {
            crate::guard::unreachable("Mat3::column", &format!("column {j} out of range"));
        }
        Vec3::new(self.m[0][j], self.m[1][j], self.m[2][j])
    }

    pub fn transpose(&self) -> Self {
        let mut out = *self;
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = self.m[j][i];
            }
        }
        out
    }

    #[inline(always)]
    pub fn mul_vec(&self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }

    pub fn mul_mat(&self, o: &Self) -> Self {
        let mut out = *self;
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = T::ZERO;
                for k in 0..3 {
                    acc += self.m[i][k] * o.m[k][j];
                }
                out.m[i][j] = acc;
            }
        }
        out
    }

    pub fn scale_columns(&self, s: Vec3<T>) -> Self {
        let mut out = *self;
        for i in 0..3 {
            out.m[i][0] = self.m[i][0] * s.x;
            out.m[i][1] = self.m[i][1] * s.y;
            out.m[i][2] = self.m[i][2] * s.z;
        }
        out
    }

    pub fn determinant(&self) -> T {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if !(det.is_finite() && det != T::ZERO) {
            return None;
        }
        let m = &self.m;
        let mut out = Self::identity();
        for i in 0..3 {
            for j in 0..3 {
                let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
                let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
                out.m[i][j] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / det;
            }
        }
        if out.is_finite() { Some(out) } else { None }
    }

    pub fn is_orthonormal(&self, tol: T) -> bool {
        let p = self.transpose().mul_mat(self);
        let id = Self::identity();
        for i in 0..3 {
            for j in 0..3 {
                if (p.m[i][j] - id.m[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_finite(&self) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if !self.m[i][j].is_finite() {
                    return false;
                }
            }
        }
        true
    }

    pub fn to_f64(&self) -> Mat3<f64> {
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.m[i][j].to_f64();
            }
        }
        Mat3 { m: out }
    }

    pub fn from_f64(v: &Mat3<f64>) -> Self {
        let mut out = Self::identity();
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = T::from_f64(v.m[i][j]);
            }
        }
        out
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat2x3<T> {
    pub m: [[T; 3]; 2],
}

impl<T: FloatingPoint> Mat2x3<T> {
    pub fn new(m: [[T; 3]; 2]) -> Self {
        Self { m }
    }

    pub fn zero() -> Self {
        Self {
            m: [[T::ZERO; 3]; 2],
        }
    }

    #[inline(always)]
    pub fn mul_vec(&self, v: Vec3<T>) -> Vec2<T> {
        Vec2::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
        )
    }

    pub fn mul_mat3(&self, r: &Mat3<T>) -> Mat2x3<T> {
        let mut out = Self::zero();
        for i in 0..2 {
            for j in 0..3 {
                let mut acc = T::ZERO;
                for k in 0..3 {
                    acc += self.m[i][k] * r.m[k][j];
                }
                out.m[i][j] = acc;
            }
        }
        out
    }

    pub fn congruence(&self, s: &Sym3<T>) -> Sym2<T> {
        s.project_2x3(self)
    }

    pub fn row(&self, i: usize) -> Vec3<T> {
        if i > 1 {
            crate::guard::unreachable("Mat2x3::row", &format!("row {i} out of range"));
        }
        Vec3::new(self.m[i][0], self.m[i][1], self.m[i][2])
    }

    pub fn is_finite(&self) -> bool {
        for i in 0..2 {
            for j in 0..3 {
                if !self.m[i][j].is_finite() {
                    return false;
                }
            }
        }
        true
    }
}
