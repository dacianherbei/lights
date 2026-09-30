use crate::numerics::float::FloatingPoint;
use crate::numerics::matrix::Mat3;
use crate::numerics::vector::Vec3;
use crate::scene::Atom;

pub const CAGE_PAD: f64 = 1.03;
pub const TETS_PER_CELL: usize = 6;
pub const FREUDENTHAL_ORDERS: [[usize; 3]; TETS_PER_CELL] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tet<T> {
    pub rest: [Vec3<T>; 4],
    pub linear: Mat3<T>,
    pub offset: Vec3<T>,
    rest_inverse: Mat3<T>,
}

fn edges<T: FloatingPoint>(v: &[Vec3<T>; 4]) -> Mat3<T> {
    Mat3::from_columns(v[1] - v[0], v[2] - v[0], v[3] - v[0])
}

impl<T: FloatingPoint> Tet<T> {
    pub fn new(rest: [Vec3<T>; 4], moved: [Vec3<T>; 4]) -> Option<Self> {
        let rest_inverse = edges(&rest).inverse()?;
        let linear = edges(&moved).mul_mat(&rest_inverse);
        let offset = moved[0] - linear.mul_vec(rest[0]);
        Some(Self {
            rest,
            linear,
            offset,
            rest_inverse,
        })
    }

    pub fn barycentric(&self, p: Vec3<T>) -> [T; 4] {
        let l = self.rest_inverse.mul_vec(p - self.rest[0]);
        [T::ONE - l.x - l.y - l.z, l.x, l.y, l.z]
    }

    pub fn map_point(&self, p: Vec3<T>) -> Vec3<T> {
        self.linear.mul_vec(p) + self.offset
    }
}

fn axis<T: FloatingPoint>(v: Vec3<T>, k: usize) -> T {
    [v.x, v.y, v.z][k]
}

fn min_of<T: FloatingPoint>(b: &[T; 4]) -> T {
    b[0].min(b[1]).min(b[2]).min(b[3])
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cage<T> {
    pub lo: Vec3<T>,
    pub step: Vec3<T>,
    pub cells: [usize; 3],
    tets: Vec<Tet<T>>,
}

impl<T: FloatingPoint> Cage<T> {
    pub fn grid<F: Fn(Vec3<T>) -> Vec3<T>>(
        lo: Vec3<T>,
        hi: Vec3<T>,
        cells: [usize; 3],
        deform: F,
    ) -> Option<Self> {
        if cells.contains(&0) {
            return None;
        }
        let step = Vec3::new(
            (hi.x - lo.x) / T::from_f64(cells[0] as f64),
            (hi.y - lo.y) / T::from_f64(cells[1] as f64),
            (hi.z - lo.z) / T::from_f64(cells[2] as f64),
        );
        if !(step.min_component() > T::ZERO && step.is_finite()) {
            return None;
        }
        let vertex = |i: [usize; 3]| {
            Vec3::new(
                lo.x + step.x * T::from_f64(i[0] as f64),
                lo.y + step.y * T::from_f64(i[1] as f64),
                lo.z + step.z * T::from_f64(i[2] as f64),
            )
        };
        let mut tets = Vec::with_capacity(cells[0] * cells[1] * cells[2] * TETS_PER_CELL);
        for cz in 0..cells[2] {
            for cy in 0..cells[1] {
                for cx in 0..cells[0] {
                    for order in FREUDENTHAL_ORDERS.iter() {
                        let mut index = [cx, cy, cz];
                        let mut rest = [vertex(index); 4];
                        for (n, &k) in order.iter().enumerate() {
                            index[k] += 1;
                            rest[n + 1] = vertex(index);
                        }
                        let moved = [
                            deform(rest[0]),
                            deform(rest[1]),
                            deform(rest[2]),
                            deform(rest[3]),
                        ];
                        tets.push(Tet::new(rest, moved)?);
                    }
                }
            }
        }
        Some(Self {
            lo,
            step,
            cells,
            tets,
        })
    }

    pub fn around<F: Fn(Vec3<T>) -> Vec3<T>>(
        atoms: &[Atom<T>],
        cells: [usize; 3],
        deform: F,
    ) -> Option<Self> {
        let first = atoms.first()?.mean;
        let (mut lo, mut hi) = (first, first);
        for a in atoms {
            lo = Vec3::new(lo.x.min(a.mean.x), lo.y.min(a.mean.y), lo.z.min(a.mean.z));
            hi = Vec3::new(hi.x.max(a.mean.x), hi.y.max(a.mean.y), hi.z.max(a.mean.z));
        }
        let pad = T::from_f64(CAGE_PAD);
        let centre = (lo + hi) * T::HALF;
        let half = (hi - lo) * T::HALF;
        let widest = half.max_component().max(T::ONE);
        let floor = widest / T::from_f64(64.0);
        let half = Vec3::new(
            (half.x * pad).max(floor),
            (half.y * pad).max(floor),
            (half.z * pad).max(floor),
        );
        Self::grid(centre - half, centre + half, cells, deform)
    }

    pub fn tets(&self) -> &[Tet<T>] {
        &self.tets
    }

    pub fn locate(&self, p: Vec3<T>) -> usize {
        let mut cell = [0usize; 3];
        for (k, c) in cell.iter_mut().enumerate() {
            let t = ((axis(p, k) - axis(self.lo, k)) / axis(self.step, k)).floor().to_f64();
            let top = (self.cells[k] - 1) as f64;
            *c = if t.is_finite() { t.clamp(0.0, top) as usize } else { 0 };
        }
        let base = ((cell[2] * self.cells[1] + cell[1]) * self.cells[0] + cell[0]) * TETS_PER_CELL;
        let mut best = base;
        let mut best_min = min_of(&self.tets[base].barycentric(p));
        for index in base + 1..base + TETS_PER_CELL {
            let m = min_of(&self.tets[index].barycentric(p));
            if m > best_min {
                best = index;
                best_min = m;
            }
        }
        best
    }

    pub fn deform_atom(&self, atom: &Atom<T>) -> Option<Atom<T>> {
        let tet = &self.tets[self.locate(atom.mean)];
        atom.mapped_affine(&tet.linear, tet.offset)
    }

    pub fn deform(&self, atoms: &[Atom<T>]) -> Option<Vec<Atom<T>>> {
        atoms.iter().map(|a| self.deform_atom(a)).collect()
    }
}
