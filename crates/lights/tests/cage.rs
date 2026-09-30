use lights::numerics::matrix::Mat3;
use lights::numerics::quaternion::Quat;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::Atom;
use lights::scene::cage::Cage;

const CELLS: [usize; 3] = [4, 3, 2];
const RANDOM_ATOMS: usize = 2_000;
const AGREEMENT: f64 = 1e-12;

struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

fn swirl(p: Vec3<f64>) -> Vec3<f64> {
    let r2 = p.x * p.x + p.y * p.y;
    let (s, c) = (0.8 * (-r2 / 2.0).exp() + 0.3 * p.z).sin_cos();
    Vec3::new(
        c * p.x - s * p.y + 0.2 * p.z * p.z,
        s * p.x + c * p.y,
        p.z + 0.1 * p.x * p.y,
    )
}

fn lo() -> Vec3<f64> {
    Vec3::new(-2.0, -1.5, -1.0)
}

fn hi() -> Vec3<f64> {
    Vec3::new(2.0, 1.5, 1.0)
}

fn kuhn_interpolation(p: Vec3<f64>) -> Vec3<f64> {
    let (lo, hi) = (lo(), hi());
    let lo_a = [lo.x, lo.y, lo.z];
    let step = [
        (hi.x - lo.x) / CELLS[0] as f64,
        (hi.y - lo.y) / CELLS[1] as f64,
        (hi.z - lo.z) / CELLS[2] as f64,
    ];
    let q = [p.x, p.y, p.z];
    let mut cell = [0usize; 3];
    let mut frac = [0.0f64; 3];
    for k in 0..3 {
        let t = (q[k] - lo_a[k]) / step[k];
        let c = t.floor().clamp(0.0, (CELLS[k] - 1) as f64);
        cell[k] = c as usize;
        frac[k] = t - c;
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| frac[b].total_cmp(&frac[a]));
    let corner = |i: [usize; 3]| {
        swirl(Vec3::new(
            lo_a[0] + step[0] * i[0] as f64,
            lo_a[1] + step[1] * i[1] as f64,
            lo_a[2] + step[2] * i[2] as f64,
        ))
    };
    let weights = [
        1.0 - frac[order[0]],
        frac[order[0]] - frac[order[1]],
        frac[order[1]] - frac[order[2]],
        frac[order[2]],
    ];
    let mut index = cell;
    let mut out = corner(index) * weights[0];
    for n in 0..3 {
        index[order[n]] += 1;
        out = out + corner(index) * weights[n + 1];
    }
    out
}

fn atom_at(mean: Vec3<f64>) -> Atom<f64> {
    Atom::new(
        mean,
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7).expect("axis"),
        Vec3::new(0.05, 0.03, 0.02),
        0.8,
        Rgb::new(0.3, 0.6, 0.9),
    )
}

fn seam_points(stream: &mut Stream) -> Vec<Vec3<f64>> {
    let (lo, hi) = (lo(), hi());
    let at = |fx: f64, fy: f64, fz: f64| {
        Vec3::new(
            lo.x + (hi.x - lo.x) * fx,
            lo.y + (hi.y - lo.y) * fy,
            lo.z + (hi.z - lo.z) * fz,
        )
    };
    let mut points = Vec::new();
    for _ in 0..RANDOM_ATOMS {
        points.push(at(stream.next(), stream.next(), stream.next()));
    }
    for _ in 0..RANDOM_ATOMS / 4 {
        let (cx, cy, cz) = (
            (stream.next() * CELLS[0] as f64).floor(),
            (stream.next() * CELLS[1] as f64).floor(),
            (stream.next() * CELLS[2] as f64).floor(),
        );
        let (a, b) = (stream.next(), stream.next());
        points.push(at(
            (cx + a) / CELLS[0] as f64,
            (cy + a) / CELLS[1] as f64,
            (cz + b) / CELLS[2] as f64,
        ));
        points.push(at(
            (cx + 1.0) / CELLS[0] as f64,
            (cy + a) / CELLS[1] as f64,
            (cz + b) / CELLS[2] as f64,
        ));
    }
    points
}

#[test]
fn a_cage_moves_every_atom_mean_by_the_piecewise_affine_map_its_vertices_define_across_every_seam() {
    let cage = Cage::grid(lo(), hi(), CELLS, swirl).expect("the cage did not build");
    assert_eq!(cage.tets().len(), CELLS[0] * CELLS[1] * CELLS[2] * 6);
    let mut stream = Stream(0x5EED_CA6E_0000_0169);
    let points = seam_points(&mut stream);
    let mut worst = 0.0f64;
    let mut outside = 0usize;
    for (n, p) in points.iter().enumerate() {
        let tet = &cage.tets()[cage.locate(*p)];
        let smallest = tet.barycentric(*p).iter().fold(f64::INFINITY, |m, v| m.min(*v));
        if smallest < -AGREEMENT {
            outside += 1;
        }
        let moved = cage.deform_atom(&atom_at(*p)).expect("a non-degenerate tet refused an atom");
        let expected = kuhn_interpolation(*p);
        let gap = (moved.mean - expected).dot(moved.mean - expected).sqrt();
        worst = worst.max(gap);
        assert!(
            gap <= AGREEMENT,
            "point {n} at {p:?}: the cage moved the mean to {:?} and the piecewise-affine \
             interpolation of the vertex images is {expected:?}, {gap:e} apart - the atom took \
             a tet that does not contain it",
            moved.mean
        );
    }
    println!(
        "CAGE seam: {} points, {} of them on cell faces and Freudenthal diagonals, worst gap {worst:e}, \
         {outside} located outside their tet",
        points.len(),
        points.len() - RANDOM_ATOMS
    );
    assert_eq!(outside, 0, "every point inside the cage is located in a tet that contains it");
}

#[test]
fn a_cage_built_from_an_affine_map_is_that_map_on_every_atom() {
    let linear = Mat3 {
        m: [[1.2, 0.4, -0.3], [-0.2, 0.9, 0.5], [0.3, -0.1, 1.1]],
    };
    let offset = Vec3::new(0.5, -0.25, 2.0);
    let cage = Cage::grid(lo(), hi(), CELLS, |p| linear.mul_vec(p) + offset).expect("cage");
    let mut stream = Stream(0xAFF1_4E00_0000_0169);
    let mut worst = 0.0f64;
    for p in seam_points(&mut stream) {
        let atom = atom_at(p);
        let caged = cage.deform_atom(&atom).expect("cage refused");
        let direct = atom.mapped_affine(&linear, offset).expect("direct refused");
        let (a, b) = (caged.covariance(), direct.covariance());
        let cov_gap = [a.xx - b.xx, a.xy - b.xy, a.xz - b.xz, a.yy - b.yy, a.yz - b.yz, a.zz - b.zz]
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let mean_gap = (caged.mean - direct.mean).dot(caged.mean - direct.mean).sqrt();
        worst = worst.max(cov_gap).max(mean_gap);
    }
    println!("CAGE affine: worst gap against the one global map {worst:e}");
    assert!(worst <= AGREEMENT, "a cage of an affine map differs from that map by {worst:e}");
}

#[test]
fn a_cage_refuses_what_it_cannot_build() {
    assert!(Cage::grid(lo(), hi(), [0, 1, 1], swirl).is_none(), "zero cells");
    assert!(Cage::grid(lo(), lo(), CELLS, swirl).is_none(), "zero extent");
    assert!(
        Cage::grid(lo(), hi(), CELLS, |p| Vec3::new(p.x, p.y, 0.0)).is_some(),
        "a flattening deform still builds, because the rest tets are sound"
    );
    let flat = Cage::grid(lo(), hi(), CELLS, |p| Vec3::new(p.x, p.y, 0.0)).expect("flat cage");
    assert!(
        flat.deform_atom(&atom_at(Vec3::zero())).is_none(),
        "and it refuses to invent a volume for an atom it flattens"
    );
    assert!(Cage::<f64>::around(&[], CELLS, swirl).is_none(), "no atoms, no cage");
}
