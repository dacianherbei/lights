use lights::numerics::float::FloatingPoint;
use lights::numerics::matrix::Mat3;
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::Sym3;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::Atom;

const CASES: usize = 10_000;
const POINTS_PER_ATOM: usize = 16;
const STATED_KAPPA_ULPS: f64 = 256.0;

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

    fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + self.next() * (hi.ln() - lo.ln())).exp()
    }

    fn rotation(&mut self) -> Quat<f64> {
        Quat::new_wxyz(self.signed(), self.signed(), self.signed(), self.signed())
            .normalize()
            .expect("a random quaternion of zero length")
    }
}

fn explicit_congruence(a: &Mat3<f64>, s: &Sym3<f64>) -> [[f64; 3]; 3] {
    let full = [
        [s.xx, s.xy, s.xz],
        [s.xy, s.yy, s.yz],
        [s.xz, s.yz, s.zz],
    ];
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    acc += a.m[i][k] * full[k][l] * a.m[j][l];
                }
            }
            *cell = acc;
        }
    }
    out
}

fn full(s: &Sym3<f64>) -> [[f64; 3]; 3] {
    [
        [s.xx, s.xy, s.xz],
        [s.xy, s.yy, s.yz],
        [s.xz, s.yz, s.zz],
    ]
}

fn max_abs(m: &[[f64; 3]; 3]) -> f64 {
    m.iter().flatten().fold(0.0f64, |acc, v| acc.max(v.abs()))
}

fn spd_case(stream: &mut Stream, index: usize) -> Sym3<f64> {
    let q = stream.rotation();
    let s = match index % 5 {
        0 => {
            let v = stream.log_uniform(1e-2, 1e1);
            Vec3::new(v, v, v)
        }
        1 => {
            let v = stream.log_uniform(1e-2, 1e1);
            Vec3::new(v, v, stream.log_uniform(1e-2, 1e1))
        }
        2 => Vec3::new(
            stream.log_uniform(1e-3, 1e3),
            stream.log_uniform(1e-3, 1e3),
            stream.log_uniform(1e-3, 1e3),
        ),
        3 => Vec3::new(
            stream.log_uniform(1e-2, 1e1),
            stream.log_uniform(1e-2, 1e1),
            stream.log_uniform(1e-2, 1e1),
        ),
        4 => Vec3::new(
            stream.log_uniform(1e-2, 1e1),
            stream.log_uniform(1e-2, 1e1),
            stream.log_uniform(1e-2, 1e1),
        ),
        other => panic!("case class {other} is not one of the five"),
    };
    let rotation = if index % 5 == 4 { Quat::identity() } else { q };
    Atom::new(Vec3::zero(), rotation, s, 1.0, Rgb::new(1.0, 1.0, 1.0)).covariance()
}

fn eigen_errors<T: FloatingPoint>(sigma: &Sym3<f64>) -> (f64, f64, f64) {
    let narrowed = Sym3::<T>::from_f64(sigma);
    let (values, vectors) = narrowed.eigen().expect("eigen refused a finite matrix");
    let v = vectors.to_f64();
    let lambda = values.to_f64();
    let mut rebuilt = [[0.0; 3]; 3];
    for (i, row) in rebuilt.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = v.m[i][0] * lambda.x * v.m[j][0]
                + v.m[i][1] * lambda.y * v.m[j][1]
                + v.m[i][2] * lambda.z * v.m[j][2];
        }
    }
    let target = full(&narrowed.to_f64());
    let mut gap = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            gap[i][j] = rebuilt[i][j] - target[i][j];
        }
    }
    let product = v.transpose().mul_mat(&v);
    let identity = Mat3::<f64>::identity();
    let mut ortho = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            ortho = ortho.max((product.m[i][j] - identity.m[i][j]).abs());
        }
    }
    (max_abs(&gap) / max_abs(&target), ortho, v.determinant())
}

fn eigen_sweep<T: FloatingPoint>(label: &str, reconstruction_bound: f64, ortho_bound: f64) {
    let mut stream = Stream(0x9E37_79B9_7F4A_7C15);
    let (mut worst_rebuild, mut worst_ortho, mut worst_det) = (0.0f64, 0.0f64, 0.0f64);
    for index in 0..CASES {
        let sigma = spd_case(&mut stream, index);
        let (rebuild, ortho, det) = eigen_errors::<T>(&sigma);
        worst_rebuild = worst_rebuild.max(rebuild);
        worst_ortho = worst_ortho.max(ortho);
        worst_det = worst_det.max((det - 1.0).abs());
        assert!(
            det > 0.0,
            "{label} case {index}: the eigenvectors have determinant {det}, so they are a \
             reflection and no quaternion can carry them"
        );
    }
    println!(
        "EIGEN {label} cases {CASES} worst reconstruction {worst_rebuild:e} (relative to max |entry|) \
         worst orthonormality {worst_ortho:e} worst |det-1| {worst_det:e}"
    );
    assert!(
        worst_rebuild <= reconstruction_bound,
        "{label}: V diag(lambda) V^T misses sigma by {worst_rebuild:e} relative, above the stated \
         bound {reconstruction_bound:e}"
    );
    assert!(
        worst_ortho <= ortho_bound,
        "{label}: V^T V misses the identity by {worst_ortho:e}, above the stated bound {ortho_bound:e}"
    );
}

#[test]
fn the_eigensolver_rebuilds_every_symmetric_matrix_as_a_proper_rotation_and_a_diagonal() {
    eigen_sweep::<f64>("f64", 64.0 * f64::EPSILON, 64.0 * f64::EPSILON);
    eigen_sweep::<f32>("f32", 64.0 * f32::EPSILON as f64, 64.0 * f32::EPSILON as f64);
}

#[test]
fn the_eigensolver_answers_the_matrices_that_need_no_rotation() {
    for sigma in [
        Sym3::<f64>::new(4.0, 0.0, 0.0, 1.0, 0.0, 9.0),
        Sym3::<f64>::new(2.0, 0.0, 0.0, 2.0, 0.0, 2.0),
        Sym3::<f64>::zero(),
    ] {
        let (values, vectors) = sigma.eigen().expect("eigen refused a diagonal matrix");
        assert_eq!(vectors, Mat3::identity(), "a diagonal matrix needs no rotation");
        assert_eq!(values, sigma.diagonal(), "and its eigenvalues are its diagonal");
    }
    assert!(
        Sym3::<f64>::new(f64::NAN, 0.0, 0.0, 1.0, 0.0, 1.0)
            .eigen()
            .is_none(),
        "a non-finite matrix is refused, not decomposed"
    );
}

fn affine(stream: &mut Stream, index: usize) -> (Mat3<f64>, Vec3<f64>) {
    let offset = Vec3::new(stream.signed(), stream.signed(), stream.signed());
    let flip = if index % 4 == 3 { -1.0 } else { 1.0 };
    let m = [
        [flip + 0.9 * stream.signed(), 0.9 * stream.signed(), 0.9 * stream.signed()],
        [0.9 * stream.signed(), 1.0 + 0.9 * stream.signed(), 0.9 * stream.signed()],
        [0.9 * stream.signed(), 0.9 * stream.signed(), 1.0 + 0.9 * stream.signed()],
    ];
    (Mat3 { m }, offset)
}

#[test]
fn an_affine_map_of_an_atom_is_an_atom_and_its_field_is_the_pulled_back_field() {
    let mut stream = Stream(0xD1B5_4A32_D192_ED03);
    let (mut worst_cov, mut worst_field, mut worst_quadratic) = (0.0f64, 0.0f64, 0.0f64);
    let (mut mapped_count, mut reflections, mut sheared) = (0usize, 0usize, 0usize);
    let (mut worst_kappa, mut worst_static, mut worst_moved) = (0.0f64, 0.0f64, 0.0f64);
    for index in 0..CASES {
        let (linear, offset) = affine(&mut stream, index);
        let det = linear.determinant();
        if det.abs() < 0.05 {
            continue;
        }
        let symmetric = (0..3).all(|i| (0..3).all(|j| linear.m[i][j] == linear.m[j][i]));
        if !symmetric {
            sheared += 1;
        }
        if det < 0.0 {
            reflections += 1;
        }
        let atom = Atom::new(
            Vec3::new(stream.signed(), stream.signed(), 3.0 + stream.signed()),
            stream.rotation(),
            Vec3::new(
                stream.log_uniform(1e-2, 1.0),
                stream.log_uniform(1e-2, 1.0),
                stream.log_uniform(1e-2, 1.0),
            ),
            0.8,
            Rgb::new(0.3, 0.6, 0.9),
        );
        let mapped = atom
            .mapped_affine(&linear, offset)
            .expect("a non-singular map refused an atom");
        mapped_count += 1;

        let expected = explicit_congruence(&linear, &atom.covariance());
        let got = full(&mapped.covariance());
        let mut gap = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                gap[i][j] = got[i][j] - expected[i][j];
            }
        }
        worst_cov = worst_cov.max(max_abs(&gap) / max_abs(&expected));

        let original_inverse = atom.covariance().inverse_spd().expect("original precision");
        let mapped_inverse = mapped.covariance().inverse_spd().expect("mapped precision");
        let root = atom.rotation.to_mat3().scale_columns(atom.scale);
        let condition = |a: &Atom<f64>| (a.scale.max_component() / a.scale.min_component()).powi(2);
        let (kappa_static, kappa_moved) = (condition(&atom), condition(&mapped));
        worst_kappa = worst_kappa.max(kappa_moved);
        for _ in 0..POINTS_PER_ATOM {
            let u = Vec3::new(3.0 * stream.signed(), 3.0 * stream.signed(), 3.0 * stream.signed());
            let x = atom.mean + root.mul_vec(u);
            let y = linear.mul_vec(x) + offset;
            let q_exact = u.dot(u);
            let q_static = original_inverse.quad_form(x - atom.mean);
            let q_moved = mapped_inverse.quad_form(y - mapped.mean);
            let floor = q_exact.max(1.0);
            worst_quadratic = worst_quadratic.max((q_moved - q_static).abs() / floor);
            worst_static = worst_static.max((q_static - q_exact).abs() / floor / (kappa_static * f64::EPSILON));
            worst_moved = worst_moved.max((q_moved - q_exact).abs() / floor / (kappa_moved * f64::EPSILON));
            worst_field = worst_field.max((((-0.5) * q_moved).exp() - ((-0.5) * q_static).exp()).abs());
        }
    }
    println!(
        "CLOSURE f64 atoms {mapped_count} sheared {sheared} reflections {reflections} \
         worst covariance {worst_cov:e} (relative to max |entry|) worst quadratic form \
         {worst_quadratic:e} (relative, floor 1), against the exact u.u the static path misses by {worst_static:.3} and \
         the moved path by {worst_moved:.3} x its own kappa x eps (moved kappa up to {worst_kappa:e}), \
         worst field {worst_field:e} (absolute, peak 1)"
    );
    assert!(
        mapped_count > CASES / 2 && sheared == mapped_count && reflections > 0,
        "the sweep must be dominated by sheared maps and include reflections, or it proves only \
         the similarity case: {mapped_count} mapped, {sheared} sheared, {reflections} reflections"
    );
    assert!(
        worst_cov <= 1e-12,
        "the re-extracted atom's covariance misses A Sigma A^T by {worst_cov:e} relative"
    );
    assert!(
        worst_moved <= STATED_KAPPA_ULPS && worst_static <= STATED_KAPPA_ULPS,
        "against the exact quadratic form u.u, the moved atom misses by {worst_moved:.3} and the \
         static atom by {worst_static:.3} x kappa x eps, and the stated bound is {STATED_KAPPA_ULPS}: \
         the only error licensed is the conditioning of inverting a covariance, which both paths pay"
    );
}

#[test]
fn a_singular_map_is_refused_or_flattened_never_invented() {
    let atom = Atom::new(
        Vec3::new(0.1, 0.2, 3.0),
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7).expect("axis"),
        Vec3::new(0.3, 0.2, 0.1),
        0.8,
        Rgb::new(0.3, 0.6, 0.9),
    );
    let flatten = Mat3 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    };
    match atom.mapped_affine(&flatten, Vec3::zero()) {
        None => {}
        Some(mapped) => {
            assert!(
                mapped.scale.min_component() <= 1e-6 * mapped.scale.max_component(),
                "a rank-2 map produced scale {:?}, which is a volume it cannot have",
                mapped.scale
            );
            assert!(
                mapped.validate(lights::settings::MIN_SCALE).is_err(),
                "and the scene refuses to take it"
            );
        }
    }
}
