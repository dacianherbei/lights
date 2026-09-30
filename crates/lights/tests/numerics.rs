use lights::numerics::float::FloatingPoint;
use lights::numerics::matrix::{Mat2x3, Mat3};
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::{Sym2, Sym3};
use lights::numerics::vector::{Vec2, Vec3};

fn probe_values_f64() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 4.0,
        f64::EPSILON,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e-300,
        1e300,
        core::f64::consts::PI,
    ];
    let mut state: u64 = 0x1234_5678_9abc_def0;
    while v.len() < 256 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let m = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        v.push((m - 0.5) * 40.0);
    }
    v
}

#[test]
fn float_impl_parity() {
    let values = probe_values_f64();
    for raw in values.iter() {
        let a = *raw as f64;
        assert_eq!(FloatingPoint::abs(a).to_bits(), f64::abs(a).to_bits());
        assert_eq!(FloatingPoint::sqrt(a).to_bits(), f64::sqrt(a).to_bits());
        assert_eq!(FloatingPoint::exp(a).to_bits(), f64::exp(a).to_bits());
        assert_eq!(FloatingPoint::ln(a).to_bits(), f64::ln(a).to_bits());
        assert_eq!(FloatingPoint::floor(a).to_bits(), f64::floor(a).to_bits());
        assert_eq!(FloatingPoint::ceil(a).to_bits(), f64::ceil(a).to_bits());
        assert_eq!(FloatingPoint::recip(a).to_bits(), f64::recip(a).to_bits());
        assert_eq!(FloatingPoint::powi(a, 3).to_bits(), f64::powi(a, 3).to_bits());
        assert_eq!(FloatingPoint::powf(a, 1.5).to_bits(), f64::powf(a, 1.5).to_bits());
        assert_eq!(
            FloatingPoint::mul_add(a, 2.0, 3.0).to_bits(),
            f64::mul_add(a, 2.0, 3.0).to_bits()
        );
        assert_eq!(FloatingPoint::is_finite(a), f64::is_finite(a));
        assert_eq!(FloatingPoint::min(a, 1.25).to_bits(), f64::min(a, 1.25).to_bits());
        assert_eq!(FloatingPoint::max(a, 1.25).to_bits(), f64::max(a, 1.25).to_bits());
        assert_eq!(
            FloatingPoint::clamp(a, -2.0, 2.0).to_bits(),
            if a.is_nan() { a.to_bits() } else { f64::clamp(a, -2.0, 2.0).to_bits() }
        );
        let (s1, c1) = FloatingPoint::sin_cos(a);
        let (s2, c2) = f64::sin_cos(a);
        assert_eq!(s1.to_bits(), s2.to_bits());
        assert_eq!(c1.to_bits(), c2.to_bits());
        assert_eq!(FloatingPoint::to_bits_ordered(a), f64::to_bits(a));
        assert_eq!(FloatingPoint::to_f64(a).to_bits(), a.to_bits());

        let b = *raw as f32;
        assert_eq!(FloatingPoint::abs(b).to_bits(), f32::abs(b).to_bits());
        assert_eq!(FloatingPoint::sqrt(b).to_bits(), f32::sqrt(b).to_bits());
        assert_eq!(FloatingPoint::exp(b).to_bits(), f32::exp(b).to_bits());
        assert_eq!(FloatingPoint::ln(b).to_bits(), f32::ln(b).to_bits());
        assert_eq!(FloatingPoint::floor(b).to_bits(), f32::floor(b).to_bits());
        assert_eq!(FloatingPoint::ceil(b).to_bits(), f32::ceil(b).to_bits());
        assert_eq!(FloatingPoint::recip(b).to_bits(), f32::recip(b).to_bits());
        assert_eq!(FloatingPoint::powi(b, 3).to_bits(), f32::powi(b, 3).to_bits());
        assert_eq!(FloatingPoint::powf(b, 1.5).to_bits(), f32::powf(b, 1.5).to_bits());
        assert_eq!(
            FloatingPoint::mul_add(b, 2.0, 3.0).to_bits(),
            f32::mul_add(b, 2.0, 3.0).to_bits()
        );
        assert_eq!(FloatingPoint::is_finite(b), f32::is_finite(b));
        assert_eq!(FloatingPoint::min(b, 1.25).to_bits(), f32::min(b, 1.25).to_bits());
        assert_eq!(FloatingPoint::max(b, 1.25).to_bits(), f32::max(b, 1.25).to_bits());
        let (s3, c3) = FloatingPoint::sin_cos(b);
        let (s4, c4) = f32::sin_cos(b);
        assert_eq!(s3.to_bits(), s4.to_bits());
        assert_eq!(c3.to_bits(), c4.to_bits());
        assert_eq!(FloatingPoint::to_bits_ordered(b), f32::to_bits(b) as u64);
    }

    assert_eq!(<f64 as FloatingPoint>::ZERO, 0.0f64);
    assert_eq!(<f32 as FloatingPoint>::ZERO, 0.0f32);
    assert_eq!(<f64 as FloatingPoint>::HALF, 0.5f64);
    assert_eq!(<f32 as FloatingPoint>::TWO, 2.0f32);
    assert_eq!(<f64 as FloatingPoint>::PI, core::f64::consts::PI);
    assert_eq!(<f32 as FloatingPoint>::PI, core::f32::consts::PI);
    assert_eq!(<f64 as FloatingPoint>::TAU, core::f64::consts::TAU);
    assert!(<f64 as FloatingPoint>::INFINITY.is_infinite());
    assert!(<f32 as FloatingPoint>::NEG_INFINITY.is_infinite());
    assert_eq!(<f32 as FloatingPoint>::from_f64(0.25), 0.25f32);
}

#[test]
fn to_bits_ordered_is_monotone_on_positives() {
    let mut values: Vec<f64> = Vec::new();
    let mut state: u64 = 99;
    for _ in 0..1000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let m = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        values.push(1.0e-6 + m * 1.0e4);
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    values.dedup();
    for w in values.windows(2) {
        assert!(
            w[0].to_bits_ordered() < w[1].to_bits_ordered(),
            "f64 order broken at {} {}",
            w[0],
            w[1]
        );
        let a = w[0] as f32;
        let b = w[1] as f32;
        if a < b {
            assert!(a.to_bits_ordered() < b.to_bits_ordered(), "f32 order broken");
        }
    }
    let biggest_positive = values[values.len() - 1];
    for negative in [-1.0e-9f64, -1.0f64, -1.0e9f64] {
        assert!(
            negative.to_bits_ordered() > biggest_positive.to_bits_ordered(),
            "a negative depth key must sort after every positive one: this is why \
             the near-plane cull is a prerequisite for the bit-trick sort"
        );
    }
}

#[test]
fn vec_ops_generic_at_both_precisions() {
    fn check<T: FloatingPoint>(tol: f64) {
        let a = Vec3::<T>::new(T::from_f64(1.0), T::from_f64(2.0), T::from_f64(3.0));
        let b = Vec3::<T>::new(T::from_f64(-4.0), T::from_f64(5.0), T::from_f64(6.0));
        assert!((a.dot(b).to_f64() - 24.0).abs() <= tol);
        let c = a.cross(b);
        assert!((c.x.to_f64() - (-3.0)).abs() <= tol);
        assert!((c.y.to_f64() - (-18.0)).abs() <= tol);
        assert!((c.z.to_f64() - 13.0).abs() <= tol);
        assert!((a.length_squared().to_f64() - 14.0).abs() <= tol);
        assert!((a.length().to_f64() - 14.0f64.sqrt()).abs() <= tol);
        let n = a.normalize().expect("nonzero");
        assert!((n.length().to_f64() - 1.0).abs() <= tol);
        assert!(Vec3::<T>::zero().normalize().is_none());
        let scaled = a * T::from_f64(2.0);
        assert!((scaled.y.to_f64() - 4.0).abs() <= tol);
        let negated = -a;
        assert!((negated.z.to_f64() + 3.0).abs() <= tol);
        assert!((a[0].to_f64() - 1.0).abs() <= tol);
        assert!((a[2].to_f64() - 3.0).abs() <= tol);
        assert!((a.xy().y.to_f64() - 2.0).abs() <= tol);

        let p = Vec2::<T>::new(T::from_f64(3.0), T::from_f64(4.0));
        assert!((p.length().to_f64() - 5.0).abs() <= tol);
        assert!((p.dot(p).to_f64() - 25.0).abs() <= tol);
        assert!((p.max_component().to_f64() - 4.0).abs() <= tol);
        assert!((p.min_component().to_f64() - 3.0).abs() <= tol);
        assert!((p[1].to_f64() - 4.0).abs() <= tol);
        let d = (p / T::from_f64(2.0)).x.to_f64();
        assert!((d - 1.5).abs() <= tol);
        let mut m = a;
        m += b;
        assert!((m.x.to_f64() - (-3.0)).abs() <= tol);
        m -= b;
        assert!((m.x.to_f64() - 1.0).abs() <= tol);
    }
    check::<f64>(1e-12);
    check::<f32>(1e-5);
}

#[test]
fn tv_s7_quaternion_half_angle() {
    let half = (22.5f64).to_radians();
    let q45 = Quat::<f64>::new_wxyz(half.cos(), 0.0, 0.0, half.sin());
    let r = q45.to_mat3();
    let s = core::f64::consts::FRAC_1_SQRT_2;
    let expected = [[s, -s, 0.0], [s, s, 0.0], [0.0, 0.0, 1.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (r.m[i][j] - expected[i][j]).abs() < 1e-12,
                "45deg row {i} col {j}: {} vs {}",
                r.m[i][j],
                expected[i][j]
            );
        }
    }

    let quarter = (45.0f64).to_radians();
    let q90 = Quat::<f64>::new_wxyz(quarter.cos(), 0.0, 0.0, quarter.sin());
    let r90 = q90.to_mat3();
    let mapped = r90.mul_vec(Vec3::new(1.0, 0.0, 0.0));
    assert!((mapped.x - 0.0).abs() < 1e-12);
    assert!((mapped.y - 1.0).abs() < 1e-12);
    assert!((mapped.z - 0.0).abs() < 1e-12);
    assert!((r90.m[0][1] + 1.0).abs() < 1e-12);
    assert!((r90.m[1][0] - 1.0).abs() < 1e-12);
}

#[test]
fn quaternion_scalar_first_and_identity_does_not_discriminate() {
    let identity = Quat::<f64>::new_wxyz(1.0, 0.0, 0.0, 0.0);
    let r = identity.to_mat3();
    assert_eq!(r, Mat3::<f64>::identity());

    let misread = Quat::<f64>::new_wxyz(0.0, 1.0, 0.0, 0.0);
    let m = misread.to_mat3();
    let expected = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((m.m[i][j] - expected[i][j]).abs() < 1e-15);
        }
    }
    let scale = Vec3::new(2.0f64, 1.0, 0.5);
    let good = Sym3::from_factor(&r.scale_columns(scale));
    let bad = Sym3::from_factor(&m.scale_columns(scale));
    assert_eq!(
        good, bad,
        "the identity quaternion CANNOT detect the component-order bug: \
         recording this is the point"
    );

    let axis = Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f64::consts::PI / 2.0)
        .expect("unit axis");
    assert!((axis.w - (core::f64::consts::PI / 4.0).cos()).abs() < 1e-15);
    assert!((axis.z - (core::f64::consts::PI / 4.0).sin()).abs() < 1e-15);
    assert!(Quat::<f64>::from_axis_angle(Vec3::zero(), 1.0).is_none());
    assert!(Quat::<f64>::new_wxyz(0.0, 0.0, 0.0, 0.0).normalize().is_none());
    let unnormalised = Quat::<f64>::new_wxyz(2.0, 0.0, 0.0, 0.0);
    assert!((unnormalised.norm() - 2.0).abs() < 1e-15);
    let fixed = unnormalised.normalize().expect("nonzero");
    assert_eq!(fixed.to_mat3(), Mat3::<f64>::identity());
}

#[test]
fn tv_s6_covariance_order() {
    let half = (22.5f64).to_radians();
    let q = Quat::<f64>::new_wxyz(half.cos(), 0.0, 0.0, half.sin());
    let r = q.to_mat3();
    let scale = Vec3::new(2.0f64, 1.0, 1.0);

    let sigma = Sym3::from_factor(&r.scale_columns(scale));
    assert!((sigma.xx - 2.5).abs() < 1e-12, "xx {}", sigma.xx);
    assert!((sigma.yy - 2.5).abs() < 1e-12, "yy {}", sigma.yy);
    assert!((sigma.zz - 1.0).abs() < 1e-12, "zz {}", sigma.zz);
    assert!(
        (sigma.xy - 1.5).abs() < 1e-12,
        "Sigma01 must be +1.5, got {}",
        sigma.xy
    );
    assert!(sigma.xz.abs() < 1e-12 && sigma.yz.abs() < 1e-12);

    let mut rows_scaled = r;
    for j in 0..3 {
        rows_scaled.m[0][j] *= scale.x;
        rows_scaled.m[1][j] *= scale.y;
        rows_scaled.m[2][j] *= scale.z;
    }
    let s_times_r = Sym3::from_factor(&rows_scaled);
    assert!((s_times_r.xx - 4.0).abs() < 1e-12);
    assert!((s_times_r.yy - 1.0).abs() < 1e-12);
    assert!(
        s_times_r.xy.abs() < 1e-12,
        "S*R DELETES the rotation and still produces a valid PSD matrix"
    );
    assert_ne!(sigma, s_times_r);

    let transposed = Sym3::from_factor(&r.transpose().scale_columns(scale));
    assert!(
        (transposed.xy + 1.5).abs() < 1e-12,
        "R transposed gives -1.5, got {}",
        transposed.xy
    );
    assert_ne!(sigma, transposed);
}

fn deterministic_spd(seed: u64) -> Sym3<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
    };
    let m = Mat3::<f64> {
        m: [
            [next() * 2.0, next(), next()],
            [next(), next() * 2.0, next()],
            [next(), next(), next() * 2.0],
        ],
    };
    let base = Sym3::from_factor(&m);
    Sym3::new(
        base.xx + 0.5,
        base.xy,
        base.xz,
        base.yy + 0.5,
        base.yz,
        base.zz + 0.5,
    )
}

#[test]
fn sym3_cholesky_roundtrip_and_rank_deficiency() {
    for seed in 0..500u64 {
        let sigma = deterministic_spd(seed);
        let lambda = match sigma.inverse_spd() {
            Some(l) => l,
            None => panic!("seed {seed} produced a non-SPD matrix"),
        };
        let back = lambda.inverse_spd().expect("inverse of an SPD inverse is SPD");
        for (a, b) in [
            (sigma.xx, back.xx),
            (sigma.xy, back.xy),
            (sigma.xz, back.xz),
            (sigma.yy, back.yy),
            (sigma.yz, back.yz),
            (sigma.zz, back.zz),
        ] {
            assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()), "seed {seed}: {a} vs {b}");
        }
        let chol = sigma.cholesky().expect("SPD");
        let inv = chol.inverse();
        let v = Vec3::new(0.37, -1.2, 0.8);
        let whitened = inv.apply(v);
        assert!(
            (whitened.length_squared() - lambda.quad_form(v)).abs() < 1e-9,
            "whitening must reproduce the Mahalanobis form"
        );
        let solved = chol.solve(chol.apply(v));
        assert!((solved - v).length() < 1e-10);
    }

    let rank_deficient = Sym3::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    assert!(rank_deficient.cholesky().is_none());
    assert!(rank_deficient.inverse_spd().is_none());
    let indefinite = Sym3::new(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);
    assert!(indefinite.cholesky().is_none());
    let negative_leading = Sym3::new(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
    assert!(negative_leading.cholesky().is_none());
}

#[test]
fn sym2_inverse_sign_and_swap_pattern() {
    let s = Sym2::<f64>::new(4.0, 1.0, 3.0);
    let det = s.determinant();
    assert!((det - 11.0).abs() < 1e-14);
    let inv = s.inverse().expect("nonsingular");
    assert!(
        (inv.xx - 3.0 / 11.0).abs() < 1e-14,
        "the DIAGONALS SWAP: inv.xx must come from yy, got {}",
        inv.xx
    );
    assert!((inv.yy - 4.0 / 11.0).abs() < 1e-14);
    assert!((inv.xy + 1.0 / 11.0).abs() < 1e-14);
    assert!(Sym2::<f64>::new(1.0, 1.0, 1.0).inverse().is_none());

    let indefinite = Sym2::<f64>::new(1.0, 2.0, 1.0);
    assert!(
        indefinite.determinant() < 0.0,
        "this matrix must have a NEGATIVE determinant for the check below to \
         mean anything: {}",
        indefinite.determinant()
    );
    assert!(
        indefinite.inverse().is_none(),
        "S10 spells the guard as the ONE-SIDED !(det > eps) precisely so a sign \
         slip cannot survive. An |det| > 0 guard admits a negative determinant \
         and returns an INDEFINITE 'conic' - a mirrored, non-positive-definite \
         quadratic form rather than a covariance inverse, whose level sets are \
         hyperbolae. Sym2::inverse is public and ScreenFrame::conic is a public \
         field, so nothing downstream is entitled to assume a caller went \
         through the Cholesky first"
    );
    assert!(
        indefinite.cholesky().is_none(),
        "cholesky already tests !(xx > 0) and !(cc > 0) one-sidedly; inverse \
         must agree with it on the same input"
    );

    let chol = s.cholesky().expect("SPD");
    let li = chol.inverse();
    let v = Vec2::new(0.7, -0.3);
    assert!((li.apply(v).length_squared() - inv.quad_form(v)).abs() < 1e-13);
    let dilated = s.add_diagonal(0.3);
    assert!((dilated.xx - 4.3).abs() < 1e-14 && (dilated.yy - 3.3).abs() < 1e-14);
    assert!((dilated.xy - 1.0).abs() < 1e-14);
    let (hi, lo) = s.eigenvalues();
    assert!(hi > lo);
    assert!((hi * lo - det).abs() < 1e-12);
    assert!((hi + lo - 7.0).abs() < 1e-12);
}

#[test]
fn tv_r9_marginal_not_conditional_matrix_level() {
    let sigma = Sym3::<f64>::new(2.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let marginal = sigma.marginal_xy();
    assert!((marginal.xx - 2.0).abs() < 1e-14);
    assert!((marginal.xy - 0.0).abs() < 1e-14);
    assert!((marginal.yy - 1.0).abs() < 1e-14);

    let lambda = sigma.inverse_spd().expect("SPD");
    let conditional_block = Sym2::new(lambda.xx, lambda.xy, lambda.yy);
    let conditional = conditional_block.inverse().expect("nonsingular");
    assert!((conditional.xx - 1.0).abs() < 1e-12);
    assert!((conditional.yy - 1.0).abs() < 1e-12);
    assert_ne!(
        marginal, conditional,
        "TV-R9 exists because the marginal and the conditional differ here"
    );
    assert!(
        (marginal.xx - conditional.xx).abs() > 0.5,
        "the conditional value must NOT be produced by marginal_xy"
    );

    let decoupled = Sym3::<f64>::new(2.0, 0.0, 0.0, 1.0, 0.0, 1.0);
    let dl = decoupled.inverse_spd().expect("SPD");
    let dcond = Sym2::new(dl.xx, dl.xy, dl.yy).inverse().expect("nonsingular");
    let dmarg = decoupled.marginal_xy();
    assert!(
        (dmarg.xx - dcond.xx).abs() < 1e-12,
        "with zero xz and yz the two COINCIDE, which is exactly the case a naive \
         test would use"
    );
}

#[test]
fn mat3_and_mat2x3_basics() {
    let a = Mat3::<f64> {
        m: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
    };
    assert!((a.determinant() - (-3.0)).abs() < 1e-12);
    assert_eq!(a.transpose().transpose(), a);
    assert_eq!(a.row(1), Vec3::new(4.0, 5.0, 6.0));
    assert_eq!(a.column(2), Vec3::new(3.0, 6.0, 10.0));
    assert_eq!(a.mul_mat(&Mat3::identity()), a);
    assert_eq!(a.mul_vec(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(1.0, 4.0, 7.0));
    let scaled = a.scale_columns(Vec3::new(2.0, 1.0, 1.0));
    assert_eq!(scaled.column(0), Vec3::new(2.0, 8.0, 14.0));
    assert_eq!(scaled.column(1), a.column(1));
    assert!(!a.is_orthonormal(1e-9));
    assert!(Mat3::<f64>::identity().is_orthonormal(1e-12));
    assert!(a.is_finite());

    let j = Mat2x3::<f64>::new([[50.0, 0.0, -5.0], [0.0, 50.0, -10.0]]);
    assert_eq!(j.mul_vec(Vec3::new(1.0, 1.0, 1.0)), Vec2::new(45.0, 40.0));
    let sigma = Sym3::<f64>::new(0.04, 0.0, 0.0, 0.01, 0.0, 0.0025);
    let screen = j.congruence(&sigma);
    assert!((screen.xx - 100.0625).abs() < 1e-12, "xx {}", screen.xx);
    assert!((screen.yy - 25.25).abs() < 1e-12, "yy {}", screen.yy);
    assert!(
        (screen.xy - 0.125).abs() < 1e-12,
        "the perspective shear term comes ENTIRELY from the third column of J; \
         a zero here means it was dropped. got {}",
        screen.xy
    );
    assert_eq!(screen, sigma.project_2x3(&j));
}

#[test]
fn tv_s9_third_jacobian_row_is_irrelevant() {
    let sigma_world = Sym3::<f64>::new(0.025, 0.015, 0.0, 0.025, 0.0, 0.0025);
    let angle = (30.0f64).to_radians();
    let w = Mat3::<f64> {
        m: [
            [angle.cos(), 0.0, angle.sin()],
            [0.0, 1.0, 0.0],
            [-angle.sin(), 0.0, angle.cos()],
        ],
    };
    let sigma_cam = sigma_world.congruence(&w);
    let mu = w.mul_vec(Vec3::new(0.5, 0.3, 8.0));
    let f = 600.0f64;
    let j = Mat2x3::<f64>::new([
        [f / mu.z, 0.0, -f * mu.x / (mu.z * mu.z)],
        [0.0, f / mu.z, -f * mu.y / (mu.z * mu.z)],
    ]);
    let two_by_three = sigma_cam.project_2x3(&j);

    for third in [Vec3::new(0.0, 0.0, 1.0), mu.normalize().expect("nonzero")] {
        let j3 = Mat3::from_rows(j.row(0), j.row(1), third);
        let full = sigma_cam.congruence(&j3);
        assert!((full.xx - two_by_three.xx).abs() < 1e-13);
        assert!((full.xy - two_by_three.xy).abs() < 1e-13);
        assert!((full.yy - two_by_three.yy).abs() < 1e-13);
    }
    assert!(
        (two_by_three.xx - 289.7033860835).abs() < 1e-8,
        "got {}",
        two_by_three.xx
    );
}

#[test]
fn tv_s14_determinant_guards_are_nan_safe() {
    let nan = Sym2::<f64>::new(f64::NAN, 0.0, 1.0);
    assert!(
        nan.inverse().is_none(),
        "a NaN covariance must be CULLED. `det == 0.0` does not catch NaN \
         (NaN == 0.0 is false) and neither does `det <= eps` (NaN <= x is \
         false); only the NEGATED form !(det > eps) is false on NaN."
    );
    let det = nan.determinant();
    assert!(
        !(det == 0.0),
        "recorded: a bare equality guard would fall through here"
    );
    assert!(
        !(det <= 1e-9),
        "recorded: a `det <= eps` guard would also fall through here"
    );
    assert!(!(det > 1e-9), "the negated form is the only one that culls");

    assert!(Sym2::<f64>::new(f64::INFINITY, 0.0, 1.0).inverse().is_none());
    assert!(Sym2::<f64>::new(1.0, f64::NAN, 1.0).inverse().is_none());

    let cancelling = Sym2::<f32>::new(1.0e8, 1.0e8, 1.0e8);
    assert_eq!(
        cancelling.determinant(),
        0.0,
        "f32 cancellation at 1e8 loses the determinant entirely"
    );
    assert!(
        cancelling.inverse().is_none(),
        "the cancellation sub-case and the NaN sub-case test DIFFERENT things \
         and neither is redundant"
    );

    assert!(Sym3::<f64>::new(f64::NAN, 0.0, 0.0, 1.0, 0.0, 1.0).cholesky().is_none());
    assert!(
        Sym3::<f64>::new(f64::INFINITY, 0.0, 0.0, 1.0, 0.0, 1.0)
            .cholesky()
            .is_none()
    );
    assert!(
        Sym3::<f64>::new(1.0, 0.0, 0.0, 1.0, f64::NAN, 1.0)
            .inverse_spd()
            .is_none()
    );
}

#[test]
fn cholesky_rejects_a_non_positive_third_pivot() {
    let flat = Sym3::<f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    assert!(
        flat.cholesky().is_none(),
        "a covariance with no extent in z is rank deficient in the THIRD pivot, \
         which is a different branch from the first two"
    );
    let negative_z = Sym3::<f64>::new(1.0, 0.0, 0.0, 1.0, 0.0, -1.0);
    assert!(negative_z.cholesky().is_none());
    let coupled = Sym3::<f64>::new(1.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    assert!(
        coupled.cholesky().is_none(),
        "d22 = zz - l20^2 = 0 here while both earlier pivots are positive"
    );
    let barely = Sym3::<f64>::new(1.0, 0.0, 1.0, 1.0, 0.0, 1.0 + 1e-9);
    assert!(barely.cholesky().is_some(), "and a hair above zero is accepted");

    let pivot_only = Sym3::<f64>::new(1.0, 0.0, 2.0, 1.0, 0.0, 1.0);
    assert!(
        pivot_only.cholesky().is_none(),
        "d22 = 1 - 4 = -3: the pivot guard and the is_finite net are two \
         independent reasons this returns None, and either alone suffices"
    );
}

#[test]
fn composing_two_rotations_is_the_product_of_their_matrices() {
    let a = match Quat::<f64>::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.7) {
        Some(q) => q,
        None => panic!("axis-angle failed"),
    };
    let b = match Quat::<f64>::from_axis_angle(Vec3::new(0.36, 0.48, 0.8), 1.1) {
        Some(q) => q,
        None => panic!("axis-angle failed"),
    };

    let composed = a.mul(b).to_mat3();
    let product = a.to_mat3().mul_mat(&b.to_mat3());
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (composed.m[i][j] - product.m[i][j]).abs() < 1.0e-14,
                "(a*b) must rotate exactly as a-then-b: element ({i},{j}) is {} against {}. \
                 The order is the matrix order — a.mul(b) applies b FIRST — and a quaternion \
                 type that cannot compose cannot nudge an orientation by a delta, which is \
                 what a scene that changes needs",
                composed.m[i][j],
                product.m[i][j]
            );
        }
    }

    let back = match b.inverse() {
        Some(q) => q,
        None => panic!("a unit quaternion must be invertible"),
    };
    let round = a.mul(b).mul(back);
    assert!(
        (round.w - a.w).abs() < 1.0e-14
            && (round.x - a.x).abs() < 1.0e-14
            && (round.y - a.y).abs() < 1.0e-14
            && (round.z - a.z).abs() < 1.0e-14,
        "q * r * r^-1 must return q, got {round:?} against {a:?}"
    );

    let mut walked = Quat::<f64>::identity();
    let step = match Quat::<f64>::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.05) {
        Some(q) => q,
        None => panic!("axis-angle failed"),
    };
    for _ in 0..40 {
        walked = walked.mul(step);
    }
    let once = match Quat::<f64>::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 2.0) {
        Some(q) => q,
        None => panic!("axis-angle failed"),
    };
    let point = Vec3::new(0.3, -0.7, 1.1);
    let stepped = walked.rotate(point);
    let direct = once.rotate(point);
    assert!(
        (stepped - direct).length() < 1.0e-13,
        "forty deltas of 0.05 rad must land where one rotation of 2.0 rad does: {stepped:?} \
         against {direct:?}. This is the per-frame case, and it is the one that accumulates"
    );

    let degenerate = Quat::<f64>::new_wxyz(0.0, 0.0, 0.0, 0.0);
    assert!(
        degenerate.inverse().is_none(),
        "a zero quaternion has no inverse and must say so rather than returning infinities"
    );
}
