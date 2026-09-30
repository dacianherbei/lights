use lights::camera::{Camera, CameraSpec, Projection};
use lights::numerics::matrix::Mat3;
use lights::numerics::symmetric::Sym3;
use lights::numerics::vector::{Vec2, Vec3};
use lights::settings::PixelConvention;

fn forward_camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        width,
        height,
    )
}

#[test]
fn look_at_builds_a_right_handed_z_forward_y_down_frame() {
    let spec = forward_camera(800, 600);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    assert_eq!(cam.rotation, Mat3::<f64>::identity());
    assert_eq!(cam.translation, Vec3::zero());
    assert_eq!(cam.to_camera(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));

    let offset = CameraSpec::look_at(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
        [0.0, -1.0, 0.0],
        45.0,
        800,
        600,
    );
    let moved = Camera::<f64>::from_spec(&offset, 0.2).expect("camera");
    let at_eye = moved.to_camera(Vec3::new(1.0, 2.0, 3.0));
    assert!(at_eye.length() < 1e-14, "the eye must map to the origin");
    let ahead = moved.to_camera(Vec3::new(1.0, 2.0, 8.0));
    assert!((ahead.z - 5.0).abs() < 1e-14, "forward is +z");

    let right = cam.rotation.row(0);
    let down = cam.rotation.row(1);
    let fwd = cam.rotation.row(2);
    assert!((right.cross(down) - fwd).length() < 1e-14, "right x down = forward");

    assert!(Camera::<f64>::from_spec(&forward_camera(0, 600), 0.2).is_err());
    let degenerate = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        45.0,
        8,
        8,
    );
    assert!(Camera::<f64>::from_spec(&degenerate, 0.2).is_err());
    let parallel_up = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        45.0,
        8,
        8,
    );
    assert!(Camera::<f64>::from_spec(&parallel_up, 0.2).is_err());
}

#[test]
fn tv_s1_isotropic_on_axis() {
    let spec = forward_camera(800, 600).with_intrinsics([1000.0, 1000.0], [399.5, 299.5]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    let mu = Vec3::new(0.0, 0.0, 5.0);
    let sigma = Sym3::<f64>::new(0.0025, 0.0, 0.0, 0.0025, 0.0, 0.0025);

    let centre = cam.project(mu).expect("in front");
    assert!((centre.x - 399.5).abs() < 1e-12 && (centre.y - 299.5).abs() < 1e-12);

    let (clamped, fired) = cam.clamp_for_jacobian(mu);
    assert!(!fired);
    let j = cam.jacobian(clamped);
    assert!((j.m[0][0] - 200.0).abs() < 1e-12, "f/z must be 200, got {}", j.m[0][0]);
    assert!((j.m[1][1] - 200.0).abs() < 1e-12);
    assert!(j.m[0][2].abs() < 1e-12 && j.m[1][2].abs() < 1e-12);

    let raw = sigma.project_2x3(&j);
    assert!((raw.xx - 100.0).abs() < 1e-10, "sigma_px = f*sigma/z = 10 px");
    assert!((raw.yy - 100.0).abs() < 1e-10);
    assert!(raw.xy.abs() < 1e-12);

    let dilated = raw.add_diagonal(0.3);
    assert!((dilated.determinant() - 10060.09).abs() < 1e-6);
    let conic = dilated.inverse().expect("nonsingular");
    assert!((conic.xx - 1.0 / 100.3).abs() < 1e-14);
    assert!((conic.yy - 1.0 / 100.3).abs() < 1e-14);

    let half_dilated = 3.0 * dilated.xx.sqrt();
    let half_raw = 3.0 * raw.xx.sqrt();
    assert!((half_dilated - 30.04497).abs() < 1e-4, "got {half_dilated}");
    assert!(
        (half_raw - 30.0).abs() < 1e-9,
        "exactly 30.0 without dilation: a half-extent of 30.0 proves the dilation \
         was NOT applied. got {half_raw}"
    );
    assert!(half_dilated > half_raw);
}

#[test]
fn tv_s1b_half_pixel_conventions_are_identical() {
    let corner = Camera::<f64>::from_spec(
        &forward_camera(800, 600).with_convention(PixelConvention::CornerSample),
        0.2,
    )
    .expect("camera");
    let centre = Camera::<f64>::from_spec(
        &forward_camera(800, 600).with_convention(PixelConvention::CentreSample),
        0.2,
    )
    .expect("camera");

    assert!((corner.principal.x - 399.5).abs() < 1e-15);
    assert!((centre.principal.x - 400.0).abs() < 1e-15);
    assert_eq!(corner.focal, centre.focal);

    let mu = Vec3::new(0.37, -0.21, 4.5);
    let pc = corner.project(mu).expect("in front");
    let pn = centre.project(mu).expect("in front");

    let mut worst: f64 = 0.0;
    for py in (0..600).step_by(37) {
        for px in (0..800).step_by(41) {
            let dc = pc - corner.sample_point(px, py);
            let dn = pn - centre.sample_point(px, py);
            worst = worst.max((dc.x - dn.x).abs()).max((dc.y - dn.y).abs());
        }
    }
    assert!(
        worst < 1e-12,
        "the offset vector d is the same physical sampling in both conventions; \
         worst deviation {worst}"
    );

    let mixed = pc - centre.sample_point(400, 300);
    let honest = pc - corner.sample_point(400, 300);
    assert!(
        ((mixed.x - honest.x).abs() - 0.5).abs() < 1e-12,
        "MIXING one convention's principal point with the other's sample point \
         shifts the image exactly half a pixel in each axis"
    );
    assert!(((mixed.y - honest.y).abs() - 0.5).abs() < 1e-12);
}

#[test]
fn tv_s3_jacobian_exact() {
    let spec = forward_camera(640, 480).with_intrinsics([500.0, 500.0], [320.0, 240.0]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    let mu = Vec3::new(1.0, 2.0, 10.0);
    let (clamped, fired) = cam.clamp_for_jacobian(mu);
    assert!(!fired);
    let j = cam.jacobian(clamped);
    let expected = [[50.0, 0.0, -5.0], [0.0, 50.0, -10.0]];
    for i in 0..2 {
        for k in 0..3 {
            assert!(
                (j.m[i][k] - expected[i][k]).abs() < 1e-12,
                "J[{i}][{k}] = {} expected {}",
                j.m[i][k],
                expected[i][k]
            );
        }
    }
    let p = cam.project(mu).expect("in front");
    assert!((p.x - 370.0).abs() < 1e-12 && (p.y - 340.0).abs() < 1e-12);
    assert!(cam.project(Vec3::new(1.0, 2.0, 0.0)).is_none());
    assert!(cam.project(Vec3::new(1.0, 2.0, -3.0)).is_none());
}

fn tv_s10_setup() -> (Camera<f64>, Vec3<f64>, Sym3<f64>) {
    let angle = (30.0f64).to_radians();
    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [-angle.sin(), 0.0, angle.cos()],
        [0.0, -1.0, 0.0],
        45.0,
        800,
        600,
    )
    .with_intrinsics([600.0, 600.0], [400.0, 300.0]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    let sigma_world = Sym3::<f64>::new(0.025, 0.015, 0.0, 0.025, 0.0, 0.0025);
    (cam, Vec3::new(0.5, 0.3, 8.0), sigma_world)
}

#[test]
fn tv_s10_view_transform_and_screen_covariance() {
    let (cam, mu_world, sigma_world) = tv_s10_setup();
    let mu = cam.to_camera(mu_world);
    assert!((mu.x - 4.4330127019).abs() < 1e-9, "mu_cam.x {}", mu.x);
    assert!((mu.y - 0.3).abs() < 1e-12);
    assert!((mu.z - 6.6782032303).abs() < 1e-9, "mu_cam.z {}", mu.z);

    let sigma_cam = cam.covariance_to_camera(&sigma_world);
    assert!((sigma_cam.xx - 0.019375).abs() < 1e-12);
    assert!((sigma_cam.yy - 0.025).abs() < 1e-12);
    assert!((sigma_cam.zz - 0.008125).abs() < 1e-12);
    assert!((sigma_cam.xy - 0.0129903811).abs() < 1e-9);
    assert!((sigma_cam.yz + 0.0075).abs() < 1e-12);

    let shifted = CameraSpec::look_at(
        [7.0, -3.0, 2.0],
        [7.0, -3.0, 3.0],
        [0.0, -1.0, 0.0],
        45.0,
        800,
        600,
    );
    let shifted_cam = Camera::<f64>::from_spec(&shifted, 0.2).expect("camera");
    assert_eq!(
        shifted_cam.covariance_to_camera(&sigma_world),
        sigma_world,
        "covariances are translation-invariant"
    );

    let centre = cam.project(mu).expect("in front");
    assert!((centre.x - 798.2819224604).abs() < 1e-7, "centre.x {}", centre.x);
    assert!((centre.y - 326.9533576313).abs() < 1e-7, "centre.y {}", centre.y);

    let (clamped, fired) = cam.clamp_for_jacobian(mu);
    assert!(!fired);
    let j = cam.jacobian(clamped);
    assert!((j.m[0][0] - 89.8445254376).abs() < 1e-9);
    let raw = sigma_cam.project_2x3(&j);
    assert!((raw.xx - 289.7033860835).abs() < 1e-7, "raw.xx {}", raw.xx);
    assert!((raw.yy - 207.3725342603).abs() < 1e-7, "raw.yy {}", raw.yy);

    let dilated = raw.add_diagonal(0.3);
    assert!((dilated.determinant() - 37565.1641167934).abs() < 1e-4);
    let half = Vec2::new(3.0 * dilated.xx.sqrt(), 3.0 * dilated.yy.sqrt());
    assert!((half.x - 51.08845735).abs() < 1e-6, "half.x {}", half.x);
    assert!((half.y - 43.23254339).abs() < 1e-6, "half.y {}", half.y);
    let (hi, _) = dilated.eigenvalues();
    assert!(
        3.0 * hi.sqrt() > half.y * 1.35,
        "the eigen circle is much wider than the tight box in y"
    );
}

#[test]
fn tv_s11_y_flip_shear_magnitude() {
    let (cam, mu_world, sigma_world) = tv_s10_setup();
    let mu = cam.to_camera(mu_world);
    let sigma_cam = cam.covariance_to_camera(&sigma_world);
    let (clamped, _) = cam.clamp_for_jacobian(mu);
    let j = cam.jacobian(clamped);
    let raw = sigma_cam.project_2x3(&j);
    assert!(
        (raw.xy.abs() - 150.5342951464).abs() < 1e-7,
        "assert the EXACT MAGNITUDE, not the sign: got {}",
        raw.xy
    );

    let mut half_wrong = j;
    half_wrong.m[1][1] = -half_wrong.m[1][1];
    let bad = sigma_cam.project_2x3(&half_wrong);
    assert!(
        (bad.xy.abs() - 139.5571012166).abs() < 1e-7,
        "the half-wrong y flip gives 139.557, got {}",
        bad.xy
    );
    assert_eq!(
        bad.xy.signum(),
        -raw.xy.signum(),
        "the half-wrong flip has the SAME sign as a correct full flip, so a \
         sign-only assertion would PASS against it"
    );

    let mut full = j;
    full.m[1][1] = -full.m[1][1];
    full.m[1][2] = -full.m[1][2];
    let flipped = sigma_cam.project_2x3(&full);
    assert!((flipped.xy.abs() - 150.5342951464).abs() < 1e-7);
    assert!((flipped.xx - raw.xx).abs() < 1e-9 && (flipped.yy - raw.yy).abs() < 1e-9);
}

#[test]
fn tv_s12_jacobian_clamp() {
    let spec = forward_camera(800, 600).with_intrinsics([600.0, 600.0], [400.0, 300.0]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    let limits = cam.jacobian_limits();
    assert!((limits.x - 0.8666666667).abs() < 1e-9, "lim_x {}", limits.x);

    let far_off = Vec3::new(5.0, 0.0, 1.0);
    let (clamped, fired) = cam.clamp_for_jacobian(far_off);
    assert!(fired, "the clamp must report that it fired");
    let j = cam.jacobian(clamped);
    assert!((j.m[0][2] + 520.0).abs() < 1e-9, "clamped J02 {}", j.m[0][2]);
    let unclamped_j = cam.jacobian(far_off);
    assert!((unclamped_j.m[0][2] + 3000.0).abs() < 1e-9);
    let centre = cam.project(far_off).expect("in front");
    assert!(
        (centre.x - (600.0 * 5.0 + 400.0)).abs() < 1e-9,
        "the projected centre uses the UNCLAMPED position"
    );

    let inside = Vec3::new(0.5, 0.0, 1.0);
    let (kept, not_fired) = cam.clamp_for_jacobian(inside);
    assert!(!not_fired);
    assert_eq!(kept, inside);
    let inside_centre = cam.project(inside).expect("in front");
    assert!((inside_centre.x - (600.0 * 0.5 + 400.0)).abs() < 1e-9);
}

#[test]
fn camera_from_intrinsics_non_square_focal() {
    let spec = forward_camera(1800, 1012).with_intrinsics([505.718889, 505.5], [899.5, 505.5]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    assert!((cam.focal.x - 505.718889).abs() < 1e-12);
    assert!((cam.focal.y - 505.5).abs() < 1e-12);
    assert!(
        cam.focal.x != cam.focal.y,
        "the demo camera's f_x and f_y differ by 0.0433%: assert both independently"
    );

    let p = cam.project(Vec3::new(1.0, 2.0, 10.0)).expect("in front");
    let expect_x = 505.718889 * 0.1 + 899.5;
    let expect_y = 505.5 * 0.2 + 505.5;
    assert!((p.x - expect_x).abs() < 1e-10, "px {} vs {expect_x}", p.x);
    assert!((p.y - expect_y).abs() < 1e-10, "py {} vs {expect_y}", p.y);

    let swapped_x = 505.5 * 0.1 + 899.5;
    assert!(
        (p.x - swapped_x).abs() > 1e-3,
        "an off-axis point whose x and y differ is what detects a focal swap"
    );
}

#[test]
fn ray_direction_is_unit_and_matches_sample_point() {
    let spec = forward_camera(800, 600).with_intrinsics([1000.0, 990.0], [399.5, 299.5]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    for py in (0..600).step_by(83) {
        for px in (0..800).step_by(97) {
            let ray = cam.ray(px, py);
            assert!(
                (ray.direction.length() - 1.0).abs() < 1e-15,
                "ray direction must be unit"
            );
            assert_eq!(ray.origin, Vec3::zero());
            let back = cam
                .project(ray.origin + ray.direction * 7.5)
                .expect("in front");
            let sample = cam.sample_point(px, py);
            assert!(
                (back - sample).length() < 1e-11,
                "ray and splat must share one pixel-coordinate convention: \
                 {back:?} vs {sample:?}"
            );
        }
    }
}

#[test]
fn orthographic_projection_is_affine_and_constant() {
    let spec = forward_camera(400, 300).orthographic(2.0);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    assert_eq!(cam.projection, Projection::Orthographic);
    assert!((cam.focal.y - 75.0).abs() < 1e-12);

    let a = cam.jacobian(Vec3::new(0.0, 0.0, 5.0));
    let b = cam.jacobian(Vec3::new(3.0, -2.0, 40.0));
    assert_eq!(a, b, "an orthographic Jacobian is exact and constant");
    assert!(a.m[0][2] == 0.0 && a.m[1][2] == 0.0);

    let near = cam.project(Vec3::new(1.0, 0.5, 3.0)).expect("orthographic");
    let far = cam.project(Vec3::new(1.0, 0.5, 300.0)).expect("orthographic");
    assert_eq!(near, far, "orthographic projection ignores depth");

    let (clamped, fired) = cam.clamp_for_jacobian(Vec3::new(500.0, 0.0, 1.0));
    assert!(!fired && clamped == Vec3::new(500.0, 0.0, 1.0));

    let ray = cam.ray(10, 20);
    assert_eq!(ray.direction, Vec3::new(0.0, 0.0, 1.0));
    assert!(ray.origin.z == 0.0);
}

#[test]
fn screen_disc_bound_is_conservative() {
    let spec = forward_camera(800, 600).with_intrinsics([900.0, 900.0], [399.5, 299.5]);
    let cam = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    let centre = Vec3::new(1.2, -0.7, 6.0);
    let radius = 0.9;
    let (disc_centre, disc_radius) = cam
        .screen_disc_bound(centre, radius)
        .expect("sphere is in front");

    let mut worst: f64 = 0.0;
    for i in 0..64 {
        for k in 0..64 {
            let u = (i as f64 + 0.5) / 64.0 * core::f64::consts::PI;
            let v = (k as f64 + 0.5) / 64.0 * core::f64::consts::TAU;
            let dir = Vec3::new(u.sin() * v.cos(), u.sin() * v.sin(), u.cos());
            let surface = centre + dir * radius;
            if surface.z <= 0.0 {
                continue;
            }
            let p = cam.project(surface).expect("in front");
            worst = worst.max((p - disc_centre).length());
        }
    }
    assert!(
        worst <= disc_radius,
        "the bound must never reject a contributor: worst {worst} vs {disc_radius}"
    );
    assert!(disc_radius < 4.0 * worst, "the bound must not be absurdly loose");

    for centre in [
        Vec3::new(8.0, 0.0, 6.0),
        Vec3::new(0.0, 9.0, 5.0),
        Vec3::new(-11.0, 4.0, 4.0),
        Vec3::new(0.0, 0.0, 6.0),
    ] {
        let (dc, dr) = cam
            .screen_disc_bound(centre, radius)
            .expect("sphere is in front");
        let mut off_axis_worst: f64 = 0.0;
        for i in 0..48 {
            for k in 0..48 {
                let u = (i as f64 + 0.5) / 48.0 * core::f64::consts::PI;
                let v = (k as f64 + 0.5) / 48.0 * core::f64::consts::TAU;
                let dir = Vec3::new(u.sin() * v.cos(), u.sin() * v.sin(), u.cos());
                let surface = centre + dir * radius;
                if surface.z <= 0.0 {
                    continue;
                }
                let p = cam.project(surface).expect("in front");
                off_axis_worst = off_axis_worst.max((p - dc).length());
            }
        }
        assert!(
            off_axis_worst <= dr,
            "a STRONGLY off-axis sphere is where the (z^2 + x^2) term earns its \
             place: at {centre:?} the true extent is {off_axis_worst} and the \
             bound is {dr}. Dropping the off-axis term under-bounds by up to \
             |mu| / z, which is 67 percent at x = 8, z = 6."
        );
    }

    assert!(cam.screen_disc_bound(Vec3::new(0.0, 0.0, 0.5), 0.9).is_none());
    assert!(cam.screen_disc_bound(Vec3::new(0.0, 0.0, -5.0), 0.1).is_none());
}

#[test]
fn camera_precision_roundtrip() {
    let spec = forward_camera(320, 240);
    let wide = Camera::<f64>::from_spec(&spec, 0.2).expect("camera");
    let narrow = Camera::<f32>::from_f64(&wide);
    let back = narrow.to_f64();
    assert!((back.focal.x - wide.focal.x).abs() < 1e-3);
    assert_eq!(back.width, wide.width);
    assert_eq!(back.convention, wide.convention);
    let single = Camera::<f32>::from_spec(&spec, 0.2).expect("camera");
    assert!((single.focal.y as f64 - wide.focal.y).abs() < 1e-3);
}

#[test]
fn the_camera_is_the_sole_owner_of_the_pixel_convention() {
    let corner = forward_camera(64, 64).with_convention(PixelConvention::CornerSample);
    let centre = forward_camera(64, 64).with_convention(PixelConvention::CentreSample);
    let a = match Camera::<f64>::from_spec(&corner, 0.2) {
        Ok(c) => c,
        Err(e) => panic!("corner camera: {e:?}"),
    };
    let b = match Camera::<f64>::from_spec(&centre, 0.2) {
        Ok(c) => c,
        Err(e) => panic!("centre camera: {e:?}"),
    };
    let pa = a.sample_point(7, 9);
    let pb = b.sample_point(7, 9);
    assert_eq!(
        pb.x - pa.x,
        0.5,
        "the convention must move the sample point by exactly half a pixel in x"
    );
    assert_eq!(pb.y - pa.y, 0.5, "and half a pixel in y");
    assert_eq!(
        a.convention,
        PixelConvention::CornerSample,
        "the camera carries the convention it was built with"
    );
    assert_eq!(b.convention, PixelConvention::CentreSample);
}

#[test]
fn a_sub_rect_render_blits_back_into_the_pixels_the_whole_frame_would_have_drawn() {
    let methods = lights::method::RenderMethod::implemented();
    assert!(
        !methods.is_empty(),
        "no method is built, so this case would pass by measuring nothing"
    );

    let full_spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        128,
        96,
    )
    .with_convention(PixelConvention::CornerSample);

    let mut scene = lights::scene::Scene::<f64>::new();
    for (index, colour) in [
        lights::numerics::vector::Rgb::new(0.9, 0.2, 0.1),
        lights::numerics::vector::Rgb::new(0.1, 0.8, 0.3),
        lights::numerics::vector::Rgb::new(0.2, 0.3, 0.9),
        lights::numerics::vector::Rgb::new(0.9, 0.8, 0.2),
    ]
    .iter()
    .enumerate()
    {
        let x = -0.45 + 0.3 * index as f64;
        let atom = lights::scene::Atom::isotropic(
            lights::numerics::vector::Vec3::new(x, 0.15 * index as f64 - 0.2, 3.0),
            0.19,
            0.85,
            *colour,
        );
        scene
            .push(atom, lights::settings::MIN_SCALE)
            .expect("scene rejected an atom");
    }

    let mut settings = lights::settings::SharedSettings::<f64>::contract();
    settings.threads = 1;
    settings.background = lights::numerics::vector::Rgb::new(0.05, 0.06, 0.09);

    let (rx, ry, rw, rh) = (40u32, 24u32, 48u32, 40u32);
    let patch_spec = full_spec.sub_rect(rx, ry, rw, rh).expect("sub_rect");
    assert_eq!((patch_spec.width, patch_spec.height), (rw, rh));

    for method in methods.iter() {
        let request = |camera: CameraSpec| lights::dispatch::RenderRequest {
            precision: lights::dispatch::Precision::F64,
            kernel: lights::kernel::KernelFamily::Gaussian,
            method: method.clone(),
            settings,
            camera,
            scene: scene.clone(),
            collect: lights::dispatch::Collect::Pixels,
            shapes: None,
        };
        let whole = lights::dispatch::render(&request(full_spec.clone())).expect("full render");
        let patch = lights::dispatch::render(&request(patch_spec.clone())).expect("patch render");
        assert_eq!((patch.image.width, patch.image.height), (rw, rh));

        let mut assembled = whole.image.clone();
        assembled
            .blit_from(&patch.image, None, rx, ry)
            .expect("blit_from");
        let gap = assembled
            .difference(&whole.image)
            .expect("difference")
            .max_abs();
        let differing = assembled
            .pixels
            .iter()
            .zip(whole.image.pixels.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "SUBRECT {} differing {} of {} worst {:e} at ({},{})",
            method.id().name(),
            differing,
            whole.image.pixels.len(),
            gap.0,
            gap.1,
            gap.2
        );
        assert!(
            gap.0 <= f64::EPSILON,
            "{}: rendering ONLY the damaged rect through a camera whose principal point is \
             shifted by (-{rx},-{ry}) must reproduce the pixels the whole frame drew there, \
             to within ONE ULP. {differing} of {} pixels differ and the worst is {:e} at \
             ({},{}), which is more than f64::EPSILON = {:e}. The window is a camera, not a \
             clip, and the only licensed difference is the association of the principal-point \
             shift",
            method.id().name(),
            whole.image.pixels.len(),
            gap.0,
            gap.1,
            gap.2,
            f64::EPSILON
        );
    }

    assert!(
        full_spec.sub_rect(100, 0, 40, 10).is_err(),
        "a rect that runs off the right edge must be refused, not silently clamped"
    );
    assert!(
        full_spec.sub_rect(0, 0, 0, 10).is_err(),
        "a zero-width rect must be refused"
    );
    let mut small = lights::image::FloatImage::<f64>::new(8, 8);
    let patch = lights::image::FloatImage::<f64>::new(4, 4);
    assert!(
        small.blit_from(&patch, None, 6, 6).is_err(),
        "a blit that runs off the target must be refused rather than writing outside it"
    );
}
