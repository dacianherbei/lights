use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::{DegenerateReason, LightsError};
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::kernel::gaussian::Gaussian;
use lights::numerics::quaternion::Quat;
use lights::numerics::symmetric::Sym3;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::prepare::PreparedScene;
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, SharedSettings};

fn white() -> Rgb<f64> {
    Rgb::new(1.0, 1.0, 1.0)
}

#[test]
fn atom_covariance_is_r_s_and_not_s_r() {
    let half = (22.5f64).to_radians();
    let q = Quat::<f64>::new_wxyz(half.cos(), 0.0, 0.0, half.sin());
    let atom = Atom::new(Vec3::zero(), q, Vec3::new(2.0, 1.0, 1.0), 1.0, white());
    let sigma = atom.covariance();
    assert!((sigma.xy - 1.5).abs() < 1e-12, "Sigma01 must be +1.5, got {}", sigma.xy);
    assert!((sigma.xx - 2.5).abs() < 1e-12);

    let isotropic = Atom::isotropic(Vec3::new(1.0, 2.0, 3.0), 0.05, 0.8, white());
    let iso_sigma = isotropic.covariance();
    assert!((iso_sigma.xx - 0.0025).abs() < 1e-15);
    assert!(iso_sigma.xy.abs() < 1e-18 && iso_sigma.xz.abs() < 1e-18);
    assert_eq!(isotropic.rotation, Quat::identity());
}

#[test]
fn push_normalises_the_quaternion() {
    let mut scene = Scene::<f64>::new();
    let doubled = Quat::<f64>::new_wxyz(2.0, 0.0, 0.0, 0.0);
    scene
        .push(
            Atom::new(Vec3::zero(), doubled, Vec3::new(0.3, 0.2, 0.1), 1.0, white()),
            MIN_SCALE,
        )
        .expect("an unnormalised quaternion is normalised, not rejected");

    let stored = scene.atoms()[0];
    assert!((stored.rotation.norm() - 1.0).abs() < 1e-15);
    let sigma = stored.covariance();
    assert!(
        (sigma.xx - 0.09).abs() < 1e-15,
        "without normalisation Sigma would be scaled by ||q||^4 = 16: {}",
        sigma.xx
    );
    assert!((sigma.yy - 0.04).abs() < 1e-15);
    assert!((sigma.zz - 0.01).abs() < 1e-15);

    let mut broken = Scene::<f64>::new();
    match broken.push(
        Atom::new(
            Vec3::zero(),
            Quat::new_wxyz(0.0, 0.0, 0.0, 0.0),
            Vec3::splat(0.1),
            1.0,
            white(),
        ),
        MIN_SCALE,
    ) {
        Err(LightsError::DegenerateAtom {
            index,
            reason: DegenerateReason::UnnormalisableRotation,
        }) => assert_eq!(index, 0),
        other => panic!("expected UnnormalisableRotation, got {other:?}"),
    }
}

#[test]
fn push_floors_scales_and_rejects_non_finite_values() {
    let mut scene = Scene::<f64>::new();
    scene
        .push(
            Atom::new(
                Vec3::zero(),
                Quat::identity(),
                Vec3::new(0.0, -1.0, 0.5),
                1.0,
                white(),
            ),
            MIN_SCALE,
        )
        .expect("scales are floored at insertion, once");
    let stored = scene.atoms()[0];
    assert_eq!(stored.scale.x, MIN_SCALE);
    assert_eq!(stored.scale.y, MIN_SCALE);
    assert_eq!(stored.scale.z, 0.5);
    assert!(
        stored.covariance().cholesky().is_some(),
        "the floor exists so that Lambda = Sigma^-1 exists for the ray path"
    );

    let mut bad = Scene::<f64>::new();
    match bad.push(
        Atom::new(
            Vec3::new(f64::NAN, 0.0, 0.0),
            Quat::identity(),
            Vec3::splat(0.1),
            1.0,
            white(),
        ),
        MIN_SCALE,
    ) {
        Err(LightsError::DegenerateAtom {
            reason: DegenerateReason::NonFiniteParameter,
            ..
        }) => (),
        other => panic!("expected NonFiniteParameter, got {other:?}"),
    }
    assert!(bad.is_empty(), "a rejected atom must not enter the scene");
}

#[test]
fn scene_reports_bounds_and_converts_precision() {
    let mut scene = Scene::<f64>::new();
    scene
        .push(Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.5, 1.0, white()), MIN_SCALE)
        .expect("valid");
    scene
        .push(Atom::isotropic(Vec3::new(2.0, 0.0, 5.0), 0.25, 1.0, white()), MIN_SCALE)
        .expect("valid");
    assert_eq!(scene.len(), 2);
    assert!(!scene.is_empty());

    let (lo, hi) = scene.bounds(3.0);
    assert!((lo.x - (-1.5)).abs() < 1e-12, "lo.x {}", lo.x);
    assert!((hi.x - 2.75).abs() < 1e-12, "hi.x {}", hi.x);

    let half = (22.5f64).to_radians();
    let q = Quat::<f64>::new_wxyz(half.cos(), 0.0, 0.0, half.sin());
    let mut rotated = Scene::<f64>::new();
    rotated
        .push(
            Atom::new(Vec3::zero(), q, Vec3::new(2.0, 1.0, 1.0), 1.0, white()),
            MIN_SCALE,
        )
        .expect("valid");
    let (rlo, rhi) = rotated.bounds(1.0);
    let sigma = rotated.atoms()[0].covariance();
    assert!(
        (rhi.x - sigma.xx.sqrt()).abs() < 1e-12,
        "the AABB half-extent is sqrt(Sigma_xx), which ALREADY INCLUDES the \
         rotation. Building it from the scale vector alone gives 2.0 here \
         instead of sqrt(2.5) = {}, and the error grows with the rotation.",
        sigma.xx.sqrt()
    );
    assert!((rhi.x - 2.5f64.sqrt()).abs() < 1e-12, "rhi.x {}", rhi.x);
    assert!(
        (rhi.x - 2.0).abs() > 0.4,
        "an axis-aligned atom cannot detect this: the test needs a rotation"
    );
    assert!((rhi.y - 2.5f64.sqrt()).abs() < 1e-12);
    assert!((rlo.x + 2.5f64.sqrt()).abs() < 1e-12);
    assert!((rhi.z - 1.0).abs() < 1e-12);

    let narrow = Scene::<f32>::from_f64(&scene);
    assert_eq!(narrow.len(), 2);
    let back = narrow.to_f64();
    assert!((back.atoms()[0].mean.z - 5.0).abs() < 1e-6);
}

#[test]
fn prepared_scene_shares_one_covariance_and_rejects_degenerates() {
    let settings = SharedSettings::<f64>::contract();
    let mut scene = Scene::<f64>::new();
    let half = (22.5f64).to_radians();
    let q = Quat::<f64>::new_wxyz(half.cos(), 0.0, 0.0, half.sin());
    scene
        .push(
            Atom::new(
                Vec3::new(0.0, 0.0, 5.0),
                q,
                Vec3::new(0.4, 0.2, 0.1),
                0.9,
                Rgb::new(0.2, 0.4, 0.6),
            ),
            settings.scale_floor,
        )
        .expect("valid");

    let prepared =
        PreparedScene::<f64, Gaussian>::build(&scene, &[], &settings).expect("SPD atom");
    assert_eq!(prepared.len(), 1);
    assert!(!prepared.is_empty());

    let view = prepared.atom(0);
    assert_eq!(view.index, 0);
    assert_eq!(*view.sigma, scene.atoms()[0].covariance());
    assert!((view.opacity - 0.9).abs() < 1e-15);
    assert!((view.scale_max - 0.4).abs() < 1e-15);
    assert!((view.scale_min - 0.1).abs() < 1e-15);

    let v = Vec3::new(0.3, -0.1, 0.2);
    let via_lambda = view.lambda.quad_form(v);
    let via_whiten = view.whiten.apply(v).length_squared();
    assert!(
        (via_lambda - via_whiten).abs() < 1e-10,
        "lambda and whiten must describe the same metric: {via_lambda} vs {via_whiten}"
    );
    let roundtrip = view.lambda.inverse_spd().expect("SPD");
    assert!((roundtrip.xx - view.sigma.xx).abs() < 1e-12);
    assert!(prepared.resident_bytes() > 0);

    assert!(matches!(
        PreparedScene::<f64, Gaussian>::build(&Scene::<f64>::new(), &[], &settings),
        Err(LightsError::EmptyScene)
    ));
}

#[test]
fn prepared_scene_rejects_a_scale_below_the_floor() {
    let settings = SharedSettings::<f64>::contract();
    let mut scene = Scene::<f64>::new();
    scene
        .push(
            Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.1, 1.0, white()),
            settings.scale_floor,
        )
        .expect("valid");
    assert!(PreparedScene::<f64, Gaussian>::build(&scene, &[], &settings).is_ok());

    let mut tiny = Scene::<f64>::new();
    tiny.push(
        Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 1.0e-30, 1.0, white()),
        1.0e-30,
    )
    .expect("pushed under a permissive floor");
    match PreparedScene::<f64, Gaussian>::build(&tiny, &[], &settings) {
        Err(LightsError::DegenerateAtom {
            index,
            reason: DegenerateReason::NonPositiveScale,
        }) => assert_eq!(index, 0),
        Err(other) => panic!("expected NonPositiveScale, got {other:?}"),
        Ok(_) => panic!("a scale below the floor must not build"),
    }
}

#[test]
fn covariance_stores_six_entries_so_asymmetry_is_unrepresentable() {
    let s = Sym3::<f64>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let v = Vec3::new(1.0, 0.0, 0.0);
    let w = Vec3::new(0.0, 1.0, 0.0);
    assert_eq!(
        v.dot(s.apply(w)),
        w.dot(s.apply(v)),
        "an asymmetric covariance is not a representable value"
    );
    assert_eq!(core::mem::size_of::<Sym3<f64>>(), 6 * 8);
}

fn tilted_needle<T: lights::numerics::float::FloatingPoint>(thinness: T) -> Scene<T> {
    let half = T::from_f64((22.5f64).to_radians());
    let (s, c) = half.sin_cos();
    let q = Quat::<T>::new_wxyz(c, s * T::HALF, s * T::HALF, s * T::HALF)
        .normalize()
        .expect("unit");
    let mut scene = Scene::<T>::new();
    scene
        .push(
            Atom::new(
                Vec3::new(T::ZERO, T::ZERO, T::from_f64(5.0)),
                q,
                Vec3::new(T::ONE, T::ONE, thinness),
                T::ONE,
                Rgb::new(T::ONE, T::ONE, T::ONE),
            ),
            T::from_f64(1.0e-30),
        )
        .expect("valid at insertion");
    scene
}

#[test]
fn prepared_scene_rejects_a_covariance_that_is_not_spd_at_this_precision() {
    let mut narrow = SharedSettings::<f32>::contract();
    narrow.scale_floor = 1.0e-30;
    let scene = tilted_needle::<f32>(1.0e-4);
    assert!(
        scene.atoms()[0].covariance().cholesky().is_none(),
        "a 1e-4 axis ratio on a tilted atom loses positive-definiteness in f32"
    );
    match PreparedScene::<f32, Gaussian>::build(&scene, &[], &narrow) {
        Err(LightsError::DegenerateAtom {
            index,
            reason: DegenerateReason::NonSpdCovariance,
        }) => assert_eq!(index, 0),
        Err(other) => panic!("expected NonSpdCovariance, got {other:?}"),
        Ok(_) => panic!(
            "a non-SPD covariance must be an ERROR, never a silent fallback to \
             zero or identity: Lambda would then be meaningless and every ray \
             through the atom would return a wrong r^2"
        ),
    }

    let mut wide = SharedSettings::<f64>::contract();
    wide.scale_floor = 1.0e-30;
    let same_atom = tilted_needle::<f64>(1.0e-4);
    assert!(
        PreparedScene::<f64, Gaussian>::build(&same_atom, &[], &wide).is_ok(),
        "the SAME atom is fine in f64: this is the precision axis producing a \
         number rather than an opinion"
    );
}

#[test]
fn lambda_and_whiten_come_from_one_cholesky() {
    let settings = SharedSettings::<f64>::contract();
    let scene = tilted_needle::<f64>(0.05);
    let prepared =
        PreparedScene::<f64, Gaussian>::build(&scene, &[], &settings).expect("SPD in f64");
    let view = prepared.atom(0);
    for v in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.31, -0.72, 0.44),
    ] {
        let via_lambda = view.lambda.quad_form(v);
        let via_whiten = view.whiten.apply(v).length_squared();
        assert!(
            (via_lambda - via_whiten).abs() <= 1e-9 * (1.0 + via_lambda.abs()),
            "lambda must be derived from the SAME factor as whiten: {via_lambda} \
             vs {via_whiten}"
        );
        let reference = view.sigma.inverse_spd().expect("SPD");
        assert!(
            (reference.quad_form(v) - via_lambda).abs() <= 1e-9 * (1.0 + via_lambda.abs()),
            "and it must still equal the independently computed inverse"
        );
    }
}

#[test]
fn resident_bytes_charges_each_stored_vector_exactly_once() {
    let settings64 = SharedSettings::<f64>::contract();
    let mut scene64 = Scene::<f64>::new();
    for i in 0..7 {
        scene64
            .push(
                Atom::isotropic(Vec3::new(0.1 * i as f64, 0.0, 5.0), 0.1, 0.9, white()),
                settings64.scale_floor,
            )
            .expect("valid");
    }
    let prepared64 =
        PreparedScene::<f64, Gaussian>::build(&scene64, &[], &settings64).expect("built");
    let per_atom_64 = 24 + 48 + 48 + 48 + 8 + 24 + 0 + 8 + 8;
    assert_eq!(per_atom_64, 216);
    assert_eq!(
        prepared64.resident_bytes(),
        7 * per_atom_64,
        "PreparedScene holds mean(24) sigma(48) lambda(48) whiten(48) opacity(8) \
         colour(24) params(0) scale_max(8) scale_min(8) per atom at f64"
    );

    let settings32 = SharedSettings::<f32>::contract();
    let mut scene32 = Scene::<f32>::new();
    for i in 0..7 {
        scene32
            .push(
                Atom::isotropic(
                    Vec3::new(0.1 * i as f32, 0.0, 5.0),
                    0.1,
                    0.9,
                    Rgb::new(1.0f32, 1.0, 1.0),
                ),
                settings32.scale_floor,
            )
            .expect("valid");
    }
    let prepared32 =
        PreparedScene::<f32, Gaussian>::build(&scene32, &[], &settings32).expect("built");
    let per_atom_32 = 12 + 24 + 24 + 24 + 4 + 12 + 0 + 4 + 4;
    assert_eq!(per_atom_32, 108);
    assert_eq!(prepared32.resident_bytes(), 7 * per_atom_32);
}

fn transparency_scene(include_transparent: bool) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    let visible = [
        (-0.05f64, 4.0f64, Rgb::new(1.0, 0.0, 0.0)),
        (0.05, 5.0, Rgb::new(0.0, 1.0, 0.0)),
    ];
    for (x, z, colour) in visible {
        scene
            .push(
                Atom::isotropic(Vec3::new(x, 0.0, z), 0.08, 0.7, colour),
                MIN_SCALE,
            )
            .expect("visible atom");
        if include_transparent {
            scene
                .push(
                    Atom::isotropic(Vec3::new(x, 0.0, z + 0.25), 0.08, 0.0, white()),
                    MIN_SCALE,
                )
                .expect("transparent atom");
        }
    }
    scene
}

fn transparency_request(include_transparent: bool, method: RenderMethod) -> RenderRequest {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings,
        camera: CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            48,
            48,
        ),
        scene: transparency_scene(include_transparent),
        collect: Collect::PixelsAndCounters,
        shapes: None,
    }
}

#[test]
fn a_fully_transparent_atom_never_reaches_the_renderer_and_the_image_does_not_move() {
    let settings = SharedSettings::<f64>::contract();
    let opaque_only = transparency_scene(false);
    let with_transparent = transparency_scene(true);
    assert_eq!(opaque_only.len(), 2);
    assert_eq!(with_transparent.len(), 4, "the scene still stores every atom");

    let lean = PreparedScene::<f64, Gaussian>::build(&opaque_only, &[], &settings)
        .expect("built");
    let padded = PreparedScene::<f64, Gaussian>::build(&with_transparent, &[], &settings)
        .expect("built");
    assert_eq!(lean.len(), 2);
    assert_eq!(
        padded.len(),
        2,
        "an atom with opacity 0 costs a Cholesky, an inverse, nine parallel-array \
         slots, a projection, a tile binning and a per-pixel kernel evaluation for a \
         contribution that is exactly zero"
    );

    let mut methods = 0usize;
    for method in RenderMethod::implemented() {
        methods += 1;
        let plain = render(&transparency_request(false, method.clone())).expect("render");
        let padded_render = render(&transparency_request(true, method.clone())).expect("render");
        assert_eq!(
            plain.image.pixels, padded_render.image.pixels,
            "{method:?}: an atom with opacity 0 contributes exactly nothing, so culling \
             it must be invisible in the rendered image"
        );
        assert_eq!(
            padded_render.counters.atoms_prepared, plain.counters.atoms_prepared,
            "{method:?}: and the cull must be visible in the counters"
        );

        let mut all_transparent = transparency_request(false, method.clone());
        all_transparent.scene = Scene::<f64>::new();
        for x in [-0.05f64, 0.05] {
            all_transparent
                .scene
                .push(
                    Atom::isotropic(Vec3::new(x, 0.0, 5.0), 0.08, 0.0, white()),
                    MIN_SCALE,
                )
                .expect("transparent atom");
        }
        let empty =
            render(&all_transparent).expect("a scene that culls to nothing still renders");
        assert_eq!(
            empty.counters.atoms_prepared, 2,
            "{method:?}: the renderers refuse an empty PreparedScene, so the cull must \
             never be the thing that empties one: a scene with no visible atom keeps \
             its atoms"
        );
        for p in empty.image.pixels.iter() {
            assert_eq!(*p, Rgb::zero(), "{method:?}: nothing visible leaves the background");
        }
    }
    assert!(
        methods > 0,
        "no method was built into this binary, so this case asserted nothing"
    );
}

#[test]
fn a_mutated_scene_is_indistinguishable_from_one_built_that_way() {
    let methods = RenderMethod::implemented();
    assert!(
        !methods.is_empty(),
        "no method is built, so this case would pass by measuring nothing"
    );

    let atom = |x: f64, colour: Rgb<f64>| {
        Atom::isotropic(Vec3::new(x, 0.0, 4.0), 0.18, 0.8, colour)
    };
    let red = Rgb::new(0.9, 0.2, 0.1);
    let green = Rgb::new(0.1, 0.8, 0.3);
    let blue = Rgb::new(0.2, 0.3, 0.9);

    let mut edited = Scene::<f64>::new();
    for (x, colour) in [(-0.4, red), (0.0, red), (0.4, red)] {
        edited.push(atom(x, colour), MIN_SCALE).expect("push");
    }
    edited.set(1, atom(0.05, green), MIN_SCALE).expect("set");
    let removed = edited.remove(0).expect("remove");
    assert_eq!(removed.colour, red, "remove must hand back the atom it took out");
    edited.push(atom(-0.4, blue), MIN_SCALE).expect("push");
    assert_eq!(edited.len(), 3);

    let mut built = Scene::<f64>::new();
    for (x, colour) in [(0.05, green), (0.4, red), (-0.4, blue)] {
        built.push(atom(x, colour), MIN_SCALE).expect("push");
    }

    let spec = CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        40.0,
        64,
        48,
    );
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 1;
    for method in methods.iter() {
        let request = |scene: Scene<f64>| RenderRequest {
            precision: Precision::F64,
            kernel: KernelFamily::Gaussian,
            method: method.clone(),
            settings,
            camera: spec.clone(),
            scene,
            collect: Collect::Pixels,
            shapes: None,
        };
        let a = render(&request(edited.clone())).expect("edited render");
        let b = render(&request(built.clone())).expect("built render");
        assert_eq!(
            a.image.pixels,
            b.image.pixels,
            "{}: a scene reached by set + remove + push must render BIT-IDENTICALLY to one \
             built with the same atoms in the same order. Anything else means a mutator \
             skipped the ingest that push performs, and the scene is no longer a pure \
             function of its atoms",
            method.id().name()
        );
    }

    let mut floored = Scene::<f64>::new();
    floored.push(atom(0.0, red), MIN_SCALE).expect("push");
    floored
        .set(
            0,
            Atom::new(
                Vec3::new(0.0, 0.0, 4.0),
                Quat::identity(),
                Vec3::new(MIN_SCALE * 0.001, 0.2, 0.2),
                0.8,
                red,
            ),
            MIN_SCALE,
        )
        .expect("set must apply the floor rather than refuse");
    assert_eq!(
        floored.atoms()[0].scale.x,
        MIN_SCALE,
        "set must re-apply the ingest floor exactly as push does — a mutator that writes \
         an unclamped scale is how a degenerate covariance reaches the renderer"
    );

    let mut unnormalised = Scene::<f64>::new();
    unnormalised.push(atom(0.0, red), MIN_SCALE).expect("push");
    let bad = Atom::new(
        Vec3::new(0.0, 0.0, 4.0),
        Quat::new_wxyz(0.0, 0.0, 0.0, 0.0),
        Vec3::splat(0.2),
        0.8,
        red,
    );
    match unnormalised.set(0, bad, MIN_SCALE) {
        Err(LightsError::DegenerateAtom { index, reason }) => {
            assert_eq!(index, 0);
            assert_eq!(reason, DegenerateReason::UnnormalisableRotation);
        }
        other => panic!("set must refuse an unnormalisable rotation, got {other:?}"),
    }
    assert_eq!(
        unnormalised.atoms()[0].colour,
        red,
        "a REFUSED write must leave the scene untouched"
    );

    match unnormalised.set(7, atom(0.0, red), MIN_SCALE) {
        Err(LightsError::AtomOutOfRange { index, len }) => {
            assert_eq!((index, len), (7, 1));
        }
        other => panic!("set past the end must be an error, got {other:?}"),
    }
    assert!(unnormalised.remove(7).is_err(), "remove past the end must be an error");

    unnormalised.clear();
    assert!(unnormalised.is_empty(), "clear must empty the scene");
}
