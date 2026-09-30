use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::LightsError;
use lights::kernel::KernelFamily;
use lights::kernel::gaussian::Gaussian;
use lights::method::raycast::{AccelKind, RayCastConfig};
use lights::method::splat::{SortKind, SplatConfig};
use lights::method::{ImageFormation, MethodId, RenderMethod, TraversalOrder, create_strategy};
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, SharedSettings};

fn every_method() -> [RenderMethod; 2] {
    [
        RenderMethod::SplatRaster(SplatConfig::contract()),
        RenderMethod::RayCast(RayCastConfig::contract()),
    ]
}

fn demo_scene() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    scene
        .push(
            Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.05, 1.0, Rgb::new(1.0, 0.0, 0.0)),
            MIN_SCALE,
        )
        .expect("valid");
    scene
}

fn request(method: RenderMethod, precision: Precision) -> RenderRequest {
    RenderRequest {
        precision,
        kernel: KernelFamily::Gaussian,
        method,
        settings: SharedSettings::<f64>::contract(),
        camera: CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            64,
            48,
        ),
        scene: demo_scene(),
        collect: Collect::Pixels,
        shapes: None,
    }
}

#[test]
fn every_method_variant_names_itself_when_it_fails() {
    for method in every_method() {
        let id = method.id();
        let settings = SharedSettings::<f64>::contract();
        match create_strategy::<f64, Gaussian>(&method, &settings) {
            Err(LightsError::MethodNotImplemented { method: named }) => assert_eq!(
                named, id,
                "a method must name ITSELF, never fall back to another"
            ),
            Err(other) => panic!("{id:?} returned the wrong error: {other:?}"),
            Ok(_) => assert!(
                id.implemented(),
                "{id:?} built a strategy while reporting implemented() == false"
            ),
        }
    }
}

#[test]
fn unimplemented_methods_never_produce_an_image() {
    for method in every_method() {
        let id = method.id();
        for precision in Precision::ALL {
            let outcome = render(&request(method.clone(), precision));
            match outcome {
                Err(LightsError::MethodNotImplemented { method: named }) => {
                    assert_eq!(named, id, "{precision:?}: wrong method named")
                }
                Err(other) => panic!("{id:?} at {precision:?} returned {other:?}"),
                Ok(image) => assert!(
                    id.implemented(),
                    "{id:?} produced an image of {}x{} while unimplemented",
                    image.image.width,
                    image.image.height
                ),
            }
        }
    }
}

#[test]
fn method_identity_table_is_exhaustive_and_stable() {
    assert_eq!(
        MethodId::ALL,
        [MethodId::SplatRaster, MethodId::RayCast],
        "L-117, PM ruling 2026-09-28: RayMarch, VoxelDda and ConeBeam were pruned. \
         None had a written implementation or a written consumer, and each cost a \
         stub file in every build and a slot in every test that walks the enum. A \
         third method re-enters with its implementation, never as a parked stub"
    );
    let mut names: Vec<&'static str> = MethodId::ALL.iter().map(|m| m.name()).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "method names must be distinct");

    assert_eq!(MethodId::SplatRaster.order(), TraversalOrder::ObjectOrder);
    assert_eq!(MethodId::RayCast.order(), TraversalOrder::ImageOrder);

    for (method, id) in every_method().iter().zip(MethodId::ALL.iter()) {
        assert_eq!(method.id(), *id, "RenderMethod and MethodId must agree");
    }

    let listed = RenderMethod::implemented();
    assert_eq!(
        listed.len(),
        MethodId::ALL.iter().filter(|m| m.implemented()).count(),
        "implemented() must agree with the MethodId table"
    );
    for method in listed {
        assert!(method.id().implemented());
    }
}

#[test]
fn the_implemented_flag_matches_what_actually_builds() {
    assert_eq!(
        MethodId::SplatRaster.implemented(),
        cfg!(feature = "method-splat"),
        "SplatRaster must claim to be implemented exactly when its cargo feature \
         compiles its code"
    );
    assert_eq!(
        MethodId::RayCast.implemented(),
        cfg!(feature = "method-raycast"),
        "RayCast must claim to be implemented exactly when its cargo feature \
         compiles its code"
    );
    let settings = SharedSettings::<f64>::contract();
    for method in every_method() {
        let id = method.id();
        let built = create_strategy::<f64, Gaussian>(&method, &settings).is_ok();
        assert_eq!(
            built,
            id.implemented(),
            "{id:?}: implemented() and create_strategy() disagree"
        );
    }

    let listed = RenderMethod::implemented();
    let expected = usize::from(cfg!(feature = "method-splat"))
        + usize::from(cfg!(feature = "method-raycast"));
    assert_eq!(listed.len(), expected);
}

#[test]
fn each_variant_carries_its_own_configuration() {
    let splat = SplatConfig {
        tile: 32,
        sort: SortKind::PerTileSort,
    };
    let raycast = RayCastConfig {
        accel: AccelKind::BruteForce,
        tile: 8,
        max_hits_per_pixel: 17,
    };
    assert_eq!(RenderMethod::SplatRaster(splat).tile(), 32);
    assert_eq!(RenderMethod::RayCast(raycast).tile(), 8);

    assert_eq!(RenderMethod::splat_reference().id(), MethodId::SplatRaster);
    assert_eq!(RenderMethod::splat_contract().id(), MethodId::SplatRaster);
    assert_eq!(RenderMethod::raycast_production().id(), MethodId::RayCast);
    assert_eq!(RenderMethod::raycast_contract().id(), MethodId::RayCast);
    assert_eq!(RayCastConfig::production().accel, AccelKind::TileBins);
    assert_eq!(RayCastConfig::contract().accel, AccelKind::BruteForce);
    assert_ne!(
        RenderMethod::SplatRaster(splat),
        RenderMethod::splat_contract(),
        "pruning a method must delete its knobs with it"
    );
}

#[test]
fn dispatch_reaches_the_method_axis_through_every_other_axis() {
    let axis = [
        (
            RenderMethod::splat_contract(),
            MethodId::SplatRaster,
            cfg!(feature = "method-splat"),
        ),
        (
            RenderMethod::raycast_contract(),
            MethodId::RayCast,
            cfg!(feature = "method-raycast"),
        ),
    ];
    for (method, id, built) in axis {
        for precision in Precision::ALL {
            for kernel in KernelFamily::ALL {
                for collect in [
                    Collect::Pixels,
                    Collect::PixelsAndCounters,
                    Collect::PixelsAndTolerance,
                ] {
                    let mut r = request(method.clone(), precision);
                    r.kernel = kernel;
                    r.collect = collect;
                    match render(&r) {
                        Ok(outcome) => {
                            assert!(
                                built,
                                "{id:?} {precision:?}/{kernel:?}/{collect:?} rendered \
                                 with its feature off"
                            );
                            assert_eq!(outcome.method, id);
                            assert_eq!(outcome.precision, precision);
                            assert_eq!((outcome.image.width, outcome.image.height), (64, 48));
                            assert_eq!(
                                outcome.image.pixels.len(),
                                64 * 48,
                                "every axis point must return the same FloatImage<f64> \
                                 type"
                            );
                        }
                        Err(LightsError::MethodNotImplemented { method: named }) => {
                            assert!(!built);
                            assert_eq!(named, id)
                        }
                        Err(LightsError::ContractViolation(violation)) => {
                            assert_eq!(
                                collect,
                                Collect::PixelsAndTolerance,
                                "{id:?} {precision:?}/{kernel:?}: only the audit \
                                 collect gates on the kernel contract"
                            );
                            assert!(
                                !kernel.curve_is_gaussian() || !kernel.blend_is_over(),
                                "{id:?} {precision:?}/{kernel:?}: a family whose \
                                 curve IS a plain gaussian under an Over blend \
                                 must not be refused: {violation:?}"
                            );
                        }
                        Err(e) => panic!(
                            "{id:?} {precision:?}/{kernel:?}/{collect:?} must reach the \
                             method axis and render: {e:?}"
                        ),
                    }
                }
            }
        }
    }
    assert_eq!(Precision::F32.bytes(), 4);
    assert_eq!(Precision::F64.bytes(), 8);
    assert_eq!(Precision::F32.name(), "f32");
    assert_eq!(Precision::ALL.len(), 2);
}

#[test]
fn dispatch_reports_scene_and_camera_failures_before_the_method() {
    let mut empty = request(RenderMethod::splat_contract(), Precision::F64);
    empty.scene = Scene::<f64>::new();
    assert!(matches!(render(&empty), Err(LightsError::EmptyScene)));

    let mut zero_viewport = request(RenderMethod::splat_contract(), Precision::F64);
    zero_viewport.camera.width = 0;
    assert!(matches!(
        render(&zero_viewport),
        Err(LightsError::ViewportZero { width: 0, .. })
    ));
}

#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn phase_camera() -> lights::camera::Camera<f64> {
    let settings = SharedSettings::<f64>::contract();
    match lights::camera::Camera::<f64>::from_spec(
        &CameraSpec::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 45.0, 64, 48),
        settings.near_plane,
    ) {
        Ok(c) => c,
        Err(e) => panic!("camera: {e:?}"),
    }
}

#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn prepared_of(
    scene: &Scene<f64>,
) -> lights::scene::prepare::PreparedScene<f64, Gaussian> {
    let settings = SharedSettings::<f64>::contract();
    let shape = lights::kernel::CurveShape::gaussian();
    match lights::scene::prepare::PreparedScene::<f64, Gaussian>::build_uniform(
        scene, &shape, &settings,
    ) {
        Ok(p) => p,
        Err(e) => panic!("prepare: {e:?}"),
    }
}

#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn other_scene() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    match scene.push(
        Atom::isotropic(Vec3::new(1.2, 0.7, 5.0), 0.05, 1.0, Rgb::new(0.0, 0.0, 1.0)),
        MIN_SCALE,
    ) {
        Ok(()) => scene,
        Err(e) => panic!("valid: {e:?}"),
    }
}

#[test]
#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn the_build_phase_charges_only_the_atom_count_and_nothing_else() {
    use lights::method::ImageFormation;

    let settings = SharedSettings::<f64>::contract();
    let scene = demo_scene();
    let prepared = prepared_of(&scene);
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let mut strategy = match create_strategy::<f64, Gaussian>(&method, &settings) {
            Ok(s) => s,
            Err(e) => panic!("{:?}: {e:?}", method.id()),
        };
        let report = match strategy.build(&prepared) {
            Ok(r) => r,
            Err(e) => panic!("{:?}: {e:?}", method.id()),
        };
        let c = report.counters;
        assert_eq!(
            c.atoms_prepared,
            prepared.len() as u64,
            "{:?}: build must charge the atom count it was handed",
            method.id()
        );
        assert_eq!(
            (
                c.atoms_projected,
                c.atoms_culled_near,
                c.atoms_culled_det,
                c.atoms_culled_support,
                c.atoms_clamped,
                c.tile_atom_pairs,
                c.sort_keys,
                c.candidates_tested,
                c.kernel_evaluations,
                c.composited_hits,
                c.early_outs,
                c.pixels_visited,
            ),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "{:?}: ImageFormation::build is a per-method hook with no occupant. \
             Every per-atom quantity a renderer needs — the Cholesky, the whitening \
             transform, Lambda — is computed once in PreparedScene::build, BELOW the \
             method axis, and dispatch runs that before create_strategy, so it never \
             lands inside RenderReport.build. If any of these counters is non-zero the \
             phase has acquired real work and the published three-phase breakdown means \
             something different from what it meant when this was written",
            method.id()
        );
        covered += 1;
    }
    assert!(
        covered >= 1,
        "the build-phase assertion ran on {covered} methods, so it measured nothing"
    );
}

#[test]
#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn rebuilding_on_a_new_scene_unbinds_the_camera() {
    use lights::image::Framebuffer;
    use lights::method::ImageFormation;
    use lights::probe::{NoProbe, ProbeSink};

    let settings = SharedSettings::<f64>::contract();
    let camera = phase_camera();
    let first = prepared_of(&demo_scene());
    let second = prepared_of(&other_scene());
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let id = method.id();
        let mut strategy = match create_strategy::<f64, Gaussian>(&method, &settings) {
            Ok(s) => s,
            Err(e) => panic!("{id:?}: {e:?}"),
        };
        if let Err(e) = strategy.build(&first) {
            panic!("{id:?}: {e:?}");
        }
        if let Err(e) = strategy.bind(&first, &camera) {
            panic!("{id:?}: {e:?}");
        }
        if let Err(e) = strategy.build(&second) {
            panic!("{id:?}: {e:?}");
        }
        let mut fb = match Framebuffer::<f64, <Gaussian as lights::kernel::Kernel<f64>>::Blend>::new(
            camera.width,
            camera.height,
            method.tile(),
        ) {
            Ok(f) => f,
            Err(e) => panic!("{id:?}: {e:?}"),
        };
        let mut sink = ProbeSink::<f64>::counters_only(camera.width, camera.height);
        let stale = strategy.render::<NoProbe>(&first, &mut fb, &mut sink);
        assert!(
            stale.is_err(),
            "{id:?}: after build() on a new scene, render() of the scene bind() last saw \
             still succeeded, so build() left the binding in place. Since L-59 the epoch \
             guard refuses a render of the NEW scene on its own, which is why this probe \
             renders the OLD one: it is the only render that reads whether the binding \
             was dropped rather than whether the epochs differ"
        );
        let outcome = strategy.render::<NoProbe>(&second, &mut fb, &mut sink);
        assert!(
            outcome.is_err(),
            "{id:?}: render() ignores the scene it is handed and paints whatever bind() \
             left behind, so a build() on a DIFFERENT scene followed by render() silently \
             produced the PREVIOUS scene's image. build() is the per-scene phase; it must \
             drop the binding it invalidates so the next render fails loudly instead of \
             reporting a stale frame as the new one"
        );
        covered += 1;
    }
    assert!(
        covered >= 1,
        "the rebind assertion ran on {covered} methods, so it measured nothing"
    );
}

#[test]
#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn the_thread_pool_is_built_once_per_renderer_not_once_per_render() {
    use lights::image::Framebuffer;
    use lights::method::{ImageFormation, thread_pool_constructions};
    use lights::probe::{NoProbe, ProbeSink};

    const RENDERS: usize = 5;
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = 3;
    let camera = phase_camera();
    let prepared = prepared_of(&demo_scene());
    let mut covered = 0usize;
    for method in RenderMethod::implemented() {
        let id = method.id();
        let before = thread_pool_constructions();
        let mut strategy = match create_strategy::<f64, Gaussian>(&method, &settings) {
            Ok(s) => s,
            Err(e) => panic!("{id:?}: {e:?}"),
        };
        if let Err(e) = strategy.build(&prepared) {
            panic!("{id:?}: {e:?}");
        }
        if let Err(e) = strategy.bind(&prepared, &camera) {
            panic!("{id:?}: {e:?}");
        }
        for _ in 0..RENDERS {
            let mut fb =
                match Framebuffer::<f64, <Gaussian as lights::kernel::Kernel<f64>>::Blend>::new(
                    camera.width,
                    camera.height,
                    method.tile(),
                ) {
                    Ok(f) => f,
                    Err(e) => panic!("{id:?}: {e:?}"),
                };
            let mut sink = ProbeSink::<f64>::counters_only(camera.width, camera.height);
            if let Err(e) = strategy.render::<NoProbe>(&prepared, &mut fb, &mut sink) {
                panic!("{id:?}: {e:?}");
            }
        }
        let built = thread_pool_constructions() - before;
        assert_eq!(
            built, 1,
            "{id:?}: {RENDERS} renders at threads=3 constructed {built} rayon thread \
             pools. A pool construction spawns and joins its worker threads, so \
             paying it per render turns an explicit thread count into a per-frame \
             tax: a fit that renders ten thousand candidate frames pays it ten \
             thousand times, and the resulting slowdown reads as evidence against \
             tile parallelism rather than against this line. The thread count is \
             fixed at construction, so the pool belongs to the renderer, not to the \
             render. This test is the only one in this file that sets threads above \
             one, which is what makes the process-wide counter readable here"
        );
        covered += 1;
    }
    assert!(
        covered >= 1,
        "the pool-construction assertion ran on {covered} methods"
    );
}

#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn epoch_scene(shift: f64) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    scene
        .push(
            Atom::isotropic(
                Vec3::new(shift, 0.0, 5.0),
                0.05,
                1.0,
                Rgb::new(1.0, 0.0, 0.0),
            ),
            MIN_SCALE,
        )
        .expect("valid");
    scene
}

#[test]
#[cfg(any(feature = "method-splat", feature = "method-raycast"))]
fn a_render_refuses_a_prepared_scene_that_bind_never_saw() {
    use lights::scene::prepare::PreparedScene;

    let settings = SharedSettings::<f64>::contract();
    let camera = phase_camera();
    let shape = lights::kernel::CurveShape::gaussian();

    let bound = PreparedScene::<f64, Gaussian>::build_uniform(&epoch_scene(0.0), &shape, &settings)
        .expect("prepare a");
    let other =
        PreparedScene::<f64, Gaussian>::build_uniform(&epoch_scene(0.7), &shape, &settings)
            .expect("prepare b");
    assert_ne!(
        bound.epoch(),
        other.epoch(),
        "two prepared scenes must never share an epoch, or this whole check is \
         a comparison of one number with itself"
    );

    let mut covered = 0usize;
    for method in every_method() {
        let id = method.id();
        let mut strategy = match create_strategy::<f64, Gaussian>(&method, &settings) {
            Ok(s) => s,
            Err(LightsError::MethodNotImplemented { .. }) => continue,
            Err(e) => panic!("{id:?}: {e:?}"),
        };
        strategy.build(&bound).expect("build");
        strategy.bind(&bound, &camera).expect("bind");

        let mut fb = lights::image::Framebuffer::<f64, lights::blend::Over>::new(
            camera.width,
            camera.height,
            method.tile(),
        )
        .expect("framebuffer");
        let mut sink = lights::probe::ProbeSink::<f64>::counters_only(camera.width, camera.height);

        match strategy.render::<lights::probe::NoProbe>(&other, &mut fb, &mut sink) {
            Err(LightsError::SceneNotBound {
                method: named,
                bound: b,
                given: g,
            }) => {
                assert_eq!(named, id);
                assert_eq!(b, bound.epoch());
                assert_eq!(g, other.epoch());
            }
            Err(e) => panic!("{id:?} refused for the wrong reason: {e:?}"),
            Ok(_) => panic!(
                "{id:?} rendered the scene it was BOUND to and called it the scene \
                 it was HANDED. Both methods copy everything they need at bind and \
                 then ignore the render argument, so the wrong image comes back \
                 with no error at all — which is the shape a host driving a \
                 mutating scene across frames hits first"
            ),
        }
        match strategy.sample_pixel(&other, 3, 4) {
            Err(LightsError::SceneNotBound { method: named, .. }) => assert_eq!(named, id),
            Err(e) => panic!("{id:?} sample_pixel refused for the wrong reason: {e:?}"),
            Ok(_) => panic!("{id:?} sampled a pixel of a scene it was never bound to"),
        }

        assert!(
            strategy
                .render::<lights::probe::NoProbe>(&bound, &mut fb, &mut sink)
                .is_ok(),
            "{id:?}: the scene it WAS bound to must still render, or the guard \
             refuses everything and the test above passes by accident"
        );
        covered += 1;
    }
    assert!(
        covered >= 1,
        "no method was built into this binary, so this case asserted nothing"
    );
}
