use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::image::FloatImage;
use lights::kernel::{Carrier, KernelFamily, KernelId, Lobe};
use lights::method::RenderMethod;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, SharedSettings};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

fn scene() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    scene
        .push(
            Atom::isotropic(Vec3::new(0.0, 0.0, 5.0), 0.30, 1.0, Rgb::new(1.0, 0.8, 0.4)),
            MIN_SCALE,
        )
        .expect("valid");
    scene
}

fn request(method: RenderMethod, kernel: KernelFamily) -> RenderRequest {
    RenderRequest {
        precision: Precision::F64,
        kernel,
        method,
        settings: SharedSettings::<f64>::contract(),
        camera: CameraSpec::look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            45.0,
            WIDTH,
            HEIGHT,
        ),
        scene: scene(),
        collect: Collect::Pixels,
        shapes: None,
    }
}

fn rendered(method: &RenderMethod, kernel: KernelFamily) -> FloatImage<f64> {
    match render(&request(method.clone(), kernel)) {
        Ok(outcome) => outcome.image,
        Err(e) => panic!("{} under {:?}: {e:?}", kernel.name(), method.id()),
    }
}

fn max_channel_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let mut worst = 0.0f64;
    for (l, r) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst
            .max((l.x - r.x).abs())
            .max((l.y - r.y).abs())
            .max((l.z - r.z).abs());
    }
    worst
}

fn ink(image: &FloatImage<f64>, background: Rgb<f64>) -> f64 {
    let mut worst = 0.0f64;
    for p in image.pixels.iter() {
        worst = worst
            .max((p.x - background.x).abs())
            .max((p.y - background.y).abs())
            .max((p.z - background.z).abs());
    }
    worst
}

fn methods() -> Vec<RenderMethod> {
    let built = RenderMethod::implemented();
    assert!(
        !built.is_empty(),
        "RenderMethod::implemented() is empty, so every check below would pass \
         by testing nothing"
    );
    built
}

#[test]
fn every_kernel_family_is_reachable_through_render() {
    let background = SharedSettings::<f64>::contract().background;
    for method in methods() {
        let mut ids: Vec<KernelId> = Vec::new();
        for family in KernelFamily::ALL {
            let outcome = match render(&request(method.clone(), family)) {
                Ok(o) => o,
                Err(e) => panic!(
                    "{} is declared in KernelFamily::ALL and render() refused it \
                     under {:?}: {e:?}. Every family in ALL must be instantiable \
                     through dispatch_kernel, which is the whole point of the \
                     enum",
                    family.name(),
                    method.id()
                ),
            };
            assert_eq!(outcome.kernel.id, family.id(), "{}", family.name());
            assert_eq!(outcome.kernel.name, family.kernel_name(), "{}", family.name());
            assert_eq!(outcome.kernel.signed, family.is_signed(), "{}", family.name());
            assert_eq!(
                outcome.kernel.modulated,
                family.is_modulated(),
                "{}",
                family.name()
            );
            assert!(
                ink(&outcome.image, background) > 1.0e-6,
                "{} under {:?} drew nothing at all, so a comparison against it \
                 would be a comparison of two empty frames",
                family.name(),
                method.id()
            );
            if !ids.contains(&outcome.kernel.id) {
                ids.push(outcome.kernel.id);
            }
        }
        assert_eq!(
            ids.len(),
            8,
            "src/kernel/ holds eight Kernel<T> implementations and each carries \
             its own KernelId; render() under {:?} reached {} of them. Four of \
             them were unreachable until L-56: the enum declared four variants \
             over six kernels",
            method.id(),
            ids.len()
        );
    }
}

#[test]
fn the_modulation_hook_fires_on_a_render_path() {
    let background = SharedSettings::<f64>::contract().background;
    for method in methods() {
        let axis = rendered(&method, KernelFamily::GaborColour(Carrier::Axis));
        let quadrature = rendered(&method, KernelFamily::GaborColour(Carrier::Quadrature));
        let plain = rendered(&method, KernelFamily::Gaussian);

        assert!(
            ink(&axis, background) > 1.0e-6 && ink(&quadrature, background) > 1.0e-6,
            "{:?}: one of the two gabor renders is an empty frame",
            method.id()
        );
        assert!(
            max_channel_gap(&axis, &quadrature) > 1.0e-3,
            "{:?}: gabor-colour at phase 0 and at phase pi/2 produced the same \
             image. The two families differ ONLY in CurveShape::phase, so this \
             is the assertion that the family's parameter travels shape() -> \
             params_from -> prepare_screen -> modulate_screen on a render() \
             path",
            method.id()
        );
        assert!(
            max_channel_gap(&axis, &plain) > 1.0e-3,
            "{:?}: gabor-colour eval_screen IS the bare gaussian envelope and \
             its blend is Over, so a renderer that never calls modulate_screen \
             produces a frame bit-identical to the plain gaussian. This check \
             fails exactly when the K::MODULATED branch is dead",
            method.id()
        );

        let first = rendered(&method, KernelFamily::GaussHermiteColour(Lobe::First));
        let second = rendered(&method, KernelFamily::GaussHermiteColour(Lobe::Second));
        assert!(
            max_channel_gap(&first, &second) > 1.0e-3,
            "{:?}: the two hermite orders produced the same image, so \
             CurveShape::order is not reaching the kernel",
            method.id()
        );
        assert!(
            max_channel_gap(&first, &plain) > 1.0e-3,
            "{:?}: gauss-hermite-colour eval_screen is also the bare gaussian \
             envelope, so this is the same dead-branch check on the second \
             modulated kernel",
            method.id()
        );
    }
}

#[test]
fn the_density_arms_fold_their_modulation_into_eval_and_declare_themselves_unmodulated() {
    for family in KernelFamily::ALL {
        assert_eq!(
            family.is_modulated(),
            matches!(
                family,
                KernelFamily::GaborColour(_) | KernelFamily::GaussHermiteColour(_)
            ),
            "{} disagrees with the two families whose eval is the bare envelope",
            family.name()
        );
        assert_eq!(
            family.eval_is_peak_normalised(),
            !matches!(
                family,
                KernelFamily::GaborDensity(Carrier::Quadrature)
                    | KernelFamily::GaussHermiteDensity(_)
            ),
            "{}: peak-normalisation is a property of the PARAMETERS, not of the \
             kernel. gabor-density folds cos(carrier + phase) into eval, so at \
             the origin it returns cos(phase) — which is 1.0 at zero phase and \
             misses only at quadrature. Reading this off the kernel alone gets \
             two of the three gabor-density families wrong",
            family.name()
        );
        assert_eq!(
            family.is_signed(),
            matches!(
                family,
                KernelFamily::GaborDensity(_) | KernelFamily::GaussHermiteDensity(_)
            ),
            "{}: the signed families are exactly the two Additive arms",
            family.name()
        );
    }
    assert!(
        !KernelFamily::GaussianNormalised.is_signed()
            && !KernelFamily::GaussianNormalised.blend_is_over(),
        "signed and not-Over are INDEPENDENT and this family is the standing \
         proof: it composites Normalised, so blend_is_over is false, and it \
         never goes negative, so is_signed is false. A check written as \
         is_signed == !blend_is_over passes on every other family and is wrong \
         here"
    );
    assert_eq!(KernelFamily::FIT_SWEEP.len(), 8);
    assert_eq!(KernelFamily::MODULATED_SWEEP.len(), 10);
    assert_eq!(
        KernelFamily::FIT_SWEEP.len() + KernelFamily::MODULATED_SWEEP.len(),
        KernelFamily::ALL.len(),
        "ALL must be the concatenation of the two sweeps and nothing else, so \
         that a family added to the enum cannot be swept by neither"
    );
    for family in KernelFamily::MODULATED_SWEEP {
        assert!(
            !family.curve_is_gaussian(),
            "{}: the audit and agreement paths gate on curve_is_gaussian and \
             assume a plain gaussian profile. A carrier or a hermite factor is \
             not one, so these families must be REFUSED there rather than \
             compared",
            family.name()
        );
    }
}
