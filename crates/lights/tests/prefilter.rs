use lights::agreement::{ContractViolation, assert_contract};
use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::LightsError;
use lights::image::FloatImage;
use lights::kernel::gaussian::{Gaussian, GaussianParams};
use lights::kernel::{Kernel, KernelFamily};
use lights::method::RenderMethod;
use lights::method::splat::SplatConfig;
use lights::numerics::symmetric::Sym2;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::demo::orthographic_camera;
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, PIXEL_BOX_VARIANCE, REFERENCE_DILATION, SharedSettings};

const VIEWPORT: u32 = 128;
const ATOM_SIGMA: f64 = 3.0;
const ATOM_OPACITY: f64 = 0.01;
const DEEP_CUTOFF: f64 = 1.0e-14;
const PEAK_PIXEL: u32 = 64;

fn simpson_second_moment(intervals: usize) -> f64 {
    let a = -0.5f64;
    let b = 0.5f64;
    let n = intervals * 2;
    let h = (b - a) / (n as f64);
    let mut total = a * a + b * b;
    for i in 1..n {
        let x = a + h * (i as f64);
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        total += weight * x * x;
    }
    total * h / 3.0
}

fn single_atom() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    let atom = Atom::isotropic(
        Vec3::new(0.5, 0.5, 4.0),
        ATOM_SIGMA,
        ATOM_OPACITY,
        Rgb::new(1.0, 1.0, 1.0),
    );
    match scene.push(atom, MIN_SCALE) {
        Ok(()) => scene,
        Err(e) => panic!("the probe atom must be renderable: {e:?}"),
    }
}

#[cfg(feature = "method-splat")]
fn deep_tail(mut settings: SharedSettings<f64>) -> SharedSettings<f64> {
    settings.threads = 1;
    settings.blend.alpha_cutoff = DEEP_CUTOFF;
    settings
}

#[cfg(feature = "method-splat")]
fn render_one(settings: SharedSettings<f64>) -> FloatImage<f64> {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::SplatRaster(SplatConfig::contract()),
        settings,
        camera: orthographic_camera(VIEWPORT, VIEWPORT, (VIEWPORT as f64) * 0.5),
        scene: single_atom(),
        collect: Collect::Pixels,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o.image,
        Err(e) => panic!("render failed: {e:?}"),
    }
}

#[cfg(feature = "method-splat")]
fn integral(image: &FloatImage<f64>) -> f64 {
    image.pixels.iter().map(|p| p.x).sum()
}

#[test]
fn pixel_box_variance_is_the_second_moment_of_a_unit_box() {
    let quadrature = simpson_second_moment(500);
    assert!(
        (PIXEL_BOX_VARIANCE - quadrature).abs() < 1.0e-15,
        "PIXEL_BOX_VARIANCE is {PIXEL_BOX_VARIANCE}, but the second moment of a \
         unit box recomputed by Simpson quadrature is {quadrature}. \
         `approximately one pixel of variance` means the variance OF one pixel, \
         which is 1/12 = 0.08333, not variance 1.0: a unit box has second \
         moment 1/12, and Mip-Splatting's published 0.1 is that rounded up. \
         Shipping 1.0 is a filter 3.46x wider in variance and would blur every \
         12px glyph into illegibility, with the report then charging the blur \
         to the representation rather than to this constant"
    );
    assert!(
        (PIXEL_BOX_VARIANCE - 1.0).abs() > 0.9,
        "the constant has been set to a standard deviation of one pixel rather \
         than a variance of one pixel"
    );
}

#[test]
#[cfg(feature = "method-splat")]
fn the_gain_preserves_the_integral_and_not_the_peak() {
    let filtered_settings = deep_tail(SharedSettings::<f64>::prefiltered());
    let mut plain_settings = filtered_settings;
    plain_settings.screen_dilation = 0.0;

    let filtered = render_one(filtered_settings);
    let plain = render_one(plain_settings);

    let filtered_integral = integral(&filtered);
    let plain_integral = integral(&plain);
    assert!(
        plain_integral > 0.0,
        "the unfiltered render produced no signal at all, so nothing below \
         this line is measuring anything"
    );
    let integral_error = (filtered_integral - plain_integral).abs() / plain_integral;
    assert!(
        integral_error < 1.0e-10,
        "the prefilter moved the rendered integral by {integral_error} \
         relative ({filtered_integral} against {plain_integral}). EWA Eq. 22 \
         carries the same prefactor on the resample filter as on the \
         unconvolved footprint, so `peak *= gain` with `cov += sI` gives \
         integral opacity*2*pi*sqrt(det cov) exactly, independent of s. An \
         integral that moves means energy_compensation is not applying the \
         gain the dilation requires"
    );

    let filtered_peak = filtered.get(PEAK_PIXEL, PEAK_PIXEL).x;
    let plain_peak = plain.get(PEAK_PIXEL, PEAK_PIXEL).x;
    assert!(
        plain_peak > 0.0,
        "the probe pixel does not sit on the atom centre, so the peak \
         comparison below would be comparing two tail values"
    );
    let peak_ratio = filtered_peak / plain_peak;
    let closed_form = (ATOM_SIGMA * ATOM_SIGMA) / (ATOM_SIGMA * ATOM_SIGMA + PIXEL_BOX_VARIANCE);
    assert!(
        (peak_ratio - closed_form).abs() < 1.0e-12,
        "the peak ratio is {peak_ratio} but the closed form for an isotropic \
         atom is sigma^2/(sigma^2 + s) = {closed_form}. The integral test above \
         is preserved by ANY dilation whose gain matches it, so it cannot tell \
         a one-pixel-variance filter from a two-pixel-variance one; only the \
         peak can, and this is the assertion that pins which filter was applied"
    );
    assert!(
        (peak_ratio - 1.0).abs() > 1.0e-3,
        "the peak did not move at all (ratio {peak_ratio}). The prefilter is \
         supposed to preserve the integral and NOT the peak; a test that only \
         checked the integral would pass with the gain hardwired to one and \
         the dilation hardwired to zero"
    );
}

#[test]
fn prefiltered_is_refused_by_assert_contract_with_dilation_enabled() {
    let settings = SharedSettings::<f64>::prefiltered();
    let scene = single_atom();
    let outcome = assert_contract(&settings, &settings, 0, 0, &scene);
    match outcome {
        Err(LightsError::ContractViolation(ContractViolation::DilationEnabled)) => (),
        other => panic!(
            "prefiltered() must be refused by the cross-method contract with \
             DilationEnabled, got {other:?}. The agreement instrument bounds \
             SplatRaster against RayCast for the unfiltered footprint; a \
             dilated covariance on one side would surface as a geometry \
             divergence and be read as a defect in the renderer"
        ),
    }
    assert!(
        settings.energy_compensation,
        "prefiltered() must also enable energy_compensation, or the dilation \
         is applied without the prefactor that makes it a filter"
    );
    assert!(
        settings.screen_dilation == PIXEL_BOX_VARIANCE,
        "prefiltered() must dilate by exactly PIXEL_BOX_VARIANCE, got {}",
        settings.screen_dilation
    );
}

#[test]
fn contract_still_has_no_prefilter_at_all() {
    let settings = SharedSettings::<f64>::contract();
    assert!(
        settings.screen_dilation == 0.0,
        "contract() must carry exactly zero screen dilation, got {}",
        settings.screen_dilation
    );
    assert!(
        !settings.energy_compensation,
        "contract() must leave energy compensation off"
    );

    let cov = Sym2::new(9.0, 0.0, 4.0);
    let (_, unchanged, gain) =
        <Gaussian as Kernel<f64>>::prefilter(&GaussianParams, cov, settings.screen_dilation);
    assert!(
        unchanged == cov && gain == 1.0,
        "under contract() the kernel prefilter must return the covariance \
         untouched and a unit gain, got cov {unchanged:?} gain {gain}. This is \
         the fact that makes the unprefiltered column of the P2 report mean \
         `no pixel prefilter` rather than `some other prefilter`"
    );

    let dilated = SharedSettings::<f64>::prefiltered();
    let (_, widened, compensated) =
        <Gaussian as Kernel<f64>>::prefilter(&GaussianParams, cov, dilated.screen_dilation);
    assert!(
        widened.xx > cov.xx && widened.yy > cov.yy && compensated < 1.0,
        "under prefiltered() the same call must widen the covariance and \
         return a gain below one, got cov {widened:?} gain {compensated}"
    );
}

#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn render_with(method: RenderMethod, settings: SharedSettings<f64>) -> FloatImage<f64> {
    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method,
        settings,
        camera: orthographic_camera(VIEWPORT, VIEWPORT, (VIEWPORT as f64) * 0.5),
        scene: single_atom(),
        collect: Collect::Pixels,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o.image,
        Err(e) => panic!("render: {e:?}"),
    }
}

#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn worst_pixel(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    let mut worst = 0.0f64;
    for (l, r) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst
            .max((l.x - r.x).abs())
            .max((l.y - r.y).abs())
            .max((l.z - r.z).abs());
    }
    worst
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_prefilter_is_splat_only_so_the_ray_path_refuses_it_rather_than_ignoring_it() {
    let contract = deep_tail(SharedSettings::<f64>::contract());
    let prefiltered = deep_tail(SharedSettings::<f64>::prefiltered());

    let splat_moved = worst_pixel(
        &render_with(RenderMethod::splat_contract(), contract),
        &render_with(RenderMethod::splat_contract(), prefiltered),
    );
    assert!(
        splat_moved > 1.0e-6,
        "the splat path must read the prefilter, or this case is measuring nothing: \
         worst pixel moved {splat_moved:.3e}"
    );

    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::raycast_contract(),
        settings: prefiltered,
        camera: orthographic_camera(VIEWPORT, VIEWPORT, (VIEWPORT as f64) * 0.5),
        scene: single_atom(),
        collect: Collect::Pixels,
        shapes: None,
    };
    match render(&request) {
        Err(LightsError::PrefilterUnsupported { setting, method }) => {
            assert_eq!(setting, "screen_dilation");
            assert_eq!(method, lights::method::MethodId::RayCast);
        }
        Err(e) => panic!("the ray path must refuse the prefilter BY NAME, got {e:?}"),
        Ok(_) => panic!(
            "THE RAY PATH RENDERED A PREFILTERED CONFIGURATION. Only \
             src/method/splat/project.rs:140 and :151 read screen_dilation and \
             energy_compensation; no file under src/method/raycast/ reads either. \
             Rendering here means an unprefiltered image was produced under a \
             prefiltered configuration, in silence — the defect this refusal exists \
             to end, and the reason assert_contract refuses a non-zero dilation. If \
             the ray caster has GAINED a prefilter, this case and that refusal must \
             be rewritten together, never deleted"
        ),
    }
}

#[test]
#[cfg(feature = "fit")]
fn no_selectable_profile_reaches_the_reference_dilation() {
    use lights::fit::harness::ProfileChoice;
    let choices = ProfileChoice::ALL;
    assert!(
        !choices.is_empty(),
        "ProfileChoice::ALL is empty, so this case would pass by measuring nothing"
    );
    let mut compensating = 0usize;
    for choice in choices.iter() {
        let settings = choice.settings();
        assert!(
            settings.screen_dilation == 0.0 || settings.screen_dilation == PIXEL_BOX_VARIANCE,
            "profile {} renders with screen_dilation {}. The only two dilations any \
             experiment in this crate has run are 0 and PIXEL_BOX_VARIANCE = 1/12, the \
             variance of a one-pixel box. docs/plan/TWO_HORIZON_PLAN.md described the crate \
             as shipping 0.3 'as one constant' for six days on the strength of a constant no \
             selector can reach",
            choice.name(),
            settings.screen_dilation
        );
        assert!(
            settings.screen_dilation != REFERENCE_DILATION,
            "profile {} reaches REFERENCE_DILATION = {REFERENCE_DILATION}. That constant is \
             the Mip-Splatting 0.1 + 0.2 sum the backlog wants DECOMPOSED; if a selectable \
             profile starts using it, every sentence about the decomposition changes meaning \
             and the plan must be re-read before the next experiment runs",
            choice.name()
        );
        if settings.energy_compensation {
            compensating += 1;
        }
    }
    assert_eq!(
        compensating, 1,
        "exactly ONE selectable profile must keep the Mip-Splatting gain. The plan said it \
         was false in BOTH, which stopped being true at commit 6ac92f33 on 2026-09-08, and \
         nothing pinned it"
    );
}
