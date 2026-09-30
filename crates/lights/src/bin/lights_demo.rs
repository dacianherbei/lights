use lights::draw::{Clip, DrawList, DrawOp, DrawShape, Ink};
use lights::primitive::{FeatureId, Part, Primitive, PrimitiveSet, Region};
use lights::outline::{Contour, Edge, Filled};
use lights::Similarity2;
use lights::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render, render_depth_limited};
use lights::error::{LightsError, Result};
use lights::image::{FloatImage, RasterLayer, write_png, write_png_normalized};
use lights::kernel::{Beta, Carrier, KernelFamily, Lobe};
use lights::method::{MethodId, RenderMethod};
use lights::camera::Camera;
use lights::kernel::gaussian::Gaussian;
use lights::numerics::quaternion::Quat;
use lights::renderer::Renderer;
use std::f64::consts::TAU;
use lights::numerics::vector::{Rgb, Vec2, Vec3};
use lights::surfel::{Surfel, SurfelScene, render_surfel_layer, render_surfels};
use lights::scene::{Atom, Scene};
use lights::scene::showcase::{
    RING_ATOMS, SURFEL_SHELL_COUNT, demo_object, material_shell, orbit_camera, shell_camera, still_camera, surfel_shell,
};
use lights::settings::{
    DirectionalLight, MIN_SCALE, OrderingModel, PIXEL_BOX_VARIANCE, PixelConvention, SharedSettings,
};
use lights::camera::CameraSpec;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "support/invocation.rs"]
mod invocation;

const USAGE: &str = "\
lights-demo --out-dir <DIR> [options]

Renders one PANEL per feature of the renderer, plus one orbit that moves, and
writes demo.tsv beside the images. `scripts/demo_page.py <DIR>` turns that
directory into the page.

  --out-dir <DIR>     REQUIRED. Every PNG and demo.tsv land here.
  --width <N>         still panel width (default 512)
  --height <N>        still panel height (default 288)
  --frames <N>        orbit frames (default 24; 0 skips the orbit)
  --reps <N>          timed repetitions per panel (default 5, first discarded)

Exit 0 clean; 2 no --out-dir, or a flag this binary does not know.";

#[derive(Clone)]
struct Panel {
    group: &'static str,
    id: String,
    title: String,
    shows: String,
    registers: &'static str,
    request: RenderRequest,
}

struct Timing {
    min_ms: f64,
    mean_ms: f64,
    spread_ms: f64,
    render_ms: f64,
}

struct Args {
    out_dir: PathBuf,
    width: u32,
    height: u32,
    frames: usize,
    reps: usize,
}

fn parse(argv: &[String]) -> std::result::Result<Args, String> {
    let mut out_dir: Option<PathBuf> = None;
    let mut width = 512u32;
    let mut height = 288u32;
    let mut frames = 24usize;
    let mut reps = 5usize;
    let mut skip = false;
    for index in 0..argv.len() {
        if skip {
            skip = false;
            continue;
        }
        let flag = argv[index].as_str();
        let value = argv.get(index + 1);
        let number = |name: &str| -> std::result::Result<usize, String> {
            match value {
                Some(v) => v.parse::<usize>().map_err(|_| format!("{name} needs a number")),
                None => Err(format!("{name} needs a value")),
            }
        };
        match flag {
            "--help" => return Err(USAGE.to_string()),
            "-h" => return Err(USAGE.to_string()),
            "--out-dir" => {
                match value {
                    Some(v) => out_dir = Some(PathBuf::from(v)),
                    None => return Err("--out-dir needs a value".to_string()),
                }
                skip = true;
            }
            "--width" => {
                width = number("--width")? as u32;
                skip = true;
            }
            "--height" => {
                height = number("--height")? as u32;
                skip = true;
            }
            "--frames" => {
                frames = number("--frames")?;
                skip = true;
            }
            "--reps" => {
                reps = number("--reps")?;
                skip = true;
            }
            other => {
                return Err(format!(
                    "unknown flag {other}\n\nA flag this binary does not \
                     understand is REFUSED rather than ignored.\n\n{USAGE}"
                ));
            }
        }
    }
    match out_dir {
        Some(dir) => Ok(Args {
            out_dir: dir,
            width: width.max(32),
            height: height.max(32),
            frames,
            reps: reps.max(2),
        }),
        None => Err("--out-dir is REQUIRED so this binary cannot write to the \
                     repository root by accident"
            .to_string()),
    }
}

fn resting_floor() -> Result<SurfelScene<f64>> {
    let mut scene = SurfelScene::new();
    for iz in -10..=18 {
        for ix in -14..=14 {
            let checker = if (ix + iz) % 2 == 0 { 0.55 } else { 0.45 };
            scene.push(Surfel::new(
                Vec3::new(f64::from(ix) * 0.25, 0.0, f64::from(iz) * 0.25),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec2::new(0.25, 0.25),
                0.99,
                Rgb::splat(checker),
            )?);
        }
    }
    Ok(scene)
}

fn resting_atoms() -> Result<Scene<f64>> {
    let colours = [
        Rgb::new(0.9, 0.2, 0.2),
        Rgb::new(0.2, 0.8, 0.3),
        Rgb::new(0.2, 0.3, 0.9),
        Rgb::new(0.9, 0.8, 0.2),
        Rgb::new(0.8, 0.3, 0.8),
    ];
    let mut scene = Scene::new();
    for (i, (height, colour)) in [1.0, 0.5, 0.0, -0.5, -1.0].iter().zip(colours.iter()).enumerate() {
        let x = -1.6 + 0.8 * i as f64;
        scene.push(Atom::isotropic(Vec3::new(x, -height * 0.35, 0.5), 0.35, 0.9, *colour), 1.0e-6)?;
    }
    Ok(scene)
}

fn base(width: u32, height: u32) -> SharedSettings<f64> {
    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.04, 0.05, 0.07);
    let _ = (width, height);
    settings
}

fn request(
    precision: Precision,
    kernel: KernelFamily,
    method: RenderMethod,
    camera: CameraSpec,
    scene: Scene<f64>,
    settings: SharedSettings<f64>,
) -> RenderRequest {
    RenderRequest {
        precision,
        kernel,
        method,
        settings,
        camera,
        scene,
        collect: Collect::PixelsAndCounters,
        shapes: None,
    }
}

fn needle_scene() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    let colours = [
        Rgb::new(0.95, 0.45, 0.25),
        Rgb::new(0.30, 0.80, 0.95),
        Rgb::new(0.80, 0.85, 0.35),
    ];
    for (index, colour) in colours.iter().enumerate() {
        let angle = 0.6 * index as f64;
        let half = angle * 0.5;
        let atom = Atom::new(
            Vec3::new(-0.5 + 0.5 * index as f64, 0.0, 4.0),
            Quat::new_wxyz(half.cos(), 0.0, 0.0, half.sin()),
            Vec3::new(0.32, 0.007, 0.007),
            0.9,
            *colour,
        );
        if scene.push(atom, MIN_SCALE).is_err() {
            return scene;
        }
    }
    scene
}

fn still_orthographic(width: u32, height: u32) -> CameraSpec {
    orbit_camera(width, height, 0.045).orthographic(0.95)
}

fn needle_camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 34.0, width, height)
        .with_convention(PixelConvention::CornerSample)
}

fn panels(args: &Args, methods: &[RenderMethod]) -> Vec<Panel> {
    let w = args.width;
    let h = args.height;
    let settings = base(w, h);
    let mut out: Vec<Panel> = Vec::new();
    let default_method = methods[0].clone();

    for method in methods.iter() {
        out.push(Panel {
            group: "the method axis",
            id: format!("method-{}", method.id().name()),
            title: format!("{} — the same scene, the other way round", method.id().name()),
            shows: format!(
                "Image formation as an interchangeable axis. {} walks the scene in {:?} \
                 order. Both resolve occlusion, and the agreement harness requires them to \
                 land inside a per-atom tolerance bound at every pixel.",
                method.id().name(),
                method.id().order()
            ),
            registers: "L-20, L-118",
            request: request(
                Precision::F64,
                KernelFamily::Gaussian,
                method.clone(),
                still_camera(w, h),
                demo_object(),
                settings,
            ),
        });
    }

    let kernels = [
        (KernelFamily::Gaussian, "the gaussian, alpha-composited"),
        (KernelFamily::GaussianNormalised, "the gaussian, coverage-normalised"),
        (KernelFamily::SuperGaussian(Beta::Half), "super-gaussian, beta 1/2 — heavier tail"),
        (KernelFamily::SuperGaussian(Beta::Four), "super-gaussian, beta 4 — flatter top"),
        (KernelFamily::CompactBump, "the compact bump — finite support, no tail at all"),
    ];
    for (kernel, label) in kernels.iter() {
        out.push(Panel {
            group: "the kernel axis",
            id: format!("kernel-{}", kernel.name().replace([' ', '.'], "-")),
            title: (*label).to_string(),
            shows: format!(
                "The atom CURVE is an axis of its own: the same six atoms, the same camera, \
                 one token different. `KernelFamily` reaches every kernel src/kernel/ \
                 implements — {} entries over eight `Kernel<T>` impls, with the beta ladder \
                 and the carrier and lobe ladders expanded.",
                KernelFamily::ALL.len()
            ),
            registers: "L-56, L-58",
            request: request(
                Precision::F64,
                *kernel,
                default_method.clone(),
                still_camera(w, h),
                demo_object(),
                settings,
            ),
        });
    }

    let modulated = [
        (
            KernelFamily::GaborColour(Carrier::Axis),
            "gabor, colour-modulated — a cosine carrier across the envelope",
            "The envelope is the plain gaussian and the carrier multiplies the COLOUR, \
             not the weight (src/method/splat/execute.rs, src/method/raycast/execute.rs \
             both gate it on `K::MODULATED`). Set that flag to false and this panel is \
             bit-identical to the first gaussian panel above, which is what \
             the_modulation_hook_fires_on_a_render_path asserts.",
        ),
        (
            KernelFamily::GaborColour(Carrier::Quadrature),
            "gabor at quadrature — the same carrier, a quarter cycle shifted",
            "One field different from the panel before it: phase pi/2 instead of 0. The \
             carrier is zero at the atom centre here, so the centre loses its ink and \
             the envelope is unchanged. That difference is the proof the family's own \
             parameter travels shape() to params_from to prepare_screen and into the \
             renderer.",
        ),
        (
            KernelFamily::GaussHermiteColour(Lobe::Second),
            "gauss-hermite, order 2 — a signed polynomial factor",
            "The modulation is a probabilists' Hermite polynomial along the atom's own \
             axis. At order 2 it is -1 at the centre, so the COLOUR is negated there and \
             the panel shows a dark core inside a bright ring. That is the kernel doing \
             what it says, not an artefact.",
        ),
        (
            KernelFamily::GaborDensity(Carrier::Diagonal),
            "gabor density — additive, and it goes negative",
            "The density arms fold the carrier into eval_screen instead of declaring \
             MODULATED, and they composite Additive, whose SIGNED is true. This is the \
             only group on this page whose kernel may return a negative weight, and the \
             audit and agreement paths REFUSE it rather than compare it — \
             curve_is_gaussian is false for every modulated family.",
        ),
    ];
    for (kernel, label, note) in modulated.iter() {
        out.push(Panel {
            group: "the modulated kernels",
            id: format!("modulated-{}", kernel.name().replace([' ', '.'], "-")),
            title: (*label).to_string(),
            shows: (*note).to_string(),
            registers: "L-56, L-58",
            request: request(
                Precision::F64,
                *kernel,
                default_method.clone(),
                still_camera(w, h),
                demo_object(),
                settings,
            ),
        });
    }

    let carrier_prefilter = [
        (
            SharedSettings::<f64>::contract(),
            "the carrier at 0.25 cyc/px, no pixel prefilter",
            "The control. With no pixel dilation the carrier attenuation factor is exactly \
             1.0, so this panel is the raw plane wave across the envelope and the contract \
             profile is unchanged by L-57.",
        ),
        (
            SharedSettings::<f64>::prefiltered(),
            "the same carrier, band-limited to the pixel box",
            "`prefilter` now attenuates the carrier amplitude by sinc(pi f dx w) sinc(pi f dy w) \
             with w = sqrt(12 * pixel_variance), the exact average of a plane wave over a unit \
             box. At 0.25 cycles per pixel that is a factor of 0.900. It used to be 1.0, and the \
             measured over-brightness ran to 31.98% at 0.40 cyc/px against a 400x400 box \
             quadrature. Both panels render through the SPLAT path: prefilter has exactly one \
             call site in the crate and it is splat-only, so under a raycast-only build this \
             pair is ABSENT rather than quietly unattenuated.",
        ),
    ];
    let splat_only = methods.iter().find(|m| m.id() == MethodId::SplatRaster);
    for (profile, label, note) in carrier_prefilter.iter() {
        let through = match splat_only {
            Some(m) => m.clone(),
            None => continue,
        };
        out.push(Panel {
            group: "the modulated kernels",
            id: format!(
                "carrier-prefilter-{}",
                if profile.screen_dilation > 0.0 { "on" } else { "off" }
            ),
            title: (*label).to_string(),
            shows: (*note).to_string(),
            registers: "L-57",
            request: request(
                Precision::F64,
                KernelFamily::GaborColour(Carrier::Axis),
                through,
                still_camera(w, h),
                demo_object(),
                *profile,
            ),
        });
    }

    let raycast_only = methods.iter().find(|m| m.id() == MethodId::RayCast);
    let carrier_family = KernelFamily::GaborColour(Carrier::Axis);
    let elevations = [
        ("carrier-elevation-flat", "the carrier in the atom's xy plane", 0.0),
        ("carrier-elevation-up", "the same carrier tilted +0.9 rad out of it", 0.9),
        ("carrier-elevation-down", "and tilted -0.9 rad, the mirror of it", -0.9),
    ];
    for (id, label, elevation) in elevations.iter() {
        let through = match raycast_only {
            Some(m) => m.clone(),
            None => continue,
        };
        let deep = demo_object();
        let tilted = carrier_family.shape().with_elevation(*elevation);
        let mut spec = request(
            Precision::F64,
            carrier_family,
            through,
            still_camera(w, h),
            deep.clone(),
            settings,
        );
        spec.shapes = Some(vec![tilted; deep.len()]);
        out.push(Panel {
            group: "the carrier out of plane",
            id: (*id).to_string(),
            title: (*label).to_string(),
            shows: "An atom's SHAPE has always been fully three-dimensional -- the covariance \
                    carries all six terms -- but its MODULATION DIRECTION was pinned to the xy \
                    plane, so no atom could carry a stripe oriented along the view axis. \
                    `CurveShape::with_elevation` lifts it. These three render through the RAY \
                    CASTER and are ABSENT under a splat-only build, deliberately: the screen \
                    path ignores elevation entirely and is bit-identical at every angle, which \
                    is the correct outcome for splat and not an oversight -- a screen-space \
                    atom lives in a plane. Read the +0.9 and -0.9 panels as a PAIR: they share \
                    cos(e) exactly, so their in-plane components are identical and only the \
                    sign of the z term differs. Any visible difference between those two is the \
                    out-of-plane term and nothing else, which is also the only assertion that \
                    can prove the term is live."
                .to_string(),
            registers: "L-151, L-120, L-135",
            request: spec,
        });
    }

    let curve_kernel = KernelFamily::SuperGaussian(Beta::One);
    let mixed_scene = demo_object();
    let atom_count = mixed_scene.len();
    let flat = KernelFamily::SuperGaussian(Beta::Eight).shape();
    let heavy = KernelFamily::SuperGaussian(Beta::Half).shape();
    let mut alternating = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        alternating.push(if index % 2 == 0 { flat } else { heavy });
    }
    let per_atom = [
        (
            "one curve for the whole scene — the control",
            None,
            "curve-uniform",
        ),
        (
            "every other atom flatter than its neighbour",
            Some(alternating),
            "curve-per-atom",
        ),
    ];
    for (label, shapes, id) in per_atom.into_iter() {
        let mut spec = request(
            Precision::F64,
            curve_kernel,
            default_method.clone(),
            still_camera(w, h),
            demo_object(),
            settings,
        );
        spec.shapes = shapes;
        out.push(Panel {
            group: "a curve per atom",
            id: id.to_string(),
            title: label.to_string(),
            shows: "`RenderRequest::shapes` carries one `CurveShape` per atom, so a single                     render can mix curves that no `KernelFamily` can name together. The left                     panel hands dispatch `None`, which builds the same uniform slice                     `build_uniform` used to and is bit-identical to it; the right panel spells                     out beta 8 and beta 1/2 alternating around the ring, and the flatter atoms                     read as harder-edged discs against their neighbours. A list of the wrong                     length is refused rather than padded, because the layer below falls back to                     a default curve per index and would otherwise give the tail atoms a shape                     nobody asked for."
                .to_string(),
            registers: "L-55, L-135",
            request: spec,
        });
    }

    for precision in Precision::ALL.iter() {
        out.push(Panel {
            group: "the precision axis",
            id: format!("precision-{}", precision.name()),
            title: format!("{} — the whole pipeline instantiated at one width", precision.name()),
            shows: "Precision is a type parameter, not a cast at the end: the scene, the \
                    camera, the covariances and the accumulators are all T. The two images \
                    are meant to differ only where the arithmetic does, which the difference \
                    panel measures."
                .to_string(),
            registers: "L-118",
            request: request(
                *precision,
                KernelFamily::Gaussian,
                default_method.clone(),
                still_camera(w, h),
                demo_object(),
                settings,
            ),
        });
    }

    out.push(Panel {
        group: "the camera",
        id: "camera-orthographic".to_string(),
        title: "orthographic — parallel rays, an exact Jacobian".to_string(),
        shows: "Under an orthographic camera the projection Jacobian is constant and exact, \
                so EWA is not an approximation and the two methods are mathematically \
                identical at every pixel. It is a test instrument that happens to look like \
                a feature."
            .to_string(),
        registers: "L-118",
        request: request(
            Precision::F64,
            KernelFamily::Gaussian,
            default_method.clone(),
            still_orthographic(w, h),
            demo_object(),
            settings,
        ),
    });

    let mut prefiltered = base(w, h);
    prefiltered.screen_dilation = PIXEL_BOX_VARIANCE;
    prefiltered.energy_compensation = true;
    let splat = methods.iter().find(|m| m.id() == MethodId::SplatRaster);
    for (label, chosen, note) in [
        ("contract — no pixel prefilter at all", settings,
         "`screen_dilation = 0`. Every cross-method agreement result in this crate is measured \
          here, because src/agreement.rs refuses to compare two methods when either carries a \
          dilation."),
        ("prefiltered — a one-pixel box, and the gain kept", prefiltered,
         "`screen_dilation = PIXEL_BOX_VARIANCE = 1/12`, the exact variance of a box one pixel \
          wide, plus `energy_compensation`, which keeps Mip-Splatting's `sqrt(|S|/|S+sI|)` \
          prefactor instead of discarding it. It preserves the INTEGRAL, not the peak. This is \
          the ONLY dilation any experiment in this crate has ever rendered with: the 0.3 that \
          the backlog called `one constant` reaches only `reference()`, which no profile \
          selector can choose. Both panels are rendered through the SPLAT path because the \
          prefilter is splat-only: src/method/raycast/implementation.rs REFUSES a non-zero \
          dilation rather than ignoring it, so under a raycast-only build this group is absent \
          rather than quietly wrong."),
    ]
    .iter()
    {
        let carrier = match splat {
            Some(m) => m.clone(),
            None => continue,
        };
        out.push(Panel {
            group: "the prefilter",
            id: format!("prefilter-{}", if chosen.screen_dilation > 0.0 { "on" } else { "off" }),
            title: (*label).to_string(),
            shows: (*note).to_string(),
            registers: "L-37, L-18",
            request: request(
                Precision::F64,
                KernelFamily::Gaussian,
                carrier,
                still_camera(w, h),
                demo_object(),
                *chosen,
            ),
        });
    }

    out.push(Panel {
        group: "anisotropy",
        id: "anisotropy-needles".to_string(),
        title: "three 45:1 needles — the case the tile binner is judged on".to_string(),
        shows: "Each atom is a rotated ellipsoid with a 45:1 axis ratio. Until 2026-09-15 the \
                ray caster bounded one of these by the SPHERE that contains it and tested \
                every tile the sphere's projected disc touched; it now bins the exact \
                silhouette conic of the ellipsoid. The tile-atom pair count below is that \
                difference, and the image is required to be bit-identical either way."
            .to_string(),
        registers: "L-20",
        request: request(
            Precision::F64,
            KernelFamily::Gaussian,
            default_method.clone(),
            needle_camera(w, h),
            needle_scene(),
            settings,
        ),
    });

    let raycast = methods.iter().find(|m| m.id() == MethodId::RayCast);
    if let Some(carrier) = raycast {
        let light = match DirectionalLight::new(Vec3::new(1.0, 0.0, 0.0), 0.2, 0.8) {
            Some(l) => l,
            None => lights::guard::unreachable("lights_demo::panels", "a unit x light failed to normalise"),
        };
        for (id, title, note, chosen_light) in [
            (
                "shading-flat",
                "the demo object, flat — every atom wears its baked colour",
                "The ring, its core and the four plates rendered by the ray caster with no light: \
                 the colour is baked per atom and composited, which is all a UI renderer ever \
                 needed. This is the control for the panel beside it.",
                None,
            ),
            (
                "shading-lit",
                "the same object under a light from +x — the gradient normal",
                "The same atoms under a directional light from +x, ambient 0.2 and diffuse 0.8. \
                 No atom carries a normal: at each ray's closest approach the ray caster takes \
                 Lambda (x* - mu), the density gradient, which is the normal of the ellipsoidal \
                 level set through that point — the SILHOUETTE normal, perpendicular to the ray. \
                 So each atom lights on the side facing the light and sits at ambient on the \
                 other, and a light placed at the camera would show almost nothing; \
                 tests/shading.rs measures both and the numbers are in the consolidated research \
                 record. The splat path has only a 2-D conic and refuses a light rather than \
                 rendering flat under it, so this group is absent under a build without the ray \
                 caster.",
                Some(light),
            ),
        ] {
            let mut chosen = settings;
            chosen.light = chosen_light;
            out.push(Panel {
                group: "shading",
                id: id.to_string(),
                title: title.to_string(),
                shows: note.to_string(),
                registers: "L-122, L-141",
                request: request(
                    Precision::F64,
                    KernelFamily::Gaussian,
                    carrier.clone(),
                    still_camera(w, h),
                    demo_object(),
                    chosen,
                ),
            });
        }
    }

    out
}

fn time_panel(panel: &Panel, reps: usize) -> Result<(RenderOutcome, Timing)> {
    let mut outcome = render(&panel.request)?;
    let mut samples: Vec<f64> = Vec::new();
    let mut render_ns = 0u128;
    for _ in 0..reps {
        let started = std::time::Instant::now();
        outcome = render(&panel.request)?;
        samples.push(started.elapsed().as_secs_f64() * 1.0e3);
        render_ns = outcome.report.render.wall_nanos;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_ms = samples[0];
    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    let spread_ms = samples[samples.len() - 1] - min_ms;
    Ok((
        outcome,
        Timing {
            min_ms,
            mean_ms,
            spread_ms,
            render_ms: render_ns as f64 / 1.0e6,
        },
    ))
}

fn row(cells: &[String]) -> Vec<String> {
    cells.to_vec()
}

fn write_panel(dir: &Path, name: &str, image: &FloatImage<f64>) -> Result<String> {
    let file = format!("{name}.png");
    write_png(&dir.join(&file), &image.to_clamped_u8())?;
    Ok(file)
}

const COLUMNS: [&str; 18] = [
    "kind", "group", "id", "frame", "title", "shows", "registers", "file", "width", "height",
    "method", "kernel", "precision", "atoms", "tile_pairs", "kernel_evals", "ms_min", "ms_detail",
];

fn run(args: &Args) -> Result<ExitCode> {
    std::fs::create_dir_all(&args.out_dir)?;
    let methods = RenderMethod::implemented_for_timing();
    if methods.is_empty() {
        println!("no method is compiled in; nothing to show");
        return Ok(ExitCode::SUCCESS);
    }
    let threads = if SharedSettings::<f64>::contract().threads == 0 {
        rayon::current_num_threads()
    } else {
        SharedSettings::<f64>::contract().threads
    };
    println!(
        "lights-demo: {} panels at {}x{}, {} orbit frames, {} reps, {threads} threads",
        panels(args, &methods).len(),
        args.width,
        args.height,
        args.frames,
        args.reps
    );

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut by_id: Vec<(String, FloatImage<f64>)> = Vec::new();

    for panel in panels(args, &methods).iter() {
        let (outcome, timing) = time_panel(panel, args.reps)?;
        let file = write_panel(&args.out_dir, &panel.id, &outcome.image)?;
        println!(
            "  {:<28} {:>8.3} ms  min  ({:.3} mean, spread {:.3})",
            panel.id, timing.min_ms, timing.mean_ms, timing.spread_ms
        );
        rows.push(row(&[
            "panel".to_string(),
            panel.group.to_string(),
            panel.id.clone(),
            String::new(),
            panel.title.clone(),
            panel.shows.clone(),
            panel.registers.to_string(),
            file,
            args.width.to_string(),
            args.height.to_string(),
            outcome.method.name().to_string(),
            outcome.kernel.name.to_string(),
            outcome.precision.name().to_string(),
            panel.request.scene.len().to_string(),
            outcome.counters.tile_atom_pairs.to_string(),
            outcome.counters.kernel_evaluations.to_string(),
            format!("{:.3}", timing.min_ms),
            format!(
                "mean {:.3} ms, spread {:.3} ms over {} reps, render phase {:.3} ms, {} threads, \
                 framebuffer {:.2} MiB (the PADDED tile grid, reallocated by every render() call)",
                timing.mean_ms,
                timing.spread_ms,
                args.reps,
                timing.render_ms,
                threads,
                outcome.report.framebuffer_bytes as f64 / (1024.0 * 1024.0)
            ),
        ]));
        by_id.push((panel.id.clone(), outcome.image));
    }

    let differences = [
        ("method-splat-raster", "method-ray-cast", "difference-methods",
         "splat-raster against ray-cast, per pixel",
         "Two independent walks of the same scene. What is left is the EWA linearisation: \
          the splat path approximates each atom's screen footprint with a Jacobian taken at \
          its centre, the ray path evaluates the exact quadratic along every ray. The \
          agreement harness turns this picture into a per-atom bound and refuses rather \
          than compares when its preconditions are violated.",
         "L-118"),
        ("prefilter-off", "prefilter-on", "difference-prefilter",
         "contract against prefiltered, per pixel",
         "What a one-pixel box costs and buys. Aspect 3 measured this on a 30-degree edge: the \
          prefilter LOWERS relative L2 at every rung of a minification ladder and holds it flat \
          into the small scales, and it WIDENS the 10-90 transition from 0.38-0.57 scene units \
          to 0.63-2.94, so the same edge stops meeting a sub-unit sharpness criterion from 1x \
          down. An arm can be closer to the truth and further from the requirement at once. \
          Read the peak below against the scene: every atom here is many pixels across, so a \
          one-pixel box has little to do. The prefilter is a claim about atoms APPROACHING one \
          pixel, which is the minification regime and not this panel.",
         "L-37"),
        ("precision-f32", "precision-f64", "difference-precision",
         "f32 against f64, per pixel",
         "The same code, one type parameter apart. This is the picture behind every claim \
          in the register that a result is precision-independent.",
         "L-118"),
    ];
    for (left, right, name, title, shows, registers) in differences.iter() {
        let found_left = by_id.iter().find(|(id, _)| id == left);
        let found_right = by_id.iter().find(|(id, _)| id == right);
        if found_left.is_none() || found_right.is_none() {
            continue;
        }
        let (a, b) = match (found_left, found_right) {
            (Some(a), Some(b)) => (a, b),
            (None, Some(_)) => continue,
            (Some(_), None) => continue,
            (None, None) => continue,
        };
        let diff = a.1.difference(&b.1)?;
        let (peak, px, py) = diff.max_abs();
        let file = format!("{name}.png");
        let scale = write_png_normalized(&args.out_dir.join(&file), &diff)?;
        println!("  {name:<28} peak {peak:.3e} at ({px},{py})");
        rows.push(row(&[
            "panel".to_string(),
            "the difference".to_string(),
            (*name).to_string(),
            String::new(),
            (*title).to_string(),
            (*shows).to_string(),
            (*registers).to_string(),
            file,
            args.width.to_string(),
            args.height.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            format!(
                "peak absolute difference {peak:.6e} at pixel ({px},{py}); the image is \
                 normalised by {scale:.6e}, so white is that peak and not 1.0"
            ),
        ]));
    }

    let plate = |width: u32, height: u32| -> FloatImage<f64> {
        let mut out = FloatImage::<f64>::new(width, height);
        for py in 0..height {
            for px in 0..width {
                let checker = ((px / 24) + (py / 24)) % 2 == 0;
                let ramp = px as f64 / width.max(1) as f64;
                let base = if checker { 0.34 } else { 0.22 };
                out.set(px, py, Rgb::new(base + 0.10 * ramp, base, base + 0.16 * (1.0 - ramp)));
            }
        }
        out
    };

    if let Some(method) = methods.first() {
        let request = RenderRequest {
            precision: Precision::F64,
            kernel: KernelFamily::Gaussian,
            method: method.clone(),
            settings: base(args.width, args.height),
            camera: still_camera(args.width, args.height),
            scene: demo_object(),
            collect: Collect::PixelsAndCoverage,
            shapes: None,
        };
        let outcome = render(&request)?;
        let premultiplied = match outcome.premultiplied {
            Some(p) => p,
            None => return Err(LightsError::InvalidCamera("coverage was not collected")),
        };
        let alpha = match outcome.coverage {
            Some(a) => a,
            None => return Err(LightsError::InvalidCamera("coverage was not collected")),
        };
        let over = premultiplied.composite_over(&alpha, &plate(args.width, args.height))?;
        let glyph_size = (args.height / 3).max(8);
        let stroke = |d: f64| ((0.5 - d.abs() + glyph_size as f64 * 0.06).clamp(0.0, 1.0) * 255.0).round() as u8;
        let mut glyph = vec![0u8; (glyph_size * glyph_size) as usize];
        let centre = glyph_size as f64 * 0.5;
        for gy in 0..glyph_size {
            for gx in 0..glyph_size {
                let (dx, dy) = (gx as f64 + 0.5 - centre, gy as f64 + 0.5 - centre);
                let ring = stroke((dx * dx + dy * dy).sqrt() - glyph_size as f64 * 0.36);
                let bar = if dy.abs() <= glyph_size as f64 * 0.36 { stroke(dx) } else { 0 };
                glyph[(gy * glyph_size + gx) as usize] = ring.max(bar);
            }
        }
        let icon_size = (args.height / 5).max(6);
        let mut icon = Vec::with_capacity((icon_size * icon_size * 4) as usize);
        for iy in 0..icon_size {
            for ix in 0..icon_size {
                let (u, v) = (ix as f64 / icon_size as f64, iy as f64 / icon_size as f64);
                let edge = (u.min(1.0 - u).min(v).min(1.0 - v) * icon_size as f64).min(1.0);
                icon.extend_from_slice(&[(40.0 + 200.0 * u) as u8, (200.0 - 120.0 * v) as u8, 220, (edge * 200.0) as u8]);
            }
        }
        let glyph_layer = RasterLayer::from_coverage8(glyph_size, glyph_size, &glyph, Rgb::new(0.98, 0.92, 0.55))?
            .placed(args.width, args.height, (args.width / 2) as i32 - glyph_size as i32 / 2, (args.height / 2) as i32 - glyph_size as i32 / 2)?;
        let icon_layer = RasterLayer::from_rgba8(icon_size, icon_size, &icon)?
            .placed(args.width, args.height, (args.width / 12) as i32, (args.height / 10) as i32)?;
        let rastered = icon_layer.over(&glyph_layer.over(&over)?)?;
        let (w, h) = (args.width as f64, args.height as f64);
        let unit = h / 24.0;
        let mut list = DrawList::<f64>::new();
        let card_min = Vec2::new(w * 0.56, h * 0.08);
        let card_max = Vec2::new(w * 0.96, h * 0.52);
        let header = Vec2::new(card_max.x, card_min.y + unit * 3.0);
        list.push(DrawOp::new(DrawShape::Rect { min: card_min, max: card_max, corner_radius: 6.0 }, Ink::straight(Rgb::new(0.10, 0.12, 0.16), 0.85)))?;
        list.push(DrawOp::new(DrawShape::Rect { min: card_min, max: card_max, corner_radius: 6.0 }, Ink::opaque(Rgb::new(0.22, 0.36, 0.62))).clipped(Clip::new(card_min, Vec2::new(card_max.x, header.y))))?;
        list.push(DrawOp::new(DrawShape::RectOutline { min: card_min, max: card_max, corner_radius: 6.0, half_width: 0.5 }, Ink::opaque(Rgb::splat(0.55))))?;
        let pin_in = Vec2::new(card_min.x, (header.y + card_max.y) * 0.5);
        let wire: Vec<Vec2<f64>> = (0..=24)
            .map(|k| {
                let t = k as f64 / 24.0;
                let x = w * 0.30 + (pin_in.x - w * 0.30) * t;
                let y = h * 0.80 + (pin_in.y - h * 0.80) * (3.0 * t * t - 2.0 * t * t * t);
                Vec2::new(x, y)
            })
            .collect();
        list.push(DrawOp::new(DrawShape::Polyline { points: wire, closed: false, half_width: unit * 0.15 }, Ink::straight(Rgb::new(0.95, 0.75, 0.30), 0.8)))?;
        list.push(DrawOp::new(DrawShape::Disc { centre: pin_in, radius: unit * 0.5 }, Ink::opaque(Rgb::new(0.95, 0.75, 0.30))))?;
        list.push(DrawOp::new(DrawShape::Ring { centre: pin_in, radius: unit * 0.5, half_width: unit * 0.1 }, Ink::opaque(Rgb::splat(0.1))))?;
        let r = unit;
        for i in 0..5 {
            let centre = Vec2::new(card_min.x + unit * 1.5 + i as f64 * r * 2.2 + r, card_max.y - unit * 2.0);
            let points: Vec<Vec2<f64>> = (0..10)
                .map(|k| {
                    let theta = std::f64::consts::PI * k as f64 / 5.0 - std::f64::consts::FRAC_PI_2;
                    let radius = if k % 2 == 0 { r } else { r * 0.45 };
                    Vec2::new(centre.x + radius * theta.cos(), centre.y + radius * theta.sin())
                })
                .collect();
            list.push(DrawOp::new(DrawShape::Polyline { points: points.clone(), closed: true, half_width: 0.5 }, Ink::opaque(Rgb::splat(0.6))))?;
            let fraction = (2.5 - i as f64).clamp(0.0, 1.0);
            if fraction > 0.0 {
                let right = centre.x - r + 2.0 * r * fraction;
                let clip = Clip::new(card_min, Vec2::new(right, card_max.y));
                list.push(DrawOp::new(DrawShape::Polygon { points }, Ink::opaque(Rgb::new(0.98, 0.80, 0.25))).clipped(clip))?;
            }
        }
        let drawn = list.render(args.width, args.height)?.over(&over)?;
        let ops = list.len();
        let zoom = 2.0;
        let focus = Vec2::new((card_min.x + card_max.x) * 0.5, card_max.y - unit * 2.0);
        let view = Similarity2::new(zoom, 0.0, Vec2::new(w * 0.5 - zoom * focus.x, h * 0.5 - zoom * focus.y))?;
        let viewed = list.render_through(view, args.width, args.height)?.over(&over)?;
        let mark = PrimitiveSet {
            ink: Rgb::new(0.45, 0.90, 0.85),
            parts: vec![
                Part { id: FeatureId(0), shape: Primitive::Disc { centre: Vec2::new(16.0, 16.0), radius: 10.0 } },
                Part { id: FeatureId(1), shape: Primitive::Segment { a: Vec2::new(4.0, 28.0), b: Vec2::new(28.0, 4.0), half_width: 1.5 } },
            ],
            regions: vec![Region {
                id: FeatureId(3),
                shape: Filled::new(vec![
                    demo_circle(Vec2::new(40.0, 16.0), 8.0, true),
                    demo_circle(Vec2::new(50.0, 16.0), 8.0, true),
                    demo_circle(Vec2::new(45.0, 16.0), 3.0, false),
                ]),
            }],
            holes: vec![Part { id: FeatureId(2), shape: Primitive::Disc { centre: Vec2::new(16.0, 16.0), radius: 7.0 } }],
        };
        let mark_scale = h / 3.0 / 32.0;
        let mark_view = Similarity2::new(mark_scale, 0.0, Vec2::new(w * 0.06, h * 0.60))?;
        let sdf_drawn = mark.render(mark_view, (args.width, args.height))?.over(&over)?;
        let (peak, _, _) = alpha.max_abs();
        for (name, title, shows, image) in [
            (
                "alpha-coverage",
                "the coverage plane on its own",
                "`Over` accumulates premultiplied colour beside transmittance, so the alpha a                  compositor needs was always in the framebuffer - until 2026-09-15 no public                  call handed it out. White is fully covered, black is untouched background.                  Peak coverage on this scene is stated below."
                    .to_string(),
                alpha.clone(),
            ),
            (
                "alpha-over-plate",
                "the same render composited over a canvas it never saw",
                "This is what a node-graph editor actually needs: draw the scene, hand back                  premultiplied colour and coverage, and let the caller lay it over whatever is                  already on screen. The checkered plate below the atoms was never part of the                  render - `composite_over` put it there afterwards, and the identity                  `resolve(acc, bg) == resolve(acc, 0) + bg*(1 - coverage)` is what makes that                  the same picture the renderer would have produced with that background."
                    .to_string(),
                over,
            ),
            (
                "raster-over-render",
                "a glyph and an icon laid over the atom render",
                "Text and bitmaps do not enter the atom scene: each arrives as its own raster pass \
                 and is composited over the render. The ring-and-bar mark in the middle is a \
                 single-channel COVERAGE bitmap with an ink colour, which is how a glyph shaped by \
                 the system renderer reaches lights (lights takes no font dependency). The tile at \
                 the top left is straight-alpha RGBA, which is how a decoded icon arrives. Both \
                 are drawn by the demo, not by a font or an image file, and neither carries depth: \
                 a raster layer sits wholly in front of what it is laid over."
                    .to_string(),
                rastered,
            ),
            (
                "display-list-over-render",
                "a node card, a wire and a rating drawn by the display list",
                format!(
                    "The UI half of lights: an ordered `DrawList` of {ops} ops, each with its own \
                     premultiplied ink and optional clip, rendered to one raster layer and laid \
                     over the atom render. A translucent card, its header and 1-pixel border are \
                     rects; the wire is ONE polyline, so where its segments meet it is painted \
                     once; the pin is a disc and a ring. The five stars are optical's Rating \
                     widget ported off egui at a value of 2.5: each is an outline polyline and a \
                     filled even-odd polygon, and the third is half-filled by a clip. Coverage is \
                     a clamped signed distance, the same law as the analytic SDF arm."
                ),
                drawn,
            ),
            (
                "display-list-zoomed",
                "the same display list under a 2x zoom and a pan",
                format!(
                    "The list above, unchanged, rendered through a `Similarity2` view: scale \
                     {zoom}, the rating row panned to the centre of the frame. No op was \
                     rewritten; each device pixel is mapped back into list space and the signed \
                     distance found there is multiplied by the scale, so edges stay one device \
                     pixel soft at any zoom. Clips move with the view, which is why the third \
                     star is still cut at its middle. The atom render underneath is NOT zoomed: \
                     it takes a 3-D camera, and this view is the UI half's only."
                ),
                viewed,
            ),
            (
                "sdf-pass-over-render",
                "the analytic SDF pass, in the default build, laid over the atom render",
                format!(
                    "ADR-019's champion for crisp 2-D content is not a render method: it is a \
                     primitive set - two parts, a disc and a capsule, minus one hole - whose \
                     coverage is clamp(1/2 - scale * sdf, 0, 1) at each pixel centre, returned \
                     as a raster layer and composited over the render like the glyph above. \
                     The mark at the lower left is drawn through a Similarity2 of scale \
                     {mark_scale:.3}, so its edges stay one device pixel soft. Until L-157 this \
                     pass lived in the fitter behind the non-default fit feature; it is now \
                     lights::primitive, and the fitter's analytic-sdf arm calls the same code. \
                     Beside it is a Tier 0 region (L-170): two overlapping counter-clockwise \
                     circles and a small clockwise one at their overlap, winding +1 +1 -1. \
                     Under the non-zero rule the middle stays FILLED, and no seam shows where \
                     the contours cross inside the ink, because the region's distance is taken \
                     to its boundary pieces alone."
                ),
                sdf_drawn,
            ),
        ] {
            let file = write_panel(&args.out_dir, name, &image)?;
            println!("  {name:<28} peak coverage {peak:.4}");
            rows.push(row(&[
                "panel".to_string(),
                "compositing".to_string(),
                name.to_string(),
                String::new(),
                title.to_string(),
                shows,
                if name == "raster-over-render" {
                    "L-75"
                } else if name == "display-list-over-render" {
                    "L-76"
                } else if name == "display-list-zoomed" {
                    "L-77"
                } else if name == "sdf-pass-over-render" {
                    "L-157"
                } else {
                    "L-38"
                }
                .to_string(),
                file,
                args.width.to_string(),
                args.height.to_string(),
                outcome.method.name().to_string(),
                outcome.kernel.name.to_string(),
                outcome.precision.name().to_string(),
                request.scene.len().to_string(),
                String::new(),
                String::new(),
                String::new(),
                format!(
                    "peak coverage {peak:.6} over the frame; the plate is generated by the                      demo, not rendered"
                ),
            ]));
        }
    }

    {
        let spec = shell_camera(args.width, args.height);
        let camera = Camera::<f64>::from_spec(&spec, 1.0e-3)?;
        let shell = surfel_shell(SURFEL_SHELL_COUNT);
        let right = camera.rotation.transpose().mul_vec(Vec3::new(1.0, 0.0, 0.0));
        let toward = camera.rotation.transpose().mul_vec(Vec3::new(0.0, 0.0, -1.0));
        for (name, title, shows, direction, material, entry) in [
            (
                "surfel-flat",
                "a sphere of 900 surfels, flat",
                "A surfel is the second primitive kind the PM ruled on 2026-09-16: a gaussian \
                 disc that STORES its normal. This shell is 900 of them on a Fibonacci sphere, \
                 rendered by their own pass with no light - the control for the two panels \
                 beside it.",
                None,
                false,
                "L-122",
            ),
            (
                "surfel-side",
                "the same shell under a light from the right",
                "Ambient 0.2, diffuse 0.8, light from image-right. Unlike the atom panels in the \
                 shading group, the terminator runs down the MIDDLE of the sphere, because each \
                 surfel is lit by the normal it carries rather than by the silhouette normal a \
                 ray finds at its closest approach. tests/surfel.rs holds every pixel of the \
                 opaque interior to the ideal Lambert sphere. The faint hexagonal texture is \
                 not noise: each disc is flat at its own normal, so the shell is faceted at the \
                 spacing of its surfels.",
                Some(right),
                false,
                "L-122",
            ),
            (
                "surfel-headlight",
                "the same shell with the light at the camera",
                "The case the gradient normal cannot serve: a light at the camera kept 0.256 of \
                 the flat luminance on an atom and shows only a rim. A surfel facing the camera \
                 keeps all of it, and this shell keeps about 0.80 - bright in the middle, \
                 darkening toward the limb, which is what a lit sphere looks like.",
                Some(toward),
                false,
                "L-122",
            ),
            (
                "surfel-material",
                "the same shell, glossy, with a lamp cap",
                "Every surfel now carries a material. The body is glossy (specular 0.6, \
                 Blinn-Phong exponent 48) under a light from image-right, so a white highlight \
                 sits where the stored normal bisects the light and the view - tests/surfel.rs \
                 holds it within 4 px of the ideal sphere's half-vector pixel for three light \
                 directions. The cap at the top EMITS: its colour is its emission and does not \
                 depend on the light, which is how a lamp is drawn. The highlight is faceted \
                 for the same reason the shading is: each disc is flat at its own normal.",
                Some(right),
                true,
                "L-154",
            ),
        ] {
            let light = match direction {
                Some(d) => match DirectionalLight::new(d, 0.2, 0.8) {
                    Some(l) => Some(l),
                    None => lights::guard::unreachable(
                        "lights_demo::surfel",
                        "a camera axis failed to normalise",
                    ),
                },
                None => None,
            };
            let scene = if material { material_shell(&shell) } else { shell.clone() };
            let (colour, coverage) = render_surfels(&scene, &camera, light.as_ref())?;
            let (peak, _, _) = coverage.max_abs();
            let file = write_panel(&args.out_dir, name, &colour)?;
            println!("  {name:<28} peak coverage {peak:.4}");
            rows.push(row(&[
                "panel".to_string(),
                "surfel".to_string(),
                name.to_string(),
                String::new(),
                title.to_string(),
                shows.to_string(),
                entry.to_string(),
                file,
                args.width.to_string(),
                args.height.to_string(),
                "surfel pass".to_string(),
                "gaussian disc".to_string(),
                "f64".to_string(),
                shell.len().to_string(),
                String::new(),
                String::new(),
                String::new(),
                format!(
                    "the surfel pass is not a RenderMethod: it runs under every build, and its \
                     premultiplied colour and coverage are what composite_over takes; peak \
                     coverage {peak:.6}"
                ),
            ]));
        }
    }

    if let Some(caster) = methods.iter().find(|m| m.id() == MethodId::RayCast) {
        let spec = CameraSpec::look_at(
            [0.0, -2.5, -5.0],
            [0.0, 0.0, 0.5],
            [0.0, -1.0, 0.0],
            45.0,
            args.width,
            args.height,
        );
        let light = match DirectionalLight::new(Vec3::new(0.3, -1.0, -0.4), 0.25, 0.75) {
            Some(l) => l,
            None => lights::guard::unreachable("lights_demo::resting", "a fixed light failed to normalise"),
        };
        let settings = SharedSettings {
            ordering: OrderingModel::HitDepth,
            light: Some(light),
            ..base(args.width, args.height)
        };
        let camera = Camera::<f64>::from_spec(&spec, settings.near_plane)?;
        let floor = resting_floor()?;
        let layer = render_surfel_layer(&floor, &camera, Some(&light))?;
        let scene = resting_atoms()?;
        let request = RenderRequest {
            precision: Precision::F64,
            kernel: KernelFamily::Gaussian,
            method: caster.clone(),
            settings,
            camera: spec,
            scene,
            collect: Collect::PixelsAndCoverage,
            shapes: None,
        };
        for (name, title, shows, limited) in [
            (
                "resting-depthless",
                "five atoms resting on a surfel floor, composited with no depth",
                "ARTIFACT, SHOWN ON PURPOSE. Five atoms sit on a floor of 841 surfels at heights \
                 of +1 to -1 sigma, and the atom pass is laid over the floor pass by \
                 composite_over, which carries no depth. The halves the floor should bury are \
                 drawn on top of it: tests/interpenetration.rs measured 3,001 of 19,200 pixels \
                 wrong at 160x120, worst 0.39 (L-153).",
                false,
            ),
            (
                "resting-depth",
                "the same scene with the floor's depth carried into the atom pass",
                "The surfel pass also hands out the depth at which each pixel turns opaque \
                 (transmittance at or below 1/1024), and the ray caster rejects every atom hit \
                 beyond it through the ray's t_max. The same composite is then exact: 0 of \
                 19,200 pixels above 1/255 against a depth-interleaved reference, worst 3.9e-5, \
                 at every clearance from -1 to 5 sigma. The HARD ARC where the floor cuts each \
                 atom is not the limit: the ray caster places an atom at ONE depth per ray, its \
                 peak t*, so a ray keeps or drops the whole atom and the reference does the same. \
                 A volumetric truth would clip the density at the plane and soften that arc; \
                 nothing in the crate measures one yet. The splat raster refuses a depth limit \
                 rather than ignoring it, so this group draws through the ray caster only.",
                true,
            ),
        ] {
            let outcome = if limited {
                render_depth_limited(&request, &layer.depth)?
            } else {
                render(&request)?
            };
            let (colour, coverage) = match (outcome.premultiplied.as_ref(), outcome.coverage.as_ref()) {
                (Some(c), Some(a)) => (c, a),
                (None, _) => lights::guard::unreachable("lights_demo::resting", "no premultiplied colour"),
                (Some(_), None) => lights::guard::unreachable("lights_demo::resting", "no coverage"),
            };
            let image = colour.composite_over(coverage, &layer.colour)?;
            let file = write_panel(&args.out_dir, name, &image)?;
            println!("  {name:<28} depth-limited {limited}");
            rows.push(row(&[
                "panel".to_string(),
                "resting on a floor".to_string(),
                name.to_string(),
                String::new(),
                title.to_string(),
                shows.to_string(),
                "L-153, L-156".to_string(),
                file,
                args.width.to_string(),
                args.height.to_string(),
                outcome.method.name().to_string(),
                outcome.kernel.name.to_string(),
                outcome.precision.name().to_string(),
                request.scene.len().to_string(),
                outcome.counters.tile_atom_pairs.to_string(),
                outcome.counters.kernel_evaluations.to_string(),
                String::new(),
                format!(
                    "{} of {} floor pixels carry a finite depth",
                    layer.depth.finite_count(),
                    args.width * args.height
                ),
            ]));
        }
    }

    if let Some(method) = methods.first() {
        let full = still_camera(args.width, args.height);
        let rect = (
            args.width / 2 - args.width / 6,
            args.height / 2 - args.height / 6,
            args.width / 3,
            args.height / 3,
        );
        let window = full.sub_rect(rect.0, rect.1, rect.2, rect.3)?;
        let request = |camera: CameraSpec, scene: Scene<f64>| RenderRequest {
            precision: Precision::F64,
            kernel: KernelFamily::Gaussian,
            method: method.clone(),
            settings: base(args.width, args.height),
            camera,
            scene,
            collect: Collect::Pixels,
            shapes: None,
        };
        let before = render(&request(full.clone(), demo_object()))?;
        let mut edited = demo_object();
        let moved = edited.atoms()[RING_ATOMS];
        let mut lifted = moved;
        lifted.mean = moved.mean + Vec3::new(0.0, -0.20, 0.0);
        lifted.colour = Rgb::new(0.25, 0.95, 1.0);
        edited.set(RING_ATOMS, lifted, MIN_SCALE)?;
        let patch = render(&request(window, edited))?;
        let mut assembled = before.image.clone();
        assembled.blit_from(&patch.image, None, rect.0, rect.1)?;

        let whole_pixels = u64::from(args.width) * u64::from(args.height);
        let patch_pixels = u64::from(rect.2) * u64::from(rect.3);
        let share = 100.0 * patch_pixels as f64 / whole_pixels as f64;
        for (name, title, shows, image, width, height) in [
            (
                "damage-patch",
                "the damaged rect, rendered on its own",
                format!(
                    "The bright core moved up and turned cyan. Only the {}x{} window around it was \
                     shaded - {patch_pixels} of {whole_pixels} pixels, {share:.1}% of the \
                     frame. The window is not a clip: it is a CAMERA, built by shifting the \
                     principal point by (-{}, -{}) and keeping the focal length, so every ray \
                     through it is the ray the whole frame would have cast.",
                    rect.2, rect.3, rect.0, rect.1
                ),
                patch.image.clone(),
                rect.2,
                rect.3,
            ),
            (
                "damage-assembled",
                "the patch blitted back into the frame before the edit",
                "The frame below was rendered ONCE, before the edit; only the rect above was \
                 redrawn and blitted in. LOOK AT THE SEAM - it is deliberate, and it is the \
                 rule this panel exists to state: a damaged rect must cover the union of the \
                 OLD and NEW footprints of everything that changed. This one does not, so the \
                 moved atom's former glow survives outside the rect and its new one stops at \
                 the edge. Where the scene does NOT change the window is exact: tests/camera.rs \
                 renders the same scene through the full camera and through the shifted one on \
                 a 128x96 frame and finds the ray caster reproduces every pixel EXACTLY (0 of \
                 12,288 differ) and the splat path to one ULP (468 differ, worst 2.22e-16) - \
                 because splat projects an atom centre into screen space and the ray caster \
                 carries the pixel into world space."
                    .to_string(),
                assembled,
                args.width,
                args.height,
            ),
        ] {
            let file = write_panel(&args.out_dir, name, &image)?;
            println!("  {name:<28} {patch_pixels} of {whole_pixels} pixels shaded ({share:.1}%)");
            rows.push(row(&[
                "panel".to_string(),
                "damage".to_string(),
                name.to_string(),
                String::new(),
                title.to_string(),
                shows,
                "L-41".to_string(),
                file,
                width.to_string(),
                height.to_string(),
                patch.method.name().to_string(),
                patch.kernel.name.to_string(),
                patch.precision.name().to_string(),
                RING_ATOMS.to_string(),
                String::new(),
                String::new(),
                String::new(),
                format!(
                    "the rect is {}x{} at ({},{}): {patch_pixels} of {whole_pixels} pixels, \
                     {share:.1}% of the frame, and its framebuffer is {:.2} MiB against the \
                     full frame's {:.2} MiB",
                    rect.2,
                    rect.3,
                    rect.0,
                    rect.1,
                    patch.report.framebuffer_bytes as f64 / (1024.0 * 1024.0),
                    before.report.framebuffer_bytes as f64 / (1024.0 * 1024.0)
                ),
            ]));
        }
    }

    #[cfg(feature = "fit")]
    {
        use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
        use lights::fit::scale::{Scale, ScaleClass};
        let class = ScaleClass::soft_shadow();
        for (numerator, label) in [(1u32, "1x"), (2u32, "2x"), (4u32, "4x")] {
            let scale = Scale::new(numerator, 1)?;
            let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
            let truth = class.truth(scale, DEFAULT_MAX_DEPTH, &mut budget)?;
            let name = format!("scale-n2s-{label}");
            let file = write_panel(&args.out_dir, &name, &truth)?;
            println!("  {name:<28} {}x{}", truth.width, truth.height);
            rows.push(row(&[
                "panel".to_string(),
                "resolution independence".to_string(),
                name,
                String::new(),
                format!("the soft-shadow class at {label}"),
                format!(
                    "ONE continuous object, sampled at {label}. The card's edge is hard and \
                     resolves to whatever the sampling rate allows; the shadow's penumbra is a \
                     SCENE length, so it covers about {} device pixels here against 15 at 1x - a \
                     shadow whose sigma were fixed in pixels would hold that number flat and \
                     the class would be measuring its own blur radius shrinking. This is the \
                     only content class in the crate carrying a hard and a soft edge in one \
                     image, and until 2026-09-15 it could not enter the zoom ladder at all \
                     because no analytic sampler for it existed.",
                    15 * numerator
                ),
                "L-42".to_string(),
                file,
                truth.width.to_string(),
                truth.height.to_string(),
                String::new(),
                String::new(),
                "f64".to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                format!(
                    "box-averaged at {} samples per axis per pixel; the shadow is an \
                     erf of the rounded-rect distance, exact for a straight edge and \
                     approximate within 2 sigma of the four corners",
                    scale.supersample()
                ),
            ]));
        }
    }

    #[cfg(feature = "fit")]
    {
        use lights::fit::atom2d::camera as fit_camera;
        use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
        use lights::fit::greedy::{FitRequest, fit, fit_budget, render_or_background};
        use lights::fit::metric::{ExactReference, psnr};
        use lights::fit::target::Target;
        use lights::settings::SharedSettings;

        const FIT_TILE: u32 = 16;
        const FIT_RENDERS: u64 = 40_000_000;

        let target = Target::<f64>::t2_straight_edge();
        let (tw, th) = target.viewport;
        let mut cells = coverage_budget(DEFAULT_CELL_LIMIT);
        let reference = ExactReference::from_target(&target, DEFAULT_MAX_DEPTH, &mut cells)?;
        let undecided = reference.undecided_fraction();
        let file = write_panel(&args.out_dir, "distortion-reference", reference.image())?;
        println!("  {:<28} {tw}x{th} exact", "distortion-reference");
        rows.push(row(&[
            "panel".to_string(),
            "distortion".to_string(),
            "distortion-reference".to_string(),
            String::new(),
            "the exact raster the numbers are against".to_string(),
            "An adaptive quadrature of a signed distance field, not a render: each \
             pixel is subdivided until every cell is farther from the edge than its \
             own half-diagonal, and then contributes its exact area. There is no \
             sampling noise in this image, which is the whole reason a decibel \
             figure against it means anything. The crate's other reference - a \
             stochastic path tracer's frame, about 2x overbright with a clipped sky - \
             is still banned, and the guard that used to ban BOTH now tells them \
             apart."
                .to_string(),
            "L-130".to_string(),
            file,
            tw.to_string(),
            th.to_string(),
            String::new(),
            String::new(),
            "f64".to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            format!(
                "undecided area fraction {undecided:.3e}, which is the reference's \
                 own error bound; from_target refuses anything above 1e-4"
            ),
        ]));

        for count in [1usize, 2, 4] {
            let mut request =
                FitRequest::new(KernelFamily::Gaussian, SharedSettings::<f64>::prefiltered());
            request.max_atoms = count;
            request.tile = FIT_TILE;
            let mut budget = fit_budget(FIT_RENDERS);
            let outcome = fit(&target, reference.image(), &request, &mut budget)?;
            let settings = SharedSettings::<f64>::prefiltered();
            let drawn = render_or_background(
                &outcome.atoms,
                &fit_camera(tw, th),
                KernelFamily::Gaussian,
                &settings,
                FIT_TILE,
            )?;
            let decibels = psnr(&drawn, &reference)?;
            let name = format!("distortion-{count}-atoms");
            let file = write_panel(&args.out_dir, &name, &drawn)?;
            println!("  {name:<28} {decibels:.2} dB");
            let stopped = if outcome.improvement_exhausted {
                "the FITTER stopped, not the allowance"
            } else {
                "the allowance stopped, not the fitter"
            };
            rows.push(row(&[
                "panel".to_string(),
                "distortion".to_string(),
                name,
                String::new(),
                format!("{count} atoms, {decibels:.2} dB"),
                format!(
                    "The greedy matching pursuit's answer at {count} atoms, scored \
                     against the exact raster beside it. {decibels:.2} dB is a number \
                     about THIS FITTER, not about the representation: the crate's only \
                     search is a coordinate ladder, so every atom count it publishes is \
                     an upper bound on cost and a lower bound on quality. Here {stopped}. \
                     The panel exists so that a gradient fitter, when one is built, has \
                     a picture and a figure to beat rather than a paragraph."
                ),
                "L-130".to_string(),
                file,
                tw.to_string(),
                th.to_string(),
                "splat".to_string(),
                KernelFamily::Gaussian.name().to_string(),
                "f64".to_string(),
                count.to_string(),
                String::new(),
                String::new(),
                String::new(),
                format!(
                    "{} full re-renders were spent placing and refining {count} \
                     atoms; psnr is 10*log10(1/mse) with mse averaged over three \
                     channels per pixel",
                    outcome.renders
                ),
            ]));
        }
    }

    #[cfg(feature = "fit")]
    {
        use lights::fit::atom2d::{Atom2d, camera as fit_camera};
        use lights::fit::coverage::{
            DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget, coverage_image_to_colour,
            exact_coverage,
        };
        use lights::fit::edge::{Criterion, measure_target};
        use lights::fit::greedy::{FitRequest, fit, fit_budget, render_or_background};
        use lights::fit::target::Target;
        use lights::numerics::vector::Vec2;
        use lights::settings::SharedSettings;

        const FIT_TILE: u32 = 16;
        const FIT_RENDERS: u64 = 40_000_000;
        const STROKE_LONG: f64 = 10.0;
        const STROKE_ACROSS: f64 = 0.75;

        let target = Target::<f64>::t5_thin_stroke();
        let (tw, th) = target.viewport;
        let mut cells = coverage_budget(DEFAULT_CELL_LIMIT);
        let (coverage, _) = exact_coverage(&target, DEFAULT_MAX_DEPTH, &mut cells)?;
        let truth = coverage_image_to_colour(&target, &coverage);
        let settings = SharedSettings::<f64>::prefiltered();
        let criterion = Criterion::registered();

        let hand = [Atom2d::from_sigma(
            Vec2::new(32.0, 32.0),
            Vec2::new(STROKE_LONG, STROKE_ACROSS),
            30.0f64.to_radians(),
            1.0,
            Rgb::new(1.0, 1.0, 1.0),
        )];
        let mut request =
            FitRequest::new(KernelFamily::Gaussian, SharedSettings::<f64>::prefiltered());
        request.max_atoms = 12;
        request.tile = FIT_TILE;
        let mut budget = fit_budget(FIT_RENDERS);
        let outcome = fit(&target, &truth, &request, &mut budget)?;

        for (name, atoms, title) in [
            (
                "criterion-one-hand-atom",
                hand.to_vec(),
                "one hand-placed atom, and it PASSES",
            ),
            (
                "criterion-twelve-fitted-atoms",
                outcome.atoms.clone(),
                "twelve fitted atoms, and they FAIL",
            ),
        ] {
            let drawn = render_or_background(
                &atoms,
                &fit_camera(tw, th),
                KernelFamily::Gaussian,
                &settings,
                FIT_TILE,
            )?;
            let metric = measure_target(&drawn, &target, &criterion)?;
            let residual = {
                let difference = truth.difference(&drawn)?;
                let mut acc = 0.0f64;
                for q in difference.pixels.iter() {
                    acc += q.x * q.x + q.y * q.y + q.z * q.z;
                }
                acc
            };
            let verdict = if metric.meets { "MEETS" } else { "MISSES" };
            let file = write_panel(&args.out_dir, name, &drawn)?;
            println!("  {name:<28} {verdict}, residual {residual:.3}");
            rows.push(row(&[
                "panel".to_string(),
                "the loss and the criterion".to_string(),
                name.to_string(),
                String::new(),
                title.to_string(),
                format!(
                    "The fitter minimises an L2 sum over pixels. Every published verdict in \
                     docs/lights/P2_RESULTS.md comes from a GEOMETRIC criterion instead: peak \
                     at least 0.90 of the target contrast, full width at half maximum 1.0 \
                     plus or minus 0.5. These two pictures are the gap between those \
                     functions. This one {verdict} the criterion at peak {:.3} and width \
                     {:.3}, and sits at residual energy {residual:.3}. The atom that passes \
                     costs more than ten times the L2 of the twelve that do not, so every \
                     atom count this project has published is an upper bound on cost and a \
                     lower bound on quality.",
                    metric.peak_frac, metric.fwhm
                ),
                "L-44".to_string(),
                file,
                tw.to_string(),
                th.to_string(),
                "splat".to_string(),
                KernelFamily::Gaussian.name().to_string(),
                "f64".to_string(),
                atoms.len().to_string(),
                String::new(),
                String::new(),
                String::new(),
                format!(
                    "judged probes {}, peak {:.4}, fwhm {:.4}, residual energy {residual:.4}",
                    metric.judged, metric.peak_frac, metric.fwhm
                ),
            ]));
        }
    }

    let mut warm_note = String::new();
    if args.frames > 0 {
        let fw = (args.width * 3) / 4;
        let fh = (args.height * 3) / 4;
        let settings = base(fw, fh);
        let mut frame_ms: Vec<f64> = Vec::new();
        {
            let still = demo_object();
            let shape = KernelFamily::Gaussian.shape();
            let cold_request = request(
                Precision::F32,
                KernelFamily::Gaussian,
                methods[0].clone(),
                orbit_camera(fw, fh, 0.0),
                still.clone(),
                settings,
            );
            let mut cold: Vec<f64> = Vec::new();
            for turn in 0..args.reps {
                let mut moved = cold_request.clone();
                moved.camera = orbit_camera(fw, fh, turn as f64 / args.reps as f64);
                let started = std::time::Instant::now();
                let _ = render(&moved)?;
                cold.push(started.elapsed().as_secs_f64() * 1.0e3);
            }
            let settings32 = SharedSettings::<f32>::from_f64(&settings);
            let scene32 = Scene::<f32>::from_f64(&still);
            let mut renderer = Renderer::<f32, Gaussian>::new(
                &methods[0],
                &settings32,
                &shape,
                fw,
                fh,
                &scene32,
            )?;
            let mut warm: Vec<f64> = Vec::new();
            for turn in 0..args.reps {
                let spec = orbit_camera(fw, fh, turn as f64 / args.reps as f64);
                let camera = Camera::<f32>::from_spec(&spec, settings32.near_plane)?;
                let started = std::time::Instant::now();
                renderer.render_frame(&camera)?;
                let _ = renderer.resolve(settings32.background);
                warm.push(started.elapsed().as_secs_f64() * 1.0e3);
            }
            cold.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            warm.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            warm_note = format!(
                "the same static scene through the one-shot render() and through a held \
                 Renderer, {} camera moves each, best of: {:.3} ms cold against {:.3} ms warm, \
                 {:.2}x - the cold path rebuilds the scene, the strategy and a {:.2} MiB \
                 framebuffer on every call",
                args.reps,
                cold[0],
                warm[0],
                cold[0] / warm[0].max(1.0e-9),
                renderer.framebuffer().resident_bytes() as f64 / (1024.0 * 1024.0)
            );
            println!("  warm path: {warm_note}");
        }
        let spin_axis = Vec3::new(0.18, 1.0, 0.12);
        let settings32 = SharedSettings::<f32>::from_f64(&settings);
        let shape = KernelFamily::Gaussian.shape();
        let mut spun = demo_object();
        let mut engine = Renderer::<f32, Gaussian>::new(
            &methods[0],
            &settings32,
            &shape,
            fw,
            fh,
            &Scene::<f32>::from_f64(&spun),
        )?;
        for frame in 0..args.frames {
            let turn = frame as f64 / args.frames as f64;
            let delta = match Quat::from_axis_angle(spin_axis, TAU / args.frames as f64) {
                Some(q) => q,
                None => Quat::identity(),
            };
            let mut next = Scene::<f64>::new();
            for atom in spun.atoms().iter() {
                let mut moved = *atom;
                moved.mean = delta.rotate(atom.mean);
                moved.rotation = delta.mul(atom.rotation);
                next.push(moved, MIN_SCALE)?;
            }
            spun = next;
            let spec = orbit_camera(fw, fh, turn * 0.5);
            let camera = Camera::<f32>::from_spec(&spec, settings32.near_plane)?;
            let started = std::time::Instant::now();
            engine.set_scene(&Scene::<f32>::from_f64(&spun))?;
            let counters = engine.render_frame_counted(&camera)?;
            let image = engine.resolve(settings32.background);
            let elapsed = started.elapsed().as_secs_f64() * 1.0e3;
            let file = write_panel(&args.out_dir, &format!("orbit-{frame:03}"), &image.to_f64())?;
            frame_ms.push(elapsed);
            rows.push(row(&[
                "frame".to_string(),
                "in motion".to_string(),
                "orbit".to_string(),
                frame.to_string(),
                String::new(),
                String::new(),
                "L-121, L-123, L-129".to_string(),
                file,
                fw.to_string(),
                fh.to_string(),
                engine.method().name().to_string(),
                "gaussian".to_string(),
                "f32".to_string(),
                spun.len().to_string(),
                counters.tile_atom_pairs.to_string(),
                counters.kernel_evaluations.to_string(),
                format!("{elapsed:.3}"),
                String::new(),
            ]));
        }
        frame_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = frame_ms[0];
        let median = frame_ms[frame_ms.len() / 2];
        let worst = frame_ms[frame_ms.len() - 1];
        println!(
            "  orbit: {} frames at {fw}x{fh}, {min:.2} / {median:.2} / {worst:.2} ms \
             (best / median / worst), median {:.1} frames per second",
            args.frames,
            1000.0 / median
        );
        rows.push(row(&[
            "panel".to_string(),
            "in motion".to_string(),
            "orbit".to_string(),
            String::new(),
            format!("one turn of the camera, {} frames", args.frames),
            format!(
                "A still image cannot show that the renderer can be driven frame by frame, \
                 so this one moves. Each frame is one full `render()` call — build, bind, \
                 clear, shade — at {fw}x{fh} f32 on {} atoms across {threads} threads. \
                 Best / median / worst are {min:.2} / {median:.2} / {worst:.2} ms, a median \
                 of {:.1} frames per second. THE SCENE CHANGES TOO: every frame each atom's \
                 mean is rotated and its orientation composed with a delta quaternion through \
                 `Quat::mul`, then written back with `Scene::set`/`push` so the ingest floor is \
                 re-applied on every write. Before 2026-09-15 neither call existed and a scene \
                 was write-once. Nothing in lights owns this cycle: the camera and the scene are \
                 both moved from outside and the crate stays headless.",
                demo_object().len(),
                1000.0 / median
            ),
            "L-121, L-123".to_string(),
            "orbit-000.png".to_string(),
            fw.to_string(),
            fh.to_string(),
            String::new(),
            String::new(),
            "f32".to_string(),
            demo_object().len().to_string(),
            String::new(),
            String::new(),
            format!("{min:.3}"),
            format!(
                "median {median:.3} ms = {:.1} fps, worst {worst:.3} ms over {} frames, \
                 {threads} threads. WARM PATH: {warm_note}",
                1000.0 / median,
                args.frames
            ),
        ]));
    }

    let header = vec![
        format!("threads\t{threads}"),
        format!("panel_size\t{}x{}", args.width, args.height),
        format!("orbit_frames\t{}", args.frames),
        format!("reps\t{}", args.reps),
        "warning\tdebug builds are several times slower than release; every \
         millisecond here belongs to the build profile that produced it"
            .to_string(),
    ];
    let columns: Vec<&str> = COLUMNS.to_vec();
    let text = invocation::table(&header, &columns, &rows);
    std::fs::write(args.out_dir.join("demo.tsv"), text)?;
    println!("\ndemo.tsv written to {}", args.out_dir.display());
    println!("the page: python3 crates/lights/scripts/demo_page.py {}", args.out_dir.display());
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(message) => {
            println!("{message}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            println!("lights-demo failed: {e:?}");
            let _ = LightsError::InvalidCamera("");
            ExitCode::from(4)
        }
    }
}

fn demo_circle(centre: Vec2<f64>, radius: f64, counter_clockwise: bool) -> Contour {
    let pi = std::f64::consts::PI;
    let sign = if counter_clockwise { 1.0 } else { -1.0 };
    Contour {
        edges: vec![
            Edge::Arc { centre, radius, from: 0.0, to: sign * pi },
            Edge::Arc { centre, radius, from: sign * pi, to: sign * 2.0 * pi },
        ],
    }
}
