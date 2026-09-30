use lights::agreement::run_pair;
use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderRequest};
use lights::error::{LightsError, Result};
use lights::image::{FloatImage, write_png, write_png_normalized};
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::numerics::vector::Rgb;
use lights::scene::Scene;
use lights::scene::demo::{
    narrow_camera, sphere_camera, spheres, spheres_anisotropic,
    ordering_probe_scene, spheres_anisotropic_no_ground, spheres_no_ground,
    stress_scene,
};
use lights::settings::{OrderingModel, SharedSettings};
use std::path::PathBuf;
use std::process::ExitCode;

#[path = "support/invocation.rs"]
mod invocation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Strict,
    Loose,
}

#[derive(Clone, Debug, PartialEq)]
enum SceneChoice {
    Spheres,
    SpheresNoGround,
    SpheresAnisotropic,
    SpheresAnisotropicNoGround,
    Needles,
    Stress { atoms: usize },
}

#[derive(Clone, Debug)]
struct Args {
    scene: SceneChoice,
    camera: &'static str,
    width: u32,
    height: u32,
    precision: Precision,
    mode: Mode,
    out_dir: PathBuf,
}

const USAGE: &str = "\
lights-agreement --out-dir <DIR> [options]

  --out-dir <DIR>     REQUIRED. Where the four PNGs are written.
  --scene <NAME>      spheres | spheres-no-ground |
                      spheres-anisotropic (default) |
                      spheres-anisotropic-no-ground | needles | stress:<N>
  --camera <NAME>     spheres (default, 121 degree horizontal field) | narrow
  --width <N>         default 640
  --height <N>        default 360
  --precision <NAME>  f32 | f64 (default)
  --mode <NAME>       strict (default) | loose

Exits nonzero if any pixel exceeds its justified per-pixel tolerance.";

fn parse() -> std::result::Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut scene = SceneChoice::SpheresAnisotropic;
    let mut camera = "spheres";
    let mut width = 640u32;
    let mut height = 360u32;
    let mut precision = Precision::F64;
    let mut mode = Mode::Strict;
    let mut out_dir: Option<PathBuf> = None;
    let mut skip = false;

    for index in 0..raw.len() {
        if skip {
            skip = false;
            continue;
        }
        let flag = raw[index].as_str();
        let value = raw.get(index + 1);
        let need = |v: Option<&String>| -> std::result::Result<String, String> {
            match v {
                Some(s) => Ok(s.clone()),
                None => Err(format!("{flag} needs a value")),
            }
        };
        match flag {
            "--help" | "-h" => return Err(USAGE.to_string()),
            "--out-dir" => {
                out_dir = Some(PathBuf::from(need(value)?));
                skip = true;
            }
            "--camera" => {
                camera = match need(value)?.as_str() {
                    "spheres" => "spheres",
                    "narrow" => "narrow",
                    other => return Err(format!("unknown camera {other}")),
                };
                skip = true;
            }
            "--scene" => {
                let name = need(value)?;
                scene = match name.as_str() {
                    "needles" => SceneChoice::Needles,
                    "spheres" => SceneChoice::Spheres,
                    "spheres-no-ground" => SceneChoice::SpheresNoGround,
                    "spheres-anisotropic" => SceneChoice::SpheresAnisotropic,
                    "spheres-anisotropic-no-ground" => {
                        SceneChoice::SpheresAnisotropicNoGround
                    }
                    other => match other.strip_prefix("stress:") {
                        Some(count) => match count.parse::<usize>() {
                            Ok(n) => SceneChoice::Stress { atoms: n },
                            Err(e) => return Err(format!("stress count: {e}")),
                        },
                        None => return Err(format!("unknown scene {other}")),
                    },
                };
                skip = true;
            }
            "--width" => {
                width = need(value)?.parse::<u32>().map_err(|e| format!("width: {e}"))?;
                skip = true;
            }
            "--height" => {
                height = need(value)?
                    .parse::<u32>()
                    .map_err(|e| format!("height: {e}"))?;
                skip = true;
            }
            "--precision" => {
                precision = match need(value)?.as_str() {
                    "f32" => Precision::F32,
                    "f64" => Precision::F64,
                    other => return Err(format!("unknown precision {other}")),
                };
                skip = true;
            }
            "--mode" => {
                mode = match need(value)?.as_str() {
                    "strict" => Mode::Strict,
                    "loose" => Mode::Loose,
                    other => return Err(format!("unknown mode {other}")),
                };
                skip = true;
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }

    match out_dir {
        Some(dir) => Ok(Args {
            scene,
            camera,
            width,
            height,
            precision,
            mode,
            out_dir: dir,
        }),
        None => Err(format!(
            "--out-dir is REQUIRED so this binary cannot write to the repository \
             root by accident\n\n{USAGE}"
        )),
    }
}

fn build_scene(choice: &SceneChoice) -> Scene<f64> {
    match choice {
        SceneChoice::Spheres => spheres::<f64>(),
        SceneChoice::SpheresNoGround => spheres_no_ground::<f64>(),
        SceneChoice::SpheresAnisotropic => spheres_anisotropic::<f64>(),
        SceneChoice::SpheresAnisotropicNoGround => {
            spheres_anisotropic_no_ground::<f64>()
        }
        SceneChoice::Needles => ordering_probe_scene::<f64>(),
        SceneChoice::Stress { atoms } => stress_scene::<f64>(*atoms, 0x51e55, 0.35),
    }
}

fn build_camera(name: &str, width: u32, height: u32) -> CameraSpec {
    match name {
        "narrow" => narrow_camera(width, height),
        "spheres" => sphere_camera(width, height),
        other => lights::guard::unreachable("lights_agreement::build_camera", other),
    }
}

const REPORT_COLUMNS: [&str; 22] = [
    "scene",
    "camera",
    "width",
    "height",
    "precision",
    "mode",
    "atoms",
    "pixels",
    "failing_pixels",
    "worst_pixel_x",
    "worst_pixel_y",
    "worst_difference",
    "worst_tolerance",
    "worst_ratio",
    "rms_difference",
    "tolerance_span",
    "tolerance_min",
    "tolerance_max",
    "clamped_atoms",
    "determinant_culled_atoms",
    "difference_png_scale",
    "tolerance_png_scale",
];

fn number(value: f64) -> String {
    format!("{value}")
}

fn write_report(
    args: &Args,
    atoms: usize,
    report: &lights::agreement::AgreementReport,
    difference_scale: f64,
    tolerance_scale: f64,
) -> Result<()> {
    let header = vec![
        format!(
            "reading\tdifference.png and tolerance.png are each normalised by \
             their own scale, recorded here, and MUST be read together"
        ),
        format!(
            "tolerance\tthe STRICT-mode per-pixel bound, which covers the EWA \
             geometry and nothing else"
        ),
        format!("clean\t{}", report.is_clean()),
    ];
    let row = vec![
        format!("{:?}", args.scene),
        args.camera.to_string(),
        args.width.to_string(),
        args.height.to_string(),
        args.precision.name().to_string(),
        format!("{:?}", args.mode).to_lowercase(),
        atoms.to_string(),
        report.pixels.to_string(),
        report.failures.to_string(),
        report.worst_pixel.0.to_string(),
        report.worst_pixel.1.to_string(),
        number(report.worst_difference),
        number(report.worst_tolerance),
        number(report.worst_ratio),
        number(report.rms_difference),
        number(report.tolerance_span),
        number(report.tolerance_min),
        number(report.tolerance_max),
        report.clamped_atoms.to_string(),
        report.determinant_culled_atoms.to_string(),
        number(difference_scale),
        number(tolerance_scale),
    ];
    let text = invocation::table(&header, &REPORT_COLUMNS, &[row]);
    std::fs::write(args.out_dir.join("report.tsv"), text)?;
    Ok(())
}

fn signed_difference(a: &FloatImage<f64>, b: &FloatImage<f64>) -> FloatImage<f64> {
    FloatImage {
        width: a.width,
        height: a.height,
        pixels: a
            .pixels
            .iter()
            .zip(b.pixels.iter())
            .map(|(l, r)| Rgb::new((l.x - r.x).abs(), (l.y - r.y).abs(), (l.z - r.z).abs()))
            .collect(),
    }
}

fn run(args: &Args) -> Result<bool> {
    std::fs::create_dir_all(&args.out_dir)?;

    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.02, 0.03, 0.05);
    let ordering = match args.mode {
        Mode::Strict => OrderingModel::CentreDepth,
        Mode::Loose => OrderingModel::HitDepth,
    };

    let base = RenderRequest {
        precision: args.precision,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::splat_contract(),
        settings,
        camera: build_camera(args.camera, args.width, args.height),
        scene: build_scene(&args.scene),
        collect: Collect::PixelsAndTolerance,
        shapes: None,
    };

    let (report, splat, ray, tolerance) = if ordering == OrderingModel::HitDepth {
        println!(
            "mode: LOOSE. The rasteriser sorts by centre depth and the ray \
             caster by t*, so the two run in genuinely different orders. The \
             tolerance below is the STRICT-mode one, which covers the EWA \
             geometry and nothing else, so the failing pixels it reports are \
             exactly the ordering error - isolated, not hidden. This mode exists \
             to measure that error separately, never to widen a failing bound.\n"
        );
        let (_, strict_splat, _, strict_tolerance) = run_pair(
            &base,
            &RenderMethod::splat_contract(),
            &RenderMethod::raycast_contract(),
        )?;
        let mut loose = base.clone();
        loose.settings.ordering = ordering;
        loose.method = RenderMethod::raycast_contract();
        let loose_ray = lights::dispatch::render(&loose)?;
        let mut loose_report =
            lights::agreement::compare(&strict_splat.image, &loose_ray.image, &strict_tolerance)?;
        loose_report.clamped_atoms = strict_splat.report.clamped_atoms;
        loose_report.determinant_culled_atoms = strict_splat.report.determinant_culled_atoms;
        (loose_report, strict_splat, loose_ray, strict_tolerance)
    } else {
        run_pair(
            &base,
            &RenderMethod::splat_contract(),
            &RenderMethod::raycast_contract(),
        )?
    };

    write_png(
        &args.out_dir.join("splat.png"),
        &splat.image.to_clamped_u8(),
    )?;
    write_png(&args.out_dir.join("raycast.png"), &ray.image.to_clamped_u8())?;
    let difference = signed_difference(&splat.image, &ray.image);
    let difference_scale =
        write_png_normalized(&args.out_dir.join("difference.png"), &difference)?;
    let tolerance_scale =
        write_png_normalized(&args.out_dir.join("tolerance.png"), &tolerance)?;

    write_report(
        args,
        base.scene.len(),
        &report,
        difference_scale,
        tolerance_scale,
    )?;

    println!("scene        {:?}", args.scene);
    println!("camera       {} {}x{}", args.camera, args.width, args.height);
    println!("precision    {}", args.precision.name());
    println!("atoms        {}", base.scene.len());
    println!();
    println!("pixels                 {}", report.pixels);
    println!("failing pixels         {}", report.failures);
    println!("worst pixel            {:?}", report.worst_pixel);
    println!("  measured difference  {:.12}", report.worst_difference);
    println!("  its tolerance        {:.12}", report.worst_tolerance);
    println!("  ratio                {:.6}", report.worst_ratio);
    println!("rms difference         {:.12}", report.rms_difference);
    println!(
        "tolerance span         {:.3} ({:.3e} to {:.3e})",
        report.tolerance_span, report.tolerance_min, report.tolerance_max
    );
    println!("clamped atoms          {}", report.clamped_atoms);
    println!("determinant-culled     {}", report.determinant_culled_atoms);
    println!();
    println!(
        "difference.png is normalised by {:.6e}, tolerance.png by {:.6e}. Read \
         them TOGETHER: anyone looking at the difference image alone will \
         conclude the renderer is broken at the periphery, which is exactly what \
         the tolerance image exists to refute.",
        difference_scale, tolerance_scale
    );
    println!(
        "report.tsv     {} - its first line is the command that produced it",
        args.out_dir.join("report.tsv").display()
    );
    Ok(report.is_clean())
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => {
            println!("\nAGREEMENT CLEAN");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            eprintln!("\nAGREEMENT FAILED: at least one pixel exceeds its justified tolerance");
            ExitCode::FAILURE
        }
        Err(LightsError::ContractViolation(v)) => {
            eprintln!(
                "\nCONTRACT VIOLATED before any pixel was compared: {v:?}. Each \
                 precondition has a characteristic fake-bug magnitude, so a \
                 harness that silently breaks one reports a plausible defect \
                 that does not exist."
            );
            ExitCode::from(3)
        }
        Err(e) => {
            eprintln!("\nERROR: {e}");
            ExitCode::from(4)
        }
    }
}
