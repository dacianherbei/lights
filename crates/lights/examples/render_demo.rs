use lights::dispatch::{Collect, Precision, RenderRequest, render};
use lights::error::Result;
use lights::image::write_png;
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::numerics::vector::Rgb;
use lights::scene::demo::{DemoImageLine, demo_closing_line, demo_out_dir, sphere_camera, spheres};
use lights::settings::SharedSettings;

fn main() -> Result<()> {
    let out_dir = demo_out_dir(std::env::args().nth(1));
    std::fs::create_dir_all(&out_dir)?;

    let width = 960u32;
    let height = 540u32;

    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.05, 0.07, 0.12);

    let request = RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::splat_contract(),
        settings,
        camera: sphere_camera(width, height),
        scene: spheres::<f64>(),
        collect: Collect::PixelsAndCounters,
        shapes: None,
    };

    println!(
        "the six-sphere geometry as 3D gaussians, sigma = R/3, through \
         its own camera, {width}x{height}, {} atoms",
        request.scene.len()
    );
    println!("writing to {}", out_dir.display());
    println!();

    let mut written = 0usize;
    for method in RenderMethod::implemented() {
        let mut one_token_different = request.clone();
        one_token_different.method = method;

        let outcome = render(&one_token_different)?;
        let name = format!("demo-{}.png", outcome.method.name());
        write_png(&out_dir.join(&name), &outcome.image.to_clamped_u8())?;
        written += 1;

        let summary = DemoImageLine {
            method: outcome.method.name().to_string(),
            width: outcome.image.width,
            height: outcome.image.height,
            file: name,
            build_nanos: outcome.report.build.wall_nanos,
            bind_nanos: outcome.report.bind.wall_nanos,
            render_nanos: outcome.report.render.wall_nanos,
            clamped_atoms: outcome.report.clamped_atoms,
            atoms_projected: outcome.counters.atoms_projected,
            kernel_evaluations: outcome.counters.kernel_evaluations,
            composited_hits: outcome.counters.composited_hits,
            early_outs: outcome.counters.early_outs,
        };
        for line in summary.lines() {
            println!("{line}");
        }
    }

    println!();
    println!("{}", demo_closing_line(written));
    Ok(())
}
