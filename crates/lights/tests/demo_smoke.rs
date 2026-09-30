#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
use lights::agreement::run_pair;
use lights::dispatch::{Collect, Precision, RenderRequest, render};
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
use lights::error::LightsError;
use lights::image::{read_png, write_png};
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
use lights::image::write_png_normalized;
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
use lights::method::MethodId;
use lights::numerics::vector::Rgb;
use lights::scene::demo::{
    DemoImageLine, demo_closing_line, demo_out_dir, sphere_camera, spheres,
};
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
use lights::scene::demo::{spheres_anisotropic_no_ground, spheres_no_ground};
use lights::settings::SharedSettings;

fn base(width: u32, height: u32) -> RenderRequest {
    let mut settings = SharedSettings::<f64>::contract();
    settings.background = Rgb::new(0.05, 0.07, 0.12);
    RenderRequest {
        precision: Precision::F64,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::splat_contract(),
        settings,
        camera: sphere_camera(width, height),
        scene: spheres::<f64>(),
        collect: Collect::Pixels,
        shapes: None,
    }
}

#[test]
fn demo_renders_both_methods_and_png_roundtrips() {
    let width = 160u32;
    let height = 90u32;
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };

    let mut rendered = 0usize;
    for method in RenderMethod::implemented() {
        let mut request = base(width, height);
        request.method = method;
        let outcome = match render(&request) {
            Ok(o) => o,
            Err(e) => panic!("{e:?}"),
        };
        assert_eq!((outcome.image.width, outcome.image.height), (width, height));

        let quantised = outcome.image.to_clamped_u8();
        let path = dir.path().join(format!("{}.png", outcome.method.name()));
        match write_png(&path, &quantised) {
            Ok(()) => (),
            Err(e) => panic!("write: {e:?}"),
        }
        let read_back = match read_png(&path) {
            Ok(i) => i,
            Err(e) => panic!("read: {e:?}"),
        };
        assert_eq!(
            read_back, quantised,
            "the PNG codec round trip must be exact, so a difference between two \
             written images is a rendering difference and never an encoding one"
        );

        let (peak, _, _) = outcome.image.max_abs();
        assert!(
            peak > 0.5,
            "{:?}: the demo scene must actually render: peak {peak}",
            outcome.method
        );
        let (lo, _) = outcome.image.min_max_channel();
        assert!(
            lo >= 0.0,
            "{:?}: an unsigned kernel must never drive a channel negative",
            outcome.method
        );
        rendered += 1;
    }
    assert_eq!(
        rendered,
        usize::from(cfg!(feature = "method-splat"))
            + usize::from(cfg!(feature = "method-raycast"))
    );
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_demo_scene_agrees_across_methods_once_the_ground_atom_is_dropped() {
    let width = 160u32;
    let height = 90u32;
    let mut request = base(width, height);
    request.scene = spheres_anisotropic_no_ground::<f64>();

    let (report, splat, ray, tolerance) = match run_pair(
        &request,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Ok(v) => v,
        Err(e) => panic!("run_pair: {e:?}"),
    };
    assert_eq!(splat.method, MethodId::SplatRaster);
    assert_eq!(ray.method, MethodId::RayCast);
    assert_eq!(report.clamped_atoms, 0);
    assert!(
        report.is_clean(),
        "the FLOAT images are compared, never the 8-bit ones, because a 1e-12 \
         tolerance cannot survive quantisation: {} of {} pixels fail, worst {:?} \
         measured {} tolerance {}",
        report.failures,
        report.pixels,
        report.worst_pixel,
        report.worst_difference,
        report.worst_tolerance
    );

    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => panic!("tempdir: {e}"),
    };
    let difference = match splat.image.difference(&ray.image) {
        Ok(d) => d,
        Err(e) => panic!("{e:?}"),
    };
    match write_png_normalized(&dir.path().join("difference.png"), &difference) {
        Ok(scale) => assert!(scale.is_finite() && scale > 0.0),
        Err(e) => panic!("{e:?}"),
    }
    match write_png_normalized(&dir.path().join("tolerance.png"), &tolerance) {
        Ok(scale) => assert!(scale.is_finite() && scale > 0.0),
        Err(e) => panic!("{e:?}"),
    }
}

#[test]
#[cfg(all(feature = "method-splat", feature = "method-raycast"))]
fn the_ground_atom_is_refused_by_the_contract_rather_than_quietly_excluded() {
    let mut request = base(160, 90);
    request.scene = spheres::<f64>();
    match run_pair(
        &request,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Err(LightsError::ContractViolation(v)) => {
            let text = format!("{v:?}");
            assert!(
                text.contains("ClampedAtomsPresent"),
                "the ground atom's Jacobian clamp has no ray-casting counterpart, \
                 so the pair must be REFUSED with the count reported rather than \
                 the atom being silently dropped: {text}"
            );
        }
        Ok((report, _, _, _)) => panic!(
            "the full sphere scene must not pass the contract: {report:?}"
        ),
        Err(other) => panic!("wrong failure: {other:?}"),
    }

    let mut clean = base(160, 90);
    clean.scene = spheres_no_ground::<f64>();
    match run_pair(
        &clean,
        &RenderMethod::splat_contract(),
        &RenderMethod::raycast_contract(),
    ) {
        Ok((report, _, _, _)) => assert_eq!(report.clamped_atoms, 0),
        Err(e) => panic!("dropping the ground atom must make the pair legal: {e:?}"),
    }
}

#[test]
fn nothing_in_this_path_computes_a_photometric_metric_against_a_path_traced_reference() {
    let sources = [
        include_str!("../src/agreement.rs"),
        include_str!("../src/scene/demo.rs"),
        include_str!("../src/bin/lights_agreement.rs"),
        include_str!("../examples/render_demo.rs"),
    ];
    let forbidden = [
        concat!("ps", "nr"),
        concat!("ss", "im"),
        concat!("m", "se"),
        concat!("mean_squared", "_error"),
    ];
    for text in sources {
        let lowered = text.to_ascii_lowercase();
        for token in forbidden {
            assert!(
                !lowered.contains(token),
                "the reference is a stochastic path tracer whose committed render is \
                 about 2x overbright from a verified double-count bug and whose \
                 entire sky is clipped to pure white. A photometric number \
                 against it would move under sample count and RNG seed alone in \
                 a renderer that is CORRECT. The sound comparison is camera and \
                 projection agreement, which is tests/sphere_geometry.rs"
            );
        }
    }
    assert!(
        sources[3].contains("sphere_camera"),
        "the demo must go through the sphere scene's own camera"
    );
}

#[test]
fn the_examples_one_argument_is_an_output_directory_and_its_absence_is_a_temp_dir() {
    let asked = demo_out_dir(Some("/somewhere/a/reader/chose".to_string()));
    assert_eq!(
        asked,
        std::path::PathBuf::from("/somewhere/a/reader/chose"),
        "the single optional argument IS the output directory and nothing is \
         appended to it; a reader who names a directory gets that directory"
    );

    let defaulted = demo_out_dir(None);
    assert_eq!(
        defaulted,
        std::env::temp_dir().join("lights-render-demo"),
        "with no argument the example must write somewhere disposable and \
         named, never the working directory it happens to be run from"
    );
    assert_ne!(
        defaulted,
        std::env::temp_dir(),
        "the default must be a NAMED sub-directory, or the example scatters \
         PNGs across the system temp directory"
    );
}

#[test]
fn the_examples_summary_reports_milliseconds_from_nanoseconds_and_names_every_counter() {
    let line = DemoImageLine {
        method: "splat".to_string(),
        width: 960,
        height: 540,
        file: "demo-splat.png".to_string(),
        build_nanos: 1_500_000,
        bind_nanos: 2_250_000,
        render_nanos: 123_456_789,
        clamped_atoms: 7,
        atoms_projected: 11,
        kernel_evaluations: 13,
        composited_hits: 17,
        early_outs: 19,
    };
    let lines = line.lines();
    assert_eq!(lines.len(), 3, "{lines:?}");

    assert!(lines[0].starts_with("splat "), "{:?}", lines[0]);
    assert!(lines[0].contains("960x540"), "{:?}", lines[0]);
    assert!(lines[0].ends_with("-> demo-splat.png"), "{:?}", lines[0]);

    assert!(
        lines[1].contains("build 1.500 ms"),
        "the report carries NANOSECONDS and the summary prints milliseconds; a \
         missing conversion reads as a renderer a million times slower: {:?}",
        lines[1]
    );
    assert!(lines[1].contains("bind 2.250 ms"), "{:?}", lines[1]);
    assert!(
        lines[1].contains("render 123.457 ms"),
        "three decimals, rounded and not truncated: {:?}",
        lines[1]
    );
    assert!(lines[1].contains("clamped atoms 7"), "{:?}", lines[1]);

    for (name, value) in [
        ("atoms projected", 11u64),
        ("kernel evaluations", 13),
        ("composited", 17),
        ("early outs", 19),
    ] {
        assert!(
            lines[2].contains(&format!("{name} {value}")),
            "the counter line must carry {name} = {value}: {:?}",
            lines[2]
        );
    }

    let mut swapped = line.clone();
    swapped.atoms_projected = line.kernel_evaluations;
    swapped.kernel_evaluations = line.atoms_projected;
    assert_ne!(
        swapped.lines()[2],
        lines[2],
        "the counters must be printed under their own names, or two of them \
         could be transposed with the line unchanged"
    );
}

#[test]
fn the_examples_closing_line_counts_what_it_actually_wrote() {
    for written in [1usize, 2, 5] {
        let text = demo_closing_line(written);
        assert!(
            text.starts_with(&format!("{written} methods rendered")),
            "the closing line must report the number of images this run wrote, \
             not a constant: {text}"
        );
        assert!(
            text.contains("The only thing that changed between them is the \
                           RenderMethod field"),
            "the sentence is the point of the example and must survive: {text}"
        );
    }
}
