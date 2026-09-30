#![cfg(feature = "fit")]
use lights::error::LightsError;
use lights::fit::atom2d::{Atom2d, camera};
use lights::fit::atoms_io::{
    FIT_COLUMNS, FIT_MAGIC, FIT_VERSION, from_text, read_atoms, to_text, write_atoms,
};
use lights::fit::greedy::render_or_background;
use lights::fit::scale::{Scale, scaled_atoms};
use lights::kernel::KernelFamily;
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

const TILE: u32 = 16;
const VIEWPORT: (u32, u32) = (64, 48);

fn awkward_atoms() -> Vec<Atom2d> {
    vec![
        Atom2d {
            centre: Vec2::new(31.234_567_890_123_456, -0.5),
            log_scale: Vec2::new(2.718_281_828_459_045, -1.414_213_562_373_095_1),
            angle: -3.141_592_653_589_793,
            opacity: 0.999_999_999_999_999_9,
            colour: Rgb::new(0.1, 0.000_000_1, 1.0),
        },
        Atom2d {
            centre: Vec2::new(0.0, 63.999_999_999_999_99),
            log_scale: Vec2::new(-2.0, 5.0),
            angle: 0.000_000_000_000_001,
            opacity: 0.000_1,
            colour: Rgb::new(1.0, 1.0, 1.0),
        },
        Atom2d {
            centre: Vec2::new(-12.5, 12.5),
            log_scale: Vec2::new(0.0, 0.0),
            angle: 1.570_796_326_794_896_6,
            opacity: 0.5,
            colour: Rgb::new(0.0, 0.0, 0.0),
        },
    ]
}

fn bits_of(atoms: &[Atom2d]) -> Vec<[u64; 9]> {
    atoms.iter().map(|a| a.bits()).collect()
}

fn drawn(atoms: &[Atom2d], viewport: (u32, u32)) -> Vec<[u64; 3]> {
    let settings = SharedSettings::<f64>::prefiltered();
    let image = match render_or_background(
        atoms,
        &camera(viewport.0, viewport.1),
        KernelFamily::Gaussian,
        &settings,
        TILE,
    ) {
        Ok(i) => i,
        Err(e) => panic!("the render failed: {e}"),
    };
    image
        .pixels
        .iter()
        .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
        .collect()
}

fn parsed(text: &str) -> (Vec<Atom2d>, (u32, u32)) {
    match from_text(text) {
        Ok(v) => v,
        Err(e) => panic!("the fit file did not parse: {e}"),
    }
}

#[test]
fn the_round_trip_is_bit_exact_and_carries_the_viewport() {
    let atoms = awkward_atoms();
    let text = to_text(&atoms, VIEWPORT);
    let (back, viewport) = parsed(&text);
    assert_eq!(
        bits_of(&atoms),
        bits_of(&back),
        "the round trip is not bit-exact, so a saved fit is an approximation of itself"
    );
    assert_eq!(
        viewport, VIEWPORT,
        "the viewport did not survive the round trip, and without it the centres mean nothing"
    );
    let header = match text.lines().next() {
        Some(l) => l,
        None => panic!("the text has no header"),
    };
    assert_eq!(
        header,
        format!(
            "{FIT_MAGIC}\t{FIT_VERSION}\t{}\t{}\t{}",
            atoms.len(),
            VIEWPORT.0,
            VIEWPORT.1
        ),
        "the header line is not what the format declares"
    );
    assert_eq!(
        text.lines().nth(1),
        Some(FIT_COLUMNS.join("\t").as_str()),
        "the column line is not what the format declares"
    );
    assert_eq!(
        text.lines().count(),
        2 + atoms.len(),
        "the file is not two header lines plus one row per atom"
    );
}

#[test]
fn a_saved_fit_redraws_to_the_same_bits_at_the_base_and_along_the_ladder() {
    let atoms = awkward_atoms();
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => panic!("no temp directory: {e}"),
    };
    let path = dir.path().join("frozen.fit");
    match write_atoms(&path, &atoms, VIEWPORT) {
        Ok(()) => (),
        Err(e) => panic!("the fit did not write: {e}"),
    }
    let (back, viewport) = match read_atoms(&path) {
        Ok(v) => v,
        Err(e) => panic!("the fit did not read: {e}"),
    };
    assert_eq!(viewport, VIEWPORT);
    assert_eq!(
        drawn(&atoms, VIEWPORT),
        drawn(&back, viewport),
        "the saved fit redraws to different bits at the base viewport"
    );

    let mut rungs = 0usize;
    for (n, d) in [(2u32, 1u32), (1, 2), (3, 2), (1, 1)] {
        let scale = match Scale::new(n, d) {
            Ok(s) => s,
            Err(e) => panic!("scale {n}/{d} is not a scale: {e}"),
        };
        let at = (
            ((viewport.0 as u64 * n as u64) / d as u64) as u32,
            ((viewport.1 as u64 * n as u64) / d as u64) as u32,
        );
        assert_eq!(
            drawn(&scaled_atoms(&atoms, scale), at),
            drawn(&scaled_atoms(&back, scale), at),
            "the saved fit redraws to different bits at rung {n}/{d}, which is the whole \
             point of saving Atom2d rather than the three-dimensional projection"
        );
        rungs += 1;
    }
    assert_eq!(
        rungs, 4,
        "the ladder check compared {rungs} rungs; a comparison of none passes trivially"
    );
}

#[test]
fn every_header_field_is_checked_and_each_refusal_names_what_it_saw() {
    let good = to_text(&awkward_atoms(), VIEWPORT);
    let cases: [(String, &str); 7] = [
        (good.replacen(FIT_MAGIC, "lights-scene", 1), "lights-fit"),
        (
            good.replacen(
                &format!("{FIT_MAGIC}\t{FIT_VERSION}"),
                &format!("{FIT_MAGIC}\t99"),
                1,
            ),
            "version",
        ),
        (
            good.replacen("\t3\t64\t48", "\t4\t64\t48", 1),
            "declares 4 atoms",
        ),
        (good.replacen("\t3\t64\t48", "\t3\t0\t48", 1), "viewport"),
        (
            good.replacen("centre.x", "centre_x", 1),
            "column line",
        ),
        (
            good.replacen("\t3\t64\t48", "\t3\t64", 1),
            "fields",
        ),
        (
            good.replace("-3.141592653589793", "not-a-number"),
            "is not a number",
        ),
    ];
    let mut refused = 0usize;
    for (text, expect) in cases.iter() {
        match from_text(text) {
            Ok(_) => panic!("this file was accepted and must not have been: expected {expect}"),
            Err(LightsError::SceneFormat(what)) => {
                assert!(
                    what.contains(expect),
                    "the refusal is {what:?} and does not name {expect:?}"
                );
                refused += 1;
            }
            Err(LightsError::ViewportZero { width, height }) => {
                assert_eq!((width, height), (0, 48));
                refused += 1;
            }
            Err(e) => panic!("wrong refusal for {expect}: {e}"),
        }
    }
    assert_eq!(
        refused,
        cases.len(),
        "only {refused} of {} malformed files were refused",
        cases.len()
    );
    match from_text(&good) {
        Ok((a, v)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(v, VIEWPORT);
        }
        Err(e) => panic!("the well-formed file was refused: {e}"),
    }
}
