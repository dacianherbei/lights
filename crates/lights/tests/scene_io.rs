use lights::error::LightsError;
use lights::numerics::float::FloatingPoint;
use lights::numerics::quaternion::Quat;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::io::{SCENE_COLUMNS, SCENE_MAGIC, SCENE_VERSION, from_text, to_text};
use lights::scene::{Atom, Scene};

const FLOOR: f64 = 1.0e-6;

fn awkward_scene() -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    let rotations = [
        Quat::from_axis_angle(Vec3::new(0.3, -0.7, 0.2), 1.234_567_890_123_456_7),
        Quat::from_axis_angle(Vec3::new(-1.0, 0.0, 0.5), -2.718_281_828_459_045),
        Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.000_001),
    ];
    for (i, r) in rotations.iter().enumerate() {
        let q = match r {
            Some(q) => *q,
            None => panic!("axis {i} did not normalise"),
        };
        let f = (i as f64) + 1.0;
        let atom = Atom::new(
            Vec3::new(
                core::f64::consts::PI * f,
                -core::f64::consts::E / f,
                1.0 / 3.0 * f,
            ),
            q,
            Vec3::new(0.123_456_789_012_345_67 * f, 2.5 / f, 0.000_012_5 * f),
            0.9 / f,
            Rgb::new(1.0 / 7.0, 2.0 / 7.0 * f, 0.5),
        );
        match scene.push(atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("atom {i} was refused: {e}"),
        }
    }
    scene
}

fn bits_of(scene: &Scene<f64>) -> Vec<u64> {
    let mut out = Vec::new();
    for a in scene.atoms().iter() {
        for v in [
            a.mean.x,
            a.mean.y,
            a.mean.z,
            a.rotation.w,
            a.rotation.x,
            a.rotation.y,
            a.rotation.z,
            a.scale.x,
            a.scale.y,
            a.scale.z,
            a.colour.x,
            a.colour.y,
            a.colour.z,
            a.opacity,
        ] {
            out.push(v.to_bits_ordered());
        }
    }
    out
}

#[test]
fn a_scene_round_trips_bit_for_bit_and_not_merely_closely() {
    let scene = awkward_scene();
    assert_eq!(scene.len(), 3);
    let text = to_text(&scene);
    let back = match from_text(&text, FLOOR) {
        Ok(s) => s,
        Err(e) => panic!("the scene did not read back: {e}"),
    };
    assert_eq!(back.len(), scene.len());
    let before = bits_of(&scene);
    let after = bits_of(&back);
    assert_eq!(before.len(), 42, "three atoms carry fourteen scalars each");
    assert_eq!(
        before, after,
        "the round trip is not bit-exact. Rust's {{:?}} for f64 emits the \
         shortest decimal that parses back to the same bits, so a difference \
         here means a cell stopped going through it"
    );
    assert!(
        text.starts_with(&format!("{SCENE_MAGIC}\t{SCENE_VERSION}\t3\n")),
        "the header must carry the magic, the version and the atom count: {:?}",
        text.lines().next()
    );
}

#[test]
fn the_narrow_precision_survives_the_same_round_trip() {
    let wide = awkward_scene();
    let narrow = Scene::<f32>::from_f64(&wide);
    let text = to_text(&narrow);
    let back = match from_text(&text, FLOOR) {
        Ok(s) => s,
        Err(e) => panic!("{e}"),
    };
    let again = Scene::<f32>::from_f64(&back);
    assert_eq!(
        narrow.atoms().len(),
        again.atoms().len(),
        "the f32 scene lost an atom in the round trip"
    );
    for (i, (a, b)) in narrow.atoms().iter().zip(again.atoms().iter()).enumerate() {
        assert_eq!(
            a.mean.x.to_bits_ordered(),
            b.mean.x.to_bits_ordered(),
            "atom {i} moved at f32"
        );
        assert_eq!(a.scale.z.to_bits_ordered(), b.scale.z.to_bits_ordered());
        assert_eq!(a.rotation.w.to_bits_ordered(), b.rotation.w.to_bits_ordered());
        assert_eq!(a.opacity.to_bits_ordered(), b.opacity.to_bits_ordered());
    }
}

#[test]
fn a_truncated_file_is_refused_rather_than_read_as_a_smaller_scene() {
    let text = to_text(&awkward_scene());
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5, "header, columns and three rows");
    let cut = format!("{}\n{}\n{}\n", lines[0], lines[1], lines[2]);
    match from_text(&cut, FLOOR) {
        Ok(s) => panic!(
            "a file cut to one row read as a {}-atom scene instead of failing",
            s.len()
        ),
        Err(LightsError::SceneFormat(what)) => assert!(
            what.contains("declares 3") && what.contains("carries 1"),
            "the refusal must name both counts, got {what:?}"
        ),
        Err(e) => panic!("wrong refusal: {e}"),
    }
}

#[test]
fn a_reordered_column_line_is_refused_rather_than_read_wrong() {
    let text = to_text(&awkward_scene());
    let mut columns: Vec<&str> = SCENE_COLUMNS.to_vec();
    columns.swap(0, 1);
    let swapped = text.replace(&SCENE_COLUMNS.join("\t"), &columns.join("\t"));
    assert_ne!(swapped, text, "the column line was not found to swap");
    match from_text(&swapped, FLOOR) {
        Ok(_) => panic!(
            "a file whose first two columns are swapped was read as if nothing \
             had happened, which is the worst failure a geometry file can have"
        ),
        Err(LightsError::SceneFormat(what)) => {
            assert!(what.contains("column line"), "{what}")
        }
        Err(e) => panic!("wrong refusal: {e}"),
    }
}

#[test]
fn a_hand_written_file_goes_through_the_same_ingest_as_push() {
    let body = format!(
        "{SCENE_MAGIC}\t{SCENE_VERSION}\t1\n{}\n{}\n",
        SCENE_COLUMNS.join("\t"),
        [
            "0.0", "0.0", "0.0", "2.0", "0.0", "0.0", "0.0", "1e-12", "1.0", "1.0", "0.5", "0.5",
            "0.5", "0.5",
        ]
        .join("\t")
    );
    let scene = match from_text(&body, FLOOR) {
        Ok(s) => s,
        Err(e) => panic!("the hand-written file was refused: {e}"),
    };
    let atom = scene.atoms()[0];
    assert!(
        (atom.rotation.w - 1.0).abs() < 1.0e-15,
        "the unnormalised quaternion w=2 was stored as {}: reading must go \
         through the one ingest that normalises, not around it",
        atom.rotation.w
    );
    assert_eq!(
        atom.scale.x, FLOOR,
        "the sub-floor scale 1e-12 was stored as {}: reading must clamp through \
         the same path push does",
        atom.scale.x
    );
    let written = to_text(&scene);
    let again = match from_text(&written, FLOOR) {
        Ok(s) => s,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(
        bits_of(&scene),
        bits_of(&again),
        "an ingested scene must be a fixed point of write-then-read"
    );
}

#[test]
fn a_file_from_another_version_is_refused() {
    let text = to_text(&awkward_scene());
    let other = text.replacen(
        &format!("{SCENE_MAGIC}\t{SCENE_VERSION}\t"),
        &format!("{SCENE_MAGIC}\t{}\t", SCENE_VERSION + 1),
        1,
    );
    match from_text(&other, FLOOR) {
        Ok(_) => panic!("a file claiming a future version was read anyway"),
        Err(LightsError::SceneFormat(what)) => assert!(what.contains("version"), "{what}"),
        Err(e) => panic!("wrong refusal: {e}"),
    }
}

#[test]
fn the_file_path_api_writes_and_reads_the_same_scene() {
    let scene = awkward_scene();
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => panic!("no temp directory: {e}"),
    };
    let path = dir.path().join("fitted.scene");
    match lights::scene::io::write_scene(&path, &scene) {
        Ok(()) => (),
        Err(e) => panic!("the scene did not write: {e}"),
    }
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(
        text,
        to_text(&scene),
        "the file on disk differs from the text the formatter produces"
    );
    let back = match lights::scene::io::read_scene(&path, FLOOR) {
        Ok(s) => s,
        Err(e) => panic!("the scene did not read: {e}"),
    };
    assert_eq!(
        bits_of(&scene),
        bits_of(&back),
        "the file round trip is not bit-exact"
    );
}
