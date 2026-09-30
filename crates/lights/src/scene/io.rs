use crate::error::{LightsError, Result};
use crate::numerics::float::FloatingPoint;
use crate::numerics::quaternion::Quat;
use crate::numerics::vector::{Rgb, Vec3};
use crate::scene::{Atom, Scene};
use std::path::Path;

pub const SCENE_MAGIC: &str = "lights-scene";
pub const SCENE_VERSION: u32 = 1;
pub const SCENE_COLUMNS: [&str; 14] = [
    "mean_x",
    "mean_y",
    "mean_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
    "scale_x",
    "scale_y",
    "scale_z",
    "colour_r",
    "colour_g",
    "colour_b",
    "opacity",
];

fn cell(v: f64) -> String {
    format!("{v:?}")
}

pub fn to_text<T: FloatingPoint>(scene: &Scene<T>) -> String {
    let wide = scene.to_f64();
    let mut out = String::new();
    out.push_str(SCENE_MAGIC);
    out.push('\t');
    out.push_str(&SCENE_VERSION.to_string());
    out.push('\t');
    out.push_str(&wide.len().to_string());
    out.push('\n');
    out.push_str(&SCENE_COLUMNS.join("\t"));
    out.push('\n');
    for a in wide.atoms().iter() {
        let cells = [
            cell(a.mean.x),
            cell(a.mean.y),
            cell(a.mean.z),
            cell(a.rotation.w),
            cell(a.rotation.x),
            cell(a.rotation.y),
            cell(a.rotation.z),
            cell(a.scale.x),
            cell(a.scale.y),
            cell(a.scale.z),
            cell(a.colour.x),
            cell(a.colour.y),
            cell(a.colour.z),
            cell(a.opacity),
        ];
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    out
}

fn scalar(text: &str, column: &str, row: usize) -> Result<f64> {
    match text.parse::<f64>() {
        Ok(v) => Ok(v),
        Err(e) => Err(LightsError::SceneFormat(format!(
            "row {row}, column {column}: {text:?} is not a number ({e})"
        ))),
    }
}

pub fn from_text(text: &str, scale_floor: f64) -> Result<Scene<f64>> {
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(l) => l,
        None => return Err(LightsError::SceneFormat("the file is empty".to_string())),
    };
    let parts: Vec<&str> = header.split('\t').collect();
    if parts.len() != 3 || parts[0] != SCENE_MAGIC {
        return Err(LightsError::SceneFormat(format!(
            "the first line must be {SCENE_MAGIC}<tab>version<tab>count, got {header:?}"
        )));
    }
    let version = match parts[1].parse::<u32>() {
        Ok(v) => v,
        Err(e) => {
            return Err(LightsError::SceneFormat(format!(
                "version {:?} is not a number ({e})",
                parts[1]
            )));
        }
    };
    if version != SCENE_VERSION {
        return Err(LightsError::SceneFormat(format!(
            "version {version} was written by a different format; this build reads {SCENE_VERSION}"
        )));
    }
    let declared = match parts[2].parse::<usize>() {
        Ok(v) => v,
        Err(e) => {
            return Err(LightsError::SceneFormat(format!(
                "atom count {:?} is not a number ({e})",
                parts[2]
            )));
        }
    };
    let columns = match lines.next() {
        Some(l) => l,
        None => {
            return Err(LightsError::SceneFormat(
                "the column line is missing".to_string(),
            ));
        }
    };
    if columns != SCENE_COLUMNS.join("\t") {
        return Err(LightsError::SceneFormat(format!(
            "the column line is {columns:?}; this build writes {:?}, and a file whose \
             columns were reordered would be read silently wrong",
            SCENE_COLUMNS.join("\t")
        )));
    }

    let mut scene = Scene::<f64>::new();
    for (row, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() != SCENE_COLUMNS.len() {
            return Err(LightsError::SceneFormat(format!(
                "row {row} has {} fields, not {}",
                cells.len(),
                SCENE_COLUMNS.len()
            )));
        }
        let mut v = [0.0f64; 14];
        for (i, c) in cells.iter().enumerate() {
            v[i] = scalar(c, SCENE_COLUMNS[i], row)?;
        }
        let atom = Atom::new(
            Vec3::new(v[0], v[1], v[2]),
            Quat::new_wxyz(v[3], v[4], v[5], v[6]),
            Vec3::new(v[7], v[8], v[9]),
            v[13],
            Rgb::new(v[10], v[11], v[12]),
        );
        scene.push(atom, scale_floor)?;
    }
    if scene.len() != declared {
        return Err(LightsError::SceneFormat(format!(
            "the header declares {declared} atoms and the file carries {}; a file cut \
             short would otherwise read as a smaller scene rather than as an error",
            scene.len()
        )));
    }
    Ok(scene)
}

pub fn write_scene<T: FloatingPoint>(path: &Path, scene: &Scene<T>) -> Result<()> {
    std::fs::write(path, to_text(scene))?;
    Ok(())
}

pub fn read_scene(path: &Path, scale_floor: f64) -> Result<Scene<f64>> {
    let text = std::fs::read_to_string(path)?;
    from_text(&text, scale_floor)
}
