use crate::error::{LightsError, Result};
use crate::fit::atom2d::Atom2d;
use crate::numerics::vector::{Rgb, Vec2};
use std::path::Path;

pub const FIT_MAGIC: &str = "lights-fit";
pub const FIT_VERSION: u32 = 1;
pub const FIT_COLUMNS: [&str; 9] = [
    "centre.x",
    "centre.y",
    "log_scale.x",
    "log_scale.y",
    "angle",
    "opacity",
    "colour.r",
    "colour.g",
    "colour.b",
];

fn cell(v: f64) -> String {
    format!("{v:?}")
}

pub fn to_text(atoms: &[Atom2d], viewport: (u32, u32)) -> String {
    let mut out = String::new();
    out.push_str(FIT_MAGIC);
    out.push('\t');
    out.push_str(&FIT_VERSION.to_string());
    out.push('\t');
    out.push_str(&atoms.len().to_string());
    out.push('\t');
    out.push_str(&viewport.0.to_string());
    out.push('\t');
    out.push_str(&viewport.1.to_string());
    out.push('\n');
    out.push_str(&FIT_COLUMNS.join("\t"));
    out.push('\n');
    for a in atoms.iter() {
        let cells = [
            cell(a.centre.x),
            cell(a.centre.y),
            cell(a.log_scale.x),
            cell(a.log_scale.y),
            cell(a.angle),
            cell(a.opacity),
            cell(a.colour.x),
            cell(a.colour.y),
            cell(a.colour.z),
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
            "fit row {row}, column {column}: {text:?} is not a number ({e})"
        ))),
    }
}

fn count(text: &str, what: &str) -> Result<u32> {
    match text.parse::<u32>() {
        Ok(v) => Ok(v),
        Err(e) => Err(LightsError::SceneFormat(format!(
            "the fit header's {what} is {text:?}, which is not a count ({e})"
        ))),
    }
}

pub fn from_text(text: &str) -> Result<(Vec<Atom2d>, (u32, u32))> {
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(l) => l,
        None => return Err(LightsError::SceneFormat("the fit file is empty".to_string())),
    };
    let fields: Vec<&str> = header.split('\t').collect();
    if fields.len() != 5 {
        return Err(LightsError::SceneFormat(format!(
            "the fit header carries {} fields and must carry 5: magic, version, count, width, \
             height",
            fields.len()
        )));
    }
    if fields[0] != FIT_MAGIC {
        return Err(LightsError::SceneFormat(format!(
            "{:?} is not a {FIT_MAGIC} file",
            fields[0]
        )));
    }
    let version = count(fields[1], "version")?;
    if version != FIT_VERSION {
        return Err(LightsError::SceneFormat(format!(
            "this is a version {version} fit file and this build reads version {FIT_VERSION}"
        )));
    }
    let declared = count(fields[2], "atom count")? as usize;
    let viewport = (count(fields[3], "width")?, count(fields[4], "height")?);
    if viewport.0 == 0 || viewport.1 == 0 {
        return Err(LightsError::ViewportZero {
            width: viewport.0,
            height: viewport.1,
        });
    }

    let columns = match lines.next() {
        Some(l) => l,
        None => {
            return Err(LightsError::SceneFormat(
                "the fit file has a header and no column line".to_string(),
            ));
        }
    };
    if columns != FIT_COLUMNS.join("\t") {
        return Err(LightsError::SceneFormat(format!(
            "the fit column line is {columns:?} and this build writes {:?}",
            FIT_COLUMNS.join("\t")
        )));
    }

    let mut atoms: Vec<Atom2d> = Vec::with_capacity(declared);
    for (row, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() != FIT_COLUMNS.len() {
            return Err(LightsError::SceneFormat(format!(
                "fit row {row} carries {} cells and the format has {}",
                cells.len(),
                FIT_COLUMNS.len()
            )));
        }
        let mut v = [0.0f64; 9];
        for i in 0..9 {
            v[i] = scalar(cells[i], FIT_COLUMNS[i], row)?;
        }
        let atom = Atom2d {
            centre: Vec2::new(v[0], v[1]),
            log_scale: Vec2::new(v[2], v[3]),
            angle: v[4],
            opacity: v[5],
            colour: Rgb::new(v[6], v[7], v[8]),
        };
        if !atom.is_finite() {
            return Err(LightsError::SceneFormat(format!(
                "fit row {row} is not finite"
            )));
        }
        atoms.push(atom);
    }
    if atoms.len() != declared {
        return Err(LightsError::SceneFormat(format!(
            "the fit header declares {declared} atoms and the file carries {}",
            atoms.len()
        )));
    }
    Ok((atoms, viewport))
}

pub fn write_atoms(path: &Path, atoms: &[Atom2d], viewport: (u32, u32)) -> Result<()> {
    match path.parent() {
        Some(dir) => std::fs::create_dir_all(dir)?,
        None => (),
    }
    std::fs::write(path, to_text(atoms, viewport))?;
    Ok(())
}

pub fn read_atoms(path: &Path) -> Result<(Vec<Atom2d>, (u32, u32))> {
    let text = std::fs::read_to_string(path)?;
    from_text(&text)
}
