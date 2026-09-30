use crate::draw::{Clip, DrawList, DrawOp, DrawShape, Ink};
use crate::error::{LightsError, Result};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::{Rgb, Vec2};
use std::collections::BTreeMap;

pub const CAPTURE_COLUMNS: [&str; 18] = [
    "shape", "kind", "clip_x0", "clip_y0", "clip_x1", "clip_y1", "x0", "y0", "x1", "y1", "radius", "width",
    "blur", "align", "closed", "rgba", "textured", "points",
];

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stand {
    Native,
    Proxy,
    Absent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapturedOp<T> {
    pub shape: usize,
    pub kind: String,
    pub stand: Stand,
    pub op: Option<DrawOp<T>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Capture<T> {
    pub header: BTreeMap<String, String>,
    pub ops: Vec<CapturedOp<T>>,
}

struct Row<'a> {
    line: usize,
    cells: Vec<&'a str>,
}

impl<'a> Row<'a> {
    fn fail(&self, reason: String) -> LightsError {
        LightsError::CaptureFormat {
            line: self.line,
            reason,
        }
    }

    fn cell(&self, name: &str) -> Result<&'a str> {
        match CAPTURE_COLUMNS.iter().position(|c| *c == name) {
            Some(i) => Ok(self.cells[i]),
            None => Err(self.fail(format!("no column {name}"))),
        }
    }

    fn number<T: FloatingPoint>(&self, name: &str) -> Result<T> {
        let text = self.cell(name)?;
        match text.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(T::from_f64(v)),
            Ok(_) | Err(_) => Err(self.fail(format!("{name} is {text:?}, not a finite number"))),
        }
    }

    fn list<T: FloatingPoint>(&self, name: &str) -> Result<Vec<T>> {
        let text = self.cell(name)?;
        text.split(',')
            .map(|v| match v.parse::<f64>() {
                Ok(n) if n.is_finite() => Ok(T::from_f64(n)),
                Ok(_) | Err(_) => Err(self.fail(format!("{name} is {text:?}, not a list of numbers"))),
            })
            .collect()
    }

    fn corner<T: FloatingPoint>(&self, x: &str, y: &str) -> Result<Vec2<T>> {
        Ok(Vec2::new(self.number(x)?, self.number(y)?))
    }

    fn points<T: FloatingPoint>(&self) -> Result<Vec<Vec2<T>>> {
        let text = self.cell("points")?;
        text.split(';')
            .map(|pair| match pair.split_once(',') {
                Some((x, y)) => match (x.parse::<f64>(), y.parse::<f64>()) {
                    (Ok(x), Ok(y)) if x.is_finite() && y.is_finite() => Ok(Vec2::new(T::from_f64(x), T::from_f64(y))),
                    (Ok(_) | Err(_), Ok(_) | Err(_)) => Err(self.fail(format!("point {pair:?} is not two finite numbers"))),
                },
                None => Err(self.fail(format!("point {pair:?} is not x,y"))),
            })
            .collect()
    }

    fn ink<T: FloatingPoint>(&self) -> Result<Ink<T>> {
        let text = self.cell("rgba")?;
        if text == "uv" {
            return Ok(Ink::opaque(Rgb::splat(T::HALF)));
        }
        let channels: Vec<T> = self.list::<T>("rgba")?;
        let full = T::from_f64(255.0);
        match channels.as_slice() {
            [r, g, b, a] => Ok(Ink::premultiplied(Rgb::new(*r / full, *g / full, *b / full), *a / full)),
            _ => Err(self.fail(format!("rgba is {text:?}, not four channels"))),
        }
    }

    fn largest_radius<T: FloatingPoint>(&self) -> Result<T> {
        Ok(self.list::<T>("radius")?.into_iter().fold(T::ZERO, |m, r| m.max(r)))
    }

    fn half_width<T: FloatingPoint>(&self) -> Result<T> {
        Ok(self.number::<T>("width")? * T::HALF)
    }
}

fn outline<T: FloatingPoint>(row: &Row, min: Vec2<T>, max: Vec2<T>, radius: T) -> Result<DrawShape<T>> {
    let half_width = row.half_width::<T>()?;
    let shift = match row.cell("align")? {
        "inside" => -half_width,
        "outside" => half_width,
        "middle" => T::ZERO,
        other => return Err(row.fail(format!("align is {other:?}, not inside, middle or outside"))),
    };
    let centre = (min + max) * T::HALF;
    let lo = min - Vec2::splat(shift);
    let hi = max + Vec2::splat(shift);
    Ok(DrawShape::RectOutline {
        min: Vec2::new(lo.x.min(centre.x), lo.y.min(centre.y)),
        max: Vec2::new(hi.x.max(centre.x), hi.y.max(centre.y)),
        corner_radius: (radius + shift).max(T::ZERO),
        half_width,
    })
}

fn convert<T: FloatingPoint>(row: &Row) -> Result<(Stand, Option<DrawShape<T>>)> {
    let kind = row.cell("kind")?;
    let native = |shape: DrawShape<T>| Ok((Stand::Native, Some(shape)));
    let proxy = |shape: DrawShape<T>| Ok((Stand::Proxy, Some(shape)));
    match kind {
        "rect_fill" => {
            let grow = row.number::<T>("blur")? * T::HALF;
            let shape = DrawShape::Rect {
                min: row.corner::<T>("x0", "y0")? - Vec2::splat(grow),
                max: row.corner::<T>("x1", "y1")? + Vec2::splat(grow),
                corner_radius: row.largest_radius()?,
            };
            if grow > T::ZERO || row.cell("textured")? == "1" { proxy(shape) } else { native(shape) }
        }
        "rect_stroke" => native(outline(
            row,
            row.corner("x0", "y0")?,
            row.corner("x1", "y1")?,
            row.largest_radius()?,
        )?),
        "circle_fill" => native(DrawShape::Disc {
            centre: row.corner("x0", "y0")?,
            radius: row.number("radius")?,
        }),
        "circle_stroke" => native(DrawShape::Ring {
            centre: row.corner("x0", "y0")?,
            radius: row.number("radius")?,
            half_width: row.half_width()?,
        }),
        "line" => native(DrawShape::Segment {
            a: row.corner("x0", "y0")?,
            b: row.corner("x1", "y1")?,
            half_width: row.half_width()?,
        }),
        "path_fill" => native(DrawShape::Polygon { points: row.points()? }),
        "path_stroke" => native(DrawShape::Polyline {
            points: row.points()?,
            closed: row.cell("closed")? == "1",
            half_width: row.half_width()?,
        }),
        "glyph" => proxy(DrawShape::Rect {
            min: row.corner("x0", "y0")?,
            max: row.corner("x1", "y1")?,
            corner_radius: T::ZERO,
        }),
        "text_other" | "mesh_triangle" => proxy(DrawShape::Polygon { points: row.points()? }),
        "ellipse_fill" | "ellipse_stroke" | "bezier" | "callback" => Ok((Stand::Absent, None)),
        other => Err(row.fail(format!("unknown kind {other:?}"))),
    }
}

impl<T: FloatingPoint> Capture<T> {
    pub fn parse(text: &str) -> Result<Self> {
        let mut header = BTreeMap::new();
        let mut columns_seen = false;
        let mut ops = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line_number = index + 1;
            if let Some(rest) = line.strip_prefix('#') {
                let (key, value) = rest.split_once('\t').unwrap_or((rest, ""));
                header.entry(key.to_string()).or_insert_with(|| value.to_string());
                continue;
            }
            if !columns_seen {
                if line != CAPTURE_COLUMNS.join("\t") {
                    return Err(LightsError::CaptureFormat {
                        line: line_number,
                        reason: format!("the column header is {line:?}, not the {} capture columns", CAPTURE_COLUMNS.len()),
                    });
                }
                columns_seen = true;
                continue;
            }
            let row = Row {
                line: line_number,
                cells: line.split('\t').collect(),
            };
            if row.cells.len() != CAPTURE_COLUMNS.len() {
                return Err(row.fail(format!("{} cells under {} columns", row.cells.len(), CAPTURE_COLUMNS.len())));
            }
            let shape = row
                .cell("shape")?
                .parse::<usize>()
                .map_err(|_| row.fail("shape is not an index".to_string()))?;
            let (stand, drawn) = convert::<T>(&row)?;
            let op = match drawn {
                Some(shape) => {
                    let clip = Clip::new(row.corner("clip_x0", "clip_y0")?, row.corner("clip_x1", "clip_y1")?);
                    Some(DrawOp::new(shape, row.ink()?).clipped(clip))
                }
                None => None,
            };
            ops.push(CapturedOp {
                shape,
                kind: row.cell("kind")?.to_string(),
                stand,
                op,
            });
        }
        if !columns_seen {
            return Err(LightsError::CaptureFormat {
                line: text.lines().count(),
                reason: "no column header".to_string(),
            });
        }
        Ok(Self { header, ops })
    }

    pub fn draw_list(&self, admit: &[Stand]) -> (DrawList<T>, Vec<String>) {
        let mut list = DrawList::new();
        let mut refused = Vec::new();
        for captured in self.ops.iter().filter(|c| admit.contains(&c.stand)) {
            if let Some(op) = &captured.op {
                if let Err(e) = list.push(op.clone()) {
                    refused.push(format!("shape {} {}: {e}", captured.shape, captured.kind));
                }
            }
        }
        (list, refused)
    }
}
