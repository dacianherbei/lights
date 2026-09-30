use crate::error::{LightsError, Result};
use crate::numerics::vector::Vec2;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileRect {
    pub x0: i32,
    pub y0: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileSpan {
    pub tx0: i32,
    pub ty0: i32,
    pub tx1: i32,
    pub ty1: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileGrid {
    pub tile: u32,
    pub grid_x: i32,
    pub grid_y: i32,
    pub width: u32,
    pub height: u32,
}

pub const DEFAULT_TILE: u32 = 16;

pub fn floor_div(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) { q - 1 } else { q }
}

impl TileGrid {
    pub fn new(width: u32, height: u32, tile: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(LightsError::ViewportZero { width, height });
        }
        if tile == 0 {
            return Err(LightsError::InvalidCamera("tile size must be non-zero"));
        }
        let t = tile as i32;
        Ok(Self {
            tile,
            grid_x: (width as i32 + t - 1) / t,
            grid_y: (height as i32 + t - 1) / t,
            width,
            height,
        })
    }

    pub fn tile_count(&self) -> usize {
        (self.grid_x as usize) * (self.grid_y as usize)
    }

    pub fn rect(&self, tx: i32, ty: i32) -> TileRect {
        let t = self.tile as i32;
        let x0 = tx * t;
        let y0 = ty * t;
        TileRect {
            x0,
            y0,
            w: t.min(self.width as i32 - x0).max(0),
            h: t.min(self.height as i32 - y0).max(0),
        }
    }

    pub fn tile_id(&self, tx: i32, ty: i32) -> u32 {
        (ty * self.grid_x + tx) as u32
    }

    pub fn span_of_box(&self, min: Vec2<f64>, max: Vec2<f64>) -> Option<TileSpan> {
        if !(min.x.is_finite() && min.y.is_finite() && max.x.is_finite() && max.y.is_finite()) {
            return None;
        }
        let t = self.tile as i32;
        let x0 = min.x.floor();
        let y0 = min.y.floor();
        let x1 = max.x.ceil();
        let y1 = max.y.ceil();
        if !(x1 > x0) || !(y1 > y0) {
            return None;
        }
        let px0 = clamp_to_i32(x0);
        let py0 = clamp_to_i32(y0);
        let px1 = clamp_to_i32(x1);
        let py1 = clamp_to_i32(y1);

        let tx0 = floor_div(px0, t).clamp(0, self.grid_x);
        let ty0 = floor_div(py0, t).clamp(0, self.grid_y);
        let tx1 = (floor_div(px1 - 1, t) + 1).clamp(0, self.grid_x);
        let ty1 = (floor_div(py1 - 1, t) + 1).clamp(0, self.grid_y);

        if tx1 <= tx0 || ty1 <= ty0 {
            return None;
        }
        Some(TileSpan {
            tx0,
            ty0,
            tx1,
            ty1,
        })
    }

    pub fn tiles(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        (0..self.grid_y).flat_map(move |ty| (0..self.grid_x).map(move |tx| (tx, ty)))
    }
}

fn clamp_to_i32(v: f64) -> i32 {
    if v > i32::MAX as f64 {
        i32::MAX
    } else if v < i32::MIN as f64 {
        i32::MIN
    } else {
        v as i32
    }
}

pub fn for_each_tile<F: Fn(TileRect) + Sync + Send>(grid: &TileGrid, threads: usize, f: F) {
    let rects: Vec<TileRect> = grid.tiles().map(|(tx, ty)| grid.rect(tx, ty)).collect();
    if threads == 1 {
        for r in rects {
            f(r);
        }
    } else {
        rects.into_par_iter().for_each(f);
    }
}
