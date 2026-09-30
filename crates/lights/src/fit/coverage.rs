use crate::error::Result;
use rayon::prelude::*;
use crate::fit::target::{Side, Target};
use crate::guard::Budget;
use crate::image::FloatImage;
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::{Rgb, Vec2};

pub const DEFAULT_MAX_DEPTH: u32 = 10;
pub const DEFAULT_CELL_LIMIT: u64 = 64_000_000;
pub const CELL_DIAGONAL: f64 = 1.4142135623730951;
pub const SEAM_PROBE: f64 = 1.0e-7;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoverageError<T> {
    pub undecided_area: T,
    pub max_depth_reached: u32,
    pub cells_visited: u64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Cell<T> {
    centre: Vec2<T>,
    half: T,
    depth: u32,
}

pub fn coverage_budget(limit: u64) -> Budget {
    Budget::new("fit::coverage::exact_coverage", limit)
}

pub fn box_line_fraction<T: FloatingPoint>(normal: Vec2<T>, signed: T, side: T) -> T {
    let half = side * T::HALF;
    let extent_x = normal.x.abs() * half;
    let extent_y = normal.y.abs() * half;
    let major = extent_x.max(extent_y);
    let minor = extent_x.min(extent_y);
    let cut = -signed;
    if !(major > T::ZERO) {
        return if signed <= T::ZERO { T::ONE } else { T::ZERO };
    }
    if cut <= -(major + minor) {
        return T::ZERO;
    }
    if cut >= major + minor {
        return T::ONE;
    }
    let eight = T::from_f64(8.0);
    if cut <= minor - major {
        let edge = cut + major + minor;
        return edge * edge / (eight * major * minor);
    }
    if cut >= major - minor {
        let edge = major + minor - cut;
        return T::ONE - edge * edge / (eight * major * minor);
    }
    (cut + major) / (T::TWO * major)
}

fn closure_fraction<T: FloatingPoint>(target: &Target<T>, centre: Vec2<T>, half: T) -> T {
    let (signed, normal) = target.nearest_surface(centre);
    let foot = centre - normal * signed;
    let step = normal * T::from_f64(SEAM_PROBE);
    if target.classify(foot + step) == target.classify(foot - step) {
        return match target.classify(centre) {
            Side::Inside => T::ONE,
            Side::Outside => T::ZERO,
        };
    }
    box_line_fraction(normal, signed, half * T::TWO)
}

pub fn exact_coverage<T: FloatingPoint>(
    target: &Target<T>,
    max_depth: u32,
    budget: &mut Budget,
) -> Result<(FloatImage<T>, CoverageError<T>)> {
    let (width, height) = target.viewport;
    let mut image = FloatImage::<T>::new(width, height);
    let diagonal = T::from_f64(CELL_DIAGONAL);

    let (site, limit) = (budget.site(), budget.limit());
    let per_pixel: Vec<(T, T, u32, u64)> = (0..(width as u64 * height as u64))
        .into_par_iter()
        .map(|idx| -> Result<(T, T, u32, u64)> {
            let px = (idx % width as u64) as u32;
            let py = (idx / width as u64) as u32;
            let mut inside_area = T::ZERO;
            let mut undecided = T::ZERO;
            let mut deepest = 0u32;
            let mut stack: Vec<Cell<T>> = Vec::new();
            stack.push(Cell {
                centre: Vec2::new(
                    T::from_f64(px as f64) + T::HALF,
                    T::from_f64(py as f64) + T::HALF,
                ),
                half: T::HALF,
                depth: 0,
            });
            let mut cells_budget = Budget::new(site, limit);
            while let Some(cell) = stack.pop() {
                cells_budget.tick()?;
                if cell.depth > deepest {
                    deepest = cell.depth;
                }
                let side = cell.half * T::TWO;
                let area = side * side;
                if target.clearance(cell.centre) > cell.half * diagonal {
                    if target.classify(cell.centre) == Side::Inside {
                        inside_area += area;
                    }
                    continue;
                }
                if cell.depth >= max_depth {
                    undecided += area;
                    inside_area += area * closure_fraction(target, cell.centre, cell.half);
                    continue;
                }
                let quarter = cell.half * T::HALF;
                for step in [
                    Vec2::new(-quarter, -quarter),
                    Vec2::new(quarter, -quarter),
                    Vec2::new(-quarter, quarter),
                    Vec2::new(quarter, quarter),
                ] {
                    stack.push(Cell {
                        centre: cell.centre + step,
                        half: quarter,
                        depth: cell.depth + 1,
                    });
                }
            }
            Ok((inside_area, undecided, deepest, cells_budget.used()))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut undecided_area = T::ZERO;
    let mut max_depth_reached = 0u32;
    let mut cells_visited = 0u64;
    for (idx, (inside, undecided, deepest, cells)) in per_pixel.into_iter().enumerate() {
        let px = (idx as u64 % width as u64) as u32;
        let py = (idx as u64 / width as u64) as u32;
        image.set(px, py, Rgb::splat(inside));
        undecided_area += undecided;
        if deepest > max_depth_reached {
            max_depth_reached = deepest;
        }
        cells_visited += cells;
    }
    for _ in 0..cells_visited {
        budget.tick()?;
    }

    Ok((
        image,
        CoverageError {
            undecided_area,
            max_depth_reached,
            cells_visited,
        },
    ))
}

pub fn coverage_image_to_colour<T: FloatingPoint>(
    target: &Target<T>,
    coverage: &FloatImage<T>,
) -> FloatImage<T> {
    FloatImage {
        width: coverage.width,
        height: coverage.height,
        pixels: coverage
            .pixels
            .iter()
            .map(|p| {
                let c = p.x.clamp(T::ZERO, T::ONE);
                target.paper * (T::ONE - c) + target.set.ink * c
            })
            .collect(),
    }
}
