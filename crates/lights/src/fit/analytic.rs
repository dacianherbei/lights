use crate::error::Result;
use crate::fit::coverage::{coverage_image_to_colour, exact_coverage};
use crate::fit::scale::Similarity2;
use crate::fit::target::Target;
use crate::guard::Budget;
use crate::image::FloatImage;
use crate::numerics::float::FloatingPoint;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AnalyticArm {
    Exact,
    Sdf,
}

impl AnalyticArm {
    pub const ALL: [AnalyticArm; 2] = [AnalyticArm::Exact, AnalyticArm::Sdf];

    pub fn name(self) -> &'static str {
        match self {
            AnalyticArm::Exact => "analytic-exact",
            AnalyticArm::Sdf => "analytic-sdf",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AnalyticCost {
    pub primitives: usize,
    pub description_bytes: usize,
    pub sdf_evaluations: u64,
    pub wall_nanos: u128,
}

pub fn sdf_coverage<T: FloatingPoint>(target: &Target<T>) -> FloatImage<T> {
    sdf_coverage_through(target, Similarity2::identity(), target.viewport)
}

pub fn sdf_coverage_through<T: FloatingPoint>(
    target: &Target<T>,
    view: Similarity2<T>,
    viewport: (u32, u32),
) -> FloatImage<T> {
    target.set.coverage_through(view, viewport)
}

pub fn render_analytic<T: FloatingPoint>(
    target: &Target<T>,
    arm: AnalyticArm,
    max_depth: u32,
    budget: &mut Budget,
) -> Result<(FloatImage<T>, AnalyticCost)> {
    let started = std::time::Instant::now();
    let (coverage, evaluations) = match arm {
        AnalyticArm::Exact => {
            let (image, residual) = exact_coverage(target, max_depth, budget)?;
            (image, residual.cells_visited)
        }
        AnalyticArm::Sdf => {
            let (width, height) = target.viewport;
            (
                sdf_coverage(target),
                (width as u64) * (height as u64),
            )
        }
    };
    let elapsed = started.elapsed().as_nanos();
    Ok((
        coverage_image_to_colour(target, &coverage),
        AnalyticCost {
            primitives: target.primitive_count(),
            description_bytes: target.description_bytes(),
            sdf_evaluations: evaluations,
            wall_nanos: elapsed,
        },
    ))
}
