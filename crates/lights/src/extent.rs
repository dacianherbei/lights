use crate::camera::{Camera, Projection};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::{Sym2, Sym3};
use crate::numerics::vector::{Vec2, Vec3};
use crate::tiling::TileRect;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScreenExtent {
    pub centre: Vec2<f64>,
    pub form: Sym2<f64>,
    pub reach: f64,
    pub half_extent: Vec2<f64>,
}

pub fn min_quadratic_over_rect<T: FloatingPoint>(
    q: Sym2<T>,
    centre: Vec2<T>,
    lo: Vec2<T>,
    hi: Vec2<T>,
) -> T {
    let corner = Vec2::new(centre.x.clamp(lo.x, hi.x), centre.y.clamp(lo.y, hi.y));
    let mut best = quadratic_at(q, corner - centre);
    if best <= T::ZERO {
        return T::ZERO;
    }
    for edge_x in [lo.x, hi.x] {
        let dx = edge_x - centre.x;
        let want = centre.y - q.xy * dx / q.yy;
        let dy = want.clamp(lo.y, hi.y) - centre.y;
        best = best.min(quadratic_at(q, Vec2::new(dx, dy)));
    }
    for edge_y in [lo.y, hi.y] {
        let dy = edge_y - centre.y;
        let want = centre.x - q.xy * dy / q.xx;
        let dx = want.clamp(lo.x, hi.x) - centre.x;
        best = best.min(quadratic_at(q, Vec2::new(dx, dy)));
    }
    best
}

#[inline(always)]
fn quadratic_at<T: FloatingPoint>(q: Sym2<T>, d: Vec2<T>) -> T {
    q.xx * d.x * d.x + (q.xy + q.xy) * d.x * d.y + q.yy * d.y * d.y
}

impl ScreenExtent {
    pub fn min_corner(&self) -> Vec2<f64> {
        Vec2::new(
            self.centre.x - self.half_extent.x,
            self.centre.y - self.half_extent.y,
        )
    }

    pub fn max_corner(&self) -> Vec2<f64> {
        Vec2::new(
            self.centre.x + self.half_extent.x,
            self.centre.y + self.half_extent.y,
        )
    }

    pub fn misses_rect(&self, rect: &TileRect) -> bool {
        let lo = Vec2::new(rect.x0 as f64, rect.y0 as f64);
        let hi = Vec2::new((rect.x0 + rect.w) as f64, (rect.y0 + rect.h) as f64);
        min_quadratic_over_rect(self.form, self.centre, lo, hi) > self.reach
    }

    pub fn contains_point(&self, p: Vec2<f64>) -> bool {
        let d = Vec2::new(p.x - self.centre.x, p.y - self.centre.y);
        self.form.quad_form(d) <= self.reach
    }
}

pub fn ellipsoid_screen_extent(
    camera: &Camera<f64>,
    mean_cam: Vec3<f64>,
    lambda_cam: &Sym3<f64>,
    support_sq: f64,
) -> Option<ScreenExtent> {
    if !(support_sq > 0.0) || !support_sq.is_finite() {
        return None;
    }
    if !mean_cam.is_finite() || !lambda_cam.is_finite() {
        return None;
    }
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    if !(fx > 0.0 && fy > 0.0) {
        return None;
    }
    let (centre_uv, form_uv, reach) = match camera.projection {
        Projection::Orthographic => {
            let sigma = lambda_cam.inverse_spd()?;
            let form = sigma.marginal_xy().inverse()?;
            (Vec2::new(mean_cam.x, mean_cam.y), form, support_sq)
        }
        Projection::Pinhole => {
            let a_mu = lambda_cam.apply(mean_cam);
            let outside = lambda_cam.quad_form(mean_cam) - support_sq;
            if !(outside > 0.0) {
                return None;
            }
            let form = Sym2::new(
                outside * lambda_cam.xx - a_mu.x * a_mu.x,
                outside * lambda_cam.xy - a_mu.x * a_mu.y,
                outside * lambda_cam.yy - a_mu.y * a_mu.y,
            );
            if !(form.xx > 0.0) {
                return None;
            }
            let inverse = form.inverse()?;
            let g = Vec2::new(
                a_mu.x * a_mu.z - outside * lambda_cam.xz,
                a_mu.y * a_mu.z - outside * lambda_cam.yz,
            );
            let m = Vec2::new(
                inverse.xx * g.x + inverse.xy * g.y,
                inverse.xy * g.x + inverse.yy * g.y,
            );
            let reach = a_mu.z * a_mu.z - outside * lambda_cam.zz + g.dot(m);
            if !(reach > 0.0) {
                return None;
            }
            (m, form, reach)
        }
    };
    let centre = Vec2::new(
        fx * centre_uv.x + camera.principal.x,
        fy * centre_uv.y + camera.principal.y,
    );
    let form = Sym2::new(
        form_uv.xx / (fx * fx),
        form_uv.xy / (fx * fy),
        form_uv.yy / (fy * fy),
    );
    let inverse = form.inverse()?;
    let half_extent = Vec2::new((reach * inverse.xx).sqrt(), (reach * inverse.yy).sqrt());
    if !centre.is_finite() || !form.is_finite() || !half_extent.is_finite() {
        return None;
    }
    Some(ScreenExtent {
        centre,
        form,
        reach,
        half_extent,
    })
}
