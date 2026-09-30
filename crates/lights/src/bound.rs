pub const FIRST_ORDER_COEFFICIENT: f64 = 1.1594183222341823;
pub const SECOND_ORDER_COEFFICIENT: f64 = 1.0826822658929016;
pub const HEADROOM: f64 = 1.05;

use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::Sym3;
use crate::numerics::vector::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundTerms<T> {
    pub omega: T,
    pub rho_e: T,
    pub sin_theta: T,
    pub first_order: T,
    pub second_order: T,
    pub total: T,
}

pub fn atom_bound_terms_with_lambda<T: FloatingPoint>(
    mean_cam: Vec3<T>,
    sigma_zz: T,
    lambda_cam: &Sym3<T>,
    opacity: T,
) -> BoundTerms<T> {
    let zero = BoundTerms {
        omega: T::ZERO,
        rho_e: T::ZERO,
        sin_theta: T::ZERO,
        first_order: T::ZERO,
        second_order: T::ZERO,
        total: T::ZERO,
    };
    let rho_e_sq = lambda_cam.quad_form(mean_cam);
    let z = mean_cam.z;
    if !(rho_e_sq > T::ZERO) || !(z * z > T::ZERO) || !opacity.is_finite() {
        return zero;
    }
    let omega_sq = (sigma_zz / (z * z) - rho_e_sq.recip()).max(T::ZERO);
    let omega = omega_sq.sqrt();
    let first = T::from_f64(FIRST_ORDER_COEFFICIENT) * opacity * omega;
    let second = T::from_f64(SECOND_ORDER_COEFFICIENT) * opacity / rho_e_sq;
    let total = first + second;
    if !total.is_finite() {
        return zero;
    }
    let transverse = (mean_cam.x * mean_cam.x + mean_cam.y * mean_cam.y).sqrt();
    let distance = mean_cam.length();
    let sin_theta = if distance > T::ZERO {
        transverse / distance
    } else {
        T::ZERO
    };
    BoundTerms {
        omega,
        rho_e: rho_e_sq.sqrt(),
        sin_theta,
        first_order: first,
        second_order: second,
        total,
    }
}

pub fn atom_bound_with_lambda<T: FloatingPoint>(
    mean_cam: Vec3<T>,
    sigma_zz: T,
    lambda_cam: &Sym3<T>,
    opacity: T,
) -> T {
    atom_bound_terms_with_lambda(mean_cam, sigma_zz, lambda_cam, opacity).total
}

pub fn atom_bound_terms<T: FloatingPoint>(
    mean_cam: Vec3<T>,
    sigma_cam: &Sym3<T>,
    opacity: T,
) -> BoundTerms<T> {
    match sigma_cam.inverse_spd() {
        Some(lambda) => {
            atom_bound_terms_with_lambda(mean_cam, sigma_cam.zz, &lambda, opacity)
        }
        None => BoundTerms {
            omega: T::ZERO,
            rho_e: T::ZERO,
            sin_theta: T::ZERO,
            first_order: T::ZERO,
            second_order: T::ZERO,
            total: T::ZERO,
        },
    }
}

pub fn atom_bound<T: FloatingPoint>(
    mean_cam: Vec3<T>,
    sigma_cam: &Sym3<T>,
    opacity: T,
) -> T {
    atom_bound_terms(mean_cam, sigma_cam, opacity).total
}
