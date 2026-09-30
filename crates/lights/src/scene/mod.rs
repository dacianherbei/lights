pub mod cage;
pub mod demo;
pub mod hierarchy;
pub mod io;
pub mod prepare;
pub mod showcase;

use crate::error::{DegenerateReason, LightsError, Result};
use crate::numerics::float::FloatingPoint;
use crate::numerics::matrix::Mat3;
use crate::numerics::quaternion::Quat;
use crate::numerics::symmetric::Sym3;
use crate::numerics::vector::{Rgb, Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Atom<T> {
    pub mean: Vec3<T>,
    pub rotation: Quat<T>,
    pub scale: Vec3<T>,
    pub opacity: T,
    pub colour: Rgb<T>,
}

impl<T: FloatingPoint> Atom<T> {
    pub fn new(
        mean: Vec3<T>,
        rotation: Quat<T>,
        scale: Vec3<T>,
        opacity: T,
        colour: Rgb<T>,
    ) -> Self {
        Self {
            mean,
            rotation,
            scale,
            opacity,
            colour,
        }
    }

    pub fn isotropic(mean: Vec3<T>, sigma: T, opacity: T, colour: Rgb<T>) -> Self {
        Self::new(
            mean,
            Quat::identity(),
            Vec3::splat(sigma),
            opacity,
            colour,
        )
    }

    pub fn covariance(&self) -> Sym3<T> {
        let r = self.rotation.to_mat3();
        let m = r.scale_columns(self.scale);
        Sym3::from_factor(&m)
    }

    pub fn mapped_affine(&self, linear: &Mat3<T>, offset: Vec3<T>) -> Option<Self> {
        let sigma = self.covariance().congruence(linear);
        let (values, vectors) = sigma.eigen()?;
        if !(values.min_component() > T::ZERO) {
            return None;
        }
        let rotation = Quat::from_rotation_matrix(&vectors)?;
        Some(Self {
            mean: linear.mul_vec(self.mean) + offset,
            rotation,
            scale: Vec3::new(values.x.sqrt(), values.y.sqrt(), values.z.sqrt()),
            opacity: self.opacity,
            colour: self.colour,
        })
    }

    pub fn validate(&self, scale_floor: T) -> Result<()> {
        if !(self.mean.is_finite()
            && self.rotation.is_finite()
            && self.scale.is_finite()
            && self.opacity.is_finite()
            && self.colour.is_finite())
        {
            return Err(LightsError::DegenerateAtom {
                index: 0,
                reason: DegenerateReason::NonFiniteParameter,
            });
        }
        if self.scale.min_component() < scale_floor {
            return Err(LightsError::DegenerateAtom {
                index: 0,
                reason: DegenerateReason::NonPositiveScale,
            });
        }
        if self.rotation.normalize().is_none() {
            return Err(LightsError::DegenerateAtom {
                index: 0,
                reason: DegenerateReason::UnnormalisableRotation,
            });
        }
        Ok(())
    }

    pub fn to_f64(&self) -> Atom<f64> {
        Atom {
            mean: self.mean.to_f64(),
            rotation: self.rotation.to_f64(),
            scale: self.scale.to_f64(),
            opacity: self.opacity.to_f64(),
            colour: self.colour.to_f64(),
        }
    }

    pub fn from_f64(a: &Atom<f64>) -> Self {
        Atom {
            mean: Vec3::from_f64(a.mean),
            rotation: Quat::from_f64(a.rotation),
            scale: Vec3::from_f64(a.scale),
            opacity: T::from_f64(a.opacity),
            colour: Rgb::from_f64(a.colour),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scene<T> {
    atoms: Vec<Atom<T>>,
}

impl<T: FloatingPoint> Scene<T> {
    pub fn new() -> Self {
        Self { atoms: Vec::new() }
    }

    pub fn push(&mut self, atom: Atom<T>, scale_floor: T) -> Result<()> {
        let index = self.atoms.len();
        let prepared = Self::ingest(atom, index, scale_floor)?;
        self.atoms.push(prepared);
        Ok(())
    }

    pub fn set(&mut self, index: usize, atom: Atom<T>, scale_floor: T) -> Result<()> {
        if index >= self.atoms.len() {
            return Err(LightsError::AtomOutOfRange {
                index,
                len: self.atoms.len(),
            });
        }
        let prepared = Self::ingest(atom, index, scale_floor)?;
        self.atoms[index] = prepared;
        Ok(())
    }

    pub fn remove(&mut self, index: usize) -> Result<Atom<T>> {
        if index >= self.atoms.len() {
            return Err(LightsError::AtomOutOfRange {
                index,
                len: self.atoms.len(),
            });
        }
        Ok(self.atoms.remove(index))
    }

    pub fn clear(&mut self) {
        self.atoms.clear();
    }

    fn ingest(atom: Atom<T>, index: usize, scale_floor: T) -> Result<Atom<T>> {
        let mut prepared = atom;
        prepared.scale = Vec3::new(
            atom.scale.x.max(scale_floor),
            atom.scale.y.max(scale_floor),
            atom.scale.z.max(scale_floor),
        );
        prepared.rotation = match atom.rotation.normalize() {
            Some(q) => q,
            None => {
                return Err(LightsError::DegenerateAtom {
                    index,
                    reason: DegenerateReason::UnnormalisableRotation,
                });
            }
        };
        if let Err(e) = prepared.validate(scale_floor) {
            return Err(match e {
                LightsError::DegenerateAtom { reason, .. } => {
                    LightsError::DegenerateAtom { index, reason }
                }
                other => other,
            });
        }
        Ok(prepared)
    }

    pub fn atoms(&self) -> &[Atom<T>] {
        &self.atoms
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn bounds(&self, sigmas: T) -> (Vec3<T>, Vec3<T>) {
        let mut lo = Vec3::splat(T::INFINITY);
        let mut hi = Vec3::splat(T::NEG_INFINITY);
        for a in self.atoms.iter() {
            let d = a.covariance().diagonal();
            let half = Vec3::new(d.x.sqrt(), d.y.sqrt(), d.z.sqrt()) * sigmas;
            let min = a.mean - half;
            let max = a.mean + half;
            lo = Vec3::new(lo.x.min(min.x), lo.y.min(min.y), lo.z.min(min.z));
            hi = Vec3::new(hi.x.max(max.x), hi.y.max(max.y), hi.z.max(max.z));
        }
        (lo, hi)
    }

    pub fn to_f64(&self) -> Scene<f64> {
        Scene {
            atoms: self.atoms.iter().map(|a| a.to_f64()).collect(),
        }
    }

    pub fn from_f64(s: &Scene<f64>) -> Self {
        Scene {
            atoms: s.atoms.iter().map(|a| Atom::from_f64(a)).collect(),
        }
    }
}
