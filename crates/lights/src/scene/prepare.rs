use crate::error::{DegenerateReason, LightsError, Result};
use crate::kernel::{CurveShape, Kernel};
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::{LowerTri3, Sym3};
use crate::numerics::vector::{Rgb, Vec3};
use crate::scene::Scene;
use crate::settings::SharedSettings;
use core::sync::atomic::{AtomicU64, Ordering};

static EPOCHS: AtomicU64 = AtomicU64::new(1);

fn next_epoch() -> u64 {
    EPOCHS.fetch_add(1, Ordering::Relaxed)
}

pub struct PreparedScene<T: FloatingPoint, K: Kernel<T>> {
    epoch: u64,
    pub mean: Vec<Vec3<T>>,
    pub sigma: Vec<Sym3<T>>,
    pub lambda: Vec<Sym3<T>>,
    pub whiten: Vec<LowerTri3<T>>,
    pub opacity: Vec<T>,
    pub colour: Vec<Rgb<T>>,
    pub params: Vec<K::Params>,
    pub scale_max: Vec<T>,
    pub scale_min: Vec<T>,
}

#[derive(Copy, Clone, Debug)]
pub struct AtomView<'a, T, K: Kernel<T>>
where
    T: FloatingPoint,
{
    pub index: usize,
    pub mean: Vec3<T>,
    pub sigma: &'a Sym3<T>,
    pub lambda: &'a Sym3<T>,
    pub whiten: &'a LowerTri3<T>,
    pub params: &'a K::Params,
    pub opacity: T,
    pub colour: Rgb<T>,
    pub scale_max: T,
    pub scale_min: T,
}

impl<T: FloatingPoint, K: Kernel<T>> PreparedScene<T, K> {
    pub fn build(
        scene: &Scene<T>,
        params: &[K::Params],
        settings: &SharedSettings<T>,
    ) -> Result<Self> {
        if scene.is_empty() {
            return Err(LightsError::EmptyScene);
        }
        let n = scene.len();
        let any_visible = scene.atoms().iter().any(|a| a.opacity != T::ZERO);
        let mut out = Self {
            epoch: next_epoch(),
            mean: Vec::with_capacity(n),
            sigma: Vec::with_capacity(n),
            lambda: Vec::with_capacity(n),
            whiten: Vec::with_capacity(n),
            opacity: Vec::with_capacity(n),
            colour: Vec::with_capacity(n),
            params: Vec::with_capacity(n),
            scale_max: Vec::with_capacity(n),
            scale_min: Vec::with_capacity(n),
        };
        for (index, atom) in scene.atoms().iter().enumerate() {
            if atom.scale.min_component() < settings.scale_floor {
                return Err(LightsError::DegenerateAtom {
                    index,
                    reason: DegenerateReason::NonPositiveScale,
                });
            }
            let sigma = atom.covariance();
            let chol = match sigma.cholesky() {
                Some(c) => c,
                None => {
                    return Err(LightsError::DegenerateAtom {
                        index,
                        reason: DegenerateReason::NonSpdCovariance,
                    });
                }
            };
            if any_visible && atom.opacity == T::ZERO {
                continue;
            }
            let whiten = chol.inverse();
            let lambda = Sym3::from_lower_transpose(&whiten);
            out.mean.push(atom.mean);
            out.sigma.push(sigma);
            out.lambda.push(lambda);
            out.whiten.push(whiten);
            out.opacity.push(atom.opacity);
            out.colour.push(atom.colour);
            out.params.push(match params.get(index) {
                Some(p) => *p,
                None => K::Params::default(),
            });
            out.scale_max.push(atom.scale.max_component());
            out.scale_min.push(atom.scale.min_component());
        }
        Ok(out)
    }

    pub fn build_uniform(
        scene: &Scene<T>,
        shape: &CurveShape,
        settings: &SharedSettings<T>,
    ) -> Result<Self> {
        let params = vec![K::params_from(shape); scene.len()];
        Self::build(scene, &params, settings)
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn len(&self) -> usize {
        self.mean.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    pub fn atom(&self, i: usize) -> AtomView<'_, T, K> {
        if i >= self.len() {
            crate::guard::unreachable(
                "PreparedScene::atom",
                &format!("index {i} of {}", self.len()),
            );
        }
        AtomView {
            index: i,
            mean: self.mean[i],
            sigma: &self.sigma[i],
            lambda: &self.lambda[i],
            whiten: &self.whiten[i],
            params: &self.params[i],
            opacity: self.opacity[i],
            colour: self.colour[i],
            scale_max: self.scale_max[i],
            scale_min: self.scale_min[i],
        }
    }

    pub fn resident_bytes(&self) -> usize {
        stored_bytes(&self.mean)
            + stored_bytes(&self.sigma)
            + stored_bytes(&self.lambda)
            + stored_bytes(&self.whiten)
            + stored_bytes(&self.opacity)
            + stored_bytes(&self.colour)
            + stored_bytes(&self.params)
            + stored_bytes(&self.scale_max)
            + stored_bytes(&self.scale_min)
    }
}

fn stored_bytes<E>(values: &[E]) -> usize {
    values.len() * core::mem::size_of::<E>()
}
