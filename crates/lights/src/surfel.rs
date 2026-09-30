use crate::camera::{Camera, Ray};
use crate::error::{LightsError, Result};
use crate::image::{DepthImage, FloatImage};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::{Rgb, Vec2, Vec3};
use crate::settings::DirectionalLight;

pub const SURFEL_SUPPORT_SIGMAS: f64 = 3.0;
pub const OPAQUE_TRANSMITTANCE: f64 = 1.0 / 1024.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material<T> {
    pub emission: Rgb<T>,
    pub specular: T,
    pub shininess: T,
}

impl<T: FloatingPoint> Material<T> {
    pub fn matte() -> Self {
        Self {
            emission: Rgb::zero(),
            specular: T::ZERO,
            shininess: T::ONE,
        }
    }

    pub fn new(emission: Rgb<T>, specular: T, shininess: T) -> Result<Self> {
        if !(emission.is_finite() && specular.is_finite() && shininess.is_finite()) {
            return Err(LightsError::NonFinite { what: "surfel material" });
        }
        if !(emission.x >= T::ZERO && emission.y >= T::ZERO && emission.z >= T::ZERO) {
            return Err(LightsError::DegenerateSurfel { reason: "negative emission" });
        }
        if !(specular >= T::ZERO) {
            return Err(LightsError::DegenerateSurfel { reason: "negative specular weight" });
        }
        if !(shininess > T::ZERO) {
            return Err(LightsError::DegenerateSurfel { reason: "non-positive shininess" });
        }
        Ok(Self {
            emission,
            specular,
            shininess,
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Surfel<T> {
    pub centre: Vec3<T>,
    pub normal: Vec3<T>,
    pub tangent: Vec3<T>,
    pub radii: Vec2<T>,
    pub opacity: T,
    pub colour: Rgb<T>,
    pub material: Material<T>,
}

impl<T: FloatingPoint> Surfel<T> {
    pub fn new(
        centre: Vec3<T>,
        normal: Vec3<T>,
        tangent: Vec3<T>,
        radii: Vec2<T>,
        opacity: T,
        colour: Rgb<T>,
    ) -> Result<Self> {
        if !(centre.is_finite()
            && normal.is_finite()
            && tangent.is_finite()
            && radii.is_finite()
            && opacity.is_finite()
            && colour.is_finite())
        {
            return Err(LightsError::NonFinite { what: "surfel" });
        }
        let normal = match normal.normalize() {
            Some(n) => n,
            None => return Err(LightsError::DegenerateSurfel { reason: "zero normal" }),
        };
        let tangent = match (tangent - normal * tangent.dot(normal)).normalize() {
            Some(t) => t,
            None => {
                return Err(LightsError::DegenerateSurfel {
                    reason: "tangent parallel to the normal",
                });
            }
        };
        if !(radii.x > T::ZERO && radii.y > T::ZERO) {
            return Err(LightsError::DegenerateSurfel { reason: "non-positive radius" });
        }
        if !(opacity >= T::ZERO && opacity <= T::ONE) {
            return Err(LightsError::DegenerateSurfel { reason: "opacity outside [0, 1]" });
        }
        Ok(Self {
            centre,
            normal,
            tangent,
            radii,
            opacity,
            colour,
            material: Material::matte(),
        })
    }

    pub fn with_material(self, material: Material<T>) -> Self {
        Self { material, ..self }
    }

    pub fn bitangent(&self) -> Vec3<T> {
        self.normal.cross(self.tangent)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfelScene<T> {
    surfels: Vec<Surfel<T>>,
}

impl<T: FloatingPoint> SurfelScene<T> {
    pub fn new() -> Self {
        Self { surfels: Vec::new() }
    }

    pub fn push(&mut self, surfel: Surfel<T>) {
        self.surfels.push(surfel);
    }

    pub fn surfels(&self) -> &[Surfel<T>] {
        &self.surfels
    }

    pub fn len(&self) -> usize {
        self.surfels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.surfels.is_empty()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SurfelHit<T> {
    pub t: T,
    pub weight: T,
    pub shade: T,
    pub specular: T,
    pub index: u32,
}

pub fn surfel_shade<T: FloatingPoint>(
    normal: Vec3<T>,
    direction: Vec3<T>,
    light: &DirectionalLight<T>,
) -> T {
    let facing = if normal.dot(direction) > T::ZERO { -normal } else { normal };
    light.ambient + light.diffuse * facing.dot(light.direction).max(T::ZERO)
}

pub fn surfel_specular<T: FloatingPoint>(
    normal: Vec3<T>,
    direction: Vec3<T>,
    light: &DirectionalLight<T>,
    material: &Material<T>,
) -> T {
    let facing = if normal.dot(direction) > T::ZERO { -normal } else { normal };
    if !(material.specular > T::ZERO && facing.dot(light.direction) > T::ZERO) {
        return T::ZERO;
    }
    let half = match (light.direction - direction).normalize() {
        Some(h) => h,
        None => return T::ZERO,
    };
    material.specular * light.diffuse * facing.dot(half).max(T::ZERO).powf(material.shininess)
}

pub fn intersect_surfel<T: FloatingPoint>(
    surfel: &Surfel<T>,
    ray: &Ray<T>,
    light: Option<&DirectionalLight<T>>,
    index: u32,
) -> Option<SurfelHit<T>> {
    let denom = surfel.normal.dot(ray.direction);
    if !(denom.abs() > T::EPSILON) {
        return None;
    }
    let t = surfel.normal.dot(surfel.centre - ray.origin) / denom;
    if !(t >= ray.t_min && t <= ray.t_max) {
        return None;
    }
    let offset = ray.origin + ray.direction * t - surfel.centre;
    let u = offset.dot(surfel.tangent) / surfel.radii.x;
    let v = offset.dot(surfel.bitangent()) / surfel.radii.y;
    let r2 = u * u + v * v;
    let support = T::from_f64(SURFEL_SUPPORT_SIGMAS);
    if !(r2 <= support * support) {
        return None;
    }
    let weight = (-T::HALF * r2).exp();
    let shade = match light {
        Some(l) => surfel_shade(surfel.normal, ray.direction, l),
        None => T::ONE,
    };
    let specular = match light {
        Some(l) => surfel_specular(surfel.normal, ray.direction, l, &surfel.material),
        None => T::ZERO,
    };
    Some(SurfelHit {
        t,
        weight,
        shade,
        specular,
        index,
    })
}

pub fn to_camera_frame<T: FloatingPoint>(camera: &Camera<T>, surfel: &Surfel<T>) -> Surfel<T> {
    Surfel {
        centre: camera.to_camera(surfel.centre),
        normal: camera.rotation.mul_vec(surfel.normal),
        tangent: camera.rotation.mul_vec(surfel.tangent),
        radii: surfel.radii,
        opacity: surfel.opacity,
        colour: surfel.colour,
        material: surfel.material,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfelLayer<T> {
    pub colour: FloatImage<T>,
    pub coverage: FloatImage<T>,
    pub depth: DepthImage<T>,
}

pub fn render_surfels<T: FloatingPoint>(
    scene: &SurfelScene<T>,
    camera: &Camera<T>,
    light: Option<&DirectionalLight<T>>,
) -> Result<(FloatImage<T>, FloatImage<T>)> {
    let layer = render_surfel_layer(scene, camera, light)?;
    Ok((layer.colour, layer.coverage))
}

pub fn render_surfel_layer<T: FloatingPoint>(
    scene: &SurfelScene<T>,
    camera: &Camera<T>,
    light: Option<&DirectionalLight<T>>,
) -> Result<SurfelLayer<T>> {
    if scene.is_empty() {
        return Err(LightsError::EmptyScene);
    }
    let local: Vec<Surfel<T>> = scene
        .surfels()
        .iter()
        .map(|s| to_camera_frame(camera, s))
        .collect();
    let mut colour = FloatImage::new(camera.width, camera.height);
    let mut coverage = FloatImage::new(camera.width, camera.height);
    let mut depth = DepthImage::<T>::far(camera.width, camera.height);
    let opaque = T::from_f64(OPAQUE_TRANSMITTANCE);
    let mut hits: Vec<SurfelHit<T>> = Vec::with_capacity(local.len());
    for py in 0..camera.height {
        for px in 0..camera.width {
            let ray = camera.ray(px, py);
            hits.clear();
            for (i, s) in local.iter().enumerate() {
                if let Some(h) = intersect_surfel(s, &ray, light, i as u32) {
                    hits.push(h);
                }
            }
            hits.sort_unstable_by_key(|h| (h.t.to_bits_ordered(), h.index));
            let mut acc = Rgb::zero();
            let mut transmittance = T::ONE;
            for h in &hits {
                let s = &local[h.index as usize];
                let alpha = s.opacity * h.weight;
                let radiance = s.colour * h.shade + Rgb::splat(h.specular) + s.material.emission;
                acc = acc + radiance * (transmittance * alpha);
                transmittance = transmittance * (T::ONE - alpha);
                if transmittance <= opaque && !depth.get(px, py).is_finite() {
                    depth.set(px, py, h.t);
                }
            }
            colour.set(px, py, acc);
            coverage.set(px, py, Rgb::splat(T::ONE - transmittance));
        }
    }
    Ok(SurfelLayer {
        colour,
        coverage,
        depth,
    })
}
