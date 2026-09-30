use crate::blend::{BlendSettings, EarlyOut};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::{Rgb, Vec3};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OrderingModel {
    CentreDepth,
    HitDepth,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelConvention {
    CornerSample,
    CentreSample,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SupportMode {
    FixedSigmas(f64),
    CutoffSolved,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DirectionalLight<T> {
    pub direction: Vec3<T>,
    pub ambient: T,
    pub diffuse: T,
}

impl<T: FloatingPoint> DirectionalLight<T> {
    pub fn new(direction: Vec3<T>, ambient: T, diffuse: T) -> Option<Self> {
        let direction = direction.normalize()?;
        Some(Self {
            direction,
            ambient,
            diffuse,
        })
    }

    pub fn from_f64(l: &DirectionalLight<f64>) -> Self {
        Self {
            direction: Vec3::from_f64(l.direction),
            ambient: T::from_f64(l.ambient),
            diffuse: T::from_f64(l.diffuse),
        }
    }

    pub fn to_f64(&self) -> DirectionalLight<f64> {
        DirectionalLight {
            direction: self.direction.to_f64(),
            ambient: self.ambient.to_f64(),
            diffuse: self.diffuse.to_f64(),
        }
    }

    pub fn fingerprint(&self) -> (f64, f64, f64, f64, f64) {
        (
            self.direction.x.to_f64(),
            self.direction.y.to_f64(),
            self.direction.z.to_f64(),
            self.ambient.to_f64(),
            self.diffuse.to_f64(),
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SharedSettings<T> {
    pub blend: BlendSettings<T>,
    pub ordering: OrderingModel,
    pub support: SupportMode,
    pub screen_dilation: T,
    pub energy_compensation: bool,
    pub near_plane: T,
    pub background: Rgb<T>,
    pub light: Option<DirectionalLight<T>>,
    pub scale_floor: T,
    pub threads: usize,
}

impl<T: FloatingPoint> SharedSettings<T> {
    pub fn reference() -> Self {
        Self {
            blend: BlendSettings::reference(),
            ordering: OrderingModel::CentreDepth,
            support: SupportMode::FixedSigmas(REFERENCE_SIGMAS),
            screen_dilation: T::from_f64(REFERENCE_DILATION),
            energy_compensation: false,
            near_plane: T::from_f64(REFERENCE_NEAR_PLANE),
            background: Rgb::zero(),
            light: None,
            scale_floor: T::from_f64(MIN_SCALE),
            threads: 0,
        }
    }

    pub fn contract() -> Self {
        Self {
            blend: BlendSettings::contract(),
            support: SupportMode::CutoffSolved,
            screen_dilation: T::ZERO,
            energy_compensation: false,
            ..Self::reference()
        }
    }

    pub fn prefiltered() -> Self {
        Self {
            screen_dilation: T::from_f64(PIXEL_BOX_VARIANCE),
            energy_compensation: true,
            ..Self::contract()
        }
    }

    pub fn from_f64(s: &SharedSettings<f64>) -> Self {
        Self {
            blend: BlendSettings::from_f64(&s.blend),
            ordering: s.ordering,
            support: s.support,
            screen_dilation: T::from_f64(s.screen_dilation),
            energy_compensation: s.energy_compensation,
            near_plane: T::from_f64(s.near_plane),
            background: Rgb::from_f64(s.background),
            light: s.light.as_ref().map(DirectionalLight::from_f64),
            scale_floor: T::from_f64(s.scale_floor),
            threads: s.threads,
        }
    }

    pub fn to_f64(&self) -> SharedSettings<f64> {
        SharedSettings {
            blend: self.blend.to_f64(),
            ordering: self.ordering,
            support: self.support,
            screen_dilation: self.screen_dilation.to_f64(),
            energy_compensation: self.energy_compensation,
            near_plane: self.near_plane.to_f64(),
            background: self.background.to_f64(),
            light: self.light.as_ref().map(|l| l.to_f64()),
            scale_floor: self.scale_floor.to_f64(),
            threads: self.threads,
        }
    }

    pub fn fingerprint(&self) -> SettingsFingerprint {
        SettingsFingerprint {
            blend: (
                self.blend.alpha_cap.to_f64(),
                self.blend.alpha_cutoff.to_f64(),
                self.blend.transmittance_cutoff.to_f64(),
                self.blend.early_out,
            ),
            ordering: self.ordering,
            support: self.support,
            screen_dilation: self.screen_dilation.to_f64(),
            energy_compensation: self.energy_compensation,
            near_plane: self.near_plane.to_f64(),
            background: (
                self.background.x.to_f64(),
                self.background.y.to_f64(),
                self.background.z.to_f64(),
            ),
            light: self.light.as_ref().map(|l| l.fingerprint()),
            scale_floor: self.scale_floor.to_f64(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SettingsFingerprint {
    pub blend: (f64, f64, f64, EarlyOut),
    pub ordering: OrderingModel,
    pub support: SupportMode,
    pub screen_dilation: f64,
    pub energy_compensation: bool,
    pub near_plane: f64,
    pub background: (f64, f64, f64),
    pub light: Option<(f64, f64, f64, f64, f64)>,
    pub scale_floor: f64,
}

pub const PIXEL_BOX_VARIANCE: f64 = 1.0 / 12.0;
pub const REFERENCE_DILATION: f64 = 0.3;
pub const REFERENCE_ALPHA_CAP: f64 = 0.99;
pub const REFERENCE_ALPHA_CUTOFF: f64 = 1.0 / 255.0;
pub const REFERENCE_TRANSMITTANCE_CUTOFF: f64 = 1.0e-4;
pub const REFERENCE_NEAR_PLANE: f64 = 0.2;
pub const REFERENCE_SIGMAS: f64 = 3.0;
pub const JACOBIAN_CLAMP_SLACK: f64 = 1.3;
pub const MIN_SCALE: f64 = 1.0e-8;
