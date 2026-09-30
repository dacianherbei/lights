#[cfg(feature = "method-raycast")]
pub mod execute;
#[cfg(feature = "method-raycast")]
pub mod prepare;

#[cfg(feature = "method-raycast")]
mod implementation;
#[cfg(feature = "method-raycast")]
pub use implementation::RayCast;

#[cfg(not(feature = "method-raycast"))]
mod absent;
#[cfg(not(feature = "method-raycast"))]
pub use absent::RayCast;

use crate::tiling::DEFAULT_TILE;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RayCastConfig {
    pub accel: AccelKind,
    pub tile: u32,
    pub max_hits_per_pixel: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccelKind {
    BruteForce,
    TileBins,
}

impl RayCastConfig {
    pub fn production() -> Self {
        Self {
            accel: AccelKind::TileBins,
            tile: DEFAULT_TILE,
            max_hits_per_pixel: 4096,
        }
    }

    pub fn contract() -> Self {
        Self {
            accel: AccelKind::BruteForce,
            tile: DEFAULT_TILE,
            max_hits_per_pixel: 4096,
        }
    }
}

impl Default for RayCastConfig {
    fn default() -> Self {
        Self::production()
    }
}
