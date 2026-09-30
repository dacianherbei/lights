#[cfg(feature = "method-splat")]
pub mod bin;
#[cfg(feature = "method-splat")]
pub mod execute;
#[cfg(feature = "method-splat")]
pub mod project;

#[cfg(feature = "method-splat")]
mod implementation;
#[cfg(feature = "method-splat")]
pub use implementation::SplatRaster;

#[cfg(not(feature = "method-splat"))]
mod absent;
#[cfg(not(feature = "method-splat"))]
pub use absent::SplatRaster;

use crate::tiling::DEFAULT_TILE;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SplatConfig {
    pub tile: u32,
    pub sort: SortKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SortKind {
    GlobalSort,
    PerTileSort,
}

impl SplatConfig {
    pub fn reference() -> Self {
        Self {
            tile: DEFAULT_TILE,
            sort: SortKind::GlobalSort,
        }
    }

    pub fn contract() -> Self {
        Self {
            tile: DEFAULT_TILE,
            sort: SortKind::GlobalSort,
        }
    }
}

impl Default for SplatConfig {
    fn default() -> Self {
        Self::reference()
    }
}
