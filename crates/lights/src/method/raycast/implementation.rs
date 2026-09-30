use crate::blend::BlendMode;
use crate::camera::Camera;
use crate::error::{LightsError, Result};
use crate::image::{DepthImage, Framebuffer, TileAccum};
use crate::kernel::Kernel;
use crate::method::raycast::execute::{Hit, limited, render_tile, shade_pixel};
use crate::method::raycast::prepare::{RayAtom, TileBinIndex, prepare_atoms, to_world_ray};
use crate::method::raycast::{AccelKind, RayCastConfig};
use crate::method::{ImageFormation, MethodId, PhaseReport};
use crate::numerics::float::FloatingPoint;
use crate::probe::{Counter, Counters, NoProbe, Probe, ProbeSink};
use crate::scene::prepare::PreparedScene;
use crate::settings::SharedSettings;
use crate::tiling::TileGrid;
use rayon::prelude::*;
use std::time::Instant;

pub struct RayCast<T: FloatingPoint, K: Kernel<T>> {
    settings: SharedSettings<T>,
    config: RayCastConfig,
    atoms: Vec<RayAtom<T, K>>,
    index: Option<TileBinIndex>,
    camera: Option<Camera<T>>,
    bound_epoch: Option<u64>,
    grid: Option<TileGrid>,
    clamped: u64,
    bind_counters: Counters,
    pool: Option<rayon::ThreadPool>,
    limit: Option<DepthImage<T>>,
}

impl<T: FloatingPoint, K: Kernel<T>> RayCast<T, K> {
    pub fn new(config: RayCastConfig, settings: &SharedSettings<T>) -> Result<Self> {
        if settings.screen_dilation != T::ZERO {
            return Err(LightsError::PrefilterUnsupported {
                setting: "screen_dilation",
                method: MethodId::RayCast,
            });
        }
        if settings.energy_compensation {
            return Err(LightsError::PrefilterUnsupported {
                setting: "energy_compensation",
                method: MethodId::RayCast,
            });
        }
        if config.tile == 0 {
            return Err(LightsError::InvalidCamera(
                "raycast tile size must be non-zero",
            ));
        }
        if config.max_hits_per_pixel == 0 {
            return Err(LightsError::InvalidCamera(
                "raycast max_hits_per_pixel must be non-zero",
            ));
        }
        Ok(Self {
            settings: *settings,
            config,
            atoms: Vec::new(),
            index: None,
            camera: None,
            bound_epoch: None,
            grid: None,
            clamped: 0,
            bind_counters: Counters::default(),
            pool: crate::method::build_thread_pool(settings.threads),
            limit: None,
        })
    }

    pub fn config(&self) -> RayCastConfig {
        self.config
    }

    pub fn grid(&self) -> Option<TileGrid> {
        self.grid
    }

    pub fn ray_atoms(&self) -> &[RayAtom<T, K>] {
        &self.atoms
    }

    pub fn bin_index(&self) -> Option<&TileBinIndex> {
        self.index.as_ref()
    }

    pub fn bind_counters(&self) -> Counters {
        self.bind_counters
    }

    fn bound_to(&self, scene: &PreparedScene<T, K>) -> Result<(TileGrid, Camera<T>)> {
        let (grid, camera) = self.bound_state()?;
        match self.bound_epoch {
            Some(bound) if bound == scene.epoch() => Ok((grid, camera)),
            Some(bound) => Err(LightsError::SceneNotBound {
                method: MethodId::RayCast,
                bound,
                given: scene.epoch(),
            }),
            None => Err(LightsError::InvalidCamera(
                "bind() must run before render()",
            )),
        }
    }

    fn check_limit(&self, grid: &TileGrid) -> Result<()> {
        match self.limit.as_ref() {
            Some(depth) if depth.width() != grid.width || depth.height() != grid.height => {
                Err(LightsError::SizeMismatch {
                    a: (grid.width, grid.height),
                    b: (depth.width(), depth.height()),
                })
            }
            Some(_) => Ok(()),
            None => Ok(()),
        }
    }

    fn bound_state(&self) -> Result<(TileGrid, Camera<T>)> {
        match (self.grid, self.camera) {
            (Some(g), Some(c)) => Ok((g, c)),
            _ => Err(LightsError::InvalidCamera(
                "raycast: bind() must run before render()",
            )),
        }
    }
}

impl<T: FloatingPoint, K: Kernel<T>> ImageFormation<T, K> for RayCast<T, K> {
    fn id(&self) -> MethodId {
        MethodId::RayCast
    }

    fn build(&mut self, scene: &PreparedScene<T, K>) -> Result<PhaseReport> {
        let started = Instant::now();
        if scene.is_empty() {
            return Err(LightsError::EmptyScene);
        }
        self.grid = None;
        self.camera = None;
        self.index = None;
        self.atoms.clear();
        self.clamped = 0;
        self.bind_counters = Counters::default();
        let mut counters = Counters::default();
        counters.add(Counter::AtomsPrepared, scene.len() as u64);
        Ok(PhaseReport {
            wall_nanos: started.elapsed().as_nanos(),
            resident_bytes: scene.resident_bytes(),
            counters,
        })
    }

    fn bind(&mut self, scene: &PreparedScene<T, K>, camera: &Camera<T>) -> Result<PhaseReport> {
        let started = Instant::now();
        let grid = TileGrid::new(camera.width, camera.height, self.config.tile)?;
        let (counters, clamped) =
            prepare_atoms::<T, K>(scene, camera, &self.settings, &mut self.atoms);
        let mut counters = counters;
        self.index = match self.config.accel {
            AccelKind::BruteForce => None,
            AccelKind::TileBins => {
                let index = TileBinIndex::build::<T, K>(&self.atoms, camera, &grid)?;
                counters.add(Counter::TileAtomPairs, index.pair_count());
                Some(index)
            }
        };
        self.grid = Some(grid);
        self.camera = Some(*camera);
        self.bound_epoch = Some(scene.epoch());
        self.clamped = clamped;
        self.bind_counters = counters;
        Ok(PhaseReport {
            wall_nanos: started.elapsed().as_nanos(),
            resident_bytes: self.resident_bytes(),
            counters,
        })
    }

    fn render<P: Probe<T>>(
        &self,
        scene: &PreparedScene<T, K>,
        fb: &mut Framebuffer<T, K::Blend>,
        sink: &mut ProbeSink<T>,
    ) -> Result<PhaseReport> {
        let started = Instant::now();
        let (grid, camera) = self.bound_to(scene)?;
        if fb.grid() != grid {
            return Err(LightsError::SizeMismatch {
                a: (grid.width, grid.height),
                b: (fb.width(), fb.height()),
            });
        }
        self.check_limit(&grid)?;
        let limit = self.limit.as_ref();
        let mut tiles = fb.split_tiles_mut(&grid)?;
        let atoms = &self.atoms;
        let index = self.index.as_ref();
        let settings = &self.settings;
        let config = &self.config;

        let shade = move |acc: &mut TileAccum<'_, T, K::Blend>| -> Result<P> {
            let mut probe = P::for_tile(acc.rect);
            let mut hits: Vec<Hit<T>> = Vec::new();
            render_tile::<T, K, P>(
                atoms,
                index,
                &camera,
                acc.rect,
                settings,
                config,
                limit,
                acc,
                &mut probe,
                &mut hits,
            )?;
            Ok(probe)
        };

        let outcomes: Vec<Result<P>> = if settings.threads == 1 {
            tiles.iter_mut().map(|acc| shade(acc)).collect()
        } else if settings.threads == 0 {
            tiles.par_iter_mut().map(|acc| shade(acc)).collect()
        } else {
            match self.pool.as_ref() {
                Some(pool) => {
                    pool.install(|| tiles.par_iter_mut().map(|acc| shade(acc)).collect())
                }
                None => tiles.par_iter_mut().map(|acc| shade(acc)).collect(),
            }
        };
        drop(tiles);

        let mut probes = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            probes.push(outcome?);
        }
        for probe in probes {
            probe.merge_into(sink);
        }
        let counters = sink.counters;
        sink.counters.merge(&self.bind_counters);
        Ok(PhaseReport {
            wall_nanos: started.elapsed().as_nanos(),
            resident_bytes: self.resident_bytes(),
            counters,
        })
    }

    fn sample_pixel(
        &self,
        scene: &PreparedScene<T, K>,
        px: u32,
        py: u32,
    ) -> Result<<K::Blend as BlendMode<T>>::Accum> {
        let (grid, camera) = self.bound_to(scene)?;
        if px >= grid.width || py >= grid.height {
            return Err(LightsError::SizeMismatch {
                a: (px, py),
                b: (grid.width, grid.height),
            });
        }
        let candidates: Option<&[u32]> = match self.index.as_ref() {
            Some(bins) => Some(bins.candidates(px, py)),
            None => None,
        };
        self.check_limit(&grid)?;
        let to_world = camera.rotation.transpose();
        let ray = limited(
            to_world_ray(&to_world, camera.eye, &camera.ray(px, py)),
            self.limit.as_ref(),
            px,
            py,
        );
        let mut cell = <K::Blend as BlendMode<T>>::init();
        let mut probe = NoProbe;
        let mut hits: Vec<Hit<T>> = Vec::new();
        shade_pixel::<T, K, NoProbe>(
            &self.atoms,
            candidates,
            &ray,
            0,
            &self.settings,
            self.config.max_hits_per_pixel,
            &mut cell,
            &mut probe,
            &mut hits,
        )?;
        Ok(cell)
    }

    fn determinant_culled_atoms(&self) -> u64 {
        0
    }

    fn clamped_atoms(&self) -> u64 {
        self.clamped
    }

    fn resident_bytes(&self) -> usize {
        self.atoms.len() * core::mem::size_of::<RayAtom<T, K>>()
            + match self.index.as_ref() {
                Some(bins) => bins.resident_bytes(),
                None => 0,
            }
            + match self.limit.as_ref() {
                Some(depth) => {
                    (depth.width() as usize) * (depth.height() as usize) * core::mem::size_of::<T>()
                }
                None => 0,
            }
    }

    fn limit_depth(&mut self, limit: Option<&DepthImage<T>>) -> Result<()> {
        self.limit = limit.cloned();
        Ok(())
    }
}
