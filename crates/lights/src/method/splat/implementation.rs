use crate::blend::BlendMode;
use crate::camera::Camera;
use crate::error::{LightsError, Result};
use crate::image::{DepthImage, Framebuffer, TileAccum};
use crate::kernel::Kernel;
use crate::method::splat::SplatConfig;
use crate::method::splat::bin::{SortKey, bin_and_sort};
use crate::method::splat::execute::{BatchEntry, gather_batch, render_tile, shade_pixel};
use crate::method::splat::project::{ScreenAtom, project_scene};
use crate::method::{ImageFormation, MethodId, PhaseReport};
use crate::numerics::float::FloatingPoint;
use crate::probe::{Counter, Counters, NoProbe, Probe, ProbeSink};
use crate::scene::prepare::PreparedScene;
use crate::settings::{OrderingModel, SharedSettings};
use crate::tiling::TileGrid;
use rayon::prelude::*;
use std::time::Instant;

pub struct SplatRaster<T: FloatingPoint, K: Kernel<T>> {
    settings: SharedSettings<T>,
    config: SplatConfig,
    grid: Option<TileGrid>,
    camera: Option<Camera<T>>,
    bound_epoch: Option<u64>,
    screen: Vec<ScreenAtom<T, K>>,
    keys: Vec<SortKey>,
    ranges: Vec<(u32, u32)>,
    clamped: u64,
    determinant_culled: u64,
    bind_counters: Counters,
    pool: Option<rayon::ThreadPool>,
}

impl<T: FloatingPoint, K: Kernel<T>> SplatRaster<T, K> {
    pub fn new(config: SplatConfig, settings: &SharedSettings<T>) -> Result<Self> {
        if !K::HAS_SCREEN_MARGINAL {
            return Err(LightsError::KernelNotProjectable {
                kernel: K::NAME,
                method: MethodId::SplatRaster,
            });
        }
        match settings.ordering {
            OrderingModel::CentreDepth => (),
            OrderingModel::HitDepth => {
                return Err(LightsError::OrderingUnsupported {
                    ordering: settings.ordering,
                    method: MethodId::SplatRaster,
                });
            }
        }
        if settings.light.is_some() {
            return Err(LightsError::LightingUnsupported {
                method: MethodId::SplatRaster,
            });
        }
        if config.tile == 0 {
            return Err(LightsError::InvalidCamera("splat tile size must be non-zero"));
        }
        Ok(Self {
            settings: *settings,
            config,
            grid: None,
            camera: None,
            bound_epoch: None,
            screen: Vec::new(),
            keys: Vec::new(),
            ranges: Vec::new(),
            clamped: 0,
            determinant_culled: 0,
            bind_counters: Counters::default(),
            pool: crate::method::build_thread_pool(settings.threads),
        })
    }

    pub fn config(&self) -> SplatConfig {
        self.config
    }

    pub fn grid(&self) -> Option<TileGrid> {
        self.grid
    }

    pub fn screen_atoms(&self) -> &[ScreenAtom<T, K>] {
        &self.screen
    }

    pub fn keys(&self) -> &[SortKey] {
        &self.keys
    }

    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    pub fn bind_counters(&self) -> Counters {
        self.bind_counters
    }

    fn bound_to(&self, scene: &PreparedScene<T, K>) -> Result<(TileGrid, Camera<T>)> {
        let (grid, camera) = self.bound_state()?;
        match self.bound_epoch {
            Some(bound) if bound == scene.epoch() => Ok((grid, camera)),
            Some(bound) => Err(LightsError::SceneNotBound {
                method: MethodId::SplatRaster,
                bound,
                given: scene.epoch(),
            }),
            None => Err(LightsError::InvalidCamera(
                "bind() must run before render()",
            )),
        }
    }

    fn bound_state(&self) -> Result<(TileGrid, Camera<T>)> {
        match (self.grid, self.camera) {
            (Some(g), Some(c)) => Ok((g, c)),
            _ => Err(LightsError::InvalidCamera(
                "splat: bind() must run before render()",
            )),
        }
    }
}

impl<T: FloatingPoint, K: Kernel<T>> ImageFormation<T, K> for SplatRaster<T, K> {
    fn id(&self) -> MethodId {
        MethodId::SplatRaster
    }

    fn build(&mut self, scene: &PreparedScene<T, K>) -> Result<PhaseReport> {
        let started = Instant::now();
        if scene.is_empty() {
            return Err(LightsError::EmptyScene);
        }
        self.grid = None;
        self.camera = None;
        self.screen.clear();
        self.keys.clear();
        self.ranges.clear();
        self.clamped = 0;
        self.determinant_culled = 0;
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
        let (mut counters, clamped, determinant_culled) =
            project_scene::<T, K>(scene, camera, &self.settings, &mut self.screen);
        let pairs = bin_and_sort::<T, K>(
            &self.screen,
            &grid,
            <K::Blend as BlendMode<T>>::ORDER_DEPENDENT,
            self.config.sort,
            &mut self.keys,
            &mut self.ranges,
        );
        counters.add(Counter::TileAtomPairs, pairs);
        counters.add(Counter::SortKeys, pairs);
        self.grid = Some(grid);
        self.camera = Some(*camera);
        self.bound_epoch = Some(scene.epoch());
        self.clamped = clamped;
        self.determinant_culled = determinant_culled;
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
        let mut tiles = fb.split_tiles_mut(&grid)?;
        let screen = &self.screen;
        let keys = &self.keys;
        let ranges = &self.ranges;
        let settings = &self.settings;

        let shade = move |batch: &mut Vec<BatchEntry<T, K>>,
                          slot: usize,
                          acc: &mut TileAccum<'_, T, K::Blend>|
         -> P {
            let mut probe = P::for_tile(acc.rect);
            let range = match ranges.get(slot) {
                Some(r) => *r,
                None => (0, 0),
            };
            gather_batch(screen, keys, range, batch);
            if !batch.is_empty() {
                render_tile::<T, K, P>(batch, acc.rect, &camera, settings, acc, &mut probe);
            }
            probe
        };
        let empty_batch = Vec::<BatchEntry<T, K>>::new;

        let probes: Vec<P> = if settings.threads == 1 {
            let mut batch = empty_batch();
            tiles
                .iter_mut()
                .enumerate()
                .map(|(i, acc)| shade(&mut batch, i, acc))
                .collect()
        } else if settings.threads == 0 {
            tiles
                .par_iter_mut()
                .enumerate()
                .map_init(empty_batch, |batch, (i, acc)| shade(batch, i, acc))
                .collect()
        } else {
            match self.pool.as_ref() {
                Some(pool) => pool.install(|| {
                    tiles
                        .par_iter_mut()
                        .enumerate()
                        .map_init(empty_batch, |batch, (i, acc)| shade(batch, i, acc))
                        .collect()
                }),
                None => tiles
                    .par_iter_mut()
                    .enumerate()
                    .map_init(empty_batch, |batch, (i, acc)| shade(batch, i, acc))
                    .collect(),
            }
        };
        drop(tiles);

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
        let tx = (px / grid.tile) as i32;
        let ty = (py / grid.tile) as i32;
        let slot = grid.tile_id(tx, ty) as usize;
        let range = match self.ranges.get(slot) {
            Some(r) => *r,
            None => (0, 0),
        };
        let mut batch: Vec<BatchEntry<T, K>> = Vec::new();
        gather_batch(&self.screen, &self.keys, range, &mut batch);
        let mut cell = <K::Blend as BlendMode<T>>::init();
        let mut probe = NoProbe;
        shade_pixel::<T, K, NoProbe>(
            &batch,
            camera.sample_point(px, py),
            0,
            &self.settings,
            &mut cell,
            &mut probe,
        );
        Ok(cell)
    }

    fn clamped_atoms(&self) -> u64 {
        self.clamped
    }

    fn determinant_culled_atoms(&self) -> u64 {
        self.determinant_culled
    }

    fn resident_bytes(&self) -> usize {
        self.screen.len() * core::mem::size_of::<ScreenAtom<T, K>>()
            + self.keys.len() * core::mem::size_of::<SortKey>()
            + self.ranges.len() * core::mem::size_of::<(u32, u32)>()
    }

    fn limit_depth(&mut self, limit: Option<&DepthImage<T>>) -> Result<()> {
        match limit {
            None => Ok(()),
            Some(_) => Err(LightsError::DepthLimitUnsupported {
                method: MethodId::SplatRaster,
            }),
        }
    }
}
