use crate::camera::Camera;
use crate::error::Result;
use crate::image::{DepthImage, FloatImage, Framebuffer};
use crate::kernel::{CurveShape, Kernel};
use crate::method::{
    ImageFormation, MethodId, RenderMethod, RenderReport, RenderStrategy, create_strategy,
};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::Rgb;
use crate::probe::{CounterProbe, Counters, NoProbe, ProbeSink};
use crate::scene::Scene;
use crate::scene::prepare::PreparedScene;
use crate::settings::SharedSettings;

pub struct Renderer<T: FloatingPoint, K: Kernel<T>> {
    strategy: RenderStrategy<T, K>,
    framebuffer: Framebuffer<T, K::Blend>,
    prepared: PreparedScene<T, K>,
    settings: SharedSettings<T>,
    shape: CurveShape,
    method: MethodId,
    report: RenderReport,
    counters: Counters,
    frames: u64,
}

impl<T: FloatingPoint, K: Kernel<T>> Renderer<T, K> {
    pub fn new(
        method: &RenderMethod,
        settings: &SharedSettings<T>,
        shape: &CurveShape,
        width: u32,
        height: u32,
        scene: &Scene<T>,
    ) -> Result<Self> {
        let prepared = PreparedScene::<T, K>::build_uniform(scene, shape, settings)?;
        let mut strategy = create_strategy::<T, K>(method, settings)?;
        let build = strategy.build(&prepared)?;
        let framebuffer = Framebuffer::<T, K::Blend>::new(width, height, method.tile())?;
        let mut report = RenderReport::default();
        report.build = build;
        report.framebuffer_bytes = framebuffer.resident_bytes();
        Ok(Self {
            strategy,
            framebuffer,
            prepared,
            settings: *settings,
            shape: *shape,
            method: method.id(),
            report,
            counters: Counters::default(),
            frames: 0,
        })
    }

    pub fn set_scene(&mut self, scene: &Scene<T>) -> Result<()> {
        self.prepared = PreparedScene::<T, K>::build_uniform(scene, &self.shape, &self.settings)?;
        self.report.build = self.strategy.build(&self.prepared)?;
        Ok(())
    }

    pub fn limit_depth(&mut self, limit: Option<&DepthImage<T>>) -> Result<()> {
        self.strategy.limit_depth(limit)
    }

    pub fn render_frame(&mut self, camera: &Camera<T>) -> Result<()> {
        self.frame::<NoProbe>(camera)
    }

    pub fn render_frame_counted(&mut self, camera: &Camera<T>) -> Result<Counters> {
        self.frame::<CounterProbe<T>>(camera)?;
        Ok(self.counters)
    }

    fn frame<P: crate::probe::Probe<T>>(&mut self, camera: &Camera<T>) -> Result<()> {
        self.report.bind = self.strategy.bind(&self.prepared, camera)?;
        self.framebuffer.clear();
        let mut sink =
            ProbeSink::<T>::counters_only(self.framebuffer.width(), self.framebuffer.height());
        self.report.render = self
            .strategy
            .render::<P>(&self.prepared, &mut self.framebuffer, &mut sink)?;
        self.report.clamped_atoms = self.strategy.clamped_atoms();
        self.report.determinant_culled_atoms = self.strategy.determinant_culled_atoms();
        self.counters = sink.counters;
        self.frames += 1;
        Ok(())
    }

    pub fn resolve(&self, background: Rgb<T>) -> FloatImage<T> {
        self.framebuffer.resolve_float(background)
    }

    pub fn resolve_premultiplied(&self) -> (FloatImage<T>, FloatImage<T>) {
        self.framebuffer.resolve_premultiplied()
    }

    pub fn framebuffer(&self) -> &Framebuffer<T, K::Blend> {
        &self.framebuffer
    }

    pub fn report(&self) -> &RenderReport {
        &self.report
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn method(&self) -> MethodId {
        self.method
    }

    pub fn resident_bytes(&self) -> usize {
        self.framebuffer.resident_bytes()
            + self.strategy.resident_bytes()
            + self.prepared.resident_bytes()
    }
}
