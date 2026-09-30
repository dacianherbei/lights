use crate::error::{LightsError, Result};
use crate::fit::analytic::{AnalyticArm, render_analytic};
use crate::fit::atom2d::{Atom2d, camera};
use crate::fit::content::{CONTROL_ABSENT, Content};
use crate::fit::coverage::coverage_budget;
use crate::fit::greedy::{FitRequest, Reading, fit, fit_budget, prefix, render_or_background};
use crate::fit::nterm::{Arm, NTermRequest, positions, relative_l2};
use crate::fit::strip::horizontal_strip;
use crate::fit::sweep::{self, Concurrency};
use crate::fit::transform::{Ranking, Transform, max_levels, truncate_and_reconstruct};
use crate::image::FloatImage;
use crate::kernel::KernelFamily;
use crate::numerics::vector::Rgb;

pub const ABSENT_HATCH_PERIOD: u32 = 4;
pub const ABSENT_INK: (f64, f64, f64) = (1.0, 0.0, 1.0);
pub const ABSENT_GROUND: (f64, f64, f64) = (0.0, 0.0, 0.0);
pub const SEPARATOR: (f64, f64, f64) = (0.25, 0.0, 0.35);
pub const TRUTH_LABEL: &str = "truth";
pub const ZOOMS: [u32; 2] = [1, 4];

pub fn rgb(triple: (f64, f64, f64)) -> Rgb<f64> {
    Rgb::new(triple.0, triple.1, triple.2)
}

pub fn separator() -> Rgb<f64> {
    rgb(SEPARATOR)
}

pub fn absent_panel(width: u32, height: u32) -> FloatImage<f64> {
    let ink = rgb(ABSENT_INK);
    let ground = rgb(ABSENT_GROUND);
    let mut image = FloatImage::<f64>::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let band = ((x + y) / ABSENT_HATCH_PERIOD) % 2;
            image.set(x, y, if band == 0 { ink } else { ground });
        }
    }
    image
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Column {
    Truth,
    Control,
    Fitted(Arm),
}

impl Column {
    pub fn name(self) -> String {
        match self {
            Column::Truth => TRUTH_LABEL.to_string(),
            Column::Control => "control".to_string(),
            Column::Fitted(arm) => arm.name(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TermCount {
    pub requested: usize,
    pub rendered: usize,
    pub available: usize,
}

impl TermCount {
    pub fn resolve(requested: usize, available: usize) -> Self {
        Self {
            requested,
            rendered: requested.min(available),
            available,
        }
    }

    pub fn at_ceiling(self) -> bool {
        self.requested > self.available
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    pub column: Column,
    pub label: String,
    pub terms: Option<TermCount>,
    pub reading: Reading,
    pub relative_l2: Option<f64>,
    pub absent: bool,
    pub has_fitter: bool,
    pub exhausted: Option<bool>,
    pub image: FloatImage<f64>,
}

impl Panel {
    pub fn at_own_ceiling(&self) -> bool {
        match self.terms {
            None => false,
            Some(t) => t.rendered == t.available,
        }
    }

    pub fn budget_stop(&self) -> bool {
        self.at_own_ceiling() && self.exhausted == Some(false)
    }

    pub fn stop(&self) -> &'static str {
        if !self.at_own_ceiling() {
            return "";
        }
        match self.exhausted {
            Some(true) => " (fitter exhausted)",
            Some(false) => " (BUDGET STOP)",
            None => "",
        }
    }

    pub fn caption(&self) -> String {
        let terms = match self.terms {
            None => String::new(),
            Some(t) => {
                if t.at_ceiling() {
                    format!(
                        " n={} (CEILING, {} requested)",
                        t.rendered, t.requested
                    )
                } else {
                    format!(" n={}", t.rendered)
                }
            }
        };
        let reading = match self.reading {
            Reading::Prefix => " PREFIX",
            Reading::OwnFit => "",
        };
        let error = match self.relative_l2 {
            None => String::new(),
            Some(e) => format!(" relL2={e:.4}"),
        };
        format!("{}{terms}{reading}{}{error}", self.label, self.stop())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TermChoice {
    SharedCeiling,
    Explicit(usize),
}

impl TermChoice {
    pub fn name(self) -> String {
        match self {
            TermChoice::SharedCeiling => "shared".to_string(),
            TermChoice::Explicit(n) => format!("{n}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedArm {
    Atoms {
        kernel: KernelFamily,
        atoms: Vec<Atom2d>,
        improvement_exhausted: bool,
    },
    Wavelet {
        transform: Transform,
        ranking: Ranking,
        levels: u32,
        available: usize,
    },
}

impl PreparedArm {
    pub fn exhausted(&self) -> Option<bool> {
        match self {
            PreparedArm::Atoms {
                improvement_exhausted,
                ..
            } => Some(*improvement_exhausted),
            PreparedArm::Wavelet { .. } => None,
        }
    }

    pub fn available(&self) -> usize {
        match self {
            PreparedArm::Atoms { atoms, .. } => atoms.len(),
            PreparedArm::Wavelet { available, .. } => *available,
        }
    }
}

pub fn prepare_arm(
    content: &Content,
    truth: &FloatImage<f64>,
    arm: Arm,
    request: &NTermRequest,
) -> Result<PreparedArm> {
    match arm {
        Arm::Atoms(kernel) => {
            let scaffold = content.fit_scaffold();
            let mut fit_request = FitRequest::new(kernel, request.settings);
            fit_request.max_atoms = request.max_atoms;
            fit_request.tile = request.tile;
            fit_request.refine_iterations = request.refine_iterations;
            fit_request.refine_tail = request.refine_tail;
            fit_request.initialisation = request.initialisation;
            let mut budget = fit_budget(request.render_limit);
            let outcome = fit(&scaffold, truth, &fit_request, &mut budget)?;
            Ok(PreparedArm::Atoms {
                kernel,
                atoms: outcome.atoms,
                improvement_exhausted: outcome.improvement_exhausted,
            })
        }
        Arm::Wavelet(transform, ranking) => {
            let (width, height) = content.viewport();
            let levels = match request.levels {
                Some(l) => l,
                None => max_levels(width, height),
            };
            Ok(PreparedArm::Wavelet {
                transform,
                ranking,
                levels,
                available: positions(width, height),
            })
        }
    }
}

pub fn render_prepared(
    prepared: &PreparedArm,
    truth: &FloatImage<f64>,
    viewport: (u32, u32),
    terms: TermCount,
    request: &NTermRequest,
) -> Result<(FloatImage<f64>, Reading)> {
    match prepared {
        PreparedArm::Atoms { kernel, atoms, .. } => {
            let spec = camera(viewport.0, viewport.1);
            let (slice, reading) = prefix(atoms, terms.rendered);
            let image = render_or_background(
                slice,
                &spec,
                *kernel,
                &request.settings,
                request.tile,
            )?;
            Ok((image, reading))
        }
        PreparedArm::Wavelet {
            transform,
            ranking,
            levels,
            ..
        } => Ok((
            truncate_and_reconstruct(*transform, truth, *levels, terms.rendered, *ranking)?,
            Reading::OwnFit,
        )),
    }
}

pub fn truth_panel(truth: &FloatImage<f64>) -> Panel {
    Panel {
        column: Column::Truth,
        label: TRUTH_LABEL.to_string(),
        terms: None,
        reading: Reading::OwnFit,
        relative_l2: None,
        absent: false,
        has_fitter: false,
        exhausted: None,
        image: truth.clone(),
    }
}

pub fn control_panel(
    content: &Content,
    truth: &FloatImage<f64>,
    request: &NTermRequest,
) -> Result<Panel> {
    let (width, height) = content.viewport();
    match content {
        Content::Raster(_) => Ok(Panel {
            column: Column::Control,
            label: CONTROL_ABSENT.to_string(),
            terms: None,
            reading: Reading::OwnFit,
            relative_l2: None,
            absent: true,
            has_fitter: false,
            exhausted: None,
            image: absent_panel(width, height),
        }),
        Content::Analytic(target) => {
            let mut cells = coverage_budget(request.cell_limit);
            let (image, _) = render_analytic(
                target,
                AnalyticArm::Exact,
                request.quadrature_depth,
                &mut cells,
            )?;
            let error = relative_l2(truth, &image)?;
            Ok(Panel {
                column: Column::Control,
                label: AnalyticArm::Exact.name().to_string(),
                terms: None,
                reading: Reading::OwnFit,
                relative_l2: Some(error),
                absent: false,
                has_fitter: false,
                exhausted: None,
                image,
            })
        }
    }
}

pub fn arm_panel(
    prepared: &PreparedArm,
    truth: &FloatImage<f64>,
    viewport: (u32, u32),
    arm: Arm,
    requested: usize,
    request: &NTermRequest,
) -> Result<Panel> {
    let terms = TermCount::resolve(requested, prepared.available());
    let (image, reading) = render_prepared(prepared, truth, viewport, terms, request)?;
    let error = relative_l2(truth, &image)?;
    Ok(Panel {
        column: Column::Fitted(arm),
        label: arm.name(),
        terms: Some(terms),
        reading,
        relative_l2: Some(error),
        absent: false,
        has_fitter: arm.has_fitter(),
        exhausted: prepared.exhausted(),
        image,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassPanels {
    pub content: String,
    pub viewport: (u32, u32),
    pub control: String,
    pub shared_ceiling: usize,
    pub requested: usize,
    pub panels: Vec<Panel>,
}

impl ClassPanels {
    pub fn column_labels(&self) -> Vec<String> {
        self.panels.iter().map(|p| p.column.name()).collect()
    }

    pub fn any_at_ceiling(&self) -> bool {
        self.panels
            .iter()
            .any(|p| p.terms.map(|t| t.at_ceiling()).unwrap_or(false))
    }

    pub fn any_budget_stop(&self) -> bool {
        self.panels.iter().any(|p| p.budget_stop())
    }
}

pub fn class_panels(
    content: &Content,
    arms: &[Arm],
    choice: TermChoice,
    base: &NTermRequest,
) -> Result<ClassPanels> {
    let mut classes = class_sweep(
        std::slice::from_ref(content),
        arms,
        choice,
        base,
        Concurrency::Serial,
    )?;
    match classes.pop() {
        Some(class) => Ok(class),
        None => Err(LightsError::InvalidCamera(
            "class_sweep returned no class for the one content it was given",
        )),
    }
}

pub fn class_sweep(
    contents: &[Content],
    arms: &[Arm],
    choice: TermChoice,
    base: &NTermRequest,
    concurrency: Concurrency,
) -> Result<Vec<ClassPanels>> {
    let truths = sweep::map(concurrency, contents, |content| {
        let mut cells = coverage_budget(base.cell_limit);
        content.truth(base.quadrature_depth, &mut cells)
    })?;

    let mut jobs: Vec<(usize, Arm)> = Vec::with_capacity(contents.len() * arms.len());
    for position in 0..contents.len() {
        for arm in arms.iter() {
            jobs.push((position, *arm));
        }
    }
    let ready = sweep::map(concurrency, &jobs, |(position, arm)| {
        let mut request = base.clone();
        request.arm = *arm;
        prepare_arm(&contents[*position], &truths[*position], *arm, &request)
    })?;

    let positions: Vec<usize> = (0..contents.len()).collect();
    sweep::map(concurrency, &positions, |position| {
        let content = &contents[*position];
        let truth = &truths[*position];
        let viewport = content.viewport();
        let own = &ready[position * arms.len()..(position + 1) * arms.len()];
        let mut shared_ceiling = usize::MAX;
        for prepared in own.iter() {
            shared_ceiling = shared_ceiling.min(prepared.available());
        }
        if own.is_empty() {
            shared_ceiling = 0;
        }
        let requested = match choice {
            TermChoice::SharedCeiling => shared_ceiling,
            TermChoice::Explicit(n) => n,
        };
        let mut panels = vec![truth_panel(truth), control_panel(content, truth, base)?];
        for (arm, prepared) in arms.iter().zip(own.iter()) {
            let mut request = base.clone();
            request.arm = *arm;
            panels.push(arm_panel(
                prepared, truth, viewport, *arm, requested, &request,
            )?);
        }
        Ok(ClassPanels {
            content: content.name().to_string(),
            viewport,
            control: content.control_label().to_string(),
            shared_ceiling,
            requested,
            panels,
        })
    })
}

pub fn strip_for_class(class: &ClassPanels, gap: u32) -> Result<FloatImage<f64>> {
    let images: Vec<FloatImage<f64>> = class.panels.iter().map(|p| p.image.clone()).collect();
    horizontal_strip(&images, gap, separator())
}

pub fn caption(class: &ClassPanels, zoom: u32) -> String {
    let panels: Vec<String> = class.panels.iter().map(|p| p.caption()).collect();
    let scale = if zoom > 1 {
        format!(
            " Upscaling is nearest-neighbour, so every pixel an arm actually produced stays a \
             flat {zoom}x{zoom} square and no value is invented."
        )
    } else {
        String::new()
    };
    let control = if class.control == CONTROL_ABSENT {
        "the analytic column is the magenta hatch and carries no image: this class has no SDF \
         description, so the control is ABSENT and was NOT replaced by a weak invented arm"
    } else {
        "the analytic column is exact box coverage of the description, the ceiling nothing can \
         beat, and it coincides with the truth panel bit for bit"
    };
    let budget = if class.any_budget_stop() {
        " A column marked BUDGET STOP placed every atom the budget allowed and had NOT run out \
         of improvements: its count is the budget, not the representation."
    } else {
        ""
    };
    let ceiling = if class.any_at_ceiling() {
        " A column marked CEILING ran out of terms below the request: that is where the fitter \
         stopped, not where the representation did."
    } else {
        ""
    };
    format!(
        "{} at {zoom}x, {}x{} px source panels, left to right: {}. Purple bars are separators, \
         never image content; {control}.{ceiling}{budget} The transform columns got their ranked \
         optimum for free and have no fitter; the atom columns got greedy matching pursuit, \
         measured as a lower bound.{scale}",
        class.content,
        class.viewport.0,
        class.viewport.1,
        panels.join(" | ")
    )
}
