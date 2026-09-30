pub mod analytic;
pub mod aspect2;
pub mod aspect3;
pub mod atom2d;
pub mod atoms_io;
pub mod backward;
pub mod content;
pub mod coverage;
pub mod descent;
pub mod edge;
pub mod glyph_scale;
pub mod greedy;
pub mod harness;
pub mod metric;
pub mod nterm;
pub mod report;
pub mod scale;
pub mod strip;
pub mod sweep;
pub mod target;
pub mod transform;

pub use analytic::{AnalyticArm, AnalyticCost, render_analytic, sdf_coverage, sdf_coverage_through};
pub use atom2d::{
    Atom2d, DZ, ORTHOGRAPHIC_FOV_DEGREES, Z0, camera, depth_of, image_to_world, pixel_centre,
    shifted_camera, to_atom, to_scene, world_to_image,
};
pub use coverage::{
    CELL_DIAGONAL, CoverageError, DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, box_line_fraction,
    coverage_budget, coverage_image_to_colour, exact_coverage,
};
pub use descent::{
    ADAM_BETA1, ADAM_BETA2, ADAM_DECAY_LEVELS, ADAM_EPSILON, ADAM_RATE, ADAM_RATE_SWEEP,
    ADAM_STEPS, AdamState, CHOSEN_STEP,
    DEFAULT_DESCENT_BUDGET, DIRECTION_SEED, DescentTrace, GradientSource, LINE_SEARCH_HALVINGS,
    LineStep, P, adam_with, descend_with, fit_with, gradient_from,
    RENDERS_PER_GRADIENT, STEP_SWEEP, SweepRow as StepSweepRow, adam, adam_at_gain, box_of,
    clamp_to, descend,
    descent_budget, directional_difference,
    fit as descent_fit, fit_budget as descent_fit_budget, gradient, line_search, loss_at, pack,
    parameter_name, parameter_scales, projected_descent, step_sweep, unit_direction, unpack,
};
pub use edge::{
    BIN_WIDTH, Criterion, EdgeMetric, HIGH_LEVEL, INTERIOR_DEPTH, LOW_LEVEL, Mapping, PLATEAU_FAR,
    PLATEAU_NEAR, ProfileBin, SDF_ARM_TRUE_FWHM, SDF_ARM_TRUE_TRANSITION, THIN_BIN_SAMPLES, TargetMetric, measure, measure_mapped, measure_shifted,
    measure_target, measure_target_mapped, measure_target_shifted,
};
pub use harness::{
    CUTOFF_SWEEP, DILATION_SWEEP, FIELD_COPIES, FIELD_SIZE, GRID_SHIFT, ProbeRequest,
    ProfileChoice, TileCost, run, tile_cost, tiled_atoms,
};
pub use sweep::{Concurrency, map as sweep_map};
pub use strip::{DEFAULT_GAP, DEFAULT_ZOOM, horizontal_strip, upscale};
pub use report::{
    BIN_COLUMNS, BinRow, Header, OVERFIT_FLAG_PX, PROBE_COLUMNS, ProbeRow, ROW_COLUMNS, Report, Row, SENSITIVITY_COLUMNS, SWEEP_COLUMNS,
    SensitivityRow, Sheet, SweepRow, never_met, overfit_flagged, print_report, write_report,
};
pub use greedy::{
    CENTRE_STEP, COLOUR_STEP, DEFAULT_FIT_BUDGET, DEFAULT_REFINE_ITERATIONS, DEFAULT_REFINE_TAIL,
    ELONGATION_GAIN, FitOutcome, FitRequest, FitStep, Initialisation, LADDER_LEVELS,
    LOG_SCALE_STEP, MAX_LOG_SCALE, MAX_OPACITY, MIN_LOG_SCALE, PREFIX_SHORTFALL, Reading,
    SEED_SIGMA, TENSOR_RADIUS,
    UNIT_STEP_SCALE, background_image, centre_step, fit, fit_budget, log_scale_bounds,
    prefix, render_atoms, render_or_background, residual_energy, residual_peak, seed_atom,
    seed_atom_scaled, seed_pixel, seed_sigma, splat_method, structure_tensor, tensor_axis,
};
pub use scale::{
    Convention, Judgement, MAX_SCALE_NUMERATOR, MAX_SUPERSAMPLE, SUPERSAMPLES_PER_SCENE_UNIT,
    Sampler, Scale, ScaleClass, Similarity2, box_average, judge_fwhm, judge_transition, measure_scene,
    scaled_atoms, scaled_primitive, scaled_target,
};
pub use target::{
    CORNER_DEGREES, CURVE_RADIUS, EDGE_DEGREES, EdgeProbe, FeatureId, GLYPH_EM,
    GLYPH_NOMINAL_PIVOT, GLYPH_TURN_DEGREES, Part, Primitive, ProbeKind, SUBPIXEL_X, SUBPIXEL_Y,
    STROKE_HALF_WIDTH, Side, Target, curve_centre, glyph_pivot, glyph_point,
};
