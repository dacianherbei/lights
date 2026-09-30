#![cfg(feature = "fit")]
use lights::fit::content::{
    CONTROL_ABSENT, Content, N1_DISTANCE_FLOOR, N1_STOPS, N2_BLUR_SIGMA, N2_BLUR_SIGMAS, N2_CENTRE,
    N2_SHADOW_OFFSET, N2_SHADOW_STRENGTH, N3_LATTICES, N4_EDGES, RASTER_HEIGHT, RASTER_WIDTH,
    blur_separable, gaussian_taps, n1_field, n1_sample, n2_coverage, n2_field, n2_shadow_plane,
    n3_field, n4_field, n4_region, pixel_sample,
};
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
use lights::fit::greedy::{FitRequest, fit, fit_budget};
use lights::fit::target::{FeatureId, Part, Primitive, Target};
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

const N1_CHECKSUM: u64 = 0x4d29_1b27_3b8a_15bb;
const N2_CHECKSUM: u64 = 0xf37f_9983_14f4_b4bc;
const N3_CHECKSUM: u64 = 0x9294_ad8f_c462_1374;
const N4_CHECKSUM: u64 = 0x4cff_d3db_fbbe_1368;

fn checksum(image: &FloatImage<f64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in image.pixels.iter() {
        for bits in [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()] {
            for byte in bits.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

fn identical(a: &FloatImage<f64>, b: &FloatImage<f64>) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.pixels.iter().zip(b.pixels.iter()).all(|(p, q)| {
            p.x.to_bits() == q.x.to_bits()
                && p.y.to_bits() == q.y.to_bits()
                && p.z.to_bits() == q.z.to_bits()
        })
}

#[test]
fn every_generator_is_bit_reproducible_and_pinned_to_a_checksum() {
    let cases: [(&str, fn() -> FloatImage<f64>, u64); 4] = [
        ("n1", n1_field, N1_CHECKSUM),
        ("n2", n2_field, N2_CHECKSUM),
        ("n3", n3_field, N3_CHECKSUM),
        ("n4", n4_field, N4_CHECKSUM),
    ];
    for (name, generate, expected) in cases {
        let first = generate();
        let second = generate();
        assert!(
            identical(&first, &second),
            "{name} did not generate the same field twice: a class whose numbers move \
             between calls cannot carry a reproducible N-term curve"
        );
        assert_eq!(
            first.width, RASTER_WIDTH,
            "{name} must be generated at the declared width"
        );
        assert_eq!(
            first.height, RASTER_HEIGHT,
            "{name} must be generated at the declared height"
        );
        assert_eq!(
            checksum(&first),
            expected,
            "{name} no longer hashes to the value recorded when the generation rule was \
             written down. The rule was pre-registered \
             BEFORE the first number was produced, and retuning a class after seeing results \
             is exactly the move that would void the experiment, so this constant is here to \
             make such a change impossible to land silently: got 0x{:016x}",
            checksum(&first)
        );
        for p in first.pixels.iter() {
            assert!(
                p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                "{name} produced a non-finite pixel"
            );
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for (_, _, expected) in cases {
        assert!(
            seen.insert(expected),
            "two classes hash to the same value, so the checksum instrument is measuring \
             nothing"
        );
    }
}

#[test]
fn the_shepard_field_weights_by_inverse_square_distance_and_nothing_else() {
    for (px, py) in [(0u32, 0u32), (17, 5), (31, 31), (63, 63), (40, 22)] {
        let p = pixel_sample(px, py);
        let mut weight = 0.0f64;
        let mut acc = Rgb::<f64>::zero();
        for (x, y, r, g, b) in N1_STOPS {
            let dx = p.x - x;
            let dy = p.y - y;
            let w = 1.0 / (dx * dx + dy * dy).max(N1_DISTANCE_FLOOR);
            weight += w;
            acc = acc + Rgb::new(r, g, b) * w;
        }
        let expected = acc * (1.0 / weight);
        let got = n1_sample(p);
        assert_eq!(
            (got.x.to_bits(), got.y.to_bits(), got.z.to_bits()),
            (
                expected.x.to_bits(),
                expected.y.to_bits(),
                expected.z.to_bits()
            ),
            "the colour field at ({px},{py}) is not the inverse-SQUARE Shepard sum this test \
             computes independently. An inverse-distance or inverse-cube power would still \
             produce a smooth multi-stop field and would still look right in a strip, which \
             is why the exponent is pinned bit for bit rather than by eye"
        );
    }
}

#[test]
fn the_colour_field_stays_inside_the_hull_of_its_own_stops() {
    let image = n1_field();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (_, _, r, g, b) in N1_STOPS {
        for (index, v) in [r, g, b].iter().enumerate() {
            lo[index] = lo[index].min(*v);
            hi[index] = hi[index].max(*v);
        }
    }
    for p in image.pixels.iter() {
        for (index, v) in [p.x, p.y, p.z].iter().enumerate() {
            assert!(
                *v >= lo[index] - 1.0e-12 && *v <= hi[index] + 1.0e-12,
                "a Shepard interpolant is a convex combination of its stops, so channel \
                 {index} may not leave [{}, {}] and it reached {v}",
                lo[index],
                hi[index]
            );
        }
    }
    let stop = N1_STOPS[2];
    let near = n1_sample(Vec2::new(stop.0 + 1.0e-4, stop.1));
    assert!(
        (near.x - stop.2).abs() < 1.0e-4
            && (near.y - stop.3).abs() < 1.0e-4
            && (near.z - stop.4).abs() < 1.0e-4,
        "arbitrarily close to a stop the field must approach that stop's colour, got {near:?}"
    );
}

#[test]
fn the_blur_is_symmetric_about_its_own_centre() {
    let taps = gaussian_taps(N2_BLUR_SIGMA, N2_BLUR_SIGMAS);
    let radius = (taps.len() - 1) / 2;
    assert_eq!(
        radius,
        (N2_BLUR_SIGMA * N2_BLUR_SIGMAS).ceil() as usize,
        "the truncated kernel must carry the declared radius"
    );
    let total: f64 = taps.iter().sum();
    assert!(
        (total - 1.0).abs() < 1.0e-12,
        "the taps must be renormalised over the window that was actually kept, got {total}"
    );
    for i in 0..=radius {
        assert_eq!(
            taps[radius - i].to_bits(),
            taps[radius + i].to_bits(),
            "tap {i} either side of the centre must be bit-identical"
        );
    }

    let side = 33u32;
    let centre = (side / 2) as usize;
    let mut plane = vec![0.0f64; (side as usize) * (side as usize)];
    plane[centre * (side as usize) + centre] = 1.0;
    let out = blur_separable(&plane, side, side, &taps);
    for y in 0..side as usize {
        for x in 0..side as usize {
            let mirrored = (2 * centre - y) * (side as usize) + (2 * centre - x);
            assert_eq!(
                out[y * (side as usize) + x].to_bits(),
                out[mirrored].to_bits(),
                "the blur of a centred impulse must be symmetric through its own centre at \
                 ({x},{y}); a one-sided window would still look like a blur and would move \
                 the shadow off its declared offset"
            );
        }
    }
    let mass: f64 = out.iter().sum();
    assert!(
        (mass - 1.0).abs() < 1.0e-12,
        "an impulse well inside the field must keep its mass, got {mass}"
    );
}

#[test]
fn the_soft_shadow_is_soft_and_the_rectangle_over_it_is_not() {
    let image = n2_field();
    let rect = n2_coverage(Vec2::new(N2_CENTRE.0, N2_CENTRE.1));
    let crisp_shadow = n2_coverage(Vec2::new(
        N2_CENTRE.0 + N2_SHADOW_OFFSET.0,
        N2_CENTRE.1 + N2_SHADOW_OFFSET.1,
    ));
    let blurred = n2_shadow_plane();
    let width = RASTER_WIDTH as usize;

    let mut shadow_step = 0.0f64;
    let mut rect_step = 0.0f64;
    let mut pairs = 0usize;
    for y in 0..RASTER_HEIGHT {
        for x in 0..RASTER_WIDTH - 1 {
            let left = (y as usize) * width + (x as usize);
            let right = left + 1;
            let step = (image.get(x + 1, y).x - image.get(x, y).x).abs();
            if rect[left] > 0.0 || rect[right] > 0.0 {
                rect_step = rect_step.max(step);
                continue;
            }
            pairs += 1;
            shadow_step = shadow_step.max(step);
        }
    }
    assert!(
        pairs > 2000,
        "only {pairs} horizontal pairs fall outside the rectangle, so this test is barely \
         looking at the shadow"
    );
    assert!(
        shadow_step > 0.02,
        "the shadow must actually be present and varying, largest step {shadow_step}"
    );
    assert!(
        shadow_step < 0.15,
        "outside the rectangle the only feature is the blurred shadow, so no two \
         neighbouring pixels may differ by {shadow_step}. An unblurred shadow steps by about \
         {:.3} here, which is what this bound exists to refuse",
        crisp_shadow
            .iter()
            .zip(crisp_shadow.iter().skip(1))
            .map(|(a, b)| (b - a).abs())
            .fold(0.0f64, f64::max)
            * N2_SHADOW_STRENGTH
            * (0.93 - 0.08)
    );
    assert!(
        rect_step > 0.25,
        "the rectangle composited on top is crisp by construction and must still step hard, \
         largest step {rect_step}"
    );

    let softest = blurred
        .iter()
        .zip(blurred.iter().skip(1))
        .map(|(a, b)| (b - a).abs())
        .fold(0.0f64, f64::max);
    let hardest = crisp_shadow
        .iter()
        .zip(crisp_shadow.iter().skip(1))
        .map(|(a, b)| (b - a).abs())
        .fold(0.0f64, f64::max);
    assert!(
        softest < hardest / 5.0,
        "the blurred shadow plane steps by {softest} and the unblurred one by {hardest}: the \
         blur is supposed to be the whole difference between them"
    );
}

#[test]
fn the_band_limited_lattice_period_is_the_finest_declared_lattice() {
    let period = N3_LATTICES.iter().copied().min().unwrap_or(1);
    assert_eq!(period, 8, "the finest declared lattice is the one measured here");
    let image = n3_field();
    let mut sums = vec![0.0f64; period as usize];
    let mut counts = vec![0.0f64; period as usize];
    for y in 0..RASTER_HEIGHT {
        for x in 1..RASTER_WIDTH - 1 {
            let slope = (image.get(x + 1, y).x - image.get(x - 1, y).x).abs() * 0.5;
            let phase = (((x as f64) + 0.5) % (period as f64)).floor() as usize;
            sums[phase] += slope;
            counts[phase] += 1.0;
        }
    }
    let mean: Vec<f64> = sums
        .iter()
        .zip(counts.iter())
        .map(|(s, c)| s / c)
        .collect();
    let at_node = (mean[(period as usize) - 1] + mean[0]) * 0.5;
    let at_midpoint = (mean[(period as usize) / 2 - 1] + mean[(period as usize) / 2]) * 0.5;
    assert!(
        at_node < 0.65 * at_midpoint,
        "a smoothstep interpolant has zero slope AT a lattice node and its steepest slope \
         halfway between two, so binning the measured slope by position modulo {period} must \
         show a trough at the nodes: node {at_node:.6} against midpoint {at_midpoint:.6}. A \
         linear interpolant would flatten this profile, and a different lattice would move \
         the trough. Full profile: {mean:?}"
    );
    for value in mean.iter() {
        assert!(*value > 0.0, "the field must vary along x at every phase");
    }
}

#[test]
fn the_four_occlusion_edges_are_where_they_were_placed() {
    let step = 0.1f64;
    let along = [-0.5f64, -0.25, 0.25, 0.5];
    for (index, (degrees, ax, ay)) in N4_EDGES.iter().enumerate() {
        let radians = degrees.to_radians();
        let (sin, cos) = radians.sin_cos();
        let direction = Vec2::new(cos, sin);
        let normal = Vec2::new(-sin, cos);
        let anchor = Vec2::new(*ax, *ay);

        let mut nearest = f64::INFINITY;
        for (other, (other_degrees, ox, oy)) in N4_EDGES.iter().enumerate() {
            if other == index {
                continue;
            }
            let (other_sin, other_cos) = other_degrees.to_radians().sin_cos();
            let other_normal = Vec2::new(-other_sin, other_cos);
            let offset = Vec2::new(anchor.x - ox, anchor.y - oy);
            nearest = nearest.min(offset.dot(other_normal).abs());
        }
        let reach = along.iter().fold(0.0f64, |a, t| a.max(t.abs())) + step;
        assert!(
            nearest > reach + 0.4,
            "edge {index} sits {nearest:.4} from the nearest other edge line and this test \
             probes {reach:.4} away from its anchor: any closer and a flip it saw could \
             belong to a different edge, so the instrument would be measuring the wrong thing"
        );

        for t in [0.0].iter().chain(along.iter()) {
            let base = Vec2::new(anchor.x + direction.x * t, anchor.y + direction.y * t);
            let outward = n4_region(Vec2::new(
                base.x + normal.x * step,
                base.y + normal.y * step,
            ));
            let inward = n4_region(Vec2::new(
                base.x - normal.x * step,
                base.y - normal.y * step,
            ));
            assert!(
                outward != inward,
                "edge {index} was declared at {degrees} degrees through ({ax}, {ay}); \
                 crossing it {step} either side at offset {t} along its own direction must \
                 switch which noise field is visible, and it did not. The direction and the \
                 normal are recomputed here from the declared angle rather than read back \
                 from the code under test"
            );
        }
    }

    let mut near = 0usize;
    for py in 0..RASTER_HEIGHT {
        for px in 0..RASTER_WIDTH {
            if n4_region(pixel_sample(px, py)) {
                near += 1;
            }
        }
    }
    let total = (RASTER_WIDTH * RASTER_HEIGHT) as usize;
    assert!(
        near > total / 4 && near < total * 3 / 4,
        "the four edges must actually partition the patch; {near} of {total} pixels took the \
         near field"
    );

    let image = n4_field();
    let mut crossings = 0usize;
    for y in 0..RASTER_HEIGHT {
        for x in 0..RASTER_WIDTH - 1 {
            if n4_region(pixel_sample(x, y)) != n4_region(pixel_sample(x + 1, y))
                && (image.get(x + 1, y).x - image.get(x, y).x).abs() > 0.05
            {
                crossings += 1;
            }
        }
    }
    assert!(
        crossings > 20,
        "only {crossings} of the horizontal occlusion crossings carry a hard step: the two \
         noise fields are supposed to be independently seeded, so the switch must be visible"
    );
}

#[test]
fn the_new_classes_declare_no_analytic_control_instead_of_inventing_one() {
    let all = Content::all();
    assert_eq!(all.len(), 8, "the run is eight content classes");
    let mut names = std::collections::BTreeSet::new();
    for content in all.iter() {
        assert!(
            names.insert(content.name().to_string()),
            "two classes share the name {}",
            content.name()
        );
    }
    for content in Content::carried_forward() {
        assert!(
            content.analytic_control().is_some(),
            "{} is an SDF class carried forward and keeps its exact control arm",
            content.name()
        );
        assert_ne!(content.control_label(), CONTROL_ABSENT);
    }
    for content in Content::newly_generated() {
        assert!(
            content.analytic_control().is_none(),
            "{} has no shape description an SDF could render exactly, so manufacturing a \
             control for it would reintroduce the very bias the new classes exist to escape",
            content.name()
        );
        assert_eq!(
            content.control_label(),
            CONTROL_ABSENT,
            "the missing control must be printed as the literal ABSENT"
        );
    }
}

#[test]
fn every_class_hands_back_a_truth_image_of_its_own_viewport() {
    for content in Content::all() {
        let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
        let truth = match content.truth(DEFAULT_MAX_DEPTH, &mut budget) {
            Ok(t) => t,
            Err(e) => panic!("{} produced no truth image: {e}", content.name()),
        };
        let (width, height) = content.viewport();
        assert_eq!((truth.width, truth.height), (width, height));
        let scaffold = content.fit_scaffold();
        assert_eq!(
            scaffold.viewport,
            content.viewport(),
            "{} hands the fitter a scaffold of the wrong size",
            content.name()
        );
        let (lo, hi) = truth.min_max_channel();
        assert!(
            lo.is_finite() && hi.is_finite() && hi > lo,
            "{} produced a flat or non-finite truth image [{lo}, {hi}]",
            content.name()
        );
        assert!(
            width.is_power_of_two() && height.is_power_of_two(),
            "{} is {width}x{height}: a viewport that is not a power of two would force the \
             transform arms to pad, and the padding would be a confound neither family asked \
             for",
            content.name()
        );
    }
}

#[test]
fn the_fitter_reads_only_the_viewport_of_a_raster_scaffold() {
    let content = Content::n1_colour_field();
    let mut budget = coverage_budget(DEFAULT_CELL_LIMIT);
    let truth = match content.truth(DEFAULT_MAX_DEPTH, &mut budget) {
        Ok(t) => t,
        Err(e) => panic!("truth: {e}"),
    };
    let plain = content.fit_scaffold();
    let decorated = Target::ink_on_paper(
        "decorated",
        plain.viewport,
        vec![Part {
            id: FeatureId(0),
            shape: Primitive::Disc {
                centre: Vec2::new(19.0, 41.0),
                radius: 7.0,
            },
        }],
        Vec::new(),
        Vec::new(),
    );

    let mut request = FitRequest::new(
        KernelFamily::Gaussian,
        SharedSettings::<f64>::prefiltered(),
    );
    request.max_atoms = 1;
    request.refine_iterations = 1;
    request.refine_tail = 0;

    let mut first_budget = fit_budget(200_000);
    let first = match fit(&plain, &truth, &request, &mut first_budget) {
        Ok(o) => o,
        Err(e) => panic!("fitting the empty scaffold: {e}"),
    };
    let mut second_budget = fit_budget(200_000);
    let second = match fit(&decorated, &truth, &request, &mut second_budget) {
        Ok(o) => o,
        Err(e) => panic!("fitting the decorated scaffold: {e}"),
    };

    assert!(!first.atoms.is_empty(), "the fit placed no atom to compare");
    assert_eq!(
        first.atoms.iter().map(|a| a.bits()).collect::<Vec<_>>(),
        second.atoms.iter().map(|a| a.bits()).collect::<Vec<_>>(),
        "a raster class has no shape description, so it hands the unchanged fitter a \
         scaffold that carries nothing but the viewport. Two scaffolds differing only in \
         their primitives must fit identically, or the fitter is consulting geometry that \
         the four new classes do not have"
    );
}
