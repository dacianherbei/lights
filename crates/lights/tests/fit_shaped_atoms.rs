#![cfg(feature = "fit")]
use lights::agreement::ContractViolation;
use lights::dispatch::Collect;
use lights::error::LightsError;
use lights::fit::atom2d::{Atom2d, ShapedAtom2d, camera};
use lights::fit::greedy::{render_atoms, render_shaped_atoms};
use lights::image::FloatImage;
use lights::kernel::{Beta, CurveShape, KernelFamily};
use lights::numerics::vector::{Rgb, Vec2};
use lights::settings::SharedSettings;

const SIDE: u32 = 32;
const TILE: u32 = 16;
const KERNEL: KernelFamily = KernelFamily::SuperGaussian(Beta::One);

fn pair() -> [Atom2d; 2] {
    [
        Atom2d::from_sigma(
            Vec2::new(10.0, 16.0),
            Vec2::new(3.0, 2.0),
            0.3,
            0.9,
            Rgb::new(0.9, 0.2, 0.1),
        ),
        Atom2d::from_sigma(
            Vec2::new(22.0, 16.0),
            Vec2::new(2.5, 3.5),
            -0.4,
            0.8,
            Rgb::new(0.1, 0.3, 0.9),
        ),
    ]
}

fn worst_gap(a: &FloatImage<f64>, b: &FloatImage<f64>) -> f64 {
    let mut worst = 0.0f64;
    for (p, q) in a.pixels.iter().zip(b.pixels.iter()) {
        worst = worst
            .max((p.x - q.x).abs())
            .max((p.y - q.y).abs())
            .max((p.z - q.z).abs());
    }
    worst
}

fn plain(atoms: &[Atom2d]) -> FloatImage<f64> {
    let settings = SharedSettings::<f64>::prefiltered();
    match render_atoms(atoms, &camera(SIDE, SIDE), KERNEL, &settings, TILE, Collect::Pixels) {
        Ok(o) => o.image,
        Err(e) => panic!("the plain fit render failed: {e}"),
    }
}

fn shaped(atoms: &[ShapedAtom2d], collect: Collect) -> Result<FloatImage<f64>, LightsError> {
    let settings = SharedSettings::<f64>::prefiltered();
    render_shaped_atoms(atoms, &camera(SIDE, SIDE), KERNEL, &settings, TILE, collect).map(|o| o.image)
}

fn column_band(image: &FloatImage<f64>, x0: u32, x1: u32) -> Vec<Rgb<f64>> {
    let mut out = Vec::new();
    for y in 0..image.height {
        for x in x0..x1 {
            out.push(image.get(x, y));
        }
    }
    out
}

fn band_gap(a: &[Rgb<f64>], b: &[Rgb<f64>]) -> f64 {
    let mut worst = 0.0f64;
    for (p, q) in a.iter().zip(b.iter()) {
        worst = worst
            .max((p.x - q.x).abs())
            .max((p.y - q.y).abs())
            .max((p.z - q.z).abs());
    }
    worst
}

#[test]
fn the_fitter_renders_each_shaped_atom_with_its_own_curve() {
    let atoms = pair();
    let uniform = plain(&atoms);
    let mixed = [
        ShapedAtom2d::new(atoms[0], CurveShape::with_beta(Beta::Eight.value())),
        ShapedAtom2d::new(atoms[1], KERNEL.shape()),
    ];
    let image = match shaped(&mixed, Collect::Pixels) {
        Ok(i) => i,
        Err(e) => panic!("the shaped fit render failed: {e}"),
    };
    let left = band_gap(&column_band(&image, 0, 8), &column_band(&uniform, 0, 8));
    let right = band_gap(&column_band(&image, 26, SIDE), &column_band(&uniform, 26, SIDE));
    assert!(
        left > 0.01,
        "atom 0 carries beta 8 and its side of the frame matched the uniform beta-1 render \
         (gap {left}), so the fitter's render path dropped the per-atom shape list"
    );
    assert!(
        right < 1e-3,
        "atom 1 carries the kernel's own curve, so the far side of atom 1 must match the \
         uniform render; gap {right} means atom 0's shape leaked onto atom 1"
    );
}

#[test]
fn a_shaped_list_repeating_the_kernel_curve_is_bit_identical_to_the_plain_fit() {
    let atoms = pair();
    let uniform = plain(&atoms);
    let same: Vec<ShapedAtom2d> = atoms
        .iter()
        .map(|a| ShapedAtom2d::new(*a, KERNEL.shape()))
        .collect();
    let image = match shaped(&same, Collect::Pixels) {
        Ok(i) => i,
        Err(e) => panic!("the shaped fit render failed: {e}"),
    };
    assert_eq!(worst_gap(&image, &uniform), 0.0);
    for (p, q) in image.pixels.iter().zip(uniform.pixels.iter()) {
        assert_eq!(
            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()),
            (q.x.to_bits(), q.y.to_bits(), q.z.to_bits())
        );
    }
}

#[test]
fn a_tolerance_audit_refuses_a_shaped_atom_whose_curve_is_not_gaussian() {
    let atoms = pair();
    let gaussian: Vec<ShapedAtom2d> = atoms
        .iter()
        .map(|a| ShapedAtom2d::new(*a, CurveShape::gaussian()))
        .collect();
    if let Err(e) = shaped(&gaussian, Collect::PixelsAndTolerance) {
        panic!("an all-gaussian shaped list must pass the tolerance audit's gate: {e}");
    }
    let heavy = [
        ShapedAtom2d::new(atoms[0], CurveShape::gaussian()),
        ShapedAtom2d::new(atoms[1], CurveShape::with_beta(Beta::Four.value())),
    ];
    match shaped(&heavy, Collect::PixelsAndTolerance) {
        Err(LightsError::ContractViolation(ContractViolation::KernelCurveNotGaussian { .. })) => (),
        Err(e) => panic!("the refusal must name the non-gaussian curve: {e}"),
        Ok(_) => panic!(
            "the kernel family is gaussian but atom 1 carries beta 4, and the tolerance audit \
             bounds a gaussian tail; the gate must read each atom's curve, not the family's"
        ),
    }
}
