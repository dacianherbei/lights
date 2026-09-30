#![cfg(feature = "fit")]
use lights::fit::atom2d::camera;
use lights::fit::content::Content;
use lights::fit::coverage::{DEFAULT_CELL_LIMIT, DEFAULT_MAX_DEPTH, coverage_budget};
use lights::fit::greedy::{FitRequest, fit, fit_budget, render_or_background};
use lights::fit::nterm::{Arm, REACHABLE_ARMS};
use lights::kernel::KernelFamily;
use lights::settings::SharedSettings;

const P4_SOURCE: &str = include_str!("../src/bin/lights_p4.rs");
const REPORT_PAGE: &str = include_str!("../scripts/report_page.py");

const WITHDRAWN: [&str; 2] = ["not reachable through KernelFamily", "sits at frequency zero"];

#[test]
fn the_p4_banner_and_the_report_page_do_not_call_the_signed_kernels_unreachable() {
    for (name, text) in [("src/bin/lights_p4.rs", P4_SOURCE), ("scripts/report_page.py", REPORT_PAGE)] {
        for claim in WITHDRAWN {
            assert!(
                !text.contains(claim),
                "{name} still says {claim:?}, but KernelFamily carries every signed kernel and \
                 the fitter renders each one at a nonzero carrier"
            );
        }
        assert!(
            text.contains("fixed carrier"),
            "{name} no longer states which half of the signed arms is missing: the carrier is \
             fixed, not searched"
        );
    }
}

#[test]
fn the_p4_ladder_still_carries_no_signed_arm() {
    let signed: Vec<&Arm> = REACHABLE_ARMS
        .iter()
        .filter(|arm| match arm {
            Arm::Atoms(kernel) => KernelFamily::MODULATED_SWEEP.contains(kernel),
            Arm::Wavelet(_, _) => false,
        })
        .collect();
    assert!(
        signed.is_empty(),
        "REACHABLE_ARMS now carries {signed:?}: the p4 banner's 'two of the six arms are \
         missing' is false and must be rewritten with it"
    );
}

#[test]
fn every_modulated_kernel_is_fitted_at_a_carrier_that_is_not_the_gaussian() {
    let content = Content::n1_colour_field();
    let mut coverage = coverage_budget(DEFAULT_CELL_LIMIT);
    let truth = match content.truth(DEFAULT_MAX_DEPTH, &mut coverage) {
        Ok(t) => t,
        Err(e) => panic!("truth: {e}"),
    };
    let scaffold = content.fit_scaffold();
    let spec = camera(truth.width, truth.height);
    let mut fitted = 0usize;
    for kernel in KernelFamily::MODULATED_SWEEP {
        let mut request = FitRequest::new(kernel, SharedSettings::<f64>::prefiltered());
        request.max_atoms = 1;
        request.refine_iterations = 1;
        request.refine_tail = 0;
        let mut budget = fit_budget(200_000);
        let outcome = match fit(&scaffold, &truth, &request, &mut budget) {
            Ok(o) => o,
            Err(e) => panic!("the greedy fitter refused {}: {e}", kernel.name()),
        };
        assert!(!outcome.atoms.is_empty(), "{} fitted no atom", kernel.name());
        let signed = match render_or_background(&outcome.atoms, &spec, kernel, &request.settings, request.tile) {
            Ok(i) => i,
            Err(e) => panic!("rendering the {} fit: {e}", kernel.name()),
        };
        let plain = match render_or_background(
            &outcome.atoms,
            &spec,
            KernelFamily::Gaussian,
            &request.settings,
            request.tile,
        ) {
            Ok(i) => i,
            Err(e) => panic!("rendering the {} atoms as gaussians: {e}", kernel.name()),
        };
        assert_ne!(
            signed.pixels,
            plain.pixels,
            "the {} fit renders bit-identical to the gaussian, so its carrier sits at zero",
            kernel.name()
        );
        fitted += 1;
    }
    assert!(fitted >= 1, "no modulated kernel was fitted");
    assert_eq!(fitted, KernelFamily::MODULATED_SWEEP.len());
}
