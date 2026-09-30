use lights::method::MethodId;
use std::path::PathBuf;

fn repo_file(rel: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

const ADR: &str = "docs/architecture/ADR-019_lights_sdf_pass.md";

#[test]
fn adr_019_records_where_the_sdf_champion_lives() {
    let adr = repo_file(ADR);
    assert!(adr.contains("**Status**: Accepted"), "{ADR} carries no accepted status");
    for cited in ["L-118", "L-73", "L-75", "L-144", "RasterLayer", "composite_over", "Target::combined"] {
        assert!(adr.contains(cited), "{ADR} does not cite {cited:?}");
    }
}

#[test]
fn no_render_method_is_the_sdf_pass() {
    assert!(!MethodId::ALL.is_empty(), "MethodId::ALL is empty, so this test would check nothing");
    let named: Vec<&str> = MethodId::ALL
        .iter()
        .map(|m| m.name())
        .filter(|n| n.contains("sdf") || n.contains("analytic"))
        .collect();
    assert!(named.is_empty(), "ADR-019 rules the SDF pass out of MethodId, yet {named:?} is a method");
}

#[test]
fn the_sdf_pass_lands_beside_the_raster_pass_it_composites_with() {
    let image = repo_file("crates/lights/src/image.rs");
    assert!(image.contains("pub struct RasterLayer<T>"), "src/image.rs lost RasterLayer, the pass ADR-019 composites through");
    assert!(image.contains("pub fn composite_over("), "src/image.rs lost composite_over");
}
