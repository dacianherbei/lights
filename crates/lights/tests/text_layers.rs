use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ruling() -> String {
    let path = crate_root().join("../../docs/lights/TEXT_LAYERS.md");
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("the text-layer ruling (L-160) must exist at {}: {e}", path.display()))
}

fn ruled_line<'a>(text: &'a str, key: &str) -> &'a str {
    let lines: Vec<&str> = text
        .lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .collect();
    assert_eq!(lines.len(), 1, "the ruling must state `{key}` exactly once, found {}", lines.len());
    lines[0].trim()
}

fn linked_libraries() -> BTreeSet<String> {
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).unwrap();
    let mut names = BTreeSet::new();
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            inside = line == "[dependencies]";
            continue;
        }
        if inside && !line.is_empty() && !line.starts_with('#') {
            let name = line.split('=').next().unwrap().trim();
            names.insert(name.to_string());
        }
    }
    names
}

#[test]
fn the_text_ruling_names_exactly_the_libraries_lights_links() {
    let text = ruling();
    let ruled: BTreeSet<String> = ruled_line(&text, "lights links:")
        .split(',')
        .map(|s| s.trim().trim_matches('`').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let linked = linked_libraries();
    assert!(!linked.is_empty(), "the [dependencies] scan found nothing: it must not pass by reading nothing");
    assert_eq!(
        ruled, linked,
        "L-160 rules that font layers 1-3 live outside lights: a library dependency added \
         without amending docs/lights/TEXT_LAYERS.md is a scope change nobody ruled on"
    );

    let precedent = ruled_line(&text, "tier 2 precedent:").trim_matches('`');
    assert!(
        !precedent.starts_with("src/fit/"),
        "the derived-atom format must copy a CORE precedent: anything under src/fit/ \
         makes text opt-in, which the layering guard and optical both forbid"
    );
    let source = fs::read_to_string(crate_root().join(precedent))
        .unwrap_or_else(|e| panic!("tier 2 precedent {precedent} must exist: {e}"));
    assert!(source.contains("_MAGIC: &str") && source.contains("_VERSION: u32"));
    assert!(source.contains("written by a different format"), "the precedent must refuse on a version mismatch");
    assert!(!source.contains(concat!("crate", "::fit")));
}

#[test]
fn a_glyph_is_one_region_and_the_withdrawn_partition_stays_quoted() {
    let text = ruling();
    let ruled = ruled_line(&text, "glyph partition:").trim();
    assert_eq!(
        ruled, "one Region, holes empty",
        "L-172 (PM, 2026-09-29) rules that a glyph is ONE Region whose membership is the \
         non-zero winding of all its contours, with holes empty and no overlap resolution; \
         the ruled line now says {ruled:?}"
    );
    assert!(
        text.contains("union of parts minus union of holes"),
        "the withdrawn parts-and-holes ruling must stay QUOTED on the page. A withdrawal \
         says which way the tree moved, and a reader who finds only the new rule cannot \
         tell whether the partition was rejected or never considered"
    );
    let primitive = fs::read_to_string(crate_root().join("src/primitive.rs"))
        .unwrap_or_else(|e| panic!("src/primitive.rs must exist: {e}"));
    assert!(
        primitive.contains("pub struct Region"),
        "the ruling names a Region as the thing a glyph becomes, so the core must define one"
    );
    assert!(
        primitive.contains("regions"),
        "PrimitiveSet must carry the regions the ruling says a glyph converts to, or the \
         document claims a shape the tree cannot hold"
    );
}
