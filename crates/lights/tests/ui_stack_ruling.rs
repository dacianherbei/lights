use std::path::PathBuf;

fn repo_file(rel: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn component(map: &str, id: &str) -> String {
    let key = format!("id = \"{id}\"");
    let found: Vec<&str> = map
        .split("[[component]]")
        .filter(|block| block.lines().any(|l| l.trim() == key))
        .collect();
    assert_eq!(found.len(), 1, "war_map.toml carries {} components with {key}", found.len());
    found[0].to_string()
}

const RETIRED_STACK: [&str; 3] = ["dear-imgui", "imnodes", "SDL3"];

#[test]
fn war_map_registers_lights_as_live_and_optical_on_its_real_stack() {
    let map = repo_file("docs/war_map.toml");
    let lights = component(&map, "lights");
    for stale in ["group = \"Legacy\"", "status = \"Deferred\"", "Archived", "not carried forward"] {
        assert!(!lights.contains(stale), "lights component still says {stale:?} (L-113 ruling (b))");
    }
    let optical = component(&map, "optical");
    for stale in RETIRED_STACK {
        assert!(!optical.contains(stale), "optical component still names {stale:?}");
    }
    for real in ["eframe", "egui", "lights"] {
        assert!(optical.contains(real), "optical component does not name {real:?}");
    }
}

#[test]
fn claude_md_does_not_name_the_retired_ui_stack() {
    let text = repo_file("CLAUDE.md");
    let hits: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("optical") && RETIRED_STACK.iter().any(|s| l.contains(s)))
        .map(|(i, l)| (i + 1, l))
        .collect();
    assert!(hits.is_empty(), "CLAUDE.md still puts optical on the retired stack: {hits:?}");
}
