use std::path::PathBuf;
use std::process::Command;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    let mut p = crate_root();
    p.pop();
    p.pop();
    p
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lights_loop_pages_{}_{name}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }
    dir
}

fn entry_blocks(text: &str) -> Vec<Vec<&str>> {
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        if line == "[[entry]]" {
            blocks.push(Vec::new());
        } else if let Some(block) = blocks.last_mut() {
            block.push(line);
        }
    }
    blocks
}

fn visible_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => {
                in_tag = true;
                text.push(' ');
            }
            '>' => in_tag = false,
            c if !in_tag => text.push(c),
            _ => {}
        }
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn head_ids() -> Vec<String> {
    let text = std::fs::read_to_string(crate_root().join("head_open.toml"))
        .unwrap_or_else(|e| panic!("head_open.toml could not be read: {e}"));
    let mut ids = Vec::new();
    for block in entry_blocks(&text) {
        if !block.iter().any(|l| l.trim() == "status = \"open\"") {
            continue;
        }
        for line in block.iter() {
            let Some(rest) = line.strip_prefix("id = \"") else {
                continue;
            };
            if let Some(end) = rest.find('"') {
                ids.push(rest[..end].to_string());
            }
        }
    }
    ids
}

fn register_counts() -> (usize, usize) {
    let mut open = 0;
    let mut total = 0;
    for file in ["head_open.toml", "tail.toml"] {
        let text = std::fs::read_to_string(crate_root().join(file))
            .unwrap_or_else(|e| panic!("{file} could not be read: {e}"));
        for block in entry_blocks(&text) {
            total += 1;
            if block.iter().any(|l| l.trim() == "status = \"open\"") {
                open += 1;
            }
        }
    }
    (open, total)
}

fn pages(out: &PathBuf) -> (bool, String, String) {
    let o = Command::new("python3")
        .current_dir(crate_root())
        .arg("scripts/burndown_host.py")
        .arg("pages")
        .arg("--out-dir")
        .arg(out)
        .output()
        .unwrap_or_else(|e| panic!("python3 could not run burndown_host.py pages: {e}"));
    (
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn the_host_regenerates_the_register_page_to_a_named_file() {
    let out = scratch("register");
    let (ok, stdout, stderr) = pages(&out);
    assert!(
        ok,
        "`burndown_host.py pages` failed. L-148: a context driven by `claude -p` has no \
         artifact tool, so the PM's pages can only stay current if the HOST writes them to a \
         fixed path where staleness shows as a diff. stdout {stdout} stderr {stderr}"
    );
    let page = out.join("burndown.html");
    let html = std::fs::read_to_string(&page)
        .unwrap_or_else(|e| panic!("{} was not written: {e}; stdout {stdout}", page.display()));
    assert!(html.contains("<title>"), "{} is not a page: {:.200}", page.display(), html);
    let (open, total) = register_counts();
    let count = format!("{open} open of {total} entries");
    assert!(
        visible_text(&html).contains(count.as_str()),
        "the regenerated page does not say {count:?}, the live register's standing; it was \
         rendered from something other than head_open.toml + tail.toml"
    );
    let ids = head_ids();
    if ids.is_empty() {
        assert!(
            html.contains("Nothing open."),
            "head_open.toml has no open entry, yet the page does not say so"
        );
    }
    for id in ids.iter() {
        assert!(
            html.contains(id.as_str()),
            "the regenerated page does not carry {id}, an entry open in head_open.toml; it was \
             rendered from something other than the live register"
        );
    }
    assert!(
        stdout.contains(&page.display().to_string()),
        "the host must NAME the file it wrote, because the close bar is `regenerate, and name \
         the file`: {stdout}"
    );
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn the_research_page_is_written_exactly_when_its_record_is_in_the_tree() {
    let out = scratch("research");
    let (ok, stdout, stderr) = pages(&out);
    assert!(ok, "stdout {stdout} stderr {stderr}");
    let record = repo_root().join("docs/lights/RESEARCH.json");
    let page = out.join("research.html");
    assert_eq!(
        page.exists(),
        record.exists(),
        "research.html exists={} while {} exists={}: a page written without its record is \
         fabricated, and a record without its page is the silent staleness L-148 is about. \
         stdout {stdout}",
        page.exists(),
        record.display(),
        record.exists()
    );
    if !record.exists() {
        assert!(
            stdout.contains("skipped research"),
            "a page the host could not build must be SAID to be skipped: {stdout}"
        );
    }
    std::fs::remove_dir_all(&out).ok();
}

#[test]
fn the_default_page_directory_is_the_committed_one() {
    let o = Command::new("python3")
        .current_dir(crate_root().join("scripts"))
        .arg("-c")
        .arg("import burndown_host; print(burndown_host.PAGES_DIR)")
        .output()
        .unwrap_or_else(|e| panic!("python3 could not import burndown_host: {e}"));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let dir = String::from_utf8_lossy(&o.stdout).trim().to_string();
    assert_eq!(
        PathBuf::from(&dir),
        repo_root().join("docs/lights/pages"),
        "the host writes the pages somewhere other than docs/lights/pages, which is where the \
         PM's publishing session reads them from"
    );
}
