#![allow(dead_code)]
use std::path::PathBuf;

pub const HEADER_KEY: &str = "invocation";

pub fn quoted(argument: &str) -> String {
    let plain = !argument.is_empty()
        && argument.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '/' || c == ':' || c == '='
        });
    if plain {
        return argument.to_string();
    }
    format!("'{}'", argument.replace('\'', "'\\''"))
}

pub fn command_line() -> String {
    std::env::args()
        .map(|a| quoted(&a))
        .collect::<Vec<String>>()
        .join(" ")
}

pub fn header_line() -> String {
    format!("#{HEADER_KEY}\t{}", command_line())
}

pub fn stamp(tables: &[PathBuf]) -> std::io::Result<usize> {
    let line = header_line();
    let opener = format!("#{HEADER_KEY}\t");
    let mut stamped = 0usize;
    for path in tables.iter() {
        let text = std::fs::read_to_string(path)?;
        let mut body = text.as_str();
        if let Some((first, rest)) = text.split_once('\n') {
            if first.starts_with(opener.as_str()) {
                body = rest;
            }
        }
        std::fs::write(path, format!("{line}\n{body}"))?;
        stamped += 1;
    }
    Ok(stamped)
}

pub fn table(header: &[String], columns: &[&str], rows: &[Vec<String>]) -> String {
    let mut text = header_line();
    text.push('\n');
    for line in header.iter() {
        text.push('#');
        text.push_str(line);
        text.push('\n');
    }
    text.push_str(&columns.join("\t"));
    text.push('\n');
    for cells in rows.iter() {
        text.push_str(&cells.join("\t"));
        text.push('\n');
    }
    text
}
