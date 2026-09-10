//! The ```rust blocks on the front page and the landing page compile against
//! this crate. Each block is pasted into a `main` with the crate's front-door
//! imports and `cargo check`ed in a scratch crate that depends on this one by
//! path, so a block that does not compile fails here rather than in a reader's
//! editor. Weight-free: nothing is run.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const PAGES: [&str; 2] = ["README.md", "site/src/handwritten/index.mdx"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Every ```rust fence on a page, with the indentation a tab component adds removed.
fn rust_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut buf: Option<Vec<&str>> = None;
    for line in text.lines() {
        let stripped = line.trim_start();
        match &mut buf {
            None if fence_tag(stripped) == Some("rust") => buf = Some(Vec::new()),
            Some(lines) if stripped.starts_with("```") => {
                blocks.push(dedent(lines));
                buf = None;
            }
            Some(lines) => lines.push(line),
            None => {}
        }
    }
    blocks
}

/// The language a fence opens with: `rust` for "```rust" and "```rust title=x", not for "```rusty".
fn fence_tag(line: &str) -> Option<&str> {
    line.strip_prefix("```")?.split_whitespace().next()
}

fn dedent(lines: &[&str]) -> String {
    let indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|l| {
            if l.len() >= indent {
                &l[indent..]
            } else {
                l.trim_start()
            }
        })
        .map(|l| format!("{l}\n"))
        .collect()
}

#[test]
fn the_user_page_rust_snippets_check() {
    let root = repo_root();
    let scratch = std::env::temp_dir().join(format!("loudkit-snippet-{}", std::process::id()));
    let _ = fs::remove_dir_all(&scratch);
    fs::create_dir_all(&scratch).unwrap();

    let mut manifest = format!(
        "[package]\nname = \"snippet\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n\
         [dependencies]\nloudkit = {{ path = {:?} }}\n",
        root.join("rust")
    );
    let mut bins = Vec::new();
    for page in PAGES {
        let text = fs::read_to_string(root.join(page)).unwrap();
        let blocks = rust_blocks(&text);
        assert!(
            !blocks.is_empty(),
            "{page} shows no rust block; the gate is looking at nothing"
        );
        for (i, body) in blocks.into_iter().enumerate() {
            let name = format!("{}_{}", page.replace(['/', '.'], "_"), i + 1);
            let main = format!(
                // No `use` is injected: the snippet must carry its own, or a
                // reader who copies it does not get a program.
                "fn main() -> Result<(), String> {{\n{body}    Ok(())\n}}\n"
            );
            fs::write(scratch.join(format!("{name}.rs")), &main).unwrap();
            manifest.push_str(&format!(
                "\n[[bin]]\nname = {name:?}\npath = \"{name}.rs\"\n"
            ));
            bins.push((page, i + 1, body));
        }
    }
    fs::write(scratch.join("Cargo.toml"), manifest).unwrap();
    // The crate's own lock pins every dependency the scratch crate can reach,
    // so the check resolves offline from what `cargo test` already fetched.
    fs::copy(root.join("rust/Cargo.lock"), scratch.join("Cargo.lock")).unwrap();

    // The same target directory as this test build, so the dependencies'
    // check artefacts are reused instead of rebuilt from nothing.
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("rust/target"));
    let output = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--bins", "--manifest-path"])
        .arg(scratch.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target)
        .output()
        .expect("cargo runs");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = fs::remove_dir_all(&scratch);
    assert!(
        output.status.success(),
        "a rust block on a user page does not compile:\n{stderr}\n--- the blocks ---\n{}",
        bins.iter()
            .map(|(page, i, body)| format!("{page} block {i}:\n{body}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
