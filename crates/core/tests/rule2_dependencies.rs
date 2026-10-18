//! Primary outcome and compliance tests for Rule 2 (`FIX-11.1`):
//! - Every external dependency across all workspace manifests is declared once in the root
//!   `[workspace.dependencies]` with `default-features = false`.
//! - Every crate and example manifest references external dependencies via `workspace = true`.
//! - `cargo metadata --format-version 1 --no-deps` confirms `uses_default_features: false`
//!   for every registry dependency across every workspace member.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/core must be two levels below workspace root")
        .to_path_buf()
}

fn collect_member_manifests(root: &Path) -> Vec<PathBuf> {
    let mut manifests = Vec::new();
    for dir_name in ["crates", "examples"] {
        let dir = root.join(dir_name);
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let manifest = entry.path().join("Cargo.toml");
                if manifest.is_file() {
                    manifests.push(manifest);
                }
            }
        }
    }
    manifests.sort();
    manifests
}

#[test]
fn root_workspace_dependencies_all_disable_default_features() {
    let root = workspace_root();
    let root_toml = fs::read_to_string(root.join("Cargo.toml")).expect("read root Cargo.toml");

    let mut in_workspace_deps = false;
    let mut violations = Vec::new();
    let mut dep_count = 0usize;

    for (line_idx, raw_line) in root_toml.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            in_workspace_deps = line == "[workspace.dependencies]";
            continue;
        }
        if in_workspace_deps {
            if let Some((dep_name, rhs)) = line.split_once('=') {
                let dep_name = dep_name.trim();
                let rhs = rhs.trim();
                dep_count += 1;
                if !rhs.contains("default-features = false") {
                    violations.push(format!(
                        "Cargo.toml:{}: workspace dependency `{dep_name}` does not specify `default-features = false`: {rhs}",
                        line_idx + 1
                    ));
                }
            }
        }
    }

    assert!(
        dep_count > 0,
        "Expected `[workspace.dependencies]` in root Cargo.toml to declare accepted dependencies"
    );
    assert!(
        violations.is_empty(),
        "Rule 2 violation in root `[workspace.dependencies]`:\n{}",
        violations.join("\n")
    );
}

#[test]
fn member_manifests_reference_external_dependencies_via_workspace_true() {
    let root = workspace_root();
    let manifests = collect_member_manifests(&root);
    assert!(!manifests.is_empty(), "No member Cargo.toml files found");

    let mut violations = Vec::new();

    for manifest in manifests {
        let rel = manifest.strip_prefix(&root).unwrap_or(&manifest);
        let content = fs::read_to_string(&manifest).expect("read member Cargo.toml");

        let mut in_dep_section = false;
        let mut multiline_dep: Option<(String, usize, String)> = None;

        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }

            if let Some((dep_name, start_line, mut acc)) = multiline_dep.take() {
                acc.push(' ');
                acc.push_str(line);
                if line.contains('}') {
                    if !acc.contains("path =") && !acc.contains("workspace = true") {
                        violations.push(format!(
                            "{}:{}: external dependency `{dep_name}` must use `workspace = true`: {acc}",
                            rel.display(),
                            start_line
                        ));
                    }
                } else {
                    multiline_dep = Some((dep_name, start_line, acc));
                }
                continue;
            }

            if line.starts_with('[') {
                in_dep_section = line == "[dependencies]"
                    || line == "[dev-dependencies]"
                    || line == "[build-dependencies]"
                    || (line.starts_with("[target.") && line.ends_with(".dependencies]"));
                continue;
            }

            if in_dep_section {
                if let Some((dep_name, rhs)) = line.split_once('=') {
                    let dep_name = dep_name.trim().to_string();
                    let rhs = rhs.trim().to_string();
                    if rhs.starts_with('{') && !rhs.contains('}') {
                        multiline_dep = Some((dep_name, line_idx + 1, rhs));
                    } else if !rhs.contains("path =") && !rhs.contains("workspace = true") {
                        violations.push(format!(
                            "{}:{}: external dependency `{dep_name}` must use `workspace = true`: {rhs}",
                            rel.display(),
                            line_idx + 1
                        ));
                    }
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Rule 2 violation: member manifests must reference external dependencies via `workspace = true`:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cargo_metadata_confirms_no_external_dependency_uses_default_features() {
    let root = workspace_root();
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()
        .expect("failed to execute `cargo metadata`");

    assert!(
        output.status.success(),
        "`cargo metadata` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("valid utf-8 from cargo metadata");

    // In `cargo metadata --format-version 1 --no-deps`, each dependency entry is a JSON object
    // containing `"name":"<dep>"`, `"source":...`, and `"uses_default_features":true|false`.
    // Registry dependencies have `"source":"registry+..."` whereas workspace path dependencies
    // have `"source":null`.
    let mut violations = Vec::new();
    let mut checked_registry_deps = 0usize;

    let mut search_pos = 0;
    let needle = "\"source\":\"registry+";
    while let Some(rel_idx) = stdout[search_pos..].find(needle) {
        let abs_idx = search_pos + rel_idx;
        let obj_start = stdout[..abs_idx]
            .rfind('{')
            .expect("dependency JSON object start");
        let obj_end = stdout[abs_idx..]
            .find('}')
            .map(|i| abs_idx + i + 1)
            .expect("dependency JSON object end");
        let obj_slice = &stdout[obj_start..obj_end];

        let name = obj_slice
            .split("\"name\":\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .unwrap_or("<unknown>");

        checked_registry_deps += 1;
        if obj_slice.contains("\"uses_default_features\":true") {
            violations.push(format!(
                "Registry dependency `{name}` has `uses_default_features: true` in metadata object: {obj_slice}"
            ));
        }

        search_pos = obj_end;
    }

    assert!(
        checked_registry_deps > 0,
        "Expected to inspect at least one registry dependency in `cargo metadata`"
    );
    assert!(
        violations.is_empty(),
        "Rule 2 violation: external dependencies with `uses_default_features: true`:\n{}",
        violations.join("\n")
    );
}
