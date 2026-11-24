// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Build the KVMem hipfire adapter: compile the C ABI shim and the portable
//! KVMem host library into one static archive.
//!
//! `KVMEM_ROOT` selects the kvmem checkout (default: the sibling directory
//! `<hipfire>/../kvmem`). The library is host-only C++17 (`kvmem/CMakeLists.txt`
//! reads the same `kvmem/host-sources.txt` manifest), so a plain C++ compiler is
//! enough; the hipfire kernels never see a kvmem symbol.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn git_head(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let head = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!head.is_empty()).then_some(head)
}

fn digest_files(root: &Path, paths: &mut [PathBuf]) -> String {
    paths.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    let mut hasher = Sha256::new();
    for relative in paths.iter() {
        let display = relative.to_string_lossy().replace('\\', "/");
        hasher.update(display.as_bytes());
        hasher.update([0]);
        let path = root.join(relative);
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "hipfire-kvmem: cannot read {} for pin check: {error}",
                path.display()
            )
        });
        hasher.update(bytes);
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory).unwrap_or_else(|error| {
        panic!(
            "hipfire-kvmem: cannot read {}: {error}",
            directory.display()
        )
    });
    for entry in entries {
        let path = entry.expect("reading a directory entry").path();
        if path.is_dir() {
            collect_files(&path, files);
        } else if path.is_file() {
            files.push(path);
        }
    }
}

fn include_digest(root: &Path) -> String {
    let include = root.join("kvmem/include");
    let mut files = Vec::new();
    collect_files(&include, &mut files);
    if files.is_empty() {
        panic!("hipfire-kvmem: {} has no headers", include.display());
    }
    let mut relative: Vec<PathBuf> = files
        .iter()
        .map(|path| {
            path.strip_prefix(&include)
                .expect("header under include")
                .to_path_buf()
        })
        .collect();
    digest_files(&include, &mut relative)
}

/// The pin format is intentionally tiny: `key=value` lines, comments allowed.
fn read_pin(path: &Path) -> BTreeMap<String, String> {
    let text = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("hipfire-kvmem: cannot read pin {}: {error}", path.display())
    });
    let mut values = BTreeMap::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("{}:{}: expected key=value", path.display(), lineno + 1));
        values.insert(key.trim().to_owned(), value.trim().to_owned());
    }
    for key in [
        "commit",
        "abi_version",
        "host_sources_sha256",
        "include_sha256",
        "shim_sha256",
    ] {
        if !values.contains_key(key) {
            panic!("hipfire-kvmem: pin {} is missing {key}", path.display());
        }
    }
    values
}

fn header_abi_version(root: &Path) -> u32 {
    let path = root.join("backends/hipfire/hipfire_kvmem_shim.h");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("hipfire-kvmem: cannot read {}: {error}", path.display()));
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#define HK_ABI_VERSION") else {
            continue;
        };
        let digits: String = rest
            .trim()
            .trim_end_matches('u')
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if let Ok(value) = digits.parse() {
            return value;
        }
    }
    panic!(
        "hipfire-kvmem: HK_ABI_VERSION is missing from {}",
        path.display()
    );
}

fn verify_pin(root: &Path) -> Option<String> {
    let path = root.join("backends/hipfire/kvmem.pin");
    if !path.is_file() {
        if env_flag("KVMEM_REQUIRE_PIN") {
            panic!(
                "hipfire-kvmem: {} is missing and KVMEM_REQUIRE_PIN is set",
                path.display()
            );
        }
        if !env_flag("KVMEM_ALLOW_UNPINNED") {
            println!(
                "cargo:warning=hipfire-kvmem: {} is missing; using an unpinned kvmem checkout",
                path.display()
            );
        }
        return None;
    }

    let pin = read_pin(&path);
    let header_abi = header_abi_version(root);
    let pin_abi: u32 = pin["abi_version"]
        .parse()
        .unwrap_or_else(|_| panic!("hipfire-kvmem: invalid abi_version in {}", path.display()));
    if header_abi != pin_abi {
        panic!("hipfire-kvmem: abi_version mismatch: pin {pin_abi}, header {header_abi}");
    }

    let manifest_digest = {
        let mut paths = vec![PathBuf::from("host-sources.txt")];
        digest_files(&root.join("kvmem"), &mut paths)
    };
    let actual = [
        ("host_sources_sha256", manifest_digest),
        ("include_sha256", include_digest(root)),
        (
            "shim_sha256",
            digest_files(
                root,
                &mut [
                    PathBuf::from("backends/hipfire/hipfire_kvmem_shim.h"),
                    PathBuf::from("backends/hipfire/hipfire_kvmem_shim.cpp"),
                ],
            ),
        ),
    ];
    for (key, digest) in actual {
        if pin[key] != digest {
            panic!(
                "hipfire-kvmem: {key} mismatch: pin {}, actual {digest}; \
                 run `python3 scripts/hipfire_pin.py --write` in kvmem",
                pin[key]
            );
        }
    }
    pin.get("commit")
        .filter(|commit| *commit != "unknown")
        .cloned()
}

fn verify_expected_commit(root: &Path) {
    let Ok(expected) = std::env::var("KVMEM_EXPECTED_COMMIT") else {
        return;
    };
    if expected.is_empty() {
        return;
    }
    let expected = expected.to_ascii_lowercase();
    if expected.len() < 7
        || expected.len() > 40
        || !expected.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        panic!("hipfire-kvmem: invalid KVMEM_EXPECTED_COMMIT {expected:?}");
    }
    let head = git_head(root).unwrap_or_else(|| {
        panic!(
            "hipfire-kvmem: cannot verify KVMEM_EXPECTED_COMMIT: {} has no readable git HEAD",
            root.display()
        )
    });
    if !head.starts_with(&expected) {
        panic!("hipfire-kvmem: commit mismatch: expected {expected}, actual {head}");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=KVMEM_ROOT");
    println!("cargo:rerun-if-env-changed=KVMEM_REQUIRE_PIN");
    println!("cargo:rerun-if-env-changed=KVMEM_ALLOW_UNPINNED");
    println!("cargo:rerun-if-env-changed=KVMEM_EXPECTED_COMMIT");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = std::env::var_os("KVMEM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../kvmem"));
    let root = root.canonicalize().unwrap_or(root);

    let include = root.join("kvmem/include");
    let shim_dir = root.join("backends/hipfire");
    let host_src = root.join("kvmem/src/host");
    let shim = shim_dir.join("hipfire_kvmem_shim.cpp");
    let host_manifest = root.join("kvmem/host-sources.txt");
    let pin_path = shim_dir.join("kvmem.pin");

    for probe in [
        include.join("kvmem/raw_kv_store.hpp"),
        shim.clone(),
        host_src.join("raw_kv_store.cpp"),
        host_manifest.clone(),
    ] {
        if !probe.exists() {
            panic!(
                "hipfire-kvmem: missing {} — point KVMEM_ROOT at a kvmem checkout \
                 (kvmem/include + kvmem/src/host + kvmem/host-sources.txt + backends/hipfire)",
                probe.display()
            );
        }
    }

    let pin_commit = verify_pin(&root);
    verify_expected_commit(&root);
    let commit = git_head(&root)
        .or(pin_commit)
        .unwrap_or_else(|| "unknown".to_owned());

    let manifest_text = fs::read_to_string(&host_manifest).unwrap_or_else(|error| {
        panic!(
            "hipfire-kvmem: cannot read {}: {error}",
            host_manifest.display()
        )
    });
    let sources: Vec<&str> = manifest_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    if sources.is_empty() {
        panic!(
            "hipfire-kvmem: {} lists no host sources",
            host_manifest.display()
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .pic(true)
        // Third-party host library: warnings are kvmem's to triage, not ours.
        .warnings(false)
        .flag_if_supported("-O2")
        .flag_if_supported("-fno-strict-aliasing")
        // Enable `#pragma omp simd` reductions in the host recall kernels
        // (page-softmax dot product). SIMD-only: no OpenMP runtime is linked.
        .flag_if_supported("-fopenmp-simd")
        .include(&include)
        .include(&shim_dir)
        .include(&host_src);

    let commit_define = format!("\"{}\"", commit.replace('\\', "\\\\").replace('"', "\\\""));
    build.define("KVMEM_COMMIT", Some(commit_define.as_str()));

    for source in sources {
        let relative = Path::new(source);
        if relative.is_absolute() || relative.components().any(|part| part.as_os_str() == "..") {
            panic!("hipfire-kvmem: invalid host source path in manifest: {source}");
        }
        let path = root.join("kvmem").join(relative);
        if !relative.starts_with("src/host") {
            panic!("hipfire-kvmem: invalid host source path in manifest: {source}");
        }
        if !path.exists() {
            panic!(
                "hipfire-kvmem: kvmem host source {} is missing — the kvmem source \
                 list moved; update kvmem/host-sources.txt",
                path.display()
            );
        }
        build.file(path);
    }
    build.file(&shim);
    build.compile("kvmem_hipfire_shim");

    // Host-only C++: the Rust link needs the C++ runtime and pthreads.
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target == "macos" || target == "ios" || target == "tvos" {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    if target != "windows" {
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

    // Full rebuild hooks: a kvmem-side edit must not leave a stale archive.
    println!("cargo:rerun-if-changed={}", include.display());
    println!("cargo:rerun-if-changed={}", host_src.display());
    println!("cargo:rerun-if-changed={}", shim_dir.display());
    println!("cargo:rerun-if-changed={}", host_manifest.display());
    println!("cargo:rerun-if-changed={}", pin_path.display());
}
