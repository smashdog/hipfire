// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Build the KVMem hipfire adapter: compile the C ABI shim and the portable
//! KVMem host library into one static archive.
//!
//! `KVMEM_ROOT` selects the kvmem checkout (default: the sibling directory
//! `<workspace>/../kvmem`, i.e. `/mnt/ai/project/kvmem` for this layout). The
//! library is host-only C++17 (`kvmem/CMakeLists.txt`'s `add_library(kvmem)`
//! source list, no CUDA/HIP), so a plain C++ compiler is enough; the hipfire
//! kernels never see a kvmem symbol.

use std::path::PathBuf;

/// Source list copied from `kvmem/CMakeLists.txt` (`add_library(kvmem ...)`).
/// Keeping it explicit (instead of globbing) makes a kvmem-side file addition
/// visible here as a build failure rather than silently missing symbols.
const KVMEM_HOST_SOURCES: &[&str] = &[
    "kvmem_store.cpp",
    "kvmem_runtime.cpp",
    "raw_kv_store.cpp",
    "mean_key_retrieval.cpp",
    "graph_recall.cpp",
    "layerwise_scoring.cpp",
    "layerwise_query.cpp",
    "layerwise_request.cpp",
    "layerwise_selection.cpp",
    "layerwise_probe_selection.cpp",
    "layerwise_memory.cpp",
    "layerwise_pressure.cpp",
    "layerwise_output_scope.cpp",
    "layerwise_window.cpp",
    "layerwise_import.cpp",
    "layerwise_options.cpp",
    "query_projection_tape.cpp",
    "memory_contract.cpp",
    "kv_content.cpp",
    "native_kv_payload.cpp",
    "host_kv_storage.cpp",
    "spill_io.cpp",
    "session_memory.cpp",
    "prefix_checkpoint.cpp",
    "checkpoint_policy.cpp",
    "rope.cpp",
];

fn main() {
    println!("cargo:rerun-if-env-changed=KVMEM_ROOT");
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

    for probe in [
        include.join("kvmem/raw_kv_store.hpp"),
        shim.clone(),
        host_src.join("raw_kv_store.cpp"),
    ] {
        if !probe.exists() {
            panic!(
                "hipfire-kvmem: missing {} — point KVMEM_ROOT at a kvmem checkout \
                 (kvmem/include + kvmem/src/host + backends/hipfire)",
                probe.display()
            );
        }
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

    for f in KVMEM_HOST_SOURCES {
        let p = host_src.join(f);
        if !p.exists() {
            panic!(
                "hipfire-kvmem: kvmem host source {} is missing — the kvmem source \
                 list moved; update KVMEM_HOST_SOURCES in build.rs",
                p.display()
            );
        }
        build.file(p);
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
}
