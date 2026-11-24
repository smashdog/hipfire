// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runtime ABI diagnostics exposed by the shim. Layout itself is guarded at
//! compile time by `src/abi.rs`.

use hipfire_kvmem::{abi_version, kvmem_commit, HK_ABI_VERSION};

#[test]
fn shim_reports_the_expected_abi_and_commit() {
    assert_eq!(abi_version(), HK_ABI_VERSION);
    let commit = kvmem_commit();
    assert!(
        !commit.is_empty(),
        "shim must stamp a commit (or 'unknown')"
    );
}
