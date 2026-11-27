// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Read-only KVMem launch capability advertisement.
//!
//! This is deliberately independent of model loading, GPU discovery and the
//! daemon. KVMem uses it to qualify a worker before a model is selected.

use serde_json::{json, Value};

const MIB: u64 = 1 << 20;

pub(crate) fn report() -> Value {
    let kvmem = cfg!(feature = "kvmem");
    let layerwise = kvmem.then(|| {
        json!({
            "available": true,
            "config_source": "environment",
            "current_layer_q": true,
            "query_replay": false,
            "position_policy": "original",
            "model_architectures": ["qwen35"],
            "model_sources": ["hfq"],
            "max_devices": 1,
            "max_concurrency": 8,
            "minimum_window_tokens": 384,
            "minimum_reserve_blocks": 1,
            "parallel_modes": ["single"],
            "kv_formats": ["q8"],
            "vision_residency": ["resident", "cpu"],
            "default_resources": {
                "activation_ram_bytes": 64 * MIB,
                "activation_disk_bytes": 32 * MIB,
                "staging_bytes": 8 * MIB,
                "metadata_bytes": 1024 * MIB,
                "activation_metadata_bytes": 96 * MIB,
            },
            "speculative": {
                "none": {"available": true},
                "mtp": {
                    "available": true,
                    "draft_width_model_owned": true,
                    "adaptive": false,
                    "ngram": false,
                    "max_concurrency": 8,
                    "max_devices": 1,
                    "draft_asset": "embedded",
                    "history_policies": ["first-layer", "upstream-global"],
                    "default_history_policy": "first-layer",
                    "state_modes": ["native"],
                },
                "dflash2": {
                    "available": true,
                    "draft_width_model_owned": true,
                    "adaptive": false,
                    "ngram": false,
                    "max_concurrency": 8,
                    "max_devices": 1,
                    "draft_asset": "paired_hfq",
                    "model_sources": ["hfq"],
                    "history": "independent-native-window",
                },
            },
        })
    });

    json!({
        "schema_version": 1,
        "backend": "hipfire",
        "query_scoring_revision": 2,
        "recall_strategy_revision": if kvmem { json!(1) } else { Value::Null },
        "compiled_with_kvmem": kvmem,
        "kvmem_contract_version": if kvmem { json!(1) } else { Value::Null },
        "block_tokens": [128],
        "kv_formats": if kvmem { json!(["q8"]) } else { json!([]) },
        "model_sources": ["hfq"],
        "model_architectures": ["qwen35"],
        "max_devices": 1,
        "max_concurrency": 8,
        "graph_alpha_supported": false,
        "independent_graph_lrfu": false,
        "streaming": true,
        "concurrent_requests": true,
        "multimodal": true,
        "recall_strategies": {
            "legacy": {
                "available": kvmem,
                "config_source": "cli",
            },
            "layerwise": layerwise.unwrap_or(Value::Null),
        },
        "speculative": {
            "none": {"available": true},
            "mtp": {
                "available": true,
                "draft_width_model_owned": true,
                "draft_asset": "embedded",
            },
            "dflash2": {
                "available": true,
                "draft_width_model_owned": true,
                "draft_asset": "paired_hfq",
                "model_sources": ["hfq"],
            },
        },
        "vision": {
            "available": true,
            "model_sources": ["hfq"],
            "sidecar_arch_ids": [5, 6],
            "residency": ["resident", "cpu"],
            "image_types": ["image/png", "image/jpeg"],
            "max_images_per_request": 1,
            "remote_urls": false,
        },
        "auth": {
            "api_key_file": true,
            "raw_api_key_arg": false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::report;
    use clap::Parser;

    #[test]
    fn cli_accepts_the_read_only_capability_flag() {
        let cli = crate::Cli::try_parse_from(["hipfire", "--kvmem-capabilities"]).unwrap();
        assert!(cli.kvmem_capabilities);
    }

    #[test]
    fn report_declares_the_kvmem_ui_contract() {
        let report = report();
        assert_eq!(report["schema_version"], 1);
        assert_eq!(report["backend"], "hipfire");
        assert_eq!(report["block_tokens"], serde_json::json!([128]));
        assert_eq!(report["model_sources"], serde_json::json!(["hfq"]));
        assert_eq!(report["recall_strategies"]["legacy"]["available"], true);
        assert_eq!(
            report["recall_strategies"]["layerwise"]["available"],
            cfg!(feature = "kvmem")
        );
        assert_eq!(report["vision"]["max_images_per_request"], 1);
        assert_eq!(report["auth"]["api_key_file"], true);
        if cfg!(feature = "kvmem") {
            assert_eq!(report["recall_strategy_revision"], 1);
        } else {
            assert!(report["recall_strategy_revision"].is_null());
        }
    }
}
