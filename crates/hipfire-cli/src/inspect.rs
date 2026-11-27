// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire inspect`: read a GGUF, HFQ, or safetensors-directory header and
//! report what hipfire would do with it. Read-only: no GPU, daemon, model load,
//! or tensor payload read.

use anyhow::{bail, Context, Result};
use clap::Args;
use std::io::Read;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub(crate) struct InspectArgs {
    /// A GGUF/HFQ model file or a safetensors model directory.
    pub path: PathBuf,
    /// Emit the KVMem HFQ/safetensors health report as JSON.
    #[arg(long)]
    pub json: bool,
}

/// The four GGUF magic bytes.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

pub(crate) fn inspect_command(args: InspectArgs) -> Result<()> {
    if args.json {
        let report = inspect_json(&args.path)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(&args.path)
        .with_context(|| format!("could not open {}", args.path.display()))?
        .read_exact(&mut magic)
        .with_context(|| format!("{} is too short to be a model file", args.path.display()))?;
    if &magic != GGUF_MAGIC {
        bail!(
            "{} is not a GGUF file; `hipfire inspect` reads GGUF files only",
            args.path.display()
        );
    }
    let report = hipfire_loader::gguf_native::inspect_report(
        &args.path,
        hipfire_loader::gguf_native::gguf_native_enabled(),
    )
    .map_err(anyhow::Error::msg)?;
    print!("{report}");
    Ok(())
}

/// Header/index-only model report for KVMem's CPU-side health check.
pub(crate) fn inspect_json(path: &std::path::Path) -> Result<serde_json::Value> {
    use hipfire_runtime::loader_api::ModelSource;
    use hipfire_runtime::model_source::ModelSource as SourceTrait;
    use serde_json::{json, Value};

    let path_text = path.to_string_lossy();
    let source = ModelSource::from_path(&path_text)
        .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
    let (kind, arch_id, tensor_count, format_version, names, metadata) = match &source {
        ModelSource::Hfq(hfq) => (
            "hfq",
            hfq.arch_id,
            hfq.tensors().len(),
            Some(hfq.format_version),
            hfq.tensors()
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            serde_json::from_str::<Value>(&hfq.metadata_json).unwrap_or_else(|_| json!({})),
        ),
        ModelSource::Dir(dir) => (
            "safetensors-dir",
            dir.arch_id(),
            dir.tensor_names().len(),
            None,
            dir.tensor_names(),
            serde_json::from_str::<Value>(SourceTrait::metadata_json(dir))
                .unwrap_or_else(|_| json!({})),
        ),
    };
    let is_dir = kind == "safetensors-dir";
    let qwen35 = matches!(arch_id, 5 | 6);
    let has_trunk = names.iter().any(|name| {
        name.ends_with("lm_head.weight")
            || name.ends_with("model.embed_tokens.weight")
            || name.ends_with("model.layers.0.self_attn.q_proj.weight")
    });
    let vision_present = names
        .iter()
        .any(|name| name.ends_with("model.visual.patch_embed.proj.weight"));
    let architecture = match arch_id {
        5 => "qwen35",
        6 => "qwen35-moe",
        20 => "dflash-draft",
        21 => "mtp-head",
        22 | 23 => "draft-sidecar",
        _ => "unknown",
    };
    let selected_metadata = json!({
        "model_type": metadata.get("model_type"),
        "architectures": metadata.get("architectures"),
        "num_experts": metadata.get("num_experts"),
        "quant_method": metadata.get("quant_method"),
        "torch_dtype": metadata.get("torch_dtype"),
    });
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Ok(json!({
        "schema_version": 1,
        "source": kind,
        "path": canonical,
        "arch_id": arch_id,
        "architecture": architecture,
        "format_version": format_version,
        "tensor_count": tensor_count,
        "can_serve": qwen35 && has_trunk,
        "kv_types": ["q8"],
        "metadata": selected_metadata,
        "vision": {
            "present": vision_present,
            "sidecar_supported": qwen35 && !is_dir,
        },
        "speculative": {
            "mtp": {"supported": qwen35 && has_trunk && !is_dir},
            "dflash2": {"supported": qwen35 && has_trunk && !is_dir, "draft_asset": "paired_hfq"},
        },
        "limitations": if is_dir {
            json!(["vision-sidecar-unsupported", "dflash-unsupported"])
        } else {
            json!([])
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    fn scratch(label: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hipfire-inspect-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_hfq(path: &std::path::Path, arch_id: u32, tensors: &[&str]) {
        let metadata = b"{}";
        let mut index = Vec::new();
        index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for tensor in tensors {
            index.extend_from_slice(&(tensor.len() as u16).to_le_bytes());
            index.extend_from_slice(tensor.as_bytes());
            index.push(1); // quant type
            index.push(1); // dimensions
            index.extend_from_slice(&1u32.to_le_bytes());
            index.extend_from_slice(&0u32.to_le_bytes());
            index.extend_from_slice(&0u64.to_le_bytes());
        }
        let data_offset = 32 + metadata.len() + index.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HFQM");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&arch_id.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&32u64.to_le_bytes());
        bytes.extend_from_slice(&(data_offset as u64).to_le_bytes());
        bytes.extend_from_slice(metadata);
        bytes.extend_from_slice(&index);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_safetensors(dir: &std::path::Path) {
        std::fs::write(
            dir.join("config.json"),
            json!({"model_type":"qwen3.5"}).to_string(),
        )
        .unwrap();
        let header = json!({"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes());
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    #[test]
    fn cli_accepts_json_output() {
        let cli =
            crate::Cli::try_parse_from(["hipfire", "inspect", "model.hfq", "--json"]).unwrap();
        match cli.command {
            Some(crate::Commands::Inspect(args)) => assert!(args.json),
            _ => panic!("expected inspect command"),
        }
    }

    #[test]
    fn hfq_json_reports_architecture_and_embedded_vision() {
        let dir = scratch("hfq");
        let path = dir.join("qwen3.8-27b.mq4-xts");
        write_hfq(
            &path,
            5,
            &["model.visual.patch_embed.proj.weight", "lm_head.weight"],
        );
        let report = inspect_json(&path).unwrap();
        assert_eq!(report["source"], "hfq");
        assert_eq!(report["arch_id"], 5);
        assert_eq!(report["architecture"], "qwen35");
        assert_eq!(report["tensor_count"], 2);
        assert_eq!(report["vision"]["present"], true);
        assert_eq!(report["speculative"]["dflash2"]["supported"], true);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn safetensors_json_reports_directory_limits() {
        let dir = scratch("safetensors");
        write_safetensors(&dir);
        let report = inspect_json(&dir).unwrap();
        assert_eq!(report["source"], "safetensors-dir");
        assert_eq!(report["arch_id"], 5);
        assert_eq!(report["vision"]["present"], false);
        assert_eq!(report["speculative"]["dflash2"]["supported"], false);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_non_model_file_is_named_and_refused() {
        let dir = scratch("invalid");
        let path = dir.join("not-a-model.bin");
        std::fs::write(&path, b"not a model").unwrap();
        let err = inspect_json(&path).unwrap_err();
        assert!(err.to_string().contains("not-a-model.bin"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
