// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Start-time KVMem option snapshot.
//!
//! The controller must not read the process environment. [`KvMemOptions`] is
//! built once by the engine from the resolved Legacy config plus the process
//! snapshot; `KvMemCtl::construct` only consumes the typed fields below.

use super::*;

/// Diagnostic switches. Defaults are all off; callers that need a probe inject
/// this struct instead of mutating process-global environment in tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KvMemDiagnostics {
    /// `HIPFIRE_KVMEM_TRACE`: policy/prefix/pack trace lines.
    pub(crate) trace: bool,
    /// `HIPFIRE_KVMEM_TRACE_KEEP`: per-step keep-set tracing.
    pub(crate) trace_keep: bool,
    /// `HIPFIRE_KVMEM_TRACE_PAGES`: page pass tracing.
    pub(crate) trace_pages: bool,
    /// `HIPFIRE_KVMEM_VERIFY`: spill/restore byte checks.
    pub(crate) verify: bool,
    /// `HIPFIRE_KVMEM_TIME`: host-side timing breakdowns.
    pub(crate) time: bool,
}

/// Typed start-time snapshot consumed by the controller.
///
/// Values are intentionally raw options: field validation and Legacy-over-env
/// precedence stay in [`KvMemCtl::construct`], except for the page-count
/// lexeme and strategy spelling, which fail closed here because they gate
/// admission. The engine is the only place that reads the process snapshot.
#[derive(Debug, Clone)]
pub(crate) struct KvMemOptions {
    pub(crate) strategy: RecallStrategy,
    pub(crate) plan_select_interval: Option<usize>,
    pub(crate) plan_pages: Option<usize>,
    pub(crate) plan_budget_tokens: Option<usize>,
    pub(crate) sink_blocks: Option<usize>,
    pub(crate) recent_blocks: Option<usize>,
    pub(crate) gen_reserve: Option<u64>,
    pub(crate) checkpoint_policy: Option<String>,
    pub(crate) checkpoint_count: Option<usize>,
    pub(crate) checkpoint_mib: Option<u64>,
    pub(crate) checkpoint_interval: Option<u64>,
    pub(crate) checkpoint_min_gap: Option<u64>,
    pub(crate) lrfu: Option<bool>,
    pub(crate) lrfu_lambda: Option<f64>,
    pub(crate) lrfu_top: Option<f64>,
    pub(crate) select_method: Option<String>,
    pub(crate) watermark_high: Option<f64>,
    pub(crate) watermark_low: Option<f64>,
    pub(crate) query_span: Option<usize>,
    pub(crate) layerwise_mtp_history: Option<String>,
    pub(crate) host_bytes: Option<u64>,
    pub(crate) content_groups: Option<usize>,
    pub(crate) content_cache_dir: Option<String>,
    pub(crate) content_cache_bytes: Option<u64>,
    /// `HIPFIRE_KVMEM_DN_IDLE`: demote an idle session's DeltaNet state to a
    /// host image and restore it on the next continuation. Default off.
    pub(crate) dn_idle: Option<bool>,
    /// `HIPFIRE_KVMEM_DN_RING_DIR`: mirror page-aligned DN checkpoints here so
    /// they survive a process restart. Defaults to `<cache dir>/dn-ring`.
    pub(crate) dn_ring_dir: Option<String>,
    /// `HIPFIRE_KVMEM_DN_DISK_MIB`: disk mirror cap; `0` disables persistence.
    pub(crate) dn_disk_mib: Option<u64>,
    pub(crate) spill_dir: Option<String>,
    pub(crate) spill_bytes: Option<u64>,
    pub(crate) layerwise_activation_ram_mib: Option<String>,
    pub(crate) layerwise_activation_disk_mib: Option<String>,
    pub(crate) layerwise_staging_mib: Option<String>,
    pub(crate) layerwise_metadata_mib: Option<String>,
    pub(crate) layerwise_activation_metadata_mib: Option<String>,
    pub(crate) batch_xfer: bool,
    pub(crate) diagnostics: KvMemDiagnostics,
}

impl KvMemOptions {
    /// Build the snapshot from the resolved Legacy config and a raw value
    /// getter. The getter is the engine's process snapshot; this function itself
    /// performs no I/O.
    pub(crate) fn resolve<F>(
        config: Option<&hipfire_config::kvmem::ResolvedConfig>,
        raw: F,
    ) -> Result<KvMemOptions, String>
    where
        F: FnMut(&str) -> Option<String>,
    {
        resolve_fields(config, raw)
    }
}

impl Default for KvMemOptions {
    fn default() -> Self {
        Self {
            strategy: RecallStrategy::Legacy,
            plan_select_interval: None,
            plan_pages: None,
            plan_budget_tokens: None,
            sink_blocks: None,
            recent_blocks: None,
            gen_reserve: None,
            checkpoint_policy: None,
            checkpoint_count: None,
            checkpoint_mib: None,
            checkpoint_interval: None,
            checkpoint_min_gap: None,
            lrfu: None,
            lrfu_lambda: None,
            lrfu_top: None,
            select_method: None,
            watermark_high: None,
            watermark_low: None,
            query_span: None,
            layerwise_mtp_history: None,
            host_bytes: None,
            content_groups: None,
            content_cache_dir: None,
            content_cache_bytes: None,
            dn_idle: None,
            dn_ring_dir: None,
            dn_disk_mib: None,
            spill_dir: None,
            spill_bytes: None,
            layerwise_activation_ram_mib: None,
            layerwise_activation_disk_mib: None,
            layerwise_staging_mib: None,
            layerwise_metadata_mib: None,
            layerwise_activation_metadata_mib: None,
            batch_xfer: true,
            diagnostics: KvMemDiagnostics::default(),
        }
    }
}

fn enabled(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("on") | Some("true"))
}

/// Preserve the two historical page-count contracts: the resolved-config
/// branch rejects any non-decimal or out-of-range value, while the plain
/// env branch keeps the old `env_usize` leniency (trim, invalid/empty = unset).
fn resolve_plan_pages(config_present: bool, raw: Option<String>) -> Result<Option<usize>, String> {
    if config_present {
        let Some(value) = raw else {
            return Ok(None);
        };
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(
                "kvmem: HIPFIRE_SLOTS_PAGED_PAGES requires a positive decimal integer".to_string(),
            );
        }
        let pages = value
            .parse::<usize>()
            .map_err(|_| "kvmem: physical page count overflow".to_string())?;
        if pages == 0 || pages > i32::MAX as usize {
            return Err("kvmem: physical page count must be in 1..=2147483647".to_string());
        }
        Ok(Some(pages))
    } else {
        Ok(raw
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .and_then(|v| v.parse::<usize>().ok()))
    }
}

/// Build the snapshot from the resolved Legacy config and a raw value getter.
/// The getter is the engine's process snapshot; this function itself performs
/// no I/O and tests inject an explicit snapshot instead of mutating environment.
fn resolve_fields<F>(
    config: Option<&hipfire_config::kvmem::ResolvedConfig>,
    mut raw: F,
) -> Result<KvMemOptions, String>
where
    F: FnMut(&str) -> Option<String>,
{
    let plan_pages_raw = raw("HIPFIRE_SLOTS_PAGED_PAGES");
    let mut get = |name: &str| {
        raw(name)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let select_interval = get("HIPFIRE_KVMEM_SELECT_INTERVAL").and_then(|v| v.parse().ok());
    let plan_pages = resolve_plan_pages(config.is_some(), plan_pages_raw)?;
    let plan_budget_tokens = get("HIPFIRE_KVMEM_BUDGET_TOKENS").and_then(|v| v.parse().ok());
    let strategy = match config {
        // A resolved Legacy config owns the public contract; the controller
        // already rejects any non-Legacy resolution for it.
        Some(_) => RecallStrategy::Legacy,
        None => resolve_recall_strategy(get("HIPFIRE_KVMEM_RECALL_STRATEGY"))?,
    };
    Ok(KvMemOptions {
        strategy,
        plan_select_interval: select_interval,
        plan_pages,
        plan_budget_tokens,
        sink_blocks: get("HIPFIRE_KVMEM_SINK_BLOCKS").and_then(|v| v.parse().ok()),
        recent_blocks: get("HIPFIRE_KVMEM_RECENT_BLOCKS").and_then(|v| v.parse().ok()),
        gen_reserve: get("HIPFIRE_KVMEM_GEN_RESERVE").and_then(|v| v.parse().ok()),
        checkpoint_policy: get("HIPFIRE_KVMEM_CHECKPOINT_POLICY"),
        checkpoint_count: get("HIPFIRE_KVMEM_CHECKPOINT_COUNT").and_then(|v| v.parse().ok()),
        checkpoint_mib: get("HIPFIRE_KVMEM_CHECKPOINT_MIB").and_then(|v| v.parse().ok()),
        checkpoint_interval: get("HIPFIRE_KVMEM_CHECKPOINT_INTERVAL").and_then(|v| v.parse().ok()),
        checkpoint_min_gap: get("HIPFIRE_KVMEM_CHECKPOINT_MIN_GAP").and_then(|v| v.parse().ok()),
        lrfu: get("HIPFIRE_KVMEM_LRFU").map(|v| enabled(Some(&v))),
        lrfu_lambda: get("HIPFIRE_KVMEM_LRFU_LAMBDA").and_then(|v| v.parse().ok()),
        lrfu_top: get("HIPFIRE_KVMEM_LRFU_TOP").and_then(|v| v.parse().ok()),
        select_method: get("HIPFIRE_KVMEM_SELECT_METHOD"),
        watermark_high: get("HIPFIRE_KVMEM_GPU_HIGH_WATERMARK").and_then(|v| v.parse().ok()),
        watermark_low: get("HIPFIRE_KVMEM_GPU_LOW_WATERMARK").and_then(|v| v.parse().ok()),
        query_span: get("HIPFIRE_KVMEM_QUERY_SPAN").and_then(|v| v.parse().ok()),
        layerwise_mtp_history: get("HIPFIRE_KVMEM_LAYERWISE_MTP_HISTORY"),
        host_bytes: get("HIPFIRE_KVMEM_HOST_BYTES").and_then(|v| v.parse().ok()),
        content_groups: get("HIPFIRE_KVMEM_CONTENT_GROUPS").and_then(|v| v.parse().ok()),
        content_cache_dir: get("HIPFIRE_KVMEM_CACHE_DIR"),
        content_cache_bytes: get("HIPFIRE_KVMEM_CACHE_BYTES").and_then(|v| v.parse().ok()),
        dn_idle: get("HIPFIRE_KVMEM_DN_IDLE").map(|v| enabled(Some(&v))),
        dn_ring_dir: get("HIPFIRE_KVMEM_DN_RING_DIR"),
        dn_disk_mib: get("HIPFIRE_KVMEM_DN_DISK_MIB").and_then(|v| v.parse().ok()),
        spill_dir: get("HIPFIRE_KVMEM_SPILL_DIR"),
        spill_bytes: get("HIPFIRE_KVMEM_SPILL_BYTES").and_then(|v| v.parse().ok()),
        layerwise_activation_ram_mib: get("HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_RAM_MIB"),
        layerwise_activation_disk_mib: get("HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_DISK_MIB"),
        layerwise_staging_mib: get("HIPFIRE_KVMEM_LAYERWISE_STAGING_MIB"),
        layerwise_metadata_mib: get("HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB"),
        layerwise_activation_metadata_mib: get("HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_METADATA_MIB"),
        batch_xfer: !matches!(
            get("HIPFIRE_KVMEM_BATCH_XFER").as_deref(),
            Some("0") | Some("off") | Some("false")
        ),
        diagnostics: KvMemDiagnostics {
            trace: enabled(get("HIPFIRE_KVMEM_TRACE").as_deref()),
            trace_keep: enabled(get("HIPFIRE_KVMEM_TRACE_KEEP").as_deref()),
            trace_pages: enabled(get("HIPFIRE_KVMEM_TRACE_PAGES").as_deref()),
            verify: enabled(get("HIPFIRE_KVMEM_VERIFY").as_deref()),
            time: enabled(get("HIPFIRE_KVMEM_TIME").as_deref()),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(pairs: &[(&str, &str)]) -> Result<KvMemOptions, String> {
        KvMemOptions::resolve(None, |name| {
            pairs
                .iter()
                .find_map(|(key, value)| (*key == name).then(|| (*value).to_string()))
        })
    }

    #[test]
    fn snapshot_captures_policy_and_diagnostics_without_io() {
        let options = snapshot(&[
            ("HIPFIRE_KVMEM_RECALL_STRATEGY", "layerwise"),
            ("HIPFIRE_KVMEM_SELECT_INTERVAL", "32"),
            ("HIPFIRE_SLOTS_PAGED_PAGES", "4096"),
            ("HIPFIRE_KVMEM_BUDGET_TOKENS", "65536"),
            ("HIPFIRE_KVMEM_TRACE", "on"),
            ("HIPFIRE_KVMEM_VERIFY", "1"),
            ("HIPFIRE_KVMEM_BATCH_XFER", "off"),
            ("HIPFIRE_KVMEM_DN_IDLE", "on"),
            ("HIPFIRE_KVMEM_DN_RING_DIR", "/tmp/dn-ring"),
            ("HIPFIRE_KVMEM_DN_DISK_MIB", "256"),
        ])
        .expect("snapshot");
        assert_eq!(options.strategy, RecallStrategy::Layerwise);
        assert_eq!(options.plan_select_interval, Some(32));
        assert_eq!(options.plan_pages, Some(4096));
        assert_eq!(options.plan_budget_tokens, Some(65536));
        assert!(options.diagnostics.trace);
        assert!(options.diagnostics.verify);
        assert!(!options.diagnostics.trace_keep);
        assert!(!options.batch_xfer);
        assert_eq!(options.dn_idle, Some(true));
        assert_eq!(options.dn_ring_dir.as_deref(), Some("/tmp/dn-ring"));
        assert_eq!(options.dn_disk_mib, Some(256));
    }

    #[test]
    fn plan_pages_config_keeps_the_strict_lexeme_contract() {
        for bad in [
            "",
            "0",
            "-1",
            "+1",
            "1.5",
            " 1",
            "1 ",
            "１２８",
            "2147483648",
        ] {
            assert!(
                resolve_plan_pages(true, Some(bad.into())).is_err(),
                "pages={bad:?}"
            );
        }
        assert_eq!(
            resolve_plan_pages(true, Some("2147483647".into())).unwrap(),
            Some(i32::MAX as usize)
        );
        assert_eq!(resolve_plan_pages(true, None).unwrap(), None);
    }

    #[test]
    fn plan_pages_env_keeps_the_old_lenient_contract() {
        for (value, expected) in [
            (" 1 ", Some(1)),
            ("+1", Some(1)),
            ("0", Some(0)),
            ("", None),
            ("invalid", None),
            ("999999999999999999999999999999999999", None),
        ] {
            assert_eq!(
                resolve_plan_pages(false, Some(value.into())).unwrap(),
                expected,
                "pages={value:?}"
            );
        }
        assert_eq!(resolve_plan_pages(false, None).unwrap(), None);
    }

    #[test]
    fn strategy_spelling_is_fail_closed() {
        assert!(snapshot(&[("HIPFIRE_KVMEM_RECALL_STRATEGY", "layers")]).is_err());
    }
}
