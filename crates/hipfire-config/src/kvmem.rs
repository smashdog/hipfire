// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure Legacy KVMem control-plane resolution. No environment, filesystem, GPU,
//! or process-global reads: callers supply the raw, presence-preserving layers.
//! Native consumers use `ResolvedConfig`, not a second compatibility parser.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const CONTRACT_VERSION: u32 = 1;
pub const BLOCK_TOKENS: u32 = 128;
pub const INTEGER_MAX: u64 = 2_147_483_647;
const MIB: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Default,
    ImplicitBudget,
    /// An older IPC envelope did not retain env-versus-developer provenance.
    ProcessSnapshot,
    Developer,
    Environment,
    Cli,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawValue {
    pub value: String,
    pub source: Source,
    /// Original flag/env/developer key; also identifies legacy byte/block units.
    pub spelling: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    /// Canonical field names; only explicitly present inputs appear here.
    pub values: BTreeMap<String, RawValue>,
    /// Old policy controls are not aliases for the new Query/session controls.
    pub compatibility: BTreeMap<String, RawValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum QueryScorer {
    PageSoftmax = 0,
    MeanQCosine = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum ScoreAgg {
    Mean = 0,
    Max = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecallStrategy {
    Legacy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueReport {
    /// Exact input lexeme (in the original spelling's units), or default lexeme.
    pub requested: String,
    pub effective: serde_json::Value,
    pub source: Source,
    pub spelling: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedConfig {
    pub contract_version: u32,
    pub enabled: bool,
    pub budget_tokens: u32,
    pub gen_reserve_tokens: u32,
    pub sink_tokens: u32,
    pub recent_tokens: u32,
    pub host_bytes: u64,
    pub disk_bytes: u64,
    pub disk_path: Option<PathBuf>,
    pub sessions: u32,
    pub cache_min_tokens: u32,
    pub block_tokens: u32,
    pub recall_strategy: RecallStrategy,
    pub query_scorer: QueryScorer,
    pub score_agg: ScoreAgg,
    pub lrfu: bool,
    pub lrfu_lambda: f64,
    pub lrfu_top_fraction: f64,
    pub graph_alpha: f64,
    pub query_max_tokens: u32,
    pub query_replay_max_tokens: u32,
    pub requested: RawConfig,
    /// Per-field requested/effective/source evidence, including unit conversions.
    pub report: BTreeMap<String, ValueReport>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("KVMem {parameter}: {message}")]
pub struct ResolveError {
    pub parameter: String,
    pub message: String,
}

type Result<T> = std::result::Result<T, ResolveError>;

fn error(parameter: impl Into<String>, message: impl Into<String>) -> ResolveError {
    ResolveError {
        parameter: parameter.into(),
        message: message.into(),
    }
}

// Canonical field, public CLI spelling, current/compatibility environment suffixes.
const FIELDS: &[(&str, &str, &[&str])] = &[
    ("enabled", "--kvmem", &["", "ENABLED"]),
    (
        "budget_tokens",
        "--kvmem-budget",
        &["BUDGET", "BUDGET_TOKENS"],
    ),
    (
        "gen_reserve_tokens",
        "--kvmem-gen-reserve",
        &["GEN_RESERVE"],
    ),
    (
        "sink_tokens",
        "--kvmem-sink-tokens",
        &["SINK_TOKENS", "SINK_BLOCKS"],
    ),
    (
        "recent_tokens",
        "--kvmem-recent-tokens",
        &["RECENT_TOKENS", "RECENT_BLOCKS"],
    ),
    (
        "host_bytes",
        "--kvmem-host-mib",
        &["HOST_MIB", "HOST_BYTES"],
    ),
    (
        "disk_bytes",
        "--kvmem-disk-mib",
        &["DISK_MIB", "SPILL_BYTES"],
    ),
    (
        "disk_path",
        "--kvmem-disk-path",
        &["DISK_PATH", "SPILL_DIR"],
    ),
    ("sessions", "--kvmem-sessions", &["SESSIONS"]),
    (
        "cache_min_tokens",
        "--kvmem-cache-min-tokens",
        &["CACHE_MIN_TOKENS"],
    ),
    ("block_tokens", "--kvmem-block-tokens", &["BLOCK_TOKENS"]),
    (
        "recall_strategy",
        "--kvmem-recall-strategy",
        &["RECALL_STRATEGY"],
    ),
    ("query_scorer", "--kvmem-query-scorer", &["QUERY_SCORER"]),
    ("score_agg", "--kvmem-score-agg", &["SCORE_AGG"]),
    ("lrfu", "--kvmem-lrfu", &["LRFU"]),
    ("lrfu_lambda", "--kvmem-lrfu-lambda", &["LRFU_LAMBDA"]),
    (
        "lrfu_top_fraction",
        "--kvmem-lrfu-top-frac",
        &["LRFU_TOP_FRAC", "LRFU_TOP"],
    ),
    ("graph_alpha", "--kvmem-graph-alpha", &["GRAPH_ALPHA"]),
    (
        "query_max_tokens",
        "--kvmem-query-max-tokens",
        &["QUERY_MAX_TOKENS"],
    ),
    (
        "query_replay_max_tokens",
        "--kvmem-query-replay-max-tokens",
        &["QUERY_REPLAY_MAX_TOKENS"],
    ),
];

pub const RETIRED_STORAGE_FLAGS: &[&str] = &[
    "--kvmem-nvme-gb",
    "--kvmem-nvme-dir",
    "--kvmem-raw-k-nvme",
    "--kvmem-session-nvme-gb",
    "--kvmem-session-cache-dir",
];

pub fn reject_retired_storage_flag(name: &str) -> Result<()> {
    let name = name.split('=').next().unwrap_or(name);
    if RETIRED_STORAGE_FLAGS.contains(&name)
        || name.starts_with("--kvmem-nvme-")
        || name.starts_with("--kvmem-session-nvme-")
    {
        return Err(error(name, "retired; use --kvmem-host-mib H with --kvmem-disk-mib D --kvmem-disk-path PATH (no automatic conversion)"));
    }
    Ok(())
}

fn env_name(spelling: &str) -> String {
    crate::developer_env_for_key(spelling).unwrap_or_else(|| spelling.to_owned())
}

fn env_suffix(spelling: &str) -> Option<String> {
    let name = env_name(spelling);
    if name == "HIPFIRE_KVMEM" {
        Some(String::new())
    } else {
        name.strip_prefix("HIPFIRE_KVMEM_").map(str::to_owned)
    }
}

fn field_for(spelling: &str) -> Option<&'static str> {
    if spelling == "--no-kvmem" {
        return Some("enabled");
    }
    if spelling == "--no-kvmem-lrfu" {
        return Some("lrfu");
    }
    let suffix = env_suffix(spelling);
    FIELDS.iter().find_map(|(key, flag, names)| {
        (*flag == spelling || suffix.as_deref().is_some_and(|s| names.contains(&s))).then_some(*key)
    })
}

fn compatibility_key(spelling: &str) -> Option<String> {
    let suffix = env_suffix(spelling)?;
    (matches!(
        suffix.as_str(),
        "QUERY_SPAN"
            | "CONTENT_GROUPS"
            | "SELECT_METHOD"
            | "GPU_HIGH_WATERMARK"
            | "GPU_LOW_WATERMARK"
    ) || suffix.starts_with("LAYERWISE_"))
    .then_some(suffix)
}

impl RawConfig {
    /// Explicit CLI wins over environment, then developer config. Unknown CLI
    /// names fail closed; unrelated environment/developer keys are ignored.
    pub fn from_maps(
        cli: &BTreeMap<String, String>,
        environment: &BTreeMap<String, String>,
        developer: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let mut raw = Self::default();
        for (map, source) in [
            (developer, Source::Developer),
            (environment, Source::Environment),
            (cli, Source::Cli),
        ] {
            for (spelling, value) in map {
                raw.insert(spelling, value, source)?;
            }
        }
        Ok(raw)
    }

    /// Preserve the provenance of the already-resolved global developer layer;
    /// this reads neither `ACTIVE_PROCESS_CONFIG` nor the ambient environment.
    pub fn from_global(
        global: &crate::ResolvedConfig,
        cli: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let mut raw = Self::default();
        for (key, value) in &global.values {
            if !crate::is_developer_key(key) {
                continue;
            }
            let text = snapshot_value(key, &value.value);
            let source = if matches!(value.source, crate::ConfigSource::LegacyEnv { .. }) {
                Source::Environment
            } else {
                Source::Developer
            };
            let spelling = if source == Source::Environment {
                env_name(key)
            } else {
                key.clone()
            };
            raw.insert(&spelling, &text, source)?;
        }
        for (key, value) in cli {
            raw.insert(key, value, Source::Cli)?;
        }
        Ok(raw)
    }

    /// Compatibility for old JSONL clients: their developer map is already a
    /// process-start snapshot, not permission to re-read the daemon environment.
    pub fn from_process_values(values: &crate::ConfigLayer) -> Result<Self> {
        let mut raw = Self::default();
        for (key, value) in &values.values {
            raw.insert(key, &snapshot_value(key, value), Source::ProcessSnapshot)?;
        }
        Ok(raw)
    }

    fn insert(&mut self, spelling: &str, text: &str, source: Source) -> Result<()> {
        reject_retired_storage_flag(spelling)?;
        let (key, compatibility) = if let Some(key) = field_for(spelling) {
            (key.to_owned(), false)
        } else if let Some(key) = compatibility_key(spelling) {
            (key, true)
        } else if source == Source::Cli {
            return Err(error(spelling, "unknown parameter"));
        } else {
            return Ok(());
        };
        let values = if compatibility {
            &mut self.compatibility
        } else {
            &mut self.values
        };
        if let Some(previous) = values.get(&key) {
            if previous.source > source {
                return Ok(());
            }
            if previous.source == source {
                return Err(error(
                    spelling,
                    format!(
                        "conflicts with {} in the same input layer",
                        previous.spelling
                    ),
                ));
            }
        }
        let text = match spelling {
            "--no-kvmem" | "--no-kvmem-lrfu" => "0",
            _ => text,
        };
        values.insert(
            key,
            RawValue {
                value: text.into(),
                source,
                spelling: spelling.into(),
            },
        );
        Ok(())
    }

    fn value(&self, key: &str, default: &str) -> RawValue {
        self.values.get(key).cloned().unwrap_or_else(|| RawValue {
            value: default.into(),
            source: Source::Default,
            spelling: FIELDS.iter().find(|f| f.0 == key).unwrap().1.into(),
        })
    }

    fn integer(&self, key: &str, default: &str, minimum: u64) -> Result<u64> {
        let raw = self.value(key, default);
        let suffix = env_suffix(&raw.spelling).unwrap_or_default();
        // The public integer bound is in MiB. Old HOST/SPILL_BYTES retain their
        // byte unit, so permit exactly the corresponding maximum byte budget.
        let legacy_bytes = matches!(suffix.as_str(), "HOST_BYTES" | "SPILL_BYTES");
        let limit = if legacy_bytes {
            INTEGER_MAX * MIB
        } else {
            INTEGER_MAX
        };
        let mut value = integer(&raw, minimum, limit)?;
        if matches!(suffix.as_str(), "SINK_BLOCKS" | "RECENT_BLOCKS") {
            value = value
                .checked_mul(u64::from(BLOCK_TOKENS))
                .filter(|v| *v <= INTEGER_MAX)
                .ok_or_else(|| {
                    error(
                        &raw.spelling,
                        "block-to-token conversion exceeds 2147483647",
                    )
                })?;
        }
        if matches!(key, "host_bytes" | "disk_bytes") && !legacy_bytes {
            value = value
                .checked_mul(MIB)
                .ok_or_else(|| error(&raw.spelling, "MiB-to-byte overflow"))?;
        }
        Ok(value)
    }

    /// `None` means the existing env/developer-only Layerwise route owns its
    /// policy. No Legacy defaults are injected into that route. Formal flags
    /// cannot silently switch strategy or alter Layerwise's old algorithm.
    pub fn resolve(&self) -> Result<Option<ResolvedConfig>> {
        self.validate_raw_shape()?;
        // Environment-only tuning spellings *without* an explicit HIPFIRE_KVMEM
        // are the historical env route's inert knobs (`*_HOST_BYTES`,
        // `*_CONTENT_GROUPS`, ...): they never enabled KVMem on their own, and
        // old binaries simply ignored them. Delegate those to the env route so a
        // stray exported value cannot poison the process policy. An explicit
        // HIPFIRE_KVMEM (either value) opts into typed validation, so the same
        // spellings share the one contract; new formal env spellings without an
        // enable fail closed below.
        let env_only_legacy = !self.values.contains_key("enabled")
            && !self.values.is_empty()
            && self
                .values
                .values()
                .all(|raw| raw.source == Source::Environment)
            && self.values.values().all(|raw| {
                env_suffix(&raw.spelling).is_some_and(|suffix| {
                    matches!(
                        suffix.as_str(),
                        "" | "BUDGET_TOKENS"
                            | "GEN_RESERVE"
                            | "SINK_BLOCKS"
                            | "RECENT_BLOCKS"
                            | "HOST_BYTES"
                            | "SPILL_BYTES"
                            | "SPILL_DIR"
                            | "LRFU"
                            | "LRFU_LAMBDA"
                            | "LRFU_TOP"
                            | "RECALL_STRATEGY"
                    )
                })
            });
        if env_only_legacy {
            return Ok(None);
        }
        let strategy = self.value("recall_strategy", "legacy");
        let layerwise = match strategy.value.as_str() {
            "legacy" => false,
            "layerwise" => true,
            "0" if strategy.source != Source::Cli => false,
            "1" if strategy.source != Source::Cli => true,
            _ => return Err(error(&strategy.spelling, "expected legacy or layerwise")),
        };
        if layerwise {
            for raw in self.values.values() {
                // Only pre-existing env/developer spellings can delegate to
                // Layerwise. New public controls have no Layerwise consumer.
                let suffix = env_suffix(&raw.spelling).unwrap_or_default();
                if raw.source == Source::Cli
                    || !matches!(
                        suffix.as_str(),
                        "" | "BUDGET_TOKENS"
                            | "GEN_RESERVE"
                            | "SINK_BLOCKS"
                            | "RECENT_BLOCKS"
                            | "HOST_BYTES"
                            | "SPILL_BYTES"
                            | "SPILL_DIR"
                            | "LRFU"
                            | "LRFU_LAMBDA"
                            | "LRFU_TOP"
                            | "RECALL_STRATEGY"
                    )
                {
                    return Err(error(&raw.spelling, "UNSUPPORTED: formal Legacy controls cannot be combined with Layerwise; retain the existing env-only Layerwise path"));
                }
            }
            return Ok(None);
        }

        let budget_tokens =
            self.integer("budget_tokens", "32768", 1)? as u32 / BLOCK_TOKENS * BLOCK_TOKENS;
        let gen_reserve_tokens =
            self.integer("gen_reserve_tokens", "1024", 1)? as u32 / BLOCK_TOKENS * BLOCK_TOKENS;
        let sink_tokens = self.integer("sink_tokens", "0", 0)? as u32 / BLOCK_TOKENS * BLOCK_TOKENS;
        let recent_tokens =
            self.integer("recent_tokens", "64", 0)? as u32 / BLOCK_TOKENS * BLOCK_TOKENS;
        let host_bytes = self.integer("host_bytes", "16384", 1)?;
        let disk_bytes = self.integer("disk_bytes", "0", 0)?;
        let sessions = self.integer("sessions", "4", 1)? as u32;
        let cache_min_tokens = self.integer("cache_min_tokens", "0", 0)? as u32;
        let block_tokens = self.integer("block_tokens", "128", 1)? as u32;
        let query_max_tokens = self.integer("query_max_tokens", "512", 1)? as u32;
        let query_replay_max_tokens = self.integer("query_replay_max_tokens", "2048", 1)? as u32;
        let query_scorer_raw = self.value("query_scorer", "page-softmax");
        let query_scorer = match query_scorer_raw.value.as_str() {
            "meanq-cosine" => QueryScorer::MeanQCosine,
            "page-softmax" => QueryScorer::PageSoftmax,
            _ => {
                return Err(error(
                    &query_scorer_raw.spelling,
                    "expected meanq-cosine or page-softmax",
                ))
            }
        };
        let cosine = query_scorer == QueryScorer::MeanQCosine;
        // Max preserves a retrieval signal that is strong in a critical layer
        // instead of diluting it across layers; 32k NIAH at a 4k residency
        // budget exposed stable misses with Mean on Swift 27B.
        let agg_default = "max";
        let agg_raw = self.value("score_agg", agg_default);
        let score_agg = match agg_raw.value.as_str() {
            "mean" => ScoreAgg::Mean,
            "max" => ScoreAgg::Max,
            _ => return Err(error(&agg_raw.spelling, "expected mean or max")),
        };
        let lrfu_default = if cosine { "1" } else { "0" };
        let lrfu = boolean(&self.value("lrfu", lrfu_default))?;
        let lrfu_lambda = real(&self.value("lrfu_lambda", "0.5"), true)?;
        let lrfu_top_fraction = real(&self.value("lrfu_top_fraction", "0.9"), false)?;
        let graph_alpha = real(&self.value("graph_alpha", "0"), true)?;
        let implicit_enabled = self.values.contains_key("budget_tokens");
        let enabled_default = if implicit_enabled { "1" } else { "0" };
        let enabled = boolean(&self.value("enabled", enabled_default))?;
        let disk_path = match self.values.get("disk_path") {
            Some(v) if v.value.is_empty() => return Err(error(&v.spelling, "must not be empty")),
            Some(v) => Some(PathBuf::from(&v.value)),
            None => None,
        };

        if budget_tokens < 256 {
            return Err(error(
                "--kvmem-budget",
                "floor to G=128 must leave at least 256 tokens",
            ));
        }
        if gen_reserve_tokens < 128 {
            return Err(error(
                "--kvmem-gen-reserve",
                "floor to G=128 must leave at least 128 tokens",
            ));
        }
        if u64::from(sink_tokens) + u64::from(recent_tokens) > u64::from(budget_tokens) {
            return Err(error(
                "--kvmem-sink-tokens/--kvmem-recent-tokens",
                "normalized S+T exceeds B; protection is never silently reduced",
            ));
        }
        if (disk_bytes > 0) != disk_path.is_some() {
            return Err(error(
                "--kvmem-disk-mib/--kvmem-disk-path",
                "positive D requires a path; D=0 forbids a path",
            ));
        }
        if !lrfu
            && (self.values.contains_key("lrfu_lambda")
                || self.values.contains_key("lrfu_top_fraction"))
        {
            return Err(error(
                "--kvmem-lrfu",
                "LRFU tuning requires enabled LRFU; remove --no-kvmem-lrfu or its tuning",
            ));
        }
        if graph_alpha != 0.0 {
            return Err(error(
                "--kvmem-graph-alpha",
                "UNSUPPORTED: native graph scoring is not implemented; only 0 is supported",
            ));
        }
        if block_tokens != BLOCK_TOKENS {
            return Err(error(
                "--kvmem-block-tokens",
                "UNSUPPORTED: this backend has fixed G=128",
            ));
        }
        if cache_min_tokens != 0 {
            return Err(error(
                "--kvmem-cache-min-tokens",
                "UNSUPPORTED: no native cache-min consumer; only 0 is supported",
            ));
        }
        if !enabled
            && self
                .values
                .keys()
                .any(|key| !matches!(key.as_str(), "enabled" | "lrfu" | "disk_bytes"))
        {
            return Err(error(
                "--kvmem",
                "settings require --kvmem or a positive --kvmem-budget; remove --no-kvmem",
            ));
        }
        if !enabled && (lrfu || disk_bytes > 0) {
            return Err(error("--kvmem", "LRFU/SSD settings require enabled KVMem"));
        }
        let mut notes = vec![
            "Native difference: graph-alpha defaults to 0 (public page-softmax default is 0.2); nonzero is UNSUPPORTED.".into(),
            "G=128; B/R/S/T floor to whole blocks; B>=256, R>=128; S/T=0 means literal zero protected tokens, not one page.".into(),
            "Q/replay limits are positive (defaults 512/2048); zero is not unlimited. QUERY_SPAN and CONTENT_GROUPS are not aliases.".into(),
            "This is configuration validation only: native C/pool/scratch admission, SSD path writability, Query, sessions and payload behavior require native validation.".into(),
        ];
        if implicit_enabled && !self.values.contains_key("enabled") {
            notes.push(
                "enabled is implicit from the explicitly present positive budget_tokens input."
                    .into(),
            );
        }
        self.validate_compatibility(&mut notes)?;
        let mut resolved = ResolvedConfig {
            contract_version: CONTRACT_VERSION,
            enabled,
            budget_tokens,
            gen_reserve_tokens,
            sink_tokens,
            recent_tokens,
            host_bytes,
            disk_bytes,
            disk_path,
            sessions,
            cache_min_tokens,
            block_tokens,
            recall_strategy: RecallStrategy::Legacy,
            query_scorer,
            score_agg,
            lrfu,
            lrfu_lambda,
            lrfu_top_fraction,
            graph_alpha,
            query_max_tokens,
            query_replay_max_tokens,
            requested: self.clone(),
            report: BTreeMap::new(),
            notes,
        };
        let effective = serde_json::to_value(&resolved).expect("finite serializable config");
        for (key, default) in [
            ("enabled", enabled_default),
            ("budget_tokens", "32768"),
            ("gen_reserve_tokens", "1024"),
            ("sink_tokens", "0"),
            ("recent_tokens", "64"),
            ("host_bytes", "16384"),
            ("disk_bytes", "0"),
            ("disk_path", ""),
            ("sessions", "4"),
            ("cache_min_tokens", "0"),
            ("block_tokens", "128"),
            ("recall_strategy", "legacy"),
            ("query_scorer", "page-softmax"),
            ("score_agg", agg_default),
            ("lrfu", lrfu_default),
            ("lrfu_lambda", "0.5"),
            ("lrfu_top_fraction", "0.9"),
            ("graph_alpha", "0"),
            ("query_max_tokens", "512"),
            ("query_replay_max_tokens", "2048"),
        ] {
            let mut raw = self.value(key, default);
            if key == "enabled" && implicit_enabled && !self.values.contains_key("enabled") {
                raw.source = Source::ImplicitBudget;
                raw.value = "absent (implicit from positive budget)".into();
                raw.spelling = self.values["budget_tokens"].spelling.clone();
            }
            resolved.report.insert(
                key.into(),
                ValueReport {
                    requested: raw.value,
                    effective: effective[key].clone(),
                    source: raw.source,
                    spelling: raw.spelling,
                },
            );
        }
        Ok(Some(resolved))
    }

    fn validate_raw_shape(&self) -> Result<()> {
        for (key, raw) in &self.values {
            if field_for(&raw.spelling) != Some(key.as_str())
                || matches!(raw.source, Source::Default | Source::ImplicitBudget)
            {
                return Err(error(key, "invalid raw input provenance or field spelling"));
            }
            if matches!(raw.spelling.as_str(), "--no-kvmem" | "--no-kvmem-lrfu") && raw.value != "0"
            {
                return Err(error(
                    &raw.spelling,
                    "negative flags must retain their explicit false value",
                ));
            }
        }
        for (key, raw) in &self.compatibility {
            if compatibility_key(&raw.spelling).as_deref() != Some(key.as_str())
                || matches!(raw.source, Source::Default | Source::ImplicitBudget)
            {
                return Err(error(key, "invalid compatibility input provenance"));
            }
        }
        Ok(())
    }

    fn validate_compatibility(&self, notes: &mut Vec<String>) -> Result<()> {
        for (key, raw) in &self.compatibility {
            match key.as_str() {
                "QUERY_SPAN" => {
                    if integer(raw, 1, 512)? != 1 {
                        return Err(error(&raw.spelling, "UNSUPPORTED with formal Legacy scoring: QUERY_SPAN is a legacy scoring ring, not --kvmem-query-max-tokens"));
                    }
                }
                "CONTENT_GROUPS" => {
                    if integer(raw, 0, 1024)? != 0 {
                        return Err(error(&raw.spelling, "UNSUPPORTED with Legacy session retention: CONTENT_GROUPS is not --kvmem-sessions"));
                    }
                }
                "SELECT_METHOD" if matches!(raw.value.as_str(), "retrieval" | "0") => {}
                _ => return Err(error(&raw.spelling, "UNSUPPORTED with the formal Legacy policy; remove this old policy override (no silent strategy conversion)")),
            }
            notes.push(format!(
                "{}={} is a legacy compatibility control, not a formal Query/session alias.",
                raw.spelling, raw.value
            ));
        }
        Ok(())
    }
}

fn snapshot_value(key: &str, value: &crate::ConfigValue) -> String {
    match value {
        crate::ConfigValue::String(value) => value.clone(),
        crate::ConfigValue::Integer(value) => value.to_string(),
        // Do not turn a developer TOML float 1.0 into the integer lexeme "1".
        crate::ConfigValue::Float(value) => format!("{value:?}"),
        crate::ConfigValue::Bool(value) if matches!(field_for(key), Some("enabled" | "lrfu")) => {
            if *value {
                "1".into()
            } else {
                "0".into()
            }
        }
        crate::ConfigValue::Bool(value) => value.to_string(),
        // Preserve explicit presence: null must not become a missing default.
        crate::ConfigValue::Null => String::new(),
    }
}

fn integer(raw: &RawValue, minimum: u64, maximum: u64) -> Result<u64> {
    if raw.value.is_empty() || !raw.value.bytes().all(|c| c.is_ascii_digit()) {
        return Err(error(
            &raw.spelling,
            "requires a nonempty ASCII unsigned decimal integer",
        ));
    }
    let value = raw
        .value
        .parse::<u64>()
        .ok()
        .filter(|n| *n <= maximum)
        .ok_or_else(|| error(&raw.spelling, format!("exceeds {maximum}")))?;
    if value < minimum {
        return Err(error(&raw.spelling, "must be positive"));
    }
    Ok(value)
}

fn boolean(raw: &RawValue) -> Result<bool> {
    match raw.value.as_str() {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        _ => Err(error(&raw.spelling, "expected 1/0, true/false or on/off")),
    }
}

fn real(raw: &RawValue, allow_zero: bool) -> Result<f64> {
    let value =
        raw.value.parse::<f64>().ok().filter(|v| {
            v.is_finite() && *v <= 1.0 && if allow_zero { *v >= 0.0 } else { *v > 0.0 }
        });
    value.ok_or_else(|| {
        error(
            &raw.spelling,
            if allow_zero {
                "requires a finite number in [0,1]"
            } else {
                "requires a finite number in (0,1]"
            },
        )
    })
}

impl ResolvedConfig {
    /// Re-resolve the supplied evidence, not env, so IPC cannot smuggle a
    /// different effective policy or bypass validation using a typed payload.
    pub fn validate(&self) -> Result<()> {
        if self.contract_version != CONTRACT_VERSION {
            return Err(error(
                "contract_version",
                format!(
                    "unsupported {}; expected {CONTRACT_VERSION}",
                    self.contract_version
                ),
            ));
        }
        if self.requested.resolve()?.as_ref() != Some(self) {
            return Err(error(
                "resolved config",
                "effective values/report differ from strict resolution of requested inputs",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }
    fn resolve(pairs: &[(&str, &str)]) -> Result<ResolvedConfig> {
        RawConfig::from_maps(&map(pairs), &BTreeMap::new(), &BTreeMap::new())?
            .resolve()
            .map(|v| v.unwrap())
    }

    #[test]
    fn v01_strict_integer_lexemes_and_bounds() {
        for (key, flag, _) in FIELDS {
            if !matches!(
                *key,
                "budget_tokens"
                    | "gen_reserve_tokens"
                    | "sink_tokens"
                    | "recent_tokens"
                    | "host_bytes"
                    | "disk_bytes"
                    | "sessions"
                    | "cache_min_tokens"
                    | "block_tokens"
                    | "query_max_tokens"
                    | "query_replay_max_tokens"
            ) {
                continue;
            }
            for bad in [
                "-1",
                "+1",
                "1.5",
                "",
                "no",
                "2147483648",
                " 1",
                "1 ",
                "１２８",
                "999999999999999999999999999999999999",
            ] {
                assert!(
                    resolve(&[("--kvmem", "1"), (flag, bad)]).is_err(),
                    "{flag}={bad}"
                );
            }
        }
        for flag in [
            "--kvmem-budget",
            "--kvmem-gen-reserve",
            "--kvmem-host-mib",
            "--kvmem-sessions",
            "--kvmem-query-max-tokens",
            "--kvmem-query-replay-max-tokens",
        ] {
            assert!(resolve(&[("--kvmem", "1"), (flag, "0")]).is_err());
        }
        let max = resolve(&[
            ("--kvmem", "1"),
            ("--kvmem-budget", "2147483647"),
            ("--kvmem-gen-reserve", "2147483647"),
            ("--kvmem-host-mib", "2147483647"),
            ("--kvmem-sessions", "2147483647"),
            ("--kvmem-query-max-tokens", "2147483647"),
            ("--kvmem-query-replay-max-tokens", "2147483647"),
        ])
        .unwrap();
        assert_eq!(max.host_bytes, INTEGER_MAX * MIB);
        assert_eq!(max.sessions, INTEGER_MAX as u32);
        assert_eq!(max.budget_tokens, 2_147_483_520);
        assert_eq!(max.gen_reserve_tokens, 2_147_483_520);
        assert_eq!(max.query_max_tokens, INTEGER_MAX as u32);
        assert_eq!(max.query_replay_max_tokens, INTEGER_MAX as u32);
    }

    #[test]
    fn v02_finite_floats_enums_and_independent_scoring() {
        for flag in [
            "--kvmem-lrfu-lambda",
            "--kvmem-lrfu-top-frac",
            "--kvmem-graph-alpha",
        ] {
            for bad in ["-0.1", "1.1", "NaN", "Inf", "-inf", "", "0.5x"] {
                assert!(resolve(&[("--kvmem", "1"), ("--kvmem-lrfu", "1"), (flag, bad)]).is_err());
            }
        }
        for flag in [
            "--kvmem-recall-strategy",
            "--kvmem-query-scorer",
            "--kvmem-score-agg",
            "--kvmem-unknown",
        ] {
            assert!(resolve(&[("--kvmem", "1"), (flag, "bad")]).is_err());
        }
        assert!(resolve(&[
            ("--kvmem", "1"),
            ("--kvmem-lrfu", "1"),
            ("--kvmem-lrfu-top-frac", "0")
        ])
        .is_err());
        for scorer in ["meanq-cosine", "page-softmax"] {
            let defaults = resolve(&[("--kvmem", "1"), ("--kvmem-query-scorer", scorer)]).unwrap();
            assert_eq!(defaults.score_agg, ScoreAgg::Max);
            assert_eq!(defaults.lrfu, scorer == "meanq-cosine");
            for agg in ["mean", "max"] {
                let c = resolve(&[
                    ("--kvmem", "1"),
                    ("--kvmem-query-scorer", scorer),
                    ("--kvmem-score-agg", agg),
                ])
                .unwrap();
                assert_eq!(c.lrfu, scorer == "meanq-cosine");
                assert_eq!(c.score_agg == ScoreAgg::Max, agg == "max");
            }
        }
        for lambda in ["0", "0.5", "1"] {
            for top in ["0.1", "0.9", "1"] {
                resolve(&[
                    ("--kvmem", "1"),
                    ("--kvmem-lrfu", "1"),
                    ("--kvmem-lrfu-lambda", lambda),
                    ("--kvmem-lrfu-top-frac", top),
                ])
                .unwrap();
            }
        }
    }

    #[test]
    fn v03_v04_alignment_and_protection() {
        for (text, expected) in [("383", 256), ("384", 384), ("385", 384)] {
            let c = resolve(&[
                ("--kvmem-budget", text),
                ("--kvmem-gen-reserve", text),
                ("--kvmem-sink-tokens", text),
                ("--kvmem-recent-tokens", "0"),
            ])
            .unwrap();
            assert_eq!(
                (
                    c.budget_tokens,
                    c.gen_reserve_tokens,
                    c.sink_tokens,
                    c.recent_tokens
                ),
                (expected, expected, expected, 0)
            );
            assert_eq!(c.report["budget_tokens"].requested, text);
        }
        for text in ["0", "127", "128", "129"] {
            let c = resolve(&[("--kvmem", "1"), ("--kvmem-recent-tokens", text)]).unwrap();
            assert_eq!(c.recent_tokens, text.parse::<u32>().unwrap() / 128 * 128);
        }
        resolve(&[
            ("--kvmem-budget", "256"),
            ("--kvmem-sink-tokens", "129"),
            ("--kvmem-recent-tokens", "129"),
        ])
        .unwrap();
        assert!(resolve(&[
            ("--kvmem-budget", "256"),
            ("--kvmem-sink-tokens", "256"),
            ("--kvmem-recent-tokens", "128")
        ])
        .is_err());
        assert!(resolve(&[("--kvmem-budget", "255")]).is_err());
        assert!(resolve(&[("--kvmem", "1"), ("--kvmem-gen-reserve", "127")]).is_err());
    }

    #[test]
    fn v09_enable_defaults_and_conflicts() {
        assert!(!resolve(&[]).unwrap().enabled);
        let c = resolve(&[("--kvmem", "1")]).unwrap();
        assert_eq!(
            (
                c.budget_tokens,
                c.gen_reserve_tokens,
                c.host_bytes,
                c.sessions
            ),
            (32768, 1024, 16384 * MIB, 4)
        );
        assert_eq!((c.sink_tokens, c.recent_tokens, c.graph_alpha), (0, 0, 0.0));
        assert!(!c.lrfu);
        assert!(!resolve(&[("--no-kvmem", "1")]).unwrap().enabled);
        assert!(resolve(&[("--kvmem-budget", "8192")]).unwrap().enabled);
        assert!(resolve(&[("--no-kvmem", "1"), ("--kvmem-budget", "8192")]).is_err());
        assert!(resolve(&[("--kvmem", "1"), ("--no-kvmem", "1")]).is_err());
        assert!(resolve(&[("--kvmem-sessions", "2")]).is_err());
    }

    #[test]
    fn environment_only_legacy_spellings_delegate_to_the_env_route() {
        let env = map(&[("HIPFIRE_KVMEM_HOST_BYTES", "17179869184")]);
        assert!(
            RawConfig::from_maps(&BTreeMap::new(), &env, &BTreeMap::new())
                .unwrap()
                .resolve()
                .unwrap()
                .is_none()
        );
        let env = map(&[
            ("HIPFIRE_KVMEM", "1"),
            ("HIPFIRE_KVMEM_BUDGET_TOKENS", "32768"),
            ("HIPFIRE_KVMEM_HOST_BYTES", "17179869184"),
            ("HIPFIRE_KVMEM_SELECT_METHOD", "retrieval"),
            ("HIPFIRE_KVMEM_QUERY_SPAN", "1"),
        ]);
        let typed = RawConfig::from_maps(&BTreeMap::new(), &env, &BTreeMap::new())
            .unwrap()
            .resolve()
            .unwrap()
            .expect("an explicit HIPFIRE_KVMEM opts into the typed contract");
        assert_eq!(
            (typed.budget_tokens, typed.host_bytes),
            (32768, 17179869184)
        );
        let env = map(&[
            ("HIPFIRE_KVMEM", "1"),
            ("HIPFIRE_KVMEM_RECALL_STRATEGY", "layerwise"),
            ("HIPFIRE_KVMEM_CONTENT_GROUPS", "2"),
        ]);
        assert!(
            RawConfig::from_maps(&BTreeMap::new(), &env, &BTreeMap::new())
                .unwrap()
                .resolve()
                .unwrap()
                .is_none()
        );
        let env = map(&[("HIPFIRE_KVMEM_SESSIONS", "2")]);
        assert!(
            RawConfig::from_maps(&BTreeMap::new(), &env, &BTreeMap::new())
                .unwrap()
                .resolve()
                .is_err()
        );
    }

    #[test]
    fn v10_storage_pairing_and_lrfu_conflicts() {
        for args in [
            vec![("--kvmem", "1"), ("--kvmem-disk-mib", "1")],
            vec![("--kvmem", "1"), ("--kvmem-disk-path", "/not/accessed")],
            vec![
                ("--kvmem", "1"),
                ("--no-kvmem-lrfu", "1"),
                ("--kvmem-lrfu-lambda", "0.5"),
            ],
            vec![
                ("--kvmem", "1"),
                ("--no-kvmem-lrfu", "1"),
                ("--kvmem-lrfu-top-frac", "0.9"),
            ],
        ] {
            assert!(resolve(&args).is_err());
        }
        let c = resolve(&[
            ("--kvmem", "1"),
            ("--kvmem-disk-mib", "1"),
            ("--kvmem-disk-path", "/not/accessed"),
        ])
        .unwrap();
        assert_eq!(c.disk_bytes, MIB);
    }

    #[test]
    fn v11_unsupported_and_retired() {
        for (flag, value) in [
            ("--kvmem-graph-alpha", "0.2"),
            ("--kvmem-block-tokens", "64"),
            ("--kvmem-cache-min-tokens", "4096"),
            ("--kvmem-recall-strategy", "layerwise"),
        ] {
            assert!(resolve(&[("--kvmem", "1"), (flag, value)])
                .unwrap_err()
                .to_string()
                .contains("UNSUPPORTED"));
        }
        for flag in RETIRED_STORAGE_FLAGS {
            let message = resolve(&[(flag, "1")]).unwrap_err().to_string();
            assert!(message.contains("retired") && message.contains("--kvmem-host-mib"));
        }
    }

    #[test]
    fn map_precedence_units_and_no_layerwise_default_injection() {
        let cli = map(&[("--kvmem-budget", "08193"), ("--kvmem-host-mib", "8")]);
        let env = map(&[
            ("HIPFIRE_KVMEM_BUDGET_TOKENS", "not parsed when shadowed"),
            ("HIPFIRE_KVMEM_HOST_BYTES", "17179869184"),
            ("HIPFIRE_KVMEM_SINK_BLOCKS", "2"),
            ("HIPFIRE_KVMEM_LRFU", "1"),
            ("HIPFIRE_KVMEM_LRFU_TOP", "0.8"),
        ]);
        let dev = map(&[
            ("developer.kvmem_budget_tokens", "2048"),
            ("developer.kvmem_recent_blocks", "3"),
        ]);
        let c = RawConfig::from_maps(&cli, &env, &dev)
            .unwrap()
            .resolve()
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                c.budget_tokens,
                c.sink_tokens,
                c.recent_tokens,
                c.host_bytes
            ),
            (8192, 256, 384, 8 * MIB)
        );
        assert_eq!(c.lrfu_top_fraction, 0.8);
        assert_eq!(c.report["budget_tokens"].source, Source::Cli);
        assert_eq!(c.report["budget_tokens"].requested, "08193");
        assert_eq!(c.report["recent_tokens"].source, Source::Developer);
        let bytes = RawConfig::from_maps(
            &map(&[("--kvmem", "1")]),
            &map(&[("HIPFIRE_KVMEM_HOST_BYTES", "17179869184")]),
            &BTreeMap::new(),
        )
        .unwrap()
        .resolve()
        .unwrap()
        .unwrap();
        assert_eq!(bytes.host_bytes, 17179869184);
        let lw = map(&[
            ("HIPFIRE_KVMEM", "1"),
            ("HIPFIRE_KVMEM_RECALL_STRATEGY", "layerwise"),
            ("HIPFIRE_KVMEM_CONTENT_GROUPS", "4"),
        ]);
        assert!(
            RawConfig::from_maps(&BTreeMap::new(), &lw, &BTreeMap::new())
                .unwrap()
                .resolve()
                .unwrap()
                .is_none()
        );
        assert!(RawConfig::from_maps(&cli, &lw, &BTreeMap::new())
            .unwrap()
            .resolve()
            .is_err());
        for (key, value) in [
            ("HIPFIRE_KVMEM_QUERY_SPAN", "512"),
            ("HIPFIRE_KVMEM_CONTENT_GROUPS", "4"),
            ("HIPFIRE_KVMEM_SINK_BLOCKS", "2147483647"),
        ] {
            assert!(RawConfig::from_maps(
                &map(&[("--kvmem", "1")]),
                &map(&[(key, value)]),
                &BTreeMap::new()
            )
            .unwrap()
            .resolve()
            .is_err());
        }
    }

    #[test]
    fn older_process_envelope_normalizes_only_its_supplied_snapshot() {
        let mut old: crate::ProcessConfig = serde_json::from_value(serde_json::json!({
            "schema_version": crate::CONFIG_SCHEMA_VERSION,
            "values": {"values": {"developer.kvmem_budget_tokens": "4097", "developer.kvmem_sink_blocks": "2"}}
        })).unwrap();
        assert!(old.kvmem.is_none());
        old.resolve_missing_kvmem().unwrap();
        let c = old.kvmem.unwrap();
        assert_eq!((c.budget_tokens, c.sink_tokens), (4096, 256));
        assert_eq!(c.report["budget_tokens"].source, Source::ProcessSnapshot);
    }

    #[test]
    fn serde_roundtrip_revalidates_effective_values() {
        let config = resolve(&[
            ("--kvmem-budget", "8193"),
            ("--kvmem-query-scorer", "meanq-cosine"),
        ])
        .unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let mut roundtrip: ResolvedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, config);
        roundtrip.validate().unwrap();
        roundtrip.budget_tokens += 128;
        assert!(roundtrip.validate().is_err());
    }
}
