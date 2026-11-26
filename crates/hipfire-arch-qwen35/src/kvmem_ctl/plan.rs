// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

impl KvMemCtl {
    /// Default-only constructor for crate-internal fixtures. Runtime callers
    /// must pass an explicit [`KvMemOptions`] snapshot.
    pub(crate) fn new_default(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
    ) -> Result<Self, String> {
        Self::new_with_options(n_slots, plan, geom, KvMemOptions::default(), None)
    }

    /// Deterministic test seam: no process environment is consulted.
    #[cfg(test)]
    pub(crate) fn new_with_strategy(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
        strategy: RecallStrategy,
    ) -> Result<Self, String> {
        let mut options = KvMemOptions::default();
        options.host_bytes = Some(16 << 20);
        options.strategy = strategy;
        Self::new_with_options(n_slots, plan, geom, options, None)
    }

    pub(crate) fn new_with_config(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
        config: Option<&hipfire_config::kvmem::ResolvedConfig>,
        options: KvMemOptions,
    ) -> Result<Self, String> {
        Self::new_with_options(n_slots, plan, geom, options, config)
    }

    pub(crate) fn new_with_options(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
        mut options: KvMemOptions,
        config: Option<&hipfire_config::kvmem::ResolvedConfig>,
    ) -> Result<Self, String> {
        let legacy = if let Some(config) = config {
            if !config.enabled
                || config.block_tokens as usize != KV_BLOCK_TOKENS
                || config.budget_tokens as usize / KV_BLOCK_TOKENS != plan.budget_blocks
            {
                return Err(
                    "kvmem: resolved Legacy configuration differs from admitted pool geometry"
                        .into(),
                );
            }
            options.strategy = RecallStrategy::Legacy;
            Some(config.clone())
        } else {
            None
        };
        Self::construct(n_slots, plan, geom, options, legacy)
    }

    pub(crate) fn diagnostics(&self) -> KvMemDiagnostics {
        self.diagnostics
    }

    fn construct(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
        options: KvMemOptions,
        legacy: Option<hipfire_config::kvmem::ResolvedConfig>,
    ) -> Result<Self, String> {
        let strategy = options.strategy;
        if n_slots == 0 || geom.n_fa_layers == 0 || geom.head_dim == 0 {
            return Err("kvmem: zero-sized controller request".to_string());
        }
        let k_width = geom.n_kv_heads * geom.head_dim;
        let q_width = geom.n_heads * geom.head_dim;
        // P4.1② G3 (design §A.8): under Layerwise + native MTP the host co-archive
        // is the COMPOSITE `kvmem-main-mtp-pages-v4` record — the trunk's
        // `n_fa_layers` ordinals PLUS one appended head aux plane at ordinal
        // `n_fa_layers` (ninfer `model = layer_types.size()`). The core raw store
        // lays each layer out as an INDEPENDENT per-block allocation
        // (raw_kv_store.cpp:154/404), so appending the aux ordinal leaves every
        // trunk layer's bytes untouched (the trunk write/restore loops address
        // `0..n_fa_layers` only). Legacy / AR-only keep `n_fa_layers` exactly, so
        // the default record is byte-neutral. The trunk GPU paged arena is
        // UNCHANGED; only this host record gains the aux layer.
        let head_plane = strategy.is_layerwise() && geom.mtp_head;
        // P1b: Legacy + native MTP can page the live head KV too. The switch is
        // opt-in until the bounded carrier has an E2E run; `HEAD_LIVE=0`
        // explicitly keeps the historical full-resident head.
        let head_live = !strategy.is_layerwise()
            && geom.mtp_head
            && hipfire_config::developer_var("HIPFIRE_KVMEM_HEAD_LIVE")
                .is_ok_and(|v| !matches!(v.trim(), "0" | "false" | "off"));
        let head_composite = head_plane || head_live;
        let n_record_layers = if head_composite {
            geom.n_fa_layers + 1
        } else {
            geom.n_fa_layers
        };
        let raw_cfg = RawStoreConfig {
            n_layer: n_record_layers as u32,
            n_embd_k: k_width as u32,
            n_embd_v: k_width as u32,
            block_tokens: KV_BLOCK_TOKENS as u32,
            k_row_bytes: 0,
            k_gpu_row_bytes: geom.per_pos_bytes as u64,
            v_gpu_row_bytes: geom.per_pos_v_bytes as u64,
        };
        let sink = legacy.as_ref().map_or_else(
            || options.sink_blocks.unwrap_or(1),
            |c| c.sink_tokens as usize / KV_BLOCK_TOKENS,
        );
        let recent = legacy.as_ref().map_or_else(
            || options.recent_blocks.unwrap_or(1),
            |c| c.recent_tokens as usize / KV_BLOCK_TOKENS,
        );
        if sink + recent > plan.budget_blocks {
            return Err(format!(
                "kvmem: sink ({sink}) + recent ({recent}) blocks exceed the \
                 residency budget ({}); lower HIPFIRE_KVMEM_SINK_BLOCKS / \
                 HIPFIRE_KVMEM_RECENT_BLOCKS or raise the budget",
                plan.budget_blocks
            ));
        }
        let budget_tokens = (plan.budget_blocks * KV_BLOCK_TOKENS) as u32;
        let gen_reserve = legacy.as_ref().map_or_else(
            || options.gen_reserve.unwrap_or(DEFAULT_GEN_RESERVE),
            |c| u64::from(c.gen_reserve_tokens),
        );
        if gen_reserve == 0 {
            return Err("kvmem: HIPFIRE_KVMEM_GEN_RESERVE must be positive".to_string());
        }
        let checkpoint_retention = match options.checkpoint_policy.as_deref() {
            Some("spread") => {
                let count = options.checkpoint_count.unwrap_or(4).clamp(1, 64);
                let mib = options.checkpoint_mib.unwrap_or(512).max(1);
                let interval = options.checkpoint_interval.unwrap_or(4096).max(1);
                let min_gap = options.checkpoint_min_gap.unwrap_or(0);
                Some((count, mib << 20, interval, min_gap))
            }
            _ => None,
        };
        // P2: persistent DN checkpoints are host-resident. The GPU pool remains
        // a fast cache; the ring is the ownership tier and the fallback when
        // the pool misses. 0 disables the ring (GPU pool only). An optional
        // immutable disk mirror seeds the ring after a process restart.
        let dn_host_mib: u64 = hipfire_config::developer_var("HIPFIRE_KVMEM_DN_HOST_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(512);
        // S2: opt-in idle-session DN demotion. The live recurrent state of a
        // decoding slot cannot be paged, so this only releases an idle slot
        // between turns and restores the exact image on continuation.
        let dn_idle = options.dn_idle.unwrap_or(false);
        let explicit_dn_ring_dir = options
            .dn_ring_dir
            .clone()
            .filter(|d| !d.is_empty())
            .map(PathBuf::from);
        let dn_disk_mib = options
            .dn_disk_mib
            .unwrap_or(if explicit_dn_ring_dir.is_some() {
                512
            } else {
                0
            });
        if dn_host_mib > 0 {
            eprintln!(
                "[kvmem] P2 DN host ring: {} MiB (HIPFIRE_KVMEM_DN_HOST_MIB); idle demote={}; disk={} MiB",
                dn_host_mib,
                if dn_idle { "on" } else { "off" },
                dn_disk_mib,
            );
        }
        if dn_disk_mib > 0 && explicit_dn_ring_dir.is_none() {
            let has_cache_dir = options
                .content_cache_dir
                .as_ref()
                .is_some_and(|d| !d.is_empty());
            if !has_cache_dir {
                eprintln!(
                    "[kvmem] P2 DN ring: disk mirror requested with no ring/cache directory; memory only"
                );
            }
        }

        // ── opt-in policy knobs (defaults reproduce the historical behaviour) ──
        // LRFU replaces the raw retrieval score with a decaying heavy-hitter
        // history in the core's `selection_score`; lambda/top tune that decay.
        let lrfu = legacy
            .as_ref()
            .map_or_else(|| options.lrfu.unwrap_or(false), |c| c.lrfu);
        let lrfu_lambda = legacy
            .as_ref()
            .map_or_else(|| options.lrfu_lambda.unwrap_or(0.5), |c| c.lrfu_lambda);
        let lrfu_top = legacy
            .as_ref()
            .map_or_else(|| options.lrfu_top.unwrap_or(0.9), |c| c.lrfu_top_fraction);
        let select_method = match legacy
            .as_ref()
            .map(|_| "retrieval".to_string())
            .or_else(|| options.select_method.clone())
        {
            Some(v) => parse_select_method(&v).ok_or_else(|| {
                format!("kvmem: HIPFIRE_KVMEM_SELECT_METHOD={v} invalid (retrieval|h2o|recency)")
            })?,
            None => 0,
        };
        if select_method == 1 {
            return Err(
                "kvmem: SELECT_METHOD=h2o needs the per-block attention-heat tap \
                 (hipfire-h2o-heat patch); use retrieval or recency for now"
                    .to_string(),
            );
        }
        // The core KvMemStore constructor enforces this; fail here with a clear
        // message instead of an opaque C++ exception at store creation.
        if lrfu && select_method != 0 {
            return Err(
                "kvmem: LRFU requires SELECT_METHOD=retrieval (core constraint)".to_string(),
            );
        }
        if lrfu {
            if !(0.0..=1.0).contains(&lrfu_lambda) {
                return Err(format!(
                    "kvmem: LRFU_LAMBDA={lrfu_lambda} out of range [0,1]"
                ));
            }
            if !(lrfu_top > 0.0 && lrfu_top <= 1.0) {
                return Err(format!("kvmem: LRFU_TOP={lrfu_top} out of range (0,1]"));
            }
        }
        // Watermark ratios are stored on the config; the page thresholds are
        // derived from the CORE watermark functions after the stores exist.
        let wm_high_env = if legacy.is_none() {
            options.watermark_high
        } else {
            None
        };
        let wm_low_env = if legacy.is_none() {
            options.watermark_low
        } else {
            None
        };
        let watermark_enabled = wm_high_env.is_some() || wm_low_env.is_some();
        let wm_high = wm_high_env.unwrap_or(0.95);
        let wm_low = wm_low_env.unwrap_or(0.85);
        if watermark_enabled
            && (!(wm_high > 0.0 && wm_high <= 1.0)
                || !(wm_low > 0.0 && wm_low <= 1.0)
                || wm_low > wm_high)
        {
            return Err(format!(
                "kvmem: GPU_HIGH/LOW_WATERMARK must satisfy 0 < low <= high <= 1 \
                 (got high={wm_high}, low={wm_low})"
            ));
        }

        // Query span: 1 keeps the single-row mean-K retrieval (byte-identical);
        // >1 accumulates a per-slot host Q ring and scores by page-softmax.
        let query_span = if legacy.is_some() {
            1
        } else {
            options.query_span.unwrap_or(1)
        };
        if !(1..=512).contains(&query_span) {
            return Err(format!(
                "kvmem: HIPFIRE_KVMEM_QUERY_SPAN={query_span} out of range [1,512]"
            ));
        }

        // ── Layerwise fail-closed policy guards (design §4, R2d-D1) ──
        // Pure decision table (see resolve_layerwise_policy): under Layerwise
        // each global working-set knob the per-layer subsystem replaces is an
        // explicit Err; Legacy is admitted unconditionally (range checks above
        // already ran), so the default serve stays byte-neutral.
        resolve_layerwise_policy(strategy, select_method, lrfu, watermark_enabled, query_span)?;

        // P4.1③ H5: resolve the head-history policy (FirstLayer auto-default under
        // Layerwise+MTP; UpstreamGlobal opt-in via env). Legacy / AR-only carry
        // Unspecified unread, so the default serve stays byte-neutral.
        let layerwise_mtp_history = resolve_layerwise_mtp_history(
            options.layerwise_mtp_history.clone(),
            strategy,
            geom.mtp_head,
        )?;

        // Recall execution model. Legacy (the default) keeps the historical
        // whole-working-set path byte-identical. Layerwise enables the per-layer
        // composite-slot subsystem: D2 adds the write-through sealed host
        // archive; the driver + packed GPU residency land in D3/D4. The strategy
        // is injected by `new` (env) or `new_with_strategy` (tests) so every
        // downstream Layerwise-only path shares one switch.

        let kv_cfg = KvStoreConfig {
            block_tokens: KV_BLOCK_TOKENS as u32,
            select_budget: budget_tokens,
            prefill_budget: 0,
            gen_budget: budget_tokens,
            sink_blocks: sink as u32,
            recent_blocks: recent as u32,
            select_method,
            lrfu: lrfu as u32,
            lrfu_lambda,
            lrfu_top_fraction: lrfu_top,
            gpu_high_watermark: wm_high,
            gpu_low_watermark: wm_low,
            ..KvStoreConfig::default()
        };

        let host_bytes = legacy.as_ref().map_or_else(
            || options.host_bytes.unwrap_or(DEFAULT_HOST_BYTES).max(1),
            |c| c.host_bytes,
        );
        // 期3 G3: internal content groups (fairness-only; ninfer
        // `--kvmem-sessions` analogue). 0 keeps the historical LRU catalog.
        let content_groups = options.content_groups.unwrap_or(0);
        if content_groups > 1024 {
            return Err(format!(
                "kvmem: HIPFIRE_KVMEM_CONTENT_GROUPS={content_groups} out of range [0,1024]"
            ));
        }
        // 期3 G3 single-tier: `HIPFIRE_KVMEM_CACHE_DIR` owns the PERSISTENT
        // pack tier. Packs are the one persistent copy, billed under
        // `HIPFIRE_KVMEM_CACHE_BYTES` and separate from the spill account;
        // there is no second storage account (the cache metrics stay 0).
        let content_cache_dir = options.content_cache_dir.clone().filter(|d| !d.is_empty());
        if let Some(dir) = content_cache_dir.as_ref() {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("kvmem cache dir {}: {e}", dir.display()))?;
            if let Some(spill) = options.spill_dir.clone() {
                if !spill.is_empty() && PathBuf::from(&spill) == dir {
                    return Err("kvmem: HIPFIRE_KVMEM_CACHE_DIR must differ from \
                                HIPFIRE_KVMEM_SPILL_DIR (separate accounts)"
                        .to_string());
                }
            }
        }
        let content_cache_dir = content_cache_dir.map(PathBuf::from);
        let layout_identity = layout_fingerprint(&raw_cfg);
        // P2 disk mirror: an explicit ring directory wins; otherwise the
        // content cache directory hosts `<cache>/dn-ring`. Persistence is off
        // unless `HIPFIRE_KVMEM_DN_DISK_MIB` (or an explicit ring dir, whose
        // default is 512 MiB) asks for it.
        let dn_ring_dir = explicit_dn_ring_dir
            .or_else(|| content_cache_dir.as_ref().map(|dir| dir.join("dn-ring")));
        let dn_ring_bytes = dn_disk_mib.saturating_mul(1 << 20);
        let dn_ring = if dn_host_mib == 0 {
            DnRing::new(0)
        } else if dn_ring_bytes > 0 {
            match dn_ring_dir.as_ref() {
                Some(dir) => DnRing::new_persistent(
                    dn_host_mib.saturating_mul(1 << 20),
                    dir,
                    dn_ring_bytes,
                    layout_identity,
                    options.diagnostics.trace,
                )
                .map_err(|e| format!("kvmem DN ring: {e}"))?,
                None => DnRing::new(dn_host_mib.saturating_mul(1 << 20)),
            }
        } else {
            DnRing::new(dn_host_mib.saturating_mul(1 << 20))
        };

        // 期3 G3 P1.1: pack files live beside the content account; scanning
        // here rebuilds the cold index after a restart (metadata only).
        let cold_bytes = options
            .content_cache_bytes
            .unwrap_or(DEFAULT_CONTENT_CACHE_BYTES)
            .max(1);
        let cold_dir = content_cache_dir;
        let cold_packs = match cold_dir.as_ref() {
            Some(dir) => ColdPackIndex::scan(
                dir,
                layout_identity,
                n_record_layers as u32,
                KV_BLOCK_TOKENS as u32,
                1 << 20,
            ),
            None => ColdPackIndex::default(),
        };
        let (spill_dir, spill_bytes) = match legacy.as_ref() {
            Some(c) => (c.disk_path.clone(), c.disk_bytes),
            None => (
                options.spill_dir.clone().map(PathBuf::from),
                options.spill_bytes.unwrap_or(DEFAULT_SPILL_BYTES).max(1),
            ),
        };
        // With a disk tier, a RAM charge that exceeds `host_bytes` demotes to
        // the spill file instead of failing. The admission headroom check only
        // guards the hard-failure case; the tiered path keeps its historical
        // "accept and spill" contract.
        let host_headroom_enforced = spill_dir.is_none();
        let storage = match spill_dir {
            Some(dir) => {
                std::fs::create_dir_all(&dir)
                    .map_err(|e| format!("kvmem spill dir {}: {e}", dir.display()))?;
                Storage::new_tiered(host_bytes, &dir, spill_bytes)
            }
            None => Storage::new_ram(host_bytes),
        }
        .map_err(|e| format!("kvmem storage: {e}"))?;

        let mut slots = Vec::with_capacity(n_slots);
        let ring_cap = if query_span > 1 {
            n_record_layers * query_span * q_width
        } else {
            0
        };
        for _ in 0..n_slots {
            slots.push(SlotState {
                raw: RawStore::new(raw_cfg, Some(&storage))
                    .map_err(|e| format!("kvmem raw store: {e}"))?,
                kv: KvStore::new(&kv_cfg).map_err(|e| format!("kvmem store: {e}"))?,
                last_select_full: 0,
                fed: 0,
                turn_start: 0,
                first_decode_done: false,
                q_ring: vec![0f32; ring_cap],
                q_ring_len: 0,
                q_ring_cursor: 0,
                lw_incarnation: 0,
                lw_seq: 0,
                lw_pressure: None,
                lw_output_pressure: None,
                lw_keep: Vec::new(),
                lw_query_due: false,
                request_query: None,
                last_query: None,
                convo_group: 0,
                last_freeze_id: None,
                lw_import_blocks: Cell::new(None),
                lw_pending_output: None,
                lw_head: None,
                lw_aux_selection: None,
            });
        }

        // Derive the watermark hysteresis band from the CORE watermark functions
        // applied to the per-slot resident budget pool (budget_blocks pages):
        // keep_target <= spill_trigger <= budget_blocks, so the band can only
        // make residency more conservative than the default, never overshoot the
        // physical pool. The keep target narrows the core selection budget via
        // set_runtime_select_budget (re-applied on reset_slot).
        let watermark = if watermark_enabled {
            let high_tok = slots[0].kv.gpu_high_watermark_tokens(budget_tokens);
            let low_tok = slots[0].kv.gpu_low_watermark_tokens(budget_tokens);
            let trigger = (high_tok as usize / KV_BLOCK_TOKENS).clamp(1, plan.budget_blocks);
            let keep = (low_tok as usize / KV_BLOCK_TOKENS).clamp(1, trigger);
            if keep < sink + recent {
                return Err(format!(
                    "kvmem: watermark keep target ({keep} blocks) is below sink+recent \
                     ({}); raise GPU_LOW_WATERMARK or lower SINK/RECENT_BLOCKS",
                    sink + recent
                ));
            }
            for st in &slots {
                st.kv
                    .set_runtime_select_budget((keep * KV_BLOCK_TOKENS) as u32)
                    .map_err(|e| format!("kvmem watermark select budget: {e}"))?;
            }
            Some((trigger, keep))
        } else {
            None
        };

        if (options.diagnostics.time || options.diagnostics.trace)
            && (lrfu
                || select_method != 0
                || watermark_enabled
                || query_span > 1
                || strategy.is_layerwise())
        {
            eprintln!(
                "[kvmem] policy: strategy={strategy:?} select_method={select_method} lrfu={lrfu} \
                 lambda={lrfu_lambda} top={lrfu_top} watermark={watermark:?} \
                 query_span={query_span} budget_blocks={} content_groups={content_groups}",
                plan.budget_blocks
            );
        }

        // Resolve the layerwise activation quotas only under Layerwise; Legacy
        // carries the unread ninfer seeds so a malformed quota env can never
        // change the byte-neutral default path (design §3.11).
        let layerwise_quotas = if strategy.is_layerwise() {
            resolve_layerwise_quotas(
                options.layerwise_activation_ram_mib.clone(),
                options.layerwise_activation_disk_mib.clone(),
                options.layerwise_staging_mib.clone(),
                options.layerwise_metadata_mib.clone(),
                options.layerwise_activation_metadata_mib.clone(),
                options.spill_dir.clone().map(PathBuf::from),
            )?
        } else {
            LayerwiseQuotas::default()
        };

        Ok(Self {
            slots,
            storage,
            host_capacity: host_bytes,
            host_record_bytes: (n_record_layers as u64).saturating_mul(
                (geom.per_pos_bytes as u64).saturating_add(geom.per_pos_v_bytes as u64),
            ),
            host_headroom_enforced,
            raw_cfg,
            kv_cfg,
            legacy,
            budget_blocks: plan.budget_blocks,
            interval: plan.interval,
            diagnostics: options.diagnostics,
            n_fa_layers: geom.n_fa_layers,
            mtp_head: geom.mtp_head,
            n_heads: geom.n_heads,
            n_kv_heads: geom.n_kv_heads,
            head_dim: geom.head_dim,
            k_width,
            kv_mode: geom.kv_mode,
            q_width,
            k_page_bytes: KV_BLOCK_TOKENS * geom.per_pos_bytes,
            v_page_bytes: KV_BLOCK_TOKENS * geom.per_pos_v_bytes,
            page_k: vec![0u8; KV_BLOCK_TOKENS * geom.per_pos_bytes],
            page_v: vec![0u8; KV_BLOCK_TOKENS * geom.per_pos_v_bytes],
            wt_k: vec![
                0u8;
                if strategy.is_layerwise() || head_live {
                    geom.max_rows * geom.per_pos_bytes
                } else {
                    0
                }
            ],
            wt_v: vec![
                0u8;
                if strategy.is_layerwise() || head_live {
                    geom.max_rows * geom.per_pos_v_bytes
                } else {
                    0
                }
            ],
            sums: vec![0f32; n_record_layers * q_width],
            q_row: vec![0f32; q_width],
            // Trunk ordinals carry one tapped Q row each; the appended head aux
            // ordinal (head_plane only) is zeroed so the core retrieval scorer
            // skips it — trunk selection scores `0..n_fa_layers` exactly as before.
            counts: {
                let mut c = vec![1u32; n_record_layers];
                if head_plane {
                    c[geom.n_fa_layers] = 0;
                }
                c
            },
            xfer_slab: None,
            batch_xfer: options.batch_xfer,
            strategy,
            layerwise_quotas,
            select_method,
            watermark,
            query_span,
            q_span: vec![0f32; ring_cap],
            q_layer_rows: vec![0u32; if query_span > 1 { n_record_layers } else { 0 }],
            head_plane,
            head_live,
            layerwise_mtp_history,
            aux_ordinal: geom.n_fa_layers as u32,
            n_record_layers,
            gen_reserve,
            checkpoint_retention,
            // P4.3: bounded frozen-prefix catalog; empty until the engine
            // freezes a prefix (PREFIX_PIN on). 期3 G3: with content groups the
            // bound is the core version limit C = max(2N, 1); otherwise 4.
            prefix_catalog: PrefixCatalog::new(if content_groups > 0 {
                (2 * content_groups).max(1)
            } else {
                4
            }),
            content_groups,
            content_retirements_total: 0,
            cold_dir,
            cold_bytes,
            cold_packs,
            cold_displaced: Vec::new(),
            cold_pack_hits_total: 0,
            cold_pack_misses_total: 0,
            spill_pages_total: 0,
            restore_pages_total: 0,
            prefix_freezes_total: 0,
            prefix_hits_total: 0,
            prefix_misses_total: 0,
            prefix_evictions_total: 0,
            prefix_catalog_hits_total: 0,
            prefix_restore_hits_total: 0,
            prefix_demotions_total: 0,
            draft_images_total: 0,
            draft_image_hits_total: 0,
            draft_image_misses_total: 0,
            draft_sidecars_total: 0,
            draft_sidecar_bytes: 0,
            dn_ring,
            dn_idle,
            state_host_bytes: 0,
            state_demotions_total: 0,
            state_promotions_total: 0,
            state_restores_total: 0,
        })
    }

    /// P4.4: the KV tier this controller resolves to (per-tier layerwise format
    /// identity + admission). The layout adapter and backend identity read it.
    pub fn kv_mode(&self) -> KvMode {
        self.kv_mode
    }

    /// Resolved recall execution model (`HIPFIRE_KVMEM_RECALL_STRATEGY`).
    /// The engine gates the Layerwise subsystem (write-through archive,
    /// per-layer driver, packed residency) on this; Legacy is the default.
    pub fn strategy(&self) -> RecallStrategy {
        self.strategy
    }

    /// Resolved layerwise activation quotas (design §3.11, ③-E1). The Layerwise
    /// carrier creates its projected-Q tape accounts against these; Legacy never
    /// reads them. Borrowed for the carrier's lifetime (the accounts are built
    /// per query from the copied limits, so no borrow outlives the controller).
    pub fn layerwise_quotas(&self) -> &LayerwiseQuotas {
        &self.layerwise_quotas
    }

    /// Borrow slot `slot`'s statistics store (`RawStore`) and policy (`KvStore`)
    /// for the Layerwise driver (design §3.9a/§3.9b). `LayerwiseDriver<'a>`
    /// borrows these for the query's lifetime; the caller must NOT [`Self::reset_slot`]
    /// while the borrow is live — reset replaces the `SlotState` wholesale (a new
    /// `RawStore`/`KvStore`), invalidating it, so the driver is rebuilt per query.
    /// `None` when the slot is out of range. The stores stay owned by this
    /// controller: the driver only reads statistics and sets the runtime select
    /// budget through them (both `&self` operations on the C++ handles).
    pub fn layerwise_stores(&self, slot: usize) -> Option<(&RawStore, &KvStore)> {
        self.slots.get(slot).map(|st| (&st.raw, &st.kv))
    }

    /// Synthesize a fresh, page-aligned selection policy for one layerwise
    /// query (design §3.9g refinement). The core layerwise picker
    /// (`prepare_selection`) admits ONLY a canonical index of sealed full pages
    /// (`total_tokens % block == 0`, every block full), but the shared per-slot
    /// `kv` store tracks ALL fed tokens — including the partial tail — for the
    /// Legacy path. So the layerwise carrier owns this fresh policy indexing the
    /// `fed / 128` sealed canonical pages; the partial tail lives only in the
    /// statistics store (`raw`) and is restored as the suffix stripe. Scoped to
    /// the query and never shared, so the Legacy `kv` store is untouched
    /// (byte-neutral). The driver's `begin_query` sets the runtime select budget,
    /// so the config's initial budget is only a placeholder here.
    pub fn layerwise_policy(&self, fed: u32) -> Result<KvStore, String> {
        let full_pages = fed / KV_BLOCK_TOKENS as u32;
        let policy =
            KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem layerwise policy: {e}"))?;
        policy
            .register_append(full_pages * KV_BLOCK_TOKENS as u32)
            .map_err(|e| format!("kvmem layerwise policy append: {e}"))?;
        Ok(policy)
    }

    /// 期4 (KVMem/nInfer DFlash2 integration): the draft's local-attention
    /// lower bound in logical token positions — the start of the continuous
    /// recent suffix the Layerwise selection retains
    /// (nInfer `dflash_context_floor = main_frontier − recent_blocks·page_tokens`).
    /// The draft ring, proposal positions and cyclic slot addressing stay
    /// logical; only the target verify uses compact positions. 0 under Legacy,
    /// where the whole context stays resident and the draft span is unchanged.
    pub fn draft_conditioning_floor(&self, slot: usize, position: usize) -> usize {
        if !self.strategy.is_layerwise() {
            return 0;
        }
        let fed = self
            .fed_tokens(slot)
            .map(|f| f as usize)
            .unwrap_or(position)
            .min(position);
        let recent = self.kv_cfg.recent_blocks as usize * KV_BLOCK_TOKENS;
        fed.saturating_sub(recent)
    }

    /// Per-slot resident page budget (full pages + partial tail).
    pub fn budget_blocks(&self) -> usize {
        self.budget_blocks
    }

    /// Always-resident sink pages (`HIPFIRE_KVMEM_SINK_BLOCKS`). The layerwise
    /// Query selection charges these inside the same budget as any mandatory
    /// (media) pins, so an admission check must add them.
    pub fn sink_blocks(&self) -> usize {
        self.kv_cfg.sink_blocks as usize
    }

    /// Always-resident recent pages (`HIPFIRE_KVMEM_RECENT_BLOCKS`). Together
    /// with the sink they form the fixed allocation every layerwise selection
    /// budget must cover.
    pub fn recent_blocks(&self) -> usize {
        self.kv_cfg.recent_blocks as usize
    }

    /// P2.2a: generation reserve R (ninfer `reserve_tokens`).
    pub fn gen_reserve(&self) -> u64 {
        self.gen_reserve
    }

    /// Layerwise window capacity: resident budget B + generation reserve R
    /// (ninfer `device_tokens = selected_tokens + reserve_tokens`).
    pub fn window_capacity_tokens(&self) -> u64 {
        (self.budget_blocks as u64) * KV_BLOCK_TOKENS as u64 + self.gen_reserve
    }
}
