// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The hipfire authors
// hipfire — see LICENSE or NOTICE in the project root.

//! D3-A deterministic discriminator (host-only, no GPU, no model sampling).
//!
//! The earlier recall discriminators mixed selection quality with generation
//! nondeterminism (near-tie sampling / carrier rebuilds). This harness isolates
//! *selection*: attention mass is injected exactly, the production
//! `LayerwisePrefillPressure` and the Legacy global `pick_topk_blocks` picker
//! consume it, and the retained page sets are compared byte-for-byte across
//! repeats.
//!
//! S1 (heterogeneous need): three layers each depend on a different page while
//! the documented global budget holds two. The per-layer picker must keep each
//! layer's needed page; the global picker drops one of the three.
//! S2 (negative control): identical mass in every layer must not manufacture a
//! divergence — the per-layer pickers agree with each other and keep the needed
//! page.
//!
//! Outcome routing (parity audit §8.3): S1 pass + S3 attribution on the real
//! engine ⇒ selection mechanism sound (fix quotas); S1 fail ⇒ mechanism defect.

use hipfire_kvmem::{
    BackendIdentity, BlockCopies, BlockDescriptor, KvStore, KvStoreConfig, LayerMemorySnapshot,
    MemoryCapabilities, MemoryCatalog, MemoryDescriptor, MemorySnapshot, MemoryStamp,
    PayloadLayout, PrefillAcceptance, PrefillMass, PrefillOptions, PrefillPressure, PressureChoice,
    RawStoreConfig, SessionIdentity,
};

const BLOCK: u32 = 4;
const PAGES: u64 = 8;
const TOKENS: u64 = PAGES * BLOCK as u64; // 32
const ROW_BYTES: usize = 36;
const LAYERS: [u32; 3] = [0, 1, 2];

fn session() -> SessionIdentity {
    SessionIdentity {
        id: 42,
        incarnation: 3,
    }
}

fn raw_config() -> RawStoreConfig {
    RawStoreConfig {
        n_layer: LAYERS.len() as u32,
        n_embd_k: 8,
        n_embd_v: 8,
        block_tokens: BLOCK,
        k_row_bytes: 0,
        k_gpu_row_bytes: ROW_BYTES as u64,
        v_gpu_row_bytes: ROW_BYTES as u64,
    }
}

fn descriptor() -> MemoryDescriptor {
    let layout =
        PayloadLayout::from_raw(&raw_config(), "hipfire-q8-packed-v1", &LAYERS).expect("layout");
    let cap = MemoryCapabilities {
        page_tokens: BLOCK,
        max_position: 1 << 20,
        native_payload: 1,
        sparse_view: 1,
        original_rope_positions: 1,
        pre_rope_statistics: 1,
        state_checkpoint: 1,
        asynchronous_transfer: 0,
    };
    let backend = BackendIdentity {
        backend: "hipfire".into(),
        revision: "kvmem-v046".into(),
        model: "d3-three-layer-fixture".into(),
        state_schema: "state-v1".into(),
    };
    MemoryDescriptor::new(&backend, &layout, &cap).expect("descriptor")
}

fn stamp() -> MemoryStamp {
    MemoryStamp {
        valid: 1,
        session: session(),
        history_revision: 7,
        view_version: 1,
        frontier: TOKENS,
        execution_history: 3,
    }
}

fn block(i: u64) -> BlockDescriptor {
    BlockDescriptor {
        session: session(),
        id: i,
        original_start: i * BLOCK as u64,
        valid_tokens: BLOCK,
        content_version: 1,
        statistics_version: 1,
    }
}

/// Every page resident in every layer, one shared accepted history.
fn catalog() -> MemoryCatalog {
    let blocks: Vec<BlockCopies> = (0..PAGES)
        .map(|i| BlockCopies {
            block: block(i),
            host_version: 1,
        })
        .collect();
    let resident: Vec<u64> = (0..PAGES).collect();
    let layers: Vec<LayerMemorySnapshot> = LAYERS
        .iter()
        .map(|&model_layer| LayerMemorySnapshot {
            model_layer,
            memory: MemorySnapshot {
                stamp: stamp(),
                blocks: blocks.clone(),
                resident: resident.clone(),
            },
        })
        .collect();
    MemoryCatalog::new(&descriptor(), &layers).expect("catalog")
}

fn pressure() -> PrefillPressure {
    PrefillPressure::new(
        session(),
        &LAYERS,
        &PrefillOptions {
            block_tokens: BLOCK,
            select_tokens: 2 * BLOCK as u64, // two pages per layer
            reserve_tokens: 0,
            interval: 1,
            stride: 0, // sample the last row only (deterministic)
            half_life: 8192.0,
        },
    )
    .expect("pressure")
}

/// One injected row: `needed` gets `share`, the rest split the remainder.
fn mass_row(needed: u64, share: f64) -> Vec<f64> {
    let mut row = vec![(1.0 - share) / (PAGES - 1) as f64; PAGES as usize];
    row[needed as usize] = share;
    row
}

/// Accept one whole-model prefill chunk whose sampled row is the last token.
fn accept(pressure: &mut PrefillPressure, needed: &[u64]) -> bool {
    let rows = pressure.rows(0, TOKENS as u32, 0, false).expect("rows");
    assert_eq!(
        rows,
        vec![TOKENS as u32 - 1],
        "stride-0 samples the last row"
    );
    let history: Vec<u64> = (0..PAGES).map(|i| i * BLOCK as u64).collect();
    let layers: Vec<PrefillMass> = LAYERS
        .iter()
        .zip(needed)
        .map(|(&model_layer, &page)| PrefillMass {
            model_layer,
            starts: history.clone(),
            history: history.clone(),
            masses: mass_row(page, 0.4),
        })
        .collect();
    pressure
        .accept(&PrefillAcceptance {
            history: stamp(),
            count: TOKENS,
            accepted: TOKENS,
            local_begin: 0,
            rows,
            layers,
        })
        .expect("accept")
}

fn chosen(choices: &[PressureChoice], layer: u32) -> Vec<u64> {
    choices
        .iter()
        .find(|c| c.model_layer == layer)
        .expect("layer choice")
        .selected
        .clone()
}

/// Legacy global picker over the same canonical pages.
fn legacy_pick(scores: &[f64]) -> Vec<u32> {
    let store = KvStore::new(&KvStoreConfig {
        block_tokens: BLOCK,
        select_budget: 2 * BLOCK, // two pages, the documented global budget
        sink_blocks: 0,
        recent_blocks: 0,
        ..KvStoreConfig::default()
    })
    .expect("legacy store");
    store.register_append(TOKENS as u32).expect("append");
    store.set_retrieval_scores(scores).expect("scores");
    store.pick_topk_blocks(&[]).expect("pick")
}

#[test]
fn s1_heterogeneous_need_separates_per_layer_from_global_selection() {
    let mut pressure = pressure();
    assert!(accept(&mut pressure, &[2, 5, 7]));
    let catalog = catalog();
    let choices = pressure
        .select(&catalog, &[], 2 * BLOCK as u64, 0)
        .expect("select");
    let read = choices.read_all().expect("read");

    // The per-layer picker keeps every layer's own needed page.
    for (layer, needed) in LAYERS.iter().zip([2u64, 5, 7]) {
        let selected = chosen(&read, *layer);
        assert_eq!(
            selected.len(),
            2,
            "layer {layer} picks two pages: {selected:?}"
        );
        assert!(
            selected.contains(&needed),
            "layer {layer} lost page {needed}: {selected:?}"
        );
    }

    // The same canonical pages under ONE global ranking: pages 2/5/7 are the
    // top three, the budget holds two, so one needed page is dropped.
    let mut scores = vec![0.05f64; PAGES as usize];
    scores[2] = 0.34;
    scores[5] = 0.33;
    scores[7] = 0.33;
    let picked = legacy_pick(&scores);
    assert_eq!(picked.len(), 2, "global budget holds two pages: {picked:?}");
    let dropped = [2u32, 5, 7].iter().any(|id| !picked.contains(id));
    assert!(dropped, "global picker kept every needed page: {picked:?}");
}

#[test]
fn s2_identical_mass_does_not_manufacture_divergence() {
    let mut pressure = pressure();
    assert!(accept(&mut pressure, &[2, 2, 2]));
    let catalog = catalog();
    let choices = pressure
        .select(&catalog, &[], 2 * BLOCK as u64, 0)
        .expect("select");
    let read = choices.read_all().expect("read");

    let first = chosen(&read, 0);
    assert!(first.contains(&2), "needed page retained: {first:?}");
    for layer in [1u32, 2] {
        assert_eq!(
            chosen(&read, layer),
            first,
            "identical mass must pick identically"
        );
    }

    // Negative control: the Legacy ranking built from the same distribution
    // keeps the same needed page; no artificial Layerwise "win".
    let row = mass_row(2, 0.4);
    let picked = legacy_pick(&row.into_iter().map(|v| v).collect::<Vec<_>>());
    assert!(
        picked.contains(&2),
        "legacy keeps the shared needed page: {picked:?}"
    );
}

#[test]
fn selection_is_byte_stable_across_repeats() {
    let mut pressure = pressure();
    assert!(accept(&mut pressure, &[2, 5, 7]));
    let catalog = catalog();
    let a = pressure
        .select(&catalog, &[], 2 * BLOCK as u64, 0)
        .expect("select a")
        .read_all()
        .expect("read a");
    let b = pressure
        .select(&catalog, &[], 2 * BLOCK as u64, 0)
        .expect("select b")
        .read_all()
        .expect("read b");
    assert_eq!(a, b, "identical accepted mass must select identically");
    for layer in LAYERS {
        assert_eq!(chosen(&a, layer), chosen(&b, layer));
    }
}
