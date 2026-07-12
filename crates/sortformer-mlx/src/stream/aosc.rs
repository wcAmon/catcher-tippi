//! AOSC speaker-cache state machine, ported from NeMo's `SortformerModules`
//! (`nemo/collections/asr/modules/sortformer_modules.py`). This is the
//! *synchronous* eval path (`streaming_update`, NOT `streaming_update_async`);
//! training-only branches (random speaker permutation `permute_spk`/`spk_perm`,
//! `scores_add_rnd` noise) are intentionally omitted and noted where skipped.
//!
//! All math operates on a single sequence (batch size 1); the NeMo batch
//! dimension is dropped. Rows are plain `Vec<f32>`: an embedding row has
//! `emb_dim` values, a preds row has `num_speakers` values.
//!
//! Index-based loops are deliberate: they mirror NeMo's `[frame][speaker]`
//! tensor indexing so the port reads against the Python line-for-line.
#![allow(clippy::needless_range_loop)]

/// Epsilon used to clamp probabilities before the log in `get_log_pred_scores`.
/// NeMo `pred_score_threshold` default (`sortformer_modules.py` :110). Not part
/// of `StreamingConfig` because the checkpoint never overrides it.
const PRED_SCORE_THRESHOLD: f32 = 0.25;

/// Streaming / AOSC hyper-parameters. Every value NeMo reads off `self.*` is
/// taken from here, so the pure logic is testable with any speaker count.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub chunk_len: usize,
    pub left_context: usize,
    pub right_context: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub update_period: usize,
    pub sil_frames_per_spk: usize,
    pub scores_boost_latest: f32,
    pub sil_threshold: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
    pub emb_dim: usize,
    pub num_speakers: usize,
}

impl StreamingConfig {
    /// Low-latency v2 preset for `diar_streaming_sortformer_4spk-v2.1`.
    ///
    /// AOSC hyper-parameters are the checkpoint's `model_config.yaml`
    /// (`streaming` section); the low-latency buffer geometry is the design's
    /// "chunk 6 frames, right context 7" setting (~1.04 s latency). `emb_dim`
    /// is `fc_d_model` (pre-encode output width). `update_period` 188 matches
    /// the venv-effective `spkcache_update_period` (the NeMo demo's 144 targets
    /// a `spkcache_refresh_rate` attribute absent in NeMo 2.7.3).
    pub fn low_latency_v2() -> Self {
        StreamingConfig {
            chunk_len: 6,
            left_context: 1,
            right_context: 7,
            fifo_len: 188,
            spkcache_len: 188,
            update_period: 188,
            sil_frames_per_spk: 3,
            scores_boost_latest: 0.05,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            emb_dim: 512,
            num_speakers: 4,
        }
    }
}

/// Mutable streaming state carried across chunks. Mirrors the sync-path fields
/// of NeMo's `StreamingSortformerState` (`sortformer_modules.py` :30-56); the
/// async-only length trackers and `spk_perm` are not needed here.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Speaker cache embeddings, at most `spkcache_len` rows after compression.
    pub spkcache: Vec<Vec<f32>>,
    /// Preds aligned to `spkcache`. `None` until the first compression, then
    /// carried/gathered (never refreshed from the forward's preds again).
    pub spkcache_preds: Option<Vec<Vec<f32>>>,
    /// FIFO queue embeddings.
    pub fifo: Vec<Vec<f32>>,
    /// Preds aligned to `fifo`; refreshed from this step's preds every update.
    pub fifo_preds: Vec<Vec<f32>>,
    /// Running mean silence embedding.
    pub mean_sil_emb: Vec<f32>,
    /// Number of silence frames accumulated into `mean_sil_emb`.
    pub n_sil_frames: u64,
}

impl StreamingState {
    /// Initial sync-path state: empty cache/fifo, no cache preds yet, zeroed
    /// silence profile (`init_streaming_state`, async=False, :360-371).
    pub fn new(config: &StreamingConfig) -> Self {
        StreamingState {
            spkcache: Vec::new(),
            spkcache_preds: None,
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.0; config.emb_dim],
            n_sil_frames: 0,
        }
    }
}

/// Synchronous `streaming_update` (`sortformer_modules.py` :526-609).
///
/// `chunk_embs` holds `lc + chunk_len + rc` pre-encode rows; `preds` covers the
/// `[spkcache | fifo | lc | chunk | rc]` rows, each `num_speakers` floats.
/// Trims the contexts, refreshes `fifo_preds` from this step's preds, appends
/// the chunk to the fifo, pops `update_period` frames into the spkcache on
/// overflow (updating the silence profile), and compresses the spkcache via
/// AOSC once it exceeds `spkcache_len`.
pub fn streaming_update(
    state: &mut StreamingState,
    config: &StreamingConfig,
    chunk_embs: &[Vec<f32>],
    preds: &[Vec<f32>],
    lc: usize,
    rc: usize,
) {
    // Current tensor lengths (NOT the config maxima): sync mode grows these
    // until a compression truncates the cache back to spkcache_len (:549-553).
    let spkcache_len = state.spkcache.len();
    let fifo_len = state.fifo.len();
    let chunk_len = chunk_embs.len() - lc - rc;

    // Skipped: inverse spk_perm remap of preds (:554-560) — training only.

    // Refresh fifo_preds from this step's preds over the fifo region (:562).
    state.fifo_preds = preds[spkcache_len..spkcache_len + fifo_len].to_vec();
    // Trim contexts off the chunk embeddings (:563).
    let chunk = &chunk_embs[lc..chunk_len + lc];
    // Chunk preds sit after [spkcache | fifo | lc] in preds (:564).
    let chunk_preds =
        &preds[spkcache_len + fifo_len + lc..spkcache_len + fifo_len + chunk_len + lc];

    // Append the chunk to the fifo (:567-568).
    state.fifo.extend(chunk.iter().cloned());
    state.fifo_preds.extend(chunk_preds.iter().cloned());

    if fifo_len + chunk_len > config.fifo_len {
        // Pop rule (:570-574). The min-bound `chunk_len - fifo_len + fifo_len_cur`
        // can be non-positive; usize saturating_sub then yields 0, matching the
        // `max(update_period, ...)` result.
        let mut pop = config.update_period;
        pop = pop.max((chunk_len + fifo_len).saturating_sub(config.fifo_len));
        pop = pop.min(fifo_len + chunk_len);

        let pop_embs: Vec<Vec<f32>> = state.fifo[..pop].to_vec();
        let pop_preds: Vec<Vec<f32>> = state.fifo_preds[..pop].to_vec();

        // Silence profile updates ONLY from popped frames (:578-583).
        get_silence_profile(state, config, &pop_embs, &pop_preds);

        state.fifo.drain(..pop);
        state.fifo_preds.drain(..pop);

        // Append popped frames to the spkcache (:588).
        state.spkcache.extend(pop_embs.iter().cloned());
        // spkcache_preds carries only after the first compression (:589-590).
        if let Some(sp) = state.spkcache_preds.as_mut() {
            sp.extend(pop_preds.iter().cloned());
        }

        if state.spkcache.len() > config.spkcache_len {
            // First compression: seed spkcache_preds from the preds over the old
            // spkcache region plus the popped preds (:592-593).
            if state.spkcache_preds.is_none() {
                let mut sp = preds[..spkcache_len].to_vec();
                sp.extend(pop_preds.iter().cloned());
                state.spkcache_preds = Some(sp);
            }
            // permute_spk = self.training = false (eval only) (:599).
            let cache_preds = state.spkcache_preds.take().unwrap();
            let (new_cache, new_preds) =
                compress_spkcache(&state.spkcache, &cache_preds, &state.mean_sil_emb, config);
            state.spkcache = new_cache;
            state.spkcache_preds = Some(new_preds);
        }
    }
}

/// `_get_silence_profile` (:636-667). A frame counts as silence when the sum of
/// its preds is below `sil_threshold`; the mean silence embedding is updated as
/// a running mean over all silence frames seen so far.
fn get_silence_profile(
    state: &mut StreamingState,
    config: &StreamingConfig,
    emb_seq: &[Vec<f32>],
    preds: &[Vec<f32>],
) {
    let emb_dim = config.emb_dim;
    let mut sil_count: u64 = 0;
    let mut sil_sum = vec![0.0f32; emb_dim];
    for (emb, pred) in emb_seq.iter().zip(preds.iter()) {
        let s: f32 = pred.iter().sum();
        if s < config.sil_threshold {
            sil_count += 1;
            for d in 0..emb_dim {
                sil_sum[d] += emb[d];
            }
        }
    }
    if sil_count == 0 {
        return; // has_new_sil.any() == False (:660-661)
    }
    let upd_n = state.n_sil_frames + sil_count;
    let denom = upd_n.max(1) as f32; // clamp(min=1) (:666)
    for d in 0..emb_dim {
        let old_sum = state.mean_sil_emb[d] * state.n_sil_frames as f32;
        state.mean_sil_emb[d] = (old_sum + sil_sum[d]) / denom;
    }
    state.n_sil_frames = upd_n;
}

/// `_get_log_pred_scores` (:669-686). Log-based per-(frame, speaker) score,
/// high for confident non-overlapped speech:
/// `log p - log(1-p) + Σ_s log(1-p_s) - log 0.5`, with both logs clamped at
/// `PRED_SCORE_THRESHOLD`.
fn get_log_pred_scores(preds: &[Vec<f32>], n_spk: usize) -> Vec<Vec<f32>> {
    let ln_half = 0.5f32.ln();
    preds
        .iter()
        .map(|p| {
            let log_p: Vec<f32> = p
                .iter()
                .map(|&x| x.max(PRED_SCORE_THRESHOLD).ln())
                .collect();
            let log_1p: Vec<f32> = p
                .iter()
                .map(|&x| (1.0 - x).max(PRED_SCORE_THRESHOLD).ln())
                .collect();
            let log_1p_sum: f32 = log_1p.iter().sum();
            (0..n_spk)
                .map(|s| log_p[s] - log_1p[s] + log_1p_sum - ln_half)
                .collect()
        })
        .collect()
}

/// `_disable_low_scores` (:782-808). Non-speech (pred <= 0.5) -> `-inf`. Then,
/// for any speaker with at least `min_pos_scores_per_spk` positive-scored
/// frames, its remaining non-positive (overlapped) speech frames -> `-inf`.
fn disable_low_scores(
    preds: &[Vec<f32>],
    scores: &mut [Vec<f32>],
    n_spk: usize,
    min_pos_scores_per_spk: usize,
) {
    let n_frames = scores.len();
    // is_speech from the ORIGINAL preds; disable non-speech scores first (:800-801).
    let is_speech: Vec<Vec<bool>> = preds
        .iter()
        .map(|p| p.iter().map(|&x| x > 0.5).collect())
        .collect();
    for f in 0..n_frames {
        for s in 0..n_spk {
            if !is_speech[f][s] {
                scores[f][s] = f32::NEG_INFINITY;
            }
        }
    }
    // Positive-score counts per speaker (evaluated after the non-speech mask,
    // so -inf never counts as positive) (:805-806).
    let mut pos_count = vec![0usize; n_spk];
    for f in 0..n_frames {
        for s in 0..n_spk {
            if scores[f][s] > 0.0 {
                pos_count[s] += 1;
            }
        }
    }
    for f in 0..n_frames {
        for s in 0..n_spk {
            let is_pos = scores[f][s] > 0.0;
            if !is_pos && is_speech[f][s] && pos_count[s] >= min_pos_scores_per_spk {
                scores[f][s] = f32::NEG_INFINITY;
            }
        }
    }
}

/// `_boost_topk_scores` (:611-634). Raise the `n_boost_per_spk` highest scores
/// per speaker by `-scale_factor * ln(0.5)` (a positive bump). Disabled (`-inf`)
/// frames stay `-inf`. Ties broken by frame index (ascending) for determinism;
/// NeMo's `torch.topk(sorted=False)` leaves ties unspecified.
fn boost_topk_scores(scores: &mut [Vec<f32>], n_spk: usize, n_boost_per_spk: usize, scale: f32) {
    let n_frames = scores.len();
    let k = n_boost_per_spk.min(n_frames);
    if k == 0 {
        return;
    }
    let bump = -scale * 0.5f32.ln();
    for s in 0..n_spk {
        let mut idx: Vec<usize> = (0..n_frames).collect();
        // Descending by score; ties -> lower frame index first.
        idx.sort_by(|&a, &b| {
            scores[b][s]
                .partial_cmp(&scores[a][s])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &f in idx.iter().take(k) {
            scores[f][s] += bump; // -inf + finite stays -inf
        }
    }
}

/// One entry selected by `get_topk_indices`: the source frame index and whether
/// the slot is disabled (filled with mean silence on gather).
struct Selected {
    frame: usize,
    disabled: bool,
}

/// `_get_topk_indices` (:688-719). `scores` already includes the appended
/// silence-pad rows (`sil_frames_per_spk` rows of `+inf`). Flattens as
/// speaker-major bands, picks the `spkcache_len` highest, then sorts the chosen
/// flat indices ascending so survivors come out grouped by speaker and, within
/// each speaker, in arrival order. `-inf` picks and silence-pad picks are marked
/// disabled.
fn get_topk_indices(
    scores: &[Vec<f32>],
    n_spk: usize,
    spkcache_len: usize,
    sil_frames_per_spk: usize,
) -> Vec<Selected> {
    let n_frames = scores.len(); // includes the silence pad
    let n_frames_no_sil = n_frames - sil_frames_per_spk;

    // Flatten speaker-major: flat = s * n_frames + f (matches permute(0,2,1)).
    let mut flat: Vec<(f32, usize)> = Vec::with_capacity(n_spk * n_frames);
    for s in 0..n_spk {
        for f in 0..n_frames {
            flat.push((scores[f][s], s * n_frames + f));
        }
    }
    // Top spkcache_len by value; ties -> lower flat index (deterministic).
    flat.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    flat.truncate(spkcache_len);

    // `-inf` picks become the sentinel that sorts to the end (:710-711).
    let mut chosen: Vec<usize> = flat
        .into_iter()
        .map(|(v, flat_idx)| {
            if v == f32::NEG_INFINITY {
                usize::MAX
            } else {
                flat_idx
            }
        })
        .collect();
    // Sort to restore original order; sentinels last (:714).
    chosen.sort_unstable();

    chosen
        .into_iter()
        .map(|flat_idx| {
            if flat_idx == usize::MAX {
                Selected {
                    frame: 0,
                    disabled: true,
                }
            } else {
                let frame = flat_idx % n_frames; // remainder(., n_frames) (:716)
                // Silence-pad frames are disabled -> mean silence (:717-718).
                if frame >= n_frames_no_sil {
                    Selected {
                        frame: 0,
                        disabled: true,
                    }
                } else {
                    Selected {
                        frame,
                        disabled: false,
                    }
                }
            }
        })
        .collect()
}

/// `_gather_spkcache_and_preds` (:721-759). Gather embeddings/preds at the
/// selected frames; disabled slots take `mean_sil_emb` and zero preds.
fn gather_spkcache_and_preds(
    emb_seq: &[Vec<f32>],
    preds: &[Vec<f32>],
    selected: &[Selected],
    mean_sil_emb: &[f32],
    n_spk: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut emb_out = Vec::with_capacity(selected.len());
    let mut preds_out = Vec::with_capacity(selected.len());
    for sel in selected {
        if sel.disabled {
            emb_out.push(mean_sil_emb.to_vec());
            preds_out.push(vec![0.0; n_spk]);
        } else {
            emb_out.push(emb_seq[sel.frame].clone());
            preds_out.push(preds[sel.frame].clone());
        }
    }
    (emb_out, preds_out)
}

/// `_compress_spkcache` (:838-896, `permute_spk=False`). Keep the `spkcache_len`
/// most important frames of `emb_seq`, ordered by speaker then arrival order,
/// reserving `sil_frames_per_spk` mean-silence slots per speaker.
fn compress_spkcache(
    emb_seq: &[Vec<f32>],
    preds: &[Vec<f32>],
    mean_sil_emb: &[f32],
    config: &StreamingConfig,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_spk = config.num_speakers;
    let n_frames = emb_seq.len();

    // Per-speaker budgets (:862-865).
    let per_spk = config.spkcache_len / n_spk - config.sil_frames_per_spk;
    let strong = (per_spk as f32 * config.strong_boost_rate).floor() as usize;
    let weak = (per_spk as f32 * config.weak_boost_rate).floor() as usize;
    let min_pos = (per_spk as f32 * config.min_pos_scores_rate).floor() as usize;

    // Base scores, then disable low scores (:867-868).
    let mut scores = get_log_pred_scores(preds, n_spk);
    disable_low_scores(preds, &mut scores, n_spk, min_pos);

    // Skipped: _get_max_perm_index / _permute_speakers (:870-872) — training only.

    // Boost newly added frames (index >= spkcache_len in the compress input)
    // (:876-877). Uses the config maximum, not the current length.
    if config.scores_boost_latest > 0.0 {
        for f in config.spkcache_len..n_frames {
            for s in 0..n_spk {
                scores[f][s] += config.scores_boost_latest;
            }
        }
    }

    // Skipped: scores_add_rnd noise (:879-881) — training only.

    // Strong then weak boosting (:884-886).
    boost_topk_scores(&mut scores, n_spk, strong, 2.0);
    boost_topk_scores(&mut scores, n_spk, weak, 1.0);

    // Append silence pad rows of +inf (:888-890).
    if config.sil_frames_per_spk > 0 {
        for _ in 0..config.sil_frames_per_spk {
            scores.push(vec![f32::INFINITY; n_spk]);
        }
    }

    let selected = get_topk_indices(
        &scores,
        n_spk,
        config.spkcache_len,
        config.sil_frames_per_spk,
    );
    gather_spkcache_and_preds(emb_seq, preds, &selected, mean_sil_emb, n_spk)
}
