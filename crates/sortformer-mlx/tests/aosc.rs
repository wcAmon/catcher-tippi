//! Pure-logic unit tests for the AOSC streaming state machine.
//!
//! These are synthetic: no MLX, no model. Embeddings are "tagged" (every
//! dimension set to a constant `tag`) so a frame's identity survives the
//! FIFO/spkcache shuffling and we can assert exactly which frames lived or
//! died. Speaker predictions are supplied per frame; a small tag->pred map
//! lets the test harness rebuild the `preds` argument from the *real* current
//! contents of `spkcache`/`fifo`, so the harness never re-implements the logic
//! under test.

use std::collections::HashMap;

use sortformer_mlx::stream::{StreamingConfig, StreamingState, streaming_update};

fn cfg() -> StreamingConfig {
    StreamingConfig {
        chunk_len: 4,
        left_context: 1,
        right_context: 2,
        fifo_len: 8,
        spkcache_len: 16,
        update_period: 8,
        sil_frames_per_spk: 1,
        scores_boost_latest: 0.05,
        sil_threshold: 0.2,
        strong_boost_rate: 0.75,
        weak_boost_rate: 1.5,
        min_pos_scores_rate: 0.5,
        emb_dim: 4,
        num_speakers: 2,
    }
}

fn frame(tag: f32) -> Vec<f32> {
    vec![tag; 4]
}

fn speech(spk: usize) -> Vec<f32> {
    let mut p = vec![0.05; 2];
    p[spk] = 0.95;
    p
}

fn silence() -> Vec<f32> {
    vec![0.01; 2]
}

/// Test harness that tracks each emitted frame's pred by its tag, so it can
/// reconstruct the full `preds` vector ([spkcache | fifo | lc | chunk | rc])
/// from the state's actual current contents before every `streaming_update`.
struct Harness {
    cfg: StreamingConfig,
    state: StreamingState,
    /// tag bits -> pred row for that frame.
    tag_preds: HashMap<u32, Vec<f32>>,
}

impl Harness {
    fn new(cfg: StreamingConfig) -> Self {
        let state = StreamingState::new(&cfg);
        Harness {
            cfg,
            state,
            tag_preds: HashMap::new(),
        }
    }

    /// Pred for an embedding currently in the cache/fifo, recovered from its
    /// tag (all dims equal). Unknown tags (e.g. inserted `mean_sil_emb`) count
    /// as zero-activity silence.
    fn pred_for(&self, emb: &[f32]) -> Vec<f32> {
        let tag = emb[0];
        self.tag_preds
            .get(&tag.to_bits())
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.cfg.num_speakers])
    }

    /// Push one chunk. `chunk` is the list of (tag, pred) for the chunk frames.
    /// Left/right context rows are synthesized as throwaway silence (they are
    /// trimmed before entering the fifo, and excluded from chunk/fifo preds).
    fn push(&mut self, chunk: Vec<(f32, Vec<f32>)>) {
        let lc = self.cfg.left_context;
        let rc = self.cfg.right_context;

        // Build preds: spkcache region | fifo region | lc | chunk | rc.
        let mut preds: Vec<Vec<f32>> = Vec::new();
        for emb in &self.state.spkcache {
            preds.push(self.pred_for(emb));
        }
        for emb in &self.state.fifo {
            preds.push(self.pred_for(emb));
        }
        for _ in 0..lc {
            preds.push(silence());
        }
        for (tag, p) in &chunk {
            self.tag_preds.insert(tag.to_bits(), p.clone());
            preds.push(p.clone());
        }
        for _ in 0..rc {
            preds.push(silence());
        }

        // Build chunk embeddings: lc throwaway | chunk (tagged) | rc throwaway.
        let mut embs: Vec<Vec<f32>> = Vec::new();
        for _ in 0..lc {
            embs.push(frame(-1.0));
        }
        for (tag, _) in &chunk {
            embs.push(frame(*tag));
        }
        for _ in 0..rc {
            embs.push(frame(-1.0));
        }

        streaming_update(&mut self.state, &self.cfg, &embs, &preds, lc, rc);
    }

    /// Speech chunk of 4 frames with the given (tag, speaker) assignments.
    fn push_speech(&mut self, frames: &[(f32, usize)]) {
        let chunk = frames.iter().map(|&(t, s)| (t, speech(s))).collect();
        self.push(chunk);
    }
}

/// Recover the integer tag of a tagged embedding (all dims equal).
fn tag_of(emb: &[f32]) -> f32 {
    emb[0]
}

#[test]
fn contexts_are_trimmed_and_chunk_lands_in_fifo() {
    let mut h = Harness::new(cfg());
    h.push(vec![
        (10.0, speech(0)),
        (11.0, speech(1)),
        (12.0, silence()),
        (13.0, speech(0)),
    ]);

    // lc=1, rc=2 rows were trimmed; only the 4 chunk rows entered the fifo.
    assert_eq!(
        h.state.fifo,
        vec![frame(10.0), frame(11.0), frame(12.0), frame(13.0)]
    );
    assert_eq!(
        h.state.fifo_preds,
        vec![speech(0), speech(1), silence(), speech(0)]
    );
    assert!(h.state.spkcache.is_empty());
    assert!(h.state.spkcache_preds.is_none());
}

#[test]
fn fifo_overflow_pops_update_period_frames_into_spkcache() {
    let mut h = Harness::new(cfg());
    // Three chunks of 4 speech frames each -> after the 3rd, fifo would be 12
    // (> fifo_len 8), so the pop rule fires.
    h.push_speech(&[(1.0, 0), (2.0, 0), (3.0, 0), (4.0, 0)]);
    h.push_speech(&[(5.0, 0), (6.0, 0), (7.0, 0), (8.0, 0)]);
    h.push_speech(&[(9.0, 0), (10.0, 0), (11.0, 0), (12.0, 0)]);

    // pop = clamp(update_period=8, min = chunk_len - fifo_len + fifo_len_cur = 4-8+8=4,
    //             max = fifo_len_cur + chunk_len = 8+4=12) = 8.
    assert_eq!(h.state.spkcache.len(), 8);
    let cached: Vec<f32> = h.state.spkcache.iter().map(|e| tag_of(e)).collect();
    assert_eq!(cached, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    // The remaining 4 stay in the fifo, in order.
    let fifo: Vec<f32> = h.state.fifo.iter().map(|e| tag_of(e)).collect();
    assert_eq!(fifo, vec![9.0, 10.0, 11.0, 12.0]);
}

#[test]
fn silence_profile_accumulates_only_below_threshold_frames() {
    let mut h = Harness::new(cfg());
    // First 4 frames are silence (tags 1..4), next 4 are speech; the 3rd chunk
    // triggers a pop of the first 8 frames -> only the 4 silence rows update
    // the running mean silence embedding.
    h.push(vec![
        (1.0, silence()),
        (2.0, silence()),
        (3.0, silence()),
        (4.0, silence()),
    ]);
    h.push_speech(&[(5.0, 0), (6.0, 1), (7.0, 0), (8.0, 1)]);
    h.push_speech(&[(9.0, 0), (10.0, 1), (11.0, 0), (12.0, 1)]);

    assert_eq!(h.state.n_sil_frames, 4);
    // mean of tags {1,2,3,4} = 2.5 across every dim.
    assert_eq!(h.state.mean_sil_emb, vec![2.5; 4]);
}

#[test]
fn compression_respects_per_speaker_quota_and_arrival_order() {
    let mut h = Harness::new(cfg());
    // 7 chunks of 4 frames (tags 1..28). Compression fires on the 7th push,
    // when the spkcache would reach 24 (> spkcache_len 16).
    //   tags 1..4   -> silence (feed the silence profile, never survive)
    //   tags 5..14  -> speaker 0
    //   tags 15..24 -> speaker 1
    // Grouping by speaker keeps spk0's (lower) tags before spk1's (higher),
    // each in arrival order, so survivors are globally strictly increasing.
    let spk_of = |tag: f32| -> Option<usize> {
        if tag <= 4.0 {
            None // silence
        } else if tag <= 14.0 {
            Some(0)
        } else {
            Some(1)
        }
    };
    let mut tag = 1.0f32;
    for _ in 0..7 {
        let chunk: Vec<(f32, Vec<f32>)> = (0..4)
            .map(|_| {
                let t = tag;
                tag += 1.0;
                match spk_of(t) {
                    None => (t, silence()),
                    Some(s) => (t, speech(s)),
                }
            })
            .collect();
        h.push(chunk);
    }

    // spkcache is compressed to exactly spkcache_len.
    assert_eq!(h.state.spkcache.len(), h.cfg.spkcache_len);
    assert_eq!(
        h.state.spkcache_preds.as_ref().map(|p| p.len()),
        Some(h.cfg.spkcache_len)
    );

    // Reserved silence slots: sil_frames_per_spk * num_speakers = 2, each equal
    // to mean_sil_emb (which is 2.5 from the popped silence frames).
    assert_eq!(h.state.mean_sil_emb, vec![2.5; 4]);
    let sil_slots = h
        .state
        .spkcache
        .iter()
        .filter(|e| **e == h.state.mean_sil_emb)
        .count();
    assert_eq!(sil_slots, 2);

    // Surviving speech tags, in spkcache order.
    let survivors: Vec<f32> = h
        .state
        .spkcache
        .iter()
        .map(|e| tag_of(e))
        .filter(|&t| t > 4.0) // exclude the mean-sil (2.5) slots
        .collect();

    // Strictly increasing overall (speaker grouping preserves global order here).
    for w in survivors.windows(2) {
        assert!(
            w[0] < w[1],
            "survivors not strictly increasing: {survivors:?}"
        );
    }

    // Per-speaker floor: each speaker keeps at least `strong` speech frames.
    // per_spk = 16/2 - 1 = 7; strong = floor(7 * 0.75) = 5.
    let spk0 = survivors.iter().filter(|&&t| t <= 14.0).count();
    let spk1 = survivors.iter().filter(|&&t| t > 14.0).count();
    assert!(spk0 >= 5, "spk0 kept {spk0} (< strong floor 5)");
    assert!(spk1 >= 5, "spk1 kept {spk1} (< strong floor 5)");
}

#[test]
fn latest_frames_survive_ties_via_score_boost() {
    let mut h = Harness::new(cfg());
    // All 24 pre-compression frames are speaker 0 with identical preds, so
    // every base score ties. The 8 newest frames (tags 17..24, popped on the
    // compressing step, index >= spkcache_len in the compress input) receive
    // +scores_boost_latest and must all survive the cull.
    let mut tag = 1.0f32;
    for _ in 0..7 {
        let chunk: Vec<(f32, usize)> = (0..4)
            .map(|_| {
                let t = tag;
                tag += 1.0;
                (t, 0)
            })
            .collect();
        h.push_speech(&chunk);
    }

    assert_eq!(h.state.spkcache.len(), h.cfg.spkcache_len);

    let survivors: Vec<f32> = h
        .state
        .spkcache
        .iter()
        .map(|e| tag_of(e))
        .filter(|&t| t >= 1.0) // exclude the zero-valued mean-sil slots
        .collect();

    // 14 speech slots (16 - 2 reserved silence) survive.
    assert_eq!(survivors.len(), 14);

    // Every newest-group tag (17..=24) survived the tie-break.
    for t in 17..=24 {
        assert!(
            survivors.contains(&(t as f32)),
            "newest frame {t} was culled despite the latest-score boost: {survivors:?}"
        );
    }
    // And some of the older frames were dropped (only 6 of 16 old survive).
    let old_survivors = survivors.iter().filter(|&&t| t <= 16.0).count();
    assert_eq!(old_survivors, 6);
}
