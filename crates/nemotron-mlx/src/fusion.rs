//! ASR/diarization fusion state machine.
//!
//! Joins two streams that share the same 80ms frame grid: timed ASR tokens
//! (see [`crate::model::TimedToken`]) and per-frame diarization speaker
//! probabilities (`[f32; 4]`, one probability per of up to four speakers).
//! The output is a list of [`SpeakerSegment`]s: zero or more finalized
//! segments plus, at most, one trailing tentative segment for tokens whose
//! speaker cannot yet be decided.
//!
//! This module is a pure state machine: it has no dependency on
//! `sortformer-mlx`, no model code, and no MLX. It never touches the
//! tokenizer or OpenCC; text is produced by a caller-supplied `detokenize`
//! closure passed into [`Fusion::segments`].

use crate::model::TimedToken;

/// One contiguous run of tokens attributed to a single speaker.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SpeakerSegment {
    pub speaker: u8,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(rename = "final")]
    pub is_final: bool,
}

/// Tunable parameters for the fusion state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionConfig {
    /// Minimum average speaker probability required to attribute a token to
    /// that speaker rather than inheriting the previous token's speaker.
    pub threshold: f32,
    /// Half-width (in frames) of the smoothing window used when averaging
    /// per-speaker probabilities around a token's frame.
    pub smooth_frames: usize,
    /// Finalized segments shorter than this are merged into a neighbor
    /// (anti-flicker).
    pub min_turn_ms: u64,
    /// Duration of one diarization/ASR frame, in milliseconds.
    pub frame_ms: u64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            smooth_frames: 3,
            min_turn_ms: 500,
            frame_ms: 80,
        }
    }
}

/// Streaming ASR/diarization fusion state.
///
/// Tokens and diarization frames are pushed incrementally as they become
/// available; [`Fusion::segments`] recomputes the full segment list from
/// the raw streams on every call. This keeps the state trivial (three
/// fields) at the cost of O(n) work per call, which is fine because
/// recordings are minutes-scale.
pub struct Fusion {
    cfg: FusionConfig,
    tokens: Vec<TimedToken>,
    diar: Vec<[f32; 4]>,
    flushed: bool,
}

/// One run of same-speaker tokens, referenced as a half-open index range
/// into a flat, ordered `(token, speaker)` slice.
struct Group {
    speaker: u8,
    start: usize,
    end: usize,
}

impl Fusion {
    pub fn new(config: FusionConfig) -> Self {
        Self {
            cfg: config,
            tokens: Vec::new(),
            diar: Vec::new(),
            flushed: false,
        }
    }

    /// Appends ASR tokens. Callers must push tokens in non-decreasing frame
    /// order (as produced by the streaming decoder).
    pub fn push_tokens(&mut self, tokens: &[TimedToken]) {
        self.tokens.extend_from_slice(tokens);
    }

    /// Appends diarization probability frames, in frame order.
    pub fn push_diar_frames(&mut self, frames: &[[f32; 4]]) {
        self.diar.extend_from_slice(frames);
    }

    /// Marks every currently- and future-pushed token as decidable using
    /// whatever diarization frames exist at the time [`Fusion::segments`]
    /// is called. This is terminal for the *decidability rule* (it never
    /// un-flushes), but it does not stop `push_tokens`/`push_diar_frames`
    /// from accepting more data.
    pub fn flush(&mut self) {
        self.flushed = true;
    }

    /// Clears all pushed tokens, diarization frames, and the flushed flag.
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.diar.clear();
        self.flushed = false;
    }

    /// Recomputes the full segment list from the raw streams.
    ///
    /// `detokenize` is called once per output segment with that segment's
    /// token ids, in frame order; the caller is expected to compose the
    /// tokenizer and `opencc::to_traditional`.
    pub fn segments(&self, detokenize: impl Fn(&[u32]) -> String) -> Vec<SpeakerSegment> {
        let diar_len = self.diar.len() as u64;
        let smooth = self.cfg.smooth_frames as u64;

        let is_decidable = |frame: u64| -> bool { self.flushed || frame + smooth < diar_len };

        // Partition tokens (preserving order) into decidable and
        // undecidable (tentative-tail) groups.
        let mut labeled: Vec<(TimedToken, u8)> = Vec::new();
        let mut tail_tokens: Vec<TimedToken> = Vec::new();
        let mut prev_speaker: Option<u8> = None;
        for &token in &self.tokens {
            if is_decidable(token.frame) {
                let speaker = attribute_speaker(&self.diar, token.frame, &self.cfg, prev_speaker);
                prev_speaker = Some(speaker);
                labeled.push((token, speaker));
            } else {
                tail_tokens.push(token);
            }
        }

        apply_anti_flicker(&mut labeled, &self.cfg);

        let final_groups = group_labeled(&labeled);
        let mut result: Vec<SpeakerSegment> = final_groups
            .iter()
            .map(|g| {
                build_segment(
                    &labeled[g.start..g.end],
                    g.speaker,
                    self.cfg.frame_ms,
                    true,
                    &detokenize,
                )
            })
            .collect();

        let last_speaker = final_groups.last().map(|g| g.speaker).unwrap_or(0);
        if let Some(tail) = build_tail(&tail_tokens, last_speaker, self.cfg.frame_ms, &detokenize) {
            result.push(tail);
        }

        result
    }
}

/// Attribution: averages each speaker's probability over the frames
/// `[frame - smooth_frames, frame + smooth_frames]`, clipped to the
/// available diarization frames, and returns the argmax (ties broken
/// toward the lower speaker index). If the winning average is below
/// `cfg.threshold`, the previous token's speaker is inherited instead (the
/// first token defaults to speaker 0). If there are no diarization frames
/// at all, the previous speaker (or 0) is returned directly.
fn attribute_speaker(
    diar: &[[f32; 4]],
    frame: u64,
    cfg: &FusionConfig,
    prev_speaker: Option<u8>,
) -> u8 {
    if diar.is_empty() {
        return prev_speaker.unwrap_or(0);
    }
    let last_idx = diar.len() as u64 - 1;
    let smooth = cfg.smooth_frames as u64;
    let lo = frame.saturating_sub(smooth).min(last_idx);
    let hi = (frame + smooth).min(last_idx);

    let mut sums = [0f32; 4];
    let mut count = 0u32;
    for idx in lo..=hi {
        let probs = diar[idx as usize];
        for (s, sum) in sums.iter_mut().enumerate() {
            *sum += probs[s];
        }
        count += 1;
    }

    let mut best_speaker = 0usize;
    let mut best_sum = sums[0];
    for (s, &sum) in sums.iter().enumerate().skip(1) {
        if sum > best_sum {
            best_sum = sum;
            best_speaker = s;
        }
    }
    let best_avg = best_sum / count as f32;
    if best_avg < cfg.threshold {
        prev_speaker.unwrap_or(0)
    } else {
        best_speaker as u8
    }
}

/// Grouping: scans a flat, frame-ordered `(token, speaker)` slice and
/// groups consecutive equal-speaker runs into [`Group`]s.
fn group_labeled(labeled: &[(TimedToken, u8)]) -> Vec<Group> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < labeled.len() {
        let speaker = labeled[start].1;
        let mut end = start + 1;
        while end < labeled.len() && labeled[end].1 == speaker {
            end += 1;
        }
        groups.push(Group {
            speaker,
            start,
            end,
        });
        start = end;
    }
    groups
}

fn group_duration_ms(labeled: &[(TimedToken, u8)], group: &Group, frame_ms: u64) -> u64 {
    let first_frame = labeled[group.start].0.frame;
    let last_frame = labeled[group.end - 1].0.frame;
    (last_frame + 1 - first_frame) * frame_ms
}

/// Anti-flicker: repeatedly finds the leftmost finalized group shorter than
/// `min_turn_ms` whose merge target (previous group, or the next group if
/// this is the first) is itself long enough (>= `min_turn_ms`), reassigns
/// all of that short group's tokens to the target's speaker in place, and
/// re-groups. Reassigning tokens (rather than splicing opaque segment
/// objects together) means that once a short blip is absorbed, it also
/// transparently reunites with a same-speaker run on its far side during
/// the next re-group pass.
///
/// A short group whose only available neighbor is *also* short is left
/// alone: there is no legitimate (long enough) turn to absorb it into, and
/// forcing a merge would arbitrarily collapse two genuinely distinct short
/// turns (see `attributes_tokens_to_dominant_smoothed_speaker`, where two
/// single-token segments for two different speakers must both survive).
/// With only one group total there are no neighbors at all, so nothing to
/// do either.
///
/// Known trade-off: a short turn is collapsed into whichever ADJACENT turn
/// is long enough — not only in the classic same-speaker (A-B-A) flicker
/// case — so in a 3+-speaker sandwich (long A, short B, long C with A != C)
/// the short B is absorbed into A rather than kept distinct.
fn apply_anti_flicker(labeled: &mut [(TimedToken, u8)], cfg: &FusionConfig) {
    loop {
        let groups = group_labeled(labeled);
        if groups.len() <= 1 {
            return;
        }

        let mut merged = false;
        for (i, group) in groups.iter().enumerate() {
            let duration = group_duration_ms(labeled, group, cfg.frame_ms);
            if duration >= cfg.min_turn_ms {
                continue;
            }
            let target_idx = if i == 0 { 1 } else { i - 1 };
            let target = &groups[target_idx];
            let target_duration = group_duration_ms(labeled, target, cfg.frame_ms);
            if target_duration < cfg.min_turn_ms {
                // The neighbor is also short; nothing legitimate to absorb
                // into. Leave this group as-is and consider the next one.
                continue;
            }
            for entry in labeled.iter_mut().take(group.end).skip(group.start) {
                entry.1 = target.speaker;
            }
            merged = true;
            break;
        }

        if !merged {
            return;
        }
    }
}

/// Builds one finalized [`SpeakerSegment`] from a contiguous slice of
/// `(token, speaker)` pairs sharing the same speaker.
fn build_segment(
    entries: &[(TimedToken, u8)],
    speaker: u8,
    frame_ms: u64,
    is_final: bool,
    detokenize: &impl Fn(&[u32]) -> String,
) -> SpeakerSegment {
    let first_frame = entries.first().expect("segment must be non-empty").0.frame;
    let last_frame = entries.last().expect("segment must be non-empty").0.frame;
    let ids: Vec<u32> = entries.iter().map(|(t, _)| t.id).collect();
    SpeakerSegment {
        speaker,
        start_ms: first_frame * frame_ms,
        end_ms: (last_frame + 1) * frame_ms,
        text: detokenize(&ids),
        is_final,
    }
}

/// Tail construction: builds the single tentative segment (if any) holding
/// all undecidable tokens, attributed to the last finalized speaker (or 0
/// if nothing has been finalized yet). Returns `None` when there are no
/// undecidable tokens.
fn build_tail(
    tail_tokens: &[TimedToken],
    last_speaker: u8,
    frame_ms: u64,
    detokenize: &impl Fn(&[u32]) -> String,
) -> Option<SpeakerSegment> {
    if tail_tokens.is_empty() {
        return None;
    }
    let entries: Vec<(TimedToken, u8)> = tail_tokens.iter().map(|&t| (t, last_speaker)).collect();
    Some(build_segment(
        &entries,
        last_speaker,
        frame_ms,
        false,
        detokenize,
    ))
}
