//! Tests for the ASR/diarization fusion state machine.

use nemotron_mlx::fusion::{Fusion, FusionConfig};
use nemotron_mlx::model::TimedToken;

fn tok(id: u32, frame: u64) -> TimedToken {
    TimedToken { id, frame }
}

fn spk(s: usize) -> [f32; 4] {
    let mut p = [0.05; 4];
    p[s] = 0.9;
    p
}

fn sil() -> [f32; 4] {
    [0.05; 4]
}

fn ids(text_stub: &str) -> impl Fn(&[u32]) -> String + '_ {
    move |ids| format!("{text_stub}:{}", ids.len())
}

#[test]
fn attributes_tokens_to_dominant_smoothed_speaker() {
    let mut fusion = Fusion::new(FusionConfig::default());

    // 20 frames of speaker 0, then 20 frames of speaker 1.
    let mut frames = Vec::new();
    for _ in 0..20 {
        frames.push(spk(0));
    }
    for _ in 0..20 {
        frames.push(spk(1));
    }
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 5), tok(2, 25)]);

    let segments = fusion.segments(ids("t"));
    assert_eq!(segments.len(), 2);

    assert_eq!(segments[0].speaker, 0);
    assert_eq!(segments[0].start_ms, 5 * 80);
    assert_eq!(segments[0].end_ms, (5 + 1) * 80);
    assert!(segments[0].is_final);
    assert_eq!(segments[0].text, "t:1");

    assert_eq!(segments[1].speaker, 1);
    assert_eq!(segments[1].start_ms, 25 * 80);
    assert_eq!(segments[1].end_ms, (25 + 1) * 80);
    assert!(segments[1].is_final);
    assert_eq!(segments[1].text, "t:1");
}

#[test]
fn silence_inherits_previous_speaker() {
    let mut fusion = Fusion::new(FusionConfig::default());

    // Speaker 0 frames, then silence frames, with a decidable token in the
    // silence region. The token should still be attributed to speaker 0 and
    // merged into the same segment as the earlier speaker-0 token.
    let mut frames = Vec::new();
    for _ in 0..10 {
        frames.push(spk(0));
    }
    for _ in 0..30 {
        frames.push(sil());
    }
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 3), tok(2, 20)]);
    fusion.flush();

    let segments = fusion.segments(ids("t"));
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].speaker, 0);
    assert!(segments[0].is_final);
    assert_eq!(segments[0].text, "t:2");
}

#[test]
fn first_token_in_silence_defaults_to_speaker_zero() {
    let mut fusion = Fusion::new(FusionConfig::default());

    let frames = vec![sil(); 30];
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 10)]);
    fusion.flush();

    let segments = fusion.segments(ids("t"));
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].speaker, 0);
    assert!(segments[0].is_final);
}

#[test]
fn short_turns_merge_into_previous_neighbor() {
    let mut fusion = Fusion::new(FusionConfig::default());

    // speaker 0 long run (with two tokens spanning it, so its derived
    // segment is legitimately long), then speaker 1 for 400ms (5 frames at
    // 80ms, with two tokens), then speaker 0 long run again (two tokens).
    // The middle segment is shorter than the 500ms min_turn_ms and both its
    // neighbors are long, so it merges into the previous (speaker 0)
    // segment and, since that reunites two runs of the same speaker, the
    // whole thing collapses into a single final segment.
    let mut frames = Vec::new();
    for _ in 0..30 {
        frames.push(spk(0));
    }
    for _ in 0..5 {
        frames.push(spk(1));
    }
    for _ in 0..30 {
        frames.push(spk(0));
    }
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[
        tok(1, 2),
        tok(2, 25),
        tok(3, 31),
        tok(4, 32),
        tok(5, 40),
        tok(6, 60),
    ]);
    fusion.flush();

    let segments = fusion.segments(ids("t"));
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].speaker, 0);
    assert!(segments[0].is_final);
    // All six tokens (including the short-turn middle ones) are included.
    assert_eq!(segments[0].text, "t:6");
}

#[test]
fn tokens_beyond_diar_horizon_form_tentative_tail() {
    let mut fusion = Fusion::new(FusionConfig::default());

    // Only 10 diar frames available; a token at frame 30 cannot be decided
    // (frame + smooth_frames >= diar.len()) until flush.
    let frames = vec![spk(0); 10];
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 5), tok(2, 30)]);

    let segments = fusion.segments(ids("t"));
    assert_eq!(segments.len(), 2);

    assert_eq!(segments[0].speaker, 0);
    assert!(segments[0].is_final);
    assert_eq!(segments[0].text, "t:1");

    assert_eq!(segments[1].speaker, 0);
    assert!(!segments[1].is_final);
    assert_eq!(segments[1].text, "t:1");
}

#[test]
fn flush_finalizes_tail_and_is_terminal() {
    let mut fusion = Fusion::new(FusionConfig::default());

    let frames = vec![spk(0); 10];
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 5), tok(2, 30)]);
    fusion.flush();

    let segments = fusion.segments(ids("t"));
    assert!(segments.iter().all(|s| s.is_final));
    assert!(!segments.iter().any(|s| !s.is_final));

    // Calling segments() again is stable.
    let segments_again = fusion.segments(ids("t"));
    assert_eq!(segments, segments_again);
}

#[test]
fn reset_clears_everything() {
    let mut fusion = Fusion::new(FusionConfig::default());

    let frames = vec![spk(0); 10];
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 5)]);
    fusion.flush();
    assert!(!fusion.segments(ids("t")).is_empty());

    fusion.reset();
    assert!(fusion.segments(ids("t")).is_empty());

    // Fusion is usable again after reset.
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 5)]);
    fusion.flush();
    assert_eq!(fusion.segments(ids("t")).len(), 1);
}

#[test]
fn segment_times_derive_from_frames() {
    let mut fusion = Fusion::new(FusionConfig::default());

    let frames = vec![spk(0); 20];
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 2), tok(2, 3), tok(3, 4)]);
    fusion.flush();

    let segments = fusion.segments(ids("t"));
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_ms, 2 * 80);
    assert_eq!(segments[0].end_ms, (4 + 1) * 80);
}

/// Edge case beyond the 8 required scenarios: `flush()` is not a terminal
/// lock on the streams themselves — it only marks "everything decidable so
/// far, and forever after" for this `Fusion`. Pushing more tokens/frames
/// after a flush is accepted, and those new tokens are immediately
/// decidable (using whatever diar frames exist at `segments()` time), since
/// `flushed` stays `true` once set.
#[test]
fn pushes_after_flush_remain_decidable() {
    let mut fusion = Fusion::new(FusionConfig::default());

    let frames = vec![spk(0); 10];
    fusion.push_diar_frames(&frames);
    fusion.push_tokens(&[tok(1, 5)]);
    fusion.flush();
    assert!(fusion.segments(ids("t")).iter().all(|s| s.is_final));

    // Push a token far beyond the diar horizon that existed at flush time.
    // Without a second flush() call it would normally be undecidable, but
    // since this Fusion was already flushed, it must resolve immediately.
    fusion.push_tokens(&[tok(2, 500)]);
    let segments = fusion.segments(ids("t"));
    assert!(segments.iter().all(|s| s.is_final));
    assert!(!segments.iter().any(|s| !s.is_final));
}
