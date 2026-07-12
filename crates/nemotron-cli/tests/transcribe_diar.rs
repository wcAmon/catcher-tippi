//! End-to-end test of the streaming ASR + diarization + fusion pipeline that
//! backs `catcher transcribe --diar-model`. Drives the crates directly (not
//! the compiled binary) so the test can inspect the segment list without
//! shelling out or scraping stdout.

use std::sync::{Mutex, MutexGuard};

use nemotron_mlx::{
    fusion::{Fusion, FusionConfig},
    model::StreamingTranscriber,
    opencc,
    tokenizer::Tokenizer,
    weights::Artifact,
};
use sortformer_mlx::stream::StreamingDiarizer;

/// MLX evaluates onto a process-global Metal command buffer that is not safe
/// for concurrent submission (see `crates/catcher-ffi/tests/ffi_lifecycle.rs`).
/// This file only has one MLX-driving test today, so the mutex is a no-op in
/// practice, but it is kept so a second MLX test added later doesn't have to
/// rediscover the "already encoding to this command buffer" failure mode.
static MLX_PIPELINE: Mutex<()> = Mutex::new(());

fn serialize_mlx() -> MutexGuard<'static, ()> {
    MLX_PIPELINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn read_wav_samples(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open fixture wav");
    reader
        .samples::<i16>()
        .map(|sample| sample.expect("decode sample") as f32 / 32768.0)
        .collect()
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn speaker_attributed_transcription_matches_conversation_fixture() {
    let _guard = serialize_mlx();

    let asr_model = std::env::var("NEMOTRON_MLX_ARTIFACT")
        .expect("set NEMOTRON_MLX_ARTIFACT to the converted Catcher ASR artifact directory");
    let diar_model = std::env::var("SORTFORMER_MLX_ARTIFACT")
        .expect("set SORTFORMER_MLX_ARTIFACT to the converted Sortformer artifact directory");

    let audio_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.wav"
    );
    let turns_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.json"
    );

    let samples = read_wav_samples(audio_path);

    let artifact = Artifact::load(&asr_model).expect("load ASR artifact");
    let mut transcriber =
        StreamingTranscriber::new(&artifact, "auto", 3).expect("build streaming transcriber");
    let tokenizer = Tokenizer::from_json(
        std::path::Path::new(&asr_model).join("tokenizer.json"),
        0,
        13_087,
    )
    .expect("load tokenizer");
    let mut diarizer =
        StreamingDiarizer::from_artifact_dir(&diar_model).expect("build streaming diarizer");
    let mut fusion = Fusion::new(FusionConfig::default());

    for chunk in samples.chunks(1_600) {
        let tokens = transcriber.push_samples(chunk).expect("push ASR samples");
        if !tokens.is_empty() {
            fusion.push_tokens(&tokens);
        }
        let frames = diarizer
            .push_samples(chunk)
            .expect("push diarization samples");
        if !frames.is_empty() {
            fusion.push_diar_frames(&frames);
        }
    }
    let tokens = transcriber.finish().expect("finish ASR");
    if !tokens.is_empty() {
        fusion.push_tokens(&tokens);
    }
    let frames = diarizer.finish().expect("finish diarization");
    if !frames.is_empty() {
        fusion.push_diar_frames(&frames);
    }
    fusion.flush();

    let segments = fusion.segments(|ids| {
        tokenizer
            .decode(ids, true)
            .map(|decoded| opencc::to_traditional(&decoded))
            .unwrap_or_default()
    });

    assert!(!segments.is_empty(), "expected at least one segment");

    // All segments must be final: `fusion.flush()` marks every pushed token
    // decidable, so no tentative tail should survive.
    for segment in &segments {
        assert!(
            segment.is_final,
            "segment must be final after finish+flush: {segment:?}"
        );
        assert!(
            !segment.text.trim().is_empty(),
            "segment text must be non-empty: {segment:?}"
        );
    }

    // Distinct speakers and at least one alternation.
    let distinct_speakers: std::collections::BTreeSet<u8> =
        segments.iter().map(|s| s.speaker).collect();
    assert!(
        distinct_speakers.len() >= 2,
        "expected >=2 distinct speakers, got {distinct_speakers:?}"
    );
    let alternations = segments
        .windows(2)
        .filter(|pair| pair[0].speaker != pair[1].speaker)
        .count();
    assert!(
        alternations >= 1,
        "expected >=1 speaker alternation across {} segments",
        segments.len()
    );

    // The fixture is Chinese (AISHELL-3); the concatenated transcript must
    // never contain simplified-only probe characters, since every output
    // path (`to_traditional`) converts to Taiwan-standard Traditional
    // (s2twp).
    let concatenated: String = segments.iter().map(|s| s.text.as_str()).collect();
    for probe in ["们", "说", "这"] {
        assert!(
            !concatenated.contains(probe),
            "output must not contain simplified-only character {probe:?}: {concatenated:?}"
        );
    }

    // Loose turn-structure check against the fixture's ground-truth turns:
    // speaker identity in `segments` is arrival order, not the ground-truth
    // label, so map first-heard segment speaker -> first-labeled turn
    // speaker (in chronological order of first appearance) and require that
    // every constructed turn of >=2s overlaps a segment of the mapped
    // speaker.
    let turns_json = std::fs::read_to_string(turns_path).expect("read conversation.json");
    let turns_value: serde_json::Value =
        serde_json::from_str(&turns_json).expect("parse conversation.json");
    let turns = turns_value["turns"].as_array().expect("turns array");

    let mut first_labeled_order: Vec<u64> = Vec::new();
    for turn in turns {
        let speaker = turn["speaker"].as_u64().expect("turn speaker");
        if !first_labeled_order.contains(&speaker) {
            first_labeled_order.push(speaker);
        }
    }
    let mut first_heard_order: Vec<u8> = Vec::new();
    for segment in &segments {
        if !first_heard_order.contains(&segment.speaker) {
            first_heard_order.push(segment.speaker);
        }
    }

    let map_turn_speaker_to_segment_speaker = |turn_speaker: u64| -> Option<u8> {
        let position = first_labeled_order
            .iter()
            .position(|&s| s == turn_speaker)?;
        first_heard_order.get(position).copied()
    };

    let mut checked_turns = 0usize;
    for turn in turns {
        let start_s = turn["start_s"].as_f64().expect("turn start_s");
        let end_s = turn["end_s"].as_f64().expect("turn end_s");
        let duration_s = end_s - start_s;
        if duration_s < 2.0 {
            continue;
        }
        let turn_speaker = turn["speaker"].as_u64().expect("turn speaker");
        let Some(expected_speaker) = map_turn_speaker_to_segment_speaker(turn_speaker) else {
            continue;
        };
        let turn_start_ms = (start_s * 1000.0).round() as u64;
        let turn_end_ms = (end_s * 1000.0).round() as u64;

        let overlaps = segments.iter().any(|segment| {
            segment.speaker == expected_speaker
                && segment.start_ms < turn_end_ms
                && segment.end_ms > turn_start_ms
        });
        assert!(
            overlaps,
            "turn {turn:?} (mapped to segment speaker {expected_speaker}) has no overlapping \
             segment; segments={segments:?}"
        );
        checked_turns += 1;
    }
    assert!(
        checked_turns > 0,
        "expected at least one >=2s turn to check against; got {} turns total",
        turns.len()
    );
}
