//! Streaming parity of the push-based AOSC diarizer against NeMo's official
//! `forward_streaming` (synchronous, low-latency preset).
//!
//! `StreamingDiarizer` is fed the 47.47 s conversation fixture in 100 ms
//! pushes, then flushed. Its concatenated per-chunk speaker probabilities must
//! match `tests/fixtures/sortformer_streaming_reference.json` (`chunk_preds`,
//! 99 chunks x 6 frames x 4) within the INT8-vs-FP32 error-distribution gates
//! (see `assert_probability_gates`), and its FIFO/spkcache depths must match
//! `length_trajectory` at the first chunk, the first cache pop/compress
//! (chunk 31), and the last.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use sortformer_mlx::stream::StreamingDiarizer;

/// MLX evaluates onto a process-global Metal command buffer that is not safe
/// for concurrent submission. This binary has a single MLX-driving test today,
/// but any future addition must also take this guard so the tests run serially
/// (same pattern as tests/diarizer_parity.rs).
static MLX_PIPELINE: Mutex<()> = Mutex::new(());

fn serialize_mlx() -> MutexGuard<'static, ()> {
    MLX_PIPELINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(serde::Deserialize)]
struct Reference {
    num_chunks: usize,
    chunk_preds: Vec<Vec<Vec<f32>>>,
    length_trajectory: Vec<Lengths>,
    final_state: FinalState,
}

/// Post-`finish()` AOSC state captured from the NeMo reference run.
///
/// `mean_sil_emb` (a 512-d silence-embedding running mean) and `n_sil_frames`
/// are deliberately not asserted here: `StreamingState` keeps them as `pub`
/// fields, but `StreamingDiarizer` (the type this test drives) keeps its
/// `state: StreamingState` field private and exposes only `state_lengths()`.
/// Asserting them would need a new public accessor on `StreamingDiarizer`,
/// which is more than the "quick win" scope of this fixture-assertion pass —
/// left as a follow-up if silence-profile parity ever needs direct coverage.
#[derive(serde::Deserialize)]
struct FinalState {
    #[allow(dead_code)]
    mean_sil_emb: Vec<f32>,
    #[allow(dead_code)]
    n_sil_frames: u64,
    fifo_len: usize,
    spkcache_len: usize,
}

#[derive(serde::Deserialize, Clone, Copy)]
struct Lengths {
    #[allow(dead_code)]
    chunk: usize,
    fifo: usize,
    spkcache: usize,
}

impl Reference {
    /// Flattens `chunk_preds` (99 x 6 x 4) into per-frame speaker rows.
    fn flat_chunk_preds(&self) -> Vec<[f32; 4]> {
        self.chunk_preds
            .iter()
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|frame| [frame[0], frame[1], frame[2], frame[3]])
            })
            .collect()
    }
}

fn load_streaming_reference() -> Reference {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_streaming_reference.json"
    )))
    .unwrap()
}

fn artifact_dir() -> PathBuf {
    std::env::var_os("SORTFORMER_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set SORTFORMER_MLX_ARTIFACT")
}

fn conversation_audio() -> Vec<f32> {
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.wav"
    ))
    .unwrap();
    reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect()
}

/// Gates the streaming diarizer's per-frame speaker probabilities against the
/// NeMo `forward_streaming` reference.
///
/// The reference is captured from **FP32** NeMo (`restore_from`, no
/// quantization), while this crate runs the deployed **INT8** artifact. At the
/// steep part of the speaker sigmoid — the frame or two where a speaker turns
/// on/off — a small INT8 logit perturbation maps to a large probability delta,
/// so a handful of transition frames diverge by up to ~0.57 from the FP32
/// reference. This is inherent to INT8-vs-FP32 (the offline INT8 path shows the
/// same ~0.27 gap at the very first onset, frame 3) and is **not** a streaming
/// artifact: the large deltas are spread uniformly across all 99 chunks with no
/// growth after the chunk-31 cache compression, and flat (fully-on / fully-off)
/// frames match to the bit (`p50 == 0`).
///
/// So parity is gated on the shape of the whole error distribution rather than
/// its single worst sample: the bulk must be excellent (`mean-abs`), the tail
/// must stay thin (`p99`, transition-frame fraction), and a loose ceiling trips
/// only on catastrophic regression. A real bug — lc/rc bookkeeping, scaling, an
/// AOSC accumulation error — lifts the whole distribution and fails `mean-abs`
/// and `p99` together, not just one sample.
fn assert_probability_gates(ours: &[[f32; 4]], reference: &[[f32; 4]]) {
    let mut deltas: Vec<f32> = Vec::with_capacity(ours.len() * 4);
    let mut first_divergent_frame: Option<usize> = None;
    for (frame, (actual, expected)) in ours.iter().zip(reference).enumerate() {
        for speaker in 0..4 {
            let difference = (actual[speaker] - expected[speaker]).abs();
            if difference > 0.08 && first_divergent_frame.is_none() {
                first_divergent_frame = Some(frame);
            }
            deltas.push(difference);
        }
    }
    let count = deltas.len();
    let mean_abs = deltas.iter().sum::<f32>() / count as f32;
    let over_transition = deltas.iter().filter(|&&d| d > 0.08).count();
    let transition_fraction = over_transition as f32 / count as f32;
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |q: f64| deltas[((count as f64 - 1.0) * q) as usize];
    let (p50, p99, max_abs) = (percentile(0.5), percentile(0.99), percentile(1.0));

    // Chunk < 31 first divergence => forward / bookkeeping suspect; chunk >= 31
    // => AOSC integration suspect (the first pop + compression lands at 31).
    eprintln!(
        "streaming parity: mean-abs={mean_abs:.5} p50={p50:.5} p99={p99:.5} \
         max-abs={max_abs:.5} transition_fraction={transition_fraction:.4} \
         first_divergent_chunk={:?}",
        first_divergent_frame.map(|frame| frame / 6)
    );

    // Gate values are empirical tripwires set ~1.2-3x above the observed run
    // (mean 0.0062, p99 0.166, fraction 0.0194, max 0.568), not first-principles
    // tolerances; retighten them if the reference is ever regenerated.
    assert!(mean_abs <= 0.02, "mean-abs {mean_abs} > 0.02");
    assert!(
        p99 <= 0.25,
        "p99 {p99} > 0.25 (error tail widened — suspect a bug)"
    );
    assert!(
        transition_fraction <= 0.05,
        "transition_fraction {transition_fraction} > 0.05 (divergence became pervasive)"
    );
    assert!(
        max_abs <= 0.70,
        "max-abs {max_abs} > 0.70 (catastrophic regression)"
    );
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn streaming_chunks_match_nemo_forward_streaming() {
    let _guard = serialize_mlx();
    let reference = load_streaming_reference();
    let mut diarizer = StreamingDiarizer::from_artifact_dir(artifact_dir()).unwrap();
    assert_eq!(diarizer.frame_ms(), 80);

    let audio = conversation_audio();
    let mut ours: Vec<[f32; 4]> = Vec::new();
    // Probe FIFO/spkcache depth right after each of these chunks completes.
    let probes = [0usize, 31, reference.num_chunks - 1];
    let mut probed: Vec<Option<(usize, usize)>> = vec![None; probes.len()];
    let mut prev_chunks = 0usize;

    for piece in audio.chunks(1600) {
        ours.extend(diarizer.push_samples(piece).unwrap());
        // Every chunk emits exactly 6 frames, so ours.len() / 6 is the number of
        // completed chunks. When exactly one chunk completes in a push we can
        // attribute the observed state to that chunk (true for chunks 0 and 31,
        // which finish well before the final flush).
        let chunks_now = ours.len() / 6;
        if chunks_now == prev_chunks + 1 {
            if let Some(slot) = probes.iter().position(|&p| p == chunks_now - 1) {
                probed[slot] = Some(diarizer.state_lengths());
            }
        }
        prev_chunks = chunks_now;
    }
    ours.extend(diarizer.finish().unwrap());
    // The final chunk lands in finish(); its post-update state is the current one.
    let last_chunk = ours.len() / 6 - 1;
    if let Some(slot) = probes.iter().position(|&p| p == last_chunk) {
        probed[slot] = Some(diarizer.state_lengths());
    }

    // The fixture's `final_state` is the NeMo reference's post-finish AOSC
    // state; cross-check it against ours directly (previously deserialized
    // but never asserted).
    assert_eq!(
        diarizer.state_lengths(),
        (
            reference.final_state.fifo_len,
            reference.final_state.spkcache_len
        ),
        "post-finish (fifo_len, spkcache_len) mismatch against final_state fixture"
    );

    // Regression check for the finished-guard fix: pushing more audio after
    // finish() must be a no-op (empty output, no panic) rather than silently
    // buffering samples finish() will never revisit.
    assert!(
        diarizer.push_samples(&audio[..1600]).unwrap().is_empty(),
        "push_samples after finish() must return no frames"
    );

    let reference_preds = reference.flat_chunk_preds();
    assert_eq!(ours.len(), reference_preds.len(), "frame count mismatch");

    for (slot, &chunk) in probes.iter().enumerate() {
        let (fifo, spkcache) =
            probed[slot].unwrap_or_else(|| panic!("state not probed for chunk {chunk}"));
        let expected = reference.length_trajectory[chunk];
        assert_eq!(
            (fifo, spkcache),
            (expected.fifo, expected.spkcache),
            "length trajectory mismatch at chunk {chunk}"
        );
    }

    assert_probability_gates(&ours, &reference_preds);

    // reset() restores a fresh stream: empty cache, no completed chunks.
    diarizer.reset();
    assert_eq!(diarizer.state_lengths(), (0, 0));
    assert!(diarizer.push_samples(&audio[..1600]).unwrap().is_empty());
}
