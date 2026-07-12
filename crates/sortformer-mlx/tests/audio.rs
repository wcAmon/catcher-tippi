use sortformer_mlx::audio::MelFrontend;
use sortformer_mlx::config::SortformerConfig;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/sortformer_config.json"
));

fn fixture_config() -> SortformerConfig {
    SortformerConfig::from_json(FIXTURE).unwrap()
}

/// `extract_frames` must return the exact same rows as slicing the whole-signal
/// `extract` at those indices: both walk the identical per-frame body (centered
/// window over the preemphasized signal), so the values are bit-for-bit equal.
/// This is the streaming frontend's correctness contract: pushing audio and
/// extracting a chunk's mel window incrementally must never diverge from the
/// offline extraction the parity fixtures were captured against.
#[test]
fn extract_frames_equals_whole_signal_extract_bitwise() {
    let config = fixture_config();
    let frontend = MelFrontend::new(&config);
    let audio: Vec<f32> = (0..16_000 * 3).map(|i| ((i as f32) * 0.01).sin() * 0.4).collect();
    let whole = frontend.extract(&audio);
    for (start, count) in [(0usize, 10usize), (5, 48), (whole.len() - 7, 7)] {
        let part = frontend.extract_frames(&audio, start, count);
        assert_eq!(part.len(), count, "frame count for ({start},{count})");
        for (offset, frame) in part.iter().enumerate() {
            assert_eq!(frame, &whole[start + offset], "frame {} differs", start + offset);
        }
    }
}

#[test]
fn one_second_of_audio_yields_100_normalized_frames() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frontend = MelFrontend::new(&config);
    let audio: Vec<f32> = (0..16_000)
        .map(|index| (index as f32 * 0.02).sin() * 0.3)
        .collect();
    let frames = frontend.extract_normalized(&audio);
    assert!(
        (98..=101).contains(&frames.len()),
        "frames {}",
        frames.len()
    );
    assert!(frames.iter().all(|frame| frame.len() == config.n_mels));
    // Per-feature normalization: each mel bin is zero-mean unit-variance over time.
    for bin in 0..config.n_mels {
        let count = frames.len() as f32;
        let mean: f32 = frames.iter().map(|frame| frame[bin]).sum::<f32>() / count;
        let variance: f32 = frames
            .iter()
            .map(|frame| (frame[bin] - mean).powi(2))
            .sum::<f32>()
            / (count - 1.0);
        assert!(mean.abs() < 1e-3, "bin {bin} mean {mean}");
        assert!(
            (variance - 1.0).abs() < 2e-2,
            "bin {bin} variance {variance}"
        );
    }
}

/// `extract` must emit exactly `floor(T / hop)` frames, matching NeMo's
/// `get_seq_len` (features.py:403-407). `torch.stft(center=True)` produces
/// `floor(T/hop) + 1` columns, but NeMo's reported sequence length -- and thus
/// what the model consumes -- drops the trailing column. This pins that
/// formula for a length that is NOT a hop multiple, so `floor(T/hop) + 1`
/// (the old buggy count) is unambiguously distinct from `floor(T/hop)`.
#[test]
fn extract_emits_floor_of_length_over_hop_frames() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frontend = MelFrontend::new(&config);
    let hop_length = (config.hop_seconds * config.sample_rate as f64).round() as usize; // 160

    // 1680 = 10 * 160 + 80: floor(1680/160) = 10, but floor + 1 = 11.
    let length = 10 * hop_length + hop_length / 2;
    let audio: Vec<f32> = (0..length)
        .map(|index| (index as f32 * 0.03).sin() * 0.2)
        .collect();

    let frames = frontend.extract(&audio);
    let expected = length / hop_length; // floor division == floor(T/hop) == 10
    assert_eq!(
        frames.len(),
        expected,
        "expected floor(T/hop) = {expected} frames, got {}",
        frames.len()
    );
    // Guard the test's own premise: the old `floor(T/hop) + 1` count differs.
    assert_ne!(frames.len(), expected + 1);
}

/// torch.stft's `center=True` convention pads the signal by `n_fft / 2` and,
/// when `win_length < n_fft`, zero-pads the analysis window so it is
/// *centered* within the `n_fft` FFT frame (offset `(n_fft - win_length) / 2`
/// on each side). That centering makes frame `f`'s window centered exactly on
/// original sample `f * hop_length`.
///
/// For a single impulse placed exactly on a hop boundary (`k = m * hop`), the
/// two neighboring frames `m - 1` and `m + 1` sit at the same distance from
/// the impulse and must see (approximately) the same Hann-window weight on
/// it, so their mel energy should be roughly symmetric.
///
/// The buggy implementation instead left-aligns the 400-sample window inside
/// the 512-sample FFT buffer (no centering offset), which shifts the
/// effective analysis window 56 samples (3.5 ms) earlier. For an impulse on a
/// hop boundary this makes frame `m - 1` fall almost entirely outside the
/// (shifted) window while frame `m + 1` still catches a large fraction of it
/// -- a strongly asymmetric response that this test would catch.
#[test]
fn impulse_on_hop_boundary_yields_symmetric_neighbor_energy() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frontend = MelFrontend::new(&config);

    let hop_length = (0.01 * 16_000.0f64).round() as usize; // 160
    let center_frame = 50usize;
    let k = center_frame * hop_length; // 8000, far from either signal edge

    let mut audio = vec![0.0f32; 16_000];
    audio[k] = 1.0;

    let frames = frontend.extract_normalized(&audio);
    let left = center_frame - 1;
    let right = center_frame + 1;

    // L2 energy of the normalized mel vector, relative to the (near-zero)
    // background frames, is our proxy for "how much of the impulse this
    // frame's analysis window caught".
    let energy = |frame: &[f32]| -> f32 { frame.iter().map(|value| value * value).sum() };
    let left_energy = energy(&frames[left]);
    let right_energy = energy(&frames[right]);

    // Correctly centered: both neighbors are equidistant from the impulse,
    // so their captured energy should be within the same order of
    // magnitude (ratio close to 1). Left-aligned (buggy): frame `left`
    // barely overlaps the shifted window while frame `right` catches most
    // of it, driving the ratio toward zero.
    let ratio = left_energy / right_energy;
    assert!(
        ratio > 0.4 && ratio < 2.5,
        "expected roughly symmetric energy around frame {center_frame}: \
         frame {left} energy {left_energy}, frame {right} energy {right_energy}, ratio {ratio}"
    );
}
