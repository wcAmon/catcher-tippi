use approx::assert_abs_diff_eq;
use nemotron_mlx::audio::LogMelFrontend;

fn synthetic_audio(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| {
            let time = index as f32 / 16_000.0;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * time).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1200.0 * time).sin()
                + ((index % 31) as f32 - 15.0) * 0.0005
        })
        .collect()
}

#[test]
fn centered_features_match_the_published_frontend_reference() {
    let frontend = LogMelFrontend::nemotron();
    let frames = frontend.extract(&synthetic_audio(1600), true);

    assert_eq!(frames.len(), 10);
    assert!(frames.iter().all(|frame| frame.len() == 128));

    let bins = [0, 1, 5, 10, 20, 40, 80, 127];
    let references = [
        [
            -9.101493, -8.904302, -7.602791, -6.0763245, -2.515178, -10.496132, -7.321813,
            -8.829868,
        ],
        [
            -14.00288, -13.877358, -12.133504, -10.178206, -3.0286355, -11.540961, -9.994083,
            -9.654244,
        ],
        [
            -16.550058, -16.28926, -16.119799, -12.411482, -3.1358635, -11.955688, -10.157519,
            -9.673437,
        ],
        [
            -11.115188, -11.104209, -10.721158, -9.595677, -3.0761108, -11.387222, -9.937812,
            -9.673555,
        ],
    ];

    let mut maximum = (0.0_f32, 0, 0, 0.0_f32, 0.0_f32);
    for (frame, expected) in [0, 1, 4, 9].into_iter().zip(references) {
        for (bin, expected) in bins.into_iter().zip(expected) {
            let actual = frames[frame][bin];
            let difference = (actual - expected).abs();
            if difference > maximum.0 {
                maximum = (difference, frame, bin, actual, expected);
            }
        }
    }
    // RustFFT and NumPy/PyTorch FFTs use different accumulation orders in low-energy bins.
    assert!(maximum.0 <= 1.0e-3, "maximum reference error: {maximum:?}");
}

#[test]
fn silence_is_the_log_zero_guard() {
    let frontend = LogMelFrontend::nemotron();
    let frames = frontend.extract(&vec![0.0; 1600], true);

    for value in frames.into_iter().flatten() {
        assert_abs_diff_eq!(value, -16.635532, epsilon = 1.0e-5);
    }
}

#[test]
fn uncentered_overlap_reproduces_a_centered_frame() {
    let frontend = LogMelFrontend::nemotron();
    let audio = synthetic_audio(2000);
    let centered = frontend.extract(&audio, true);
    let start = 4 * 160 - 512 / 2;
    let uncentered = frontend.extract(&audio[start..start + 512], false);

    assert_eq!(uncentered.len(), 1);
    for (actual, expected) in uncentered[0].iter().zip(centered[4].iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 2.0e-4);
    }
}
