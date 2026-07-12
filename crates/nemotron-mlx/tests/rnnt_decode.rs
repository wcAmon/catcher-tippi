use approx::assert_abs_diff_eq;
use nemotron_mlx::model::{
    GreedyRnnt, LstmCell, LstmState, PredictionNetwork, PredictionState, QuantizedEmbedding,
    QuantizedLinear, TimedToken,
};

static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn greedy_control_advances_only_on_blank_and_caps_symbols() {
    let mut calls = Vec::new();
    let tokens = GreedyRnnt::decode_with(3, 9, 2, |frame, symbol| {
        calls.push((frame, symbol));
        match (frame, symbol) {
            (0, 0) => 4,
            (0, 1) => 9,
            (1, _) => 5,
            (2, _) => 9,
            _ => unreachable!(),
        }
    });

    // Frame 0 emits one token then blanks out (advances to frame 1). Frame 1
    // caps at two symbols without ever seeing blank, so both share frame 1.
    // Frame 2 blanks immediately and emits nothing.
    assert_eq!(
        tokens,
        vec![
            TimedToken { id: 4, frame: 0 },
            TimedToken { id: 5, frame: 1 },
            TimedToken { id: 5, frame: 1 },
        ]
    );
    assert_eq!(calls, vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]);
}

#[test]
fn lstm_cell_uses_pytorch_ifgo_gate_order_on_mlx() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let hidden = 128;
    let weight = vec![0.001; 4 * hidden * hidden];
    let mut bias = vec![0.0; 4 * hidden];
    // i=0.5, f=0.5, g=0.5, o=0.5 for zero input/state.
    let candidate_bias = 0.5_f32.atanh();
    bias[2 * hidden..3 * hidden].fill(candidate_bias);
    let cell = LstmCell::from_f32(&weight, &weight, &bias, &[0.0; 4 * 128], hidden, 128).unwrap();
    let mut state = LstmState::zeros(hidden);

    let output = cell.step_f32(&vec![0.0; hidden], &mut state).unwrap();

    let expected_cell = 0.25_f32;
    let expected_hidden = 0.5 * expected_cell.tanh();
    assert_abs_diff_eq!(state.cell()[0], expected_cell, epsilon = 5.0e-4);
    assert_abs_diff_eq!(output[0], expected_hidden, epsilon = 5.0e-4);
    assert_abs_diff_eq!(state.hidden()[0], expected_hidden, epsilon = 5.0e-4);
}

#[test]
fn prediction_cache_does_not_advance_on_blank() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let hidden = 128;
    let embedding = QuantizedEmbedding::from_f32(&vec![0.01; 4 * hidden], 4, hidden, 128).unwrap();
    let weight = vec![0.001; 4 * hidden * hidden];
    let mut bias = vec![0.0; 4 * hidden];
    bias[2 * hidden..3 * hidden].fill(0.2);
    let cell = LstmCell::from_f32(&weight, &weight, &bias, &[0.0; 4 * 128], hidden, 128).unwrap();
    let projector =
        QuantizedLinear::from_f32(&identity(hidden), hidden, hidden, &[0.0; 128], 128).unwrap();
    let network = PredictionNetwork::new(embedding, vec![cell], projector, 3).unwrap();
    let mut state = PredictionState::new(1, hidden);

    let first = network.step(3, &mut state).unwrap();
    let first_hidden = state.layers()[0].hidden().to_vec();
    let blank_again = network.step(3, &mut state).unwrap();
    assert_eq!(blank_again, first);
    assert_eq!(state.layers()[0].hidden(), first_hidden);

    network.step(1, &mut state).unwrap();
    assert_ne!(state.layers()[0].hidden(), first_hidden);
}

fn identity(dimensions: usize) -> Vec<f32> {
    let mut values = vec![0.0; dimensions * dimensions];
    for index in 0..dimensions {
        values[index * dimensions + index] = 1.0;
    }
    values
}
