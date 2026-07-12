use mlx_rs::{Array, ops::indexing::TryIndexOp};

use super::{ModelError, ModelResult, QuantizedLinear, Tensor3};
use crate::weights::Artifact;

/// A decoded token paired with the encoder frame it was emitted on.
///
/// `frame` is on the model's 80 ms subsampled grid: `frame * 80 ms` is the
/// token's approximate start time. Frame indices returned by
/// [`StreamingRnntDecoder::decode_frames`] are LOCAL to the decoded window;
/// [`super::StreamingTranscriber`] offsets them to a GLOBAL utterance frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct TimedToken {
    pub id: u32,
    pub frame: u64,
}

/// Hidden and cell vectors for one LSTM layer and one streaming utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct LstmState {
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

impl LstmState {
    pub fn zeros(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            cell: vec![0.0; hidden_size],
        }
    }

    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    pub fn cell(&self) -> &[f32] {
        &self.cell
    }
}

/// One PyTorch-compatible LSTM layer backed by two MLX INT8 matmuls.
#[derive(Debug)]
pub struct LstmCell {
    hidden_size: usize,
    input_projection: QuantizedLinear,
    recurrent_projection: QuantizedLinear,
    bias: Array,
}

impl LstmCell {
    #[allow(clippy::too_many_arguments)]
    pub fn from_f32(
        weight_ih: &[f32],
        weight_hh: &[f32],
        bias_ih: &[f32],
        bias_hh: &[f32],
        hidden_size: usize,
        group_size: usize,
    ) -> ModelResult<Self> {
        if bias_ih.len() != 4 * hidden_size || bias_hh.len() != 4 * hidden_size {
            return Err(ModelError::InvalidShape(
                "LSTM biases must contain four hidden-size gates".to_string(),
            ));
        }
        let combined_bias: Vec<f32> = bias_ih
            .iter()
            .zip(bias_hh)
            .map(|(left, right)| left + right)
            .collect();
        Ok(Self {
            hidden_size,
            input_projection: QuantizedLinear::from_f32(
                weight_ih,
                4 * hidden_size,
                hidden_size,
                &vec![0.0; 4 * hidden_size],
                group_size,
            )?,
            recurrent_projection: QuantizedLinear::from_f32(
                weight_hh,
                4 * hidden_size,
                hidden_size,
                &vec![0.0; 4 * hidden_size],
                group_size,
            )?,
            bias: Array::from_slice(&combined_bias, &[1, (4 * hidden_size) as i32])
                .as_type::<half::f16>()?,
        })
    }

    pub fn from_artifact(artifact: &Artifact, layer: usize) -> ModelResult<Self> {
        let prefix = "decoder.lstm";
        let weight_ih = format!("{prefix}.weight_ih_l{layer}");
        let weight_hh = format!("{prefix}.weight_hh_l{layer}");
        let bias_ih = format!("{prefix}.bias_ih_l{layer}");
        let bias_hh = format!("{prefix}.bias_hh_l{layer}");
        let input_projection = QuantizedLinear::from_artifact(artifact, &weight_ih, None)?;
        let recurrent_projection = QuantizedLinear::from_artifact(artifact, &weight_hh, None)?;
        let bias = artifact
            .f16_array(&bias_ih)?
            .add(&artifact.f16_array(&bias_hh)?)?;
        let hidden_size = input_projection.input_dims();
        Ok(Self {
            hidden_size,
            input_projection,
            recurrent_projection,
            bias: bias.reshape(&[1, (4 * hidden_size) as i32])?,
        })
    }

    /// Advances a single batch-one timestep using PyTorch's `i,f,g,o` gate order.
    pub fn step_f32(&self, input: &[f32], state: &mut LstmState) -> ModelResult<Vec<f32>> {
        if input.len() != self.hidden_size
            || state.hidden.len() != self.hidden_size
            || state.cell.len() != self.hidden_size
        {
            return Err(ModelError::InvalidShape(format!(
                "LSTM step requires vectors of length {}",
                self.hidden_size
            )));
        }
        let input = Array::from_slice(input, &[1, self.hidden_size as i32]);
        let previous_hidden = Array::from_slice(&state.hidden, &[1, self.hidden_size as i32]);
        let previous_cell =
            Array::from_slice(&state.cell, &[1, self.hidden_size as i32]).as_type::<half::f16>()?;
        let gates = self
            .input_projection
            .forward_array(&input)?
            .add(&self.recurrent_projection.forward_array(&previous_hidden)?)?
            .add(&self.bias)?;
        let hidden = self.hidden_size as i32;
        let input_gate = mlx_rs::ops::sigmoid(gates.try_index((.., 0..hidden))?)?;
        let forget_gate = mlx_rs::ops::sigmoid(gates.try_index((.., hidden..2 * hidden))?)?;
        let candidate = mlx_rs::ops::tanh(gates.try_index((.., 2 * hidden..3 * hidden))?)?;
        let output_gate = mlx_rs::ops::sigmoid(gates.try_index((.., 3 * hidden..4 * hidden))?)?;
        let cell = forget_gate
            .multiply(&previous_cell)?
            .add(&input_gate.multiply(&candidate)?)?;
        let next_hidden = output_gate.multiply(&mlx_rs::ops::tanh(&cell)?)?;

        let cell_f32 = cell.as_type::<f32>()?;
        let hidden_f32 = next_hidden.as_type::<f32>()?;
        mlx_rs::transforms::eval([&cell_f32, &hidden_f32])?;
        state.cell = cell_f32.try_as_slice::<f32>()?.to_vec();
        state.hidden = hidden_f32.try_as_slice::<f32>()?.to_vec();
        Ok(state.hidden.clone())
    }
}

/// INT8-on-disk embedding dequantized once to FP16 for efficient row lookup.
#[derive(Debug)]
pub struct QuantizedEmbedding {
    vocab_size: usize,
    hidden_size: usize,
    values: Array,
}

impl QuantizedEmbedding {
    pub fn from_f32(
        weight: &[f32],
        vocab_size: usize,
        hidden_size: usize,
        group_size: usize,
    ) -> ModelResult<Self> {
        if weight.len() != vocab_size * hidden_size || hidden_size % group_size != 0 {
            return Err(ModelError::InvalidShape(
                "embedding weight shape or group size is invalid".to_string(),
            ));
        }
        let weight = Array::from_slice(weight, &[vocab_size as i32, hidden_size as i32])
            .as_type::<half::f16>()?;
        let (packed, scales, biases) = mlx_rs::ops::quantize(&weight, group_size as i32, 8)?;
        let values = mlx_rs::ops::dequantize(&packed, &scales, &biases, group_size as i32, 8)?;
        Ok(Self {
            vocab_size,
            hidden_size,
            values,
        })
    }

    pub fn from_artifact(artifact: &Artifact, weight_name: &str) -> ModelResult<Self> {
        let (packed, scales, biases, group_size, shape) = artifact.quantized_arrays(weight_name)?;
        if shape.len() != 2 {
            return Err(ModelError::InvalidShape(format!(
                "embedding artifact {weight_name} must have rank 2"
            )));
        }
        let values = mlx_rs::ops::dequantize(&packed, &scales, &biases, group_size as i32, 8)?;
        Ok(Self {
            vocab_size: shape[0],
            hidden_size: shape[1],
            values,
        })
    }

    fn lookup_f32(&self, token: u32) -> ModelResult<Vec<f32>> {
        if token as usize >= self.vocab_size {
            return Err(ModelError::InvalidShape(format!(
                "embedding token {token} is outside vocabulary {}",
                self.vocab_size
            )));
        }
        let row = self.values.try_index(token as i32)?.as_type::<f32>()?;
        row.eval()?;
        Ok(row.try_as_slice::<f32>()?.to_vec())
    }
}

/// Persistent prediction-network state for one streaming utterance.
#[derive(Debug, Clone)]
pub struct PredictionState {
    layers: Vec<LstmState>,
    cached_output: Vec<f32>,
    initialized: bool,
}

impl PredictionState {
    pub fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LstmState::zeros(hidden_size))
                .collect(),
            cached_output: vec![0.0; hidden_size],
            initialized: false,
        }
    }

    pub fn layers(&self) -> &[LstmState] {
        &self.layers
    }
}

/// Embedding, stacked LSTM, and decoder projection used by RNNT.
#[derive(Debug)]
pub struct PredictionNetwork {
    embedding: QuantizedEmbedding,
    cells: Vec<LstmCell>,
    projector: QuantizedLinear,
    blank_token_id: u32,
}

impl PredictionNetwork {
    pub fn new(
        embedding: QuantizedEmbedding,
        cells: Vec<LstmCell>,
        projector: QuantizedLinear,
        blank_token_id: u32,
    ) -> ModelResult<Self> {
        if cells.is_empty()
            || cells
                .iter()
                .any(|cell| cell.hidden_size != embedding.hidden_size)
        {
            return Err(ModelError::InvalidShape(
                "prediction LSTM dimensions do not match embedding".to_string(),
            ));
        }
        Ok(Self {
            embedding,
            cells,
            projector,
            blank_token_id,
        })
    }

    pub fn from_artifact(artifact: &Artifact) -> ModelResult<Self> {
        Self::new(
            QuantizedEmbedding::from_artifact(artifact, "decoder.embedding.weight")?,
            vec![
                LstmCell::from_artifact(artifact, 0)?,
                LstmCell::from_artifact(artifact, 1)?,
            ],
            QuantizedLinear::from_artifact(
                artifact,
                "decoder.decoder_projector.weight",
                Some("decoder.decoder_projector.bias"),
            )?,
            13_087,
        )
    }

    pub fn step(&self, token: u32, state: &mut PredictionState) -> ModelResult<Vec<f32>> {
        if state.layers.len() != self.cells.len() {
            return Err(ModelError::InvalidShape(
                "prediction state has incorrect layer count".to_string(),
            ));
        }
        if state.initialized && token == self.blank_token_id {
            return Ok(state.cached_output.clone());
        }
        let mut hidden = self.embedding.lookup_f32(token)?;
        for (cell, layer_state) in self.cells.iter().zip(&mut state.layers) {
            hidden = cell.step_f32(&hidden, layer_state)?;
        }
        state.cached_output = self.projector.forward_f32(&hidden, 1)?;
        state.initialized = true;
        Ok(state.cached_output.clone())
    }
}

/// RNNT ReLU joint network and vocabulary head.
#[derive(Debug)]
pub struct JointNetwork {
    hidden_size: usize,
    head: QuantizedLinear,
}

impl JointNetwork {
    pub fn from_artifact(artifact: &Artifact) -> ModelResult<Self> {
        Ok(Self {
            hidden_size: 640,
            head: QuantizedLinear::from_artifact(
                artifact,
                "joint.head.weight",
                Some("joint.head.bias"),
            )?,
        })
    }

    pub fn score(&self, encoder: &[f32], decoder: &[f32]) -> ModelResult<u32> {
        if encoder.len() != self.hidden_size || decoder.len() != self.hidden_size {
            return Err(ModelError::InvalidShape(
                "joint inputs must both have decoder hidden size".to_string(),
            ));
        }
        let fused: Vec<f32> = encoder.iter().zip(decoder).map(|(a, b)| a + b).collect();
        let fused = mlx_rs::nn::relu(Array::from_slice(&fused, &[1, self.hidden_size as i32]))?;
        let logits = self.head.forward_array(&fused)?;
        let token = mlx_rs::ops::indexing::argmax_axis(&logits, -1, false)?.as_type::<u32>()?;
        token.eval()?;
        Ok(token.try_as_slice::<u32>()?[0])
    }
}

/// Complete RNNT head consuming encoder `[1,time,1024]` frames.
#[derive(Debug)]
pub struct StreamingRnntDecoder {
    encoder_projector: QuantizedLinear,
    prediction: PredictionNetwork,
    joint: JointNetwork,
    blank_token_id: u32,
    max_symbols_per_step: usize,
}

impl StreamingRnntDecoder {
    pub fn from_artifact(artifact: &Artifact) -> ModelResult<Self> {
        Ok(Self {
            encoder_projector: QuantizedLinear::from_artifact(
                artifact,
                "encoder_projector.weight",
                Some("encoder_projector.bias"),
            )?,
            prediction: PredictionNetwork::from_artifact(artifact)?,
            joint: JointNetwork::from_artifact(artifact)?,
            blank_token_id: 13_087,
            max_symbols_per_step: 10,
        })
    }

    pub fn new_state(&self) -> PredictionState {
        PredictionState::new(2, 640)
    }

    pub fn decode_frames(
        &self,
        encoded: &Tensor3,
        state: &mut PredictionState,
    ) -> ModelResult<Vec<TimedToken>> {
        if encoded.shape[0] != 1 || encoded.shape[2] != 1024 {
            return Err(ModelError::InvalidShape(
                "RNNT encoder input must have shape [1,time,1024]".to_string(),
            ));
        }
        let mut decoder_hidden = self.prediction.step(self.blank_token_id, state)?;
        let mut output = Vec::new();
        for (frame, chunk) in encoded.values.chunks_exact(1024).enumerate() {
            let encoder_hidden = self.encoder_projector.forward_f32(chunk, 1)?;
            for _ in 0..self.max_symbols_per_step {
                let token = self.joint.score(&encoder_hidden, &decoder_hidden)?;
                if token == self.blank_token_id {
                    break;
                }
                output.push(TimedToken {
                    id: token,
                    frame: frame as u64,
                });
                decoder_hidden = self.prediction.step(token, state)?;
            }
        }
        Ok(output)
    }
}

/// RNNT greedy-search state machine, separated from neural scoring for deterministic testing.
pub struct GreedyRnnt;

impl GreedyRnnt {
    /// Scores each encoder frame until blank or `max_symbols_per_step` emissions.
    pub fn decode_with(
        frame_count: usize,
        blank_token_id: u32,
        max_symbols_per_step: usize,
        mut score: impl FnMut(usize, usize) -> u32,
    ) -> Vec<TimedToken> {
        let mut output = Vec::new();
        for frame in 0..frame_count {
            for symbol in 0..max_symbols_per_step {
                let token = score(frame, symbol);
                if token == blank_token_id {
                    break;
                }
                output.push(TimedToken {
                    id: token,
                    frame: frame as u64,
                });
            }
        }
        output
    }
}
