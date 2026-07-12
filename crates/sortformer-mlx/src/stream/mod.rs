//! Streaming state machine for Sortformer diarization.
//!
//! Pure `Vec<f32>` logic (no MLX, no model, no I/O): the FIFO queue and the
//! AOSC (Arrival-Order Speaker Cache) that Task 7 wires to the real model.

mod aosc;
mod diarizer;

pub use aosc::{StreamingConfig, StreamingState, streaming_update};
pub use diarizer::StreamingDiarizer;
