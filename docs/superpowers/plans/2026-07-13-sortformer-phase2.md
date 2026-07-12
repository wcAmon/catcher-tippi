# Sortformer Phase 2 (Streaming + Fusion + s2twp) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streaming AOSC diarization, ASR token timestamps, Rust fusion, OpenCC s2twp, and C ABI v2 so `catcher transcribe --diar-model` prints who-said-what-when in Traditional Chinese.

**Architecture:** Port NeMo's synchronous `forward_streaming` path into `sortformer-mlx` (per-chunk full re-forward over `[spkcache|fifo|chunk]`, recurrence in the AOSC embedding state); thread frame indices through `nemotron-mlx`'s RNNT decoder; fuse both 80 ms-grid streams in a pure state machine; convert all text with ferrous-opencc s2twp; expose segments through the extended C ABI.

**Tech Stack:** Rust (edition 2024), mlx-rs 0.25.3, ferrous-opencc 0.4, NeMo 2.7.3 (dev-machine reference generation only, `/tmp/sortformer-venv`).

**Spec:** `docs/superpowers/specs/2026-07-13-sortformer-phase2-design.md` — read it before any task.

## Global Constraints

- **Ground truth overrides this plan.** Checkpoint config, NeMo source (anchors below), and generated fixtures win over any scaffold code here. Before implementing a numeric component, read the anchored NeMo lines. (Phase 1 lesson: the plan's STFT scaffold was wrong; the fixture caught it.)
- NeMo anchors (installed venv `/tmp/sortformer-venv/lib/python3.11/site-packages/`): `nemo/collections/asr/models/sortformer_diar_models.py` — `forward_streaming` :627–716, `forward_streaming_step` :718–815, `frontend_encoder` :277–304; `nemo/collections/asr/modules/sortformer_modules.py` — `streaming_feat_loader` :208–256, `streaming_update` :526–609, `_boost_topk_scores` :611, `_get_silence_profile` :636, `_get_log_pred_scores` :669, `_get_topk_indices` :688, `_gather_spkcache_and_preds` :721, `_disable_low_scores` :782, `_compress_spkcache` :838–896.
- Low-latency preset (fixed, from spec): frame 80 ms; chunk_len 6; left_context 1; right_context 7; fifo_len 188; spkcache_len 188; spkcache_update_period 188; sil_frames_per_spk 3; scores_boost_latest 0.05; sil_threshold 0.2; strong_boost_rate 0.75; weak_boost_rate 1.5; min_pos_scores_rate 0.5. Checkpoint-config values (`chunk_len 188`, `fifo_len 0`, rc 1) are training defaults — do NOT use them.
- Phase 1 parity gates stay unchanged and must stay green: mel rms ≤ 2%, encoder rms ≤ 5%, end-to-end probabilities max-abs ≤ 0.05 / mean ≤ 0.01. The published HF artifact (`wcamon/catcher-diar-mlx-int8`) is immutable — nothing in this phase regenerates or republishes it.
- Model-gated tests: `#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]` with `std::env::var_os("SORTFORMER_MLX_ARTIFACT")` (ASR side: `NEMOTRON_MLX_ARTIFACT`). Fixtures live at repo-root `tests/fixtures/`, reached via `concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/…")`.
- New dependencies: **only** `ferrous-opencc = "0.4"` (in `nemotron-mlx`). No other new crates.
- mlx-rs idiom: follow `crates/nemotron-mlx/src/model/layers.rs` (`Array::from_slice(values, &[dims as i32])`, `mlx_rs::ops::*`, `.eval()?`, `.try_as_slice::<f32>()?`). Exact mlx-rs 0.25.3 method names (transpose/softmax variants) per its docs — keep the math, adapt the names.
- All user-visible Chinese output is Traditional (s2twp). Display speaker numbers are 1-based (說話者1); internal `speaker` fields are 0-based.
- Python tooling runs in `/tmp/sortformer-venv` (Homebrew py3.11 + torch + NeMo 2.7.3); source checkpoint at `/tmp/sortformer-src`. Never print or commit the HF token in `.env`.
- Commit style: conventional commits (`feat:`, `test:`, `docs:`, `refactor:`), frequent.

---

### Task 1: Shared ops module + `xscaling` from config

**Files:**
- Create: `crates/sortformer-mlx/src/model/ops.rs`
- Modify: `crates/sortformer-mlx/src/model/mod.rs`, `src/model/encoder.rs`, `src/model/transformer.rs`, `src/config.rs`
- Test: `crates/sortformer-mlx/tests/config.rs` (extend)

**Interfaces:**
- Consumes: existing `Norm` (encoder.rs:339, transformer.rs:128 — same math, both NeMo `nn.LayerNorm`), free helpers `relu_in_place`/`silu_in_place`/`softmax_in_place`/`add_in_place` duplicated across both files.
- Produces: `pub(crate) mod ops` re-exporting `Norm` (with `from_artifact(artifact, prefix)` and `forward(&[f32], rows)`), `relu_in_place`, `silu_in_place`, `softmax_in_place`, `add_in_place`. `SortformerConfig` gains `pub xscaling: bool`.

- [ ] **Step 1: Failing test for config parsing.** In `tests/config.rs`, add:

```rust
#[test]
fn config_parses_xscaling_and_defaults_to_true_when_absent() {
    let config = SortformerConfig::load(fixture_dir()).unwrap();
    assert!(config.xscaling);
    let stripped: serde_json::Value = {
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture_path()).unwrap()).unwrap();
        // remove the field however it is nested; inspect the fixture first
        strip_xscaling(&mut v);
        v
    };
    let config = SortformerConfig::from_json(&stripped.to_string()).unwrap();
    assert!(config.xscaling, "absent xscaling must default to true (published artifact compat)");
}
```

First inspect `tests/fixtures/sortformer_config.json` to find where `xscaling` lives (expected under the encoder block; the export script may or may not have propagated it — if absent entirely, the test's first assert still passes via the default, which is the point). Match the existing test file's helpers for paths.

- [ ] **Step 2: Run `cargo test -p sortformer-mlx --test config`** — expect FAIL (no `xscaling` field).
- [ ] **Step 3: Implement.** Add `#[serde(default = "default_true")] pub xscaling: bool` to `SortformerConfig` following how its 17 existing fields are parsed (config.rs:60–78); in `Encoder::from_artifact` (encoder.rs:49) set `input_scale = if config.xscaling { (d_model as f32).sqrt() } else { 1.0 }`, replacing the hardcoded `sqrt(512)`.
- [ ] **Step 4: Extract ops.** Create `src/model/ops.rs`; move `Norm` and the four in-place helpers there once; delete the duplicates from `encoder.rs` and `transformer.rs` and import from `ops`. Keep `BatchNorm` in `encoder.rs` (encoder-only). No behavior change.
- [ ] **Step 5: Run the full suite** — `cargo test -p sortformer-mlx` green; with the artifact present also run `SORTFORMER_MLX_ARTIFACT=… cargo test -p sortformer-mlx -- --ignored` green (parity gates unchanged).
- [ ] **Step 6: Commit** `refactor: consolidate sortformer ops and parse xscaling from config`.

---

### Task 2: Attention as MLX matmuls

**Files:**
- Modify: `crates/sortformer-mlx/src/model/transformer.rs:202-235` (`SelfAttention::forward`), `crates/sortformer-mlx/src/model/encoder.rs:511-563` (`SelfAttention::forward`)
- Test: existing parity suites (`tests/encoder_parity.rs`, `tests/diarizer_parity.rs`) are the gate; add one timing probe.

**Interfaces:**
- Consumes: `Linear::forward` / `QuantizedLinear::forward_f32` (unchanged), `relative_shift` (encoder.rs, unchanged), `ops::softmax_in_place` (dropped from the attention path).
- Produces: identical signatures — `transformer.rs: fn forward(&self, input: &[f32], frames: usize) -> ModelResult<Vec<f32>>`; `encoder.rs: fn forward(&self, input: &[f32], frames: usize, positions: &Tensor3) -> ModelResult<Vec<f32>>`. Callers unaffected.

This is a behavior-preserving rewrite: replace the per-query/per-key scalar loops with batched MLX ops. Same math, same outputs within FP tolerance; the parity fixtures are the referee.

- [ ] **Step 1: Transformer attention (no positional term).** Rewrite `transformer.rs` `SelfAttention::forward` as:

```rust
fn forward(&self, input: &[f32], frames: usize) -> ModelResult<Vec<f32>> {
    let queries = self.query_net.forward(input, frames)?;
    let keys = self.key_net.forward(input, frames)?;
    let values = self.value_net.forward(input, frames)?;
    let scale = 1.0 / (self.head_dim as f32).sqrt();
    let split = [frames as i32, self.heads as i32, self.head_dim as i32];
    // [T, H, D] -> [H, T, D]
    let q = Array::from_slice(&queries, &split).transpose_axes(&[1, 0, 2])?;
    let k = Array::from_slice(&keys, &split).transpose_axes(&[1, 0, 2])?;
    let v = Array::from_slice(&values, &split).transpose_axes(&[1, 0, 2])?;
    // [H, T, T]
    let scores = q.matmul(&k.transpose_axes(&[0, 2, 1])?)?.multiply(array!(scale))?;
    let probabilities = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
    // [H, T, D] -> [T, H*D]
    let attended = probabilities
        .matmul(&v)?
        .transpose_axes(&[1, 0, 2])?
        .reshape(&[frames as i32, self.hidden_size as i32])?;
    attended.eval()?;
    self.out_projection.forward(attended.try_as_slice::<f32>()?, frames)
}
```

(Adapt method names to the mlx-rs 0.25.3 API actually exposed; `layers.rs` shows the established call style.)

- [ ] **Step 2: Encoder rel-pos attention.** Same pattern with the Transformer-XL split (current scalar code encoder.rs:511–563 is the math spec):
  - content scores: `(q + bias_u) · kᵀ` — add `bias_u` (shape `[H*D]`, reshape `[H, 1, D]`) to `q` before the matmul → `[H, T, T]`.
  - positional scores: `(q + bias_v) · pᵀ` with `p = linear_pos(positions)` reshaped `[H, P, D]` → raw `[H, T, P]`; **eval and pull back**, apply the existing `relative_shift(raw, frames, position_frames)` per head on CPU (it is a pure reindex, O(T·P) copies — not the bottleneck), then rebuild an `[H, T, T]` array from the shifted slices.
  - `scores = (content + shifted) * scale`, softmax over the last axis, `× v`, merge heads, `linear_out`.
- [ ] **Step 3: Timing probe.** In `tests/diarizer_parity.rs` add:

```rust
#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn diarize_reports_wall_time() {
    let diarizer = load_diarizer();               // reuse the file's existing loader
    let audio = fixture_audio();
    let started = std::time::Instant::now();
    diarizer.diarize(&audio).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let rtf = seconds / (audio.len() as f64 / 16_000.0);
    eprintln!("offline diarize RTF = {rtf:.3}");
    assert!(rtf < 1.0, "offline diarize slower than real time: {rtf:.3}");
}
```

- [ ] **Step 4: Verify parity.** `SORTFORMER_MLX_ARTIFACT=… cargo test -p sortformer-mlx -- --ignored --nocapture` — all Phase 1 gates green, RTF printed and < 1.0. Then `cargo test -p sortformer-mlx` (unit suite) green.
- [ ] **Step 5: Commit** `perf: batch sortformer attention through MLX matmuls`.

---

### Task 3: Two-speaker conversation fixture

**Files:**
- Create: `tools/make_conversation_fixture.py`, `tests/fixtures/conversation.wav`, `tests/fixtures/conversation.json`, `tests/fixtures/README.md`

**Interfaces:**
- Produces: `tests/fixtures/conversation.wav` — 16 kHz mono PCM16, 40–60 s, exactly 2 speakers alternating turns. `tests/fixtures/conversation.json` — the constructed turn map: `{"turns": [{"speaker": 0, "start_s": 0.0, "end_s": 4.2}, …], "source": "...", "license": "..."}`. Tasks 4, 7, 12 consume both.

- [ ] **Step 1: Source audio.** Preference order (spec): (1) **AISHELL-3** (Apache-2.0, Mandarin, openslr.org SLR93 / HF `AISHELL/AISHELL-3` — per-utterance WAVs by speaker ID, resample 44.1 kHz→16 kHz); (2) Common Voice zh-TW (CC0; HF gated — token in `.env` if needed, never print it); (3) LibriSpeech dev-clean (CC BY 4.0, English fallback — s2twp then validated by text fixtures only, per spec). Download a handful of utterances from **two different speakers** into `/tmp/phase2-fixture-src/`.
- [ ] **Step 2: Write `tools/make_conversation_fixture.py`** (venv python; deps already present: numpy, soundfile or scipy — check `pip list` first, follow `tools/export_sortformer_weights.py` conventions). It must: load the chosen utterances, resample to 16 kHz mono, alternate speakers A/B with 0.3–0.7 s silence gaps to a total of 45–55 s, peak-normalize to 0.9, write `tests/fixtures/conversation.wav` (PCM16) and the turn map JSON (speaker 0 = first speaker heard). Deterministic: fixed utterance list and gap values hardcoded in the script, no RNG.
- [ ] **Step 3: Run it** and sanity-check: `python tools/make_conversation_fixture.py && soxi tests/fixtures/conversation.wav` (or read the header via python) — 16 kHz, 1 channel, 40–60 s. Duration must exceed 35 s so streaming exercises ≥ 1 FIFO pop AND ≥ 1 spkcache compression (fifo 188 + spkcache 188 frames ≈ 30.1 s).
- [ ] **Step 4: Write `tests/fixtures/README.md`** recording: source dataset, exact utterance IDs, license (with URL), construction command, and what `conversation.json` contains.
- [ ] **Step 5: Commit** `test: add two-speaker conversation fixture` (include the WAV; ~1.6 MB is fine).

---

### Task 4: Streaming reference generation (NeMo ground truth)

**Files:**
- Create: `tools/generate_sortformer_streaming_reference.py`, `tests/fixtures/sortformer_streaming_reference.json`

**Interfaces:**
- Consumes: `/tmp/sortformer-src` (.nemo checkpoint), `/tmp/sortformer-venv`, `tests/fixtures/conversation.wav`.
- Produces: `sortformer_streaming_reference.json` with this exact shape (consumed by Tasks 5 and 7):

```json
{
  "preset": {"chunk_len": 6, "chunk_left_context": 1, "chunk_right_context": 7,
              "fifo_len": 188, "spkcache_len": 188, "spkcache_update_period": 188},
  "num_chunks": 0,
  "chunk_preds": [[[0.0, 0.0, 0.0, 0.0]]],
  "chunk0_pre_encode": {"frames": 13, "dim": 512, "values": [0.0]},
  "final_state": {"mean_sil_emb": [0.0], "n_sil_frames": 0,
                   "fifo_len": 0, "spkcache_len": 0},
  "length_trajectory": [{"chunk": 0, "fifo": 0, "spkcache": 0}]
}
```

`chunk_preds[i]` is chunk i's kept prediction frames (≤ chunk_len × 4 floats; the last chunk may be shorter). `length_trajectory` records fifo/spkcache lengths after every chunk's update.

- [ ] **Step 1: Write the script**, following `tools/generate_sortformer_reference.py` conventions (model restore from .nemo via `SortformerEncLabelModel.restore_from`, CPU, eval mode, fp32). Configure the preset **before** running: set `model.sortformer_modules.chunk_len = 6`, `.chunk_left_context = 1`, `.chunk_right_context = 7`, `.fifo_len = 188`, `.spkcache_len = 188`, `.spkcache_update_period = 188`, and `model.streaming_mode = True`, async off. Hook `forward_streaming_step` to capture each step's returned chunk preds and post-update state lengths; capture `pre_encode` output for chunk 0 via a hook on `model.encoder.pre_encode`. Feed `conversation.wav` through `process_signal` + `forward_streaming` (models file :627–716). Also assert inside the script that the concatenated `chunk_preds` equals the `total_preds` returned by `forward_streaming` (sanity that capture == truth).
- [ ] **Step 2: Run** `/tmp/sortformer-venv/bin/python tools/generate_sortformer_streaming_reference.py --nemo /tmp/sortformer-src/<checkpoint>.nemo --audio tests/fixtures/conversation.wav --output tests/fixtures/sortformer_streaming_reference.json`. Verify: `num_chunks ≈ ceil(mel_frames / 48)` for ~50 s audio (≈ 105+), `length_trajectory` shows fifo reaching 188 then dropping (pop) and spkcache hitting 188 then compressing, file size ≲ 500 KB (preds dominate; chunk0_pre_encode ≈ 13×512 floats).
- [ ] **Step 3: Commit** `test: add NeMo streaming reference for sortformer AOSC parity`.

---

### Task 5: `Encoder::pre_encode` / `forward_embedded` split

**Files:**
- Modify: `crates/sortformer-mlx/src/model/encoder.rs` (`Encoder` impl, :47-131)
- Test: `crates/sortformer-mlx/tests/encoder_parity.rs` (extend)

**Interfaces:**
- Consumes: existing `Subsampling::forward` (encoder.rs:310), `run` (:93), `input_scale`.
- Produces (Task 7 relies on these exact signatures):

```rust
impl Encoder {
    /// dw-striding conv subsampling only: mel [T_mel][128] -> [1, T_mel/8, 512], UNSCALED.
    pub fn pre_encode(&self, mel_frames: &[Vec<f32>]) -> ModelResult<Tensor3>;
    /// 17 Conformer blocks over already-pre-encoded embeddings (NeMo bypass_pre_encode=True).
    /// Applies input_scale (xscaling) to the embeddings first, then rel-pos + blocks.
    pub fn forward_embedded(&self, embedded: &Tensor3) -> ModelResult<Tensor3>;
}
```

`forward(mel)` must remain exactly `forward_embedded(pre_encode(mel))` — refactor `run` so both paths share one code body; the existing offline parity tests prove the refactor is faithful.

**Scaling subtlety (verify, don't assume):** NeMo applies xscaling inside `pos_enc` over the whole concatenated `[spkcache|fifo|chunk]` sequence each step — the cached embeddings are stored **unscaled** (raw pre_encode output). So `pre_encode` must NOT scale; `forward_embedded` scales its full input. Confirm against `frontend_encoder` (:277–304) and the chunk0 parity below.

- [ ] **Step 1: Failing parity test.** In `tests/encoder_parity.rs`:

```rust
#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn pre_encode_matches_nemo_streaming_chunk0() {
    let reference = streaming_reference();           // loads sortformer_streaming_reference.json
    let config = fixture_config();
    let encoder = load_encoder();                    // follow the file's existing loader
    let audio = conversation_audio();
    let mel = MelFrontend::new(&config).extract(&audio);
    // chunk 0: no left context, 48 chunk + 56 right-context mel frames
    let chunk0 = &mel[..(48 + 56).min(mel.len())];
    let ours = encoder.pre_encode(chunk0).unwrap();
    assert_eq!(ours.shape[1], reference.chunk0_pre_encode.frames);
    assert_relative_rms_below(&ours.values, &reference.chunk0_pre_encode.values, 0.05);
}
```

Reuse/extend the file's existing fixture-loading and rms helpers; add `conversation_audio()` reading `tests/fixtures/conversation.wav` with the existing WAV loader pattern.

- [ ] **Step 2: Run** — FAIL (`pre_encode` not defined).
- [ ] **Step 3: Implement** the split: extract the subsampling call out of `run` into `pre_encode` (returns unscaled `Tensor3`); `forward_embedded` = scale by `input_scale` + positional encoding + 17 blocks (the remainder of `run`); `forward` composes them. `forward_trace` keeps working (it can call the split internals).
- [ ] **Step 4: Run all sortformer tests** (unit + `--ignored` with artifact): new test green, all Phase 1 parity green.
- [ ] **Step 5: Commit** `feat: split sortformer encoder into pre_encode and forward_embedded`.

---

### Task 6: AOSC state machine (pure logic)

**Files:**
- Create: `crates/sortformer-mlx/src/stream/aosc.rs` (+ `src/stream/mod.rs`, add `pub mod stream;` to `lib.rs`)
- Test: `crates/sortformer-mlx/tests/aosc.rs`

**Interfaces:**
- Consumes: nothing model-side — pure `Vec<f32>` math. Port from `sortformer_modules.py` :526–609 and :611–896 (anchors in Global Constraints).
- Produces (Task 7 relies on these):

```rust
pub struct StreamingConfig {
    pub chunk_len: usize, pub left_context: usize, pub right_context: usize,
    pub fifo_len: usize, pub spkcache_len: usize, pub update_period: usize,
    pub sil_frames_per_spk: usize, pub scores_boost_latest: f32, pub sil_threshold: f32,
    pub strong_boost_rate: f32, pub weak_boost_rate: f32, pub min_pos_scores_rate: f32,
    pub emb_dim: usize, pub num_speakers: usize,
}
impl StreamingConfig { pub fn low_latency_v2() -> Self /* preset table */ }

pub struct StreamingState {
    pub spkcache: Vec<Vec<f32>>, pub spkcache_preds: Option<Vec<Vec<f32>>>,
    pub fifo: Vec<Vec<f32>>, pub fifo_preds: Vec<Vec<f32>>,
    pub mean_sil_emb: Vec<f32>, pub n_sil_frames: u64,
}
impl StreamingState { pub fn new(config: &StreamingConfig) -> Self }

/// Sync-path streaming_update. `chunk_embs` includes lc+chunk+rc rows (pre_encode output);
/// `preds` covers [spkcache | fifo | lc | chunk | rc] rows of num_speakers floats each.
/// Trims contexts, refreshes fifo_preds from this step's preds, appends the chunk to the
/// fifo, pops update_period frames into the spkcache on overflow (updating the silence
/// profile), and compresses the spkcache via AOSC when it exceeds spkcache_len.
pub fn streaming_update(
    state: &mut StreamingState, config: &StreamingConfig,
    chunk_embs: &[Vec<f32>], preds: &[Vec<f32>], lc: usize, rc: usize,
);
```

(`preds` rows are `Vec<f32>` of length `num_speakers`, not `[f32; 4]`, so the pure logic is testable with any speaker count; Task 7 adapts.)

- [ ] **Step 1: Write failing unit tests first** (`tests/aosc.rs`) — pure synthetic, no model. Cover at minimum:

```rust
fn cfg() -> StreamingConfig {
    StreamingConfig { chunk_len: 4, left_context: 1, right_context: 2, fifo_len: 8,
        spkcache_len: 16, update_period: 8, sil_frames_per_spk: 1,
        scores_boost_latest: 0.05, sil_threshold: 0.2, strong_boost_rate: 0.75,
        weak_boost_rate: 1.5, min_pos_scores_rate: 0.5, emb_dim: 4, num_speakers: 2 }
}
fn frame(tag: f32) -> Vec<f32> { vec![tag; 4] }
fn speech(spk: usize) -> Vec<f32> { let mut p = vec![0.05; 2]; p[spk] = 0.95; p }
fn silence() -> Vec<f32> { vec![0.01; 2] }

#[test]
fn contexts_are_trimmed_and_chunk_lands_in_fifo() { /* push one chunk with lc=1, rc=2;
    fifo == the 4 chunk rows; fifo_preds == the 4 chunk pred rows */ }

#[test]
fn fifo_overflow_pops_update_period_frames_into_spkcache() { /* push chunks until
    fifo would exceed 8; assert popped count == clamp rule from :539-547 and
    spkcache grew by the same rows in order */ }

#[test]
fn silence_profile_accumulates_only_below_threshold_frames() { /* pop rows where
    sum(preds) < 0.2 update mean_sil_emb as a running mean and n_sil_frames;
    speech rows do not */ }

#[test]
fn compression_respects_per_speaker_quota_and_arrival_order() { /* overfill spkcache
    with tagged frames (frame(tag) so identity is recoverable); after compress:
    len == spkcache_len; per speaker ≥ strong floor of speech frames; the reserved
    sil_frames_per_spk slots contain mean_sil_emb; surviving tags appear in strictly
    increasing original order */ }

#[test]
fn latest_frames_survive_ties_via_score_boost() { /* two equal-scoring frame groups,
    newer group wins slots (scores_boost_latest) */ }
```

Write real bodies (construct chunks, call `streaming_update` repeatedly, assert). Tags in embedding values make survivorship checkable.

- [ ] **Step 2: Run** `cargo test -p sortformer-mlx --test aosc` — FAIL (module missing).
- [ ] **Step 3: Implement `aosc.rs`** by porting the anchored NeMo functions in this order, keeping one private Rust fn per NeMo helper with the same name (`get_log_pred_scores`, `disable_low_scores`, `boost_topk_scores`, `get_silence_profile`, `get_topk_indices`, `gather_spkcache_and_preds`, `compress_spkcache`). Key facts from the source (verify each at its anchor):
  - scores: `log(p) − log(1−p) + Σ_s log(1−p_s) − log(0.5)` with NeMo's eps guard (read :669–686 for the exact clamping).
  - `disable_low_scores`: non-speech frames → `-inf`; when a speaker has ≥ `min_pos = floor(per_spk × min_pos_scores_rate)` positive frames, its non-positive frames → `-inf` (:782–808).
  - `per_spk = spkcache_len / num_speakers − sil_frames_per_spk`; strong = `floor(per_spk × strong_boost_rate)` boosted ×2; weak = `floor(per_spk × weak_boost_rate)` boosted ×1 (:611–634).
  - newest frames (index ≥ old spkcache_len) get `+scores_boost_latest`.
  - reserve `sil_frames_per_spk` `+inf` slots per speaker, filled with `mean_sil_emb` and zero preds on gather (:721–759).
  - top-k across the per-speaker score bands, then **sort selected indices ascending** — arrival order (:688–719, sort at :714).
  - pop rule on fifo overflow: `pop = clamp(update_period, min = chunk_len − fifo_len + fifo_len_cur, max = fifo_len_cur + chunk_len)` (:539–547).
  - `spkcache_preds` stays `None` until the first compression (sync path), then carries gathered preds.
- [ ] **Step 4: Run** — all `aosc.rs` tests green; `cargo test -p sortformer-mlx` green.
- [ ] **Step 5: Commit** `feat: add AOSC speaker-cache state machine`.

---

### Task 7: `StreamingDiarizer`

**Files:**
- Create: `crates/sortformer-mlx/src/stream/diarizer.rs` (re-export from `stream/mod.rs`: `pub use diarizer::StreamingDiarizer;` `pub use aosc::{StreamingConfig, StreamingState};`)
- Modify: `crates/sortformer-mlx/src/audio.rs` (add `MelFrontend::extract_frames`), `src/lib.rs`
- Test: `crates/sortformer-mlx/tests/streaming_parity.rs`, `tests/audio.rs` (extend)

**Interfaces:**
- Consumes: `Diarizer` internals (frontend/encoder/proj/layers/head — add a crate-visible accessor or build `StreamingDiarizer` from the same `from_artifact_dir` parts), `Encoder::pre_encode`/`forward_embedded` (Task 5), `streaming_update` + `StreamingConfig::low_latency_v2()` (Task 6).
- Produces (FFI Task 11 and CLI Task 12 rely on):

```rust
pub struct StreamingDiarizer { /* model parts + StreamingState + audio buffer + mel cursor */ }
impl StreamingDiarizer {
    pub fn from_artifact_dir(model_dir: impl AsRef<Path>) -> ModelResult<Self>;
    pub fn push_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<[f32; 4]>>;
    pub fn finish(&mut self) -> ModelResult<Vec<[f32; 4]>>;   // flush trailing partial chunk
    pub fn reset(&mut self);
    pub fn frame_ms(&self) -> u64;                            // 80
}
```

- [ ] **Step 1: Incremental mel, failing test first.** In `tests/audio.rs`:

```rust
#[test]
fn extract_frames_equals_whole_signal_extract_bitwise() {
    let config = fixture_config();
    let frontend = MelFrontend::new(&config);
    let audio: Vec<f32> = (0..16_000 * 3).map(|i| ((i as f32) * 0.01).sin() * 0.4).collect();
    let whole = frontend.extract(&audio);
    for (start, count) in [(0usize, 10usize), (5, 48), (whole.len() - 7, 7)] {
        let part = frontend.extract_frames(&audio, start, count);
        for (offset, frame) in part.iter().enumerate() {
            assert_eq!(frame, &whole[start + offset], "frame {} differs", start + offset);
        }
    }
}
```

Implement `extract_frames(&self, audio: &[f32], start: usize, count: usize) -> Vec<Vec<f32>>` by refactoring `extract` so both share the per-frame body (`extract` = `extract_frames(audio, 0, total)`); bit-equality is the acceptance (same code path, centered window reads only `frame*hop ± n_fft/2` samples — preemphasis must also be computed identically, watch its one-sample lookback at window edges). Run: FAIL → implement → PASS.

- [ ] **Step 2: Chunk loop.** Implement `StreamingDiarizer`:
  - State: full `audio: Vec<f32>` buffer (a recording is minutes; keep it simple), `next_chunk: usize` (chunk index), `state: StreamingState`, `finished: bool`.
  - Total mel frames available for `n` samples: same formula `extract` uses (inspect audio.rs:62-91). Chunk `k` covers mel `[k*48, k*48+48)`; process it when mel for `k*48 + 48 + 56` frames exists (right context complete). Left offset: `min(8, k*48)` mel frames (0 for chunk 0, else 8); right offset: `min(56, total_mel − end)`.
  - Per chunk: `extract_frames(audio, start − left, left + 48 + right)` → `pre_encode` → embeddings rows (lc = left/8 rounded as NeMo does: `lc = round(left_offset/8)`, `rc = ceil(right_offset/8)` — models file :802–811) → assemble `[spkcache | fifo | chunk_embs]` into one `Tensor3` → `forward_embedded` → `encoder_proj` → 18 transformer layers → sigmoid head (reuse `Diarizer::forward_hidden`'s tail; factor a shared private fn on `Diarizer` rather than duplicating) → split preds rows: spkcache/fifo regions discarded after `streaming_update` consumes them; return the chunk-region preds (`[f32; 4]` conversion).
  - `finish()`: process remaining mel frames as a final shorter chunk (rc = whatever remains, possibly 0), then mark finished.
  - `reset()`: fresh `StreamingState`, clear buffers.
- [ ] **Step 3: Parity test, failing first.** `tests/streaming_parity.rs`:

```rust
#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn streaming_chunks_match_nemo_forward_streaming() {
    let reference = load_streaming_reference();
    let mut diarizer = StreamingDiarizer::from_artifact_dir(artifact_dir()).unwrap();
    let audio = conversation_audio();
    let mut ours: Vec<[f32; 4]> = Vec::new();
    for piece in audio.chunks(1600) {              // 100 ms pushes
        ours.extend(diarizer.push_samples(piece).unwrap());
    }
    ours.extend(diarizer.finish().unwrap());
    let reference_preds: Vec<[f32; 4]> = reference.flat_chunk_preds();
    assert_eq!(ours.len(), reference_preds.len(), "frame count mismatch");
    assert_probability_gates(&ours, &reference_preds, 0.08, 0.02); // max-abs, mean-abs
}
```

Gates: INT8 streaming accumulates state divergence, so allow max-abs 0.08 / mean 0.02 (looser than offline 0.05/0.01 but same order). If observed error is far above this, treat it as a bug, not a gate problem. Also assert the reference's `length_trajectory` matches our fifo/spkcache lengths at 3 probe chunks (first, first-pop, first-compress) — expose `#[cfg(test)]`-visible lengths or a `state_lengths()` accessor.

- [ ] **Step 4: Run** with artifact — green. Full crate suite green.
- [ ] **Step 5: Commit** `feat: add streaming AOSC diarizer`.

---

### Task 8: Token timestamps (`TimedToken`)

**Files:**
- Modify: `crates/nemotron-mlx/src/model/rnnt.rs:353-378`, `src/model/stream.rs` (StreamingTranscriber), `src/model/mod.rs` (export), `crates/nemotron-cli/src/main.rs:83` (transcribe call site), `crates/catcher-ffi/src/lib.rs` (update_text token handling)
- Test: `crates/nemotron-mlx/tests/rnnt_decode.rs` (extend), `tests/real_checkpoint.rs` (extend)

**Interfaces:**
- Produces (fusion Task 10 and FFI Task 11 rely on):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct TimedToken { pub id: u32, pub frame: u64 }   // frame × 80 ms = start time

// changed signatures (no shim kept):
impl StreamingRnntDecoder { pub fn decode_frames(&self, encoded: &Tensor3, state: &mut PredictionState) -> ModelResult<Vec<TimedToken>> } // frame = LOCAL index
impl StreamingTranscriber {
    pub fn push_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<TimedToken>>;  // frame = GLOBAL
    pub fn finish(&mut self) -> ModelResult<Vec<TimedToken>>;
    pub fn transcribe_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<TimedToken>>;
}
```

- [ ] **Step 1: Failing unit test.** In `tests/rnnt_decode.rs`, extend the existing greedy-control test (`:10`, synthetic joint scores) to assert frames: a token emitted while consuming encoder frame `i` carries `frame == i`, multiple tokens on one frame share the frame, and tokens after a blank advance. Follow the file's existing synthetic-joint construction.
- [ ] **Step 2: Run** — FAIL (type mismatch). **Implement:** in `decode_frames`, `for (frame, chunk) in encoded.values.chunks_exact(1024).enumerate()` and push `TimedToken { id, frame: frame as u64 }`. In `StreamingTranscriber`, add `frames_seen: u64`; in `decode_features`, offset each returned local frame by `frames_seen`, then `frames_seen += encoded_frames` per decoded window; `reset()` zeroes it. Update all call sites: CLI (`main.rs:83` — `token_ids` becomes `tokens`, decode with `tokens.iter().map(|t| t.id).collect::<Vec<_>>()`, keep the JSON field `token_ids` emitting ids), FFI (`update_text` extends `handle.tokens: Vec<TimedToken>`; text decode maps ids), any tests.
- [ ] **Step 3: Chunked ≡ offline equivalence test.** In `tests/real_checkpoint.rs` add (model-gated like its neighbors):

```rust
#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and the downloaded checkpoint"]
fn chunked_and_offline_decodes_agree_on_ids_and_frames() {
    // build two transcribers from the artifact (reuse the file's loader pattern)
    // offline: transcribe_samples(&audio)
    // chunked: push_samples in 1600-sample pieces + finish()
    // assert_eq!(offline, chunked)   // TimedToken derives PartialEq
}
```

Write the real body following the file's existing artifact/wav helpers.

- [ ] **Step 4: Run everything:** `cargo test -p nemotron-mlx -p nemotron-cli -p catcher-ffi`, plus `--ignored` suites with `NEMOTRON_MLX_ARTIFACT` set (the FFI byte-exact transcript test must still pass — text output unchanged). 
- [ ] **Step 5: Commit** `feat: emit frame-indexed tokens from streaming ASR`.

---

### Task 9: OpenCC s2twp module

**Files:**
- Modify: `crates/nemotron-mlx/Cargo.toml` (add `ferrous-opencc = "0.4"`), `src/lib.rs` (`pub mod opencc;`)
- Create: `crates/nemotron-mlx/src/opencc.rs`
- Test: `crates/nemotron-mlx/tests/opencc.rs`

**Interfaces:**
- Produces: `pub fn to_traditional(text: &str) -> String` — s2twp conversion, `OnceLock`-cached converter, panics never (a converter construction failure is a programming error caught by tests since dictionaries are embedded).

- [ ] **Step 1: Failing tests** (`tests/opencc.rs`):

```rust
use nemotron_mlx::opencc::to_traditional;

#[test]
fn converts_simplified_to_taiwan_traditional_with_phrases() {
    assert_eq!(to_traditional("软件"), "軟體");
    assert_eq!(to_traditional("信息"), "資訊");
    assert_eq!(to_traditional("里面"), "裡面");
    assert_eq!(to_traditional("鼠标"), "滑鼠");
    assert_eq!(to_traditional("这是一个测试"), "這是一個測試");
}

#[test]
fn is_idempotent_on_traditional_and_passes_through_ascii() {
    assert_eq!(to_traditional("軟體與資訊"), "軟體與資訊");
    assert_eq!(to_traditional("hello, world 123"), "hello, world 123");
    assert_eq!(to_traditional(""), "");
}
```

- [ ] **Step 2: Run** — FAIL. **Implement** `opencc.rs`: `static CONVERTER: OnceLock<ferrous_opencc::OpenCC> = OnceLock::new();` initialized with the built-in **s2twp** config (check docs.rs/ferrous-opencc 0.4 for the exact constructor — likely `OpenCC::from_config_name("s2twp")` or a `BuiltinConfig::S2twp` enum); `to_traditional` delegates. If a fixture expectation disagrees with the library's actual s2twp output, verify against the OpenCC reference tables before changing the test.
- [ ] **Step 3: Run** — PASS. Check binary-size/compile impact is sane (`cargo build -p nemotron-mlx` still works without new system deps).
- [ ] **Step 4: Commit** `feat: add s2twp Traditional Chinese conversion`.

---

### Task 10: Fusion state machine

**Files:**
- Create: `crates/nemotron-mlx/src/fusion.rs` (`pub mod fusion;` in lib.rs)
- Test: `crates/nemotron-mlx/tests/fusion.rs`

**Interfaces:**
- Consumes: `TimedToken` (Task 8). Nothing from `sortformer-mlx`.
- Produces (FFI Task 11 and CLI Task 12 rely on):

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SpeakerSegment {
    pub speaker: u8, pub start_ms: u64, pub end_ms: u64,
    pub text: String,
    #[serde(rename = "final")] pub is_final: bool,
}

pub struct FusionConfig {
    pub threshold: f32,        // 0.5
    pub smooth_frames: usize,  // 3  (±window)
    pub min_turn_ms: u64,      // 500
    pub frame_ms: u64,         // 80
}
impl Default for FusionConfig { /* the values above */ }

pub struct Fusion { /* cfg, tokens: Vec<TimedToken>, diar: Vec<[f32;4]>, flushed: bool */ }
impl Fusion {
    pub fn new(config: FusionConfig) -> Self;
    pub fn push_tokens(&mut self, tokens: &[TimedToken]);
    pub fn push_diar_frames(&mut self, frames: &[[f32; 4]]);
    pub fn segments(&self, detokenize: impl Fn(&[u32]) -> String) -> Vec<SpeakerSegment>;
    pub fn flush(&mut self);
    pub fn reset(&mut self);
}
```

**Semantics (from the spec — implement exactly):**
1. A token at frame `f` is *decidable* when `f + smooth_frames < diar.len()` or `flushed`. Undecidable tokens form at most one tentative tail segment.
2. Attribution of a decidable token: average each speaker's probability over frames `[f − smooth_frames, f + smooth_frames]` clipped to available diar frames; take the argmax; if its average < `threshold` → inherit the previous token's speaker (first token: speaker 0). Ties break toward the lower speaker index.
3. Consecutive same-speaker decidable tokens group into a segment: `start_ms = first.frame * frame_ms`, `end_ms = (last.frame + 1) * frame_ms`, `is_final = true`.
4. Anti-flicker on the finalized list: any segment shorter than `min_turn_ms` whose neighbors exist merges into the *previous* segment (or the next, if it is the first); re-run until stable (single pass over a Vec is fine — merge left, then rescan).
5. The tentative tail: all undecidable tokens as ONE segment attributed to the last finalized speaker (or 0), `is_final = false`, omitted when empty.
6. `segments()` recomputes from the raw streams each call (recordings are minutes-scale; O(n) per call is fine and keeps the state trivial). `detokenize` receives each segment's token ids; the caller composes tokenizer + `opencc::to_traditional`.
7. `flush()` makes everything decidable with whatever diar frames exist; `reset()` clears all.

- [ ] **Step 1: Failing tests** (`tests/fusion.rs`) — write real bodies for ALL of these with a small helper DSL:

```rust
fn tok(id: u32, frame: u64) -> TimedToken { TimedToken { id, frame } }
fn spk(s: usize) -> [f32; 4] { let mut p = [0.05; 4]; p[s] = 0.9; p }
fn sil() -> [f32; 4] { [0.05; 4] }
fn ids(text_stub: &str) -> impl Fn(&[u32]) -> String + '_ {
    move |ids| format!("{text_stub}:{}", ids.len())
}

#[test] fn attributes_tokens_to_dominant_smoothed_speaker() { /* 20 frames spk0 then
    20 frames spk1; tokens at frames 5 and 25 → two final segments, speakers 0 and 1 */ }
#[test] fn silence_inherits_previous_speaker() { /* spk0 frames, silence frames, token in
    silence region → still speaker 0, same segment */ }
#[test] fn first_token_in_silence_defaults_to_speaker_zero() { }
#[test] fn short_turns_merge_into_previous_neighbor() { /* spk0 long, spk1 for 400ms
    (5 frames, tokens), spk0 long → one segment, speaker 0, text includes middle ids */ }
#[test] fn tokens_beyond_diar_horizon_form_tentative_tail() { /* diar 10 frames, tokens at
    frames 5 and 30 → one final (spk of 5) + one tentative tail holding frame-30 token */ }
#[test] fn flush_finalizes_tail_and_is_terminal() { /* after flush(), no tentative
    segments; is_final everywhere; segment list stable across calls */ }
#[test] fn reset_clears_everything() { }
#[test] fn segment_times_derive_from_frames() { /* start=first*80, end=(last+1)*80 */ }
```

- [ ] **Step 2: Run** — FAIL (module missing). **Implement `fusion.rs`** per the semantics block. ~200 lines; keep attribution, grouping, anti-flicker, and tail construction as four private functions so each test failure localizes.
- [ ] **Step 3: Run** — all green; `cargo test -p nemotron-mlx` green.
- [ ] **Step 4: Commit** `feat: add ASR-diarization fusion state machine`.

---

### Task 11: C ABI v2

**Files:**
- Modify: `crates/catcher-ffi/include/catcher.h`, `crates/catcher-ffi/src/lib.rs`, `crates/catcher-ffi/Cargo.toml` (add `sortformer-mlx = { path = "../sortformer-mlx" }`, `serde_json = "1"`), `apps/tippi/Sources/TippiCore/CatcherClient.swift:34` (pass `nil`)
- Test: `crates/catcher-ffi/tests/ffi_lifecycle.rs` (extend)

**Interfaces:**
- Consumes: `StreamingDiarizer` (Task 7), `TimedToken` (Task 8), `opencc::to_traditional` (Task 9), `Fusion`/`SpeakerSegment` (Task 10).
- Produces the v2 header (Phase 3 Swift work builds on exactly this):

```c
catcher_handle_t *catcher_create(const char *asr_model_path,
                                 const char *diar_model_path,  /* NULL = ASR only */
                                 const char *language,
                                 uint32_t lookahead);
/* catcher_start / catcher_push_audio / catcher_finish / catcher_text
   / catcher_last_error / catcher_destroy: signatures unchanged */
const char *catcher_segments(const catcher_handle_t *handle); /* UTF-8 JSON array, borrowed */
const char *catcher_warning(const catcher_handle_t *handle);  /* NULL = no warning, borrowed */
```

**Behavior (from spec):**
- `CatcherHandle` gains: `diarizer: Option<StreamingDiarizer>`, `fusion: Fusion`, `timed_tokens: Vec<TimedToken>` (replaces `tokens: Vec<u32>`), `segments_json: CString`, `warning: Option<CString>`.
- `catcher_create` with non-NULL `diar_model_path`: `StreamingDiarizer::from_artifact_dir` failure → create fails (NULL return, thread-local error). NULL path → `diarizer = None`.
- `catcher_push_audio`: push to transcriber → `fusion.push_tokens`; if diarizer present, push the same samples → `fusion.push_diar_frames`. A diarizer **runtime** error: set `warning`, drop the diarizer to `None`, continue (transcription unaffected; fusion simply stops receiving diar frames, so subsequent tokens ride the tentative tail until finish). Rebuild `text` (now `opencc::to_traditional(decoded)`) and `segments_json` after every state change.
- `catcher_finish`: transcriber finish → fusion pushes, diarizer finish → fusion pushes, then `fusion.flush()`; rebuild strings.
- `catcher_segments`: with `diarizer == None` **and never-diar** (NULL path) → `[]` constant. Detokenizer closure: `|ids| opencc::to_traditional(&tokenizer.decode(ids, true).unwrap_or_default())`.
- `catcher_text` returns the s2twp-converted full transcript (existing byte-exact FFI test asserts English audio — conversion is a no-op there, test unchanged).
- String lifetime conventions identical to `catcher_text` (document in header comments).

- [ ] **Step 1: Failing tests.** Extend `ffi_lifecycle.rs`:

```rust
#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT"]
fn ascii_transcription_with_null_diar_keeps_v1_behavior() { /* existing byte-exact test
    body + assert catcher_segments == "[]" and catcher_warning == NULL */ }

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn dual_model_create_produces_decodable_segments_json() {
    /* create(asr, diar, "auto", 3); start; push conversation.wav in pieces; finish;
       parse catcher_segments with serde_json; assert: non-empty, every element has
       speaker/start_ms/end_ms/text/final fields, all final==true after finish,
       segments sorted by start_ms */
}

#[test]
fn null_handle_segments_and_warning_are_safe() { /* catcher_segments(NULL) -> NULL? No:
    follow catcher_text's NULL-handle convention exactly (see :171) */ }
```

Write real bodies mirroring the file's existing CString/unsafe call style; update the existing tests for the new `catcher_create` arity (pass `std::ptr::null()`).

- [ ] **Step 2: Run** — FAIL (arity). **Implement** lib.rs per the behavior block; update `catcher.h` with documented signatures; regenerate nothing (header is hand-maintained).
- [ ] **Step 3: Swift fix.** In `CatcherClient.swift:34`, add the `nil` argument for `diar_model_path` (keep `CatcherServing` unchanged — Phase 3 owns the API growth). Verify: `swift build --package-path apps/tippi` (or the repo's established build command — check `apps/tippi/README` / project scripts; whatever Phase 1 used to build Tippi).
- [ ] **Step 4: Run** `cargo test -p catcher-ffi` green; gated tests with both artifacts green.
- [ ] **Step 5: Commit** `feat: extend C ABI with diarization segments (v2)`.

---

### Task 12: CLI who-said-what + end-to-end

**Files:**
- Modify: `crates/nemotron-cli/src/main.rs` (Transcribe subcommand), `README.md`
- Test: `crates/nemotron-cli/tests/transcribe_diar.rs` (new)

**Interfaces:**
- Consumes: everything above. `Transcribe` gains `#[arg(long)] diar_model: Option<PathBuf>`.

**Output contract:** with `--diar-model`, one line per segment `[mm:ss] 說話者N:text` (N = speaker+1, `format_timestamp`-style mm:ss without millis for display; keep the existing `format_timestamp` for `diarize` untouched — add `format_timestamp_short`). `--json` → the `SpeakerSegment` array via serde (same JSON as `catcher_segments`). Without `--diar-model`: today's output, but text passes through `opencc::to_traditional`.

- [ ] **Step 1: Implement the pipeline** in `run()`'s Transcribe arm: when `diar_model` is set, drive `StreamingTranscriber` + `StreamingDiarizer` + `Fusion` in a single pass over the WAV (push 1600-sample pieces to both — this exercises the real streaming path, unlike `transcribe_samples`), finish + flush, then print. Detokenizer closure = tokenizer + `to_traditional`, as in the FFI.
- [ ] **Step 2: End-to-end test** (`tests/transcribe_diar.rs`, gated on both artifact env vars): run the pipeline in-process (call the crates directly, not the binary) on `tests/fixtures/conversation.wav`; assert ≥ 2 distinct speakers appear, segments alternate at least once, all `is_final`, text non-empty, and — if the fixture is Chinese — the concatenated output contains none of the simplified-only probe characters `"们"`, `"说"`, `"这"` (all common in zh output and all changed by s2twp; skip this assertion for an English fixture). Compare segment boundaries loosely against `tests/fixtures/conversation.json` turns: each constructed turn ≥ 2 s must overlap a segment of the matching relative speaker order (speaker identity is arrival-order, so map first-heard → first-labeled).
- [ ] **Step 3: RTF acceptance (manual).** Run and record in the task report:

```bash
time target/release/catcher transcribe --model <asr> --diar-model <diar> \
  --audio tests/fixtures/conversation.wav
```

`cargo build --release` first. Wall time / audio duration must be < 1.0. Paste the numbers.

- [ ] **Step 4: README.** Extend the diarization section: `--diar-model` usage, sample output block, note that all Chinese output is Taiwan-standard Traditional (s2twp), and the 1.04 s diarization latency figure.
- [ ] **Step 5: Run the workspace suite** `cargo test --workspace` green; gated suites green with artifacts.
- [ ] **Step 6: Commit** `feat: speaker-attributed Traditional Chinese transcription in CLI`.

---

## Self-review notes (already applied)

- Spec coverage: Block 1 → Tasks 1–2; Block 2 → Tasks 3–7; Block 3 → Task 8; Block 4 → Task 10; Block 5 → Task 9; Block 6 → Tasks 11–12; fixture → Task 3; RTF criterion → Task 2 (offline probe) + Task 12 (manual acceptance).
- Type consistency: `TimedToken { id, frame }` (Tasks 8/10/11/12); `SpeakerSegment` with serde `final` rename (Tasks 10/11/12); `StreamingConfig`/`streaming_update` (Tasks 6/7); `pre_encode`/`forward_embedded` (Tasks 5/7); `to_traditional` (Tasks 9/11/12).
- Known intentional deviations from "complete code": AOSC helper internals (Task 6) and the encoder rel-pos matmul (Task 2) are specified by NeMo anchors + existing scalar code rather than verbatim plan code — the ground-truth-first constraint makes the fixtures, not this plan, the spec. mlx-rs method names are per-docs, math is fixed.
