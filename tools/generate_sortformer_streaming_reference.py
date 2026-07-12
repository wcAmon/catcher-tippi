"""Capture NeMo Streaming Sortformer ground truth for Rust streaming parity tests.

Runs NeMo's official `forward_streaming` (synchronous / low-latency preset) over an
audio file and records, for every streaming chunk, the kept per-chunk predictions and
the post-update state lengths. Task 7 asserts per-chunk parity of the Rust streaming
diarizer against this fixture.

Usage:
    python tools/generate_sortformer_streaming_reference.py \
        --nemo /tmp/sortformer-src/diar_streaming_sortformer_4spk-v2.1.nemo \
        --audio tests/fixtures/conversation.wav \
        --output tests/fixtures/sortformer_streaming_reference.json

Conventions follow tools/generate_sortformer_reference.py (restore_from, CPU, fp32,
eval, forward hooks).
"""

import argparse
import json
import math
from pathlib import Path

import soundfile
import torch
from nemo.collections.asr.models import SortformerEncLabelModel

# Low-latency preset. These OVERRIDE the checkpoint's training defaults and must be set
# on the loaded model's sortformer_modules BEFORE forward_streaming runs.
PRESET = {
    "chunk_len": 6,
    "chunk_left_context": 1,
    "chunk_right_context": 7,
    "fifo_len": 188,
    "spkcache_len": 188,
    "spkcache_update_period": 188,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True, help="Path to the .nemo checkpoint")
    parser.add_argument("--audio", required=True, help="Path to a 16 kHz mono wav")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    model = SortformerEncLabelModel.restore_from(args.nemo, map_location="cpu").eval()

    # Apply the low-latency preset (overrides checkpoint training defaults) and switch to
    # synchronous streaming eval. streaming_mode=True makes process_signal skip waveform
    # peak-normalization (models file :501); async off is the eval-parity path.
    sm = model.sortformer_modules
    for key, value in PRESET.items():
        setattr(sm, key, value)
    model.streaming_mode = True
    model.async_streaming = False
    # Fails loudly (does not silently mutate) if the preset is illegal.
    sm._check_streaming_parameters()

    audio, sample_rate = soundfile.read(args.audio, dtype="float32")
    assert sample_rate == 16_000 and audio.ndim == 1, "expected 16 kHz mono audio"
    signal = torch.tensor(audio)[None, :]
    length = torch.tensor([signal.shape[1]])

    # Hook pre_encode to grab chunk 0's pre-encoded embeddings (first call only).
    chunk0_pre_encode = {}

    def pre_encode_hook(_module, _inputs, output):
        if "value" not in chunk0_pre_encode:
            embs = output[0] if isinstance(output, tuple) else output
            chunk0_pre_encode["value"] = embs.detach().clone()

    handle = model.encoder.pre_encode.register_forward_hook(pre_encode_hook)

    # Wrap forward_streaming_step to capture each step's kept chunk_preds (the diff added
    # to total_preds) and the post-update synchronous state lengths.
    records = []
    original_step = model.forward_streaming_step

    def wrapped_step(*step_args, **step_kwargs):
        prev_total = step_kwargs["total_preds"]
        prev_len = prev_total.shape[1]
        state, total = original_step(*step_args, **step_kwargs)
        chunk_preds = total[:, prev_len:, :]
        records.append(
            {
                "chunk_preds": chunk_preds.detach().clone(),
                "fifo": int(state.fifo.shape[1]),
                "spkcache": int(state.spkcache.shape[1]),
                "mean_sil_emb": state.mean_sil_emb.detach().clone(),
                "n_sil_frames": int(state.n_sil_frames[0].item()),
            }
        )
        return state, total

    model.forward_streaming_step = wrapped_step

    with torch.no_grad():
        processed_signal, processed_signal_length = model.process_signal(
            audio_signal=signal, audio_signal_length=length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        mel_frames = int(processed_signal.shape[2])
        total_preds = model.forward_streaming(processed_signal, processed_signal_length)

    handle.remove()
    model.forward_streaming_step = original_step

    # Sanity: concatenated captured chunk_preds must equal the returned total_preds.
    captured = torch.cat([r["chunk_preds"] for r in records], dim=1)
    assert captured.shape == total_preds.shape, (
        f"captured shape {tuple(captured.shape)} != total_preds shape "
        f"{tuple(total_preds.shape)}"
    )
    assert torch.equal(captured, total_preds), (
        "captured chunk_preds do not concatenate to total_preds; capture is wrong"
    )

    expected_num_chunks = math.ceil(
        mel_frames / (sm.chunk_len * sm.subsampling_factor)
    )
    num_chunks = len(records)
    assert num_chunks == expected_num_chunks, (
        f"num_chunks {num_chunks} != ceil(mel_frames/48) {expected_num_chunks}"
    )

    final = records[-1]
    payload = {
        "preset": PRESET,
        "audio": Path(args.audio).name,
        "sample_rate": sample_rate,
        "mel_frames": mel_frames,
        "num_chunks": num_chunks,
        "chunk_preds": [
            r["chunk_preds"].squeeze(0).tolist() for r in records
        ],
        "chunk0_pre_encode": {
            "frames": int(chunk0_pre_encode["value"].shape[1]),
            "dim": int(chunk0_pre_encode["value"].shape[2]),
            "values": chunk0_pre_encode["value"].squeeze(0).flatten().tolist(),
        },
        "final_state": {
            "mean_sil_emb": final["mean_sil_emb"].squeeze(0).tolist(),
            "n_sil_frames": final["n_sil_frames"],
            "fifo_len": final["fifo"],
            "spkcache_len": final["spkcache"],
        },
        "length_trajectory": [
            {"chunk": i, "fifo": r["fifo"], "spkcache": r["spkcache"]}
            for i, r in enumerate(records)
        ],
    }

    Path(args.output).write_text(json.dumps(payload, separators=(",", ":")))
    size = Path(args.output).stat().st_size
    print(
        f"wrote {args.output}: num_chunks={num_chunks}, mel_frames={mel_frames}, "
        f"total_preds={tuple(total_preds.shape)}, "
        f"chunk0_pre_encode={payload['chunk0_pre_encode']['frames']}x"
        f"{payload['chunk0_pre_encode']['dim']}, size={size} bytes"
    )
    print("sanity assert passed: concatenated chunk_preds == total_preds")


if __name__ == "__main__":
    main()
