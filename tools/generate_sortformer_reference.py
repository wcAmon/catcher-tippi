"""Capture NeMo Sortformer intermediate activations for Rust parity tests.

Usage: python tools/generate_sortformer_reference.py MODEL.nemo WAV OUTPUT_JSON
Requires: pip install "nemo_toolkit[asr]" soundfile
Run with --dump-structure first to print the module tree and confirm the
attribute names used below match this NeMo version; adjust if they differ.
"""

import json
import sys
from pathlib import Path

import soundfile
import torch
from nemo.collections.asr.models import SortformerEncLabelModel


def summary(tensor):
    flat = tensor.detach().float().flatten()
    return {
        "shape": list(tensor.shape),
        "mean": flat.mean().item(),
        "rms": flat.square().mean().sqrt().item(),
        "first": flat[:64].tolist(),
    }


def main() -> None:
    dump_structure = "--dump-structure" in sys.argv
    arguments = [value for value in sys.argv[1:] if value != "--dump-structure"]
    if len(arguments) != 3 and not (dump_structure and len(arguments) >= 1):
        raise SystemExit(
            "usage: generate_sortformer_reference.py MODEL.nemo WAV OUTPUT_JSON"
        )
    model = SortformerEncLabelModel.restore_from(arguments[0], map_location="cpu").eval()
    if dump_structure:
        print(model)
        return
    wav_path, output_path = Path(arguments[1]), Path(arguments[2])

    audio, sample_rate = soundfile.read(wav_path, dtype="float32")
    assert sample_rate == 16_000 and audio.ndim == 1
    signal = torch.tensor(audio)[None, :]
    length = torch.tensor([signal.shape[1]])

    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value

        return hook

    handles = [model.encoder.pre_encode.register_forward_hook(capture("pre_encode"))]
    handles += [
        layer.register_forward_hook(capture(f"encoder_layer_{index}"))
        for index, layer in enumerate(model.encoder.layers)
    ]
    handles += [
        layer.register_forward_hook(capture(f"transformer_layer_{index}"))
        for index, layer in enumerate(model.transformer_encoder.layers)
    ]

    with torch.no_grad():
        features, feature_length = model.preprocessor(
            input_signal=signal, length=length
        )
        captured["features"] = features  # [1, n_mels, frames]
        encoded, encoded_length = model.encoder(
            audio_signal=features, length=feature_length
        )
        captured["encoder_out"] = encoded  # [1, d_model, frames/8]
        embeddings = encoded.transpose(1, 2)  # [1, frames/8, d_model]
        projected = model.sortformer_modules.encoder_proj(embeddings)
        captured["projected"] = projected
        transformed = model.transformer_encoder(
            encoder_states=projected,
            encoder_mask=torch.ones(projected.shape[:2]),
        )
        probabilities = model.sortformer_modules.forward_speaker_sigmoids(transformed)
        captured["probabilities"] = probabilities  # [1, frames/8, 4]

    for handle in handles:
        handle.remove()

    payload = {name: summary(tensor) for name, tensor in captured.items()}
    payload["probabilities_full"] = probabilities.flatten().tolist()
    payload["wav"] = wav_path.name
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {output_path} with {probabilities.shape[1]} frames")


if __name__ == "__main__":
    main()
