import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError("reference WAV must be mono 16 kHz signed PCM16")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: generate_transcription_reference.py MODEL_DIR WAV OUTPUT_JSON")
    model_dir, wav_path, output_path = map(Path, sys.argv[1:])
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.set_num_lookahead_tokens(3)
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    audio = read_wav(wav_path)
    first = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=16000,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="en-US",
        return_tensors="pt",
    )
    chunks = [first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]]
    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    while True:
        start = mel_frame_idx * processor.feature_extractor.hop_length - processor.feature_extractor.n_fft // 2
        if start >= len(audio):
            break
        end = start + processor.num_samples_per_audio_chunk
        chunk_audio = audio[start:end]
        if len(chunk_audio) < processor.num_samples_per_audio_chunk:
            chunk_audio = np.pad(chunk_audio, (0, processor.num_samples_per_audio_chunk - len(chunk_audio)))
        inputs = processor(
            chunk_audio,
            sampling_rate=16000,
            is_streaming=True,
            is_first_audio_chunk=False,
            language="en-US",
            return_tensors="pt",
        )
        chunks.append(inputs.input_features)
        mel_frame_idx += processor.num_mel_frames_per_audio_chunk

    with torch.no_grad():
        generate_kwargs = {
            **first,
            "input_features": (chunk for chunk in chunks),
            "return_dict_in_generate": True,
        }
        result = model.generate(**generate_kwargs)
    token_ids = result.sequences.flatten().tolist()
    output_path.write_text(
        json.dumps(
            {
                "lookahead": 3,
                "language": "en-US",
                "chunks": len(chunks),
                "token_ids": token_ids,
                "text": processor.decode(token_ids, skip_special_tokens=True),
            },
            separators=(",", ":"),
        )
    )
    print(output_path.read_text())


if __name__ == "__main__":
    main()
