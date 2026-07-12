"""Build the two-speaker conversation fixture from AISHELL-3 utterances.

Usage: python tools/make_conversation_fixture.py
Requires: pip install soundfile numpy scipy huggingface_hub

Source utterances are downloaded from the AISHELL/AISHELL-3 dataset on the
Hugging Face Hub (Apache-2.0) into /tmp/phase2-fixture-src/ if not already
present there, then resampled 44.1kHz->16kHz, concatenated with silence gaps
alternating between two speakers, peak-normalized, and written to
tests/fixtures/conversation.wav plus the matching turn map
tests/fixtures/conversation.json. See tests/fixtures/README.md for the exact
utterance IDs and license text.

Deterministic: the utterance list and gap durations below are hardcoded
(no RNG), so re-running reproduces byte-identical output.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path("/tmp/phase2-fixture-src")
OUTPUT_WAV = REPO_ROOT / "tests/fixtures/conversation.wav"
OUTPUT_JSON = REPO_ROOT / "tests/fixtures/conversation.json"

DATASET_REPO = "AISHELL/AISHELL-3"
DATASET_LICENSE = "Apache-2.0"
DATASET_URL = "https://huggingface.co/datasets/AISHELL/AISHELL-3"

TARGET_SAMPLE_RATE = 16_000
SOURCE_SAMPLE_RATE = 44_100
PEAK_TARGET = 0.9

# Speaker 0 = SSB0009 (first speaker heard), speaker 1 = SSB0011.
# Turns alternate strictly A, B, A, B, ... Utterance IDs and order are fixed
# so the fixture is reproducible; see tests/fixtures/README.md for how these
# were selected.
UTTERANCES = [
    (0, "train/wav/SSB0009/SSB00090001.wav"),
    (1, "train/wav/SSB0011/SSB00110001.wav"),
    (0, "train/wav/SSB0009/SSB00090002.wav"),
    (1, "train/wav/SSB0011/SSB00110002.wav"),
    (0, "train/wav/SSB0009/SSB00090003.wav"),
    (1, "train/wav/SSB0011/SSB00110003.wav"),
    (0, "train/wav/SSB0009/SSB00090005.wav"),
    (1, "train/wav/SSB0011/SSB00110005.wav"),
    (0, "train/wav/SSB0009/SSB00090006.wav"),
    (1, "train/wav/SSB0011/SSB00110007.wav"),
    (0, "train/wav/SSB0009/SSB00090007.wav"),
    (1, "train/wav/SSB0011/SSB00110009.wav"),
    (0, "train/wav/SSB0009/SSB00090009.wav"),
    (1, "train/wav/SSB0011/SSB00110010.wav"),
]

# Silence between consecutive turns, seconds. len(GAPS_S) == len(UTTERANCES) - 1.
GAPS_S = [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.7]


def load_utterance(relative_path: str) -> np.ndarray:
    local_path = SOURCE_DIR / relative_path
    if not local_path.exists():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=relative_path,
            local_dir=str(SOURCE_DIR),
        )
        local_path = Path(downloaded)
    audio, sample_rate = sf.read(local_path, dtype="float32")
    assert audio.ndim == 1, f"{relative_path} is not mono"
    assert sample_rate == SOURCE_SAMPLE_RATE, (
        f"{relative_path} unexpected sample rate {sample_rate}"
    )
    resampled = resample_poly(audio, TARGET_SAMPLE_RATE, sample_rate)
    return resampled.astype(np.float32)


def main() -> None:
    assert len(GAPS_S) == len(UTTERANCES) - 1

    segments: list[np.ndarray] = []
    turns: list[dict] = []
    cursor_s = 0.0
    for index, (speaker, relative_path) in enumerate(UTTERANCES):
        audio = load_utterance(relative_path)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        assert rms > 1e-4, f"{relative_path} decoded to near silence (rms={rms})"
        duration_s = len(audio) / TARGET_SAMPLE_RATE
        turns.append(
            {
                "speaker": speaker,
                "start_s": round(cursor_s, 3),
                "end_s": round(cursor_s + duration_s, 3),
            }
        )
        segments.append(audio)
        cursor_s += duration_s
        print(
            f"turn {index:2d}: speaker {speaker}  {relative_path}  "
            f"dur={duration_s:.3f}s  rms={rms:.4f}"
        )
        if index < len(GAPS_S):
            gap_s = GAPS_S[index]
            gap_samples = int(round(gap_s * TARGET_SAMPLE_RATE))
            gap = np.zeros(gap_samples, dtype=np.float32)
            segments.append(gap)
            cursor_s += gap_s
            print(f"  gap after turn {index}: {gap_s:.2f}s  rms={0.0:.4f}")

    full = np.concatenate(segments)
    peak = float(np.max(np.abs(full)))
    full = (full * (PEAK_TARGET / peak)).astype(np.float32)

    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT_WAV, full, TARGET_SAMPLE_RATE, subtype="PCM_16")

    payload = {
        "turns": turns,
        "source": DATASET_REPO,
        "license": f"{DATASET_LICENSE} ({DATASET_URL})",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    total_duration_s = len(full) / TARGET_SAMPLE_RATE
    print(
        f"wrote {OUTPUT_WAV} ({total_duration_s:.2f}s, peak={np.max(np.abs(full)):.3f}) "
        f"and {OUTPUT_JSON} ({len(turns)} turns)"
    )


if __name__ == "__main__":
    main()
