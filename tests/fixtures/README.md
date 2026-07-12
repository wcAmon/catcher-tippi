# conversation.wav / conversation.json

Two-speaker conversation fixture used by streaming-diarization parity tests
(Task 4: NeMo reference run) and end-to-end who-said-what validation (Task
12). Not a real conversation — it is a deterministic concatenation of
single-speaker utterances from a public corpus, alternating between two
speakers with short silence gaps, so the resulting audio exercises real
speaker turns without recording anyone.

## Source

[AISHELL-3](https://huggingface.co/datasets/AISHELL/AISHELL-3), a Mandarin
multi-speaker TTS corpus published by Beijing Shell Shell Technology Co.,
Ltd., mirrored on the Hugging Face Hub.

- **License:** Apache-2.0, as declared in the dataset's own `README.md`
  frontmatter (`license: apache-2.0`) at
  https://huggingface.co/datasets/AISHELL/AISHELL-3
- **Speakers used:** `SSB0009` (speaker 0, first heard) and `SSB0011`
  (speaker 1), 7 utterances each, 14 turns total.
- **Original format:** 44.1 kHz mono PCM WAV per utterance.

### Exact utterance IDs (in the order they appear in the fixture)

| Turn | Speaker | Utterance path (within the AISHELL-3 HF repo) |
| ---- | ------- | ---------------------------------------------- |
| 0  | 0 | `train/wav/SSB0009/SSB00090001.wav` |
| 1  | 1 | `train/wav/SSB0011/SSB00110001.wav` |
| 2  | 0 | `train/wav/SSB0009/SSB00090002.wav` |
| 3  | 1 | `train/wav/SSB0011/SSB00110002.wav` |
| 4  | 0 | `train/wav/SSB0009/SSB00090003.wav` |
| 5  | 1 | `train/wav/SSB0011/SSB00110003.wav` |
| 6  | 0 | `train/wav/SSB0009/SSB00090005.wav` |
| 7  | 1 | `train/wav/SSB0011/SSB00110005.wav` |
| 8  | 0 | `train/wav/SSB0009/SSB00090006.wav` |
| 9  | 1 | `train/wav/SSB0011/SSB00110007.wav` |
| 10 | 0 | `train/wav/SSB0009/SSB00090007.wav` |
| 11 | 1 | `train/wav/SSB0011/SSB00110009.wav` |
| 12 | 0 | `train/wav/SSB0009/SSB00090009.wav` |
| 13 | 1 | `train/wav/SSB0011/SSB00110010.wav` |

## Construction

Built by `tools/make_conversation_fixture.py`, which:

1. Downloads the utterances above from `AISHELL/AISHELL-3` on the Hugging
   Face Hub into `/tmp/phase2-fixture-src/` (skipped if already cached
   there).
2. Resamples each 44.1 kHz utterance to 16 kHz mono (`scipy.signal.resample_poly`).
3. Concatenates the 14 turns in the fixed order above, inserting a silence
   gap after each turn except the last. Gap durations (seconds), in order:
   `0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.7`.
4. Peak-normalizes the full signal to 0.9.
5. Writes `conversation.wav` (16 kHz mono PCM16) and `conversation.json`
   (the turn map, computed from the actual concatenation above — not
   estimated).

The utterance list and gap durations are hardcoded in the script (no RNG),
so re-running it reproduces byte-identical output.

Reconstruction command:

```sh
/tmp/sortformer-venv/bin/python tools/make_conversation_fixture.py
```

(Any Python 3.11+ environment with `numpy`, `scipy`, `soundfile`, and
`huggingface_hub` installed works; the venv path above is just what this
project used.)

## Result

- 16 kHz mono PCM16 WAV, 47.47 s total (comfortably inside the 45-55 s
  target and well past the 35 s minimum needed to exercise at least one
  FIFO pop and one speaker-cache compression during streaming).
- 14 turns, strictly alternating speaker 0 / speaker 1, each separated by a
  0.3-0.7 s silence gap.
- Every turn's RMS is nonzero (spoken audio); every gap is exact digital
  silence (zeros) before normalization.

## conversation.json

```json
{
  "turns": [{"speaker": 0, "start_s": 0.0, "end_s": 2.386}, ...],
  "source": "AISHELL/AISHELL-3",
  "license": "Apache-2.0 (https://huggingface.co/datasets/AISHELL/AISHELL-3)"
}
```

`turns` gives the exact start/end time in seconds of each speaker turn as
laid out in `conversation.wav`, with speaker 0 defined as the first speaker
heard.
