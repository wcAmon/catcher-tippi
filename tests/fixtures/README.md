# Audio fixtures

## Mandarin submit-command fixtures

`bang-wo-song-chu-zh-cn.wav` and `bang-wo-song-chu-zh-tw.wav` are deterministic
positive fixtures for 「幫我送出」, spoken by the macOS Tingting (`zh_CN`) and
Meijia (`zh_TW`) voices. `bang-wo-zh-tw.wav` and `song-chu-zh-tw.wav` are
partial-command negatives. All four files are 16 kHz mono PCM16.

Reconstruction commands:

```sh
/usr/bin/say -v Tingting -r 170 -o /tmp/bang-wo-song-chu-zh-cn.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-song-chu-zh-tw.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-zh-tw.aiff "幫我"
/usr/bin/say -v Meijia -r 170 -o /tmp/song-chu-zh-tw.aiff "送出"
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-cn.aiff tests/fixtures/bang-wo-song-chu-zh-cn.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-tw.aiff tests/fixtures/bang-wo-song-chu-zh-tw.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-zh-tw.aiff tests/fixtures/bang-wo-zh-tw.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/song-chu-zh-tw.aiff tests/fixtures/song-chu-zh-tw.wav
```

## tippi-go.wav

Deterministic negative regression for the removed English command. It remains
checked in to prove that `Tippi Go` no longer submits.

```sh
/usr/bin/say -v Samantha -r 170 -o /tmp/tippi-go.aiff "Tippy go"
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/tippi-go.aiff tests/fixtures/tippi-go.wav
```

## conversation.wav / conversation.json

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

### Ground-truth transcripts

Official transcripts from AISHELL-3 `train/content.txt` (pinyin stripped),
in fixture turn order:

| Turn | Utterance ID | Transcript (simplified) |
| ---- | ------------ | ----------------------- |
| 0  | `SSB00090001` | 伺候老婆是老公的责任 |
| 1  | `SSB00110001` | 领导干部廉洁从政自查 |
| 2  | `SSB00090002` | 阿尔泰的生物有什么 |
| 3  | `SSB00110002` | 河宕村民委员会计划生育服务室 |
| 4  | `SSB00090003` | 文成县的学校有什么 |
| 5  | `SSB00110003` | 基层医院的医生缺乏不断学习和提高水平的动力 |
| 6  | `SSB00090005` | 但有人说我非常耀眼 |
| 7  | `SSB00110005` | 下辈子不做女人 |
| 8  | `SSB00090006` | 微软推出免费增值策略 |
| 9  | `SSB00110007` | 三百六十五 |
| 10 | `SSB00090007` | 山海经地名有什么 |
| 11 | `SSB00110009` | 当你孤单你会想起谁 |
| 12 | `SSB00090009` | 我们称她为母夜叉 |
| 13 | `SSB00110010` | 搜狐娱乐讯据香港媒体报道 |

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
