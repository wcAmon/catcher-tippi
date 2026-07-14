# Tippi Chinese Submit Command Design

**Date:** 2026-07-14  
**Status:** Approved in conversation; pending written-spec review

## 1. Context

The first real cross-App voice-input acceptance run showed that the English
command is not reliable inside Mandarin speech. The main ASR rendered variants
such as `TPOP`, `PPOTIPPIGOP`, and similar fragments, while the independent KWS
did not detect `Tippi Go`. No Return was sent.

The KWS model itself does not need to change. Its `tokens.txt` contains English
phonemes and tone-marked Mandarin pinyin tokens, so the existing offline
sherpa-onnx open-vocabulary KWS runtime can recognize a Mandarin command by
changing the generated keyword definition.

## 2. Product Decision

The only submit command becomes **「幫我送出」**.

- `Tippi Go` is removed rather than retained as an alias.
- The command remains local and offline.
- The command remains independent of the main ASR transcript.
- The internal command identifier becomes `SUBMIT_ZH`.
- Stable text is held for 2,000 ms instead of 1,500 ms because the four-syllable
  Mandarin command is longer than the old English command.

## 3. Goals

- Detect a normally spoken 「幫我送出」 reliably in Mandarin speech.
- Send Return exactly once per accepted command.
- Keep the spoken command and the main ASR's approximations out of the target
  input field.
- Upgrade an already installed, valid KWS model without downloading the model
  archive again.
- Remove `Tippi Go` from runtime logic, UI copy, and user documentation. Retain
  it only as a negative regression fixture/test and in historical specs.
- Preserve all existing fail-closed target, permission, stop, duplicate-command,
  and append-only injection behavior.

## 4. Non-goals

- No second submit alias, configurable command picker, or custom wake-word UI.
- No third model and no fine-tuning or user-specific training.
- No main-ASR text matching, fuzzy matching, Backspace cleanup, or clipboard
  fallback.
- No use of sherpa timestamps for transcript cutoff.
- No automatic target focusing or Accessibility-value writes.

## 5. Keyword Definition

The Mandarin syllables map to tokens already present in the pinned model:

| Character | Pinyin | KWS tokens | Token IDs |
| --- | --- | --- | --- |
| 幫 | bāng | `b āng` | `77 224` |
| 我 | wǒ | `w ǒ` | `178 253` |
| 送 | sòng | `s òng` | `138 210` |
| 出 | chū | `ch ū` | `79 243` |

The generated `keywords.txt` contains exactly one line:

```text
b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH
```

The initial keyword boost and trigger threshold remain `1.5` and `0.25`. They
may change only if the positive and negative real-model matrix in section 10
demonstrates a regression; any change must be recorded in the implementation
commit and covered by the same matrix.

Swift exposes one command definition with these values:

- display phrase: `幫我送出`
- event identifier: `SUBMIT_ZH`
- pinyin token sequence and KWS parameters above

`KeywordModelManifest` uses this definition to generate `keywords.txt`, the
controller compares detections with the event identifier, and SwiftUI uses the
display phrase. Rust keeps the ABI-side expected identifier as `SUBMIT_ZH` and
rejects any other KWS result.

## 6. Audio and Injection Data Flow

The two-model flow is otherwise unchanged:

1. Each mono Float32 16 kHz chunk is delivered to Catcher ASR and sherpa KWS.
2. The controller advances `receivedSampleCount` by the exact chunk length.
3. Stable text uses:

   ```text
   audioEndMs = floor(receivedSampleCount * 1000 / 16000)
   stableCutoffMs = max(0, audioEndMs - 2000)
   ```

4. Without a command, only `catcher_text_before(stableCutoffMs)` is injected.
5. When KWS returns `SUBMIT_ZH`, Catcher finishes before the same sample-derived
   cutoff, the safe prefix is submitted, Return is sent once, and the ASR, KWS,
   injection coordinator, duplicate guard, and sample timeline reset.
6. An empty safe prefix sends no Return.

The KWS timestamp remains diagnostic-only. Increasing the holdback to 2 seconds
ensures that a normally paced four-syllable command and its detection latency
remain inside the withheld tail. The UI tells the user to finish the content,
pause briefly, and then say 「幫我送出」.

## 7. Existing-Installation Upgrade

The pinned ONNX files and `tokens.txt` do not change. The installer separates
verification into two layers:

1. **Runtime files:** the four pinned files (`encoder.onnx`, `decoder.onnx`,
   `joiner.onnx`, and `tokens.txt`) with their existing SHA-256 hashes.
2. **Generated files:** the current `keywords.txt` and
   `THIRD_PARTY_NOTICES.md` contents.

An installation is repairable only when its inventory consists exactly of
those four runtime files and those two generated files, and every runtime file
is valid. If either generated file is stale, the installer atomically rewrites
the generated files in place and verifies the complete installation again. It
does not invoke the downloader or extractor. An interruption between the two
atomic writes is safe because the next preparation repeats the repair.

If a runtime file is missing, corrupt, or the directory inventory is unsafe,
the installer uses the existing verified download, staging, and atomic
promotion path. The model directory name and archive hashes remain unchanged.

## 8. Error Behavior

- A generated-file repair failure sets Voice Input preparation to failed with
  a generic keyword-model error; the Transcription tab remains usable.
- A KWS load or streaming failure stops recording, resets the voice turn, and
  never injects or submits the held tail.
- A non-`SUBMIT_ZH` result is ignored as a command and cannot send Return.
- A command detected while Tippi is frontmost remains discarded and must be
  repeated after focusing the target.
- Stop never flushes held text into the target and never sends Return.

User-visible errors say 「口令模型」 or 「幫我送出」 and do not mention the
removed English command.

## 9. UI and Documentation

The Voice Input tab changes all command-facing copy:

- badge: `口令：幫我送出`
- header: `內容說完後短暫停頓，再說「幫我送出」。`
- ready state: `「幫我送出」口令模型已就緒`
- recording hint: `文字約延遲 2 秒；短暫停頓後說「幫我送出」。`
- target-retry message: `請切到目標輸入框後重說「幫我送出」`

README behavior, limitations, and acceptance instructions use the same phrase
and 2-second delay. Historical design documents are not rewritten; this spec
supersedes their English-command decisions.

## 10. Test and Acceptance Matrix

### Deterministic fixtures

Check in 16 kHz mono PCM16 positive fixtures for both Mandarin accents using
available macOS voices:

```sh
/usr/bin/say -v Tingting -r 170 -o /tmp/bang-wo-song-chu-zh-cn.aiff "幫我送出"
/usr/bin/say -v Meijia -r 170 -o /tmp/bang-wo-song-chu-zh-tw.aiff "幫我送出"
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-cn.aiff tests/fixtures/bang-wo-song-chu-zh-cn.wav
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/bang-wo-song-chu-zh-tw.aiff tests/fixtures/bang-wo-song-chu-zh-tw.wav
```

These reconstruction commands and voice names are recorded in
`tests/fixtures/README.md` beside the fixture descriptions.

### Automated coverage

- Both padded Mandarin positive fixtures detect `SUBMIT_ZH` and detect again
  after reset.
- Existing `tippi-go.wav` no longer triggers.
- Padded fixtures for 「送出」 alone and 「幫我」 alone do not trigger.
- Existing English and Mandarin conversation fixtures do not trigger.
- Installer repair tests prove that a valid old installation changes only the
  generated files without downloader, extractor, or model promotion calls.
- Missing or corrupt runtime files still exercise the full verified install
  path.
- Swift controller tests use `SUBMIT_ZH`, expect a 2,000 ms holdback, preserve
  command timestamp independence, and assert exactly one Return.
- Source checks ensure active runtime/UI/README files contain no `Tippi Go` or
  `TIPPI_GO` references.

### Real acceptance

With the rebuilt signed App and a focused TextEdit field:

- Speak ordinary content, pause about 0.5 seconds, then say 「幫我送出」 three
  times across three turns.
- All three commands must be detected.
- Each turn must press Return exactly once.
- Neither the command nor main-ASR approximations may appear in the field.
- The second and third turns must prove that the sample clock and duplicate
  guard reset.
- Saying only 「送出」, only 「幫我」, and the old `Tippi Go` must not submit.
- Stopping with held text, leaving Tippi frontmost, and focusing a non-text
  target must remain safe.

## 11. Expected Code Areas

- `apps/tippi/Sources/TippiCore/KeywordModelManifest.swift`
- `apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift`
- `apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift`
- `apps/tippi/Sources/TippiCore/VoiceInputTiming.swift`
- `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`
- `crates/catcher-ffi/src/kws.rs`
- Rust and Swift KWS, installer, controller, and UI tests
- `tests/fixtures/README.md` and new Mandarin command fixtures
- `README.md`

The Catcher ASR model, sherpa ONNX model files, C ABI shape, text injection
mechanism, and model archive stay unchanged.
