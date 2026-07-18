/**
 * 驗收測試:對已安裝的真 engine host(真模型)餵 fixture wav,驗證
 * asr-host-v1 協定端到端的辨識品質——直接呼叫 `EngineClient`(不經過
 * `server.ts`/WS),隔離「協定與模型是否正確」跟「HTTP/WS 服務層是否正確」
 * 兩件事(後者是 `service_test.ts` 的職責)。
 *
 * fixture `verify/fixtures/hello-streaming.wav`、參考文字、門檻
 * (normalized edit distance ≤ 0.25)、chunk 大小(1600 samples/chunk)
 * 逐字對齊 `crates/catcher-asr-host/tests/real_model.rs` 的
 * `transcribes_fixture_wav_end_to_end`——這是同一份驗收邏輯的第三個實作
 * (見 `asr_metric.ts` 檔頭說明),`asr_metric_test.ts` 已經把 normalize/
 * distance 的演算法本身釘住,這裡只需要專心驗證「真的餵給真 host 會得到
 * 夠接近的結果」。
 *
 * 權限:`--allow-run=bin/engine-host`(spawn 真 host)、`--allow-read=.`
 * (讀 fixture wav、machine-profile 平台判斷用的 machine-profile,經
 * `../_machine`)。
 */
import { assert } from "jsr:@std/assert@^1.0.19";
import { fromFileUrl } from "jsr:@std/path@^1.0.9/from-file-url";
import {
  buildEngineArgs,
  platformFromProfile,
  readMachineProfile,
  resolveMachineProfilePath,
} from "../reference/main.ts";
import { stableEngineBinaryPath } from "../reference/downloader.ts";
import { EngineClient } from "../reference/engine.ts";
import { normalizedLevenshtein, normalizeForAsr } from "./asr_metric.ts";
import { chunkPcm16, readWavPcm16 } from "./wav.ts";

const APP_DIR = Deno.cwd();
// why fromFileUrl(見 reference/setup.ts 的 MANIFEST_PATH 註解):裸
// `new URL(...).pathname` 在 Windows 會產生 `/C:/...` 這種非法原生路徑，
// 導致 Deno.stat/readFile 以 os error 3 失敗——Task 6 Windows 演練實測發現。
const FIXTURE_PATH = fromFileUrl(new URL("./fixtures/hello-streaming.wav", import.meta.url));
const EXPECTED_TEXT = "Hello, this is a streaming speech recognition test";
const MAX_NORMALIZED_DISTANCE = 0.25;

Deno.test("protocol：真 host 真模型轉錄 hello-streaming.wav，與參考文字的正規化編輯距離 ≤ 0.25", async () => {
  const profile = await readMachineProfile(resolveMachineProfilePath());
  const platform = platformFromProfile(profile);
  const binPath = stableEngineBinaryPath(APP_DIR, platform);
  const args = buildEngineArgs(platform, `${APP_DIR}/model`, profile);

  const client = await EngineClient.spawn(binPath, args);
  try {
    client.start();
    const pcm = await readWavPcm16(FIXTURE_PATH);
    // 逐 chunk 送完再 stop——與 real_model.rs 的黑箱手法一致:協定裡
    // stdin 是循序讀取的一行一行,host 保證在讀到 stop 之前已經處理完
    // 前面所有排隊的 audio 指令,不需要在中途等待任何回應。
    for (const chunk of chunkPcm16(pcm)) {
      client.pushPcm(chunk);
    }
    const finalText = await client.stop();

    const expectedNorm = normalizeForAsr(EXPECTED_TEXT);
    const gotNorm = normalizeForAsr(finalText);
    const distance = normalizedLevenshtein(expectedNorm, gotNorm);
    assert(
      distance <= MAX_NORMALIZED_DISTANCE,
      `normalized edit distance ${distance.toFixed(3)} > ${MAX_NORMALIZED_DISTANCE}\n` +
        `expected (normalized): ${expectedNorm}\ngot (normalized): ${gotNorm}`,
    );
  } finally {
    client.kill();
  }
});
