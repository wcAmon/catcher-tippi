/**
 * 讀取 mono 16-bit PCM WAV 檔的 `data` chunk,並依協定建議的 1600-sample
 * chunk 大小切分——僅供 `verify/` 內的 `protocol_test.ts`/`service_test.ts`
 * 共用,不是通用的 WAV parser(不處理其他 chunk 類型、不驗證取樣率/聲道數
 * 等 fmt chunk 欄位)。`verify/fixtures/hello-streaming.wav` 由
 * `crates/nemotron-cli`/`catcher-ffi` 既有測試沿用的固定格式產生,保證是
 * RIFF/WAVE、PCM16、mono、16 kHz,不需要 parser 自己驗證這些前提。
 *
 * `readWavPcm16` 移植自 `crates/catcher-asr-host/tests/real_model.rs` 的
 * `read_wav_pcm16`,語意相同:找到 `"data"` 這四個位元組的位置,之後
 * 8 bytes(4-byte chunk id 本身 + 4-byte chunk size 欄位)開始就是 PCM
 * 資料。
 */

/** 讀取 WAV 檔,回傳 `data` chunk 的原始位元組(PCM16-LE)。 */
export async function readWavPcm16(path: string): Promise<Uint8Array> {
  const bytes = await Deno.readFile(path);
  const marker = new TextEncoder().encode("data");
  let dataPos = -1;
  for (let i = 0; i + 4 <= bytes.length; i++) {
    if (
      bytes[i] === marker[0] && bytes[i + 1] === marker[1] &&
      bytes[i + 2] === marker[2] && bytes[i + 3] === marker[3]
    ) {
      dataPos = i;
      break;
    }
  }
  if (dataPos === -1) {
    throw new Error(`WAV 檔找不到 "data" chunk:${path}`);
  }
  return bytes.subarray(dataPos + 8);
}

/**
 * 把 PCM16 位元組切成固定大小的 chunk(協定建議每 chunk 1600 samples ≈
 * 100ms,見 `docs/protocol/asr-host-v1.md`)。最後一個 chunk 可能短於
 * `chunkSamples`(協定文件:chunk 長度只是建議,不是強制,host 只要求
 * 位元組數為偶數——最後一段殘餘 chunk 天然滿足這個前提,因為輸入的
 * 位元組數本身就是偶數〔PCM16 是 2 bytes/sample〕)。
 */
export function chunkPcm16(samples: Uint8Array, chunkSamples = 1600): Uint8Array[] {
  const chunkBytes = chunkSamples * 2;
  const chunks: Uint8Array[] = [];
  for (let offset = 0; offset < samples.length; offset += chunkBytes) {
    chunks.push(samples.subarray(offset, Math.min(offset + chunkBytes, samples.length)));
  }
  return chunks;
}
