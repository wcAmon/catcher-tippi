/**
 * `downsampler-core.js` 的單元測試——驗證 AudioWorklet 邏輯裡「不依賴
 * Web Audio API」的那部分數學:Int16 轉換的 clamping、固定大小分塊的邊界
 * 行為、線性內插重採樣的取樣率換算與跨批次連續性。
 *
 * why 這份測試不需要瀏覽器:`downsampler-core.js` 刻意設計成純函式(見
 * 該檔案檔頭 why 說明),完全不碰 `AudioWorkletProcessor`/`sampleRate`
 * 全域/`postMessage`,可以直接在 Deno 裡 `import` 並用一般陣列/
 * `Float32Array` 餵資料進去斷言——這正是「把 worklet 邏輯抽成可測試的
 * 純函式檔案」這個設計決策要換來的好處(見 PLAN.md Task 3 的驗證步驟)。
 *
 * 執行方式(不需要任何權限旗標——純函式測試,沒有檔案/網路 I/O):
 *   deno test recipes/tomato-ears/reference/ui/downsampler-core_test.ts
 */

import { assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1.0.19";
import {
  CHUNK_SIZE,
  createChunker,
  createLinearResampler,
  floatToInt16Sample,
  TARGET_SAMPLE_RATE,
} from "./downsampler-core.js";

// ---------------------------------------------------------------------------
// floatToInt16Sample:Int16 clamping
// ---------------------------------------------------------------------------

Deno.test("floatToInt16Sample：邊界值與零點", () => {
  assertEquals(floatToInt16Sample(0), 0);
  assertEquals(floatToInt16Sample(1), 32767); // 正側最大值
  assertEquals(floatToInt16Sample(-1), -32768); // 負側最小值(Int16 非對稱範圍)
});

Deno.test("floatToInt16Sample：超出 [-1, 1] 的樣本被 clamp，不溢位環繞", () => {
  // 若沒有 clamp，超出範圍的浮點數轉 Int16 時會溢位環繞（例如正值繞成
  // 負值），這裡驗證的正是「不會發生那種事」——輸出永遠停在 Int16 的
  // 合法邊界，而不是環繞出一個看似合法但實際上完全錯誤的數字。
  assertEquals(floatToInt16Sample(2), 32767);
  assertEquals(floatToInt16Sample(-2), -32768);
  assertEquals(floatToInt16Sample(100), 32767);
  assertEquals(floatToInt16Sample(-100), -32768);
});

Deno.test("floatToInt16Sample：中間值依 Int16 非對稱範圍正確縮放", () => {
  // 0.5 * 32767 = 16383.5，四捨五入為 16384。
  assertEquals(floatToInt16Sample(0.5), 16384);
  // -0.5 * 32768 = -16384，整數無條件捨入誤差。
  assertEquals(floatToInt16Sample(-0.5), -16384);
});

// ---------------------------------------------------------------------------
// createChunker：固定大小分塊的邊界行為
// ---------------------------------------------------------------------------

Deno.test("createChunker：CHUNK_SIZE 等於協定建議的 1600（100ms @ 16kHz）", () => {
  assertEquals(CHUNK_SIZE, 1600);
  assertEquals(TARGET_SAMPLE_RATE, 16000);
});

Deno.test("createChunker：剛好餵滿一個 chunk，恰好吐出一個 Int16Array(1600)", () => {
  const chunker = createChunker();
  const samples = new Array(CHUNK_SIZE).fill(0.1);
  const chunks = chunker.push(samples);
  assertEquals(chunks.length, 1);
  assertEquals(chunks[0].length, CHUNK_SIZE);
  assertEquals(chunks[0] instanceof Int16Array, true);
  assertEquals(chunks[0][0], floatToInt16Sample(0.1));

  // 緩衝區已經在上一次 push() 湊滿後重置——再餵 1 個樣本不會立刻吐出
  // 任何 chunk（還差 1599 個才會滿）。
  const next = chunker.push([0.2]);
  assertEquals(next.length, 0);
});

Deno.test("createChunker：一次餵超過一個 chunk 的量，跨越多個 chunk 邊界", () => {
  const chunker = createChunker();
  // 3600 = 2 個滿的 chunk（3200）+ 殘餘 400。
  const samples = new Array(3600).fill(0).map((_, i) => (i % 2 === 0 ? 0.1 : -0.1));
  const chunks = chunker.push(samples);
  assertEquals(chunks.length, 2);
  assertEquals(chunks[0].length, CHUNK_SIZE);
  assertEquals(chunks[1].length, CHUNK_SIZE);
  // 兩個吐出的 chunk 必須是不同的底層緩衝區（見 createChunker 的 why 註解：
  // transfer 之後不能重用同一塊記憶體）——對其中一個 chunk 寫入不應該
  // 影響另一個。
  assertEquals(chunks[0] === chunks[1], false);

  const remainder = chunker.flush();
  assertEquals(remainder?.length, 400);
  assertEquals(remainder instanceof Int16Array, true);

  // flush() 之後緩衝區已清空，再次 flush() 沒有殘餘可吐，回傳 null。
  assertEquals(chunker.flush(), null);
});

Deno.test("createChunker：餵空陣列不產生 chunk，也不影響既有緩衝區狀態", () => {
  const chunker = createChunker();
  assertEquals(chunker.push([]).length, 0);
  chunker.push(new Array(800).fill(0));
  assertEquals(chunker.push([]).length, 0); // 空陣列不會誤觸發「湊滿」判斷
  const remainder = chunker.flush();
  assertEquals(remainder?.length, 800);
});

Deno.test("createChunker：可以用自訂 chunkSize（不寫死 1600，供其他情境重用）", () => {
  const chunker = createChunker(4);
  const chunks = chunker.push([0.1, 0.2, 0.3, 0.4, 0.5]);
  assertEquals(chunks.length, 1);
  assertEquals(chunks[0].length, 4);
  const remainder = chunker.flush();
  assertEquals(remainder?.length, 1);
});

// ---------------------------------------------------------------------------
// createLinearResampler：48kHz → 16kHz 換算與跨批次連續性
// ---------------------------------------------------------------------------

Deno.test("createLinearResampler：48kHz→16kHz 整除比例，單次餵完整段長度精確", () => {
  // ratio = 48000 / 16000 = 3；餵入恰好 4800 個輸入樣本（= 0.1s @ 48kHz）
  // 應該產生恰好 1600 個輸出樣本（= 0.1s @ 16kHz）——這正是協定的
  // CHUNK_SIZE 為何選 1600 的來源換算。
  //
  // why 這裡用一般 Array、不用 Float32Array：這個測試要驗證的是內插
  // 「公式」本身算得對不對，Float32Array 會先把每個樣本截斷成單精度浮點
  // （例如 0.42 實際存成 0.41999998…），跟公式邏輯無關的精度損失會
  // 混進斷言誤差裡；下面的「串流連續性」測試才特別用 Float32Array，
  // 用意是驗證跟真實 AudioWorkletProcessor 收到的型別相容，兩個測試
  // 目的不同。
  const resampler = createLinearResampler(48000, 16000);
  const input = new Array(4800).fill(0.42);
  const output = resampler.push(input);
  assertEquals(output.length, 1600);
  // 常數訊號的線性內插結果仍然是同一個常數（s0 === s1 時內插無作用）。
  for (const sample of output) {
    assertEquals(sample, 0.42);
  }
});

Deno.test("createLinearResampler：整數 ratio（48kHz→16kHz）時 frac 恆為 0，等同等距抽樣", () => {
  // ratio = 3 是整數，且初始位置是 0，所以每次內插的 i 都剛好落在整數
  // 樣本點上（frac 恆為 0）——這種情況下線性內插退化成單純的等距抽樣
  // （每 3 個樣本取 1 個），並不會真的用到「兩點之間的插值」這條路徑。
  // 這個測試只驗證「等距抽樣」這個特例的正確性；真正會用到非零 frac
  // 插值路徑的驗證在下一個測試（44.1kHz→16kHz，ratio 非整數）。
  const resampler = createLinearResampler(48000, 16000);
  const input = new Array(9).fill(0).map((_, i) => i); // 0,1,2,...,8
  const output = resampler.push(input);
  // i0 依序是 0,3,6（i1=7<=8 合法）；i=9 時 i1=10>8 停止。
  assertEquals(output, [0, 3, 6]);
});

Deno.test("createLinearResampler：非整數 ratio（44.1kHz→16kHz）時線性斜坡的內插值精確落在同一條直線上", () => {
  // 44100 是常見的麥克風原生取樣率之一；44100/16000 = 2.75625，是非整數
  // ratio，會產生非零的 frac（真正走到內插公式的加權平均那一步，不是
  // 上一個測試的整數 ratio 退化情況）。輸入用線性斜坡（affine 函式）：
  // 線性內插重採樣一條線性訊號，理論上應該精確重現同一條線（只是取樣
  // 點間距變了），可以用來驗證內插公式本身沒有算錯（不是只驗證長度對了
  // 但數值錯的假陽性）。
  const inputSampleRate = 44100;
  const outputSampleRate = 16000;
  const resampler = createLinearResampler(inputSampleRate, outputSampleRate);
  const length = 4410; // 0.1s @ 44.1kHz
  const slope = 1 / length; // 輸入訊號 input[i] = i * slope，範圍落在 [0,1)
  const input = new Array(length);
  for (let i = 0; i < length; i++) input[i] = i * slope;

  const output = resampler.push(input);
  const ratio = inputSampleRate / outputSampleRate;
  // ratio 非整數時輸出長度不會剛好是 length/ratio 的整數倍，這裡不斷言
  // 具體長度，只斷言「每一個確實吐出來的樣本」都落在正確的直線上。
  for (let k = 0; k < output.length; k++) {
    const expected = k * ratio * slope; // 第 k 個輸出樣本對應的輸入軸位置
    assertAlmostEquals(output[k], expected, 1e-9);
  }
});

Deno.test("createLinearResampler：跨多次 push() 呼叫的結果，等價於一次餵完整段（串流連續性）", () => {
  // AudioWorkletProcessor.process() 每次固定收到 128 個原生取樣率樣本
  // （Web Audio 的 render quantum），不是一次收到一大段——這個測試驗證
  // `nextInputIndex`/`prevSample` 的跨批次狀態延續邏輯，是否讓「分成很多
  // 次 128-sample 的 push()」跟「一次餵完整段」得到（幾乎）相同的輸出，
  // 而不是在每個批次邊界上產生不連續的內插誤差。
  const inputSampleRate = 44100; // 非整數 ratio，讓 nextInputIndex 的分數進位真正被用到
  const outputSampleRate = 16000;
  const length = 4410; // 0.1s @ 44.1kHz，128 不能整除，最後一批會是不滿的殘餘區塊
  const input = new Float32Array(length);
  for (let i = 0; i < length; i++) input[i] = Math.sin(i * 0.01) * 0.5; // 非線性訊號，更貼近真實語音波形

  const wholeOutput = createLinearResampler(inputSampleRate, outputSampleRate).push(input);

  const streamed = [];
  const streamingResampler = createLinearResampler(inputSampleRate, outputSampleRate);
  const quantum = 128; // Web Audio render quantum
  for (let offset = 0; offset < length; offset += quantum) {
    const block = input.subarray(offset, Math.min(offset + quantum, length));
    streamed.push(...streamingResampler.push(block));
  }

  assertEquals(streamed.length, wholeOutput.length);
  for (let k = 0; k < wholeOutput.length; k++) {
    assertAlmostEquals(streamed[k], wholeOutput[k], 1e-9);
  }
});

Deno.test("createLinearResampler：1:1 比例（inputSampleRate === outputSampleRate），最後一個樣本留給下一批", () => {
  // 退化情境（ratio = 1）：常見於本機開發時麥克風原生取樣率剛好就是
  // 16kHz（少見但可能發生，例如某些虛擬音訊裝置）——驗證這個邊界情況
  // 不會除以零或跑出異常長度。
  //
  // why 單次 push() 只吐出 length-1 個輸出、不是 length 個：內插第 k 個
  // 輸出需要 i0 與 i1 兩個樣本（見演算法 why 註解），輸入陣列最後一個
  // 樣本永遠只能當「下一批的 i1／s1」用，不會在同一次 push() 裡被消耗成
  // 輸出——這是刻意的串流設計，不是傳回長度算錯。下面接著再 push 一個
  // 樣本，驗證它會在下一批被正確用上（見 nextInputIndex 的 why 註解）。
  const resampler = createLinearResampler(16000, 16000);
  const input = [0.1, 0.2, 0.3, 0.4];
  const output = resampler.push(input);
  assertEquals(output, [0.1, 0.2, 0.3]);

  const nextOutput = resampler.push([0.5]);
  assertEquals(nextOutput, [0.4]);
});
