/**
 * 錄音頁降採樣的核心數學：線性內插重採樣、Int16 轉換、固定大小分塊。
 *
 * 為什麼獨立成這個檔案（why，對齊店規第 4 條的密度要求）：
 * `AudioWorkletProcessor` 只能活在瀏覽器的 audio rendering thread 裡，沒有
 * DOM、沒有 `Deno` 全域、也拿不到瀏覽器來跑測試——如果把這裡的數學寫死在
 * `downsampler-worklet.js` 裡，就完全沒辦法用 `deno test` 驗證。這個檔案
 * 刻意不碰任何 Web Audio / DOM API（沒有 `AudioWorkletProcessor`、
 * `sampleRate` 全域、`postMessage`），純粹是幾個可以直接餵陣列進去、看
 * 陣列吐出來的函式──`downsampler-worklet.js`（瀏覽器端）與
 * `downsampler-core_test.ts`（Deno 端）是這個模組僅有的兩個呼叫端，各自
 * 用 `import` 這一份邏輯，不重複實作、不會漂移。
 */

/** 錄音頁固定往 asr-host-v1 協定的目標取樣率送資料。協定本身只接受
 * `sample_rate: 16000`（見 `PROTOCOL.md`，配方包內；源 repo 路徑
 * `docs/protocol/asr-host-v1.md` 的 `start` 指令格式），寫死在這裡比讓
 * 呼叫端每次都要手動傳入更不容易傳錯。 */
export const TARGET_SAMPLE_RATE = 16000;

/**
 * 每個 PCM chunk 累積滿的樣本數，對應協定文件裡的建議值：
 * 「audio 建議每 chunk 1600 samples（100 ms）」（16000 Hz × 0.1 s = 1600）。
 *
 * why 剛好是 100ms、不是更小或更大：
 * - 太小（例如 10ms/160 samples）→ WS binary frame 數量暴增，
 *   base64 編碼、JSON-lines 寫入 host stdin 的次數也跟著暴增，
 *   對單機迴圈（Deno ↔ engine host 子行程）是不必要的多次系統呼叫開銷；
 * - 太大（例如 1s）→ partial 結果的更新延遲被拉高到使用者可感知的程度，
 *   「即時轉錄」的體感會變差；
 * - 100ms 是串流 ASR 常見的甜蜜點，也是協定文件明文的建議值——這裡
 *   遵守協定建議，不是隨意選的數字。
 */
export const CHUNK_SIZE = 1600;

/**
 * 把一個（理論上落在 [-1, 1] 之間的）float 樣本轉成 PCM16-LE 需要的
 * Int16 整數樣本。
 *
 * why 要 clamp：麥克風增益、削波（clipping）或降採樣內插的浮點誤差都可能
 * 讓樣本值略微超出 [-1, 1]（例如 1.0000001）。Int16 是有號整數，若不
 * clamp 直接轉型，超出範圍的值會溢位環繞（例如 32768 變成 -32768），
 * 在音訊裡聽起來是刺耳的爆音雜訊，比單純削頂（clip）更難聽、更難除錯。
 *
 * why 正負兩側用不同的縮放係數（32768 vs 32767）：Int16 的範圍是
 * [-32768, 32767]，非對稱（正側比負側少一格，因為 0 算在正側）。用
 * 對稱的 32767 去乘負值會讓 -1.0 只映射到 -32767、永遠碰不到 Int16
 * 理論上的最小值 -32768；分開處理才能讓滿幅信號真正用滿整個 Int16 範圍。
 */
export function floatToInt16Sample(sample) {
  const clamped = Math.max(-1, Math.min(1, sample));
  const scaled = clamped < 0 ? clamped * 32768 : clamped * 32767;
  // Math.round 後再 clamp 一次：理論上 scaled 已經落在範圍內，但保留這道
  // 保險不吃效能（純整數比較），換來「這個函式的回傳值保證合法 Int16」
  // 這個不變量，呼叫端不必再自行檢查。
  return Math.max(-32768, Math.min(32767, Math.round(scaled)));
}

/**
 * 建立一個「串流線性內插重採樣器」：把來源取樣率（麥克風的原生取樣率，
 * 常見 44100/48000 Hz）的 float 樣本，逐次 `push()` 一批批地轉成目標取樣率
 * （16000 Hz）的 float 樣本。
 *
 * why 用線性內插、不用更精確的重採樣演算法（例如 sinc 內插/多相濾波器）：
 * 這是店規第 2 條「技術棧限縮」延伸出的取捨——AudioWorkletProcessor
 * 執行在即時音訊執行緒，任何一次 `process()` 呼叫超過預算（通常是幾毫秒）
 * 就會造成音訊卡頓（underrun）；線性內插是 O(n)、沒有濾波器狀態、沒有
 * FFT，在 audio thread 上的運算成本可忽略不計。代價是高頻會有輕微失真
 * （線性內插等效於一個粗糙的低通濾波器），但語音辨識模型輸入本來就是
 * 16kHz、且人聲的主要能量集中在較低頻段，這個失真對 ASR 準確率的影響
 * 遠小於「audio thread 掉幀導致錄到的音訊本身就不連續」的風險。
 *
 * why 要「串流」而不是每批獨立處理：`AudioWorkletProcessor.process()`
 * 固定每次收到 128 個樣本（Web Audio 的 render quantum），若每批獨立
 * 從頭內插，批次邊界上的樣本會被錯誤地「斷開」重算，累積下來的相位誤差
 * 會讓輸出出現週期性的細微爆音。這裡用 `nextInputIndex`/`prevSample`
 * 兩個閉包變數跨批次保留內插狀態，讓連續多次 `push()` 的輸出等價於
 * 「一次餵完整段音訊」的結果（`downsampler-core_test.ts` 有專門驗證
 * 這個等價性的測試）。
 */
export function createLinearResampler(inputSampleRate, outputSampleRate = TARGET_SAMPLE_RATE) {
  // ratio：每輸出 1 個目標取樣率樣本，要在來源取樣率的時間軸上跨過幾個
  // 樣本。例如 48000→16000 時 ratio=3：每 3 個輸入樣本才產生 1 個輸出樣本。
  const ratio = inputSampleRate / outputSampleRate;

  // prevSample：上一次 push() 那批資料的最後一個樣本。當內插需要「本批
  // 開頭之前」的樣本時（見下方 i0 < 0 分支）用它補上，避免每批開頭都
  // 從靜音（0）開始內插造成不連續。
  let prevSample = 0;

  // nextInputIndex：下一個輸出樣本，對應到「本次 push() 傳入的陣列」座標
  // 系裡的浮點位置。可能是負數（表示落在 prevSample 與本批第 0 個樣本
  // 之間），這是刻意設計，見下方迴圈內的 i0 < 0 分支。
  let nextInputIndex = 0;

  /**
   * 餵一批來源取樣率的 float 樣本，回傳重採樣後的目標取樣率 float 樣本
   * （長度約為 `inputSamples.length / ratio`，實際值依內插邊界而定）。
   */
  function push(inputSamples) {
    const output = [];
    let i = nextInputIndex;
    while (true) {
      const i0 = Math.floor(i);
      const i1 = i0 + 1;
      if (i1 > inputSamples.length - 1) {
        // 內插需要 i0 與 i1 兩個樣本，但 i1 這個樣本還沒送到（在下一批
        // 才會出現）——停在這裡，把浮點位置留到下一次 push() 繼續，
        // 不能用「本批最後一個樣本」湊數，否則會提早結束內插、漏樣本。
        break;
      }
      const s0 = i0 < 0 ? prevSample : inputSamples[i0];
      const s1 = inputSamples[i1];
      const frac = i - i0; // 0..1 之間，i 落在 s0/s1 之間的比例位置
      output.push(s0 + (s1 - s0) * frac);
      i += ratio;
    }
    // 把座標系換算回「下一批的相對位置」：目前的 i 是相對本批開頭算的，
    // 減去本批長度後就是相對「下一批開頭」的位置（可能是負值，落在
    // prevSample 與下一批第 0 個樣本之間）。
    nextInputIndex = i - inputSamples.length;
    if (inputSamples.length > 0) {
      prevSample = inputSamples[inputSamples.length - 1];
    }
    return output;
  }

  return { push };
}

/**
 * 建立一個「固定大小分塊器」：把陸續 `push()` 進來的 float 樣本（重採樣器
 * 的輸出）轉成 Int16、累積到剛好 `chunkSize` 個樣本就吐出一個新配置的
 * `Int16Array`。
 *
 * why 累積到「剛好」`CHUNK_SIZE` 才吐：對齊協定建議的 100ms chunk 大小
 * （見 `CHUNK_SIZE` 的 why 註解）；「剛好」而非「至少」是因為
 * `AudioWorkletProcessor` 每次 `process()` 只收到 128 個原生取樣率樣本
 * （換算成 16kHz 大約 42-43 個樣本，遠小於 1600），非跨批次累積不可能
 * 湊到 1600──這正是這個函式存在的理由。
 *
 * why 每吐一個滿的 chunk 就換一個新的 `Int16Array` 緩衝區（而不是原地
 * 清空重用）：吐出去的 `Int16Array` 會被 `downsampler-worklet.js` 用
 * `postMessage(chunk.buffer, [chunk.buffer])` transfer 給主執行緒——
 * transfer 之後這塊記憶體的所有權已經轉移，worklet 這邊的參照會變成
 * detached（長度變 0），如果不換新緩衝區、繼續在原本的 `buffer` 上寫入
 * 下一批樣本，會直接對一塊已經不屬於這裡的記憶體寫入，是未定義行為。
 */
export function createChunker(chunkSize = CHUNK_SIZE) {
  let buffer = new Int16Array(chunkSize);
  let filled = 0; // buffer 裡目前已經填了幾個樣本（< chunkSize）

  /**
   * 餵一批 float 樣本（可以跨越好幾個 chunk 邊界）。回傳這次呼叫「新湊滿」
   * 的所有 chunk（可能是空陣列、一個，或多個——取決於這批樣本的長度）。
   * 未湊滿 chunkSize 的殘餘樣本留在內部緩衝區，等下一次 push() 或
   * `flush()`。
   */
  function push(floatSamples) {
    const chunks = [];
    let offset = 0;
    while (offset < floatSamples.length) {
      // 一次只處理「填滿目前緩衝區」與「這批剩餘樣本數」兩者中較小的量，
      // 讓一次 push() 可以正確跨越多個 chunk 邊界（例如一次餵 3600 個
      // 樣本會吐出 2 個滿的 chunk，剩 400 個留在緩衝區）。
      const space = chunkSize - filled;
      const take = Math.min(space, floatSamples.length - offset);
      for (let k = 0; k < take; k++) {
        buffer[filled + k] = floatToInt16Sample(floatSamples[offset + k]);
      }
      filled += take;
      offset += take;
      if (filled === chunkSize) {
        chunks.push(buffer);
        buffer = new Int16Array(chunkSize); // 見上方 why：換新緩衝區，不重用
        filled = 0;
      }
    }
    return chunks;
  }

  /**
   * 把目前緩衝區裡「未滿 chunkSize」的殘餘樣本立刻吐出（長度 < chunkSize）。
   * 供使用者按下「停止錄音」時呼叫，避免最後不到 100ms 的音訊被直接丟棄
   * ——協定的 `audio` 指令沒有限制 chunk 必須剛好是建議長度，只是建議值，
   * 送一個較短的收尾 chunk 是合法的。緩衝區是空的（filled === 0）時回傳
   * `null`，呼叫端不必再送一個空的 binary frame。
   */
  function flush() {
    if (filled === 0) return null;
    const remainder = buffer.slice(0, filled);
    buffer = new Int16Array(chunkSize);
    filled = 0;
    return remainder;
  }

  return { push, flush };
}
