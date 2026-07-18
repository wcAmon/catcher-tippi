/**
 * `AudioWorkletProcessor`：在瀏覽器的 audio rendering thread 上執行，把
 * 麥克風原生取樣率的 mono 音訊即時降採樣成 asr-host-v1 協定要的
 * 16 kHz PCM16 chunk（1600 samples ≈ 100ms 一個），用 transferable
 * `postMessage` 送回主執行緒。
 *
 * why 邏輯分兩個檔案：實際的內插/分塊數學在 `downsampler-core.js`（見
 * 該檔案檔頭 why 說明）——這裡完全不重新實作那些數學，只負責「接
 * Web Audio 的 process() callback」與「透過 port 跟主執行緒交換資料」
 * 這兩件屬於 AudioWorkletProcessor 生命週期的事。
 *
 * why 用 `import`（本檔是一個 ES module worklet）：`AudioContext
 * .audioWorklet.addModule()` 載入的檔案本身就是一個獨立的 module graph，
 * 支援 `import`/`export`——跟主執行緒的 `app.js` 是完全分開的執行環境
 * （沒有 DOM、沒有 window），但一樣可以用相對路徑 import 同目錄下的
 * 純函式模組，不需要打包工具、不需要外部 CDN（店規：零外部資源）。
 */
import { createChunker, createLinearResampler } from "./downsampler-core.js";

class DownsamplerProcessor extends AudioWorkletProcessor {
  #resampler;
  #chunker;

  constructor() {
    super();
    // `sampleRate` 是 AudioWorkletGlobalScope 的全域變數（非 import），
    // 值等於建立 AudioContext 時的取樣率——`app.js` 用瀏覽器裝置原生
    // 取樣率建立 AudioContext（見該檔 why 說明:不指定 sampleRate 選項，
    // 讓瀏覽器用麥克風原生速率,避免額外一層瀏覽器內建重採樣),所以這裡
    // 收到的就是麥克風的原生取樣率(常見 44100/48000 Hz)。
    this.#resampler = createLinearResampler(sampleRate);
    this.#chunker = createChunker();

    // 主執行緒在使用者按下「停止錄音」時,會送一個 {type:"flush"} 訊息
    // 過來,要求把緩衝區裡不到 100ms 的殘餘樣本立刻吐出(見
    // downsampler-core.js 的 flush() why 說明),避免最後一小段音訊被
    // 直接丟棄。這裡同步處理、同步回覆,不需要非同步狀態機。
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === "flush") {
        const remainder = this.#chunker.flush();
        if (remainder !== null) {
          // transfer 語意同下方 process() 內的說明:第二個參數把
          // remainder.buffer 的所有權轉移給主執行緒,避免複製整塊記憶體。
          this.port.postMessage(remainder.buffer, [remainder.buffer]);
        }
        // 額外送一個「已沖洗完成」的標記訊息,讓主執行緒知道可以安全地
        // 送出 WS 的 stop 控制訊息、拆掉音訊圖了——如果沒有這個標記,
        // 主執行緒沒有辦法知道上面那個(可能沒有的)binary postMessage
        // 什麼時候會抵達,可能在殘餘 chunk 送達之前就搶先呼叫了 stop()。
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  /**
   * Web Audio 每個 render quantum(固定 128 個樣本)呼叫一次。
   * `inputs[0]` 是第一個輸入的所有聲道,`inputs[0][0]` 是第 0 聲道
   * (mono 錄音只用得到這一個聲道——`app.js` 的 getUserMedia 約束已經
   * 要求 `channelCount: 1`)。
   *
   * why 回傳 `true`:AudioWorkletProcessor 的 spec 規定,回傳 `true` 會
   * 讓這個 processor 保持「存活」、持續收到後續的 process() 呼叫,即使
   * 這個節點沒有連到 `audioContext.destination`(本 worklet 刻意不接
   * destination——它只是把資料轉送出去,不需要真的播放聲音,見 app.js
   * 建立 AudioWorkletNode 時 `numberOfOutputs: 0` 的 why 註解)。回傳
   * `false`(或不回傳)則瀏覽器可能在沒有向 destination 輸出的情況下
   * 判定這個節點「用不到了」而提早停止呼叫 process(),導致錄音中斷。
   */
  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel || channel.length === 0) {
      // 尚未有實際音訊資料(例如麥克風串流剛連上、還沒有第一批樣本)——
      // 什麼都不做,但仍然回傳 true 讓 processor 保持存活等下一批。
      return true;
    }

    // channel 是 Float32Array,型別上與 createLinearResampler.push() 期待
    // 的「可用索引存取的樣本陣列」相容,不需要額外轉型。
    const resampled = this.#resampler.push(channel);
    const chunks = this.#chunker.push(resampled);
    for (const chunk of chunks) {
      // transferable postMessage:第二個參數列出要「轉移所有權」而非
      // 複製的 ArrayBuffer。音訊資料在 audio thread 上頻繁產生,若每次
      // 都用結構化複製(structured clone)整塊複製過去,在高頻呼叫下會
      // 造成不必要的記憶體配置與複製開銷;transfer 是零複製的所有權轉移
      // (對應地,createChunker() 每吐一個滿的 chunk 就換一個新緩衝區,
      // 不會在 transfer 之後又寫入同一塊記憶體——見該函式的 why 註解)。
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }
    return true;
  }
}

registerProcessor("downsampler", DownsamplerProcessor);
