/**
 * `EngineClient`:asr-host-v1 協定(見 `PROTOCOL.md`,配方包內;源 repo 路徑
 * `docs/protocol/asr-host-v1.md`,凍結)的
 * Deno 端子行程包裝。管理 engine host(`catcher-asr-host` / `nemotron-asr-host`)
 * 的生命週期,把 stdin/stdout JSON-lines 轉成 TypeScript 的方法呼叫與 callback。
 *
 * 設計原則(why):
 * - **平台分支只在「選哪個 binary/args」一行**(見 `main.ts`)——`EngineClient`
 *   本身完全平台無關,只認協定,不認引擎實作細節。
 * - **`onPartial`/`onError` 是單一訂閱者的 callback,不是 event emitter**:
 *   v1 情境下一個 EngineClient 只服務一個當下的 WebSocket 連線(見
 *   `server.ts`),用簡單欄位賦值就夠,不需要引入 EventTarget 的額外複雜度。
 * - **`start`/`pushPcm` 回傳 `void`(fire-and-forget)**:協定裡這兩個指令本來
 *   就不是「一去一回」——`start` 成功不輸出任何事件,`audio` 的回應是
 *   非同步、可能連續好幾個 chunk 才有一次 partial。真正需要「一去一回」的
 *   只有 `stop`(等 final),所以只有 `stop()` 回傳 Promise。
 * - **host crash 或非預期結束 → `onError`,呼叫端自行決定是否 `kill()` +
 *   重新 `spawn()`**:`EngineClient` 不做自動重生,因為「要不要重生、重生
 *   幾次、要不要放棄」是應用層策略(server.ts/main.ts 的責任),不是這個
 *   薄包裝層該內建的行為。
 */

import { TextLineStream } from "jsr:@std/streams@^1.1.1/text-line-stream";
import { encodeBase64 } from "jsr:@std/encoding@^1.0.11/base64";

/** 協定事件的鬆散型別——只挑本模組需要的欄位,故意不用 discriminated union
 * 窮舉,避免遇到未來新增/未知事件類型時型別檢查在執行期才爆炸。 */
interface RawEvent {
  event?: unknown;
  backend?: unknown;
  text?: unknown;
  message?: unknown;
}

/** `stop()` 呼叫中的 pending 狀態:等待 host 回覆 final(或 error)。 */
interface PendingStop {
  resolve: (finalText: string) => void;
  reject: (err: Error) => void;
}

/**
 * 單一 engine host 子行程的包裝。
 *
 * 使用方式:`const client = await EngineClient.spawn(binPath, args)`——
 * `spawn` 會等到收到協定的 `ready` 事件才回傳(見協定:「行程啟動 → 載入
 * 模型 → 輸出 ready」,ready 之前不得有任何 stdout 行),`client.backend`
 * 即為 `ready.backend`。
 */
export class EngineClient {
  /** engine host 回報的後端字串(`mlx`/`dml`/`cpu`,測試用 host 可能是 `fake`)。
   * 依協定,消費端不得依此值分支行為,只能展示。 */
  readonly backend: string;

  /** 收到新的 partial 快照時呼叫。text 是「會話累積至今的全文」,不是增量
   * ——協定明文規定 client 必須整段覆寫顯示,連續兩則內容相同也是合法的。 */
  onPartial: (text: string) => void = () => {};

  /** 收到「無對應 pending 操作」的 error 事件時呼叫(例如 pushPcm 送出格式錯的
   * chunk、host 非預期結束/crash)。呼叫端可以選擇忽略、記錄,或 kill() +
   * 重新 spawn() 一個新的 EngineClient——本類別不自動重生。 */
  onError: (message: string) => void = () => {};

  #process: Deno.ChildProcess;
  #stdinWriter: WritableStreamDefaultWriter<Uint8Array>;
  #encoder = new TextEncoder();
  #pendingStop: PendingStop | null = null;
  /** 由 `kill()` 主動設定;讀取迴圈用它區分「使用者主動關閉」與「host crash」,
   * 避免主動關閉時還誤觸發一次 onError。 */
  #killedByUser = false;

  private constructor(
    process: Deno.ChildProcess,
    backend: string,
    stdinWriter: WritableStreamDefaultWriter<Uint8Array>,
  ) {
    this.#process = process;
    this.backend = backend;
    this.#stdinWriter = stdinWriter;
  }

  /** 底層子行程 PID,供上層診斷/記錄用(例如把 crash 前的 PID 寫進日誌)。 */
  get pid(): number {
    return this.#process.pid;
  }

  /**
   * 啟動 engine host 子行程,等待協定的 `ready` 事件,回傳可用的
   * `EngineClient`(`backend` 已就緒)。
   *
   * why 在這裡就同步等 ready、而不是讓呼叫端自己等:協定保證 ready 是
   * 「模型已載入完成」的訊號,呼叫端(server.ts/main.ts)在拿到
   * `EngineClient` 之前不該假設引擎可用——把這個等待收斂在 `spawn()`
   * 內,呼叫端寫法就能簡單地 `await`,不必自己重新實作「讀第一行」邏輯。
   */
  static async spawn(binPath: string, args: string[]): Promise<EngineClient> {
    const command = new Deno.Command(binPath, {
      args,
      stdin: "piped",
      stdout: "piped",
      // stderr 依協定只供人類閱讀,消費端必須忽略;但仍要 pipe 走(而非
      // "inherit"/"null")並主動排空,否則 host 寫多了會塞滿管線緩衝區,
      // 反過來卡住 host 自己的 stdout 寫入(該行程可能是同步阻塞式輸出)。
      stderr: "piped",
    });
    const process = command.spawn();
    const stdinWriter = process.stdin.getWriter();

    const lineIterator = process.stdout
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new TextLineStream())
      [Symbol.asyncIterator]();

    const first = await lineIterator.next();
    if (first.done) {
      throw new Error(
        `engine host(${binPath})未輸出任何 stdout 行就結束,可能啟動失敗(檢查 --model 路徑等參數)`,
      );
    }
    const readyEvent = parseEventLine(first.value);
    if (
      readyEvent === undefined || readyEvent.event !== "ready" ||
      typeof readyEvent.backend !== "string"
    ) {
      throw new Error(`engine host(${binPath})第一行不是合法的 ready 事件:${first.value}`);
    }

    const client = new EngineClient(process, readyEvent.backend, stdinWriter);
    // 背景執行,不 await——生命週期由子行程自身結束或 kill() 決定,
    // 迴圈內的例外一律轉成 onError/pendingStop 的 reject,不會讓這個
    // async function 拋出未被捕捉的例外(fire-and-forget 是刻意設計)。
    client.#runReadLoop(lineIterator);
    client.#drainStderr();
    return client;
  }

  /**
   * 開新會話。`lang` 對齊協定的 `start.lang` 欄位——v1 中此欄位由 host 的
   * `--language` 啟動參數決定,host 接受但忽略,這裡保留參數只是為了
   * 前向相容(未來協定版本若啟用它,呼叫端不必改介面)。
   */
  start(lang = "auto"): void {
    this.#send({ cmd: "start", lang, sample_rate: 16000 });
  }

  /** 送一個 PCM16-LE mono 16kHz 音訊 chunk(協定建議每 chunk 1600 samples ≈100ms)。 */
  pushPcm(chunk: Uint8Array): void {
    this.#send({ cmd: "audio", pcm16_b64: encodeBase64(chunk) });
  }

  /**
   * 沖洗解碼器並結束會話,resolve 為 host 回傳的 `final.text`。
   *
   * 若 host 在 stop 之後回報 `error`(協定:沖洗/解碼失敗,該會話視為已
   * 結束、不會再有 final),則這個 Promise 改以 reject 收尾——呼叫端一定會
   * 得到一個明確的了結,不會無限期懸掛。
   */
  stop(): Promise<string> {
    if (this.#pendingStop) {
      return Promise.reject(new Error("stop() 已在進行中,不可重複呼叫"));
    }
    return new Promise<string>((resolve, reject) => {
      this.#pendingStop = { resolve, reject };
      this.#send({ cmd: "stop" });
    });
  }

  /**
   * 主動終止子行程。呼叫端(通常是 server.ts 偵測到需要重啟,或應用關閉)
   * 之後可以用新的 `binPath`/`args` 呼叫 `EngineClient.spawn()` 重新建立
   * 一個實例——本類別本身不記錄「上次怎麼 spawn 的」,重生策略交給呼叫端。
   */
  kill(): void {
    this.#killedByUser = true;
    try {
      this.#process.kill();
    } catch {
      // 行程可能已經結束(例如 host 自己先 crash 了),再次 kill 會拋錯,
      // 這裡的語義是「確保結束」,不是「必須原本存活」,忽略即可。
    }
    // stdin 沒有更多資料要寫;顯式關閉讓底層資源盡快釋放,而非等 GC。
    this.#stdinWriter.close().catch(() => {});
  }

  /** 把物件序列化成一行 JSON,寫入 stdin。寫入失敗(通常代表 host 已死)
   * 一律轉成 onError,不讓 `start`/`pushPcm` 這兩個宣告為 `void` 的方法
   * 意外把 rejected promise 丟給呼叫端處理。 */
  #send(command: Record<string, unknown>): void {
    const line = this.#encoder.encode(JSON.stringify(command) + "\n");
    this.#stdinWriter.write(line).catch((err) => {
      this.onError(
        `寫入 engine host stdin 失敗(行程可能已結束):${
          err instanceof Error ? err.message : String(err)
        }`,
      );
    });
  }

  /** 背景讀取 stdout,逐行解析協定事件並派發。 */
  async #runReadLoop(lineIterator: AsyncIterator<string>): Promise<void> {
    try {
      while (true) {
        const { value, done } = await lineIterator.next();
        if (done) break; // stdout EOF:host 行程已結束(見下方善後)
        this.#handleLine(value);
      }
    } catch (err) {
      this.#handleHostFailure(
        `讀取 engine host stdout 時發生錯誤:${err instanceof Error ? err.message : String(err)}`,
      );
      return;
    }
    if (!this.#killedByUser) {
      // 協定裡唯一「host 主動結束」的合法情境是 stdin EOF(我們自己關閉,
      // 對應 kill() 設的旗標);其餘任何未經我們要求就結束 stdout 的情況
      // 都視為非預期結束(crash 或模型載入後某種致命錯誤)。
      this.#handleHostFailure("engine host 行程已非預期結束(stdout 提前關閉)");
    }
  }

  #handleLine(line: string): void {
    if (line.trim() === "") return;
    const parsed = parseEventLine(line);
    if (parsed === undefined) {
      this.onError(`無法解析 engine host 輸出行:${line}`);
      return;
    }
    switch (parsed.event) {
      case "partial":
        this.onPartial(typeof parsed.text === "string" ? parsed.text : "");
        break;
      case "final": {
        const text = typeof parsed.text === "string" ? parsed.text : "";
        if (this.#pendingStop) {
          this.#pendingStop.resolve(text);
          this.#pendingStop = null;
        } else {
          // 協定保證一個會話恰好一次 final,且只會在 stop() 之後出現;
          // 收到「無 pending stop 的 final」代表協定假設被破壞,回報但不拋出
          // (拋出會讓背景讀取迴圈意外中止,反而讓後續事件全部收不到)。
          this.onError(`收到未預期的 final 事件(無對應的 pending stop):${text}`);
        }
        break;
      }
      case "error": {
        const message = typeof parsed.message === "string" ? parsed.message : "";
        if (this.#pendingStop) {
          // 協定:stop 後的 error 表示沖洗/解碼失敗,會話視為已結束,
          // 不會再有 final——所以用 reject 了結這次 stop(),而不是放著等。
          this.#pendingStop.reject(new Error(message));
          this.#pendingStop = null;
        } else {
          this.onError(message);
        }
        break;
      }
      case "ready":
        // 協定裡 ready 只會在行程啟動時出現一次,spawn() 已經消費過了。
        // 防禦性地把重複 ready 視為異常但不致命,交給 onError 讓呼叫端知道。
        this.onError("收到未預期的重複 ready 事件");
        break;
      default:
        this.onError(`收到未知事件類型:${line}`);
    }
  }

  #handleHostFailure(message: string): void {
    if (this.#pendingStop) {
      this.#pendingStop.reject(new Error(message));
      this.#pendingStop = null;
    }
    this.onError(message);
  }

  /** 排空 stderr(協定:人類可讀日誌,消費端必須忽略)。轉發到 console.error
   * 只是方便本機除錯,不影響協定語義。 */
  async #drainStderr(): Promise<void> {
    const lines = this.#process.stderr
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new TextLineStream());
    try {
      for await (const line of lines) {
        if (line.trim() === "") continue;
        console.error(`[engine host stderr] ${line}`);
      }
    } catch {
      // stdout 迴圈已經負責回報行程層級的失敗;stderr 排空迴圈的例外
      // (通常只是「串流已被行程結束連帶關閉」)不需要重複回報。
    }
  }
}

/** 把一行 stdout 解析成 `RawEvent`;解析失敗回傳 `undefined`(呼叫端決定要
 * 回報成 onError 還是視為 spawn() 失敗)。 */
function parseEventLine(line: string): RawEvent | undefined {
  try {
    const parsed = JSON.parse(line);
    if (parsed && typeof parsed === "object") return parsed as RawEvent;
    return undefined;
  } catch {
    return undefined;
  }
}
