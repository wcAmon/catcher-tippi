/**
 * 錄音頁主執行緒邏輯:UI 事件、麥克風擷取、AudioWorklet 圖組裝、WebSocket
 * 收送。降採樣的實際數學不在這裡——見 `downsampler-worklet.js` 與
 * `downsampler-core.js`;本檔只負責「組裝」與「畫面呈現」。
 *
 * why 不用任何框架(React/Vue/…)、不打包:店規第 2 條技術棧限縮 +
 * 「無框架、無外部 CDN」的明文需求(見 PLAN.md Task 3)。這份 UI 的
 * 複雜度(幾個按鈕、一段文字、一個列表)完全不需要框架,直接操作
 * DOM 比引入一整套建置流程(npm/bundler/CDN 依賴)更貼近「單檔可讀、
 * 給重建 agent 當語義錨點」的目標。
 *
 * why 這裡不用 TypeScript:`ui/` 底下的檔案是瀏覽器直接載入的原始檔
 * (`<script src="app.js">`、`audioWorklet.addModule("downsampler-worklet.js")`),
 * 沒有建置步驟做轉譯——維持與 `downsampler-worklet.js`/
 * `downsampler-core.js` 一致的純 JS,瀏覽器不需要任何轉換就能執行。
 */

// ---------------------------------------------------------------------------
// DOM 參照
// ---------------------------------------------------------------------------

const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const backendBadge = document.getElementById("backend-badge");
const connectionStatus = document.getElementById("connection-status");
const partialText = document.getElementById("partial-text");
const finalList = document.getElementById("final-list");
const emptyState = document.getElementById("empty-state");
const copyAllButton = document.getElementById("copy-all-button");
const exportButton = document.getElementById("export-button");
const errorBanner = document.getElementById("error-banner");

// ---------------------------------------------------------------------------
// 應用狀態
// ---------------------------------------------------------------------------

/** 所有已經收到 `final` 事件的逐字稿,依收到順序累積——「final 後訊息列表
 * 累積」是明文需求,跟 partial 的「快照替換」語意相反,兩者不能共用同一個
 * 顯示邏輯。 */
const finalMessages = [];

/** 錄音時建立、停止後釋放的一組資源;放在單一物件而不是散落的模組級變數,
 * 是為了讓「開始錄音」跟「停止錄音」各自只需要處理一個參照,不必記住
 * 一長串變數名稱各自清哪個。null 代表目前沒有在錄音。 */
let session = null;

let socket = null;

// ---------------------------------------------------------------------------
// WebSocket:連上 server.ts 的 `/ws`,依協定收 ready/partial/final/error
// ---------------------------------------------------------------------------

/**
 * why 用 `location.host` 組 WS URL、不寫死埠號:`server.ts` 固定綁
 * 127.0.0.1:43117,但寫死埠號等於在這份 reference UI 裡重複一份「目前
 * 用哪個埠」的事實,跟 `main.ts` 的 `DEFAULT_PORT` 各自維護容易漂移。
 * 瀏覽器載入這個頁面時,`location.host` 本來就已經是「使用者實際連線的
 * host:port」,直接沿用最不容易出錯。
 */
function connectWebSocket() {
  socket = new WebSocket(`ws://${location.host}/ws`);

  socket.onopen = () => {
    setConnectionStatus("已連線");
  };

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleServerMessage(message);
  };

  socket.onerror = () => {
    // `onerror` 不帶有用的錯誤細節(瀏覽器 WebSocket API 的既有限制),
    // 真正的處理留給緊接著必然觸發的 `onclose`。
    setConnectionStatus("連線發生錯誤");
  };

  socket.onclose = () => {
    setConnectionStatus("連線已中斷,請重新整理頁面");
    // 連線斷了就不可能再送音訊或收到轉錄結果——錄音中就把畫面收斂回
    // 「未錄音」狀態,避免使用者對著一個實際上不會有任何回應的按鈕錄音。
    if (session !== null) {
      teardownSession();
    }
    startButton.disabled = true;
    stopButton.disabled = true;
  };
}

/** 依 server.ts 的 `ServerMessage` 型別(見該檔案 WS 協定說明)分派處理。
 * `type` 是本協定唯一允許分支的欄位——`backend` 的值本身依店規/協定文件
 * 明文規定「僅展示,不分支」,下面看得到 `backend` 只被拿去塞進畫面文字,
 * 沒有任何 `if (backend === ...)` 的邏輯分支。 */
function handleServerMessage(message) {
  switch (message.type) {
    case "ready":
      // 顯示層之外沒有任何依 backend 值做的行為分支——見上方 why 說明。
      backendBadge.textContent = `後端:${message.backend}`;
      startButton.disabled = false;
      break;
    case "partial":
      // 快照替換語意:協定明文規定 partial.text 是「會話累積至今的全文」,
      // 不是增量,所以這裡永遠是整段覆寫(textContent 賦值),絕對不能用
      // 任何形式的字串串接/append。
      partialText.textContent = message.text;
      break;
    case "final":
      appendFinalMessage(message.text);
      partialText.textContent = "";
      break;
    case "error":
      showError(message.message);
      break;
    default:
      showError(`收到未知的伺服器訊息類型:${JSON.stringify(message)}`);
  }
}

/** 更新頁面頂端的連線狀態文字(WS 連線的生命週期提示,非錯誤訊息)。 */
function setConnectionStatus(text) {
  connectionStatus.textContent = text;
}

/** 在錯誤橫幅顯示一則訊息;重複呼叫以最新一則覆蓋(單一橫幅,不堆疊)。 */
function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

/** 隱藏並清空錯誤橫幅(開始新一次錄音時呼叫,舊錯誤不再相關)。 */
function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

// ---------------------------------------------------------------------------
// final 訊息列表(累積,非替換)
// ---------------------------------------------------------------------------

/** 把一則 final 逐字稿加進資料陣列與畫面列表尾端(累積語意,與 partial 的
 * 快照替換相反),並解鎖複製/匯出按鈕。 */
function appendFinalMessage(text) {
  finalMessages.push(text);
  emptyState.hidden = true;

  const item = document.createElement("li");
  item.className = "final-item";
  item.textContent = text;
  finalList.appendChild(item);

  // 有至少一則 final 訊息才允許複製/匯出——空清單去複製或匯出一個空檔案
  // 對使用者沒有意義,不如直接把按鈕禁用掉,省去他們自己發現「怎麼複製了
  // 空白」的困惑。
  copyAllButton.disabled = false;
  exportButton.disabled = false;
}

/** 把所有 final 訊息以換行接起來,做為複製/匯出共用的文字內容。 */
function joinedTranscript() {
  return finalMessages.join("\n");
}

copyAllButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(joinedTranscript());
    flashButtonFeedback(copyAllButton, "已複製");
  } catch (err) {
    // Clipboard API 在非 HTTPS/非 localhost 來源會被拒絕——本服務固定
    // 綁 127.0.0.1(店規第 5 條),瀏覽器把 localhost/127.0.0.1 視為
    // secure context,正常情況下不會走到這裡;仍保留錯誤提示做為保險。
    showError(`複製失敗:${err instanceof Error ? err.message : String(err)}`);
  }
});

exportButton.addEventListener("click", () => {
  const blob = new Blob([joinedTranscript()], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `tomato-ears-逐字稿-${timestampForFilename()}.txt`;
  // 觸發下載:把 <a> 暫時掛進 DOM 再 click()——某些瀏覽器對「沒有掛在
  // DOM 上的元素」的合成 click() 事件不保證觸發下載行為,掛進去再移除是
  // 常見且可靠的寫法。
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // 下載已經由瀏覽器接手處理 Blob 內容,object URL 這時可以立刻釋放,
  // 不需要等待——避免遺留未釋放的記憶體(URL.revokeObjectURL 是瀏覽器
  // 建議的清理方式)。
  URL.revokeObjectURL(url);
  flashButtonFeedback(exportButton, "已匯出");
});

/** 產生匯出檔名用的本地時間戳(YYYYMMDD-HHMMSS,不含檔名非法字元)。 */
function timestampForFilename() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

/** 按鈕點擊後短暫顯示「已複製」/「已匯出」文字回饋,1.2 秒後恢復原字樣
 * ——比彈出 alert() 或另外畫一個 toast 元件簡單,對這種輕量操作已經夠用。 */
function flashButtonFeedback(button, feedbackText) {
  const original = button.textContent;
  button.textContent = feedbackText;
  setTimeout(() => {
    button.textContent = original;
  }, 1200);
}

// ---------------------------------------------------------------------------
// 錄音控制:開始/停止
// ---------------------------------------------------------------------------

startButton.addEventListener("click", () => {
  startRecording().catch((err) => {
    showError(`無法開始錄音:${err instanceof Error ? err.message : String(err)}`);
    teardownSession();
    startButton.disabled = false;
    stopButton.disabled = true;
  });
});

stopButton.addEventListener("click", () => {
  stopRecording().catch((err) => {
    showError(`停止錄音時發生錯誤:${err instanceof Error ? err.message : String(err)}`);
  });
});

async function startRecording() {
  clearError();
  startButton.disabled = true;

  // channelCount: 1——mono。downsampler-worklet.js 只讀 inputs[0][0]
  // (第 0 聲道),要求瀏覽器一開始就只給單聲道,避免立體聲輸入時
  // worklet 靜靜地把右聲道整個丟掉而沒有任何提示。
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    video: false,
  });

  // 麥克風一到手就立刻建立(部分填充的)session——而不是等所有 await 都
  // 成功後才一次性賦值。why:這個函式後面還有兩個可能失敗的非同步步驟
  // (addModule 可能因 worklet 檔案載入/解析失敗而 throw),如果 session
  // 要等全部成功才存在,中途失敗時 catch 分支呼叫的 teardownSession()
  // 會因為 session === null 而 no-op——麥克風串流已經取得、指示燈已亮,
  // 卻沒有任何人負責 stop 它,使用者會看到「錄音失敗但麥克風還開著」。
  // 讓 teardownSession() 對「已取得的任何資源」都能生效(它對每個欄位
  // 各自做 null 檢查),失敗發生在哪一步都能完整回收。
  session = {
    stream,
    audioContext: null,
    sourceNode: null,
    workletNode: null,
    requestFlush: null,
  };

  // 不傳 sampleRate 選項:讓 AudioContext 使用裝置原生取樣率。
  // why:若指定一個跟裝置原生取樣率不同的值,瀏覽器會在我們接觸得到
  // 這些樣本之前,先用它自己的(黑箱、品質未知的)重採樣器轉換一次
  // ——這樣 downsampler-worklet.js 收到的就不是「麥克風原始樣本」,
  // 而是「瀏覽器已經重採樣過一次的樣本」,等於重採樣兩次,徒增失真。
  // 維持原生取樣率,只在 worklet 裡做我們自己知道邏輯的那一次降採樣。
  const audioContext = new AudioContext();
  session.audioContext = audioContext;
  await audioContext.audioWorklet.addModule("downsampler-worklet.js");

  const sourceNode = audioContext.createMediaStreamSource(stream);
  session.sourceNode = sourceNode;
  const workletNode = new AudioWorkletNode(audioContext, "downsampler", {
    numberOfInputs: 1,
    // numberOfOutputs: 0——這個節點不產生要播放的音訊,只是把資料轉送到
    // 主執行緒,不需要接到 audioContext.destination(也因此不會有麥克風
    // 直接回放造成的回音風險)。downsampler-worklet.js 的 process() 會
    // 回傳 true 讓瀏覽器持續呼叫它,即使沒有連到 destination 也不會被
    // 提早判定為「用不到」(見該檔案 why 註解)。
    numberOfOutputs: 0,
    channelCount: 1,
    channelCountMode: "explicit",
    channelInterpretation: "speakers",
  });
  session.workletNode = workletNode;

  let pendingFlush = null;
  workletNode.port.onmessage = (event) => {
    const data = event.data;
    if (data instanceof ArrayBuffer) {
      // binary frame:worklet 湊滿(或使用者停止錄音時沖洗出的殘餘)一個
      // PCM16 chunk,依 server.ts 的 WS 協定直接送 binary frame,不额外
      // 包 JSON/base64(那一層轉換是 engine.ts 在 Deno 端做的,見該檔案
      // `pushPcm` 的 why 註解)。
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(data);
      }
      return;
    }
    if (data && data.type === "flushed") {
      // 對應 stopRecording() 裡等待的那個 Promise——見下方 why 說明。
      if (pendingFlush) pendingFlush();
    }
  };

  sourceNode.connect(workletNode);

  socket.send(JSON.stringify({ type: "start" }));

  /** 讓 stopRecording() 可以要求 worklet 沖洗殘餘 chunk,並等到瀏覽器
   * 確認「已經沖洗完成」才繼續往下走(送 WS stop、關閉音訊圖)——見
   * downsampler-worklet.js 的 flush 訊息 why 說明:沒有這個等待,
   * 最後不到 100ms 的音訊可能在還沒送到 WS 之前,音訊圖就先被拆了。 */
  session.requestFlush = () => {
    return new Promise((resolve) => {
      pendingFlush = resolve;
      workletNode.port.postMessage({ type: "flush" });
    });
  };

  startButton.disabled = true;
  stopButton.disabled = false;
}

/** flush 交握的等待上限。why 需要上限:等待 `flushed` 確認的 Promise 只有
 * worklet 回話才會 resolve——若 worklet 因任何原因不再回應(audio thread
 * 被瀏覽器暫停、worklet 內部丟出未捕捉例外後停止處理 port 訊息等),沒有
 * 上限的等待會讓 stopRecording() 永遠卡住,開始/停止兩顆按鈕同時停在
 * disabled,UI 再也無法恢復。2 秒的選值:flush 只是把記憶體裡既有的
 * 殘餘樣本 post 回主執行緒(無 I/O、無運算),正常情況毫秒級完成,2 秒
 * 已是正常耗時的千倍以上——超時幾乎必然代表 worklet 真的死了,再等下去
 * 沒有意義;同時 2 秒也短到使用者不會覺得停止按鈕壞掉。 */
const FLUSH_TIMEOUT_MS = 2000;

async function stopRecording() {
  if (session === null) return;
  stopButton.disabled = true;

  if (session.requestFlush !== null) {
    // Promise.race 加上時限:flush 交握若在 FLUSH_TIMEOUT_MS 內沒有等到
    // worklet 的 flushed 確認,放棄等待、照常走完停止流程——寧可犧牲最後
    // 不到 100ms 的殘餘音訊,也不能讓 UI 永遠卡在兩顆按鈕都 disabled 的
    // 狀態(選值理由見 FLUSH_TIMEOUT_MS 的 why 註解)。
    const flushedInTime = await Promise.race([
      session.requestFlush().then(() => true),
      new Promise((resolve) => setTimeout(() => resolve(false), FLUSH_TIMEOUT_MS)),
    ]);
    if (!flushedInTime) {
      showError("停止錄音時未收到降採樣器的沖洗確認,尾端音訊可能未送出。");
    }
  }
  // 已知且接受的損失窗口:flushed 確認只保證「殘餘 chunk 已從 worklet
  // post 到主執行緒的佇列」,從確認送達到下面 teardownSession() 拆掉
  // 音訊圖之間,audio thread 可能又處理了一兩個 render quantum(約幾
  // 毫秒的新樣本),那些樣本會隨拆圖丟失——量級遠小於一個 100ms chunk,
  // 對逐字稿無感知影響,不值得為它再加一輪交握的複雜度。

  socket.send(JSON.stringify({ type: "stop" }));
  // final 事件由 handleServerMessage 非同步處理(伺服端 stop() 需要時間
  // 沖洗解碼器),這裡不等待——UI 維持在「已停止錄音」狀態,final 抵達時
  // 自然會出現在列表裡,不需要阻塞使用者按下一次「開始錄音」。

  teardownSession();
  startButton.disabled = false;
}

/** 釋放麥克風串流與音訊圖資源。無論是正常停止還是發生錯誤,都要確保麥克風
 * 指示燈(瀏覽器/系統層級的錄音中提示)會關掉——直接呼叫這個函式比在每個
 * 錯誤分支各自重寫一次清理邏輯更不容易漏掉。
 *
 * 每個欄位各自 null 檢查(而非只檢查 session 本身):session 在
 * startRecording() 裡是「邊取得資源邊填入」的部分填充物件(見該函式的
 * why 註解),失敗可能發生在任何一個 await 之後——這裡必須對「已經取得
 * 的那部分」照常回收,對「還沒取得的那部分」安靜跳過。這個回收路徑
 * 無法在 Deno 測試中覆蓋(getUserMedia/AudioContext 只存在於瀏覽器,
 * 而 downsampler-core_test.ts 刻意只測不碰 Web API 的純函式),由
 * Task 5 的 mac 實機演練人工驗證;這段註解本身就是給重建 agent 的
 * 語義錨點,說明「為什麼不能寫成一次性賦值 + 單一 null 檢查」。 */
function teardownSession() {
  if (session === null) return;
  if (session.sourceNode !== null) session.sourceNode.disconnect();
  if (session.workletNode !== null) session.workletNode.disconnect();
  session.stream.getTracks().forEach((track) => track.stop());
  if (session.audioContext !== null) {
    session.audioContext.close().catch(() => {
      // AudioContext 可能已經處於 closed 狀態(例如使用者連續快速按了
      // 兩次停止),close() 會 reject——這裡的語義是「確保關閉」,不是
      // 「必須原本開著」,忽略即可(對齊 engine.ts kill() 的既有慣例)。
    });
  }
  session = null;
}

// ---------------------------------------------------------------------------
// 啟動
// ---------------------------------------------------------------------------

// 頁面載入時按鈕先禁用,等 WS 的 `ready` 事件抵達(見 handleServerMessage
// 的 "ready" 分支)才開放「開始錄音」——避免使用者在 WS 都還沒連上引擎、
// backend 徽章都還沒有內容時就點下去,送出一個必然被 server 端拒絕
// (無會話)的 audio chunk。
startButton.disabled = true;
stopButton.disabled = true;
copyAllButton.disabled = true;
exportButton.disabled = true;
connectWebSocket();
