/**
 * tomato-ears 的 HTTP + WebSocket 服務:serve `${appDir}/ui` 底下的靜態錄音頁,
 * `/ws` 走一個簡單的文字/binary 混合協定把瀏覽器與 {@link EngineClient} 接起來。
 *
 * 設計原則(why):
 * - **僅綁 127.0.0.1**(店規第 5 條):`Deno.serve({ hostname: "127.0.0.1", ... })`
 *   讓作業系統只在 loopback 介面監聽,從同一台機器的其他網路介面(LAN IP、
 *   `0.0.0.0` 語意上的「所有介面」)連進來會直接被拒絕——不需要應用層額外
 *   判斷來源 IP 再擋,綁定範圍本身就是防線。
 * - **`EngineClient` 是單一訂閱者的 callback,不是 event emitter**(見
 *   `engine.ts` 的說明):本模組把 `engine.onPartial`/`engine.onError`
 *   在「目前使用中的那個 WS 連線」建立時重新指派過去。v1 只服務單一本機
 *   使用者,同時只會有一個「作用中」的錄音分頁;若有新分頁連上來搶佔,
 *   舊分頁會停止收到事件(合理的降級,而非串流錯亂)。
 * - **WS text frame 是控制訊息,binary frame 是音訊資料**:分開兩種 frame
 *   類型比「全部包 JSON、音訊用 base64 塞進 text」省頻寬,也讓
 *   `downsampler-worklet.js`(Task 3)可以直接把 transferable 的
 *   `Int16Array.buffer` post 過來,不必先轉字串。
 */

import { EngineClient } from "./engine.ts";

/** 靜態檔案的副檔名 → Content-Type 對照表;僅涵蓋 `ui/` 目錄實際會用到的類型。 */
const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wav": "audio/wav",
  ".ico": "image/x-icon",
};

/** WS → server 的控制訊息(text frame,JSON)。 */
interface ClientMessage {
  type: "start" | "stop";
  /** 保留欄位,對齊協定的 start.lang;v1 UI 不送,預設由 EngineClient 帶入 "auto"。 */
  lang?: string;
}

/** server → WS 的事件訊息(text frame,JSON)。 */
type ServerMessage =
  | { type: "ready"; backend: string }
  | { type: "partial"; text: string }
  | { type: "final"; text: string }
  | { type: "error"; message: string };

/**
 * 啟動服務:靜態檔案 serve `${appDir}/ui`,`/ws` 提供錄音用的 WebSocket。
 *
 * `engine` 必須已經是 `spawn()` 完成、`ready` 狀態的 {@link EngineClient}
 * ——本函式不負責啟動/重啟引擎,那是 `main.ts` 的職責(讓「引擎生命週期」
 * 與「HTTP/WS 服務」保持關注點分離,方便各自測試——本檔的測試就是用假的
 * `EngineClient` stub 跑,完全不需要真的 spawn 子行程)。
 */
export function startServer(
  appDir: string,
  engine: EngineClient,
  port: number,
): Deno.HttpServer {
  const uiDir = `${appDir}/ui`;

  return Deno.serve(
    { hostname: "127.0.0.1", port, onListen: () => {} },
    (request) => {
      const url = new URL(request.url);
      if (url.pathname === "/ws") {
        return handleWebSocketUpgrade(request, engine);
      }
      return serveStatic(uiDir, url.pathname);
    },
  );
}

/** 把 request 升級成 WebSocket,並把音訊/控制事件接到 `engine`。 */
function handleWebSocketUpgrade(request: Request, engine: EngineClient): Response {
  if (request.headers.get("upgrade") !== "websocket") {
    return new Response("此路徑僅接受 WebSocket 升級請求", { status: 426 });
  }
  const { socket, response } = Deno.upgradeWebSocket(request);

  socket.onopen = () => {
    // 把這個引擎的事件接到「當下這個」連線——見檔頭 why 說明:同時只服務
    // 一個作用中連線,新連線會蓋掉舊連線的訂閱。
    engine.onPartial = (text) => sendMessage(socket, { type: "partial", text });
    engine.onError = (message) => sendMessage(socket, { type: "error", message });
    // 連線一建立就回報目前的 backend,UI 不需要另外問——對齊 Task 3
    // 「backend 徽章」的需求(顯示 ready 的 backend 值,僅展示不分支)。
    sendMessage(socket, { type: "ready", backend: engine.backend });
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleControlMessage(socket, engine, event.data);
      return;
    }
    // binary frame:錄音頁的 AudioWorklet 送出的 PCM16 chunk。
    const chunk = event.data instanceof ArrayBuffer
      ? new Uint8Array(event.data)
      : toUint8Array(event.data as Blob | ArrayBufferView);
    handleBinaryChunk(engine, chunk);
  };

  return response;
}

/** 把 WS message event 的資料統一成 `Uint8Array`(涵蓋 Blob/ArrayBufferView 兩種可能形態)。 */
function toUint8Array(data: Blob | ArrayBufferView): Uint8Array {
  if (data instanceof Blob) {
    // Deno 的 WebSocket 實作對 binary frame 一律回傳 ArrayBuffer,不會是
    // Blob;這個分支只是為了型別完整性與未來若換 runtime 時的保險。
    throw new Error("非預期的 Blob binary frame(預期 ArrayBuffer)");
  }
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function handleControlMessage(socket: WebSocket, engine: EngineClient, raw: string): void {
  let message: ClientMessage;
  try {
    message = JSON.parse(raw);
  } catch {
    sendMessage(socket, { type: "error", message: `無法解析的控制訊息:${raw}` });
    return;
  }
  switch (message.type) {
    case "start":
      engine.start(message.lang);
      break;
    case "stop":
      // stop() 回傳 Promise<string>;成功回推 final,失敗(協定:沖洗/解碼
      // 失敗)回推 error——兩者都要讓 UI 知道會話已經結束。
      engine.stop()
        .then((text) => sendMessage(socket, { type: "final", text }))
        .catch((err) =>
          sendMessage(socket, {
            type: "error",
            message: err instanceof Error ? err.message : String(err),
          })
        );
      break;
    default:
      sendMessage(socket, { type: "error", message: `未知的控制訊息類型:${raw}` });
  }
}

function handleBinaryChunk(engine: EngineClient, chunk: Uint8Array): void {
  engine.pushPcm(chunk);
}

/** 送一則 JSON 事件;socket 已關閉時 `send` 會拋錯,吞掉即可(競態:引擎的
 * 非同步回呼可能在使用者關分頁之後才觸發)。 */
function sendMessage(socket: WebSocket, message: ServerMessage): void {
  if (socket.readyState !== WebSocket.OPEN) return;
  try {
    socket.send(JSON.stringify(message));
  } catch {
    // 見上方註解:關閉競態,忽略。
  }
}

/**
 * 提供 `${uiDir}` 底下的靜態檔案。`/` 對映到 `index.html`;
 * 任何解析後跳出 `uiDir` 範圍的路徑(例如 `..` 穿越)一律 404,不嘗試
 * 修正或部分接受——寧可讓使用者看到「找不到頁面」,也不要意外把
 * `uiDir` 以外的檔案系統內容 serve 出去。
 */
async function serveStatic(uiDir: string, pathname: string): Promise<Response> {
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  const segments = relative.split("/");
  if (segments.some((segment) => segment === "" || segment === "..")) {
    return new Response("Not Found", { status: 404 });
  }

  const filePath = `${uiDir}/${relative}`;
  let file: Deno.FsFile;
  try {
    file = await Deno.open(filePath, { read: true });
  } catch {
    return new Response("Not Found", { status: 404 });
  }

  const ext = relative.slice(relative.lastIndexOf("."));
  const contentType = CONTENT_TYPES[ext] ?? "application/octet-stream";
  return new Response(file.readable, { headers: { "content-type": contentType } });
}
