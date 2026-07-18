/**
 * `server.ts` 的測試:靜態檔案 serve、WS 全流程(ready/start/audio/stop/
 * final/error)、以及「僅綁 127.0.0.1」的綁定檢查。
 *
 * 全程用一個手寫的假 `EngineClient`(見 `createStubEngine`),不 spawn 任何
 * 真的子行程——`EngineClient` 用了 ECMAScript `#private` 欄位,TypeScript
 * 對這種欄位是 nominal typing(帶隱藏品牌),手寫的 stub 物件字面量無法
 * 結構相容,因此傳給 `startServer` 前用 `as unknown as EngineClient` 明確
 * 斷言型別——這是刻意的測試手法,不代表 production 程式碼會這樣用
 * (production 一定是傳真正 `EngineClient.spawn()` 的結果)。
 *
 * 執行本檔所需權限旗標(dev-time 測試):
 *   deno test --allow-net --allow-read --allow-write --allow-sys=networkInterfaces \
 *     recipes/tomato-ears/reference/server_test.ts
 * - `--allow-net`:啟動測試用 server(127.0.0.1 任意埠)、WS 用戶端連線、
 *   binding 測試連到 LAN 位址。
 * - `--allow-read`/`--allow-write`:建立/讀取臨時 `ui/` 目錄(靜態檔測試)。
 * - `--allow-sys=networkInterfaces`:binding 測試需要列出本機網卡位址,
 *   找一個非 loopback 的 IPv4 位址來驗證「連不上」。
 */

import { assertEquals, assertRejects } from "jsr:@std/assert@^1.0.19";
import type { EngineClient } from "./engine.ts";
import { startServer } from "./server.ts";

/** 手寫的假 EngineClient:記錄呼叫、讓測試能手動控制 stop() 何時 resolve/reject。 */
interface StubEngine {
  backend: string;
  onPartial: (text: string) => void;
  onError: (message: string) => void;
  start: (lang?: string) => void;
  pushPcm: (chunk: Uint8Array) => void;
  stop: () => Promise<string>;
  kill: () => void;
}

interface StubEngineController {
  /** 傳給 `startServer` 用的斷言型別版本(見檔頭說明)。 */
  asEngineClient: EngineClient;
  /** 原始物件參照,測試用來直接呼叫 `onPartial`/`onError`
   * (server.ts 會在連線建立時把這兩個欄位重新指派過)。 */
  raw: StubEngine;
  startCalls: Array<string | undefined>;
  pushPcmCalls: Uint8Array[];
  killCalled: { value: boolean };
  /** `stop()` 實際被呼叫的次數——`resolveStop`/`rejectStop` 只對「最新一次」
   * 呼叫生效,呼叫端必須先用 `waitUntil(() => stopCalls.count === N)`
   * 確認伺服端已經真的呼叫過 `engine.stop()`,才能安全地觸發 resolve/reject
   * (見下方測試裡的競態說明:WS 控制訊息是非同步送達的,不能假設
   * `socket.send()` 之後伺服端就立刻處理完畢)。
   */
  stopCalls: { count: number };
  resolveStop: (text: string) => void;
  rejectStop: (err: Error) => void;
}

function createStubEngine(backend = "fake"): StubEngineController {
  const startCalls: Array<string | undefined> = [];
  const pushPcmCalls: Uint8Array[] = [];
  const killCalled = { value: false };
  const stopCalls = { count: 0 };
  let stopResolve: (text: string) => void = () => {};
  let stopReject: (err: Error) => void = () => {};

  const raw: StubEngine = {
    backend,
    onPartial: () => {},
    onError: () => {},
    start(lang) {
      startCalls.push(lang);
    },
    pushPcm(chunk) {
      pushPcmCalls.push(chunk);
    },
    stop() {
      return new Promise<string>((resolve, reject) => {
        stopResolve = resolve;
        stopReject = reject;
        stopCalls.count++;
      });
    },
    kill() {
      killCalled.value = true;
    },
  };

  return {
    asEngineClient: raw as unknown as EngineClient,
    raw,
    startCalls,
    pushPcmCalls,
    killCalled,
    stopCalls,
    resolveStop: (text) => stopResolve(text),
    rejectStop: (err) => stopReject(err),
  };
}

/** FIFO 訊息收集器:`onmessage` 到達時若已經有人在等,直接餵給等待者;
 * 否則先放進佇列。`next()` 反過來:佇列有貨先吃佇列,沒有才排隊等下一則。
 * 這個順序保證測試不會因為「訊息比呼叫 next() 早到」而漏接(競態)。 */
function createMessageCollector(socket: WebSocket) {
  const queue: unknown[] = [];
  const waiters: Array<(msg: unknown) => void> = [];
  socket.onmessage = (event) => {
    const parsed = JSON.parse(event.data as string);
    const waiter = waiters.shift();
    if (waiter) waiter(parsed);
    else queue.push(parsed);
  };
  return {
    next(): Promise<unknown> {
      const queued = queue.shift();
      if (queued !== undefined) return Promise.resolve(queued);
      return new Promise((resolve) => waiters.push(resolve));
    },
  };
}

/** `Deno.HttpServer.addr` 的型別是 `Deno.Addr`(涵蓋 Unix socket 的聯集),
 * 但 `startServer` 內部一定是用 `{ hostname, port }` 呼叫 `Deno.serve`,
 * 執行期一定是 `Deno.NetAddr`——用小斷言收斂型別,避免每個呼叫點都要重複寫。 */
function portOf(server: Deno.HttpServer): number {
  return (server.addr as Deno.NetAddr).port;
}

/**
 * 建一個臨時 appDir,底下含 `ui/` 子目錄與指定的靜態檔——`startServer` 的
 * 第一個參數是 appDir(它內部固定 serve `${appDir}/ui`,見 server.ts),
 * 不是 ui 目錄本身,回傳值對齊這個約定,呼叫端才不會傳錯層級。
 */
async function makeTempAppDir(uiFiles: Record<string, string>): Promise<string> {
  const dir = await Deno.makeTempDir({ prefix: "tomato-ears-app-" });
  await Deno.mkdir(`${dir}/ui`, { recursive: true });
  for (const [name, content] of Object.entries(uiFiles)) {
    await Deno.writeTextFile(`${dir}/ui/${name}`, content);
  }
  return dir;
}

async function connectWs(
  port: number,
): Promise<{ socket: WebSocket; collector: ReturnType<typeof createMessageCollector> }> {
  const socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  // 收集器必須在 await open 之前同步建立(見上方 why 註解),避免漏接
  // server 在 socket.onopen 裡立刻送出的 ready 事件。
  const collector = createMessageCollector(socket);
  await new Promise<void>((resolve, reject) => {
    socket.onopen = () => resolve();
    socket.onerror = (event) => reject(event);
  });
  return { socket, collector };
}

async function closeWs(socket: WebSocket): Promise<void> {
  if (socket.readyState === WebSocket.CLOSED) return;
  await new Promise<void>((resolve) => {
    socket.onclose = () => resolve();
    socket.close();
  });
}

Deno.test("靜態檔案:GET / 回傳 index.html，未知路徑 404，路徑穿越被拒", async () => {
  const uiDir = await makeTempAppDir({ "index.html": "<h1>tomato-ears</h1>" });
  const stub = createStubEngine();
  const server = startServer(uiDir, stub.asEngineClient, 0);
  try {
    const root = await fetch(`http://127.0.0.1:${portOf(server)}/`);
    assertEquals(root.status, 200);
    assertEquals(await root.text(), "<h1>tomato-ears</h1>");

    const missing = await fetch(`http://127.0.0.1:${portOf(server)}/no-such-file.js`);
    assertEquals(missing.status, 404);
    await missing.body?.cancel();

    const traversal = await fetch(`http://127.0.0.1:${portOf(server)}/../../etc/passwd`);
    // 瀏覽器/fetch 通常會先正規化掉 `..`;無論正規化後落在哪裡,
    // serveStatic 對任何解析出 `..` 片段或落在 uiDir 外的路徑一律 404。
    assertEquals(traversal.status, 404);
    await traversal.body?.cancel();
  } finally {
    await server.shutdown();
    await Deno.remove(uiDir, { recursive: true });
  }
});

/** `reference/` 目錄本身的路徑(即 `server_test.ts` 所在目錄)——真正的
 * `ui/`(Task 3 產出的 index.html/app.js/downsampler-worklet.js/style.css)
 * 就在它底下,對齊 server.ts 的 `${appDir}/ui` 慣例,不需要另外組一個
 * fixture 目錄。用 `import.meta.url` 取路徑而非寫死相對於 cwd 的字串,
 * 是延續 `engine_test.ts`/`permissions_probe_test.ts` 既有的慣例——
 * `deno test` 的 cwd 依呼叫方式不同可能是 repo 根目錄或
 * `recipes/tomato-ears/`,寫死相對路徑會因呼叫方式不同而找錯檔案。 */
const REAL_REFERENCE_DIR = new URL(".", import.meta.url).pathname;

Deno.test("靜態檔案:真正的 ui/ 四個檔案(index.html/app.js/downsampler-worklet.js/style.css)皆 200 且 Content-Type 正確", async () => {
  // 跟上一個測試不同:上一個測試用 makeTempAppDir 造的 fixture 只驗證
  // serveStatic 的路徑解析邏輯本身;這個測試改指向 Task 3 真正產出的
  // ui/ 目錄,確保「瀏覽器實際會拿到的檔案」也真的 200——避免 fixture
  // 測試通過、但真檔案缺漏或命名對不上的情況被漏掉。
  const stub = createStubEngine();
  const server = startServer(REAL_REFERENCE_DIR, stub.asEngineClient, 0);
  try {
    const expectations: Array<[path: string, contentTypePrefix: string]> = [
      ["/", "text/html"],
      ["/app.js", "text/javascript"],
      ["/downsampler-worklet.js", "text/javascript"],
      ["/style.css", "text/css"],
    ];
    for (const [path, contentTypePrefix] of expectations) {
      const response = await fetch(`http://127.0.0.1:${portOf(server)}${path}`);
      assertEquals(response.status, 200, `${path} 應回傳 200`);
      const contentType = response.headers.get("content-type") ?? "";
      assertEquals(
        contentType.startsWith(contentTypePrefix),
        true,
        `${path} 的 Content-Type 應以 "${contentTypePrefix}" 開頭,實際是 "${contentType}"`,
      );
      await response.body?.cancel();
    }

    // downsampler-core.js 不是瀏覽器直接載入的檔案(它是被 downsampler-
    // worklet.js `import` 進去的模組,而非 `<script>`/`addModule` 的直接
    // 進入點),但既然它也放在 ui/ 目錄底下,靜態 serve 邏輯本來就會
    // serve 它——AudioWorklet 的模組載入器會發一個真正的 HTTP GET 抓它,
    // 一併驗證它也是 200,避免這個間接依賴被漏測。
    const coreModule = await fetch(`http://127.0.0.1:${portOf(server)}/downsampler-core.js`);
    assertEquals(coreModule.status, 200);
    assertEquals(coreModule.headers.get("content-type")?.startsWith("text/javascript"), true);
    await coreModule.body?.cancel();
  } finally {
    await server.shutdown();
  }
});

Deno.test("WS 全流程:ready → start → binary chunk → partial → stop → final", async () => {
  const uiDir = await makeTempAppDir({ "index.html": "ok" });
  const stub = createStubEngine("fake");
  const server = startServer(uiDir, stub.asEngineClient, 0);
  const { socket, collector } = await connectWs(portOf(server));
  try {
    assertEquals(await collector.next(), { type: "ready", backend: "fake" });

    socket.send(JSON.stringify({ type: "start" }));
    // 控制訊息透過網路送達,不能假設 socket.send() 一回來伺服端就處理完了
    // ——等到「可觀察的副作用」(stub 記錄的呼叫)出現才算數。
    await waitUntil(() => stub.startCalls.length === 1);
    assertEquals(stub.startCalls, [undefined]);

    const chunk = new Uint8Array([1, 2, 3, 4]);
    socket.send(chunk.buffer);
    await waitUntil(() => stub.pushPcmCalls.length === 1);
    assertEquals(stub.pushPcmCalls[0], chunk);

    // 模擬引擎非同步吐出 partial:直接呼叫 server.ts 在 socket.onopen 時
    // 重新指派過的 onPartial(raw.onPartial 這時已經不是初始的空函式)。
    stub.raw.onPartial("你好");
    assertEquals(await collector.next(), { type: "partial", text: "你好" });

    socket.send(JSON.stringify({ type: "stop" }));
    // 同樣的競態:必須等 stub.stop() 真的被呼叫過(stopCalls.count 增加)
    // 才能呼叫 resolveStop(),否則 resolveStop() 可能作用在「舊的、還沒被
    // 換成 pending 狀態」的 no-op 上,讓真正的那次 stop() Promise 永遠掛著
    // ——這正是本測試檔開發過程中實際踩到的 bug(見 task-2-report.md)。
    await waitUntil(() => stub.stopCalls.count === 1);
    stub.resolveStop("最終文字");
    assertEquals(await collector.next(), { type: "final", text: "最終文字" });
  } finally {
    await closeWs(socket);
    await server.shutdown();
    await Deno.remove(uiDir, { recursive: true });
  }
});

Deno.test("WS：stop() reject 時回推 error（協定:沖洗/解碼失敗，不會有 final）", async () => {
  const uiDir = await makeTempAppDir({ "index.html": "ok" });
  const stub = createStubEngine("fake");
  const server = startServer(uiDir, stub.asEngineClient, 0);
  const { socket, collector } = await connectWs(portOf(server));
  try {
    assertEquals(await collector.next(), { type: "ready", backend: "fake" });
    socket.send(JSON.stringify({ type: "stop" }));
    await waitUntil(() => stub.stopCalls.count === 1); // 見上一個測試的 why 註解
    stub.rejectStop(new Error("解碼失敗"));
    assertEquals(await collector.next(), { type: "error", message: "解碼失敗" });
  } finally {
    await closeWs(socket);
    await server.shutdown();
    await Deno.remove(uiDir, { recursive: true });
  }
});

Deno.test("WS：engine.onError 直接觸發（例如 host crash）時回推 error", async () => {
  const uiDir = await makeTempAppDir({ "index.html": "ok" });
  const stub = createStubEngine("fake");
  const server = startServer(uiDir, stub.asEngineClient, 0);
  const { socket, collector } = await connectWs(portOf(server));
  try {
    assertEquals(await collector.next(), { type: "ready", backend: "fake" });
    stub.raw.onError("engine host 已非預期結束");
    assertEquals(await collector.next(), { type: "error", message: "engine host 已非預期結束" });
  } finally {
    await closeWs(socket);
    await server.shutdown();
    await Deno.remove(uiDir, { recursive: true });
  }
});

Deno.test("WS：無法解析的控制訊息回推 error，不中斷連線", async () => {
  const uiDir = await makeTempAppDir({ "index.html": "ok" });
  const stub = createStubEngine("fake");
  const server = startServer(uiDir, stub.asEngineClient, 0);
  const { socket, collector } = await connectWs(portOf(server));
  try {
    assertEquals(await collector.next(), { type: "ready", backend: "fake" });
    socket.send("not json");
    const errorMessage = await collector.next() as { type: string; message: string };
    assertEquals(errorMessage.type, "error");

    // 連線仍活著:接下來的合法控制訊息照常運作。
    socket.send(JSON.stringify({ type: "start" }));
    await waitUntil(() => stub.startCalls.length === 1);
  } finally {
    await closeWs(socket);
    await server.shutdown();
    await Deno.remove(uiDir, { recursive: true });
  }
});

/** 找一個非 loopback 的本機 IPv4 位址,供綁定測試使用。找不到就回傳
 * undefined(例如 CI 沙箱沒有一般網卡),對應測試會自動 `ignore`。 */
function findLanIPv4(): string | undefined {
  const candidates = Deno.networkInterfaces().filter(
    (iface) => iface.family === "IPv4" && iface.address !== "127.0.0.1",
  );
  return candidates[0]?.address;
}

const lanIPv4 = findLanIPv4();

Deno.test({
  name: "服務只綁 127.0.0.1:透過本機 LAN 介面位址連線應被拒絕",
  ignore: lanIPv4 === undefined,
  fn: async () => {
    const uiDir = await makeTempAppDir({ "index.html": "ok" });
    const stub = createStubEngine();
    const server = startServer(uiDir, stub.asEngineClient, 0);
    try {
      // 同一個埠號,換一個 hostname(本機的 LAN 位址而非 127.0.0.1)——
      // 因為 Deno.serve 只在 127.0.0.1 監聽,這個位址上沒有任何行程在聽
      // 這個埠,連線必須被拒絕(ECONNREFUSED)。
      await assertRejects(() => Deno.connect({ hostname: lanIPv4!, port: portOf(server) }));
    } finally {
      await server.shutdown();
      await Deno.remove(uiDir, { recursive: true });
    }
  },
});

/** 忙等直到 predicate 為真或逾時(1 秒);用短間隔輪詢取代固定 sleep,
 * 避免測試在慢機器上 flaky,也避免正常情況下浪費時間。 */
async function waitUntil(predicate: () => boolean, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error("waitUntil：等待逾時");
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
