/**
 * 驗收測試:走完整的 HTTP + WebSocket 服務堆疊(真 engine host、真模型、
 * 真 `server.ts`)——對照 `protocol_test.ts` 只測 `EngineClient` 本身,這裡
 * 驗的是「使用者實際打開瀏覽器會用到的那個路徑」也正確接起來:
 * ready → start → binary chunk(fixture wav)→ partial → stop → final。
 *
 * 權限:同 `real_service.ts` 需要的一切(spawn engine + 綁 port)+
 * `--allow-net`(WS 用戶端連線;`verify:*` 的 `--allow-net` 已因
 * `binding_test.ts` 的需要放寬為不縮圈,見 SECURITY.md「verify 與 start
 * permission 差異」章節)。
 */
import { assert, assertEquals } from "jsr:@std/assert@^1.0.19";
import { fromFileUrl } from "jsr:@std/path@^1.0.9/from-file-url";
import { startRealService } from "./real_service.ts";
import { chunkPcm16, readWavPcm16 } from "./wav.ts";

const APP_DIR = Deno.cwd();
// why fromFileUrl:同 protocol_test.ts——裸 `.pathname` 在 Windows 會產生
// 非法原生路徑,Task 6 Windows 演練實測發現(見 reference/setup.ts 註解)。
const FIXTURE_PATH = fromFileUrl(new URL("./fixtures/hello-streaming.wav", import.meta.url));

function connectWs(port: number): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
    socket.onopen = () => resolve(socket);
    socket.onerror = (event) => reject(event);
  });
}

interface ServerMessage {
  type: string;
  [key: string]: unknown;
}

/** 收集 WS text frame 事件,`next()` 佇列優先、沒有才排隊等下一則——同
 * `reference/server_test.ts` 的 `createMessageCollector` 慣例,避免訊息
 * 比呼叫 `next()` 早到而漏接。 */
function collectMessages(socket: WebSocket) {
  const queue: ServerMessage[] = [];
  const waiters: Array<(msg: ServerMessage) => void> = [];
  socket.onmessage = (event) => {
    const parsed = JSON.parse(event.data as string) as ServerMessage;
    const waiter = waiters.shift();
    if (waiter) waiter(parsed);
    else queue.push(parsed);
  };
  return {
    next(): Promise<ServerMessage> {
      const queued = queue.shift();
      if (queued !== undefined) return Promise.resolve(queued);
      return new Promise((resolve) => waiters.push(resolve));
    },
  };
}

Deno.test("service：WS 全流程(真 host 真模型) ready → start → binary chunks → partial → stop → final", async () => {
  const service = await startRealService(APP_DIR);
  try {
    const socket = await connectWs(service.port);
    const messages = collectMessages(socket);
    try {
      const ready = await messages.next();
      assertEquals(ready.type, "ready");
      assert(typeof ready.backend === "string" && (ready.backend as string).length > 0);

      socket.send(JSON.stringify({ type: "start" }));

      // 背景收集 partial,直到看到 final/error 為止——partial 可能在還沒
      // 送完全部 chunk 前就已經開始出現(真模型會邊收邊解碼)。
      let sawPartial = false;
      const waitForFinal = (async (): Promise<ServerMessage> => {
        while (true) {
          const msg = await messages.next();
          if (msg.type === "partial") {
            sawPartial = true;
            continue;
          }
          return msg;
        }
      })();

      const pcm = await readWavPcm16(FIXTURE_PATH);
      for (const chunk of chunkPcm16(pcm)) {
        socket.send(chunk);
      }
      socket.send(JSON.stringify({ type: "stop" }));

      const final = await waitForFinal;
      if (final.type === "error") {
        throw new Error(`engine 回報 error(訊息:${String(final.message)})`);
      }
      assertEquals(final.type, "final");
      assert(
        typeof final.text === "string" && (final.text as string).length > 0,
        "final.text 不應為空字串",
      );
      assert(sawPartial, "應至少收到一次 partial(fixture 音訊約 4 秒，真模型應有中途輸出)");
    } finally {
      socket.close();
    }
  } finally {
    await service.shutdown();
  }
});
