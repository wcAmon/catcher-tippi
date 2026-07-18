/**
 * `EngineClient` 黑箱協定測試,對 mac 本地 build 的 `catcher-asr-host`
 * `--fake-engine` 模式跑。不測 MLX 真推論(那是 crates/nemotron-mlx 的責任),
 * 只測「EngineClient 是否正確實作 asr-host-v1 協定的 client 端」。
 *
 * 前置:先在本 worktree build 一次(binary 不進版控,`target/` 已
 * gitignore):
 *   cargo build --release -p catcher-asr-host
 *
 * FakeEngine 語義(見 crates/catcher-asr-host/src/engine.rs):每 1600 samples
 * 累積一個 token,以「字<序號>」解碼(例如第一個 token → "字0"、第二個 → 兩個
 * token 一起解成 "字0字1"),每次 `start` 會重置累積狀態。
 *
 * 執行本檔所需權限旗標(dev-time 測試,不是 manifest.json 的 runtime 權限
 * 集合——engine_test.ts 需要 `Deno.kill()` 模擬 host crash,該 API 需要
 * *unscoped* `--allow-run`,比正式 runtime 的 `--allow-run=<binPath>`
 * 更寬,故意不寫進 manifest;另外需要 `--allow-read` 讓
 * `assertBinaryBuilt()` 能在 binary 未 build 時給出清楚的錯誤訊息,而不是
 * 讓 spawn 失敗時的通用錯誤蓋掉「你忘了 cargo build」這個更有用的線索):
 *   deno test --allow-run --allow-read recipes/tomato-ears/reference/engine_test.ts
 */

import { assert, assertEquals, assertRejects } from "jsr:@std/assert@^1.0.19";
import { fromFileUrl } from "jsr:@std/path@^1.0.9/from-file-url";
import { EngineClient } from "./engine.ts";

/** 本 worktree build 出的 fake-engine 測試用 binary(見上方前置指示)。
 * why fromFileUrl(見 reference/setup.ts 的 MANIFEST_PATH 註解):裸
 * `new URL(...).pathname` 在 Windows 會產生 `/C:/...` 這種非法原生路徑，
 * 導致 Deno.stat/readFile 以 os error 3 失敗——Task 6 Windows 演練實測發現。 */
const FAKE_HOST_PATH = fromFileUrl(
  new URL("../../../target/release/catcher-asr-host", import.meta.url),
);

async function assertBinaryBuilt(): Promise<void> {
  try {
    await Deno.stat(FAKE_HOST_PATH);
  } catch {
    throw new Error(
      `找不到測試用 binary:${FAKE_HOST_PATH}\n` +
        `請先在本 worktree 執行:cargo build --release -p catcher-asr-host`,
    );
  }
}

/** 1600 samples 的靜音 PCM16-LE(FakeEngine 只看 sample 數,不看內容)。 */
function silentChunk(samples: number): Uint8Array {
  return new Uint8Array(samples * 2);
}

/** 建一個「下一次 onPartial 呼叫」的 one-shot waiter,呼叫前先掛上再送資料,
 * 避免資料送出與掛上 listener 之間的競態。 */
function nextPartial(client: EngineClient): { wait: Promise<string>; texts: string[] } {
  const texts: string[] = [];
  let resolveNext: ((text: string) => void) | undefined;
  const wait = new Promise<string>((resolve) => {
    resolveNext = resolve;
  });
  client.onPartial = (text) => {
    texts.push(text);
    resolveNext?.(text);
    resolveNext = undefined;
  };
  return { wait, texts };
}

Deno.test("EngineClient.spawn：等 ready，backend 為 fake", async () => {
  await assertBinaryBuilt();
  const client = await EngineClient.spawn(FAKE_HOST_PATH, ["--fake-engine"]);
  try {
    assertEquals(client.backend, "fake");
  } finally {
    client.kill();
  }
});

Deno.test("start → pushPcm(1600 samples) → onPartial 收到「字0」", async () => {
  await assertBinaryBuilt();
  const client = await EngineClient.spawn(FAKE_HOST_PATH, ["--fake-engine"]);
  try {
    client.start();
    const { wait } = nextPartial(client);
    client.pushPcm(silentChunk(1600));
    assertEquals(await wait, "字0");
  } finally {
    client.kill();
  }
});

Deno.test("stop() resolve final text（累積 1600 samples 後 stop）", async () => {
  await assertBinaryBuilt();
  const client = await EngineClient.spawn(FAKE_HOST_PATH, ["--fake-engine"]);
  try {
    client.start();
    const { wait } = nextPartial(client);
    client.pushPcm(silentChunk(1600));
    assertEquals(await wait, "字0");

    const final = await client.stop();
    assertEquals(final, "字0");
  } finally {
    client.kill();
  }
});

Deno.test("二會話：第二次 start 後 FakeEngine 重置，partial/final 重新從「字0」開始", async () => {
  await assertBinaryBuilt();
  const client = await EngineClient.spawn(FAKE_HOST_PATH, ["--fake-engine"]);
  try {
    for (let session = 0; session < 2; session++) {
      client.start();
      const { wait } = nextPartial(client);
      client.pushPcm(silentChunk(1600));
      assertEquals(await wait, "字0", `session ${session} 的 partial`);
      const final = await client.stop();
      assertEquals(final, "字0", `session ${session} 的 final`);
    }
  } finally {
    client.kill();
  }
});

Deno.test("host 被殺掉後：onError 收到通知，pending stop() 被 reject", async () => {
  await assertBinaryBuilt();
  const client = await EngineClient.spawn(FAKE_HOST_PATH, ["--fake-engine"]);

  let errorMessage: string | undefined;
  client.onError = (message) => {
    errorMessage = message;
  };

  client.start();
  const stopPromise = client.stop();

  // 直接對子行程送 SIGKILL,模擬非預期 crash(不是透過 client.kill(),
  // 那會把 #killedByUser 設成 true 而刻意壓下 crash 偵測路徑)。
  Deno.kill(client.pid, "SIGKILL");

  await assertRejects(() => stopPromise);
  // 讀取迴圈是背景 async 任務,onError 的呼叫可能比 stopPromise 的 reject
  // 稍晚一點點才排進事件佇列——用短暫的忙等避開這個排程時間差,而不是
  // 依賴固定的 sleep(避免 flaky)。
  for (let i = 0; i < 50 && errorMessage === undefined; i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert(errorMessage !== undefined, "host crash 後應呼叫 onError");
});

Deno.test("spawn()：binPath 不存在時 reject（而非讓例外逸散成未捕捉的 rejection）", async () => {
  await assertRejects(
    () => EngineClient.spawn("/definitely/not/a/real/path/catcher-asr-host", ["--fake-engine"]),
  );
});
