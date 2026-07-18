/**
 * 起「真實已安裝」服務的共用 helper——真 engine host(真模型)+ 真
 * `server.ts`,供 `service_test.ts`/`binding_test.ts` 共用,避免兩份測試
 * 各自重複「讀 machine-profile → 決定平台 → spawn engine → startServer」
 * 這段組裝邏輯(與 `main.ts` 的 `run()` 同構,但 verify 需要能拿到
 * server/engine 的控制代碼做斷言與收尾,`main.ts` 的版本只負責跑到底、
 * 不回傳)。
 *
 * why 不直接 spawn `deno task start:mac` 當子行程走黑箱測試(同
 * `reference/permissions_probe_test.ts` 的 literal-task 測試手法):
 * 那種手法驗證的是「宣告旗標本身是否足以跑起服務」,`verify/` 的定位不同
 * ——這裡假設 setup 已完成、旗標已經對(那件事由
 * `reference/permissions_probe_test.ts` 在 dev-time 顧,`verify/
 * permissions_test.ts` 顧「deno.json 與 manifest.json 是否同步」),
 * `service_test.ts`/`binding_test.ts` 要驗的是「真的餵真音訊給真模型,
 * 端到端服務行為是否正確」,直接 in-process 組裝更直接也更容易在測試裡
 * 拿到 server/engine 的控制代碼做細部斷言。
 */
import {
  buildEngineArgs,
  DEFAULT_PORT,
  platformFromProfile,
  readMachineProfile,
  resolveMachineProfilePath,
} from "../reference/main.ts";
import { stableEngineBinaryPath } from "../reference/downloader.ts";
import { EngineClient } from "../reference/engine.ts";
import { startServer } from "../reference/server.ts";

export interface RealService {
  server: Deno.HttpServer;
  engine: EngineClient;
  port: number;
  /** 先關 HTTP server(不再接受新連線),再 kill engine 子行程。 */
  shutdown(): Promise<void>;
}

/**
 * 組裝一個真實可用的服務:讀 machine-profile 決定平台、spawn 已安裝的
 * engine host(真模型)、起 `server.ts`。`appDir` 必須已經是「setup 完成」
 * 狀態(`bin/engine-host[.exe]` 與 `model/` 都在),否則 `EngineClient.spawn`
 * 會直接 reject。
 */
export async function startRealService(appDir: string): Promise<RealService> {
  const profile = await readMachineProfile(resolveMachineProfilePath());
  const platform = platformFromProfile(profile);
  const binPath = stableEngineBinaryPath(appDir, platform);
  const args = buildEngineArgs(platform, `${appDir}/model`, profile);

  const engine = await EngineClient.spawn(binPath, args);
  const server = startServer(appDir, engine, DEFAULT_PORT);

  return {
    server,
    engine,
    port: DEFAULT_PORT,
    async shutdown() {
      await server.shutdown();
      engine.kill();
    },
  };
}
