/**
 * tomato-ears 的 CLI 進入點(`deno task start`):讀 machine-profile 決定平台
 * 與 engine 啟動旗標、確認 setup 已完成、spawn engine host、起 HTTP/WS 服務、
 * 嘗試自動開瀏覽器。
 *
 * 設計原則(why):
 * - **純邏輯與 I/O 分離**(同 `recipes/env-base/probe/machine-profile.ts`
 *   的既有慣例):`platformFromProfile`/`buildEngineArgs`/`isSetupComplete`
 *   等函式可以獨立單元測試,不必真的 spawn 子行程或起 HTTP 服務;
 *   只有檔案最底部的 `if (import.meta.main)` 區塊(未被測試覆蓋,如同
 *   `machine-profile.ts` 的 `if (import.meta.main)` 一樣)做真正的組裝。
 * - **不從 `recipes/env-base/` import 任何 TS 模組**:配方在店規設計裡是
 *   使用者的 agent 各自複製、獨立組裝的單位(見 Task 4 的 PLAN.md——
 *   agent 只複製 `recipes/tomato-ears/reference/` 到 `~/tmuh-apps/tomato-ears/`,
 *   不會把整個 monorepo 帶過去)。因此本檔跟 `machine-profile.ts` 之間
 *   唯一的耦合是「讀同一份 machine-profile.json 檔案」,不是程式碼 import
 *   ——`homeDir()` 因此在這裡重新寫一份(而非共用),是刻意的配方獨立性
 *   取捨,不是疏忽。
 * - **`TMUH_APPS_DIR` 覆寫整個 `~/tmuh-apps` 根目錄**(不只是
 *   `tomato-ears/` 子目錄),因為 machine-profile.json 也在同一棵樹下
 *   (`_machine/`)——這是 Task 5 mac 演練要用乾淨 `TMUH_HOME` 模擬全新安裝
 *   的前提,見 plan Task 5 段落。
 * - **開瀏覽器允許失敗**:`--allow-run=open`(mac)/`--allow-run=rundll32`
 *   (Windows,用 `rundll32 url.dll,FileProtocolHandler` 而非整個
 *   `--allow-run=cmd`,縮小可執行檔的授權範圍)兩者都可能因為使用者環境
 *   (無 GUI、SSH 連線、企業限制)而失敗——失敗不影響服務本身,只是退化成
 *   印出 URL 讓使用者手動貼到瀏覽器。
 */

// 注意:main.ts 刻意不 import `ensureDependencies`——`deno task setup` 是
// 獨立的 deno task(獨立的權限集合,含 --allow-net 對外下載),`deno task
// start` 的權限刻意不含對外網路(見檔頭 why:兩階段權限模型)。setup 是否
// 完成用 `isSetupComplete()`(純存在性檢查)判斷,不會在 start 階段觸發
// 任何下載或雜湊驗證。
import { type Platform, resolveEngineBinaryPath } from "./downloader.ts";
import { EngineClient } from "./engine.ts";
import { startServer } from "./server.ts";

/** manifest.json 的 `ports.http`(見 recipes/tomato-ears/manifest.json)。
 * 兩處必須手動保持同步——這是 glue 腳本的固有侷限,Task 4 的
 * `permissions_test.ts` 家族之後可以加一項「manifest.ports.http ===
 * 這個常數」的一致性檢查,本 task 範圍不含那個測試。 */
const DEFAULT_PORT = 43117;

/** 解析使用者家目錄:mac/Linux 用 `HOME`,Windows 用 `USERPROFILE`。
 * 與 `recipes/env-base/probe/machine-profile.ts` 的同名函式邏輯一致,
 * 刻意重複實作而非 import——理由見檔頭 why 說明(配方獨立性)。 */
function homeDir(): string {
  const home = Deno.build.os === "windows" ? Deno.env.get("USERPROFILE") : Deno.env.get("HOME");
  if (!home) {
    throw new Error("無法解析家目錄(HOME/USERPROFILE 環境變數未設定)");
  }
  return home;
}

/**
 * `~/tmuh-apps` 根目錄,可用 `TMUH_APPS_DIR` 環境變數整棵覆寫
 * (Task 5 乾淨環境演練、或使用者想換安裝位置時使用)。
 */
export function resolveTmuhAppsRoot(): string {
  return Deno.env.get("TMUH_APPS_DIR") ?? `${homeDir()}/tmuh-apps`;
}

/** 本 app 的安裝目錄(`ensureDependencies`/`resolveEngineBinaryPath` 的 `appDir`)。 */
export function resolveAppDir(): string {
  return `${resolveTmuhAppsRoot()}/tomato-ears`;
}

/** env-base 產出的 machine-profile.json 路徑(跨 app 共用,不在 app 自己的目錄下)。 */
export function resolveMachineProfilePath(): string {
  return `${resolveTmuhAppsRoot()}/_machine/machine-profile.json`;
}

/**
 * 讀取並解析 machine-profile.json。檔案不存在時丟出清楚的提示訊息
 * (指向 env-base 配方),而不是讓使用者看到原始的 `NotFound` 堆疊。
 */
export async function readMachineProfile(path: string): Promise<Record<string, unknown>> {
  let text: string;
  try {
    text = await Deno.readTextFile(path);
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) {
      throw new Error(
        `找不到 machine-profile:${path}\n` +
          `請先完成 env-base 配方(recipes/env-base/RECIPE.md)再回來安裝 tomato-ears。`,
      );
    }
    throw err;
  }
  const parsed: unknown = JSON.parse(text);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`machine-profile 內容不是合法的 JSON 物件:${path}`);
  }
  return parsed as Record<string, unknown>;
}

/**
 * 依 machine-profile 的 `os`/`arch` 決定 tomato-ears 支援的目標平台。
 * 不支援的組合(例如 Linux、Intel mac)直接 throw——tomato-ears 的
 * 引擎相依只 pin 了 macOS arm64 與 Windows x64 兩份 prebuilt binary
 * (見 manifest.json 的 `dependencies.engine`),沒有第三種可用的引擎。
 */
export function platformFromProfile(
  profile: { os?: unknown; arch?: unknown },
): Platform {
  if (profile.os === "darwin" && profile.arch === "aarch64") {
    return "macos-arm64";
  }
  if (profile.os === "windows" && (profile.arch === "x86_64" || profile.arch === "x64")) {
    return "windows-x64";
  }
  throw new Error(
    `不支援的平台組合:os=${String(profile.os)} arch=${String(profile.arch)}` +
      `(tomato-ears 目前只支援 macOS arm64 與 Windows x64)`,
  );
}

/**
 * 從 machine-profile 的 `inference` 欄位讀出本 app 先前回填的推論後端探測
 * 結果。結構取 `{ "tomato-ears": { "backend": "cpu" } }`——鍵是 app 名稱,
 * 因為 machine-profile.json 是跨 app 共用檔案(見
 * `recipes/env-base/probe/machine-profile.ts` 的冪等 merge 設計:每個 app
 * 各自管自己的鍵,互不覆寫),值目前只需要 `backend` 這一個欄位。
 *
 * why 只有 Windows 會用到:mac host 固定用 MLX(無「探測後端」這回事);
 * Windows host 依店規第 6 節做「實測探測 + 基準測量」決定 DirectML 或
 * CPU,結果應該只測一次、記住,不必每次啟動都重新探測(那個探測本身要
 * 載入模型跑基準,不便宜)。回填動作(把探測結果寫回 machine-profile)不在
 * 本 task 範圍內(那是 Windows host 自己的 BackendProber 邏輯,見
 * `apps/nemotron-asr-host` 的既有實作);main.ts 只負責「讀」。
 */
export function readBackfilledBackend(profile: Record<string, unknown>): string | undefined {
  const inference = profile.inference;
  if (!inference || typeof inference !== "object") return undefined;
  const entry = (inference as Record<string, unknown>)["tomato-ears"];
  if (!entry || typeof entry !== "object") return undefined;
  const backend = (entry as Record<string, unknown>).backend;
  return typeof backend === "string" ? backend : undefined;
}

/**
 * 組出 spawn engine host 用的命令列引數。
 *
 * 平台分支只在這一個函式裡(對齊 engine.ts 檔頭 why 說明的設計原則:
 * `EngineClient` 本身完全平台無關):
 * - 兩平台共通:`--model <modelDir> --language auto`;
 * - Windows 專屬:若 machine-profile 已經回填過 backend 探測結果,加上
 *   `--backend <值>` 跳過重新探測(見 `readBackfilledBackend` 的 why)。
 *   mac host 沒有 `--backend` 這個旗標(它只認 MLX),不能誤加。
 */
export function buildEngineArgs(
  platform: Platform,
  modelDir: string,
  profile: Record<string, unknown>,
): string[] {
  const args = ["--model", modelDir, "--language", "auto"];
  if (platform === "windows-x64") {
    const backend = readBackfilledBackend(profile);
    if (backend !== undefined) {
      args.push("--backend", backend);
    }
  }
  return args;
}

/**
 * 檢查 `deno task setup` 是否已經完成:引擎執行檔解得出來,且模型目錄
 * 底下至少有一個檔案。
 *
 * why 不驗雜湊:完整的雜湊驗證是 `verify/integrity_test.ts`(Task 4)的
 * 職責——那是使用者主動要求的驗收步驟,可以接受花比較久的時間逐檔算
 * SHA-256。`main.ts` 每次啟動都會呼叫這個檢查,只做「看起來裝過了嗎」的
 * 快速判斷(存在性),對使用者體感才不會拖慢平常的啟動流程。
 */
export async function isSetupComplete(appDir: string, platform: Platform): Promise<boolean> {
  try {
    await resolveEngineBinaryPath(appDir, platform);
  } catch {
    return false;
  }
  try {
    for await (const _entry of Deno.readDir(`${appDir}/model`)) {
      return true; // 找到至少一個檔案就視為「裝過」,不必列完整個目錄。
    }
    return false; // 目錄存在但是空的。
  } catch {
    return false; // 目錄不存在。
  }
}

/**
 * 嘗試用系統預設瀏覽器開啟 `url`。允許失敗(見檔頭 why):任何錯誤
 * (權限不足、找不到對應指令、無 GUI 環境)都被吞掉,只印一則提示,
 * 不會讓整個服務因為「開瀏覽器」這種非核心步驟而中止。
 */
export async function openBrowser(url: string): Promise<void> {
  const command = Deno.build.os === "windows"
    // rundll32 開瀏覽器是 Windows 的標準做法,比整個 --allow-run=cmd
    // (等於授權任意 shell 指令)範圍窄很多。
    ? new Deno.Command("rundll32", { args: ["url.dll,FileProtocolHandler", url] })
    : new Deno.Command("open", { args: [url] });
  try {
    const { success } = await command.output();
    if (!success) {
      console.error(`無法自動開啟瀏覽器,請手動開啟:${url}`);
    }
  } catch (err) {
    console.error(
      `無法自動開啟瀏覽器(${err instanceof Error ? err.message : String(err)}),請手動開啟:${url}`,
    );
  }
}

/**
 * 真正的組裝流程,抽成獨立函式只是為了讓 `if (import.meta.main)` 區塊能把
 * 「執行」跟「錯誤處理」分開——目標使用者幾乎沒有程式經驗(店規第 2 節),
 * 任何失敗都必須落地成一句看得懂的中文訊息 + 非零 exit code,不能是一段
 * 原始的 stack trace(見下方 catch:未捕捉的 rejection 在 Deno 預設行為下
 * 會印出完整堆疊,對目標使用者毫無意義)。
 */
async function run(): Promise<void> {
  const appDir = resolveAppDir();
  const profile = await readMachineProfile(resolveMachineProfilePath());
  const platform = platformFromProfile(profile);

  if (!(await isSetupComplete(appDir, platform))) {
    throw new Error(
      "尚未完成安裝。請先執行:deno task setup\n" +
        `(會下載 engine host 與模型檔案,驗證 SHA-256 後安裝到 ${appDir})`,
    );
  }

  const binPath = await resolveEngineBinaryPath(appDir, platform);
  const args = buildEngineArgs(platform, `${appDir}/model`, profile);

  console.log(`啟動引擎:${binPath} ${args.join(" ")}`);
  const engine = await EngineClient.spawn(binPath, args);
  console.log(`引擎就緒,backend = ${engine.backend}`);
  engine.onError = (message) => console.error(`[engine error] ${message}`);

  const server = startServer(appDir, engine, DEFAULT_PORT);
  const url = `http://127.0.0.1:${DEFAULT_PORT}/`;
  console.log(`服務已啟動:${url}`);
  await openBrowser(url);

  await server.finished;
}

if (import.meta.main) {
  try {
    await run();
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    Deno.exit(1);
  }
}
