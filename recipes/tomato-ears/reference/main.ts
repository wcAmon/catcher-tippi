/**
 * tomato-ears 的 CLI 進入點(`deno task start:mac` / `start:win`):讀
 * machine-profile 決定平台與 engine 啟動旗標、確認 setup 已完成、spawn
 * engine host、起 HTTP/WS 服務、嘗試自動開瀏覽器。
 *
 * ## cwd 相對權限模型(本檔與 manifest/deno.json 的核心約定)
 *
 * `deno task` **定義上**以 deno.json 所在目錄為工作目錄執行;配方安裝時
 * deno.json 就放在 app 目錄(`~/tmuh-apps/tomato-ears/`)根部,所以:
 * - **cwd 就是 app 目錄**——`resolveAppDir()` 預設回傳 `Deno.cwd()`,
 *   完全不做 HOME/USERPROFILE 查找(Deno 的權限旗標**不展開 `~`**,而讀
 *   HOME 環境變數又需要額外的 `--allow-env=HOME`,在宣告旗標下會直接
 *   NotCapable;cwd 模型兩個問題都不存在,權限旗標得以用 `.`/
 *   `../_machine`/`bin/engine-host` 這種跨機器不變的相對路徑靜態宣告,
 *   實測證據見 `reference/permissions_probe_test.ts`);
 * - machine-profile 在 `../_machine/machine-profile.json`(app 目錄的
 *   上一層是 `~/tmuh-apps/`,`_machine/` 是跨 app 共用目錄,對應
 *   `--allow-read=../_machine` 與 `--allow-write=../_machine` 旗標);
 * - `TMUH_APPS_DIR` 環境變數仍可整棵覆寫(值 = `~/tmuh-apps` 的替代
 *   根目錄),供測試/演練把整套安裝指到任意位置——此時呼叫端要自行給
 *   對應的權限旗標(演練環境用 dev 旗標,不受 manifest 靜態宣告限制)。
 *
 * ## 其他設計原則(why)
 *
 * - **純邏輯與 I/O 分離**(同 `recipes/env-base/probe/machine-profile.ts`
 *   的既有慣例):`platformFromProfile`/`buildEngineArgs`/`isSetupComplete`
 *   等函式可獨立單元測試,不必真的 spawn 子行程或起 HTTP 服務;只有
 *   `run()`(經 `if (import.meta.main)` 觸發)做真正的組裝,由
 *   `permissions_probe_test.ts` 的「literal `deno task start:mac`」測試
 *   以與使用者逐字相同的指令做黑箱驗證。
 * - **不從 `recipes/env-base/` import 任何 TS 模組**:配方在店規設計裡是
 *   使用者的 agent 各自複製、獨立組裝的單位(agent 只複製
 *   `recipes/tomato-ears/` 的內容到 app 目錄,不會把整個 monorepo 帶過去)。
 *   本檔跟 `machine-profile.ts` 之間唯一的耦合是「讀寫同一份
 *   machine-profile.json 檔案」,不是程式碼 import。
 * - **自動開瀏覽器(店規第 3 條)以最小系統執行檔納入縮圈清單**:
 *   start 的宣告旗標是 `--allow-run=bin/engine-host,open`(mac)/
 *   `--allow-run=bin/engine-host.exe,explorer`(win)——`open` 與
 *   `explorer` 都是單一系統執行檔,授權範圍遠窄於整個 shell
 *   (`cmd`/`sh`)。`openBrowser` 仍允許失敗(無 GUI、SSH 連線等環境):
 *   只有 spawn 擲例外才降級為印出 URL(exit code 不做為判準,理由見
 *   `openBrowser` 的 why 註解);另可設 `TMUH_NO_BROWSER` 環境變數明確
 *   關閉自動開啟(測試/無頭環境用,已同步列入 --allow-env 白名單)。
 */

// 注意:main.ts 刻意不 import `ensureDependencies`——`deno task setup:*` 是
// 獨立的 deno task(獨立的權限集合,含 --allow-net 對外下載),`deno task
// start:*` 的權限刻意不含對外網路(兩階段權限模型)。setup 是否完成用
// `isSetupComplete()`(純存在性檢查)判斷,不會在 start 階段觸發任何下載
// 或雜湊驗證。
import { type Platform, stableEngineBinaryPath } from "./downloader.ts";
import { EngineClient } from "./engine.ts";
import { startServer } from "./server.ts";

/** manifest.json 的 `ports.http`(見 recipes/tomato-ears/manifest.json)。
 * 兩處必須手動保持同步——這是 glue 腳本的固有侷限;`export` 這個常數是
 * 為了讓 `verify/permissions_test.ts` 能機械化釘住「manifest.json 的
 * `ports.http`」「這個常數」「`start:mac`/`start:win` 宣告的
 * `--allow-net=127.0.0.1:<port>`」三處port 數字永遠一致,防止未來任一處
 * 單獨改動而漂移(Task 2 遺留的一致性檢查缺口,已在 Task 4 補上)。 */
export const DEFAULT_PORT = 43117;

/**
 * app 安裝目錄:預設 = `Deno.cwd()`(cwd 模型,見檔頭——`deno task` 的
 * cwd 就是 deno.json 所在的 app 目錄);`TMUH_APPS_DIR` 設定時 =
 * `${TMUH_APPS_DIR}/tomato-ears`。
 */
export function resolveAppDir(): string {
  const override = Deno.env.get("TMUH_APPS_DIR");
  return override !== undefined ? `${override}/tomato-ears` : Deno.cwd();
}

/**
 * env-base 產出的 machine-profile.json 路徑:預設
 * `../_machine/machine-profile.json`(cwd 相對,恰好落在
 * `--allow-read=../_machine` 的宣告範圍內);`TMUH_APPS_DIR` 設定時
 * `${TMUH_APPS_DIR}/_machine/machine-profile.json`。
 */
export function resolveMachineProfilePath(): string {
  const override = Deno.env.get("TMUH_APPS_DIR");
  return override !== undefined
    ? `${override}/_machine/machine-profile.json`
    : "../_machine/machine-profile.json";
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
 * CPU,結果應該只測一次、記住,不必每次啟動都重新探測(那個探測要實際
 * 載入模型跑基準,不便宜)。首跑後由 {@link writeBackfilledBackend} 回填。
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
 * 把引擎首跑探測出的 backend 回填進 machine-profile(Windows 首跑專用,
 * 這也是 start 權限需要 `--allow-write=../_machine` 的唯一理由——mac
 * 不回填,但兩平台的旗標形狀保持一致,SECURITY.md 的審查對照表比較單純)。
 *
 * 重新從磁碟讀最新內容再 merge(而非沿用啟動時讀進來的那份快照),避免
 * 蓋掉引擎載入期間其他行程(例如使用者同時重跑 env-base 探測)寫入的
 * 欄位;只動 `inference["tomato-ears"]` 這一個鍵,其餘原樣保留——與
 * env-base 的冪等 merge 精神一致(各 app 只管自己的鍵)。
 */
export async function writeBackfilledBackend(
  profilePath: string,
  backend: string,
): Promise<void> {
  const profile = await readMachineProfile(profilePath);
  const inference = (profile.inference && typeof profile.inference === "object")
    ? profile.inference as Record<string, unknown>
    : {};
  const entry = (inference["tomato-ears"] && typeof inference["tomato-ears"] === "object")
    ? inference["tomato-ears"] as Record<string, unknown>
    : {};
  const merged = {
    ...profile,
    inference: { ...inference, "tomato-ears": { ...entry, backend } },
  };
  await Deno.writeTextFile(profilePath, JSON.stringify(merged, null, 2) + "\n");
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
 * 檢查 `deno task setup:*` 是否已經完成:穩定路徑的引擎執行檔存在
 * (setup 的最後一步是 pin,見 downloader.ts 的 `extractAndPinEngine`——
 * 它存在即代表下載+解壓+pin 全部走完),且模型目錄底下至少有一個檔案。
 *
 * why 不驗雜湊:完整的雜湊驗證是 `verify/integrity_test.ts`(Task 4)的
 * 職責——那是使用者主動要求的驗收步驟,可以接受花比較久的時間逐檔算
 * SHA-256。`main.ts` 每次啟動都會呼叫這個檢查,只做「看起來裝過了嗎」的
 * 快速判斷(存在性),對使用者體感才不會拖慢平常的啟動流程。
 */
export async function isSetupComplete(appDir: string, platform: Platform): Promise<boolean> {
  try {
    await Deno.stat(stableEngineBinaryPath(appDir, platform));
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
 * 嘗試用系統預設瀏覽器開啟 `url`(店規第 3 條:啟動後瀏覽器自動開
 * localhost 頁面)。
 *
 * - `TMUH_NO_BROWSER` 環境變數有設定(任何非空值)→ 直接跳過,只印 URL。
 *   測試(permissions_probe_test.ts 的 literal-task 黑箱測試)與無頭環境
 *   用它取得決定性行為,不然每跑一次測試就真的彈一個瀏覽器分頁。
 * - mac 用 `open <url>`,Windows 用 `explorer <url>`(explorer 帶 URL
 *   引數會交給預設瀏覽器開啟)——兩者都在 start 宣告旗標的
 *   `--allow-run` 縮圈清單內。
 * - **只有 spawn 擲例外才視為失敗**(NotCapable、找不到執行檔、無 GUI
 *   環境),降級為印出 URL;**不看 exit code**——why:Windows 的
 *   `explorer` 即使成功開啟也常回傳非零 exit code,用 exit code 判斷
 *   會對使用者謊報「無法開啟」,比不判斷更糟。
 */
export async function openBrowser(url: string): Promise<void> {
  if (Deno.env.get("TMUH_NO_BROWSER")) {
    console.log(`已依 TMUH_NO_BROWSER 略過自動開啟瀏覽器,請手動開啟:${url}`);
    return;
  }
  const command = Deno.build.os === "windows"
    ? new Deno.Command("explorer", { args: [url] })
    : new Deno.Command("open", { args: [url] });
  try {
    await command.output();
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
  const profilePath = resolveMachineProfilePath();
  const profile = await readMachineProfile(profilePath);
  const platform = platformFromProfile(profile);

  if (!(await isSetupComplete(appDir, platform))) {
    throw new Error(
      "尚未完成安裝。請先執行:deno task setup:mac(Windows:setup:win)\n" +
        `(會下載 engine host 與模型檔案,驗證 SHA-256 後安裝到 ${appDir})`,
    );
  }

  const binPath = stableEngineBinaryPath(appDir, platform);
  const args = buildEngineArgs(platform, `${appDir}/model`, profile);

  console.log(`啟動引擎:${binPath} ${args.join(" ")}`);
  const engine = await EngineClient.spawn(binPath, args);
  console.log(`引擎就緒,backend = ${engine.backend}`);
  engine.onError = (message) => console.error(`[engine error] ${message}`);

  // Windows 首跑:把實測探測出的 backend 回填進 machine-profile,下次啟動
  // buildEngineArgs 會帶 --backend 跳過重新探測(見 readBackfilledBackend)。
  if (platform === "windows-x64" && readBackfilledBackend(profile) === undefined) {
    await writeBackfilledBackend(profilePath, engine.backend);
    console.log(`已把 backend=${engine.backend} 回填進 machine-profile(下次啟動跳過探測)`);
  }

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
