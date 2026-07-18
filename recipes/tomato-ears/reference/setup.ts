/**
 * tomato-ears 的相依安裝 CLI 進入點(`deno task setup:mac` / `setup:win`):
 * 讀 machine-profile 決定平台、讀 `manifest.json` 取得相依清單、呼叫
 * `ensureDependencies`(downloader.ts)下載並驗證 SHA-256、原子安裝到
 * appDir。
 *
 * ## 為什麼是獨立於 main.ts 的檔案(而非在 main.ts 裡加一個 setup 分支)
 *
 * 兩階段權限模型(見 SECURITY.md)要求 `setup` 與 `start` 是**兩個獨立的
 * deno task**,各自宣告獨立的權限集合——setup 需要 `--allow-net`(對外
 * 下載),start 刻意不含對外網路。把兩者寫進同一個檔案,會讓單一模組同時
 * 背負兩種宣告權限交集之外的職責,審查者無法只看 deno.json 的
 * task 指令就確認「這個進入點到底會不會碰網路」。拆成兩個獨立進入點
 * 檔案,讓 SECURITY.md 的審查表可以逐一對照「檔案 ↔ task ↔ 權限」。
 *
 * 與 main.ts 共用的邏輯(`resolveAppDir`/`resolveMachineProfilePath`/
 * `readMachineProfile`/`platformFromProfile`)直接 import,不重複實作——
 * 兩者都在同一個 `reference/` 目錄內,是同一配方內的模組互相 import,
 * 不是跨配方邊界(main.ts 檔頭「不從 recipes/env-base import」的原則
 * 管的是跨配方邊界,不適用於此)。
 *
 * 執行本模組(供 `deno task setup:mac`/`setup:win` 使用)所需權限旗標,
 * 與 `manifest.json.permissions.setup:mac`/`setup:win` 逐字同步(見
 * `verify/permissions_test.ts` 的機械化比對):
 *   --allow-net
 *   --allow-read=.,../_machine
 *   --allow-write=.,../_machine
 *   --allow-run=tar
 *   --allow-env=TMUH_APPS_DIR
 * (`--allow-net` 全開的理由,見 `downloader.ts` 檔頭與 SECURITY.md:HF/
 * GitHub 下載會經過 CDN redirect,目標網域無法窮舉;完整性改由逐檔
 * SHA-256 pin 保證。)
 */

import {
  platformFromProfile,
  readMachineProfile,
  resolveAppDir,
  resolveMachineProfilePath,
} from "./main.ts";
import { ensureDependencies, type Manifest } from "./downloader.ts";
import { fromFileUrl } from "jsr:@std/path@^1.0.9/from-file-url";

/**
 * `manifest.json` 的路徑:錨定在本檔(`reference/setup.ts`)所在位置的
 * 上一層——即 app 目錄根部(`deno.json`/`manifest.json` 與 `reference/`
 * 同層,這是配方安裝時的固定佈局,見 PLAN.md)。用 `import.meta.url` 而非
 * cwd 相對字串來算這個路徑,是防禦性寫法:`deno task` 定義上一定以
 * app 目錄為 cwd(main.ts 檔頭的 cwd 模型說明),兩種算法在正常使用下
 * 解析到同一個絕對路徑,但錨定在模組自己的位置不會受呼叫方式影響。
 *
 * **必須用 `fromFileUrl` 而非裸 `new URL(...).pathname`**(Task 6 Windows
 * 演練實測發現的阻斷性 bug,見
 * `.superpowers/sdd/task-6-rehearsal-log.md`):WHATWG URL 的 `.pathname`
 * 在 Windows 磁碟機路徑上會保留 URL 規範要求的前導斜線(例如
 * `file:///C:/Users/...` → pathname `/C:/Users/...`),這個字串**不是**
 * 合法的 Windows 原生路徑——Rust/Deno 的檔案系統層會把開頭的 `/` 解讀成
 * 「相對於目前磁碟機根目錄」,於是變成尋找一個字面上名為 `C:` 的資料夾
 * (NTFS 不允許檔名含冒號),結果是 `NotFound: 系統找不到指定的路徑`
 * (os error 3)——無論宣告多寬的 `--allow-read` 都无法修正,因為問題發生
 * 在權限檢查通過之後的 OS 層級路徑解析。`fromFileUrl`(`jsr:@std/path`,
 * 本檔已透過 `downloader.ts` 間接依賴,`deno.lock` 已釘住整個套件版本,
 * 加這個子路徑匯入不需要更新 lockfile)會依平台正確轉換,mac/Linux 與
 * Windows 皆可正確處理,不需要额外的平台判斷分支。
 */
const MANIFEST_PATH = fromFileUrl(new URL("../manifest.json", import.meta.url));

async function readManifest(): Promise<Manifest> {
  const text = await Deno.readTextFile(MANIFEST_PATH);
  return JSON.parse(text) as Manifest;
}

/**
 * 真正的組裝流程,抽成獨立函式的理由與 `main.ts` 的 `run()` 相同:讓
 * `if (import.meta.main)` 區塊能把「執行」跟「錯誤處理」分開,任何失敗
 * 都落地成一句看得懂的中文訊息 + 非零 exit code(目標使用者幾乎沒有
 * 程式經驗,見店規第 2 節)。
 */
async function run(): Promise<void> {
  const appDir = resolveAppDir();
  const profilePath = resolveMachineProfilePath();
  const profile = await readMachineProfile(profilePath);
  const platform = platformFromProfile(profile);
  const manifest = await readManifest();

  console.log(`安裝 tomato-ears 相依到 ${appDir}(平台:${platform})…`);
  await ensureDependencies(manifest, appDir, platform, (msg) => console.log(msg));
  console.log(
    "安裝完成。可執行 `deno task verify:mac`(Windows:`deno task verify:win`)驗收," +
      "或直接 `deno task start:mac`/`start:win` 啟動。",
  );
}

if (import.meta.main) {
  try {
    await run();
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    Deno.exit(1);
  }
}
