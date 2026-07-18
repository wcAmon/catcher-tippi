/**
 * App 0 環境探測腳本(店規 §5,`docs/superpowers/specs/2026-07-18-mini-app-store-design.md`)。
 *
 * 探測本機硬體與 Deno 版本,寫入 `~/tmuh-apps/_machine/machine-profile.json`。
 *
 * 冪等 merge(why):App 0 可能被重跑(例如使用者升級了記憶體或 Deno),
 * 但個別 mini-app 在自己首次啟動時回填進 machine-profile 的欄位
 * (例如推論後端探測結果 `inference`)不屬於本腳本管轄,絕不能被覆寫或砍掉——
 * 那些欄位往往是實際載入模型 + 跑基準測量得來的,砍掉代表逼每個 app 重測一次。
 * 因此本腳本只覆寫自己產出的欄位(見 {@link ProbeFields}),其餘既有鍵原樣保留。
 *
 * 執行(最小權限旗標,逐一列出理由):
 *   deno run \
 *     --allow-sys=systemMemoryInfo \
 *     --allow-read=$HOME/tmuh-apps \
 *     --allow-write=$HOME/tmuh-apps \
 *     --allow-env=HOME,USERPROFILE,TMUH_APPS_DIR \
 *     recipes/env-base/probe/machine-profile.ts
 *
 * - `--allow-sys=systemMemoryInfo`:讀取實體記憶體總量(`Deno.systemMemoryInfo()`)。
 * - `--allow-read`/`--allow-write=$HOME/tmuh-apps`:讀取既有 profile 做 merge、寫入新 profile
 *   (`TMUH_APPS_DIR` 覆寫時,呼叫端要把這兩個旗標換成對應的覆寫路徑,見下方)。
 * - `--allow-env=HOME,USERPROFILE,TMUH_APPS_DIR`:解析使用者家目錄以定位
 *   `~/tmuh-apps`(mac/Linux 用 `HOME`,Windows 用 `USERPROFILE`);
 *   `TMUH_APPS_DIR` 是測試/演練情境的覆寫(見 {@link defaultProfilePath}
 *   的 why 說明),語意對齊 `recipes/tomato-ears/reference/main.ts` 的
 *   `resolveAppDir`/`resolveMachineProfilePath`——同一個變數名,同一種
 *   「整棵替代根目錄」語意,兩處各自獨立支援(不共用程式碼,見配方
 *   獨立組裝的既有慣例),但行為必須一致,否則 env-base 探測出來的路徑
 *   會跟 tomato-ears 讀 machine-profile 的路徑對不上。
 */

/** 本腳本探測出的欄位;`merge` 時只有這些鍵會被覆寫。 */
export interface ProbeFields {
  /** 探測時間戳(ISO 8601),供人工核對「這份 profile 是何時測的」。 */
  probedAt: string;
  /** `Deno.build.os`:`"darwin" | "windows" | "linux"` 等。 */
  os: string;
  /** `Deno.build.arch`:`"aarch64" | "x86_64"` 等。 */
  arch: string;
  /** 實體記憶體總量(bytes),來自 `Deno.systemMemoryInfo().total`。 */
  ramBytes: number;
  /** CPU 執行緒數,來自 `navigator.hardwareConcurrency`。 */
  cpuThreads: number;
  /** 執行本腳本的 Deno 版本,來自 `Deno.version.deno`。 */
  denoVersion: string;
}

/**
 * 探測本機事實並組成 {@link ProbeFields}。
 *
 * why 拆成獨立函式:呼叫端(`main`)才需要 I/O 權限;這個函式本身除了
 * `Deno.systemMemoryInfo()`(需要 `--allow-sys`)之外都是唯讀查詢,
 * 方便測試時把「探測」和「寫檔/merge」邏輯分開驗證。
 */
export function buildProbeFields(): ProbeFields {
  return {
    probedAt: new Date().toISOString(),
    os: Deno.build.os,
    arch: Deno.build.arch,
    ramBytes: Deno.systemMemoryInfo().total,
    cpuThreads: navigator.hardwareConcurrency,
    denoVersion: Deno.version.deno,
  };
}

/**
 * 冪等 merge:回傳新物件 = `existing` 的所有欄位,再以 `probe` 覆寫。
 *
 * why:因為 `probe` 只包含本腳本管轄的鍵(見 {@link ProbeFields}),用物件
 * 展開(而非白名單覆寫)就能保證「existing 裡任何本腳本不認識的鍵
 * (例如 app 回填的 `inference`)原樣保留」——不需要維護一份易漂移的
 * 白名單常數。純函式,不做 I/O,方便單元測試。
 */
export function mergeProfile(
  existing: Record<string, unknown> | undefined,
  probe: ProbeFields,
): Record<string, unknown> {
  return { ...(existing ?? {}), ...probe };
}

/**
 * 讀取既有 profile 檔。
 *
 * 不存在(全新安裝)或內容無法解析成物件(損毀/格式錯)一律回傳 `undefined`,
 * 讓呼叫端視同「從零開始」——不嘗試把 merge 建立在垃圾資料上。
 */
async function readExisting(
  path: string,
): Promise<Record<string, unknown> | undefined> {
  let text: string;
  try {
    text = await Deno.readTextFile(path);
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return undefined;
    throw err;
  }
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return undefined; // why: 非物件內容(例如陣列/純字串)視同損毀,不 merge
  } catch {
    return undefined; // why: 無法解析的 JSON 視同損毀,不 merge、不拋錯中斷 App 0
  }
}

/** 解析使用者家目錄:mac/Linux 用 `HOME`,Windows 用 `USERPROFILE`。 */
function homeDir(): string {
  const home = Deno.build.os === "windows"
    ? Deno.env.get("USERPROFILE")
    : Deno.env.get("HOME");
  if (!home) {
    throw new Error("無法解析家目錄(HOME/USERPROFILE 環境變數未設定)");
  }
  return home;
}

/**
 * `~/tmuh-apps/_machine/machine-profile.json` 的預設路徑(店規 §5 目錄慣例)。
 *
 * `TMUH_APPS_DIR` 覆寫(why):測試/演練情境需要把整套安裝指到非真實
 * `HOME` 的位置,但本腳本原本完全不認識這個變數(Task 5 mac 演練的卡點
 * #2,見 `.superpowers/sdd/task-5-rehearsal-log.md`)——演練當下只能靠
 * 覆寫 `HOME` 環境變數繞過,不夠直接。語意對齊
 * `recipes/tomato-ears/reference/main.ts` 的 `resolveAppDir`/
 * `resolveMachineProfilePath`:`TMUH_APPS_DIR` 的值 = `~/tmuh-apps` 的
 * 替代根目錄本身(不是它的上一層),設定時直接回傳
 * `${TMUH_APPS_DIR}/_machine/machine-profile.json`,完全不查
 * `HOME`/`USERPROFILE`(未設定時才落回原本的 `homeDir()` 邏輯)。
 */
export function defaultProfilePath(): string {
  const override = Deno.env.get("TMUH_APPS_DIR");
  if (override !== undefined) {
    return `${override}/_machine/machine-profile.json`;
  }
  return `${homeDir()}/tmuh-apps/_machine/machine-profile.json`;
}

/** 探測、merge、寫檔的完整流程,供 CLI 進入點使用。 */
export async function probeAndWrite(
  path: string,
): Promise<Record<string, unknown>> {
  const dir = path.substring(0, path.lastIndexOf("/"));
  await Deno.mkdir(dir, { recursive: true });
  const existing = await readExisting(path);
  const merged = mergeProfile(existing, buildProbeFields());
  await Deno.writeTextFile(path, JSON.stringify(merged, null, 2) + "\n");
  return merged;
}

if (import.meta.main) {
  const path = defaultProfilePath();
  const merged = await probeAndWrite(path);
  console.log(`machine-profile 已寫入:${path}`);
  console.log(JSON.stringify(merged, null, 2));
}
