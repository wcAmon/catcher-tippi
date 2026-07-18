/**
 * tomato-ears 相依下載器:依 `manifest.json` 的 `dependencies` 下載 engine host
 * 壓縮包與模型檔案,逐檔驗 SHA-256,原子安裝(`.part` → 驗 hash → rename)。
 *
 * 設計原則(why):
 * - **永不信任下載內容**:任何檔案落地前都必須通過 SHA-256 校驗,校驗失敗
 *   立刻刪除殘檔並 throw——呼叫端(`setup.ts`/`main.ts`)不需要、也不應該
 *   自行決定「差不多對就好」。這是店規第 5 條「所有外部下載一律 SHA-256
 *   pin,下載後先驗再用」的直接實作。
 * - **冪等**:已存在且雜湊相符的檔案直接略過下載,讓 `deno task setup`
 *   可以安全地重跑(續傳中斷、換網路重試、甚至單純被使用者重複執行)。
 * - **不做斷點續傳**:v1 為求正確性簡單,永遠整檔重下(不送 HTTP Range)。
 *   任何前次中斷留下的 `.part` 殘檔會被直接截斷覆寫,不影響正確性,只是
 *   犧牲一點頻寬——這在配方情境(數百 MB 模型檔,使用者網路品質不一)
 *   是可接受的取捨,值得未來若真的需要再加 Range 支援。
 * - **engine 壓縮包解壓用系統 `tar -xf`**:mac 產物是 `.tar.gz`,Windows 產物
 *   是 `.zip`;系統內建的 `tar`(macOS 為 bsdtar,Windows 10/11 內建的
 *   `tar.exe` 同樣是 bsdtar 移植版)兩種格式都認得同一套指令列語法,不需要
 *   為平台分岔解壓邏輯。
 *
 * 執行本模組(供 `deno task setup` 使用)所需權限旗標,與
 * `manifest.json.permissions.setup` 逐字同步:
 *   --allow-net
 *   --allow-read=~/tmuh-apps/tomato-ears
 *   --allow-write=~/tmuh-apps/tomato-ears
 *   --allow-run=tar
 *   --allow-env=TMUH_APPS_DIR
 * (`--allow-net` 全開的理由:HF/GitHub 下載會經過 CDN redirect,目標網域
 * 無法窮舉枚舉;完整性改由逐檔 SHA-256 pin 保證,見 SECURITY.md。)
 */

import { crypto } from "jsr:@std/crypto@^1.1.0";
import { encodeHex } from "jsr:@std/encoding@^1.0.11/hex";
import { walk } from "jsr:@std/fs@^1.0.24/walk";
import { dirname } from "jsr:@std/path@^1.0.9/dirname";

/** 兩個目標平台的識別字串,與 manifest.json 的 `dependencies.*` 鍵一致。 */
export type Platform = "macos-arm64" | "windows-x64";

/** manifest.json 裡「單一檔案」的描述(名稱 + 雜湊 + 大小)。 */
export interface ManifestFileEntry {
  name: string;
  sha256: string;
  byteCount: number;
}

/** manifest.json 的 `dependencies.engine.<platform>`:單一壓縮包。 */
export interface EngineDependency {
  url: string;
  sha256: string;
  byteCount: number;
}

/** manifest.json 的 `dependencies.model.<platform>`:多檔案的 HF repo 快照。 */
export interface ModelDependency {
  repo: string;
  baseUrl: string;
  /** Windows 端有 revision pin;mac 端目前浮動於 main(見 manifest 內 `_baseUrlNote`)。 */
  revision?: string;
  files: ManifestFileEntry[];
}

/**
 * `manifest.json` 的最小型別子集——本模組只讀 `dependencies`,其餘欄位
 * (`ports`/`permissions`/`verify`/...)用索引簽章容許存在但不逐一宣告,
 * 避免和 manifest schema 的其他變動耦合。
 */
export interface Manifest {
  name: string;
  version: string;
  dependencies: {
    engine: Record<Platform, EngineDependency>;
    model: Record<Platform, ModelDependency>;
  };
  [key: string]: unknown;
}

/**
 * 各平台引擎 host 執行檔在壓縮包內的檔名(見
 * `scripts/build-asr-host.sh`、`scripts/build-nemotron-asr-host.ps1`)。
 * `main.ts` 用這個名稱在 `resolveEngineBinaryPath` 找出實際 spawn 路徑。
 */
export function engineBinaryName(platform: Platform): string {
  return platform === "windows-x64" ? "nemotron-asr-host.exe" : "catcher-asr-host";
}

/**
 * 在 `${appDir}/bin` 底下遞迴尋找引擎執行檔。
 *
 * why 用遞迴搜尋而非固定相對路徑:mac 的 tar.gz 內含一層
 * `catcher-asr-host-v<version>-macos-arm64/` 包裝目錄,Windows 的 zip 則是
 * 扁平的(`Compress-Archive -Path publish\*` 不含外層目錄)。與其硬編兩種
 * 不同的路徑規則(未來 release 打包方式一改就壞),不如直接在解壓後的
 * `bin/` 樹裡按檔名找——對兩種封裝方式都成立,且解壓目錄本來就只有這一份
 * 引擎,不會有同名檔案誤判的疑慮。
 */
export async function resolveEngineBinaryPath(
  appDir: string,
  platform: Platform,
): Promise<string> {
  const binDir = `${appDir}/bin`;
  const targetName = engineBinaryName(platform);
  try {
    for await (const entry of walk(binDir, { includeDirs: false })) {
      if (entry.name === targetName) return entry.path;
    }
  } catch (err) {
    if (!(err instanceof Deno.errors.NotFound)) throw err;
    // binDir 尚不存在 → 視同「找不到」,落到下面統一的錯誤訊息。
  }
  throw new Error(
    `在 ${binDir} 內找不到引擎執行檔 ${targetName}(setup 是否已完成?請先跑 deno task setup)`,
  );
}

/** 計算檔案的 SHA-256(hex, lowercase)。檔案不存在時回傳 `undefined`。 */
async function fileSha256(path: string): Promise<string | undefined> {
  let file: Deno.FsFile;
  try {
    file = await Deno.open(path, { read: true });
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return undefined;
    throw err;
  }
  try {
    // 用 readable stream 餵 digest,不必把整個檔案(模型檔可達數百 MB)
    // 一次讀進記憶體。
    const digest = await crypto.subtle.digest("SHA-256", file.readable);
    return encodeHex(digest);
  } finally {
    // file.readable 已在 digest() 內被完整消費並自動關閉底層檔案描述符;
    // 若中途拋錯(digest 失敗)則 readable 可能未被消費完,保險起見仍嘗試關閉。
    try {
      file.close();
    } catch {
      // 已經被 stream 消費並關閉時再次 close() 會拋錯,忽略即可。
    }
  }
}

/** 刪除檔案,但檔案本來就不存在也視為成功(冪等清理)。 */
async function safeRemove(path: string): Promise<void> {
  try {
    await Deno.remove(path);
  } catch (err) {
    if (!(err instanceof Deno.errors.NotFound)) throw err;
  }
}

/**
 * 下載單一檔案並驗證,原子落地到 `destPath`。
 *
 * 流程:已存在且雜湊符 → 直接略過;否則下載到 `<destPath>.part`
 * (無論該 `.part` 是否為前次中斷的殘檔,一律以截斷模式開檔重下)→
 * 驗大小 → 驗雜湊 → `rename` 到最終檔名。`rename` 在同一個檔案系統內
 * 是原子操作,確保「使用者/其他行程看到的 `destPath`」要嘛是舊的完整檔案、
 * 要嘛是新驗證過的完整檔案,不會看到半吊子的內容。
 */
async function downloadVerifiedFile(
  url: string,
  sha256: string,
  byteCount: number,
  destPath: string,
  label: string,
  onProgress?: (msg: string) => void,
): Promise<void> {
  const existingHash = await fileSha256(destPath);
  if (existingHash === sha256) {
    onProgress?.(`${label}:已存在且雜湊相符,略過下載`);
    return;
  }

  await Deno.mkdir(dirname(destPath), { recursive: true });
  const partPath = `${destPath}.part`;
  onProgress?.(`${label}:下載中…`);

  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`下載失敗(${label}):HTTP ${response.status}`);
  }

  try {
    const file = await Deno.open(partPath, { create: true, write: true, truncate: true });
    // pipeTo 會在來源串流結束時關閉 file.writable(預設行為),等同關檔,
    // 不需要再手動呼叫 file.close()。
    await response.body.pipeTo(file.writable);
  } catch (err) {
    await safeRemove(partPath);
    throw new Error(`下載中斷(${label}):${err instanceof Error ? err.message : String(err)}`);
  }

  const actualSize = (await Deno.stat(partPath)).size;
  if (actualSize !== byteCount) {
    await safeRemove(partPath);
    throw new Error(
      `檔案大小不符(${label}):預期 ${byteCount} bytes,實際 ${actualSize} bytes`,
    );
  }

  const actualHash = await fileSha256(partPath);
  if (actualHash !== sha256) {
    await safeRemove(partPath);
    throw new Error(`SHA-256 不符(${label}):預期 ${sha256},實際 ${actualHash}`);
  }

  await Deno.rename(partPath, destPath);
  onProgress?.(`${label}:下載完成`);
}

/**
 * 把 engine 壓縮包解壓到 `${appDir}/bin`。
 *
 * why 先檢查「執行檔是否已存在」才解壓:壓縮包本身已經通過雜湊驗證,
 * 重複解壓同一份內容並不會產生錯誤結果,但對使用者體感沒有意義的 I/O——
 * 讓 `deno task setup` 重跑時能快速略過已完成的步驟。
 */
async function extractEngineArchive(
  archivePath: string,
  appDir: string,
  platform: Platform,
  onProgress?: (msg: string) => void,
): Promise<void> {
  const binDir = `${appDir}/bin`;
  try {
    await resolveEngineBinaryPath(appDir, platform);
    onProgress?.("engine host:已解壓,略過");
    return;
  } catch {
    // 找不到 → 尚未解壓(或上次解壓不完整),繼續往下執行。
  }

  await Deno.mkdir(binDir, { recursive: true });
  onProgress?.("engine host:解壓中…");
  const command = new Deno.Command("tar", {
    args: ["-xf", archivePath, "-C", binDir],
    stdout: "piped",
    stderr: "piped",
  });
  const { success, stderr } = await command.output();
  if (!success) {
    throw new Error(`tar 解壓失敗(${archivePath}):${new TextDecoder().decode(stderr)}`);
  }

  // 解壓後仍找不到執行檔 → 封裝格式跟預期不符,視為致命錯誤(而非靜默放過,
  // 讓後續 spawn 才在更難除錯的地方失敗)。
  await resolveEngineBinaryPath(appDir, platform);
  onProgress?.("engine host:解壓完成");
}

/**
 * 確保 `appDir` 底下已備妥指定平台的所有相依(engine host + 模型檔)。
 *
 * 逐檔行為:已存在且雜湊符 → 略過;否則下載到 `.part` → 驗大小/雜湊 →
 * 原子 rename;任何一檔驗證失敗即刪除殘檔並 throw,整個安裝視為失敗
 * (呼叫端不需要、也不應該以「部分成功」繼續)。
 *
 * 目錄配置(`~/tmuh-apps/tomato-ears/` 慣例下):
 * - `download/`:原始下載暫存(engine 壓縮包落地於此,驗證後保留供重跑時
 *   免重下);
 * - `bin/`:engine 壓縮包解壓後的執行檔與隨附檔案;
 * - `model/`:模型檔案,扁平存放(單一平台安裝,不需要再分子目錄)。
 */
export async function ensureDependencies(
  manifest: Manifest,
  appDir: string,
  platform: Platform,
  onProgress?: (msg: string) => void,
): Promise<void> {
  const engineDep = manifest.dependencies.engine[platform];
  const modelDep = manifest.dependencies.model[platform];

  const archiveName = engineDep.url.split("/").pop() ?? "engine.archive";
  const archivePath = `${appDir}/download/${archiveName}`;
  await downloadVerifiedFile(
    engineDep.url,
    engineDep.sha256,
    engineDep.byteCount,
    archivePath,
    "engine host",
    onProgress,
  );
  await extractEngineArchive(archivePath, appDir, platform, onProgress);

  for (const file of modelDep.files) {
    const url = modelDep.baseUrl + file.name;
    const destPath = `${appDir}/model/${file.name}`;
    await downloadVerifiedFile(
      url,
      file.sha256,
      file.byteCount,
      destPath,
      `model/${file.name}`,
      onProgress,
    );
  }
}
