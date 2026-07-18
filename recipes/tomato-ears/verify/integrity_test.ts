/**
 * 驗收測試:manifest.json 宣告的每一個相依檔案,對「目前平台」實際安裝在
 * appDir 底下的檔案,逐一比對是否存在且 SHA-256 相符。這是 `verify/`
 * 套件裡最基礎的一項——放在檔名字母序最前面,`deno test verify/` 預設
 * 依檔名排序執行,如果連檔案完整性都不過,後面的協定/服務測試不可能有
 * 有意義的結果。
 *
 * 平台判斷沿用 `reference/main.ts` 的 `platformFromProfile`(讀
 * `../_machine/machine-profile.json`),與 `deno task start:*`/`setup:*`
 * 用同一套邏輯——verify 驗的是「這台機器實際會執行的那個平台」,不是
 * 兩個平台都驗(使用者機器通常只裝了一個平台的相依)。
 *
 * why engine host 只驗「下載暫存的壓縮包」雜湊、不驗「解壓+pin 後的
 * 穩定路徑執行檔」雜湊:`manifest.json` 的 engine SHA-256 是對**壓縮包
 * 本身**算的,不是對解壓後的執行檔內容算的(tar.gz 解壓、複製 pin 到
 * `bin/engine-host` 都不會改變位元組內容,但如果直接對穩定路徑算雜湊
 * 去比對壓縮包的雜湊,兩者天生就不會相等——不是完整性問題,是比錯對象)。
 * `downloader.ts` 的設計本來就保留 `download/` 目錄裡驗證過的原始壓縮包
 * (見其檔頭 why 註解),所以「壓縮包雜湊符」加上「穩定路徑執行檔存在
 * 且非空檔案」兩件事合起來,才是這個平台完整性的正確驗收方式。
 *
 * 權限:`--allow-read=.,../_machine`(讀 manifest.json、appDir 底下的
 * bin/download/model、machine-profile)。
 */
import { assertEquals, assertExists } from "jsr:@std/assert@^1.0.19";
import { crypto } from "jsr:@std/crypto@^1.1.0";
import { encodeHex } from "jsr:@std/encoding@^1.0.11/hex";
import {
  platformFromProfile,
  readMachineProfile,
  resolveMachineProfilePath,
} from "../reference/main.ts";
import { type Manifest, stableEngineBinaryPath } from "../reference/downloader.ts";

/** `deno task verify:*` 定義上以 app 目錄為 cwd 執行(同 main.ts 的 cwd
 * 模型),`manifest.json` 與 `deno.json` 已在配方安裝時複製到這一層。 */
const APP_DIR = Deno.cwd();
const MANIFEST_PATH = `${APP_DIR}/manifest.json`;

async function readManifest(): Promise<Manifest> {
  const text = await Deno.readTextFile(MANIFEST_PATH);
  return JSON.parse(text) as Manifest;
}

/** 計算檔案的 SHA-256(hex, lowercase),串流讀取避免大檔案(模型可達數百
 * MB)一次讀進記憶體——與 `downloader.ts` 的 `fileSha256` 同款做法。 */
async function fileSha256(path: string): Promise<string> {
  const file = await Deno.open(path, { read: true });
  try {
    const digest = await crypto.subtle.digest("SHA-256", file.readable);
    return encodeHex(digest);
  } finally {
    try {
      file.close();
    } catch {
      // 已被 stream 消費並關閉時再次 close() 會拋錯,忽略即可。
    }
  }
}

Deno.test("integrity：manifest 相依檔案在本機安裝目錄內存在且 SHA-256 相符", async (t) => {
  const manifest = await readManifest();
  const profile = await readMachineProfile(resolveMachineProfilePath());
  const platform = platformFromProfile(profile);

  await t.step(
    "engine host：download/ 目錄內驗證過的原始壓縮包，大小與 SHA-256 皆與 manifest 相符",
    async () => {
      const engineDep = manifest.dependencies.engine[platform];
      const archiveName = engineDep.url.split("/").pop();
      if (!archiveName) {
        throw new Error(`engine dependency URL 無法解析出檔名:${engineDep.url}`);
      }
      const archivePath = `${APP_DIR}/download/${archiveName}`;
      const stat = await Deno.stat(archivePath);
      assertEquals(stat.size, engineDep.byteCount, `archive 大小不符:${archivePath}`);
      assertEquals(
        await fileSha256(archivePath),
        engineDep.sha256,
        `archive SHA-256 不符:${archivePath}`,
      );
    },
  );

  await t.step(
    "engine host：解壓 + pin 後的穩定路徑執行檔存在且非空檔案(內容雜湊的驗收對象是上一步的壓縮包，見檔頭 why 註解)",
    async () => {
      const stablePath = stableEngineBinaryPath(APP_DIR, platform);
      const stat = await Deno.stat(stablePath);
      assertExists(stat);
      if (stat.size === 0) {
        throw new Error(`穩定路徑執行檔是空檔案:${stablePath}`);
      }
    },
  );

  const modelDep = manifest.dependencies.model[platform];
  for (const file of modelDep.files) {
    await t.step(`model/${file.name}：存在且大小/SHA-256 相符`, async () => {
      const filePath = `${APP_DIR}/model/${file.name}`;
      const stat = await Deno.stat(filePath);
      assertEquals(stat.size, file.byteCount, `檔案大小不符:${filePath}`);
      assertEquals(await fileSha256(filePath), file.sha256, `SHA-256 不符:${filePath}`);
    });
  }
});
