/**
 * 驗收測試:machine-readable 版本的 SECURITY.md 第一條審查步驟——「deno.json
 * 的 task 指令旗標 == manifest.json 宣告的 permissions」,逐字元素比對
 * (而非只驗證「有對應的旗標存在」這種寬鬆檢查)。這把 SECURITY.md 文字
 * 描述的審查步驟變成每次 `deno task verify:*` 都會自動重跑的機械化防
 * 回歸——manifest.json/deno.json 之後任何一邊改動旗標而忘記同步另一邊,
 * 這個測試立刻紅燈,不必依賴人工逐字比對兩份檔案。
 *
 * 只比對 `setup:mac`/`setup:win`/`start:mac`/`start:win` 四把 task
 * ——這四個是 manifest.json.permissions 實際宣告的鍵(店規第 5 條「權限
 * 顯式宣告在 manifest」管的是「應用執行旗標」,即安裝/啟動兩階段;
 * `verify:*` 本身是驗收工具鏈的內部旗標,不在 manifest 的宣告範圍內,
 * 見 SECURITY.md「verify 與 start permission 差異」章節的說明)。
 *
 * 額外驗證(見 `reference/main.ts` 檔頭 `DEFAULT_PORT` 常數旁的 why 註解,
 * Task 2 遺留、本檔補上的一致性檢查):`manifest.json` 的 `ports.http`、
 * `reference/main.ts` 的 `DEFAULT_PORT` 常數、`start:mac`/`start:win`
 * 宣告的 `--allow-net=127.0.0.1:<port>`,三處必須是同一個數字。
 *
 * 權限:`--allow-read=.`(讀 `deno.json`/`manifest.json`)。
 */
import { assertEquals } from "jsr:@std/assert@^1.0.19";
import { DEFAULT_PORT } from "../reference/main.ts";

const APP_DIR = Deno.cwd();

interface DenoJson {
  tasks: Record<string, string>;
}

interface ManifestSubset {
  permissions: Record<string, string[]>;
  ports: { http: number; protocol: string };
}

async function readDenoJson(): Promise<DenoJson> {
  return JSON.parse(await Deno.readTextFile(`${APP_DIR}/deno.json`)) as DenoJson;
}

async function readManifest(): Promise<ManifestSubset> {
  return JSON.parse(await Deno.readTextFile(`${APP_DIR}/manifest.json`)) as ManifestSubset;
}

/**
 * 從 `deno run/test <flags...> <script>` 這種 task 指令字串裡取出
 * `--allow-*` 旗標,依原字串內的順序。task 指令目前的其餘 token 只有
 * `deno`/`run`/`test` 與最後的腳本路徑,兩者都不會以 `--allow-` 開頭,
 * 用前綴篩選足夠精確,不需要完整的 shell 語法解析器。
 */
function extractAllowFlags(taskCommand: string): string[] {
  return taskCommand.split(/\s+/).filter((token) => token.startsWith("--allow-"));
}

const TASKS_TO_CHECK = ["setup:mac", "setup:win", "start:mac", "start:win"] as const;

Deno.test("permissions：deno.json 各平台 task 的 --allow-* 旗標與 manifest.json.permissions 逐字相等", async () => {
  const denoJson = await readDenoJson();
  const manifest = await readManifest();

  for (const key of TASKS_TO_CHECK) {
    const taskCommand = denoJson.tasks[key];
    if (taskCommand === undefined) {
      throw new Error(`deno.json 缺少 task「${key}」`);
    }
    const declaredFlags = manifest.permissions[key];
    if (declaredFlags === undefined) {
      throw new Error(`manifest.json.permissions 缺少鍵「${key}」`);
    }
    assertEquals(
      extractAllowFlags(taskCommand),
      declaredFlags,
      `task「${key}」的旗標與 manifest.permissions.${key} 不一致`,
    );
  }
});

Deno.test(
  "permissions：manifest.ports.http / main.ts 的 DEFAULT_PORT / start:* 宣告的 --allow-net port 三處一致",
  async () => {
    const denoJson = await readDenoJson();
    const manifest = await readManifest();

    assertEquals(
      manifest.ports.http,
      DEFAULT_PORT,
      "manifest.json 的 ports.http 與 reference/main.ts 的 DEFAULT_PORT 不一致",
    );

    for (const key of ["start:mac", "start:win"] as const) {
      const taskCommand = denoJson.tasks[key];
      if (taskCommand === undefined) {
        throw new Error(`deno.json 缺少 task「${key}」`);
      }
      const netFlag = extractAllowFlags(taskCommand).find((flag) =>
        flag.startsWith("--allow-net=")
      );
      if (netFlag === undefined) {
        throw new Error(`task「${key}」缺少 --allow-net= 旗標`);
      }
      assertEquals(
        netFlag,
        `--allow-net=127.0.0.1:${DEFAULT_PORT}`,
        `task「${key}」宣告的 --allow-net port 與 DEFAULT_PORT 不一致`,
      );
    }
  },
);
