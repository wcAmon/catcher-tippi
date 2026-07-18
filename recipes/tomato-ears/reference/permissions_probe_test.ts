/**
 * 權限旗標語意的**實測固化測試**:manifest/deno.json 的 cwd 相對權限模型
 * (`.`/`../_machine`/`bin/engine-host`)建立在三個 Deno 行為事實上,本檔
 * 用正例+反例把每個事實釘死,防止未來 Deno 升版或旗標改寫時靜默回歸:
 *
 * 1. `--allow-read=<相對路徑>` 以 cwd 解析,範圍外(`../`)的讀取被拒;
 * 2. 權限旗標**不展開 `~`**——`--allow-read=~/tmuh-apps` 這種寫法對真實
 *    使用者是不可用的(比對不到任何實際路徑);
 * 3. `--allow-run=<相對檔案路徑>` 精確縮圈:清單內的執行檔(以相對或
 *    絕對路徑 spawn 皆可)允許,清單外(如 `/bin/echo`)被拒。
 *
 * 最後一個測試以**與使用者逐字相同的指令**(`deno task start:mac`,無任何
 * 額外旗標或覆寫)在一個組裝好的假 app 目錄跑起完整服務,驗證宣告旗標
 * 真的足以支撐 start 全流程(這關閉了「宣告旗標從未被實際執行過」的
 * 驗證缺口)。
 *
 * 執行本檔所需權限旗標(dev-time 測試;需要 spawn `deno` 子行程與 lsof,
 * 故 --allow-run 不縮圈):
 *   deno test --allow-run --allow-read --allow-write --allow-env \
 *     --allow-net=127.0.0.1:43117 \
 *     recipes/tomato-ears/reference/permissions_probe_test.ts
 */

import { assertEquals, assertStringIncludes } from "jsr:@std/assert@^1.0.19";
import { fromFileUrl } from "jsr:@std/path@^1.0.9/from-file-url";

/** 這些 probe 用 shell script 當假執行檔,Windows 上不適用;
 * Windows 端的等價驗證由 Task 6 演練(真機 ssh)覆蓋。 */
const isPosix = Deno.build.os !== "windows";

interface ProbeResult {
  code: number;
  stdout: string;
  stderr: string;
}

/** 以指定 cwd 與權限旗標跑一個子 Deno 行程執行 `scriptPath`,收集輸出。
 * `env` 全清空後只給白名單(HOME 供 Deno 找到模組快取,PATH 供解析系統
 * 指令)——避免外層測試環境的變數(尤其 TMUH_APPS_DIR)洩漏進 probe。 */
async function runDeno(
  cwd: string,
  denoArgs: string[],
  extraEnv: Record<string, string> = {},
): Promise<ProbeResult> {
  const execDir = Deno.execPath().replace(/\/[^/]+$/, "");
  const command = new Deno.Command(Deno.execPath(), {
    args: denoArgs,
    cwd,
    stdout: "piped",
    stderr: "piped",
    clearEnv: true,
    env: {
      HOME: Deno.env.get("HOME") ?? "",
      PATH: `${execDir}:/usr/bin:/bin`,
      ...extraEnv,
    },
  });
  const { code, stdout, stderr } = await command.output();
  return {
    code,
    stdout: new TextDecoder().decode(stdout),
    stderr: new TextDecoder().decode(stderr),
  };
}

Deno.test({
  name: "probe：--allow-read=. 以 cwd 解析——cwd 內可讀，../ 之外 NotCapable",
  ignore: !isPosix,
  fn: async () => {
    const root = await Deno.makeTempDir({ prefix: "perm-probe-read-" });
    try {
      await Deno.mkdir(`${root}/app`);
      await Deno.writeTextFile(`${root}/app/inside.txt`, "inside");
      await Deno.writeTextFile(`${root}/outside.txt`, "outside");
      await Deno.writeTextFile(
        `${root}/app/probe.ts`,
        `const inside = await Deno.readTextFile("inside.txt");
console.log("INSIDE:" + inside);
try {
  await Deno.readTextFile("../outside.txt");
  console.log("OUTSIDE:OK");
} catch (err) {
  console.log("OUTSIDE:" + (err instanceof Deno.errors.NotCapable ? "DENIED" : "OTHER"));
}
`,
      );

      const result = await runDeno(`${root}/app`, ["run", "--allow-read=.", "probe.ts"]);
      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "INSIDE:inside");
      assertStringIncludes(result.stdout, "OUTSIDE:DENIED");
    } finally {
      await Deno.remove(root, { recursive: true });
    }
  },
});

Deno.test({
  name: "probe：權限旗標不展開 ~——--allow-read=~/tmuh-apps 擋不出 $HOME/tmuh-apps 的實際路徑",
  ignore: !isPosix,
  fn: async () => {
    // 建一個假 HOME,底下真的放 tmuh-apps/f.txt;child 的 HOME 環境變數
    // 指向它——若 Deno 會展開 ~,--allow-read=~/tmuh-apps 就該允許這個
    // 讀取;實際結果是 NotCapable,證明 ~ 是被當成字面路徑(cwd 相對的
    // 「./~/tmuh-apps」目錄)處理的。這正是 Task 1 原宣告不可用的根因。
    const fakeHome = await Deno.makeTempDir({ prefix: "perm-probe-home-" });
    try {
      await Deno.mkdir(`${fakeHome}/tmuh-apps`, { recursive: true });
      await Deno.writeTextFile(`${fakeHome}/tmuh-apps/f.txt`, "content");
      await Deno.writeTextFile(
        `${fakeHome}/probe.ts`,
        `try {
  await Deno.readTextFile(Deno.args[0]);
  console.log("READ:OK");
} catch (err) {
  console.log("READ:" + (err instanceof Deno.errors.NotCapable ? "DENIED" : "OTHER"));
}
`,
      );

      const result = await runDeno(
        fakeHome,
        ["run", "--allow-read=~/tmuh-apps", "probe.ts", `${fakeHome}/tmuh-apps/f.txt`],
        { HOME: fakeHome },
      );
      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "READ:DENIED");
    } finally {
      await Deno.remove(fakeHome, { recursive: true });
    }
  },
});

Deno.test({
  name: "probe：--allow-run=bin/engine-host 精確縮圈——相對/絕對 spawn 皆可，/bin/echo 被拒",
  ignore: !isPosix,
  fn: async () => {
    const appDir = await Deno.makeTempDir({ prefix: "perm-probe-run-" });
    try {
      await Deno.mkdir(`${appDir}/bin`);
      await Deno.writeTextFile(`${appDir}/bin/engine-host`, "#!/bin/sh\necho ENGINE_RAN\n");
      await Deno.chmod(`${appDir}/bin/engine-host`, 0o755);
      await Deno.writeTextFile(
        `${appDir}/probe.ts`,
        `const decoder = new TextDecoder();
const rel = await new Deno.Command("bin/engine-host", { stdout: "piped" }).output();
console.log("REL:" + decoder.decode(rel.stdout).trim());
const abs = await new Deno.Command(Deno.cwd() + "/bin/engine-host", { stdout: "piped" }).output();
console.log("ABS:" + decoder.decode(abs.stdout).trim());
try {
  await new Deno.Command("/bin/echo", { args: ["x"], stdout: "piped" }).output();
  console.log("ECHO:OK");
} catch (err) {
  console.log("ECHO:" + (err instanceof Deno.errors.NotCapable ? "DENIED" : "OTHER"));
}
`,
      );

      // --allow-read=. 是 Deno.cwd() 需要的(對齊 start 旗標本來就含它)。
      const result = await runDeno(
        appDir,
        ["run", "--allow-read=.", "--allow-run=bin/engine-host", "probe.ts"],
      );
      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "REL:ENGINE_RAN");
      assertStringIncludes(result.stdout, "ABS:ENGINE_RAN");
      assertStringIncludes(result.stdout, "ECHO:DENIED");
    } finally {
      await Deno.remove(appDir, { recursive: true });
    }
  },
});

/** literal-task 測試需要 mac 本地 build 的 fake-engine host(同 engine_test.ts)。
 * why fromFileUrl(見 reference/setup.ts 的 MANIFEST_PATH 註解):裸
 * `new URL(...).pathname` 在 Windows 會產生 `/C:/...` 這種非法原生路徑，
 * 導致 Deno.stat/readFile 以 os error 3 失敗——Task 6 Windows 演練實測發現。 */
const FAKE_HOST_PATH = fromFileUrl(
  new URL("../../../target/release/catcher-asr-host", import.meta.url),
);

async function fakeHostBuilt(): Promise<boolean> {
  try {
    await Deno.stat(FAKE_HOST_PATH);
    return true;
  } catch {
    return false;
  }
}

Deno.test({
  name: "literal `deno task start:mac`：宣告旗標零覆寫，在組裝好的假 app 目錄跑起完整服務",
  ignore: Deno.build.os !== "darwin" || !(await fakeHostBuilt()),
  fn: async () => {
    // 佔埠預檢:43117 是 manifest 宣告的固定埠,被別的行程占住時直接給出
    // 清楚訊息,而不是讓 fetch 打到不相干的服務產生誤導性的失敗。
    try {
      const listener = Deno.listen({ hostname: "127.0.0.1", port: 43117 });
      listener.close();
    } catch {
      throw new Error("port 43117 已被占用,請先關閉占用的行程再跑本測試");
    }

    const root = await Deno.makeTempDir({ prefix: "perm-probe-task-" });
    const appDir = `${root}/tomato-ears`;
    const recipeDir = new URL("..", import.meta.url).pathname;
    try {
      // 組裝假 app 目錄:結構與 PLAN.md 指示 agent 組出來的一致——
      // deno.json/deno.lock + reference/ 在 app 目錄根部,`deno task` 因此
      // 以 app 目錄為 cwd(cwd 模型的前提)。
      await Deno.mkdir(`${appDir}/reference`, { recursive: true });
      await Deno.mkdir(`${appDir}/bin`, { recursive: true });
      await Deno.mkdir(`${appDir}/model`, { recursive: true });
      await Deno.mkdir(`${appDir}/ui`, { recursive: true });
      await Deno.mkdir(`${root}/_machine`, { recursive: true });

      await Deno.copyFile(`${recipeDir}deno.json`, `${appDir}/deno.json`);
      await Deno.copyFile(`${recipeDir}deno.lock`, `${appDir}/deno.lock`);
      for (const mod of ["main.ts", "engine.ts", "server.ts", "downloader.ts"]) {
        await Deno.copyFile(`${recipeDir}reference/${mod}`, `${appDir}/reference/${mod}`);
      }

      // 穩定路徑放一個 wrapper,轉呼叫本地 build 的 fake-engine host——
      // main.ts 會用 --model/--language 引數 spawn 它(引數被 wrapper 丟棄,
      // fake engine 不需要模型),真實走完 spawn→ready→serve 的宣告旗標路徑。
      await Deno.writeTextFile(
        `${appDir}/bin/engine-host`,
        `#!/bin/sh\nexec "${FAKE_HOST_PATH}" --fake-engine\n`,
      );
      await Deno.chmod(`${appDir}/bin/engine-host`, 0o755);
      await Deno.writeTextFile(`${appDir}/model/weights.bin`, "dummy");
      await Deno.writeTextFile(`${appDir}/ui/index.html`, "<h1>tomato-ears probe</h1>");
      await Deno.writeTextFile(
        `${root}/_machine/machine-profile.json`,
        JSON.stringify({ os: "darwin", arch: "aarch64" }) + "\n",
      );

      // 逐字執行使用者會執行的指令:deno task start:mac,cwd = app 目錄,
      // 環境不帶 TMUH_APPS_DIR(cwd 模型的預設路徑解析)。
      const execDir = Deno.execPath().replace(/\/[^/]+$/, "");
      const child = new Deno.Command(Deno.execPath(), {
        args: ["task", "start:mac"],
        cwd: appDir,
        stdout: "piped",
        stderr: "piped",
        clearEnv: true,
        // TMUH_NO_BROWSER=1:宣告旗標含 open(店規第 3 條的自動開瀏覽器),
        // 但測試環境不該每跑一次就真的彈出瀏覽器分頁——用配方內建的環境
        // 變數退出機制取得決定性行為,同時順帶驗證這個退出機制本身可用。
        env: {
          HOME: Deno.env.get("HOME") ?? "",
          PATH: `${execDir}:/usr/bin:/bin`,
          TMUH_NO_BROWSER: "1",
        },
      }).spawn();
      const stdoutPromise = new Response(child.stdout).text();
      const stderrPromise = new Response(child.stderr).text();

      try {
        // 輪詢等服務起來(不用固定 sleep,避免慢機器 flaky)。
        let body: string | undefined;
        const deadline = Date.now() + 20_000;
        while (Date.now() < deadline) {
          try {
            const response = await fetch("http://127.0.0.1:43117/");
            body = await response.text();
            assertEquals(response.status, 200);
            break;
          } catch {
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
        }
        if (body === undefined) {
          throw new Error(
            `服務在時限內沒有起來。stderr:\n${await stderrPromise}`,
          );
        }
        assertStringIncludes(body, "tomato-ears probe");
      } finally {
        // 關閉:lsof 找出實際監聽 43117 的行程(deno task 的孫行程
        // deno run)並終結;deno run 一死,engine host 的 stdin 得到 EOF
        // 而自行結束(協定規定),deno task 也隨之退出——不會殘留孤兒。
        // 必須用 -sTCP:LISTEN 只挑「監聽者」:不加的話,本測試行程自己的
        // fetch keep-alive 連線(ESTABLISHED)也會被列出,等於 SIGKILL
        // 自己(實際發生過:整個 deno test 以 exit 137 陣亡)。
        const lsof = await new Deno.Command("/usr/sbin/lsof", {
          args: ["-ti", "tcp:43117", "-sTCP:LISTEN"],
          stdout: "piped",
          stderr: "null",
        }).output();
        for (const line of new TextDecoder().decode(lsof.stdout).split("\n")) {
          const pid = Number.parseInt(line.trim(), 10);
          if (Number.isFinite(pid)) {
            try {
              Deno.kill(pid, "SIGKILL");
            } catch {
              // 行程可能已自行結束。
            }
          }
        }
        try {
          child.kill("SIGKILL");
        } catch {
          // 可能已隨 deno run 結束而退出。
        }
        await child.status;
      }

      const stdout = await stdoutPromise;
      await stderrPromise; // 消費完 stderr 串流(內容只有 deno task 的指令回顯)。
      // 宣告旗標下的完整啟動輸出:引擎 ready(fake backend)+ 服務啟動。
      assertStringIncludes(stdout, "引擎就緒,backend = fake");
      assertStringIncludes(stdout, "服務已啟動:http://127.0.0.1:43117/");
      // TMUH_NO_BROWSER 退出機制生效:不 spawn open,改印跳過訊息 + URL。
      // (不設 TMUH_NO_BROWSER 時,宣告旗標本來就允許 open——瀏覽器會
      // 真的開啟,那是正確行為而非測試該擋的事,所以這裡驗證的是退出
      // 路徑,實際開啟路徑由 Task 5 mac 演練以人眼驗收。)
      assertStringIncludes(stdout, "已依 TMUH_NO_BROWSER 略過自動開啟瀏覽器");
    } finally {
      await Deno.remove(root, { recursive: true });
    }
  },
});
