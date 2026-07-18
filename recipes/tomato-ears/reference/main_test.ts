/**
 * `main.ts` 的測試:只測純邏輯/I/O 輔助函式(`platformFromProfile`、
 * `buildEngineArgs`、`readBackfilledBackend`、`writeBackfilledBackend`、
 * `resolveAppDir`、`resolveMachineProfilePath`、`readMachineProfile`、
 * `isSetupComplete`、`openBrowser` 的 TMUH_NO_BROWSER 退出路徑),不測
 * `run()` 裡真正 spawn engine/起 server 的組裝流程——那段由
 * `permissions_probe_test.ts` 的「literal `deno task start:mac`」測試做
 * 黑箱驗證(以與使用者逐字相同的指令與宣告旗標執行)。
 *
 * 執行本檔所需權限旗標(dev-time 測試):
 *   deno test --allow-read --allow-write --allow-env \
 *     recipes/tomato-ears/reference/main_test.ts
 * - `--allow-read`/`--allow-write`:`isSetupComplete`/`readMachineProfile`/
 *   `writeBackfilledBackend` 測試要建臨時目錄與檔案。
 * - `--allow-env`:`resolveAppDir`/`resolveMachineProfilePath`/
 *   `openBrowser` 測試要設/刪/讀 `TMUH_APPS_DIR` 與 `TMUH_NO_BROWSER`
 *   (dev-time 測試需要 set/delete,比正式 runtime 的唯讀語意更寬,
 *   不必比照 manifest)。
 * - 刻意**不給** `--allow-run`:openBrowser 測試藉此驗證 TMUH_NO_BROWSER
 *   路徑真的沒有嘗試 spawn(若有,NotCapable 會被 openBrowser 捕捉並印出
 *   降級訊息,測試的 console 攔截就會看到 error 輸出而失敗)。
 */

import {
  assertEquals,
  assertObjectMatch,
  assertRejects,
  assertThrows,
} from "jsr:@std/assert@^1.0.19";
import {
  buildEngineArgs,
  isSetupComplete,
  openBrowser,
  platformFromProfile,
  readBackfilledBackend,
  readMachineProfile,
  resolveAppDir,
  resolveMachineProfilePath,
  writeBackfilledBackend,
} from "./main.ts";

Deno.test("platformFromProfile：darwin/aarch64 → macos-arm64", () => {
  assertEquals(platformFromProfile({ os: "darwin", arch: "aarch64" }), "macos-arm64");
});

Deno.test("platformFromProfile：windows/x86_64 → windows-x64", () => {
  assertEquals(platformFromProfile({ os: "windows", arch: "x86_64" }), "windows-x64");
});

Deno.test("platformFromProfile：不支援的組合 throw（例如 linux）", () => {
  assertThrows(() => platformFromProfile({ os: "linux", arch: "x86_64" }));
});

Deno.test("readBackfilledBackend：讀到巢狀 tomato-ears.backend", () => {
  const profile = { inference: { "tomato-ears": { backend: "cpu" } } };
  assertEquals(readBackfilledBackend(profile), "cpu");
});

Deno.test("readBackfilledBackend：欄位不存在時回傳 undefined（不 throw）", () => {
  assertEquals(readBackfilledBackend({}), undefined);
  assertEquals(readBackfilledBackend({ inference: {} }), undefined);
  assertEquals(
    readBackfilledBackend({ inference: { "other-app": { backend: "cpu" } } }),
    undefined,
  );
});

Deno.test("buildEngineArgs：mac 平台不加 --backend（即使 profile 裡有值）", () => {
  const profile = { inference: { "tomato-ears": { backend: "cpu" } } };
  const args = buildEngineArgs("macos-arm64", "/appdir/model", profile);
  assertEquals(args, ["--model", "/appdir/model", "--language", "auto"]);
});

Deno.test("buildEngineArgs：windows 平台、尚未回填 backend → 不加 --backend（讓 host 自行探測）", () => {
  const args = buildEngineArgs("windows-x64", "/appdir/model", {});
  assertEquals(args, ["--model", "/appdir/model", "--language", "auto"]);
});

Deno.test("buildEngineArgs：windows 平台、已回填 backend → 加 --backend 跳過重新探測", () => {
  const profile = { inference: { "tomato-ears": { backend: "dml" } } };
  const args = buildEngineArgs("windows-x64", "/appdir/model", profile);
  assertEquals(args, ["--model", "/appdir/model", "--language", "auto", "--backend", "dml"]);
});

/** 在暫時設定/清除 TMUH_APPS_DIR 的情況下執行 `fn`,結束後還原原值——
 * 避免測試順序影響彼此(env 是行程全域狀態)。 */
function withTmuhAppsDir(value: string | undefined, fn: () => void): void {
  const previous = Deno.env.get("TMUH_APPS_DIR");
  try {
    if (value === undefined) Deno.env.delete("TMUH_APPS_DIR");
    else Deno.env.set("TMUH_APPS_DIR", value);
    fn();
  } finally {
    if (previous === undefined) Deno.env.delete("TMUH_APPS_DIR");
    else Deno.env.set("TMUH_APPS_DIR", previous);
  }
}

Deno.test("resolveAppDir：未設定 TMUH_APPS_DIR 時 = Deno.cwd()（cwd 模型）", () => {
  withTmuhAppsDir(undefined, () => {
    assertEquals(resolveAppDir(), Deno.cwd());
  });
});

Deno.test("resolveAppDir：TMUH_APPS_DIR 設定時 = <override>/tomato-ears", () => {
  withTmuhAppsDir("/tmp/fake-tmuh-apps", () => {
    assertEquals(resolveAppDir(), "/tmp/fake-tmuh-apps/tomato-ears");
  });
});

Deno.test("resolveMachineProfilePath：未設定 TMUH_APPS_DIR 時 = ../_machine/machine-profile.json（cwd 相對）", () => {
  withTmuhAppsDir(undefined, () => {
    assertEquals(resolveMachineProfilePath(), "../_machine/machine-profile.json");
  });
});

Deno.test("resolveMachineProfilePath：TMUH_APPS_DIR 設定時 = <override>/_machine/machine-profile.json", () => {
  withTmuhAppsDir("/tmp/fake-tmuh-apps", () => {
    assertEquals(
      resolveMachineProfilePath(),
      "/tmp/fake-tmuh-apps/_machine/machine-profile.json",
    );
  });
});

Deno.test("readMachineProfile：檔案不存在時給出指向 env-base 的清楚錯誤訊息", async () => {
  await assertRejects(
    () => readMachineProfile("/tmp/definitely-not-here/machine-profile.json"),
    Error,
    "env-base",
  );
});

Deno.test("readMachineProfile：成功讀取並回傳解析後的物件", async () => {
  const dir = await Deno.makeTempDir();
  const path = `${dir}/machine-profile.json`;
  try {
    await Deno.writeTextFile(path, JSON.stringify({ os: "darwin", arch: "aarch64" }));
    const profile = await readMachineProfile(path);
    assertEquals(profile, { os: "darwin", arch: "aarch64" });
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("readMachineProfile：內容不是物件時 throw（例如純陣列）", async () => {
  const dir = await Deno.makeTempDir();
  const path = `${dir}/machine-profile.json`;
  try {
    await Deno.writeTextFile(path, JSON.stringify([1, 2, 3]));
    await assertRejects(() => readMachineProfile(path));
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("isSetupComplete：全新目錄（什麼都沒裝）回傳 false", async () => {
  const dir = await Deno.makeTempDir();
  try {
    assertEquals(await isSetupComplete(dir, "macos-arm64"), false);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("isSetupComplete：只有引擎沒有模型檔 → 仍視為未完成", async () => {
  const dir = await Deno.makeTempDir();
  try {
    await Deno.mkdir(`${dir}/bin`, { recursive: true });
    // setup 完成的判準是「穩定路徑」bin/engine-host(pin 的產物),
    // 不是壓縮包內的原始檔名——見 isSetupComplete 的 doc comment。
    await Deno.writeTextFile(`${dir}/bin/engine-host`, "fake");
    await Deno.mkdir(`${dir}/model`, { recursive: true }); // 目錄存在但是空的
    assertEquals(await isSetupComplete(dir, "macos-arm64"), false);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("isSetupComplete：原始檔名存在但穩定路徑不存在 → 未完成（pin 是 setup 的一部分）", async () => {
  const dir = await Deno.makeTempDir();
  try {
    await Deno.mkdir(`${dir}/bin`, { recursive: true });
    await Deno.writeTextFile(`${dir}/bin/catcher-asr-host`, "fake"); // 只有原始檔名
    await Deno.mkdir(`${dir}/model`, { recursive: true });
    await Deno.writeTextFile(`${dir}/model/weights.safetensors`, "fake weights");
    assertEquals(await isSetupComplete(dir, "macos-arm64"), false);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("isSetupComplete：穩定路徑引擎 + 至少一個模型檔都在 → true", async () => {
  const dir = await Deno.makeTempDir();
  try {
    await Deno.mkdir(`${dir}/bin`, { recursive: true });
    await Deno.writeTextFile(`${dir}/bin/engine-host`, "fake");
    await Deno.mkdir(`${dir}/model`, { recursive: true });
    await Deno.writeTextFile(`${dir}/model/weights.safetensors`, "fake weights");
    assertEquals(await isSetupComplete(dir, "macos-arm64"), true);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("writeBackfilledBackend：只動 inference['tomato-ears'].backend，其餘欄位原樣保留", async () => {
  const dir = await Deno.makeTempDir();
  const path = `${dir}/machine-profile.json`;
  try {
    await Deno.writeTextFile(
      path,
      JSON.stringify({
        os: "windows",
        arch: "x86_64",
        ramBytes: 17179869184,
        inference: { "other-app": { backend: "dml", benchmarkMs: 42 } },
      }),
    );

    await writeBackfilledBackend(path, "cpu");

    const rewritten = await readMachineProfile(path);
    assertObjectMatch(rewritten, {
      os: "windows",
      ramBytes: 17179869184,
      inference: {
        "other-app": { backend: "dml", benchmarkMs: 42 }, // 別的 app 的鍵不能被動到
        "tomato-ears": { backend: "cpu" },
      },
    });
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("writeBackfilledBackend：inference 欄位原本不存在也能建立", async () => {
  const dir = await Deno.makeTempDir();
  const path = `${dir}/machine-profile.json`;
  try {
    await Deno.writeTextFile(path, JSON.stringify({ os: "windows", arch: "x86_64" }));
    await writeBackfilledBackend(path, "dml");
    const rewritten = await readMachineProfile(path);
    assertEquals(readBackfilledBackend(rewritten), "dml");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("openBrowser：TMUH_NO_BROWSER 設定時完全不 spawn，只印跳過訊息 + URL", async () => {
  const url = "http://127.0.0.1:43117/";
  const logs: string[] = [];
  const errors: string[] = [];
  const originalLog = console.log;
  const originalError = console.error;
  console.log = (...args: unknown[]) => logs.push(args.map(String).join(" "));
  console.error = (...args: unknown[]) => errors.push(args.map(String).join(" "));

  const previous = Deno.env.get("TMUH_NO_BROWSER");
  try {
    Deno.env.set("TMUH_NO_BROWSER", "1");
    await openBrowser(url);
  } finally {
    console.log = originalLog;
    console.error = originalError;
    if (previous === undefined) Deno.env.delete("TMUH_NO_BROWSER");
    else Deno.env.set("TMUH_NO_BROWSER", previous);
  }

  // 跳過訊息(含 URL)走 console.log;console.error 必須完全沒動靜——
  // 本測試檔沒有 --allow-run,若 openBrowser 沒有真的短路而嘗試 spawn,
  // NotCapable 會被它內部捕捉並印出降級訊息到 console.error,這裡就會抓到。
  assertEquals(errors, []);
  assertEquals(logs.length, 1);
  assertEquals(logs[0].includes("TMUH_NO_BROWSER"), true);
  assertEquals(logs[0].includes(url), true);
});
