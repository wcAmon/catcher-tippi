/**
 * `main.ts` 的測試:只測純邏輯/I/O 輔助函式(`platformFromProfile`、
 * `buildEngineArgs`、`readBackfilledBackend`、`resolveTmuhAppsRoot`、
 * `readMachineProfile`、`isSetupComplete`),不測 `if (import.meta.main)`
 * 區塊裡真正 spawn engine/起 server/開瀏覽器的組裝流程——那段是純粹的
 * glue,對齊 `recipes/env-base/probe/machine-profile.ts` 的既有慣例
 * (`if (import.meta.main)` 底下的程式碼不寫單元測試)。
 *
 * 執行本檔所需權限旗標(dev-time 測試):
 *   deno test --allow-read --allow-write --allow-env \
 *     recipes/tomato-ears/reference/main_test.ts
 * - `--allow-read`/`--allow-write`:`isSetupComplete`/`readMachineProfile`
 *   測試要建臨時目錄與檔案。
 * - `--allow-env`:`resolveTmuhAppsRoot` 測試要設/讀 `TMUH_APPS_DIR`、
 *   `HOME`/`USERPROFILE`(比正式 runtime 的
 *   `--allow-env=TMUH_APPS_DIR` 更寬,dev-time 測試不必比照 manifest)。
 */

import { assertEquals, assertRejects, assertThrows } from "jsr:@std/assert@^1.0.19";
import {
  buildEngineArgs,
  isSetupComplete,
  platformFromProfile,
  readBackfilledBackend,
  readMachineProfile,
  resolveTmuhAppsRoot,
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

Deno.test("resolveTmuhAppsRoot：TMUH_APPS_DIR 設定時整棵覆寫", () => {
  const previous = Deno.env.get("TMUH_APPS_DIR");
  try {
    Deno.env.set("TMUH_APPS_DIR", "/tmp/fake-tmuh-apps");
    assertEquals(resolveTmuhAppsRoot(), "/tmp/fake-tmuh-apps");
  } finally {
    if (previous === undefined) Deno.env.delete("TMUH_APPS_DIR");
    else Deno.env.set("TMUH_APPS_DIR", previous);
  }
});

Deno.test("resolveTmuhAppsRoot：未設定 TMUH_APPS_DIR 時退回 ~/tmuh-apps", () => {
  const previous = Deno.env.get("TMUH_APPS_DIR");
  try {
    Deno.env.delete("TMUH_APPS_DIR");
    const root = resolveTmuhAppsRoot();
    if (!root.endsWith("/tmuh-apps")) {
      throw new Error(`預期以 /tmuh-apps 結尾,實際:${root}`);
    }
  } finally {
    if (previous !== undefined) Deno.env.set("TMUH_APPS_DIR", previous);
  }
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
    await Deno.writeTextFile(`${dir}/bin/catcher-asr-host`, "fake");
    await Deno.mkdir(`${dir}/model`, { recursive: true }); // 目錄存在但是空的
    assertEquals(await isSetupComplete(dir, "macos-arm64"), false);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("isSetupComplete：引擎執行檔 + 至少一個模型檔都在 → true", async () => {
  const dir = await Deno.makeTempDir();
  try {
    await Deno.mkdir(`${dir}/bin`, { recursive: true });
    await Deno.writeTextFile(`${dir}/bin/catcher-asr-host`, "fake");
    await Deno.mkdir(`${dir}/model`, { recursive: true });
    await Deno.writeTextFile(`${dir}/model/weights.safetensors`, "fake weights");
    assertEquals(await isSetupComplete(dir, "macos-arm64"), true);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});
