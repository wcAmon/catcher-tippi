/**
 * `machine-profile.ts` 的單元測試。
 *
 * 執行(所需權限旗標;`Deno.test()` 的 `permissions` 選項只能收斂、
 * 不能升級父行程權限,所以必須在呼叫 `deno test` 時就給對——這就是
 * 「權限旗標各測試檔自帶」的意思:旗標寫在這份檔案的說明裡,由執行者照抄):
 *   deno test --allow-read --allow-write --allow-sys=systemMemoryInfo \
 *     recipes/env-base/probe/machine-profile_test.ts
 */

import { assertEquals, assertObjectMatch } from "jsr:@std/assert@^1.0.19";
import {
  buildProbeFields,
  mergeProfile,
  probeAndWrite,
  type ProbeFields,
} from "./machine-profile.ts";

const fixedProbe: ProbeFields = {
  probedAt: "2026-01-01T00:00:00.000Z",
  os: "darwin",
  arch: "aarch64",
  ramBytes: 17179869184,
  cpuThreads: 8,
  denoVersion: "2.9.3",
};

Deno.test("mergeProfile：全新安裝(existing 為 undefined)只產出 probe 欄位", () => {
  const merged = mergeProfile(undefined, fixedProbe);
  assertEquals(merged, { ...fixedProbe });
});

Deno.test("mergeProfile：保留 existing 裡本腳本不認識的欄位(例如 app 回填的 inference)", () => {
  const existing = {
    os: "darwin",
    arch: "aarch64",
    ramBytes: 8589934592, // 舊值，應被探測值覆寫
    inference: { backend: "mlx", benchmarkMs: 42 }, // app 首跑回填，不應被動到
  };
  const merged = mergeProfile(existing, fixedProbe);
  assertEquals(merged.inference, { backend: "mlx", benchmarkMs: 42 });
  assertEquals(merged.ramBytes, fixedProbe.ramBytes);
  assertEquals(merged.denoVersion, fixedProbe.denoVersion);
});

Deno.test(
  "buildProbeFields：型別與基本合理性(需要 --allow-sys=systemMemoryInfo)",
  () => {
    const probe = buildProbeFields();
    assertEquals(typeof probe.os, "string");
    assertEquals(typeof probe.arch, "string");
    assertEquals(typeof probe.denoVersion, "string");
    if (probe.ramBytes <= 0) {
      throw new Error("ramBytes 應為正數，實測: " + probe.ramBytes);
    }
    if (probe.cpuThreads <= 0) {
      throw new Error("cpuThreads 應為正數，實測: " + probe.cpuThreads);
    }
  },
);

Deno.test(
  "probeAndWrite：冪等 merge——預先寫入含未知欄位的檔案，重跑後該欄位仍在",
  async () => {
    const dir = await Deno.makeTempDir();
    const path = `${dir}/machine-profile.json`;

    // 模擬「app 已首跑回填 inference 欄位」的既有 profile。
    await Deno.writeTextFile(
      path,
      JSON.stringify({
        os: "darwin",
        arch: "aarch64",
        ramBytes: 1, // 刻意設一個假的舊值，驗證探測會覆寫它
        inference: { backend: "mlx", benchmarkMs: 42 },
      }),
    );

    const merged = await probeAndWrite(path);

    assertObjectMatch(merged, {
      inference: { backend: "mlx", benchmarkMs: 42 },
    });
    if ((merged.ramBytes as number) <= 1) {
      throw new Error("probe 應覆寫 ramBytes 為實測值，而非保留舊的 1");
    }

    // 驗證寫入磁碟的內容與回傳值一致（下一輪重跑仍能讀到 inference）。
    const rewritten = JSON.parse(await Deno.readTextFile(path));
    assertObjectMatch(rewritten, {
      inference: { backend: "mlx", benchmarkMs: 42 },
    });

    await Deno.remove(dir, { recursive: true });
  },
);

Deno.test(
  "probeAndWrite：目標目錄不存在時自動建立（全新安裝情境）",
  async () => {
    const dir = await Deno.makeTempDir();
    const path = `${dir}/_machine/machine-profile.json`; // 子目錄尚未存在

    const merged = await probeAndWrite(path);

    assertEquals(typeof merged.os, "string");
    const onDisk = JSON.parse(await Deno.readTextFile(path));
    assertEquals(onDisk.os, merged.os);

    await Deno.remove(dir, { recursive: true });
  },
);
