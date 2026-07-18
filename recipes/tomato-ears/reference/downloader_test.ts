/**
 * `downloader.ts` 的測試:用 Deno 內建 HTTP server 在本機提供假的 engine
 * 壓縮包與模型檔案,涵蓋店規要求的三種情境——雜湊對、雜湊錯、斷點殘檔。
 *
 * 執行本檔所需權限旗標(dev-time 測試,涵蓋建 fixture 用的 tar 打包 +
 * 下載器本身解壓用的 tar 呼叫):
 *   deno test --allow-net --allow-read --allow-write --allow-run=tar \
 *     recipes/tomato-ears/reference/downloader_test.ts
 */

import { assertEquals, assertExists, assertRejects } from "jsr:@std/assert@^1.0.19";
import { encodeHex } from "jsr:@std/encoding@^1.0.11/hex";
import {
  engineBinaryName,
  type EngineDependency,
  ensureDependencies,
  type Manifest,
  type ManifestFileEntry,
  resolveEngineBinaryPath,
  stableEngineBinaryPath,
} from "./downloader.ts";

/** 對一段記憶體中的 bytes 算 SHA-256(hex)。全域 `crypto.subtle.digest`
 * 原生就吃 `BufferSource`,fixture 資料量小,不需要 downloader.ts 內
 * 為了大檔案才用的串流版本。
 *
 * why 用 `new Uint8Array(bytes)` 重新包一層:`Deno.readFile()` 等 API 回傳的
 * `Uint8Array` 型別參數是較寬的 `ArrayBufferLike`(涵蓋 SharedArrayBuffer),
 * 跟 `crypto.subtle.digest`/`Response` 建構子要求的 `BufferSource`
 * (`ArrayBuffer` 特化版)在型別檢查上對不上——實際執行沒有問題(這些
 * bytes 從來不是真的 SharedArrayBuffer),重新包一層純粹是滿足型別檢查。 */
async function sha256Hex(bytes: Uint8Array): Promise<string> {
  return encodeHex(await crypto.subtle.digest("SHA-256", new Uint8Array(bytes)));
}

/** 一個「路徑 → bytes」的本機 HTTP server,並記錄每個路徑被要求的次數
 * (用來驗證「已存在且雜湊符的檔案應該完全不再發出下載請求」的冪等行為)。 */
function startFixtureServer(files: Map<string, Uint8Array>) {
  const requestCounts = new Map<string, number>();
  const server = Deno.serve(
    { hostname: "127.0.0.1", port: 0, onListen: () => {} },
    (request) => {
      const pathname = new URL(request.url).pathname;
      requestCounts.set(pathname, (requestCounts.get(pathname) ?? 0) + 1);
      const bytes = files.get(pathname);
      if (bytes === undefined) return new Response("not found", { status: 404 });
      return new Response(new Uint8Array(bytes)); // 見 sha256Hex 的型別 why 註解
    },
  );
  const port = (server.addr as Deno.NetAddr).port;
  return { port, requestCounts, shutdown: () => server.shutdown() };
}

/** 建一個內含單一檔案(檔名對齊 macos-arm64 的引擎執行檔名)的 tar.gz,
 * 模擬 engine host release 壓縮包。用系統 `tar` 打包,與 downloader.ts
 * 解壓時用的是同一個工具,兩邊格式保證一致。 */
async function buildFakeEngineArchive(): Promise<Uint8Array> {
  const stageDir = await Deno.makeTempDir({ prefix: "tomato-ears-fixture-" });
  try {
    const binaryName = engineBinaryName("macos-arm64");
    await Deno.writeTextFile(`${stageDir}/${binaryName}`, "#!/bin/sh\necho fake engine\n");
    const archivePath = `${stageDir}/archive.tar.gz`;
    const command = new Deno.Command("tar", {
      args: ["-czf", archivePath, "-C", stageDir, binaryName],
    });
    const { success, stderr } = await command.output();
    if (!success) {
      throw new Error(`建測試 fixture 失敗:${new TextDecoder().decode(stderr)}`);
    }
    return await Deno.readFile(archivePath);
  } finally {
    await Deno.remove(stageDir, { recursive: true });
  }
}

interface Fixtures {
  engineArchive: Uint8Array;
  engineDep: EngineDependency;
  modelFileA: { bytes: Uint8Array; entry: ManifestFileEntry };
  modelFileB: { bytes: Uint8Array; entry: ManifestFileEntry };
}

/** 準備一套「完全正確」的假引擎壓縮包 + 兩個假模型檔(用來組出各測試情境
 * 的 Manifest);測試視需要另外竄改個別欄位來製造雜湊不符等情境。 */
async function buildFixtures(baseUrl: string): Promise<Fixtures> {
  const engineArchive = await buildFakeEngineArchive();
  const modelBytesA = new TextEncoder().encode("model file A content");
  const modelBytesB = new TextEncoder().encode("model file B content, a bit longer");

  return {
    engineArchive,
    engineDep: {
      url: `${baseUrl}/engine.tar.gz`,
      sha256: await sha256Hex(engineArchive),
      byteCount: engineArchive.byteLength,
    },
    modelFileA: {
      bytes: modelBytesA,
      entry: {
        name: "a.bin",
        sha256: await sha256Hex(modelBytesA),
        byteCount: modelBytesA.byteLength,
      },
    },
    modelFileB: {
      bytes: modelBytesB,
      entry: {
        name: "b.bin",
        sha256: await sha256Hex(modelBytesB),
        byteCount: modelBytesB.byteLength,
      },
    },
  };
}

function buildManifest(fixtures: Fixtures): Manifest {
  return {
    name: "tomato-ears-test",
    version: "0.0.0",
    dependencies: {
      engine: {
        "macos-arm64": fixtures.engineDep,
        // windows-x64 分支本測試不使用,但 Manifest 型別要求兩個平台鍵齊全
        // (與 manifest.json 的實際 schema 一致),放一份無效但型別正確的假資料。
        "windows-x64": fixtures.engineDep,
      },
      model: {
        "macos-arm64": {
          repo: "test/repo",
          baseUrl: "http://127.0.0.1:0/unused/", // 本測試只走 macos-arm64 分支
          files: [fixtures.modelFileA.entry, fixtures.modelFileB.entry],
        },
        "windows-x64": {
          repo: "test/repo",
          baseUrl: "http://127.0.0.1:0/unused/",
          files: [],
        },
      },
    },
    verify: "deno task verify",
  };
}

Deno.test("ensureDependencies：雜湊對——下載落地、可解壓、重跑不再發請求(冪等)", async () => {
  const fixtures = await buildFixtures("PLACEHOLDER");
  const files = new Map<string, Uint8Array>([
    ["/engine.tar.gz", fixtures.engineArchive],
    ["/model/a.bin", fixtures.modelFileA.bytes],
    ["/model/b.bin", fixtures.modelFileB.bytes],
  ]);
  const fixtureServer = startFixtureServer(files);
  const baseUrl = `http://127.0.0.1:${fixtureServer.port}`;
  fixtures.engineDep.url = `${baseUrl}/engine.tar.gz`;
  const manifest = buildManifest(fixtures);
  manifest.dependencies.model["macos-arm64"].baseUrl = `${baseUrl}/model/`;

  const appDir = await Deno.makeTempDir({ prefix: "tomato-ears-appdir-" });
  try {
    const progressLines: string[] = [];
    await ensureDependencies(manifest, appDir, "macos-arm64", (msg) => progressLines.push(msg));

    // engine host 已解壓且可被 resolveEngineBinaryPath 找到。
    const binPath = await resolveEngineBinaryPath(appDir, "macos-arm64");
    assertEquals(await Deno.readTextFile(binPath), "#!/bin/sh\necho fake engine\n");

    // 執行檔已 pin 到穩定路徑 bin/engine-host(--allow-run 靜態宣告的對象),
    // 內容與原始落點一致。
    const stablePath = stableEngineBinaryPath(appDir, "macos-arm64");
    assertEquals(await Deno.readTextFile(stablePath), "#!/bin/sh\necho fake engine\n");

    // 模型檔落地且內容正確。
    assertEquals(
      await Deno.readFile(`${appDir}/model/a.bin`),
      fixtures.modelFileA.bytes,
    );
    assertEquals(
      await Deno.readFile(`${appDir}/model/b.bin`),
      fixtures.modelFileB.bytes,
    );
    assertExists(progressLines.find((line) => line.includes("下載完成")));

    // 冪等:重跑一次,已存在且雜湊符的檔案不該再發出任何 HTTP 請求。
    const countsBeforeRerun = new Map(fixtureServer.requestCounts);
    await ensureDependencies(manifest, appDir, "macos-arm64");
    assertEquals(fixtureServer.requestCounts, countsBeforeRerun);

    // 穩定路徑被誤刪 → 重跑 setup 應能從原始落點重新 pin,仍然零 HTTP 請求
    // (壓縮包與解壓後的原樹都還在,不需要重新下載)。
    await Deno.remove(stablePath);
    await ensureDependencies(manifest, appDir, "macos-arm64");
    assertEquals(await Deno.readTextFile(stablePath), "#!/bin/sh\necho fake engine\n");
    assertEquals(fixtureServer.requestCounts, countsBeforeRerun);
  } finally {
    await fixtureServer.shutdown();
    await Deno.remove(appDir, { recursive: true });
  }
});

Deno.test("ensureDependencies：雜湊錯——刪除殘檔並 throw，後續檔案不會被下載(fail-fast)", async () => {
  const fixtures = await buildFixtures("PLACEHOLDER");
  // 刻意把 a.bin 的宣告雜湊改錯(伺服器實際吐出的內容不變)。
  fixtures.modelFileA.entry.sha256 = "0".repeat(64);

  const files = new Map<string, Uint8Array>([
    ["/engine.tar.gz", fixtures.engineArchive],
    ["/model/a.bin", fixtures.modelFileA.bytes],
    ["/model/b.bin", fixtures.modelFileB.bytes],
  ]);
  const fixtureServer = startFixtureServer(files);
  const baseUrl = `http://127.0.0.1:${fixtureServer.port}`;
  fixtures.engineDep.url = `${baseUrl}/engine.tar.gz`;
  const manifest = buildManifest(fixtures);
  manifest.dependencies.model["macos-arm64"].baseUrl = `${baseUrl}/model/`;

  const appDir = await Deno.makeTempDir({ prefix: "tomato-ears-appdir-" });
  try {
    await assertRejects(
      () => ensureDependencies(manifest, appDir, "macos-arm64"),
      Error,
      "SHA-256",
    );

    // a.bin 雜湊校驗失敗:成品與殘檔都不該留下。
    await assertRejects(() => Deno.stat(`${appDir}/model/a.bin`));
    await assertRejects(() => Deno.stat(`${appDir}/model/a.bin.part`));

    // 逐檔循序處理、失敗即停:排在 a.bin 之後的 b.bin 不該被碰過。
    await assertRejects(() => Deno.stat(`${appDir}/model/b.bin`));
  } finally {
    await fixtureServer.shutdown();
    await Deno.remove(appDir, { recursive: true });
  }
});

Deno.test("ensureDependencies：斷點殘檔——預先留下損毀的 .part 也能重下成功", async () => {
  const fixtures = await buildFixtures("PLACEHOLDER");
  const files = new Map<string, Uint8Array>([
    ["/engine.tar.gz", fixtures.engineArchive],
    ["/model/a.bin", fixtures.modelFileA.bytes],
    ["/model/b.bin", fixtures.modelFileB.bytes],
  ]);
  const fixtureServer = startFixtureServer(files);
  const baseUrl = `http://127.0.0.1:${fixtureServer.port}`;
  fixtures.engineDep.url = `${baseUrl}/engine.tar.gz`;
  const manifest = buildManifest(fixtures);
  manifest.dependencies.model["macos-arm64"].baseUrl = `${baseUrl}/model/`;

  const appDir = await Deno.makeTempDir({ prefix: "tomato-ears-appdir-" });
  try {
    // 模擬前一次下載中斷留下的殘檔:大小、內容都跟正確檔案對不上。
    await Deno.mkdir(`${appDir}/model`, { recursive: true });
    await Deno.writeTextFile(`${appDir}/model/a.bin.part`, "殘缺不全的舊資料");

    await ensureDependencies(manifest, appDir, "macos-arm64");

    assertEquals(
      await Deno.readFile(`${appDir}/model/a.bin`),
      fixtures.modelFileA.bytes,
    );
    // .part 不該殘留——成功後已 rename 成正式檔名。
    await assertRejects(() => Deno.stat(`${appDir}/model/a.bin.part`));
  } finally {
    await fixtureServer.shutdown();
    await Deno.remove(appDir, { recursive: true });
  }
});
