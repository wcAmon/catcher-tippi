/**
 * 驗收測試:確認真正啟動的服務(真 host 真模型,同 `service_test.ts` 的
 * 組裝方式)只在 127.0.0.1 監聽——從本機的另一個網路介面(LAN IPv4)嘗試
 * 連線應該被作業系統拒絕(ECONNREFUSED),而不是被 Deno 權限層擋下來
 * (若是權限層先擋,不能證明「作業系統真的沒有在那個介面監聽」,那是完全
 * 不同的失敗原因)。這正是 `deno task verify:*` 的 `--allow-net` 在這個
 * task 家族裡刻意比 `start:*`/`setup:*` 更寬(不縮圈)的唯一理由——見
 * SECURITY.md「verify 與 start permission 差異」章節的完整說明。
 *
 * 沒有非 loopback IPv4 網卡的機器(例如某些 CI 沙箱)會自動 `ignore` 這項
 * ——這是既有 `reference/server_test.ts` 綁定測試的同款容錯,不是本測試
 * 刻意放水。
 *
 * 權限:同 `real_service.ts` 需要的一切 + `--allow-sys=networkInterfaces`
 * (列出本機網卡找一個非 loopback IPv4 位址)+ `--allow-net`(嘗試連到
 * 那個位址)。
 */
import { assertRejects } from "jsr:@std/assert@^1.0.19";
import { startRealService } from "./real_service.ts";

const APP_DIR = Deno.cwd();

/** 找一個本機非 loopback 的 IPv4 位址;找不到時回傳 undefined(對應測試
 * 自動 ignore)——同 `reference/server_test.ts` 的既有慣例。 */
function findLanIPv4(): string | undefined {
  const candidates = Deno.networkInterfaces().filter(
    (iface) => iface.family === "IPv4" && iface.address !== "127.0.0.1",
  );
  return candidates[0]?.address;
}

const lanIPv4 = findLanIPv4();

Deno.test({
  name: "binding：服務只綁 127.0.0.1——透過本機 LAN 介面位址連線應被拒絕",
  ignore: lanIPv4 === undefined,
  fn: async () => {
    const service = await startRealService(APP_DIR);
    try {
      // 預期是 Deno.errors.ConnectionRefused(OS 層級拒絕),不特別限定
      // Error 子類別——不同作業系統對「連到沒人監聽的位址」的錯誤型別
      // 可能有些微差異,這裡只在意「一定會 reject」這件事本身。
      await assertRejects(() => Deno.connect({ hostname: lanIPv4!, port: service.port }));
    } finally {
      await service.shutdown();
    }
  },
});
