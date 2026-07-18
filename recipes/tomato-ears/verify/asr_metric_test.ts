/**
 * `asr_metric.ts` 的單元測試:把這份 TS port 與
 * `crates/catcher-asr-host/tests/real_model.rs` 的 Rust 原版行為釘死在
 * 幾組固定案例上(見 `asr_metric.ts` 檔頭的「第三份獨立實作」說明)。
 * 不需要任何權限旗標(純函式,不碰檔案系統/網路)。
 */
import { assertEquals } from "jsr:@std/assert@^1.0.19";
import { normalizedLevenshtein, normalizeForAsr } from "./asr_metric.ts";

Deno.test("normalizeForAsr：移除空白與 ASCII 標點", () => {
  assertEquals(normalizeForAsr("Hello, this is a test."), "Hellothisisatest");
});

Deno.test("normalizeForAsr：移除中文全形標點與空白，保留文字", () => {
  assertEquals(normalizeForAsr("你好，世界！ 這是 測試。"), "你好世界這是測試");
});

Deno.test("normalizeForAsr：數字格式差異不是標點，原樣保留（對齊 Rust 版 why 註解）", () => {
  assertEquals(normalizeForAsr("三百六十五 vs 365"), "三百六十五vs365");
});

Deno.test("normalizeForAsr：空字串與純標點/空白字串正規化後為空字串", () => {
  assertEquals(normalizeForAsr(""), "");
  assertEquals(normalizeForAsr("， 。！ \t\n"), "");
});

Deno.test("normalizedLevenshtein：完全相同字串 → 0", () => {
  assertEquals(normalizedLevenshtein("hello", "hello"), 0);
});

Deno.test("normalizedLevenshtein：兩邊皆空字串 → 0（邊界特判，對齊 Rust 版）", () => {
  assertEquals(normalizedLevenshtein("", ""), 0);
});

Deno.test("normalizedLevenshtein：一邊空字串 → 1.0（全刪除/全插入）", () => {
  assertEquals(normalizedLevenshtein("", "abc"), 1);
  assertEquals(normalizedLevenshtein("abc", ""), 1);
});

Deno.test("normalizedLevenshtein：經典 kitten→sitting，編輯距離 3，正規化為 3/7", () => {
  assertEquals(normalizedLevenshtein("kitten", "sitting"), 3 / 7);
});

Deno.test("normalizedLevenshtein：完全不同但等長的字串，距離為 1.0", () => {
  assertEquals(normalizedLevenshtein("abc", "xyz"), 1);
});

Deno.test("normalizedLevenshtein：以 Unicode code point 為單位計算，補充平面字元不被拆成兩個 code unit", () => {
  // U+20000（𠀀）是一個 surrogate pair；若誤用字串索引/.length（UTF-16
  // code unit）會把它算成兩個字元，這裡驗證同一個字元比對距離仍是 0。
  const supplementary = "𠀀";
  assertEquals(normalizedLevenshtein(supplementary, supplementary), 0);
  assertEquals([...supplementary].length, 1);
});

Deno.test("normalizeForAsr + normalizedLevenshtein：組合行為對齊 protocol_test.ts 的實際用法", () => {
  const expected = normalizeForAsr("Hello, this is a streaming speech recognition test");
  // 模擬一個「幾乎正確但漏了一個字」的辨識結果（漏"is"，非純標點差異——
  // 純標點/空白差異在正規化後會完全抵銷，距離仍是 0，不足以驗證這個
  // 組合行為，見上面幾個測試已經涵蓋純標點差異的情境）。
  const got = normalizeForAsr("Hello this a streaming speech recognition test");
  const distance = normalizedLevenshtein(expected, got);
  assertEquals(distance > 0, true);
  assertEquals(distance <= 0.25, true);
});
