/**
 * ASR 評測用文字正規化與正規化 Levenshtein 距離——從
 * `crates/catcher-asr-host/tests/real_model.rs`(Rust,`normalize_for_asr`/
 * `is_punct`/`normalized_levenshtein`,凍結行為)逐一移植過來的**第三份
 * 獨立實作**(第一份是 Rust 原版;第二份是 Windows 端 `.NET`/C# 的等價
 * 實作,見 `docs/superpowers/plans/2026-07-19-nemotron-asr-host.md`
 * 「RealModelTests.cs」段落)。三份實作必須語義一致(同一組標點集合、同一種
 * 演算法、同一個 0.25 門檻),`asr_metric_test.ts` 用固定案例把這份 TS
 * port 釘住,防止未來任一份漂移。
 *
 * why 需要三份獨立實作而非共用一個套件:Rust(engine host 本體)、C#
 * (Windows host 測試)、TS(這份 Deno 配方的驗收套件)三個語言各自服務
 * 不同的執行環境,彼此之間沒有共用 runtime 可以 import 同一份程式碼——
 * 寧可接受三份實作的維護成本(用測試案例對齊行為),也不要為了「不重複」
 * 硬拉一個跨語言依賴。
 */

/**
 * 對齊 Rust `char::is_ascii_punctuation()` 的精確範圍(逐一抄自標準庫文件):
 * U+0021..=U+002F、U+003A..=U+0040、U+005B..=U+0060、U+007B..=U+007E。
 * 用 code point 數值範圍比較而非正規表示式,避免正規表示式在字元類別裡
 * 對 `/`、`[`、`]`、`\` 等符號的跳脫規則寫錯而悄悄漏收/多收字元。
 */
function isAsciiPunctuation(codePoint: number): boolean {
  return (
    (codePoint >= 0x21 && codePoint <= 0x2f) ||
    (codePoint >= 0x3a && codePoint <= 0x40) ||
    (codePoint >= 0x5b && codePoint <= 0x60) ||
    (codePoint >= 0x7b && codePoint <= 0x7e)
  );
}

/** 逐字對照 Rust 版 `is_punct` 裡的中文全形標點 match 分支。 */
const CJK_PUNCT = new Set([
  "，",
  "。",
  "、",
  "；",
  "：",
  "?",
  "!",
  "「",
  "」",
  "『",
  "』",
  "（",
  "）",
  "《",
  "》",
  "…",
  "—",
  "·",
  "？",
  "！",
]);

/** 單一字元(Unicode code point 字串)是否視為標點,對齊 Rust 版 `is_punct`。 */
function isPunct(ch: string): boolean {
  if (CJK_PUNCT.has(ch)) return true;
  const codePoint = ch.codePointAt(0);
  return codePoint !== undefined && isAsciiPunctuation(codePoint);
}

/**
 * 比對前的正規化:去除空白與標點(ASR 評測慣例——格式差異不是辨識錯誤,
 * 但數字格式差異如「三百六十五」vs「365」仍計為錯誤,對齊 Rust 版
 * `normalize_for_asr` 的同款註解與行為)。用 `.trim().length === 0`
 * 判斷單一字元是否為空白,涵蓋一般空格/tab/換行等常見空白,足夠本場景
 * (fixture 文字皆為一般英文/中文句子,不含罕見 Unicode 空白變體)。
 */
export function normalizeForAsr(text: string): string {
  return Array.from(text)
    .filter((ch) => ch.trim().length !== 0 && !isPunct(ch))
    .join("");
}

/**
 * 字元(Unicode code point)級 Levenshtein 距離除以較長字串的字元數
 * (0.0 = 完全相同,1.0 = 完全不同),對齊 Rust 版 `normalized_levenshtein`
 * 的演算法與邊界情況(兩邊皆空字串 → 0.0)。
 *
 * why 用 `Array.from(text)` 取字元而非字串索引/`.length`:JS 字串以
 * UTF-16 code unit 為單位,補充平面字元(surrogate pair,例如部分罕見
 * CJK 擴展區字元)用字串索引會被拆成兩個 code unit,產生錯誤的字元數與
 * 距離計算;`Array.from` 依 Unicode code point 切分,與 Rust `chars()`
 * (Unicode scalar value)語意對齊。
 */
export function normalizedLevenshtein(a: string, b: string): number {
  const charsA = Array.from(a);
  const charsB = Array.from(b);
  if (charsA.length === 0 && charsB.length === 0) return 0;

  let prev: number[] = Array.from({ length: charsB.length + 1 }, (_, i) => i);
  let curr: number[] = new Array(charsB.length + 1).fill(0);

  for (let i = 0; i < charsA.length; i++) {
    curr[0] = i + 1;
    for (let j = 0; j < charsB.length; j++) {
      const cost = charsA[i] === charsB[j] ? 0 : 1;
      curr[j + 1] = Math.min(prev[j] + cost, prev[j + 1] + 1, curr[j] + 1);
    }
    [prev, curr] = [curr, prev];
  }

  return prev[charsB.length] / Math.max(charsA.length, charsB.length);
}
