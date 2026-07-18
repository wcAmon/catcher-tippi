# asr-host protocol v1

傳輸:stdin/stdout,JSON-lines(每行一則 JSON,UTF-8,`\n` 結尾)。
stderr 僅供人類閱讀的日誌,消費端必須忽略。

## Host 生命週期
1. 行程啟動 → 載入模型 → 輸出 `ready`(之前不得輸出任何 stdout 行)。
2. 之後接受指令。stdin EOF = 結束行程(exit code 0)。
3. 模型載入失敗:輸出 `error` 後以 exit code 1 結束。

## 指令(stdin →)
| 指令 | 格式 | 語義 |
|---|---|---|
| start | `{"cmd":"start","lang":"auto","sample_rate":16000}` | 開新會話。`lang`:`auto`/`en-US`/`zh-CN`/`zh-TW` 等 checkpoint locale。`sample_rate` 僅接受 16000。會話進行中再收 start → `error`(會話不中斷)。 |
| audio | `{"cmd":"audio","pcm16_b64":"<base64>"}` | mono 16 kHz PCM16-LE。建議每 chunk 1600 samples(100 ms)。無會話時收到 → `error`。 |
| stop | `{"cmd":"stop"}` | 沖洗解碼器,輸出 `final`,會話結束。無會話時收到 → `error`。 |

## 事件(stdout ←)
| 事件 | 格式 | 語義 |
|---|---|---|
| ready | `{"event":"ready","backend":"mlx"}` | 模型載入完成(win host 為 `"dml"` 或 `"cpu"`)。 |
| partial | `{"event":"partial","text":"..."}` | 會話累積轉錄的最新全文(非增量)。僅在有新 token 時輸出。 |
| final | `{"event":"final","text":"..."}` | stop 之後的定稿全文,一個會話恰好一次。 |
| error | `{"event":"error","message":"..."}` | 可恢復錯誤(格式錯、狀態錯)。行程不退出;致命錯誤才退出(exit 1)。 |

## 錯誤處理原則
- 無法解析的行 → `error`,繼續讀下一行。
- `pcm16_b64` 非法 base64 或位元組數為奇數 → `error`,會話保留。
- 文字一律 host 端已轉繁體(opencc s2twp);消費端不再轉換。
