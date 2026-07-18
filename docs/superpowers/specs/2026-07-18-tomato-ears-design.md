# tomato-ears Mini-App 設計(含 Engine Host 抽出)

日期:2026-07-18
狀態:草案(待 wake 審核)
前置:`2026-07-18-mini-app-store-design.md`(店規、配方格式、引擎政策)

## 1. 目標

把 catcher-tippi 的既有資產固化成 mini-app store 的第一個配方:
一個跨平台(macOS arm64 / Windows x64)的 **Nemotron ASR 即時轉錄應用**,
瀏覽器 UI + Deno 本機服務,由使用者的 agent 依配方組裝完成。

「固化既有資產」的含義:推論核心不重寫、不讓 agent 碰——
mac 用已驗證的 MLX runtime(catcher),Windows 用已驗證的
onnxruntime + DirectML 路線(`codex/windows-auto-backend` 分支),
兩者抽成獨立的 console engine host 發布。

## 2. 架構

```
瀏覽器 UI(getUserMedia 錄音、轉錄訊息列表、匯出 txt/json)
   ↕ WebSocket(127.0.0.1)
Deno 服務                                ← agent 依配方組裝的部分
   - 靜態 UI serve + 自動開瀏覽器
   - 首次啟動:下載 engine host + 模型,驗 SHA-256,原子安裝
   - engine host 子行程管理(啟動、重啟、逾時)
   ↕ stdin/stdout JSON-lines 協定(兩平台同一份合約)
Engine host(prebuilt,hash pin)          ← wake 發布的 binary
   - mac: catcher-asr-host(MLX / Metal)
   - win: nemotron-asr-host(onnxruntime-genai + DirectML→CPU)
```

## 3. Engine Host 抽出(catcher-tippi 端的新工程)

### 3.1 macOS:`catcher-asr-host`

- 來源:`crates/nemotron-cli` / `catcher-ffi` 既有能力,包一個 console 進入點;
- 模型:`wcamon/catcher-asr-mlx-int8`(HF,≈629 MiB,沿用既有 SHA-256 pin 機制);
- 發布:GitHub Releases(catcher-tippi repo),附 SHA-256。

### 3.2 Windows:`nemotron-asr-host`

- 來源:`codex/windows-auto-backend` 分支的 `NemotronEngine.cs`、
  `InferenceBackend.cs`(含 `InferenceBackendPolicy` 實測探測)、
  `ModelInstaller.cs` 抽出為 .NET console 專案(self-contained 單檔發布,
  含 onnxruntime / DirectML / D3D12Core DLL 與授權檔);
- 模型:`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`(公開 artifact);
- 後端選擇:沿用 InferenceBackendPolicy——DirectML 與 CPU 都實測,
  DML 需贏過門檻(0.85)才採用;結果回報給 Deno 寫進 machine-profile;
- 發布:GitHub Releases,附 SHA-256。

### 3.3 stdin/stdout 協定(v1)

JSON-lines,每行一則訊息;音訊用 base64 內嵌(v1 取簡單,之後再評估二進位框架):

```
→ {"cmd":"start","lang":"auto","sample_rate":16000}
→ {"cmd":"audio","pcm16_b64":"...."}          # 16 kHz mono PCM,每 chunk ≈100ms
→ {"cmd":"stop"}
← {"event":"ready","backend":"mlx|dml|cpu"}
← {"event":"partial","text":"..."}
← {"event":"final","text":"..."}
← {"event":"error","message":"..."}
```

合約版本寫進 manifest 的 `ports` 欄位;兩平台 host 行為以同一份協定測試驗證。

## 4. Deno 服務(agent 組裝的部分)

reference/ 內提供 95% 完成、詳細註解的模組:

| 模組 | 職責 |
|---|---|
| `server.ts` | HTTP + WebSocket,只綁 127.0.0.1,啟動後自動開瀏覽器 |
| `downloader.ts` | 依 manifest 下載 engine host + 模型,SHA-256 驗證,原子安裝 |
| `engine.ts` | 子行程生命週期 + 協定編解碼(平台分支只在「選哪個 binary」一行) |
| `ui/` | 錄音頁:開始/停止、部分結果即時顯示、訊息列表、複製/匯出 txt·json |

agent 的黏合工作僅限:讀 machine-profile 填平台參數、串接模組、跑 verify。

Deno 權限宣告(manifest 同步):
`--allow-net=127.0.0.1,huggingface.co,github.com --allow-read=~/tmuh-apps/tomato-ears --allow-write=~/tmuh-apps/tomato-ears --allow-run=<engine-host路徑>`

## 5. 驗收測試(verify/)

1. **下載完整性**:所有 dependencies 檔案存在且 SHA-256 相符;
2. **協定測試**:對 engine host 送 fixture wav(附在配方內,約 10 秒中文/英文各一),
   斷言 final 文字與參考轉錄的相似度門檻(非逐字比對,容忍小差異);
3. **服務測試**:啟動 Deno 服務,WebSocket 走完 start→audio→stop 全流程;
4. **綁定檢查**:服務不可從非 127.0.0.1 介面連上;
5. **權限檢查**:實際執行旗標 == manifest 宣告(SECURITY.md 的機械化部分)。

全數通過 agent 才可宣告完成(店規第 6 條)。

## 6. v1 範圍

**做**:麥克風即時轉錄、部分/最終結果顯示、訊息列表、複製與匯出 txt/json、
繁中輸出(host 內建 opencc s2twp,沿用既有實作)。

**不做**(留給後續 mini-app):說話者分離(diarizer)、TTS(tts-speaker,
Windows 分支的 VoxCPM2 資產屆時同法抽 host)、語音輸入注入(留在原生 Tippi)、
檔案批次轉錄、Linux 支援。

## 7. 建構順序

1. catcher-tippi:抽出並發布兩個 engine host(3.1、3.2)+ 協定測試;
2. 撰寫配方包(reference/、verify/、SPEC/PLAN/SECURITY/manifest);
3. 在乾淨的 mac 與 Windows 機器上各做一次「agent 全程組裝」演練,
   記錄 agent 卡住的每一點,回填 PLAN.md 與註解;
4. tmuh.ai 端 store 實作文件(依店規 spec 第 7 節展開)交伺服器 Claude Code。
