# Catcher-Tippi Mini-App 切分路線圖

日期:2026-07-19
狀態:方向定案(各 app 細部設計於各自 spec 展開)

## 整體方向

catcher-tippi 的資產逐步固化為 **tmuh.ai mini-app store 的配方(recipe)系列**:store 上發布的不是打包的應用程式,而是「詳細註解的參考程式碼 + 給 AI agent 的組裝指令 + 驗收測試」;使用者的 agent(Claude Code 等)照配方在自己電腦上組裝出應用。原生 app(macOS SwiftUI Tippi、Windows WPF Tippi)持續作為完整體驗版,mini-app 是可組裝、可堆疊的拆分版。

**共用地基(已完成)**:

- 凍結協定 `docs/protocol/asr-host-v1.md`(stdin/stdout JSON-lines;start/audio/stop → ready/partial/final/error)
- 引擎政策:mac = MLX prebuilt host、Windows = onnxruntime + DirectML 實測探測→CPU;engine host 一律 prebuilt + SHA-256 pin,agent 不編譯原生碼
- 店規九條與配方格式:見 `docs/store/2026-07-19-mini-app-store-handoff.md`
- 環境基座 `recipes/env-base/`(Deno、目錄慣例、machine-profile、agent 固化記憶)
- cwd 相對權限模型(Deno 旗標不展開 `~`、`--allow-run` 僅認精確路徑——實測釘死於 `permissions_probe_test.ts`)
- 品質迴圈:**配方必須通過「乾淨環境 + 零上下文 agent」的雙平台全程組裝演練**,卡點回填 PLAN.md(tomato-ears 的兩次演練各抓到一個對方平台原理上測不到的阻斷 bug:mac metallib 烤路徑、Windows `/C:/` 路徑)

## Mini-App 系列

| App | 功能 | 狀態 | 主要既有資產 |
|---|---|---|---|
| **tomato-ears** | 麥克風即時語音轉文字(瀏覽器 UI) | **v0.1.0 已完成**,已交付 store | `catcher-asr-host` v0.1.1(MLX)、`nemotron-asr-host` v0.1.0(ORT/DML)、`recipes/tomato-ears/` |
| **tomato-meeting** | 會議轉錄:說話者分離 + 分段逐字稿(誰說了什麼)、匯出 | 規劃中 | mac:`sortformer-mlx` + `catcher-diar-mlx-int8`(HF);win:`SpeakerDiarizer.cs` + pyannote segmentation(sherpa-onnx);`nemotron-mlx/fusion.rs`(token×diar 融合);Tippi 轉錄分頁 UI 模式 |
| **tomato-speaker** | 文字轉語音(輸入文字 → 播放/匯出 wav) | 規劃中 | win:`VoxCpmTtsService.cs` + VoxCPM2 模型 + `WavPlaybackService.cs`;mac 引擎待定(候選:VoxCPM2 ONNX 跨平台統一,或 MLX 移植) |
| **tomato-helper** | 語音助手:語音指令 + 本地 LLM 對話(候選組合:KWS 喚醒 + ASR + Agents-A1 本地模型) | 方向探索中 | KWS:sherpa-onnx zipformer(「幫我送出」已驗證);LLM:`Agents-A1-4B-MLX-4bit`(`codex/mac-agents-a1-chat` 進行中);ASR:沿用 tomato-ears 引擎 |

**不進 mini-app 的部分**:跨 app 文字注入(voice input)需要 OS Accessibility 特權,留在原生 Tippi;這是原生版的獨有價值。

## 堆疊原則

- 每個 mini-app = localhost 服務 + 瀏覽器 UI,port 與 API 合約宣告於 manifest,後者可組合前者(例:tomato-meeting 復用 tomato-ears 的 engine host 與協定,僅新增 diar host 或擴充事件)
- 協定擴充走版本化(asr-host-v1 凍結;diar/tts 需要新事件時開 v2 或平行協定文件,不改既有)
- 引擎 host 抽出模式已成熟:console host + stdin/stdout JSON-lines + FakeEngine 供無模型測試 + 黑箱協定測試,新 app 照抄 tomato-ears 的做法
- 每個 app 的模型/引擎一律 HF 或 GitHub Releases + hash pin(店規 9)

## 建構順序(暫定)

1. **tomato-meeting**——資產最齊(雙平台 diar 都有實作),且與 tomato-ears 共用 ASR 地基,是驗證「堆疊」的第一案
2. **tomato-speaker**——win 資產現成,mac 引擎選型是主要設計工作
3. **tomato-helper**——依賴 Agents-A1 工作(`codex/mac-agents-a1-chat`)收斂後再開 spec

每個 app 沿用既定流程:brainstorm → spec → plan → subagent-driven 實作(每 task 審查)→ 雙平台演練 → 終審 → 合併 → bundle 交付 store。
