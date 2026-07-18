# Tippi for Windows（CPU / GPU 自動選擇）

這是原本 macOS SwiftUI/MLX 應用程式的 Windows 原生版本。介面使用 WPF，並讓四個
本機模型共存：Nemotron 3.5 INT4 ONNX 負責 ASR、5.4 MB sherpa-onnx KWS 獨立偵測
「幫我送出」、Pyannote segmentation INT8 + NVIDIA TitaNet-S ONNX 負責說話者分離，
VoxCPM2 BaseLM Q8 + Acoustic F16 負責文字轉語音。ASR、KWS 與說話者模型常駐 CPU；
TTS 預設在獨立 process 自動嘗試 Vulkan GPU，載入失敗或沒有可用 GPU 時改用 CPU。
程式不需要 CUDA，也能在沒有獨立顯示卡的 Windows x64 PC 上使用。

## 使用需求

- Windows 10 或 Windows 11，x64 處理器
- VoxCPM2 的 CPU fallback 需要 AVX2／FMA／F16C／BMI2（約 2013 年後的主流 x64 CPU）
- 只使用 ASR／Voice Command／Diarization：建議至少 8 GB RAM，保留約 4 GB 可用記憶體
- 同時載入 VoxCPM2：建議 16 GB 以上 RAM；Vulkan 所需 VRAM 依驅動與上下文而異
- 首次啟動下載約 829 MiB ASR、KWS 與說話者模型
- VoxCPM2 是選用下載，按 TTS 分頁的按鈕後才下載約 3.55 GB
- 模型備妥後，ASR、Voice Command、TTS 與說話者分析都能完全離線

發佈版是 self-contained，不必另外安裝 .NET 或 Visual C++ Runtime。

## 建置

在 repository 根目錄用 PowerShell 執行：

```powershell
.\apps\tippi-windows\scripts\build.ps1
```

完成後開啟：

```text
artifacts\Tippi-win-x64\Tippi.exe
```

從原始碼建置需要 .NET 8 SDK；專案已附上經真實模型測試的 x64 CPU / DirectML native
runtime。runtime 的來源 commit、設定與 SHA-256 記錄在
[`Runtime/README.md`](Runtime/README.md)。
VoxCPM2 的 CPU/Vulkan runtime 約 85 MB，來源、建置設定及 SHA-256 記錄在
[`Runtime/tts/README.md`](Runtime/tts/README.md)。

## 功能

- 麥克風即時辨識與 WAV/MP3/M4A/WMA 音訊檔轉錄
- 自動、DirectML（失敗回退）與 CPU 三種運算設定；結果依驅動程式與 runtime 簽章快取
- 自動說話者分離，輸出時間戳與「說話者 1／2…」標籤
- 自動語言或中文、英、日、韓、德、法手動選擇
- OpenCC 台灣繁體轉換
- 複製、UTF-8 文字檔儲存
- 以 Windows Unicode `SendInput` 將語音輸入到目前程式
- 獨立 3M KWS 模型偵測「幫我送出」；ASR 文字本身不會冒充 Voice Command event
- VoxCPM2 本機 TTS 分頁，提供 4／6／10 inference steps 與播放／卸載控制
- TTS 自動後端依序嘗試 Vulkan GPU、CPU；也可手動固定其中一種
- TTS 在獨立 process 執行，合成時 ASR、Voice Command 與說話者分析仍可繼續工作
- 可續傳模型下載，並固定 revision、檔案大小及 SHA-256 驗證

說話者模型會在啟動時載入並常駐 CPU 記憶體，但推論是停止錄音或完成檔案辨識後的
第二階段 CPU 分析。這樣待機與即時 ASR 階段不會持續搶 CPU，較適合沒有獨立顯示卡的
電腦；分析時間會隨錄音長度與處理器速度增加。若第二階段失敗，程式仍會保留已完成的
完整逐字稿。

目前固定的 INT4 模型 revision 已在 Intel Iris Xe 與 NVIDIA RTX 4060 Laptop 上實測：
DirectML 可以載入，但在實際解碼時不相容，因此自動模式會選 CPU。GPU runtime 與安全回退
已整合，待上游提供可通過相同測試的 ONNX 匯出即可啟用，不會讓無獨顯電腦失去支援。

跨程式輸入受 Windows UIPI 安全規則限制：一般權限的 Tippi 無法把按鍵送到
「以系統管理員身分執行」的程式。請讓兩個程式使用相同權限層級。

## 模型與資料位置

模型安裝在：

```text
%LOCALAPPDATA%\Tippi\Models\nemotron-3.5-asr-onnx-int4-8364d9e2
%LOCALAPPDATA%\Tippi\Models\speaker-diarization-onnx-v1
%LOCALAPPDATA%\Tippi\Models\sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
%LOCALAPPDATA%\Tippi\Models\voxcpm2-gguf-q8-169f64d8
```

音訊與辨識結果不會上傳；只有首次下載或按「下載／修復模型」時會連線至固定的
Hugging Face／GitHub 模型來源。模型授權與固定來源記錄在
`THIRD_PARTY_NOTICES.md`。

ASR 與 TTS 都是直接下載已轉換的模型，不會在使用者電腦做量化。ASR 的 INT4 ONNX、
KWS release archive／四個 ONNX 檔，以及 VoxCPM2 的 BaseLM Q8_0／Acoustic F16 GGUF
都固定 revision、大小與 SHA-256；下載支援續傳，完成後才取代正式檔案。GGUF 不會
提交進 Git repository。

## 量化選擇

- **INT4（目前採用）**：ASR 檔案約 757 MiB，CPU 實測可用，最適合沒有獨顯的預設版本。
- **INT8**：通常比 INT4 使用更多磁碟與 RAM，也可能保留較好準確度；更換前仍需用相同語料測 WER。
- **FP16**：較適合 GPU，但目前找到的 Nemotron FP16 ONNX 在 DirectML 實測會產生空白結果，因此未發行。
- **FP32**：體積與 RAM 需求最高，主要作為轉換／準確度基準；目前匯出也未通過 DirectML 解碼測試。
- **1-bit / BitNet**：不是這個 FastConformer-RNNT 模型可直接套用的通用後量化格式；目前沒有已驗證、可由本程式載入的 1-bit Nemotron artifact。

## 測試

一般測試不會下載模型：

```powershell
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```

完整 ASR CPU 模型測試（需要模型，首次約 757 MiB）：

```powershell
$env:TIPPI_RUN_MODEL_TEST = '1'
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```

DirectML 真實音檔探測與 CPU 回退測試：

```powershell
$env:TIPPI_RUN_DML_MODEL_TEST = '1'
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```

說話者 CPU 模型測試：

```powershell
$env:TIPPI_RUN_DIAR_TEST = '1'
$env:TIPPI_DIAR_MODEL_DIR = "$env:LOCALAPPDATA\Tippi\Models\speaker-diarization-onnx-v1"
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```

獨立 Voice Command 真實模型矩陣（兩個正例、四個負例）：

```powershell
$env:TIPPI_RUN_KWS_TEST = '1'
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```

四模型共存測試會讓 ASR、Voice Command、Diarization 常駐 CPU，同時執行說話者分析
與 loopback VoxCPM2 TTS，並驗證產生的 WAV；所有模型需已下載：

```powershell
$env:TIPPI_RUN_TTS_TEST = '1'
$env:TIPPI_TTS_BACKEND = 'auto' # 也可用 vulkan 或 cpu
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```

只校驗 VoxCPM2 下載檔，不載入模型：

```powershell
$env:TIPPI_RUN_TTS_DOWNLOAD_TEST = '1'
dotnet test .\apps\tippi-windows.tests\Tippi.Windows.Tests.csproj -c Release
```
