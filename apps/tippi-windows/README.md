# Tippi for Windows（CPU / GPU 自動選擇）

這是原本 macOS SwiftUI/MLX 應用程式的 Windows 原生版本。介面使用 WPF，
辨識模型改為同一個 NVIDIA Nemotron 3.5 ASR Streaming 0.6B 的 INT4 ONNX
轉換版，透過 ONNX Runtime GenAI 執行。程式會用真實音檔測試 DirectML，只有在 GPU
能正確產生文字且比 CPU 明顯更快時才選用 GPU，否則自動使用 CPU。說話者分離使用
Pyannote segmentation INT8 與 NVIDIA TitaNet-S 的 ONNX 版本，透過
sherpa-onnx CPU provider 執行；整套程式不需要 NVIDIA、AMD 獨立顯示卡，
也不需要 CUDA。

## 使用需求

- Windows 10 或 Windows 11，x64 處理器
- 建議 8 GB RAM，載入模型時至少保留約 4 GB 可用記憶體
- 首次啟動需要網路下載約 797 MiB 模型；之後辨識與說話者分析完全離線

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

## 功能

- 麥克風即時辨識與 WAV/MP3/M4A/WMA 音訊檔轉錄
- 自動、DirectML（失敗回退）與 CPU 三種運算設定；結果依驅動程式與 runtime 簽章快取
- 自動說話者分離，輸出時間戳與「說話者 1／2…」標籤
- 自動語言或中文、英、日、韓、德、法手動選擇
- OpenCC 台灣繁體轉換
- 複製、UTF-8 文字檔儲存
- 以 Windows Unicode `SendInput` 將語音輸入到目前程式
- 說「幫我送出」時移除指令文字並按一次 Enter
- 可續傳模型下載，並固定 revision、檔案大小及 SHA-256 驗證

說話者分離是停止錄音或完成檔案辨識後的第二階段 CPU 分析。這樣即時 ASR
不會與另一個模型搶 CPU，較適合沒有獨立顯示卡的電腦；分析時間會隨錄音長度
與處理器速度增加。若第二階段失敗，程式仍會保留已完成的完整逐字稿。

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
```

音訊與辨識結果不會上傳；只有首次下載或按「下載／修復模型」時會連線
Hugging Face。模型授權與固定來源記錄在
`THIRD_PARTY_NOTICES.md`。

ASR 是直接下載 Hugging Face 上已轉換好的 INT4 ONNX 檔案，不是在使用者電腦本地量化。
revision、每個檔案大小與 SHA-256 都固定在 `Services/ModelManifest.cs`，下載後會逐檔驗證。

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
