<#
.SYNOPSIS
    下載並驗證 Nemotron ASR ONNX INT4 模型檔案(Windows 端)。

.DESCRIPTION
    檔案清單(name/size/sha256)靜態展開自
    apps/tippi-windows/Services/ModelManifest.cs;此腳本不解析該 .cs 檔,
    改成手動同步的靜態表——若上游 manifest 變動,需同步更新這裡。

    冪等:目的地已存在且 SHA-256 相符的檔案會直接跳過;雜湊不符則刪除重下。
    每個檔案下載完成後都會重新算雜湊比對,不符即刪檔並拋錯(fail fast,
    不留下半殘缺/錯誤內容的檔案)。

    DEVIATION(見 task-4-brief.md 原始建議 URL 樣式 .../resolve/main/<name>):
    這裡改用 .../resolve/<Revision>/<name>,Revision 即 ModelManifest.cs 的
    Revision 常數。manifest 內的 sha256 是針對該固定 revision 的內容算出的;
    若改用 resolve/main,一旦上游 main 分支之後更新,腳本會抓到雜湊不再相符
    的新內容,只會報「下載失敗」而看不出是版本飄移。resolve/<revision> 保證
    抓到的位元組與 manifest 雜湊表永遠一致。

.NOTES
    執行(遠端 Windows,cmd.exe):
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch-nemotron-onnx-model.ps1
#>

$ErrorActionPreference = "Stop"
# Invoke-WebRequest 預設進度條會逐位元組刷新終端機,對 690MB 的
# encoder.onnx.data 這類大檔案會明顯拖慢下載速度。
$ProgressPreference = "SilentlyContinue"

$Repository = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
$Revision = "8364d9e2dd9da23789b480bdbba9e423717e42ee"
$TargetDir = "C:\Users\i5491\catcher-tippi-models\nemotron-onnx-int4"

# name / size(bytes,僅供人工核對,未強制驗證——SHA-256 已是足夠嚴格的完整性檢查)/ sha256
$Files = @(
    @{ Name = "audio_processor_config.json"; Size = 413;       Sha256 = "ab28d41eb87ce3922006edeb9c3fad4d5ce451f9a56a12d84f470f02a5ec157b" }
    @{ Name = "decoder.onnx";                Size = 4696;      Sha256 = "6a9f608dcbab71ebd81ffa4c198e82a5b6bb10f1c1830a94c752c5f543454df3" }
    @{ Name = "decoder.onnx.data";           Size = 59785216;  Sha256 = "e5fd55cbeeb268f9d383e2ee72735b9fbbb13aea4bc7cd38cb73b8e16f1366c7" }
    @{ Name = "encoder.onnx";                Size = 2677548;   Sha256 = "0b05217594ec0bda442e43a90a298ac2471a3bdcea9b169de34214e61a730e17" }
    @{ Name = "encoder.onnx.data";           Size = 690089984; Sha256 = "2f27295855aeb99ab1f8cd2254418d9ad7a087ea8dbe85f5596b4d887ea7d630" }
    @{ Name = "genai_config.json";           Size = 1892;      Sha256 = "39568fbeebbe848696a1e2a01c7f33df000f72c29f2285509fd12442bda9571e" }
    @{ Name = "joint.onnx";                  Size = 2136;      Sha256 = "e2c7d2fa40a243bf82eaca36c15698c52129de9361d2875d7f223f67fcd9482d" }
    @{ Name = "joint.onnx.data";             Size = 37830656;  Sha256 = "2e0fb1c060f3777a1a76e78d5589dd54f01505a06dffbd2588e315508b402c12" }
    @{ Name = "model_config.json";           Size = 365;       Sha256 = "f41f943eeb1310a89dd58cf3e11e654a8ae1a788fceeb6cd1eacce3a6d081965" }
    @{ Name = "silero_vad.onnx";             Size = 2243022;   Sha256 = "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808" }
    @{ Name = "tokenizer.json";              Size = 642525;    Sha256 = "24e1e8335c8396884a86f06880271376ae46a29381cfc35c82c6295d407acec7" }
    @{ Name = "tokenizer_config.json";       Size = 183;       Sha256 = "ea4b35353f468fea11f436f837d9621a29b4ba9d1c73c1ed0aa5743f5a53919e" }
    @{ Name = "vocab.txt";                   Size = 64024;     Sha256 = "ca88922ac5a92c911b79985b69634d7a4c2ef604d61b71bbe2982210dd77cd43" }
)

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$downloaded = 0
$skipped = 0

foreach ($file in $Files) {
    $dest = Join-Path $TargetDir $file.Name

    if (Test-Path $dest) {
        $existingHash = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash.ToLowerInvariant()
        if ($existingHash -eq $file.Sha256) {
            Write-Host "SKIP  $($file.Name)(已存在且雜湊相符)"
            $skipped++
            continue
        }
        Write-Host "STALE $($file.Name)(雜湊不符,刪除重下:預期 $($file.Sha256),實際 $existingHash)"
        Remove-Item -Force $dest
    }

    $url = "https://huggingface.co/$Repository/resolve/$Revision/$($file.Name)"
    Write-Host "GET   $($file.Name) <- $url"
    Invoke-WebRequest -Uri $url -OutFile $dest

    $hash = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash.ToLowerInvariant()
    if ($hash -ne $file.Sha256) {
        Remove-Item -Force $dest
        throw "SHA-256 mismatch for $($file.Name): expected $($file.Sha256), got $hash"
    }
    Write-Host "OK    $($file.Name)(雜湊相符)"
    $downloaded++
}

Write-Host "全部 $($Files.Count) 個檔案就緒:$TargetDir(下載 $downloaded、跳過 $skipped)"
