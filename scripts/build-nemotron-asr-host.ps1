# 打包 nemotron-asr-host 為可發布的 zip 並產生 SHA-256(bare filename)。
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$dotnet = Join-Path $env:USERPROFILE "dotnet\dotnet.exe"
if (Test-Path publish\nemotron-asr-host) { Remove-Item -Recurse -Force publish\nemotron-asr-host }
& $dotnet publish apps\nemotron-asr-host\NemotronAsrHost.csproj -c Release -r win-x64 `
    --self-contained -o publish\nemotron-asr-host
if ($LASTEXITCODE -ne 0) { throw "publish failed" }
$version = "0.1.0"
$name = "nemotron-asr-host-v$version-windows-x64"
New-Item -ItemType Directory -Force -Path dist | Out-Null
Copy-Item docs\protocol\asr-host-v1.md publish\nemotron-asr-host\PROTOCOL.md
Compress-Archive -Path publish\nemotron-asr-host\* -DestinationPath "dist\$name.zip" -Force
$hash = (Get-FileHash "dist\$name.zip" -Algorithm SHA256).Hash.ToLower()
# Out-File -Encoding ascii 在 Windows PowerShell 上會把行尾寫成 CRLF(0D 0A);
# mac/Linux 端 `shasum -c` 只認純 LF(0A),CRLF 會讓最後一行多一個 0x0D,校驗失敗。
# 改用 File.WriteAllText 明確只塞一個 `n`(LF),不經過 PowerShell 的行尾轉換。
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "dist\$name.zip.sha256"), "$hash  $name.zip`n")
Write-Host "done: dist\$name.zip"
Write-Host "sha256: $hash"
