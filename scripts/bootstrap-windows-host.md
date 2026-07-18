# Windows 機器 bootstrap 紀錄(nemotron-asr-host)

本文件記錄如何從零把 Windows 遠端機器準備到能 build `apps/tippi-windows`,供後續 task 的 `dotnet build`/`dotnet test` 迴圈使用。所有指令均為**實際執行且成功**的版本(非最初嘗試)。

## 機器規格快照

- Hostname: `orca`
- ssh: `ssh i5491@100.91.128.2`(Tailscale IP,BatchMode 免密碼)
- OS: Windows 11, build 10.0.26200(`ver` 顯示 `10.0.26200.8875`)
- CPU: 20 threads
- GPU: RTX 4060 Laptop GPU + Iris Xe(內顯)
- RAM: 16GB
- Arch: x64
- 預設 remote shell: **cmd.exe**(指令鏈用 `&`,不是 `;`)
- 既有工具: `git 2.47.0.windows.2`、`winget v1.29.280`、PowerShell(`powershell -NoProfile -Command "..."`)、`gh` CLI 存在但 **token 失效,勿使用**
- Repo `wcAmon/catcher-tippi` 是 public,直接 `git clone https://...` 即可,免認證

## Step 1: push 分支(本機執行,非 ssh)

```bash
cd /Users/wake/Desktop/catcher-tippi/.worktrees/nemotron-asr-host
git push -u origin feat/nemotron-asr-host
```

結果:`* [new branch] feat/nemotron-asr-host -> feat/nemotron-asr-host`,upstream 設定成功。

## Step 2: 安裝 .NET 8 SDK(user-local,免管理員)

### 已知會失敗的寫法(記錄以避免重踩)

以下 one-liner(brief 原始建議)在本機 zsh 執行時,`$env:TEMP`、`$env:USERPROFILE` 會被**本機 bash/zsh** 提前展開成空字串(因為它們長得像 shell 變數 `$env`),導致遠端收到殘缺指令 `:TEMP\di.ps1`:

```bash
ssh i5491@100.91.128.2 "powershell -NoProfile -ExecutionPolicy Bypass -Command \"iwr https://dot.net/v1/dotnet-install.ps1 -OutFile $env:TEMP\\di.ps1; & $env:TEMP\\di.ps1 -Channel 8.0 -InstallDir $env:USERPROFILE\\dotnet\""
```

實際錯誤(遠端 PowerShell 收到 `-OutFile :TEMP\di.ps1`,亂碼是 cmd/ssh 對中文語系 PowerShell 的編碼問題,不影響診斷):

```
iwr : 不支援這種指定的路徑格式。
+ iwr https://dot.net/v1/dotnet-install.ps1 -OutFile :TEMP\di.ps1; & :T ...
NotImplemented: (:) [Invoke-WebRequest], NotSupportedException
```

### 實際成功寫法:本機寫 .ps1,經 stdin pipe 給遠端 PowerShell

避免所有 ssh/cmd/PowerShell 多層轉義問題,做法是把安裝腳本寫成本機檔案,再用 `cat local.ps1 | ssh ... "powershell -NoProfile -ExecutionPolicy Bypass -Command -"` 把內容經 stdin 餵給遠端 PowerShell(`-Command -` 代表從 stdin 讀取腳本)。

本機腳本(`install-dotnet.ps1`,內容如下,`$env:TEMP`/`$env:USERPROFILE` 在檔案內是純文字,只會被遠端 PowerShell 展開,不會被本機 shell 動到):

```powershell
$ErrorActionPreference = "Stop"
$dest = Join-Path $env:TEMP "di.ps1"
Invoke-WebRequest -Uri https://dot.net/v1/dotnet-install.ps1 -OutFile $dest
& $dest -Channel 8.0 -InstallDir "$env:USERPROFILE\dotnet"
```

執行:

```bash
cat install-dotnet.ps1 | ssh -o BatchMode=yes i5491@100.91.128.2 "powershell -NoProfile -ExecutionPolicy Bypass -Command -"
```

實際輸出(節錄):

```
dotnet-install: Remote file https://builds.dotnet.microsoft.com/dotnet/Sdk/8.0.423/dotnet-sdk-8.0.423-win-x64.zip size is 285072593 bytes.
dotnet-install: Downloaded file ... size is 285072593 bytes.
dotnet-install: The remote and local file sizes are equal.
dotnet-install: Extracting the archive.
dotnet-install: Adding to current process PATH: "C:\Users\i5491\dotnet\". Note: This change will not be visible if PowerShell was run as a child process.
dotnet-install: Installed version is 8.0.423
dotnet-install: Installation finished
```

### 驗證(必須用完整路徑,PATH 上的 dotnet shim 不是真 SDK)

```bash
ssh i5491@100.91.128.2 "%USERPROFILE%\dotnet\dotnet.exe --version"
```

輸出:`8.0.423`

```bash
ssh i5491@100.91.128.2 "%USERPROFILE%\dotnet\dotnet.exe --list-sdks"
```

輸出:`8.0.423 [C:\Users\i5491\dotnet\sdk]`

## Step 3: clone repo(public repo,免認證)

先確認遠端目錄不存在(若之前有部分 clone 殘留,改用 `git -C ... fetch & checkout` 而非直接 clone 失敗):

```bash
ssh i5491@100.91.128.2 "if exist C:\Users\i5491\catcher-tippi (echo EXISTS) else (echo MISSING)"
```

本次結果:`MISSING`,直接 clone:

```bash
ssh i5491@100.91.128.2 "git clone --branch feat/nemotron-asr-host https://github.com/wcAmon/catcher-tippi.git C:\Users\i5491\catcher-tippi"
```

結果:`Cloning into 'C:\Users\i5491\catcher-tippi'... Updating files: 100% (253/253), done.`

驗證 HEAD:

```bash
ssh i5491@100.91.128.2 "cd /d C:\Users\i5491\catcher-tippi & git log --oneline -1"
```

輸出:`ad25ea6 test: normalize punctuation/whitespace before ASR distance`

本機分支尖端(同時比對):

```bash
git log --oneline -1
```

輸出:`ad25ea6 test: normalize punctuation/whitespace before ASR distance` —— 一致。

## Step 4: 建置冒煙

```bash
ssh i5491@100.91.128.2 "cd /d C:\Users\i5491\catcher-tippi & %USERPROFILE%\dotnet\dotnet.exe build apps\tippi-windows\Tippi.Windows.csproj -c Release"
```

輸出:

```
  正在判斷要還原的專案...
  已還原 C:\Users\i5491\catcher-tippi\apps\tippi-windows\Tippi.Windows.csproj (1.8 秒 內)。
  Tippi.Windows -> C:\Users\i5491\catcher-tippi\apps\tippi-windows\bin\Release\net8.0-windows\Tippi.dll

建置成功。
    0 個警告
    0 個錯誤

經過時間 00:00:13.33
```

Build succeeded,0 warnings,0 errors,約 13 秒(含 restore)。

## 後續 task 依賴的固定路徑

- SDK: `%USERPROFILE%\dotnet\dotnet.exe`(= `C:\Users\i5491\dotnet\dotnet.exe`,版本 8.0.423)—— **必須用完整路徑**,PATH 上另有一個 .NET runtime shim 會誤報 "No .NET SDKs were found"。
- Repo: `C:\Users\i5491\catcher-tippi`,checkout 於 `feat/nemotron-asr-host`。
- Remote shell 慣例: cmd.exe,`&` 鏈指令、`cd /d` 切磁碟機、`%VAR%` 才會在 cmd 展開。
- 若要送 PowerShell 腳本給遠端,優先用 `cat local.ps1 | ssh ... "powershell -NoProfile -ExecutionPolicy Bypass -Command -"`(stdin pipe),避免多層 shell 轉義炸掉 `$env:` 變數。

## 附註:mac 端 Deno 安裝紀錄(tomato-ears 配方,Plan 3 Task 1)

本機(執行本文件其餘章節、負責推 branch 的 mac)安裝 Deno,供
`recipes/env-base/`、`recipes/tomato-ears/` 兩個配方使用。

安裝指令(user-local,免管理員):

```bash
curl -fsSL https://deno.land/install.sh | sh
```

實際輸出(節錄):

```
Deno was installed successfully to /Users/wake/.deno/bin/deno
Run '/Users/wake/.deno/bin/deno --help' to get started
```

驗證版本:

```bash
$ ~/.deno/bin/deno --version
deno 2.9.3 (long term support, release, aarch64-apple-darwin)
v8 14.9.207.2-rusty
typescript 6.0.3
```

**已知結果(供後續 task 核對):**

- 版本:Deno `2.9.3`(long term support)
- 路徑:`~/.deno/bin/deno`
- 安裝腳本**未自動**把 `~/.deno/bin` 寫進 shell profile(`~/.zshrc` 未被
  改動);後續指令一律用完整路徑 `~/.deno/bin/deno`,或先手動
  `export PATH="$HOME/.deno/bin:$PATH"`,不假設 `deno` 已在 PATH 上
  ——與本文件 Windows 章節「必須用完整路徑,PATH 上另有 shim 會誤報」
  的教訓一致。

Windows 端對照(見 Global Constraints):`winget install DenoLand.Deno`
(2.9.x),由後續 task 在遠端機器上實際安裝時記錄版本。
