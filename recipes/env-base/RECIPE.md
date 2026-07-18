# env-base（App 0：環境基座）

> 對應店規 `docs/superpowers/specs/2026-07-18-mini-app-store-design.md` 第 5
> 節。 這不是一個「應用」，是所有 mini-app 共用的一次性環境基座。
> **只需要跑過一次。** 跑完之後，把下方「寫入持久記憶」那一步做完， 之後安裝任何
> tmuh.ai mini-app 都不必重跑這份配方。

## 給 agent 的指示

你正在替使用者組裝 tmuh.ai mini-app store 的環境基座。全程只做四件事，
不寫任何應用邏輯：

1. **檢查是否已經跑過**——先讀 `~/tmuh-apps/_machine/machine-profile.json`
   是否存在。如果存在，且你自己的持久記憶（CLAUDE.md / memory）裡已經記錄 Deno
   版本與路徑，直接跳過本配方，回報「環境基座已就緒」即可，
   **不要重新安裝、不要重新探測**。
2. 安裝 Deno（一條指令，見下方 Step 1）。
3. 建立標準目錄 `~/tmuh-apps/` 與 `~/tmuh-apps/_machine/`。
4. 跑硬體探測腳本，產出 `~/tmuh-apps/_machine/machine-profile.json`。
5. 把探測到的環境事實寫進你自己的持久記憶（見 Step 4）。

## Step 1：安裝 Deno（user-local，免管理員）

macOS / Linux：

```bash
curl -fsSL https://deno.land/install.sh | sh
```

安裝到 `~/.deno`，把 `~/.deno/bin` 加進 `PATH`（安裝腳本結尾會印出對應的 shell
profile 指令，照做即可）。驗證：

```bash
deno --version
```

**第三種情境——`deno` 已經裝過，但 `PATH` 沒有它**（例如上一個 agent session
裝的、或使用者手動裝過、只是這次開的是新終端機）：先用

```bash
ls ~/.deno/bin/deno
```

確認可執行檔是否已存在。如果存在，**不要重跑 curl 安裝腳本**（重裝沒有
必要，還會浪費一次網路下載）——只要把 `~/.deno/bin` 加進當前 shell session 的
`PATH` 即可：

```bash
export PATH="$HOME/.deno/bin:$PATH"
```

再跑 `deno --version` 確認。**注意**：這個 `export` 只對「當前這個 shell
session」有效——如果你的執行環境每個指令都是獨立行程（shell state 不跨
指令持久化），之後每一條要用到 `deno` 的指令前都要重新 `export` 一次；
若要讓使用者「下次開新終端機也不必再處理」，把同一行 `export PATH=...`
追加進使用者的 shell profile（`~/.zshrc`/`~/.bashrc`/`~/.bash_profile`
等，依實際使用的 shell 而定）才會持久生效。

Windows：

```cmd
winget install DenoLand.Deno -e --accept-package-agreements --accept-source-agreements
```

（另見 `scripts/bootstrap-windows-host.md` 的「mac 端 deno 安裝紀錄」附註
章節，內有 Windows 對照版本號）。驗證：

```cmd
deno --version
```

**PATH 生效時機**：winget 安裝完會提示「重新啟動命令介面以使用新值」——
如果你的操作方式是每條指令都經由新的 shell/ssh session 送出（例如 ssh 遠端驅動
Windows 機），**不需要**額外處理：下一條指令天然是新 session，
會直接拿到更新後的 PATH。只有在同一個互動式 cmd/PowerShell 視窗裡連續
下指令才需要真的關掉重開視窗。

**已知結果（本次在 mac 上實際安裝，供核對用，不代表你的機器一定一樣）：** Deno
`2.9.3`（long term support），路徑 `~/.deno/bin/deno` （完整輸出見
`scripts/bootstrap-windows-host.md` 附註）。

**已知結果（Task 6 於 Windows 11 x64 實際安裝，供核對用）：** Deno
`2.9.3`（build 標示為 `stable, release`，不是 mac 的 `long term support`——winget
發行版的 build 字串不同，版本語意相同，不 構成問題），路徑
`C:\Users\<user>\AppData\Local\Microsoft\WinGet\Links\deno.exe`（shim）。
完整逐字稿見 `.superpowers/sdd/task-6-rehearsal-log.md`「階段一 Step 1」。

## Step 2：建立標準目錄

```bash
mkdir -p ~/tmuh-apps/_machine
```

Windows（cmd）對照：

```cmd
mkdir C:\Users\<user>\tmuh-apps\_machine
```

cmd 的 `mkdir` 天然遞迴建立中間目錄（等效 bash 的 `-p`），不需要額外
旗標。（也不必手動先建：下游探測腳本自己會
`Deno.mkdir(dir, {
recursive: true })`；這裡列出來是給想要提前確認目錄結構的使用者。）

目錄慣例（之後每個 mini-app 沿用）：

- `~/tmuh-apps/<app-name>/`：每個 mini-app 一個子目錄（`bin/`、`model/`、`ui/`
  等）。
- `~/tmuh-apps/_machine/`：跨 app 共用的環境資料，目前只有一個檔案
  `machine-profile.json`。

## Step 3：跑硬體探測

**在配方包目錄執行**（本文件把自己包裝成「像從 store 下載的獨立配方包」
——`recipes/env-base/` 本身就是這份配方包的根目錄，不假設你手上有整個
repo）。下面指令的 `probe/machine-profile.ts` 是**相對配方包根目錄** 的路徑，先
`cd` 到 `recipes/env-base/`（或你拿到的配方包解壓後的根目錄） 再執行：

```bash
cd recipes/env-base   # 或你的配方包解壓根目錄
deno run \
  --allow-sys=systemMemoryInfo \
  --allow-read=$HOME/tmuh-apps \
  --allow-write=$HOME/tmuh-apps \
  --allow-env=HOME,USERPROFILE,TMUH_APPS_DIR \
  probe/machine-profile.ts
```

`--allow-env` 多列的 `TMUH_APPS_DIR` 對應腳本支援的覆寫（見下方）：一般
使用者安裝不需要設定這個變數，探測腳本會照常用 `HOME`/`USERPROFILE` 算出
`~/tmuh-apps`；只有測試/演練情境（需要把整套安裝指到非真實 HOME 的位置）
才需要設定 `TMUH_APPS_DIR`，設定時探測腳本會直接把 profile 寫到
`$TMUH_APPS_DIR/_machine/machine-profile.json`，不再查 `HOME`/
`USERPROFILE`——此時 `--allow-read`/`--allow-write` 也要跟著指向
`$TMUH_APPS_DIR`（而不是 `$HOME/tmuh-apps`），例如：

```bash
TMUH_APPS_DIR=/path/to/替代根目錄 deno run \
  --allow-sys=systemMemoryInfo \
  --allow-read=$TMUH_APPS_DIR \
  --allow-write=$TMUH_APPS_DIR \
  --allow-env=HOME,USERPROFILE,TMUH_APPS_DIR \
  probe/machine-profile.ts
```

### Windows（cmd）對照 + 這裡的兩個 cmd 陷阱

```cmd
cd recipes\env-base
deno run --allow-sys=systemMemoryInfo --allow-read=%USERPROFILE%\tmuh-apps --allow-write=%USERPROFILE%\tmuh-apps --allow-env=HOME,USERPROFILE,TMUH_APPS_DIR probe/machine-profile.ts
```

若要用 `TMUH_APPS_DIR` 覆寫（測試/演練情境），**不要**照抄 bash 的 `VAR=x cmd`
單行語法直譯成 cmd 的 `set VAR=x & cmd`：

> **cmd 陷阱警告（Task 6 Windows 演練卡點 #1 實測發現）**
>
> 1. **`%VAR%` 在整行解析期展開，早於同一行的 `set` 執行**。單行
>    `set TMUH_APPS_DIR=C:\...\tmuh-apps & deno run --allow-read=%TMUH_APPS_DIR% ...`
>    裡的 `%TMUH_APPS_DIR%` 拿到的是**這行指令開始前**的舊值（未定義=
>    空字串），不是同一行 `set` 剛設的新值——結果旗標實際是
>    `--allow-read=`（空），Deno 報 `NotCapable`。**手動先建目錄也救
>    不了**：根因不是目錄不存在，是旗標本身拿到空字串。
>    **正確寫法：權限旗標直接寫字面路徑**，`set` 只留給 deno 腳本自己 在執行期用
>    `Deno.env.get("TMUH_APPS_DIR")` 讀（子行程執行期讀取， 不受 cmd
>    解析期展開影響）：
>    ```cmd
>    set TMUH_APPS_DIR=C:\Users\<user>\tmuh-apps & cd /d C:\...\recipes\env-base & deno run --allow-sys=systemMemoryInfo --allow-read=C:\Users\<user>\tmuh-apps --allow-write=C:\Users\<user>\tmuh-apps --allow-env=HOME,USERPROFILE,TMUH_APPS_DIR probe/machine-profile.ts
>    ```
> 2. **`set X=value & ...` 會把 `&` 前那個空格也算進值裡**（cmd 把 `value`
>    整段當成值，尾隨空格照留）。Windows 路徑解析對「目錄名尾
>    隨空格」通常寬容（Task 6 實測寫入路徑帶尾隨空格仍正確落地），但這
>    是僥倖不是保證——安全寫法是帶引號：`set "X=value" & ...`。

完整實測見 `.superpowers/sdd/task-6-rehearsal-log.md`「階段一 Step 2： 卡點
#1」。

這會產出（或**冪等更新**）`~/tmuh-apps/_machine/machine-profile.json` （或
`TMUH_APPS_DIR` 覆寫時的等效路徑），內容包括：

| 欄位          | 說明                                          |
| ------------- | --------------------------------------------- |
| `probedAt`    | 探測時間（ISO 8601）                          |
| `os`          | `Deno.build.os`（`darwin`/`windows`/`linux`） |
| `arch`        | `Deno.build.arch`（`aarch64`/`x86_64`）       |
| `ramBytes`    | 實體記憶體總量（bytes）                       |
| `cpuThreads`  | CPU 執行緒數                                  |
| `denoVersion` | 執行探測腳本當下的 Deno 版本                  |

**推論後端欄位（例如 `inference`）刻意留空**，由各 mini-app 首次啟動時
自己實測後回填（見店規第 6 節「硬體適配用實測探測」）——本探測腳本 是冪等
merge，重跑不會動到那些欄位，見 `recipes/env-base/probe/machine-profile.ts`
內的說明與 `machine-profile_test.ts` 的「冪等 merge」測試。

## Step 4：把環境事實寫進持久記憶

跑完 Step 3 之後，把以下事實寫進你自己的持久記憶檔（CLAUDE.md 或等效的 memory
機制），格式不拘，重點是「下次接手的 agent（或你自己下個 session）
不必重新探測」：

- Deno 版本與安裝路徑（例如「Deno 2.9.3，`~/.deno/bin/deno`，已在 PATH」）。
- `~/tmuh-apps/` 目錄已建立，`~/tmuh-apps/_machine/machine-profile.json`
  已存在。
- 明確寫一句：**「env-base 已完成，之後安裝 mini-app 不要重新安裝 Deno 或重新跑
  machine-profile 探測，除非使用者明確要求重探測（例如換機器/ 升級記憶體）。」**

## 驗收

- [ ] `deno --version` 有輸出，且版本 ≥ 2.x。
- [ ] `~/tmuh-apps/_machine/machine-profile.json` 存在，`os`/`arch`/
      `ramBytes`/`cpuThreads`/`denoVersion` 皆為非空值。
- [ ] 你的持久記憶檔已包含上述環境事實。

完成以上三項，App 0 即告完成，可以繼續安裝其他 mini-app（例如
`recipes/tomato-ears/`）。

**附註（人力步驟）**：env-base 本身（Deno 安裝 + 硬體探測）全程可以由 agent
獨力完成，不需要使用者動手。但下游某些 mini-app 安裝完成、啟動
服務後，仍會有必須由「人」完成的步驟——例如 tomato-ears 首次錄音前
瀏覽器會彈出麥克風授權提示，那個提示只能由使用者本人點擊，agent 無法
代為操作瀏覽器原生對話框。跑到那類 app 的配方文件時留意它是否有標註
這件事（tomato-ears 見 `recipes/tomato-ears/PLAN.md` Step 5 的附註）。
