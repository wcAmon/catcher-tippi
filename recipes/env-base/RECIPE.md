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

Windows：`winget install DenoLand.Deno`（另見
`scripts/bootstrap-windows-host.md` 的「mac 端 deno 安裝紀錄」附註章節， 內有
Windows 對照版本號）。

**已知結果（本次在 mac 上實際安裝，供核對用，不代表你的機器一定一樣）：** Deno
`2.9.3`（long term support），路徑 `~/.deno/bin/deno` （完整輸出見
`scripts/bootstrap-windows-host.md` 附註）。

## Step 2：建立標準目錄

```bash
mkdir -p ~/tmuh-apps/_machine
```

目錄慣例（之後每個 mini-app 沿用）：

- `~/tmuh-apps/<app-name>/`：每個 mini-app 一個子目錄（`bin/`、`model/`、`ui/`
  等）。
- `~/tmuh-apps/_machine/`：跨 app 共用的環境資料，目前只有一個檔案
  `machine-profile.json`。

## Step 3：跑硬體探測

```bash
deno run \
  --allow-sys=systemMemoryInfo \
  --allow-read=$HOME/tmuh-apps \
  --allow-write=$HOME/tmuh-apps \
  --allow-env=HOME,USERPROFILE \
  recipes/env-base/probe/machine-profile.ts
```

這會產出（或**冪等更新**）`~/tmuh-apps/_machine/machine-profile.json`，內容包括：

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
