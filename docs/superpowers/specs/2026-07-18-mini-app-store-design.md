# tmuh.ai Mini-App Store 設計規範

日期:2026-07-18
狀態:草案(待 wake 審核)
範圍:發布模式定義、配方格式、上架規則、環境基座、推論引擎政策、tmuh.ai 端功能綱要

## 1. 背景與目的

tmuh.ai 目前以 host HTML page 的方式發布內容。本設計把「應用程式發布」納入同一哲學:
**store 上放的不是打包好的執行檔,而是「製作配方(recipe)」**——prompts + 規格 +
帶註解的程式碼範例 + 驗收測試。使用者下載配方後,由自己電腦上的 agent(Claude Code 等)
照配方把應用程式組裝出來。

目標使用者幾乎沒有程式經驗,因此整個設計的最高優先指標是:
**agent 在建構過程中需要處理 bug 的次數趨近於零**。

## 2. 發布哲學:組裝,不是生成

配方裡 agent 的角色是**組裝工,不是工程師**:

- 參考程式碼是 95% 完成、帶詳細註解、已通過測試的成品;
- agent 只做四件事:**探測環境、下載並驗 hash、寫少量黏合、跑驗收測試**;
- agent 寫的程式碼越少,變異越少,bug 越少;
- 「確定性」來自驗收測試,不來自 prompt——測試過了才算完成。

## 3. 配方(Recipe)格式

每個 mini-app 在 store 上是一個固定結構的包:

```
recipe/
  manifest.json    # 名稱、版本、stack、宣告權限、外部相依(binary/模型)+ SHA-256
  SPEC.md          # 要做出什麼、驗收標準(人類可讀,也是 agent 的目標定義)
  PLAN.md          # 給 agent 的分階段建造指令(讀 machine-profile → 下載 → 黏合 → 測試)
  reference/       # 已驗證的關鍵模組(95% 完成、詳細註解)
  verify/          # 驗收測試腳本(agent 必須跑過才算完成)
  SECURITY.md      # agent 安全審查步驟 + 權限對照表
```

### manifest.json 必要欄位

| 欄位 | 說明 |
|---|---|
| `name` / `version` | 識別與版本 |
| `stack` | 只允許 `deno` / `python` / `go` |
| `permissions` | 顯式權限宣告(如 Deno 旗標 `--allow-net=127.0.0.1 --allow-read=...`) |
| `dependencies` | 每個外部下載(engine binary、模型)的 URL + 大小 + SHA-256 |
| `ports` | 佔用的 localhost port 與 API 合約版本(供 mini-app 之間堆疊組合) |
| `verify` | 驗收測試的進入指令 |

## 4. 上架規則(店規)

1. **夠小**:單一功能,一份 SPEC.md 講得完;參考程式碼約 1500 行以內。
2. **技術棧限縮**:Deno(JS/TS,首選)、Python、Go。Go 保留給需要二進位效能的場景
   (單檔編譯、零 runtime 相依)。禁止其他棧;禁止過新、agent 訓練資料不足的路線
   (例:Mojo + Vulkan——agent 無法穩定執行,已評估否決)。
3. **容易打開**:安裝最多一條指令、啟動一條指令、瀏覽器自動開 localhost 頁面。
4. **詳細註解**:reference 程式碼每個 public function 有目的說明、每段非顯然邏輯有
   why 註解。註解是給重建 agent 的語義錨點,直接降低組裝錯誤率。
5. **資安強固化**:
   - 服務只綁 `127.0.0.1`;
   - 所有外部下載一律 SHA-256 pin,下載後先驗再用;
   - 權限顯式宣告在 manifest,agent 審查=驗證「執行旗標 == 宣告」;
   - 配方附 SECURITY.md,agent 在建構完成前必須執行其中的審查步驟。
6. **驗收測試門檻**:`verify/` 測試全數通過才算建構完成,agent 不得自行宣告成功。
7. **硬體適配用實測探測**:凡涉及推論的 mini-app,後端選擇必須以「實際載入 + 基準測量」
   決定(見第 6 節),禁止只讀硬體規格猜測,禁止無條件依賴 CPU fallback。
8. **prebuilt + pin,agent 不編譯原生碼**:推論引擎與模型一律使用預先發布的
   binary/artifact,agent 只下載與驗 hash。
9. **大檔案發布通道**:模型與 engine binary 一律發布在 Hugging Face 或
   GitHub Releases,不直接放在 tmuh.ai 上。(本次僅定為條款;
   AI 模型類 mini-app 的上傳流程設計超出本次範圍。)

## 5. 環境基座(App 0)與固化記憶

Store 的第一個配方不是應用,是**環境基座**。使用者安裝第一個 mini-app 前必先跑過:

1. 安裝 Deno runtime(單一官方指令)——所有 mini-app 的統一基座;
2. 建立標準目錄 `~/tmuh-apps/`(每個 mini-app 一個子目錄;共用 `_machine/` 放環境資料);
3. 執行硬體探測,產出 `~/tmuh-apps/_machine/machine-profile.json`:
   OS、arch、GPU、RAM、可用推論後端與其實測基準結果;
4. **要求 agent 把環境事實寫進它的持久記憶**(CLAUDE.md / memory):
   Deno 版本與路徑、目錄慣例、machine-profile 位置、「已裝過的東西不要重裝」。

之後每個 mini-app 配方的 PLAN.md 第一步固定是:
「讀 machine-profile,不要重新探測、不要重新安裝基座」。

> 設計理由:環境變異是 agent 建構時最大的 bug 來源。基座把這個成本從
> 「每個 app 攤還一次」變成「全站只付一次」。

## 6. 推論引擎政策(決策表,非 agent 自由發揮)

| 平台 | 引擎 | 形式 |
|---|---|---|
| macOS arm64 | MLX(catcher runtime) | wake 發布的 prebuilt engine host,SHA-256 pin |
| Windows x64 | onnxruntime + DirectML;實測輸 CPU 則退 CPU | wake 發布的 prebuilt engine host,SHA-256 pin |
| 其他/載入失敗 | CPU execution provider | 最後保底,不是預設 |

**實測探測原則**(取自 catcher-tippi `InferenceBackendPolicy`,已在 Windows 分支驗證):
候選後端都實際載入並跑短基準;GPU 後端必須贏過 CPU 一個有意義的門檻
(參考值 0.85——內顯搬張量的成本會吃掉紙面優勢)才採用。
探測結果寫進 machine-profile,同一台機器不重複探測。

## 7. tmuh.ai 端功能綱要

(此節之後會展開成獨立實作文件,交由 tmuh.ai 伺服器上的 Claude Code 執行。)

- **上傳**:創作者上傳 recipe 包(非執行檔);伺服器做 manifest schema 驗證
  + 店規靜態檢查(stack 白名單、port 宣告、hash 欄位齊全)。
- **審查**:伺服器端 Claude Code 對每個上架配方做一次安全審查
  (prompt injection 面:配方是「會被別人的 agent 執行的指令」,
  store 端審查 + 使用者端 SECURITY.md 是兩道獨立防線)。
- **展示**:沿用現有 host HTML page 機制——每個配方一個展示頁
  + 可下載 bundle + 一鍵複製的啟動指令(使用者貼進 agent 即開始建構)。
- **版本**:配方更新 = 新版本 bundle;配方需設計成增量可重入
  (agent 可在既有安裝目錄上跑新版配方做 diff 式升級,不重建全部)。

## 8. 非目標(本次不做)

- AI 模型類 mini-app 的上傳/託管流程(僅定第 4 節第 9 條通道條款);
- Linux 引擎政策(表留空位,之後補);
- mini-app 之間的自動組合編排(v1 只定 port/API 宣告慣例);
- 跨 app 文字注入(voice input)類需要 OS 特權的功能——留在原生 Tippi app,
  不進 mini-app store。

## 9. 第一個應用

第一個上架配方為 `tomato-ears`(Nemotron ASR 即時轉錄),
設計見 `2026-07-18-tomato-ears-design.md`。
