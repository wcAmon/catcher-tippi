# tomato-ears SECURITY

給組裝 agent 的資安審查文件——對應店規(`docs/superpowers/specs/2026-07-18-mini-app-store-design.md`)
第 4 節第 5 條「資安強固化」:agent **在建構完成前必須執行本文件的審查步驟**,
不得只憑「`deno task verify:*` 全綠」就跳過。verify 的
`permissions_test.ts` 已經把第一條審查步驟機械化(見下方),但另外三條
(綁定範圍、外部 CDN、setup 的 net 全開 trade-off)需要人工或 agent 自行核對
一次,因為它們驗的是「這個配方的設計本身有沒有做對」,不是「這次安裝有沒有
做對」。

## 1. 兩階段權限模型

tomato-ears 的 Deno 執行權限拆成兩個完全獨立的 `deno task`,各自宣告獨立的
權限集合,`start` 執行期**刻意不含任何對外網路權限**:

```
deno task setup:mac / setup:win   ← 下載階段,需要對外網路
deno task start:mac / start:win   ← 運行階段,零對外網路
deno task verify:mac / verify:win ← 驗收階段(見第 4 節,權限與前兩者性質不同)
```

### 1.1 cwd 相對權限模型

`deno task` **定義上**以 `deno.json` 所在目錄為工作目錄(cwd)執行;配方
安裝時 `deno.json` 就放在 app 目錄(`~/tmuh-apps/tomato-ears/`)根部,所以
**cwd 就是 app 目錄**。這是所有權限旗標能用「跨機器不變的相對路徑」靜態
宣告在 `manifest.json`/`deno.json` 裡的關鍵前提,原因是兩個實測到的 Deno
行為事實(固化在 `reference/permissions_probe_test.ts`,正例+反例都有,
防未來 Deno 升版時回歸):

1. **Deno 的權限旗標不展開 `~`**——`--allow-read=~/tmuh-apps` 這種寫法對
   真實使用者的 `$HOME` 完全不生效(`~` 被當成字面路徑的一部分,而不是
   shell 展開後的家目錄)。
2. **`--allow-run=<路徑>` 只接受「能解析出可執行檔絕對路徑的字串」**
   ——相對路徑(對 cwd 解析)或絕對路徑都可以,但**不支援目錄前綴語意**
   (用一個目錄字串授權,執行期一律 `NotCapable`)。

因此權限旗標一律寫成 `.`(app 目錄本身)、`../_machine`(跨 app 共用目錄)、
`bin/engine-host[.exe]`(setup 解壓 + pin 出的穩定執行檔路徑)這種 cwd 相對
寫法——三者都是「這台機器上永遠位於同一個相對位置」的路徑,可以在
`manifest.json` 裡寫死,不需要每台機器動態產生旗標字串。

### 1.2 逐條權限對照表

| Task                    | 旗標                                                                        | 為什麼需要                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `setup:mac`/`setup:win` | `--allow-net`                                                               | 下載 engine host 與模型檔案,經 GitHub/Hugging Face 的 CDN,目標網域無法窮舉(見 1.3 的 trade-off 說明)                    |
|                         | `--allow-read=.,../_machine`                                                | 讀 `manifest.json`、檢查已存在的檔案是否可略過下載、讀 machine-profile 決定平台                                         |
|                         | `--allow-write=.,../_machine`                                               | 把下載內容原子落地到 `download/`/`bin/`/`model/`                                                                        |
|                         | `--allow-run=tar`                                                           | 解壓 engine host 壓縮包(mac `.tar.gz`、Windows `.zip`,系統內建 `tar`/bsdtar 兩種格式通吃)                               |
|                         | `--allow-env=TMUH_APPS_DIR`                                                 | 讀取安裝目錄覆寫環境變數(測試/演練用,一般使用者不需要設定)                                                              |
| `start:mac`/`start:win` | `--allow-net=127.0.0.1:43117`                                               | **只**綁定+監聽這一個 loopback port,不能連到任何其他位址                                                                |
|                         | `--allow-read=.,../_machine`                                                | 讀 app 目錄下的引擎/模型/UI 檔案、讀 machine-profile                                                                    |
|                         | `--allow-write=../_machine`                                                 | Windows 首跑把實測探測出的 backend 回填進 machine-profile(`writeBackfilledBackend`);mac 不回填,但旗標保持同形以簡化本表 |
|                         | `--allow-run=bin/engine-host,open`(mac)/`bin/engine-host.exe,explorer`(win) | spawn 引擎子行程 + 自動開瀏覽器(店規第 3 條),縮圈到剛好這兩個系統執行檔,不是整個 shell                                  |
|                         | `--allow-env=TMUH_APPS_DIR,TMUH_NO_BROWSER`                                 | 安裝目錄覆寫 + 測試/無頭環境跳過自動開瀏覽器                                                                            |

**`start` 執行期完全沒有 `--allow-net` 之外的網路能力**(唯一的網路權限
就是綁自己的 loopback port),這是最重要的一條防線:即使引擎子行程或
UI 程式碼被動了手腳,Deno 沙箱本身就不允許它連到任何外部網址。

### 1.3 setup 階段 `--allow-net` 全開的 trade-off(誠實說明)

`setup` 的 `--allow-net` 沒有縮圈到特定網域(不是
`--allow-net=github.com,huggingface.co`)。這是一個**刻意接受的權衡**,
不是疏漏:

- HF/GitHub 的下載會經過 CDN redirect(例如 HF 的 `resolve/` 端點會 302
  到 `cdn-lfs.huggingface.co` 之類的動態子網域),目標網域無法在
  `manifest.json` 裡窮舉列出;
- **完整性不靠網域白名單保證,靠逐檔 SHA-256 pin 保證**——`downloader.ts`
  的每一個下載動作,落地前都必須通過雜湊校驗,雜湊不符立刻刪除殘檔並
  `throw`(見 `downloader.ts` 檔頭「永不信任下載內容」)。就算 `setup`
  階段連到了非預期的網域,只要下載內容通不過 SHA-256 比對就不會被信任、
  不會被使用;
- 這個 trade-off 的範圍**只限於 `setup`**——一旦安裝完成,`start` 執行期
  完全沒有這個風險面(見 1.2 表格),使用者平常操作(開啟服務、錄音)
  的攻擊面遠比安裝那一刻小。

## 2. 審查步驟(agent 建構完成前必須執行)

### 2.1 執行旗標 == manifest 宣告(已機械化)

```bash
deno task verify:mac   # 或 verify:win
```

`verify/permissions_test.ts` 會解析 `deno.json` 的 `setup:mac`/`setup:win`/
`start:mac`/`start:win` 四個 task 指令,取出其中的 `--allow-*` 旗標,逐字
比對 `manifest.json.permissions` 對應鍵的陣列內容——**順序也要求相同**
（不是只驗證「有沒有出現」這種寬鬆檢查）。這個測試同時驗證
`manifest.json` 的 `ports.http`、`reference/main.ts` 的 `DEFAULT_PORT`
常數、`start:*` 宣告的 `--allow-net=127.0.0.1:<port>` 三處port數字互相
一致(防止任何一處單獨改動而漂移)。

如果只是想手動核對而不跑測試,打開 `manifest.json` 的 `permissions` 欄位
與 `deno.json` 的 `tasks` 欄位並排比對——四把 task 的 `--allow-*` 旗標
應該逐字相同,順序也一樣。

### 2.2 服務只綁 127.0.0.1

```bash
# 服務跑起來之後(deno task start:mac 或 verify 期間):
lsof -nP -iTCP:43117 -sTCP:LISTEN
# 應該只看到一行,ADDRESS 欄位是 127.0.0.1:43117，不是 *:43117 或 0.0.0.0:43117
```

`verify/binding_test.ts` 已經把這件事機械化(嘗試從本機 LAN 介面連線,
應該被作業系統拒絕,而非被 Deno 權限層擋下——見第 4 節的差異說明)。
`reference/server.ts` 的實作用 `Deno.serve({ hostname: "127.0.0.1", ... })`
——讓作業系統本身只在 loopback 介面監聽,不是應用層事後判斷來源 IP 再拒絕
(那種做法在有 bug 時可能被繞過,綁定範圍本身當防線才是可靠的)。

### 2.3 零外部 CDN

```bash
grep -n "http://\|https://" recipes/tomato-ears/reference/ui/*.html recipes/tomato-ears/reference/ui/*.js recipes/tomato-ears/reference/ui/*.css
# 預期:沒有任何符合結果(或只有出現在註解文字裡的說明性文字，不是可執行的
# <script src>/<link href>/import 語句)
```

`reference/ui/` 四個檔案(`index.html`/`app.js`/`downsampler-worklet.js`/
`downsampler-core.js`/`style.css`)全部是單檔實作,不 `import` 任何外部
URL、不載入任何外部字型/圖示/框架——瀏覽器打開 `http://127.0.0.1:43117/`
之後,不會有任何請求離開這台機器(録音/轉錄全程走本機 WebSocket 到本機
引擎子行程)。

## 3. 引擎 host 的角色與信任邊界

`bin/engine-host[.exe]` 是 wake 發布的 prebuilt binary(SHA-256 pin 在
`manifest.json`),**不是**這個配方的原始碼——它是一個獨立的信任邊界:
Deno 的 `--allow-run` 只保證「只能執行清單內這一個路徑的程式」,不保證
「這個程式本身做了什麼」。這個信任來自兩處:

1. `verify/integrity_test.ts` 驗證下載到的壓縮包(`download/` 目錄)雜湊
   與 `manifest.json` 宣告的完全相符——沒有中間人竄改過;
2. 引擎子行程本身**沒有任何網路權限**(它是 Deno `--allow-run` spawn 出來
   的獨立作業系統行程,不受 Deno 沙箱管轄,但引擎本身的設計是純粹的
   stdin/stdout 推論服務,不含任何網路程式碼——`docs/protocol/asr-host-v1.md`
   全程只描述 stdin/stdout 的 JSON-lines 交換,沒有任何網路相關的指令或
   事件)。

## 4. verify 與 start permission 差異(誠實揭露)

`deno task verify:mac`/`verify:win` 的權限旗標**比 `start:mac`/`start:win`
更寬**:`--allow-net` 不縮圈(而非 `127.0.0.1:43117`),並多了
`--allow-sys=networkInterfaces`。這是刻意的、僅限驗收工具鏈的放寬,原因是
`verify/binding_test.ts` 必須**證明一個否定**(「從 LAN 介面連不上」)——
如果驗證本身的 `--allow-net` 縮圈在 `127.0.0.1:43117`,那麼嘗試連線 LAN
位址會被 **Deno 權限層**擋下(`NotCapable`),而不是被**作業系統**擋下
(`ECONNREFUSED`)。這兩者是完全不同的失敗原因——前者只證明「我們自己的
測試行程被 Deno 沙箱限制住了」,無法證明「伺服器綁定範圍本身是安全的」;
後者才是真正驗證店規第 5 條「服務只綁 127.0.0.1」的方式。

這個放寬**只影響 `verify:*`(一次性驗收工具鏈,agent 建構完成後跑一次
確認,不是常駐執行的應用程式)**,`manifest.json.permissions` 完全沒有
宣告 `verify:*` 的鍵——它不在店規第 5 條「權限顯式宣告在 manifest」管的
範圍內(那條規則管的是「應用執行旗標」,即使用者平常會反覆執行的
`setup`/`start` 兩階段)。`start:*` 的 `--allow-net=127.0.0.1:43117`
縮圈維持不變,這才是使用者實際運行服務時的真實攻擊面。

`verify/permissions_test.ts` 的逐字比對範圍因此**只涵蓋四把 task**
(`setup:mac`/`setup:win`/`start:mac`/`start:win`),不包含 `verify:mac`/
`verify:win` 自己——見該測試檔頭的說明。

## 5. dev-time 測試 vs. verify 的權限差異(附帶說明,非審查步驟)

`reference/*_test.ts`(例如 `engine_test.ts` 用 `--allow-run` 不縮圈)使用
比 manifest 更寬的權限,因為它們需要模擬 host crash(`Deno.kill`)、跑本機
HTTP fixture server 等只有開發者在本 repo 才會做的事——這些測試檔**不會
被複製進最終安裝目錄**,只存在於配方原始碼樹(`reference/`)裡供未來維護
這個配方的人參考,不影響終端使用者實際執行時的權限面。PLAN.md 附錄有更
完整的 dev-test vs. verify 分工說明。
