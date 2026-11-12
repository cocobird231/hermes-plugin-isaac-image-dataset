# 安裝與跨機器設定

本套件為獨立 Hermes plugin；不依賴其他 Isaac Sim 或 RViz2 plugin。公開套件不包含場景、影像、帳號設定或主機執行設定。場景資產由使用者另行準備。

## 系統條件

- Linux、Python 3、Git，以及支援 Python plugin 的 Hermes Agent。
- NVIDIA GPU、相容驅動與 Isaac Sim 6 Python 環境。Hermes 使用的 Python 與 Isaac Sim 使用的 Python 可以不同。
- 現行採集程序使用 Linux `systemd --user`、cgroup 記憶體限制及 GPU 鎖；跨機器支援以相同系統條件為前提。
- 場景須符合本套件的採樣語意格式；現行資料流程採 schema 2 語意資料與 NuRec 場景。安裝 plugin 不會自動將任意 USD/USDZ 轉為可採樣場景。

## 方法一：clone 後安裝

將下列網址中的 `<owner>` 替換為實際 GitHub 帳號或組織：

```bash
git clone https://github.com/<owner>/hermes-isaac-image-dataset.git
cd hermes-isaac-image-dataset
bash install.sh
hermes plugins
```

在 Hermes 的 plugin 清單啟用 `isaac-image-dataset`，再開始新對話。安裝程式建立指向 clone 目錄的連結，因此安裝後應保留該目錄。預設不啟用 plugin，也不改寫其他 plugin 或 Hermes 的模型設定。

安裝時可直接提供本機所需路徑：

```bash
bash install.sh \
  --isaac-python /path/to/isaac-environment/bin/python \
  --scene /path/to/semantic-scene.usda \
  --run-root /path/to/datasets
```

`--isaac-python` 應為能載入 Isaac Sim 的 Python 入口，而非任意系統 Python。虛擬環境的 Python 符號連結會保留原入口，避免解析為系統 Python 後遺失環境。安裝時檢查檔案存在及可執行；實際 Isaac Sim 套件與 GPU 相容性在採樣、渲染時驗證。

## 方法二：由 Hermes 安裝 GitHub repo

支援 Git plugin 安裝的 Hermes 版本可直接使用：

```bash
hermes plugins install https://github.com/<owner>/hermes-isaac-image-dataset.git
hermes plugins
```

Hermes 依 repo 根目錄的 `plugin.yaml` 安裝本套件；此流程不會自動執行 `install.sh`。啟用後，使用下節的設定工具儲存必要路徑。

兩種方法擇一即可。已使用方法一建立連結時，Hermes 的原生安裝器可能拒絕在同名位置再次安裝；不應以覆寫方式混用兩種安裝方式。

## 透過 Agent 儲存路徑

未提供任何路徑仍可完成安裝；plugin 不會將 Hermes 自己的 Python 誤認為 Isaac Sim Python。首次採樣前，Agent 應依序執行：

1. 呼叫 `isaac_dataset_get_config`，檢查必要設定是否完整。
2. 僅對缺少的必要設定詢問使用者；例如 Isaac Sim Python 的位置。
3. 使用 `isaac_dataset_configure` 儲存使用者提供的路徑。
4. 再次讀取設定，確認後執行採樣與採集。

可向 Agent 提供以下指令，將範例路徑替換為實際路徑：

> 將資料集 plugin 的 Isaac Sim Python 設為 /path/to/isaac-environment/bin/python，預設場景設為 /path/to/semantic-scene.usda，資料集輸出設為 /path/to/datasets。請儲存設定，之後沿用。

預設場景可留空，改由每次採樣呼叫提供。已儲存且有效的設定會沿用；重跑安裝程式時，只更新有提供的參數。

## 設定保存位置

| 項目 | 預設位置 | 自訂方式 |
|---|---|---|
| plugin 安裝位置 | `~/.hermes/plugins/isaac-image-dataset` | `HERMES_HOME` 或 `--hermes-home` |
| 執行設定 | `~/.config/isaac-image-dataset/config.json` | `XDG_CONFIG_HOME`、`ISAAC_DATASET_CONFIG` 或 `--config` |
| 資料集工作目錄 | `~/.local/share/isaac-image-dataset/runs` | `XDG_DATA_HOME`、`--run-root` 或設定工具 |
| GPU 鎖 | `~/.cache/isaac-image-dataset/gpu.lock` | `XDG_CACHE_HOME`、`--gpu-lock-path` 或設定工具 |

以上 `~` 由執行使用者解析，沒有預設帳號、hostname 或 IP。設定檔與公開原始碼分離，採原子寫入，權限為 `0600`。若多個程式共用同一張 GPU，應使用相同的 GPU 鎖路徑。

使用 `--config` 指定非預設設定檔時，啟動 Hermes 的環境也須設定 `ISAAC_DATASET_CONFIG` 指向同一檔案。可在使用者自行管理的 shell 或服務環境保存該變數；安裝程式不會改寫 shell 設定。

## 其他安裝參數

| 參數 | 用途 |
|---|---|
| `--batch-size` | 每個渲染程序最多處理的姿態數；範圍 1–1000 |
| `--startup-timeout-seconds` | 渲染程序啟動的時間預算 |
| `--per-sample-timeout-seconds` | 每筆影像的時間預算 |
| `--sampling-timeout-seconds` | 採樣程序的時間預算 |
| `--hermes` | Hermes CLI 不在 PATH 時指定其可執行檔 |
| `--enable` | 安裝後透過 Hermes CLI 僅啟用本 plugin |

`batch_size` 是程序分批上限，與相機數量不同；相機數量及渲染品質由採樣／pipeline API 的資料集設定決定。預設渲染品質為一般。完整參數可由 `bash install.sh --help` 查閱。


## 原生安裝的掃描確認

Hermes原生Git安裝會掃描來源；啟動Isaac子程序、systemd user job與讀取資源狀態的程式可能被列為CAUTION。檢視掃描內容後，於終端完成互動確認即可繼續；不要停用掃描器。此提示與plugin啟用是不同步驟，安裝後仍需確認工具已啟用。
