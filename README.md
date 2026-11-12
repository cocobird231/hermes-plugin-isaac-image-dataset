# Isaac Image Dataset

獨立的 Hermes Agent plugin：在具備道路語意的 Isaac Sim 場景產生相機採樣姿態，使用可重用相機池拍攝 RGB，保存同幀姿態與內參，並依位置群組切分訓練、驗證及測試資料。提供逐張 checkpoint、分批執行、取消及故障恢復。

此目錄就是可發布的 repository 根目錄。所有 plugin 程式、安裝器、工具、skill 與測試均包含於此，不依賴其他 Isaac Sim／RViz2 plugin。場景與執行設定由使用者提供，不包含於程式包。

## 執行需求

- 已安裝並設定模型的 Hermes Agent，支援 `plugin.yaml`、Python 工具及 `hermes plugins install`。
- Linux、systemd user services、cgroup v2、NVIDIA GPU 與 `nvidia-smi`。已驗證 Ubuntu 24.04／Isaac Sim 6.0.1；不宣稱 Windows、macOS 或其他 Isaac 版本已有驗收。
- 可啟動 headless NuRec 的 Isaac Sim 6 Python 執行入口，可為虛擬環境 Python 或 Isaac 的 `python.sh`；該環境需具備 NumPy、Pillow、OpenUSD及Isaac所需套件。
- RAM 至少16 GiB，採集前至少12 GiB可用；整個採集job採10 GiB記憶體上限、swap 0。GPU忙碌時不會停止其他程式。
- 語意場景須符合 schema 2 `/DatasetSemantics` 原 mesh face 標籤，公尺、Z-up、靜態三角網格及可於本機解析的依賴。目前渲染支援單一普通、非SPG的NuRec Volume與對應碰撞Mesh。普通未標註USDZ不能直接作為道路採樣來源。

工具層只需 Python 標準函式庫；Isaac相關模組只在指定的子程序內載入。安裝器不安裝Isaac、GPU驅動、模型或場景。跨機器支援透過外部設定實現；執行後端仍有上述Linux環境需求。

## 安裝

可由Hermes直接安裝公開repo：

```bash
hermes plugins install https://github.com/cocobird231/hermes-plugin-isaac-image-dataset.git
```

或 clone 後執行一行安裝指令：

```bash
git clone https://github.com/cocobird231/hermes-plugin-isaac-image-dataset.git hermes-isaac-image-dataset
cd hermes-isaac-image-dataset
bash install.sh
```

Hermes原生安裝會掃描外掛程式；因插件需要啟動Isaac子程序及systemd user job，可能顯示CAUTION並要求確認。請檢視列出的用途後完成互動確認，不需要關閉掃描器。

選擇其中一種安裝方式即可。完成後執行 `hermes plugins`，確認 `isaac-image-dataset` 並啟用；也可使用 `hermes plugins enable isaac-image-dataset`。clone方式建立來源目錄的連結，因此須保留該目錄。

兩種方式均可先安裝、稍後由agent補上設定。原生Hermes安裝不會執行 `install.sh`；首次使用時agent透過下列設定工具完成配置。安裝器也接受明確參數：

```bash
bash install.sh \
  --isaac-python /path/to/isaac-environment/bin/python \
  --scene /path/to/semantic-scene.usda \
  --run-root /path/to/dataset-runs
```

完整旗標與自訂Hermes目錄見 [安裝說明](docs/installation.md)。安裝僅針對此plugin，不修改其他plugin、模型認證或GPU驅動。

## 一次設定，後續重用

Agent先呼叫 `isaac_dataset_get_config`。未配置的Isaac入口為 `null`，不會誤用Hermes自己的Python啟動模擬。使用者可提供prompt：

> 我的Isaac Python入口是 `/path/to/isaac-environment/bin/python`，語意場景是 `/path/to/semantic-scene.usda`，輸出目錄是 `/path/to/dataset-runs`。請使用dataset plugin保存這些設定，之後沿用。

Agent呼叫：

```json
{
  "settings": {
    "isaac_python": "/path/to/isaac-environment/bin/python",
    "default_scene": "/path/to/semantic-scene.usda",
    "run_root": "/path/to/dataset-runs"
  }
}
```

以上為 `isaac_dataset_configure` 的參數，不是採樣參數。工具檢查路徑與數值後，原子合併寫入設定檔，權限為0600。只更新提供的欄位；其他設定保留，既有job快照與plan不變。後續prompt未指定路徑時直接讀取已存值，不需每次詢問。

設定位置：`$XDG_CONFIG_HOME/isaac-image-dataset/config.json`；未指定XDG時使用 `~/.config/isaac-image-dataset/config.json`。`ISAAC_DATASET_CONFIG` 可改用其他位置。資料與GPU lock分別預設在XDG data／cache目錄下；`ISAAC_DATASET_RUN_ROOT`只覆蓋本次環境中的輸出位置，不會被設定工具寫回磁碟。

| 設定欄位 | 用途／預設 |
| --- | --- |
| `isaac_python` | 必須由使用者指定；保存虛擬環境入口，不解析成其他Python。 |
| `default_scene` | 預設語意場景；可不設定，改於每個job提供 `config.scene_path`。 |
| `run_root` | Job輸出根目錄，預設XDG data下的 `isaac-image-dataset/runs`。 |
| `gpu_lock_path` | 互斥檔，預設XDG cache下的 `isaac-image-dataset/gpu.lock`；多個採集器應使用同一位置。 |
| `batch_size` | 每個Isaac程序最多新增影像數，預設20，範圍1～1000。 |
| `sampling_timeout_seconds` | CPU採樣上限，預設600。 |
| `startup_timeout_seconds` | 每批固定時間預算，預設600。 |
| `per_sample_timeout_seconds` | 每張加入批次預算的係數，預設180；不是逐張timeout。 |

`default_scene: null`可清除預設場景，改由每個job傳入。`run_root`也是查詢job的根目錄；變更不會搬移或停止舊job，設定回執會提供原根目錄。已有執行中或待恢復job時請先沿用原值；若已更換，需恢復原值才能查詢、取消或接續舊job。

路徑可使用絕對路徑或 `~`。設定可隨機器重新提供；不要把私有設定檔加入Git。`configuration_complete=true`只表示必要入口已配置，不代表已完成GPU／場景相容性測試。

## 三種渲染品質

| 使用者名稱 | `config.render_quality` | 每組step後額外update |
| --- | --- | ---: |
| 效率 | `efficiency` | 0 |
| 一般（預設） | `normal` | 1 |
| 品質 | `quality` | 10 |

三種策略使用相同場景、採樣、解析度及相機內參。相機池先一次設定所有姿態，再呼叫共同的 `rep.orchestrator.step(wait_for_render=True)`；依選項增加0／1／10次Kit更新，隨後讀回與保存各相機資料。必要初始化流程保留，`step`內部也可能執行多個更新／subframes。

選項控制額外更新次數，不是渲染器解析度、曝光或Path Tracing取樣數，也不保證特定影像品質。更多update可能改變細節呈現並增加耗時，不能據此保證定位網路精度。metadata、相機證據與回執保存實際策略及執行次數。

新plan省略選項時保存 `normal`。品質與相機數綁定不可變plan，恢復時不能更換；需要其他品質請建立新job。舊多相機plan若缺少品質欄位，保留原10次策略；更舊的單相機plan保留原渲染流程，避免恢復時混用預設值。

## 開始採集

例如：「使用已存場景，採集60張廣角影像，4台相機，效率模式，依4:2:2切分。」Agent先讀取設定，再將下列參數交給 `isaac_dataset_pipeline`：

```json
{
  "config": {
    "n": 60,
    "camera_count": 4,
    "render_quality": "efficiency",
    "seed": 42,
    "sensor_height_m": 1.5,
    "pitch_deg": [-5, 10],
    "roll_deg": [-5, 5],
    "yaw_deg": [0, 360],
    "camera": {"resolution": [1280, 720], "horizontal_fov_deg": 120},
    "split_ratios": [4, 2, 2]
  }
}
```

`n`是全部影像數；`camera_count`是同時重用的相機數，整數1～8、預設1，不會將N乘上相機數。相同seed與幾何設定在品質切換下保持同一採樣結果。廣角120°為無畸變pinhole，不是魚眼或ZED；`camera.template_prim`可採用場景中受支援的置中perspective Camera內參。

預設採樣32筆、seed42、1280×720、水平FOV90°、離地1.5m、pitch−5°～+10°、roll±5°、yaw360°。採樣使用合格語意原面、坡度／間距／障礙檢查，不能將 `road_surface` 等同車道或保證車輛可通行。N請求上限10000不代表任意場景都有足夠合格空間。

## 九個工具與完成判定

| 工具 | 作用 |
| --- | --- |
| `isaac_dataset_get_config` | 讀取已存設定、缺少項目及相機／品質預設。 |
| `isaac_dataset_configure` | 保存使用者提供的路徑與runtime限制；不啟動模擬。 |
| `isaac_dataset_sample` | 新建job並產生不可變採樣plan，完成為 `sampled`。 |
| `isaac_dataset_collect` | 拍攝已有plan，完成為 `captured`。 |
| `isaac_dataset_split` | 驗證資料並按位置群組切分，完成為 `completed`。 |
| `isaac_dataset_pipeline` | 一次提交上述採樣、拍攝與切分流程。 |
| `isaac_dataset_status` | 查詢狀態、產物及實際退出證據，可等待0～30秒。 |
| `isaac_dataset_cancel` | 停止該job的專屬程序，保留已提交資料。 |
| `isaac_dataset_resume` | 接續已確認停止的失敗／取消job；不重拍已提交資料。 |

提交成功只表示受理。保存回傳的 `job_id`，持續查詢直到 `completed`，才能確認完整影像、姿態、分集與正常程序退出；`unknown`不是成功。不要因速度較慢就重複建立pipeline。

每張PNG對應XYZ＋Qx/Qy/Qz/Qw、實際相機K與來源語意面，另有影像／證據SHA和逐張checkpoint。預設4:2:2等於50%／25%／25%；同位置群組不跨集，實際數量可能因群組大小偏離比例。故障恢復會重驗來源、plan與已提交前綴，禁止改寫既有plan來更換品質。

目前每個程序完成 `batch_size` 後關閉再啟動下一批；這是插件資源策略，不是Isaac的必然限制。品質選項不會改變此策略。單次共同step亦不代表GPU一定平行完成所有視角。

## 文件、驗證與發布

- [Agent操作手冊](skills/isaac-image-dataset/SKILL.md)
- [安裝與配置](docs/installation.md)
- [內部實作、恢復與資料格式](docs/architecture.md)
- [版本紀錄](HISTORY.md)

驗證完成資料：

```bash
/path/to/isaac-environment/bin/python scripts/verify_dataset.py \
  /path/to/dataset-runs/JOB_ID --output /path/to/new-verification.json
```

需要透過已配置的Hermes模型執行文字請求時，可使用 `scripts/run_hermes_dataset.py`；請求及對話輸出留在repository以外。程式的CPU測試另需SciPy，可於具備OpenUSD的Isaac Python環境執行 `python -m unittest discover -s tests`。

本目錄為獨立repository，只納入plugin程式、通用文件與測試程式碼。模型、場景、資料集、測試結果、開發文件、日誌、私有設定與對話紀錄不納入Git。授權為MIT，見 [LICENSE](LICENSE)。
