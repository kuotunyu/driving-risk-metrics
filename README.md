# driving-risk-metrics

P1：以 BDD100K 建立可重現的自駕感知風險評估專案。

## 目前狀態

目前已完成新版套件、驗證與 CI 基礎、資料 manifest 與前處理、標準／風險加權
指標、instance-balanced semantic coverage、normalized image-band、溫度校準與
配對 bootstrap 分析、三個核准模型的 adapter、確定性訓練與評估引擎、五個 CLI
指令與 claim-safe 報告產生器，以及 locked evaluation 的專案 skill 與其
fail-closed validator。

**尚未存在任何正式訓練結果**：正式 manifest 尚未凍結、九個 (模型 × seed)
訓練工作尚未啟動、`docs/claims.yaml` 為空。任何看似分數的數字在此之前都
不是量測結果。後續依核准計畫逐項以測試先行方式加入。

舊版 CamVid 原型沒有混在目前程式中。完整舊版由本機 Git 標籤
`legacy-v0-da35026` 保存，盤點記錄在
[`docs/verification/legacy-audit.md`](docs/verification/legacy-audit.md)。

## P1-07 指標契約

- 每個 instance 產生一筆等權重 coverage 記錄，避免大型物件以像素數壓過
  小型行人或騎士。
- 小於 50% 正確像素才標記為 critical instance miss；剛好 50% 不算 miss。
- small／medium／large 面積 tertile 只從 training annotation intersection 學習。
- semantic 與 instance sample ID 預設必須完全一致；只有明確啟用 audited
  intersection 才能繼續，而且會保留每個排除 ID 與原因。
- 空間欄位固定命名為 `normalized_image_band`，只表示 top／middle／bottom
  的正規化影像區域，不代表實體距離或深度。

## 固定工作方式

- 使用 Python 3.11 與 `uv.lock`。
- 資料集與大型產物只放在專案外、D 槽的指定目錄，不提交 Git。
- 正式訓練、批次推論與 prediction artifacts 預設使用 Colab A100。
- 未經當次明確同意，不執行長時間或 unattended 的本機 RTX 4090 工作。

```powershell
uv sync --frozen --all-groups --extra train
uv run python -m drivemetrics.dev verify
uv run driving-risk --help
```

如果驗證失敗，先處理第一個失敗項目，不要跳過或降低門檻。

## 指令

七個指令都只驗證參數並呼叫套件服務，成功時把一個 JSON 狀態物件印到 stdout，
失敗時把診斷訊息印到 stderr 並以非零狀態結束。依正式流程的順序：

```powershell
driving-risk data preflight --config configs/protocols/bdd100k_semseg_v1.yaml --data-root PATH --output PATH
driving-risk train --config configs/run_segformer_b2.yaml --manifest PATH --data-root PATH --seed 17 --output-dir PATH
driving-risk calibrate --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --checkpoint PATH --data-root PATH --output-dir PATH
driving-risk evaluate --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --checkpoint PATH --data-root PATH --output-dir PATH [--temperature PATH]
driving-risk aggregate --index PATH --output-dir PATH
driving-risk audit-claims --claims docs/claims.yaml
driving-risk report --claims docs/claims.yaml --artifacts-dir PATH --output-dir site
```

核准的三個模型：`segformer_b2`、`upernet_convnextv2_tiny`、`upernet_dinov2_small`，
各有一份 `configs/run_<model>.yaml`。

`train` 與 `evaluate` 需要 `DRIVEMETRICS_RUN_PROVENANCE` 環境變數，內容是含
`commit`、`lock_sha256` 與 `hardware` 的 JSON。這個專案不猜測執行環境，缺少就
直接失敗。
