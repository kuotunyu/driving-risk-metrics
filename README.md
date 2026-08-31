# driving-risk-metrics

P1：以 BDD100K 建立可重現的自駕感知風險評估專案。

## 目前狀態

目前只有新版的套件、驗證與 CI 基礎。尚未加入正式指標、模型、訓練、
推論結果或作品集結論；後續功能會依核准計畫逐項以測試先行方式加入。

舊版 CamVid 原型沒有混在目前程式中。完整舊版由本機 Git 標籤
`legacy-v0-da35026` 保存，盤點記錄在
[`docs/verification/legacy-audit.md`](docs/verification/legacy-audit.md)。

## 固定工作方式

- 使用 Python 3.11 與 `uv.lock`。
- 資料集與大型產物只放在專案外、D 槽的指定目錄，不提交 Git。
- 正式訓練、批次推論與 prediction artifacts 預設使用 Colab A100。
- 未經當次明確同意，不執行長時間或 unattended 的本機 RTX 4090 工作。

```powershell
uv sync --frozen --all-groups
uv run python -m drivemetrics.dev verify
uv run driving-risk --help
```

如果驗證失敗，先處理第一個失敗項目，不要跳過或降低門檻。
