# driving-risk-metrics

**只看 mIoU，會漏掉哪些與易受傷害用路人相關的分割失敗？**

[English](README.en.md) · [線上報告](https://kuotunyu.github.io/driving-risk-metrics/) · [Release v1.0.2](https://github.com/kuotunyu/driving-risk-metrics/releases/tag/v1.0.2)

自駕感知安全三案作品集之一，另外兩案是 [bev-calibration-lab](https://github.com/kuotunyu/bev-calibration-lab)（[網站](https://kuotunyu.github.io/bev-calibration-lab/)），在 nuScenes 上研究相機/LiDAR calibration fault；以及 [perception-error-to-aeb](https://github.com/kuotunyu/perception-error-to-aeb)（[網站](https://kuotunyu.github.io/perception-error-to-aeb/)），在 nuPlan 上把感知誤差送進固定的 AEB policy。
三案使用不同資料集與研究設定，不代表已驗證同一套模型從感知驅動 AEB。

三個當代語意分割模型在 BDD100K 上以同一份凍結協定訓練，各跑三個 seed，最後在一組
998 張影像的 locked cohort 上評估一次。這組 cohort 從未用於訓練、checkpoint 選擇、
溫度校準或樣本挑選。三項指標的模型順序相同，但配對 bootstrap 對前兩名能否區分
提供不同強度的證據；本 repository 也呈現像素平均容易遮蔽的 instance 層級失敗。

## 重點

- 前兩名模型的 mean IoU，SegFormer-B2 減去 UperNet-ConvNeXtV2-Tiny 的配對差為 -0.010，配對 bootstrap 區間為 -0.022 到 0.002，包含零。 <!-- claim: p1.interval.miou.segformer-minus-convnextv2; rounded: 3; fields: estimate,low,high -->
- 同樣兩個模型，改看涵蓋易受傷害用路人類別的 critical-class recall，配對差為 -0.023，區間為 -0.040 到 -0.008，不含零。 <!-- claim: p1.interval.critical-recall.segformer-minus-convnextv2; rounded: 3; fields: estimate,low,high -->
- UperNet-ConvNeXtV2-Tiny 在 462 個最小 tertile 的 person instance 中，有 306 個連一半像素都沒有正確分類。 <!-- claim: p1.instances.convnextv2.person-small -->
- 同一個模型在全部 17 個最小 tertile 的 rider instance 上，都沒有正確分類到一半像素。 <!-- claim: p1.instances.convnextv2.rider-small -->
- 同一個模型在 3749 個最小 tertile 的 car instance 中，有 822 個連一半像素都沒有正確分類。 <!-- claim: p1.instances.convnextv2.car-small -->

instance 計數來自每個模型的單一訓練 seed（seed 17，三個核准 seed 中的第一個，經溫度縮放；
溫度縮放不會改變預測類別），不是跨 seed 平均，也沒有區間。這是
[`docs/protocol.md`](docs/protocol.md) 跨 seed 平均規則的例外。rider 與 motorcycle 的最小
tertile instance 太少，不足以估計失敗率；person 的樣本可支撐較穩定的 cohort 內描述。

以像素平均的分數可能看起來不錯，但這個 cohort 中多數最小 tertile 的 person 與全部最小
tertile 的 rider 都是 critical miss。下圖列出三個模型在每一個 instance 類別上的結果：car
是數量最多的類別，也是唯一多數最小 tertile instance 被正確找回的類別，所以這種失敗不只發生
在易受傷害用路人身上。

![各類別在最小 tertile 上的 critical miss，每個模型各取一個訓練 seed，由 extended-metrics.json 繪製](docs/figures/small-tertile-critical-misses.svg)

## 發現

前兩名模型的配對 bootstrap 區間在 mean IoU 上**包含零**；同樣兩個模型，改看行人與騎士等
易受傷害用路人類別的 recall，區間**不含零**。

![前兩名模型在 mean IoU 與 critical-class recall 上的配對差與 bootstrap 區間，由 rankings.json 繪製](docs/figures/headline-top-two.svg)

<details>
<summary>完整精度陳述</summary>

> SegFormer-B2 減去 UperNet-ConvNeXtV2-Tiny 的 mean IoU 配對差為 -0.010027276977824351，bootstrap 區間從 -0.02224922437284147 到 0.0023369779504553204，包含零。 <!-- claim: p1.interval.miou.segformer-minus-convnextv2 -->

> SegFormer-B2 減去 UperNet-ConvNeXtV2-Tiny 的 critical-class recall 配對差為 -0.023304517439508565，bootstrap 區間從 -0.03998178645924999 到 -0.008046430169669789，不含零。 <!-- claim: p1.interval.critical-recall.segformer-minus-convnextv2 -->

</details>

mean IoU 差值的區間包含零，並不能證明模型等效或可以互換；兩個區間的差異也不等於直接檢驗了
兩種指標的差異。
這裡比較的是語意分割指標，沒有測試煞車決策或實車安全。

下圖把同一個 mean IoU 差值拆到各類別。mean IoU 讓十九個類別各佔相同權重，所以易受傷害
用路人類別的差距進到 mean IoU 時被稀釋了：標示出來的四個類別加總起來，已超過整個差值。
長條是 `metrics.json` 中跨 seed 平均的點估計，沒有逐類別區間，而 mean IoU 的區間本身包含零。
[`tests/contract/test_committed_figures.py`](tests/contract/test_committed_figures.py)
會從證據重畫每一張圖，並在已發布的 metrics 上檢查這個拆解。

![前兩名模型 mean IoU 差值的逐類別貢獻，標示出 critical 類別，由 metrics.json 繪製](docs/figures/miou-gap-by-class.svg)

問題不在排名。三個模型的順序在三個指標下完全相同，而且這件事會和「出現反轉」一樣
直白地被報告出來：

> 以 critical_recall 為三個模型排名，順序與以 miou 排名完全相同：未觀察到反轉。 <!-- claim: p1.ranking.critical-recall.no-reversal -->
> 以 pixel_accuracy 為三個模型排名，順序與以 miou 排名完全相同：未觀察到反轉。 <!-- claim: p1.ranking.pixel-accuracy.no-reversal -->

換指標改變的不是順序，而是這個 cohort 的 bootstrap 區間對區分前兩名提供多強的證據。

![配對差與 bootstrap 區間，由 rankings.json 繪製](docs/figures/paired-differences.svg)

## 主要結果

三個 seed 平均，在 locked cohort 上量測。主表顯示到小數點後三位，並以明確標記讓
claims validator 從原始 artifact 重算顯示值；下方保留可逐位核對的完整精度。

| 模型 | mean IoU | critical-class recall | pixel accuracy |
| --- | --- | --- | --- |
| UperNet-ConvNeXtV2-Tiny <!-- claim: p1.metrics.convnextv2; rounded: 3; fields: miou,critical_recall,pixel_accuracy --> | 0.632 | 0.811 | 0.939 |
| SegFormer-B2 <!-- claim: p1.metrics.segformer; rounded: 3; fields: miou,critical_recall,pixel_accuracy --> | 0.622 | 0.787 | 0.939 |
| UperNet-DINOv2-Small <!-- claim: p1.metrics.dinov2; rounded: 3; fields: miou,critical_recall,pixel_accuracy --> | 0.474 | 0.520 | 0.914 |

<details>
<summary>完整精度與 evidence trace</summary>

| 模型 | mean IoU | critical-class recall | pixel accuracy |
| --- | --- | --- | --- |
| UperNet-ConvNeXtV2-Tiny <!-- claim: p1.metrics.convnextv2 --> | 0.6320100232208011 | 0.8105162716623479 | 0.9387763736063249 |
| SegFormer-B2 <!-- claim: p1.metrics.segformer --> | 0.6219827462429768 | 0.7872117542228393 | 0.9385890837416806 |
| UperNet-DINOv2-Small <!-- claim: p1.metrics.dinov2 --> | 0.47424706184502113 | 0.520379600009604 | 0.9141344038041649 |

</details>

> 每一個配對區間都是對加總後的 confusion 做兩階段配對 bootstrap，信心水準 0.95，重抽 5000 次，seed 為 20260831。 <!-- claim: p1.interval.method -->

## 像素指標掩蓋掉的失敗

以像素加權的指標，會讓一台公車的票數蓋過五十個行人。這個 repository 另外以等權重
評分每一個標註 instance，並在正確分類的比例低於一半時，將該 instance 記為
**critical miss**。instance 依面積分成三個 tertile，切點只從 training split 學習。

本節每一個 instance 計數都來自各模型經溫度縮放後的 seed 17，不是跨 seed 平均，也沒有區間。
各類別 critical miss 的圖在[重點](#重點)一節。

這些結果只描述 locked cohort 中，語意與 instance 標註能互相佐證的最小 tertile
instance。person 有 462 個樣本可支撐較穩定的 cohort 內描述；rider 與 motorcycle
分別只有 17 與 14 個，極端失敗值得檢查，但不足以估計其他資料或部署條件的失敗率：

> UperNet-ConvNeXtV2-Tiny 在 462 個最小 tertile 的 person instance 中，有 306 個連一半像素都沒有正確分類。 <!-- claim: p1.instances.convnextv2.person-small -->
> UperNet-ConvNeXtV2-Tiny 在全部 17 個最小 tertile 的 rider instance 上，都沒有正確分類到一半像素。 <!-- claim: p1.instances.convnextv2.rider-small -->
> UperNet-ConvNeXtV2-Tiny 在全部 14 個最小 tertile 的 motorcycle instance 上，都沒有正確分類到一半像素。 <!-- claim: p1.instances.convnextv2.motorcycle-small -->

同一個模型、同一批 instance，換成汽車：

> UperNet-ConvNeXtV2-Tiny 在 3749 個最小 tertile 的 car instance 中，有 822 個連一半像素都沒有正確分類。 <!-- claim: p1.instances.convnextv2.car-small -->
> 跨所有類別，UperNet-ConvNeXtV2-Tiny 在 4514 個最小 tertile 的 instance 中，有 1399 個超過一半像素分類錯誤。 <!-- claim: p1.instances.convnextv2.small-overall -->

這個對比顯示，同一模型的整體像素平均不能交代各類別、各 instance 尺寸的失敗分布；
它不是實車風險或安全性的量測。

instance coverage 是在語意標註與 instance 標註互相佐證的 footprint 上量測，而不是在
原始 bitmask 上。兩種標註在物體邊界會不一致，若在只有其中一方主張的像素上評分，
等於把標註瑕疵算到模型頭上：

> instance coverage 在語意與 instance 標註互相佐證的 footprint 上，為 12860 個 instance 評分，平均佐證比例為 0.94643690780947；有 115 個 instance 沒有任何被佐證的像素而被排除。 <!-- claim: p1.instances.corroboration -->
> locked cohort 的 998 張 ground-truth mask，每一張都先對照凍結的 manifest 驗證過才評分。 <!-- claim: p1.ground-truth.masks -->

## 校準不一定有幫助

溫度縮放在獨立的 calibration split 上擬合，再套用到 locked cohort。它降低了兩個模型的
校準誤差，卻讓第三個變差：

> 溫度縮放降低了 UperNet-ConvNeXtV2-Tiny 在 locked cohort 上的 expected calibration error，從 0.004609387187919981 降到 0.0032855195799122。 <!-- claim: p1.calibration.convnextv2.ece -->
> 溫度縮放降低了 UperNet-DINOv2-Small 在 locked cohort 上的 expected calibration error，從 0.005448051902032049 降到 0.003985369701553616。 <!-- claim: p1.calibration.dinov2.ece -->
> 溫度縮放反而提高了 SegFormer-B2 在 locked cohort 上的 expected calibration error，從 0.0028840449773854925 升到 0.0035196866015977167。 <!-- claim: p1.calibration.segformer.ece -->

這三次執行都往同一個方向移動；逐 seed 數值讓這項一致性可見：

> SegFormer-B2 的每一個 seed 在溫度縮放後都朝同一方向移動：校準後為 0.003590665043061051、 0.0034857099631801494、 0.0034826847985519496，校準前為 0.0028459279224686924、 0.002904871031420072、 0.0029013359782677135。 <!-- claim: p1.calibration.segformer.ece-per-seed -->

在本實驗中，原本 ECE 已低的模型經 calibration cohort 擬合的溫度縮放後，
locked cohort 的 ECE 反而增加。逐 seed 數值補充了平均值，不能據此推論所有模型或資料集。

## 樣本稀薄的類別會被標示，不會被藏起來

> 三個模型在 train 類別上的 IoU 都是 0.0，而這個類別在 cohort 的 7 張影像中總共只有 109005 個標註像素。 <!-- claim: p1.per-class.train -->

單看那個零，像是模型的失敗；把它和支撐它的樣本量放在一起看，它是關於 cohort 的陳述。
產生的報告中每一個 per-class 列都帶著自己的像素數與影像數，出現在少於 50 張影像的
類別會被標為 thin。

準確率也隨用路人出現在畫面中的位置而變化：

> 影像中間三分之一、也就是遠處用路人出現的區域，pixel accuracy 為：UperNet-ConvNeXtV2-Tiny 0.9063993962745598、SegFormer-B2 0.9039825378430191、UperNet-DINOv2-Small 0.8681091984773754。 <!-- claim: p1.bands.middle -->

這些 band 是正規化的影像列，不是深度，也不是實體距離。

## 本頁每一個數字如何被檢查

上面每一個結果句都帶著 <code>&lt;!-- claim: ... --&gt;</code> 標記。該 claim 指名
[`docs/evidence/bdd100k_semseg_v1/`](docs/evidence/bdd100k_semseg_v1) 底下的一個
artifact、其中的一個 JSON pointer，以及該 artifact 必須帶有的 protocol 與 dataset
manifest 雜湊。兩道獨立檢查強制執行這件事，指令都列在
[檢查每一個已發布數字](#檢查每一個已發布數字只用-cpu不需資料集不需-gpu)。

第一道是 `audit-claims`，證明 registry 中每一條 claim 都能從自己的 artifact 重現。第二道是
`validate_claims.py`，讀兩份 README 與 release notes、追溯每一個帶標記的句子，並且
**回報任何同時出現指標名稱與數字卻沒有標記的行**。
沒有人能追溯的數字，正是這個專案存在要防止的失敗，所以它會讓建置失敗，而不是被發布。
同一道檢查也在測試套件中執行
（[`tests/contract/test_published_documents.py`](tests/contract/test_published_documents.py)），
所以加入無法追溯數字的 pull request，其 CI 檢查會失敗。

證據同時在一般測試執行中自我檢查：任何一個已發布數字在任何被追蹤的 artifact 中被
改動，測試套件就會失敗。

研究紀錄：

- [`docs/protocol.md`](docs/protocol.md)：凍結協定與其修訂。
- [`docs/experiment-card.md`](docs/experiment-card.md)：九次執行、雜湊與方法沿革。
- [`docs/model-card.md`](docs/model-card.md)：三個架構與其允許用途。
- [`docs/dataset-card.md`](docs/dataset-card.md)：BDD100K 來源、授權與凍結切分。
- [`docs/related-work.md`](docs/related-work.md)：這些方法所依循或與之不同的既有研究，包括 Cityscapes 的 iIoU、溫度縮放與 risk–coverage 分析。

工程與發布紀錄：

- [`docs/verification/analysis-reproduction.md`](docs/verification/analysis-reproduction.md)：分析的三次獨立執行，以及哪些部分一致。
- [`docs/verification/mutation-audit.md`](docs/verification/mutation-audit.md)：純核心的 mutation 分數與每一個存活 mutant 的處置。
- [`docs/release-notes/`](docs/release-notes)：每一個版本改了什麼。

## 重現方式

需要 uv 0.11.x：`pyproject.toml` 要求 `>=0.11.18,<0.12`，所以 uv 0.12 以後的版本會拒絕執行，
CI 使用 0.11.18。可用 `pipx install uv==0.11.18` 或指定版本的安裝程式安裝
（[install.sh](https://astral.sh/uv/0.11.18/install.sh)；Windows 用
[install.ps1](https://astral.sh/uv/0.11.18/install.ps1)）。

### 檢查每一個已發布數字（只用 CPU，不需資料集，不需 GPU）

以下就是 Pages workflow 在發布報告前執行的指令：

```bash
uv sync --frozen --all-groups
uv run --frozen driving-risk audit-claims --claims docs/claims.yaml
uv run --frozen python .agents/skills/auditing-driving-risk-claims/scripts/validate_claims.py \
  --claims docs/claims.yaml --repo-root . --document README.md --document README.en.md \
  --document docs/release-notes/v1.0.0.md --document docs/release-notes/v1.0.1.md \
  --document docs/release-notes/v1.0.2.md
uv run --frozen driving-risk report --claims docs/claims.yaml --artifacts-dir docs/evidence/bdd100k_semseg_v1 --output-dir site
```

在一份全新複本、沒有 GPU 的 Windows 電腦上，sync 之後的三道指令不到十秒就跑完。
第一次 sync 會下載鎖定的套件，所需時間取決於網路。

### 完整開發檢查

```bash
uv sync --frozen --all-groups --extra train --extra report
uv run --frozen python -m drivemetrics.dev verify
uv run --frozen driving-risk --help
```

`verify` 以固定順序執行八個階段，並在第一個失敗處停止：私有檔案守衛、格式檢查、
lint、型別檢查、完整測試套件、第一方程式碼的 100% statement 與 branch 覆蓋率、
schema 契約、文件連結。沒有任何覆蓋率豁免，沒有 `pragma: no cover`，也沒有被排除的
第一方路徑。

### 正式流程（需要 BDD100K 與 GPU）

正式流程，依順序：

```bash
driving-risk data preflight --config configs/protocols/bdd100k_semseg_v1.yaml --data-root PATH --output PATH
driving-risk data tertiles --manifest PATH --labels-root PATH --instance-root PATH --output PATH
driving-risk train --config configs/run_segformer_b2.yaml --manifest PATH --data-root PATH --seed 17 --output-dir PATH --device cuda
driving-risk calibrate --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --checkpoint PATH --data-root PATH --output-dir PATH --device cuda
driving-risk evaluate --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --checkpoint PATH --data-root PATH --output-dir PATH --device cuda --temperature PATH
driving-risk index --runs-root PATH --config configs/protocols/bdd100k_semseg_v1.yaml --manifest PATH --risk-profile configs/risk_profiles/vru_priority.yaml --output PATH
driving-risk aggregate --index PATH --output-dir PATH
driving-risk gallery --index PATH --output PATH
driving-risk extended-metrics --index PATH --output PATH
driving-risk report --claims docs/claims.yaml --artifacts-dir docs/evidence/bdd100k_semseg_v1 --output-dir site
```

九次訓練在 A100 上各約 12 小時。分析則在 CPU 上從儲存的 prediction artifact 執行，
已在不同日期與不同 runtime 上執行三次；各次之間一致的部分記錄在
[`docs/verification/analysis-reproduction.md`](docs/verification/analysis-reproduction.md)。

BDD100K 不在此再散布，checkpoint 與約 54 GiB 的逐影像 prediction artifact 也不在此
散布。被提交的證據是 claim 所引用的分析輸出。

## 這個專案不支持哪些結論

- **其他地區與其他相機。** BDD100K 是特定蒐集條件下的行車紀錄器影像。沒有新證據之前，
  這裡的任何結果都不能外推到其他感測器、安裝方式或地區。
- **深度或距離。** 影像 band 是正規化的影像列。這個 repository 沒有任何深度估計。
- **即時推論。** 沒有量測也沒有主張任何延遲、吞吐量或嵌入式部署的性能。
- **核准清單以外的模型。** 只有 `segformer_b2`、`upernet_convnextv2_tiny` 與
  `upernet_dinov2_small` 在此協定下訓練過。結果是關於「以這種方式訓練的這些模型」，
  不是關於這些架構本身。
- **量產安全論證。** instance coverage 與 risk-weighted cost 是評估工具，它們不是安全
  論證，也不能取代安全論證。

## 授權

MIT，見 [LICENSE](LICENSE)。BDD100K 由其作者以自身條款散布，不在此再散布，詳見
[`docs/dataset-card.md`](docs/dataset-card.md)。

三個預訓練 backbone 是第三方權重，各 checkpoint 已發布的條款並不相同，這裡也不散布任何
checkpoint。每一個的論文與所發布的授權（附連結）列在
[`docs/model-card.md`](docs/model-card.md#third-party-pretrained-weights)。
