# Related work

This page places the methods in this repository next to the published work they
follow or depart from. It is a reading guide for reviewers, not a survey. The text
cites by author; full references with links are at the end. The dataset and
backbone papers are cited in the [dataset card](dataset-card.md) and the
[model card](model-card.md#third-party-pretrained-weights).

## Instance-balanced scoring and Cityscapes iIoU

The README motivates equal instance weighting by saying that pixel-weighted metrics
let one bus outvote fifty pedestrians. Cordts et al. made the same observation for
Cityscapes: the standard intersection-over-union is biased toward instances that
cover a large image area. Their instance-level score, iIoU, weights each
ground-truth pixel by the ratio of its class's average instance size to the size
of its own instance.

Instance coverage in this repository shares that motivation but measures something
different:

- **Per instance, not per class.** iIoU still reduces to one pixel-level score per
  class. Here every annotated instance gets its own record, the fraction of its
  pixels classified correctly, and the report counts the instances below one half
  (critical misses) by class and by size tertile.
- **Corroborated footprint, not raw mask.** Coverage is measured only where the
  semantic and instance annotations agree, so boundary disagreement between the
  two annotations is not charged to the model. Instances with no corroborated
  pixel are excluded, and the mean corroborated fraction is published.
- **No false-positive term.** iIoU keeps unweighted false positives in its
  denominator. Instance coverage is recall-like and ignores false positives, so it
  is read beside class-level scores such as mean IoU, never instead of them.
- **Size strata from training data only.** Tertile edges are learned per class
  from the training split alone.

## Temperature scaling

Guo et al. showed that modern deep networks are often poorly calibrated, and found
temperature scaling, a single scalar that divides the logits and is fitted by
negative log-likelihood on held-out data, effective on most of the datasets they
studied. This repository uses that recipe unchanged: one temperature per
checkpoint, fitted on the calibration split, applied to the locked cohort. The
README reports one model whose calibration error on the locked cohort rose after
scaling, as an observation about this experiment only.

## How calibration error is measured

Nixon et al. showed that calibration-error estimates depend on choices that are
often left implicit: whether only the top-label confidence is binned or every
class probability is, whether bins are equal-width or adaptive, and whether tiny
probabilities are thresholded. They report that conditioning on the class gives
more effective evaluations and that adaptive binning gives more stable rank
orderings.

The expected calibration error published here is computed one-vs-rest for every
class, in fixed equal-width bins weighted by their pixel counts, and averaged over
the classes with support, with no probability threshold
([`calibration.py`](../src/drivemetrics/metrics/calibration.py)). That is close to
what Nixon et al. call static calibration error. No adaptive-binning or
thresholded variant was computed, so whether the reported calibration changes hold
under those estimators has not been tested.

## Selective prediction and risk–coverage

Geifman and El-Yaniv studied selective classification for deep networks: a
classifier abstains when its confidence falls below a threshold, and the trade-off
between coverage and the error rate on accepted samples is read from a
risk–coverage curve. Their softmax response, the maximum softmax probability, is
the confidence by which this repository ranks pixels. Geifman, Uziel and El-Yaniv
later proposed the area under the risk–coverage curve (AURC) as a single summary
of a confidence function, together with an excess variant, E-AURC, that subtracts
the area achieved by the best possible ranking.

Here the curve is rebuilt at bin boundaries from a quantized confidence histogram,
and the trapezoid area is divided by the covered span, so the value is a
coverage-weighted mean selective risk. E-AURC is not computed.

## Comparing two intervals

Gelman and Stern point out that the difference between a significant and a
non-significant result is not itself statistically significant. The headline
follows the same caution: it reports two separate paired intervals, one per
metric, and states that comparing them is not a test of a difference between the
metrics. This page adds no claim to that result.

## Safety standards

Missed small vulnerable road users are an example of what ISO 21448 (SOTIF)
calls a performance insufficiency, and of what ISO/PAS 8800 calls an output
insufficiency of an AI element; UL 4600 evaluates autonomous products through a
structured safety case. This repository is not a safety case, follows none of
these processes, and claims no compliance with any of these standards. Its
measurements come from one dataset, one frozen cohort and research checkpoints,
and would need an argument about operating conditions and acceptance criteria
before they could support any safety claim.

## 中文摘要

本頁整理本 repository 的方法所依循、或與之不同的既有研究，並非文獻回顧。
等權重的 instance 評分與 Cityscapes 的 iIoU 出於相同動機：以像素加權的指標偏向面積大的
instance。不同之處在於，這裡為每一個 instance 各自記錄它在語意與 instance 標註互相佐證的
footprint 上被正確分類的比例，沒有 false positive 項，並依類別與 tertile 計數 critical miss。
溫度縮放沿用 Guo 等人的做法。發布的校準誤差是逐類別（one-vs-rest）、固定等寬分箱的版本，
沒有計算 Nixon 等人討論的 adaptive 或 thresholded 變體。
risk–coverage 分析沿用 Geifman 與 El-Yaniv 的 selective classification 框架，沒有計算 E-AURC。
本 repository 不是安全論證（safety case），
也不主張符合 ISO 21448、ISO/PAS 8800 或 UL 4600 中的任何一項。

## References

- Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B. The Cityscapes Dataset for Semantic Urban Scene Understanding. CVPR 2016. [CVF open access](https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html)
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML 2017, PMLR volume 70. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
- Nixon, J., et al. Measuring Calibration in Deep Learning. CVPR Workshops 2019. [CVF open access](https://openaccess.thecvf.com/content_CVPRW_2019/html/Uncertainty_and_Robustness_in_Deep_Visual_Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.html), [arXiv:1904.01685](https://arxiv.org/abs/1904.01685)
- Geifman, Y., El-Yaniv, R. Selective Classification for Deep Neural Networks. Advances in Neural Information Processing Systems 30 (NIPS 2017). [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
- Geifman, Y., Uziel, G., El-Yaniv, R. Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers. ICLR 2019. [arXiv:1805.08206](https://arxiv.org/abs/1805.08206)
- Gelman, A., Stern, H. The Difference Between "Significant" and "Not Significant" is not Itself Statistically Significant. The American Statistician, 60(4), 328–331, 2006. [doi:10.1198/000313006X152649](https://doi.org/10.1198/000313006X152649)
- ISO 21448:2022, Road vehicles — Safety of the intended functionality. [ISO catalogue](https://www.iso.org/standard/77490.html)
- ISO/PAS 8800:2024, Road vehicles — Safety and artificial intelligence. [ISO catalogue](https://www.iso.org/standard/83303.html)
- UL 4600, Standard for Evaluation of Autonomous Products, edition 3, 2023. [UL Standards & Engagement catalogue](https://www.shopulstandards.com/ProductDetail.aspx?productid=UL4600)
