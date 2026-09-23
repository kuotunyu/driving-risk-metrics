# Model card: the three approved segmentation models

Three models were trained under `bdd100k_semseg_v1`, one per pretraining paradigm.
Half the metrics in this study measure confidence, and pretraining shapes
confidence, so the three were chosen to differ in how they were pretrained rather
than only in size.

## What was trained

| Model name | Architecture | Pretrained backbone | Pretraining paradigm |
| --- | --- | --- | --- |
| `segformer_b2` | SegFormer-B2, hierarchical transformer encoder with an all-MLP decoder | `nvidia/mit-b2` | supervised ImageNet classification, 2021 |
| `upernet_convnextv2_tiny` | UperNet decoder on a ConvNeXtV2-Tiny backbone | `facebook/convnextv2-tiny-1k-224` | fully convolutional masked autoencoding, 2023 |
| `upernet_dinov2_small` | UperNet decoder on a DINOv2-Small backbone | `facebook/dinov2-small` | self-supervised distillation, 2023 |

Every pretrained classification head is discarded and replaced with a fresh
19-class head. `ConvNextV2Model` reports its `classifier.weight` and
`classifier.bias` as unexpected when the backbone is loaded, because the published
checkpoint is an ImageNet-1k classifier and this repository loads only its backbone.
That message is the protocol working as specified, not a defect.

## Third-party pretrained weights

The backbone weights come from third parties and carry their own published terms.
This table records what each source states, with links, as checked on 2026-09-23.
It draws no legal conclusion.

| Checkpoint | Paper | Licence as published |
| --- | --- | --- |
| `nvidia/mit-b2` | Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021, [arXiv:2105.15203](https://arxiv.org/abs/2105.15203) | The [Hugging Face card](https://huggingface.co/nvidia/mit-b2) declares `license: other` and links the [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE), which restricts the Work and its derivative works to non-commercial use and defines that as "research or evaluation purposes only". |
| `facebook/convnextv2-tiny-1k-224` | Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders", CVPR 2023, [CVF open access](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html) | The [Hugging Face card](https://huggingface.co/facebook/convnextv2-tiny-1k-224) declares `license: apache-2.0`. The [upstream ConvNeXt-V2 README](https://github.com/facebookresearch/ConvNeXt-V2#license) states that the project is released under the MIT license except the ImageNet pre-trained and fine-tuned models, which it lists as CC-BY-NC. The two sources differ, and this repository does not say which governs. |
| `facebook/dinov2-small` | Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", Transactions on Machine Learning Research 2024, [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | The [Hugging Face card](https://huggingface.co/facebook/dinov2-small) declares `license: apache-2.0`, and the [upstream DINOv2 README](https://github.com/facebookresearch/dinov2#license) states that code and model weights are released under the Apache License 2.0. |

The UperNet decoder follows Xiao et al., "Unified Perceptual Parsing for Scene
Understanding", ECCV 2018 ([CVF open access](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html)).
It is built from its configuration and trained from random initialisation; no
pretrained UperNet weights are loaded.

## Training recipe, identical across the three except where the protocol differs

| Setting | Value |
| --- | --- |
| Input | resized to 512 by 910, padded to 512 by 1024, mask pad value 255 |
| Steps | 30000, warmup 1000 |
| Effective batch size | 16, micro-batch 8 |
| Augmentation | horizontal flip, probability 0.5 |
| Optimizer | AdamW |
| Learning rate | 0.00006 for `segformer_b2`, 0.0001 for the other two |
| Weight decay | 0.01 for `segformer_b2`, 0.05 for the other two |
| Checkpoint selection | final step only |
| Seeds | 17, 42, 73 |
| Calibration | scalar temperature, fitted by multiclass negative log likelihood on the calibration split |

Checkpoint selection is `final_step_only` by design. Selecting a checkpoint on any
held-out set would make that set a tuning set, and this study has only one
evaluation, taken once.

## Checkpoints produced

Nine checkpoints, all distinct, listed with their SHA-256 and fitted temperature in
[`experiment-card.md`](experiment-card.md). The weights are not redistributed in
this repository. Their digests are published so that a reader who obtains or
reproduces them can prove they hold the same artifact the results were computed
from.

## What these models may be used for

- Reproducing the measurements in this repository from the same cohort and protocol.
- Studying how pretraining paradigm relates to confidence calibration and to
  instance-level coverage on road scenes.
- As a baseline to compare a new evaluation method against, since the prediction
  artifacts are stored and the analysis is deterministic.

No checkpoint is redistributed here. Anyone reusing these checkpoints, or weights
derived from them, should review the upstream terms listed under
[Third-party pretrained weights](#third-party-pretrained-weights); this repository
makes no statement about their legal effect.

## What these models must not be used for

- **Any deployed driving function.** These are research checkpoints trained on one
  dataset with no validation of any kind outside it.
- **A claim about the architectures in general.** Each was trained once per seed
  under one recipe. A different recipe, resolution or schedule could reorder them,
  and no such search was performed.
- **Transfer to other regions, cameras or mounting positions**, or to night, weather
  or lighting conditions not represented in the cohort, without new evidence.
- **Depth, distance or 3D reasoning.** These models produce 2D semantic labels only.
- **Real-time claims.** Latency and throughput were not measured.

## Known weaknesses, measured rather than assumed

- All three score 0.0 IoU on class `train`, which appears in 7 images of the locked
  cohort carrying 109005 labelled pixels. This is reported with its support in the
  READMEs and in the generated report.
- On the smallest area tertile, all three models fail the classes the study
  protects far more often than they fail cars. The per-class, per-tertile figures
  are in [`evidence/bdd100k_semseg_v1/extended-metrics.json`](evidence/bdd100k_semseg_v1/extended-metrics.json).
- Temperature scaling raised `segformer_b2`'s calibration error on the locked
  cohort while lowering the other two models'. The per-seed values are published so
  that this can be read rather than inferred.

## Provenance

- Protocol SHA-256 `b33c842250f6afcc7bd7c1108b29bf84f342dda5bb5420e64d6be48773c4369f`.
- Training data manifest SHA-256 `1d621fd5a86d6e7cb35d090fee0e60ae8580904a4ea2ccbb5f98b94bff84ef68`.
- Dataset provenance and licence: [`dataset-card.md`](dataset-card.md).
- Every published number and the artifact it comes from: [`claims.yaml`](claims.yaml).
