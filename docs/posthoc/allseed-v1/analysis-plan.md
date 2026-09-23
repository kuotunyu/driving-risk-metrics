# Post-release analysis: every seed's instance coverage, and intervals for the point-estimate metrics

**Pre-registered analysis plan.** Dated 2026-09-23. This document was committed to
the repository before any number below existed: instance coverage had been
computed for seed 17 only, and none of the intervals in §4 had been computed, when
it was written. Its commit is the time stamp of the pre-registration. Anything
reported later that departs from this plan will be labelled exploratory.

This is a post-release analysis of the v1.0.x release. It re-aggregates the
per-image prediction artifacts that the released evaluation already wrote. No model
is run, no prediction is made, and no published number, claim, figure, tag or
release is changed.

## 1. Why

Two deviations from the protocol are recorded in the
[experiment card](../../experiment-card.md#deviations-from-the-stated-protocol-noted-2026-09-23):

- every instance-coverage number comes from one training seed per model (seed 17),
  with no interval, although the protocol says every reported number is a mean over
  seeds with an interval, never a single seed;
- intervals exist only for paired differences in mean IoU, pixel accuracy and
  critical-class recall. Expected calibration error, Brier score, selective risk
  (AURC), per-class IoU and risk-weighted cost are seed means without intervals.

This analysis computes what those rules asked for, from the artifacts the nine
runs already produced.

## 2. Questions

1. **Seed robustness of the instance finding.** The README's instance statements are
   about small people. Averaged over the three seeds, is the small-tertile person
   critical-miss rate of each model above one half, as seed 17 suggested, and how
   far is seed 17 from the seed mean?
2. **Model separation on small people.** For each pair of models, is the difference
   in the seed-mean small-person critical-miss rate distinguishable from zero?
3. **Intervals for the point-estimate metrics** (descriptive): how uncertain are the
   published confidence-based and per-class numbers, and which paired differences
   are distinguishable?

## 3. Data and cohort boundary

- **Cohort.** The locked validation cohort: 998 images, dataset manifest SHA-256
  `d43bda8f556747f8500bc14269957f2218c52c02917358e5456ec75faa12944f`. Its
  predictions were made once, by the released evaluation; this analysis only reads
  them.
- **Runs.** The nine formal runs listed in
  [`formal_run_index.json`](../../evidence/bdd100k_semseg_v1/formal_run_index.json),
  with both their uncalibrated (`eval`) and temperature-scaled (`eval_calibrated`)
  artifact directories.
- **Ground truth.** The semantic masks verified file by file against the frozen
  manifest, the instance bitmasks whose set digest must equal the released
  `247f574897745aae23bb9aef4ec263e26c22388bedb844bce11cc75602d80caf`, and the frozen
  area tertiles (`5f9365d5b9189b49649e34fc8403f16f4934d630ceb5cf903429007a52997206`).
- **Why the locked cohort is not spent again.** The protocol's first rule is that the
  locked cohort is spent the moment it influences a decision. Nothing here is a
  decision: no checkpoint, threshold, seed, rerun or stopping point is chosen from
  these numbers, the 0.5 critical-miss threshold stays the released definition, and
  the threshold sweep of §5 is descriptive only.
- **Instance scoring.** Exactly the released rule: corroborated instances (pixels on
  which the semantic mask and the instance annotation agree), the frozen tertiles,
  and critical miss = `correct_fraction < 0.5`, applied to every seed's
  `eval_calibrated` predictions (temperature scaling does not change the predicted
  class; §6 checks this).

## 4. Statistics common to every interval

- **Resampling.** The released two-stage paired bootstrap
  (`two_stage_paired_bootstrap_statistic` in
  [`bootstrap.py`](../../../src/drivemetrics/analysis/bootstrap.py)): each resample
  draws the 998 images with replacement, shared by every run in the comparison, sums
  each run's per-image components and recomputes the metric from the sums, then draws
  the three seeds with replacement inside each model. 5,000 resamples, generator seed
  20260831, percentile intervals. A model's level is the mean over its seeds; a paired
  difference is the difference of the two models' seed means.
- **Primary intervals** (§5.1, §5.2) are Bonferroni-adjusted over three: 98.33%
  (1 − 0.05/3). The 95% interval is reported beside them as a description.
- **Secondary intervals** (§5.4) are 95%, descriptive, with no multiplicity
  adjustment, like the released intervals.
- **AURC.** A resample's pooled confidence histogram is the image-weight vector times
  each image's 65,536-level histogram. The weight vectors are generated in exactly the
  order the released function draws them, and the products are computed as exact
  integer sums, so the result equals what the released function would give, without
  holding every histogram in every resample. No coarser grid is used.

## 5. Pre-registered analysis

### 5.1 Primary: the small-person critical-miss rate of each model

For each model *m*: *R_m*, the mean over seeds of (critical misses ÷ instances) among
the corroborated small-tertile person instances of the locked cohort, with its 98.33%
interval. Decision per model:

- **majority missed:** the lower bound is above 0.5;
- **minority missed:** the upper bound is below 0.5;
- **undetermined:** otherwise.

Reported beside it: each seed's rate, and seed 17's rate minus *R_m*.

### 5.2 Primary: paired differences between models

For each of the three model pairs, in the released order (SegFormer-B2 minus
UperNet-ConvNeXtV2-Tiny, SegFormer-B2 minus UperNet-DINOv2-Small,
UperNet-ConvNeXtV2-Tiny minus UperNet-DINOv2-Small): the difference of seed-mean
small-person critical-miss rates, with its 98.33% interval. The pair is
**separable** when the interval excludes zero, and **not separable** otherwise. A
non-separable pair is not evidence that the two models are equivalent.

McNemar tests are not used: instances cluster within images, which breaks their
independence assumption, and seeds of different models are not paired.

### 5.3 Secondary: the full instance block for every seed

The released instance block (every instance class, overall and by tertile: instance
count, critical misses and mean correct fraction) for each of the nine runs, and the
seed mean for each model. Classes with very few small instances (rider, motorcycle)
are reported as counts only; no rate is interpreted for them.

### 5.4 Secondary: intervals for the point-estimate metrics

For each model's seed-mean level and for the three paired differences, 95% intervals
for:

1. classwise expected calibration error, the published definition (mean over the 19
   classes of one-vs-rest ECE with 15 equal-width bins), uncalibrated and calibrated;
2. top-label expected calibration error with 15 equal-width bins, uncalibrated and
   calibrated. It is not a published metric; it is added because it is the form most
   readers know. The bins align exactly with the stored 65,536-level confidences;
3. Brier score, uncalibrated and calibrated;
4. selective risk (AURC), uncalibrated and calibrated, by the exact method of §4;
5. IoU of each vulnerable-road-user class: person, rider, motorcycle and bicycle;
6. risk-weighted cost for the `vru_priority` and `drivable_boundary` profiles. The
   `balanced` profile is omitted because it equals one minus pixel accuracy, whose
   interval is already published.

### 5.5 Secondary: critical-miss threshold sweep

For the small-tertile person instances, the seed-mean critical-miss rate of each model
at thresholds 0.1, 0.2, …, 0.9 (strict less-than, compared as integers of correct and
footprint pixels), and whether the order of the three models at each threshold equals
the order at 0.5. Descriptive only; the released 0.5 definition does not change.

## 6. Gates before any result is read

Every gate must pass; a failure stops the analysis for investigation, and no result
is read.

1. **Integrity.** For every artifact: its payload SHA-256 equals the value recorded in
   its evaluation directory's `run_record.json`; its protocol and dataset hashes equal
   the run index; and `predicted_class` and `confusion` are byte-identical between
   `eval` and `eval_calibrated`.
2. **Reproduction of released numbers.** Computed by the new code from the artifacts,
   each must equal the released value exactly:
   - the nine paired intervals in
     [`intervals.json`](../../evidence/bdd100k_semseg_v1/intervals.json);
   - seed 17's instance block in
     [`extended-metrics.json`](../../evidence/bdd100k_semseg_v1/extended-metrics.json);
   - the seed-mean AURC values in `extended-metrics.json`;
   - the seed-mean ECE and Brier values in
     [`metrics.json`](../../evidence/bdd100k_semseg_v1/metrics.json).

## 7. What will not be done

No threshold, seed, model or subset is selected after the fact. No released number is
replaced: new numbers appear only in a separate, clearly labelled document with its
own claims file. No per-image or per-instance table, image, crop or bitmask is
published.

## 8. Risks

- Only three seeds per model, so the seed stage of every interval rests on three
  values.
- Instances cluster within images; the image-level resampling accounts for this, a
  per-instance test would not.
- The instance annotations come from the third-party mirror disclosed in the dataset
  card; the set that is read is recorded by its digest, not verified against an
  official release.
- The secondary intervals are many and unadjusted; they describe uncertainty and are
  not a family of tests.
