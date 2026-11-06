# CUED-Net R2: pre-specified analysis plan

Committed before any locked model is evaluated on the held-out split. The commit
hash of this file is quoted in the manuscript. `evaluation/p2_analysis.py` implements it.

## 1. Data
- Development cohort: 528 lesion pairs from 482 patients (258 malignant), CBIS-DDSM
  official training split. Pairs are keyed on patient, side and the abnormality id in
  the study name. Patients keep their frozen R1 folds (`data/cv_folds.json`, md5
  10b4dbd4e933575e2ffd72048869eefd). Manifest md5 5aef69eb5e82bf80eb29eb9771f392f8.
- Held-out split: 154 lesion pairs from 143 patients (60 malignant), CBIS-DDSM
  official test split, no patient overlap with development. It is not
  a pristine set. It produced the January 2026 results and ablation, and was
  evaluated twice on 1 June 2026 during reconstruction of the patch pipeline,
  which informed the use of cropped images. No R2 choice uses it.
- External: CMMD, reported separately. Labels are breast-level, inputs are whole-breast
  crops without a lesion ROI, lesion types are mixed.

## 2. Protocol
- Inner validation: 85/15 split of each outer training fold, grouped by patient,
  stratified by label, fixed across models and seeds.
- Inner validation alone sets early stopping, temperature, referral thresholds,
  uncertainty score weights and the fixed-specificity threshold.
- Early stopping starts when the encoders are unfrozen (epoch 6), patience 15,
  50 epochs at most. A pilot on inner validation only showed that counting from
  epoch 1 stopped some runs before the encoders had trained. No outer-fold or
  held-out result informed this change.
- Decision threshold: 0.5, fixed in advance, same for all methods.
- Seeds: 42, 123, 456, 789, 2024. All settings are in `training/p2_config.py`.
- Methods: CUED-Net, TMC, single-view EDL, cross-view transformer, softmax dual-view
  (used as MC-Dropout with T = 50, and as Deep-Ensemble with M = 5 seed models).
  CUED-Net with a ConvNeXt-T encoder is a backbone comparison, reported apart.
  This list was fixed before any outer-fold or held-out metric was computed.

## 3. Inference
- Unit: lesion pair, resampled by patient. B = 10,000, percentile intervals.
- Single-model metrics are computed per seed in each bootstrap sample, then averaged.
  The same resamples are used for all methods, which gives paired differences.
- Deep-Ensemble gives one prediction per case and has no per-seed row.
- Holm correction across the comparators, per metric.
- Calibration errors use a bias-shifted percentile interval, because binned
  errors are biased upward under resampling.

## 4. Primary comparisons, CUED-Net against each comparator
AUC, F1 at 0.5, AURC and E-AURC. A significant F1 difference is claimed only from
the clustered paired test. Equivalence applies to cross-validated AUC only: margin
0.03, claimed only if the 90% paired interval lies inside it. Otherwise the text
says that no significant difference was detected. The held-out split is too
small for an equivalence claim.

## 5. Added value of view discordance
With e = 1 when the prediction at 0.5 is wrong:

    M0: logit P(e=1) = b0 + b1 u_evid
    M1: logit P(e=1) = b0 + b1 u_evid + b2 u_disc

Both are fitted on inner validation with standardised inputs and scored on the
outer fold. Report the differences in error-detection AUROC and in AURC with
patient-clustered intervals, plus the error rate by evidential and discordance
tercile. Added value is claimed only if the interval of the AURC difference
excludes zero. If it does not, the paper reports a null result and keeps
discordance only as a descriptive attribution signal.

## 6. Secondary analyses
- Sensitivity, specificity, PPV, NPV, FNR, AP. PPV and NPV hold at cohort prevalence.
- Sensitivity at 90% specificity, threshold from inner validation.
- F1 at the inner-validation F1-maximising threshold.
- ECE (15 equal-width and equal-mass bins), MCE, Brier, NLL, raw and calibrated.
  Calibrated probability: sigmoid(logit(p) / T), T fitted per fold and seed by NLL.
- Referral at 90, 80 and 70% inner-validation coverage: sensitivity among accepted
  cases, share of malignant cases referred, share of errors captured.
- Dependence between components: Spearman, distance correlation, kNN mutual
  information, with 999 permutations. Pearson r alone is not used.
- Decision curves for threshold probabilities of 5 to 50%, labelled retrospective.
- Sensitivity analysis excluding breasts with more than one mass.

## 7. Held-out evaluation
All locked models are trained first. `evaluation/evaluate_test.py` then runs once
for all methods and writes a lock file. Any later run is disclosed in the paper.

## 8. Uncertainty scores of comparators
TMC and single-view EDL use vacuity, MC-Dropout the entropy of the mean over
T = 50 passes, Deep-Ensemble the entropy of the mean over five seed models, and
the cross-view transformer the predictive entropy of a single pass.

## 9. External evaluation on CMMD
- Cohort: every breast with both CC and MLO and a clinical label. Labels are
  breast-level. Inputs are label-blind whole-breast crops with no lesion ROI.
  Lesion types are mixed. Patients are assigned once, seed 2026, stratified by
  any malignant breast: 10% adaptation training, 20% adaptation validation,
  70% evaluation.
- Zero-shot: fold -1 models unchanged. Temperature and thresholds come from
  CBIS-DDSM inner validation. No CMMD case informs any choice. Reported on all
  pairs, on the evaluation patients and on the mass-containing subset of them.
- Adapted, CUED-Net and TMC only: encoders frozen with batch norm in evaluation
  mode, heads retrained for at most 30 epochs with patience 8, selection,
  temperature and thresholds on adaptation validation. Reported on the evaluation
  patients and their mass-containing subset. Zero-shot and adapted results stay in
  separate rows, and only the evaluation-patient rows are compared.

## 10. Saliency against lesion masks
Grad-CAM of the predicted class evidence for each view of every outer-fold case and
seed, on the cross-validation models, so the held-out split is not read. Masks are
placed by locating each crop in its full mammogram. Views with a match score under
0.8 are left out and counted. Measures: share of saliency energy inside the mask,
that share divided by mask area (1 means no concentration), and whether the peak
falls inside the mask, by correctness and evidential tercile, with patient-clustered
intervals. Crops are lesion-centred, so the mask covers much of the input and the
chance level is reported beside each value.

## 11. Cost
Parameters, operations and per-pair latency at batch size 1 on one GPU, for all
methods in one script, with MC-Dropout at T = 50 and Deep-Ensemble at M = 5.

## 12. Amendment of 5 November 2026
Made after the outer-fold results of sections 4 and 5 were seen and before any
locked model was evaluated on the held-out split. Sections 1 to 11 stay as written,
and their results are reported as pre-specified, including the null result of
section 5 for the fitted score (dAURC +0.0004, 95% CI -0.0137 to 0.0136).

Reason. The fitted score used vacuity and discordance only, on about 63 inner
validation pairs per fold, and ranked errors worse than the comparators. It ignored
the distance of the fused probability from the decision threshold.

Uncertainty score. CUED-Net now reports the fused predictive entropy H(p), which has
no fitted parameter, and its exact split in bits (`evaluation/decomposition.py`):

    H(p) = sum_v w_v E[H(theta_v)]               within-view ambiguity
         + sum_v w_v (H(p_v) - E[H(theta_v)])    evidential uncertainty, mutual information
         + (H(p) - sum_v w_v H(p_v))             view discordance uncertainty, weighted JS

The contribution of view discordance uncertainty is the AURC difference between H(p)
and the sum of the first two terms. The dependence between the terms is reported. For a fair
comparison, TMC and single-view EDL are scored by both vacuity and entropy.

Backbone. Each comparator is retrained with ConvNeXt-T under the same protocol, so a
method comparison is never confounded with a backbone comparison.

Selection rules, fixed before those runs. All use inner validation of the outer folds:
- Primary backbone: DenseNet-121, the configuration under review. ConvNeXt-T is a
  complete second arm with its own comparators, reported as the data show, and is not
  selected as primary. This line replaces an earlier wording of 5 November and was
  fixed before any result of the ConvNeXt-T comparators existed.
- Evidence scale: `cued_convnext_k5` replaces `cued_convnext` in the ConvNeXt-T arm only
  if it has both the lower inner-validation AURC of H(p) and the lower inner-validation
  NLL before temperature scaling. If it does, the same scale is trained for DenseNet-121
  and judged by the same rule.
The number of variants tried is reported. The outer folds informed this amendment, so
outer-fold differences that favour the amended method are labelled post hoc. The held-out
split is the confirmatory evaluation, and it runs once for all locked methods.

Confirmatory claims, in order, DenseNet-121 arm first, then the ConvNeXt-T arm:
1. View discordance uncertainty lowers AURC: H(p) against the sum of the other two terms.
2. AURC of CUED-Net H(p) against each same-backbone comparator, under vacuity and entropy
   where both exist, with Holm correction per arm.
Equivalence is never claimed on the held-out split. Comparisons across backbones are
reported as a backbone effect, not as a method effect.

Also reported, added on 5 November 2026 before any held-out evaluation, after rereading the
reviewer comments. None of these changes a confirmatory claim:
- The literal test asked by reviewer R3.2, view term added to the evidential term alone,
  and to the ambiguity term alone, beside the H(p) comparison.
- Dependence between the three terms: Spearman, distance correlation and kNN mutual
  information, with 999 permutations of whole patients among patients of equal size.
- Sensitivity among referred malignant cases, beside sensitivity among accepted cases.
- Fusion rule from saved outputs, no retraining: evidence-weighted, equal weights,
  view with more evidence, CC only, MLO only.
- Share of cases with the two views on opposite sides of 0.5, error rate in each group,
  and the error rate by terciles of the two terms.
`evaluation/p2_analysis.py` ranks CUED-Net by H(p) and keeps the score of section 5 as
`cued_net_fitted`, so both results come from one script run.

Outcome of the evidence-scale rule, 5 November 2026, from `evaluation/select_p2.py` on the 25
inner-validation sets of the outer folds. Inner AURC of H(p) 0.0955 against 0.0992, inner NLL
before temperature 0.439 against 0.502. `cued_convnext_k5` therefore replaces `cued_convnext`
in the ConvNeXt-T arm. The decision was made before the outer folds of `cued_convnext_k5`
were read, and it stands although those outer folds later showed slightly worse values
(AUC 0.886 against 0.890, AURC 0.089 against 0.086). `cued_convnext` stays in the tables as
a sensitivity result, outside the Holm family. `cued_net_k5` is trained next and judged by
the same rule against `cued_net`. Evidence scale is the one setting selected for CUED-Net
alone, and the comparators keep scale 1.

Outcome for DenseNet-121, 5 November 2026, read before the outer folds of `cued_net_k5`.
Inner AURC of H(p) 0.1276 against 0.1238 for `cued_net`, inner NLL before temperature 0.481
against 0.526. Only one of the two conditions holds, so `cued_net` stays the primary model at
scale 1. The two arms therefore differ in evidence scale, as the rule decided: DenseNet-121 at
scale 1 and ConvNeXt-T at scale 5.

## 13. Checks of what each term measures
Added 5 November 2026 and revised the same day, before the held-out split was evaluated and
before any radiologist annotation was joined to a prediction. The revision used only the
annotation tables: 82% of development pairs have the same subtlety in both views and only 22
differ in BI-RADS, so a correlation with those gaps alone would have little power. These
checks are descriptive, they sit outside the confirmatory claims, and they are reported
whatever they show. Each is read on the development outer folds and then repeated on the
held-out split in the locked evaluation.

Two ratios separate what the raw terms mix, because U_amb and U_evid both shrink as p leaves
0.5: the evidential share U_evid / (U_amb + U_evid), and the view share U_vd / H(p).

Radiologist annotations (`evaluation/anchor_p2.py`, `data/export_lesion_meta.py`). CBIS-DDSM
records subtlety, BI-RADS, shape and margins for each view of a lesion. Ratings of 0 are
treated as missing. Per case, mean across seeds. Whole patients are permuted, 9,999 times,
with Holm correction across the four tests of one arm.
1. Described differently: the CC and MLO records differ in any of the four fields (179 of 528
   development pairs, 52 of 154 held-out pairs). Effect: probability that such a case has the
   higher U_vd. Expected above 0.5.
2. U_vd against the subtlety difference between views. Expected positive.
3. H(p) against the mean subtlety, where a higher rating means a more obvious lesion.
   Expected negative.
4. Evidential share against the mean subtlety. Expected negative, since a subtle lesion should
   give less evidence.
Also reported without a test of its own: mean U_vd and error rate for subtlety differences of
0, 1, and 2 or more, and the contrast for BI-RADS differences, which rests on few cases.

Development reading of 5 November 2026, reported as read. On the outer folds, H(p) and the
evidential share fell as lesions became more obvious (Spearman -0.24 and -0.25 for DenseNet-121,
-0.21 and -0.19 for ConvNeXt-T, Holm p 0.0004). U_vd did not follow the annotations: test 1 gave
0.51 and 0.53 (Holm p 1.0 and 0.56), and test 2 gave 0.01 and 0.02. All four tests stay as
written for the held-out split. Added after that reading, and exploratory: among pairs with
different subtlety ratings, does the more obvious view carry more evidence? Development
difference in S: +0.12 (p 0.15) and +0.55 (p 0.03). It is tested once on the held-out split,
outside the Holm family, where only about 31 pairs qualify.

Controlled changes (`evaluation/term_response_p2.py`). Each change is compared with the clean
output of the same case, recomputed by the same script and model, with a patient-clustered
bootstrap.
- swap_opposite: the MLO view is replaced by the MLO view of another patient with the opposite
  label, three random draws. swap_same: the same with a patient of the same label.
- blur at sigma 1.5, 3 and 6 pixels at 224 by 224, on both views.
- occlusion of the central 25% and 50% of the patch side on both views, where the lesion sits.
Expected: U_vd and the view share rise more under swap_opposite than under swap_same, and the
evidential share rises with the severity of blur and occlusion.

The design may be corrected once on a development pilot: the fold checkpoints of `cued_net`
on the outer folds. Any correction is committed before the held-out run and is disclosed.

Development pilot of 6 November 2026, reported as read (25 fold checkpoints of `cued_net`,
2,640 case outputs per condition, mean over seeds per case, 10,000 patient-clustered bootstrap
draws). The recomputed clean outputs equal the stored outer-fold outputs (largest absolute
difference 0, AUC 0.859). swap_opposite raised U_vd by 0.046 bits [0.041, 0.052] from a clean
mean of 0.027, and swap_same by 0.013 [0.010, 0.017]. Neither swap changed U_amb or U_evid
(all four intervals include 0). The rise under swap_opposite held in every fold and every draw
(0.026 to 0.070). Blur and occlusion raised U_amb (+0.017 to +0.053) and U_evid (+0.004 to
+0.012), while U_vd stayed flat or fell (-0.009 to +0.000). The evidential share rose with
occlusion (+0.001, +0.003) but not monotonically with blur (+0.001, +0.001, -0.001). The
replacement view of a swap is drawn by label, so the AUC under swap_same rises (0.90 to 0.91) and the
AUC under either swap is not reported as performance. No design correction follows: the
held-out run uses the design as written above.
On the held-out split the changes run once, after the locked evaluation, on the reference
model of each arm.

Case figure, rule fixed 5 November 2026 (`cued_net`, outer folds, per case, mean across seeds).
Strata by terciles of U_amb + U_evid and of U_vd: low and low, correct; high and low; low and
high, correct; low and high, wrong. A case counts as wrong when most seeds err. Each stratum
gives the case whose H(p) is closest to the stratum median. No case is chosen by eye.
