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
