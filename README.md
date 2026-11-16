# CUED-Net

### Cross-view Uncertainty-guided Evidential Dual-encoder Network for Breast Mass Classification in Mammography

Code of the manuscript JBHI-00149-2026, *IEEE Journal of Biomedical and Health Informatics* (second revision, under review).

| Branch | Content |
| --- | --- |
| `r2-protocol` (this branch) | Code, analysis plan and analysis outputs of the second revision |
| `main` | Code reviewed in the first revision (commit `95d92c3`) |

Scripts of the first revision that remain on this branch (`train_cv*.py`, `ablations/`, `stats_tests2.py`,
`selective_prediction_2.py`, `cmmd_external_validation/` and the R1 plotting scripts) do not run against the
corrected models and were not used for any result of the second revision.

---

## Method

CUED-Net encodes the craniocaudal (CC) and mediolateral-oblique (MLO) views of a breast with two encoders that do not
share weights. Each encoder ends in an evidential head that gives Dirichlet parameters, a view probability `p_v` and a
vacuity `u_v`. The fused probability is the evidence-weighted mean `p = sum_v w_v p_v`, with `w_v` proportional to `1 - u_v`.

The entropy of the fused prediction splits exactly into three terms, for every case and in one forward pass:

```
H(p) = U_amb + U_evid + U_vd
U_amb  = sum_v w_v E[H(theta_v)]                 within-view ambiguity
U_evid = sum_v w_v (H(p_v) - E[H(theta_v)])      evidential uncertainty
U_vd   = H(p) - sum_v w_v H(p_v)                 view discordance uncertainty (weighted Jensen-Shannon divergence)
```

with `theta_v ~ Dir(alpha_v)`. The identity holds by construction and has no fitted weights
(`evaluation/decomposition.py`). Referral ranks cases by `H(p)`.

---

## Protocol of the second revision

- Analysis plan: `ANALYSIS_PLAN.md`, frozen at commit `bb08f67`. The rule for the case figure was fixed in `6d3ca32`.
- CBIS-DDSM masses, CC and MLO pairs, patients grouped. Five frozen outer folds (`data/cv_folds.json`),
  15% of each training pool as inner validation, five seeds (42, 123, 456, 789, 2024).
- Each metric is computed per seed and then averaged over the seeds. Intervals come from a patient-clustered bootstrap
  with B = 10,000; comparisons are CUED-Net minus comparator, with Holm correction.
- Held-out split (CBIS-DDSM official test patients): models trained on the whole development pool (fold `-1`)
  and evaluated once. The record of that evaluation is `results/r2/test_evaluation.lock.json`.
- External cohort: CMMD, zero-shot and after adaptation of the heads, with adaptation and evaluation patients kept apart.
- All settings reported in the paper are in `training/p2_config.py`.

---

## Installation

```bash
git clone https://github.com/aitlbachir/CUED-Net.git
cd CUED-Net
git checkout r2-protocol
pip install torch==2.4.1 torchvision==0.19.1   # choose the wheel that matches your CUDA version
pip install -r requirements.txt
```

Training and inference ran on Kaggle (NVIDIA T4 GPUs). The analysis scripts were last rerun on CPU with Python 3.12.10,
numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1 and matplotlib 3.11.2 (see "Checking the outputs" below).
`evaluation/term_response_p2.py` imports torch even for `--summary`.

Tests (`tests/test_no_patient_leakage.py` needs the `data/` folder on `PYTHONPATH`):

```bash
PYTHONPATH=data pytest tests
```

---

## Pipeline

### 1. Data (CBIS-DDSM and CMMD are public and are not redistributed here)

```bash
python data/build_patchs.py --kaggle <CBIS-DDSM folder> --out <patches> --id_source study
python data/build_p2_manifest.py --patches <patches> --train_csv <mass_case_description_train_set.csv> \
    --test_csv <mass_case_description_test_set.csv> --frozen_folds data/cv_folds.json --out p2_manifest.json
python data/build_lesion_masks.py --kaggle <CBIS-DDSM folder> --patch_report <patch report csv> --out masks
python data/export_lesion_meta.py --train_csv <train csv> --test_csv <test csv> --out lesion_meta.csv
python data/build_cmmd_p2.py --dicom_root <CMMD DICOM folder> --clinical_xlsx <CMMD clinical xlsx> --patches <cmmd patches> --out cmmd_p2.json
```

`results/r2/multi_mass.json` lists the pairs of breasts with more than one mass (used for a sensitivity analysis).

### 2. Training (GPU)

`training/run_queue.py` runs `training/train_p2.py` for every fold and seed of a list of jobs `<method>:cv`
(outer folds 0 to 4) or `<method>:test` (fold -1, trained on the whole development pool):

```bash
python training/run_queue.py --jobs cued_net:cv,softmax_dual:cv,tmc:cv,sv_edl:cv,cross_view:cv,cued_net:test \
    --manifest p2_manifest.json --patches <patches> --out runs
```

Methods (`train_p2.py --method`): `cued_net` (DenseNet-121, evidence scale 1), `cued_convnext_k5` (ConvNeXt-T, scale 5;
the reference ConvNeXt-T model), `cued_convnext` and `cued_net_k5` (the other scale), comparators `softmax_dual`, `tmc`,
`sv_edl`, `cross_view` and their `_convnext` versions (the `softmax_dual` and `softmax_convnext`
runs also write the MC-Dropout predictions, and Deep-Ensemble averages their five seed models of a fold), the earlier loss `cued_r1_kl`, and the robustness arms `cued_resnet50`, `cued_effnet_b0`,
`cued_no_consistency`, `cued_vdl`. `evaluation/select_p2.py` applies the evidence-scale rule on inner validation only.

### 3. Held-out split, CMMD, controlled changes, saliency and cost (GPU)

```bash
python evaluation/evaluate_test.py --manifest p2_manifest.json --patches <patches> --runs runs --out test --plan_commit bb08f67
python evaluation/term_response_p2.py --manifest p2_manifest.json --patches <patches> --runs runs --out test --methods cued_net,cued_convnext_k5
python evaluation/cmmd_p2.py --manifest cmmd_p2.json --patches <cmmd patches> --runs runs --out cmmd --phase both
python evaluation/gradcam_p2.py --manifest p2_manifest.json --patches <patches> --masks masks --runs runs --out gradcam
python evaluation/cost_p2.py --out cost.json
```

`evaluate_test.py` refuses to run twice unless a reason is given; the held-out split was evaluated once.

### 4. Analyses and manuscript outputs (CPU)

`evaluation/build_r2_outputs.py` holds the command behind every analysis file, macro file, table, figure and the
graphical abstract. It works in a folder laid out as follows (prediction CSVs extracted from the run archives):

```
<root>/work/cv/runs/<method>/fold_<k>/seed_<s>/{inner_val,outer_val}.csv   outer folds, all methods
<root>/work/heldout/<method>/fold_-1/seed_<s>/{inner_val,test}.csv          held-out split
<root>/work/s6/test/...                                                      held-out controlled changes
<root>/work/s6/cmmd/{zero_shot,adapted}/...                                  CMMD
<root>/work/e10/runs/...                                                     robustness arms and cued_net
<root>/work/pilot/response_dev/...                                           development controlled changes
<root>/work/v4cost/cost.json, <root>/analysis/{lesion_meta.csv, multi_mass.json, case_keys.json, gradcam_summary.json}
<root>/manuscript/cases/*.png                                                patches of the four cases of Fig. S2
```

```bash
python evaluation/build_r2_outputs.py --root <root> --dry_run                  # print every command
python evaluation/build_r2_outputs.py --root <root> --stage summaries          # bootstrap analyses (slow)
python evaluation/build_r2_outputs.py --root <root> --stage derived manuscript # macros, tables, figures
python evaluation/build_r2_outputs.py --root <root> --out_dir <check> --stage derived manuscript   # rebuild elsewhere to compare
```

Every number in the manuscript comes from a macro written by `evaluation/make_numbers.py`, and every table from
`evaluation/make_tables.py`; no value is typed by hand.

---

## Analysis outputs (`results/r2/`)

| File | Script | Content |
| --- | --- | --- |
| `an_cv_final.json` | `p2_analysis.py --split outer_val --multi_mass` | Outer folds: every method, comparisons, referral, view-term tests, dependence, calibration, net benefit |
| `an_test.json` | `p2_analysis.py --split test` | Held-out split, same content |
| `an_cmmd_{zero_shot,adapted}_{eval,mass}.json`, `an_cmmd_zero_shot_all.json` | `p2_analysis.py --split cmmd_eval / cmmd_mass / cmmd` | CMMD evaluation patients, mass-containing subset, all patients |
| `e10_cv.json` | `e10_analysis.py` | Robustness arms (other encoders, no consistency term, earlier view discordance loss) |
| `term_response_dev_summary.json`, `term_response_test_summary.json` | `term_response_p2.py --summary` | Response of the three terms to controlled changes |
| `anchor_dev.json`, `anchor_test.json` | `anchor_p2.py` | Terms against the radiologists' descriptions of the two views |
| `anchor_attributes_dev.json`, `anchor_attributes_test.json` | `anchor_attributes_p2.py` | Exploratory: the same test for each attribute |
| `view_term_cued_convnext_cv.json`, `view_term_cued_convnext_test.json` | `view_term_p2.py` | Exploratory: view-term tests for ConvNeXt-T at evidence scale 1 |
| `reliability_cv.json`, `reliability_test.json` | `reliability_p2.py` | Reliability bins (same arrays as the ECE) |
| `per_seed_cv.json`, `per_seed_cv.csv` | `seed_table_p2.py` | Per-seed AUC, F1, AURC and fusion-weight percentiles |
| `gradcam_summary.json` | `gradcam_p2.py` | Grad-CAM saliency against the lesion outline |
| `cost.json` | `cost_p2.py` | Parameters, floating-point operations and latency |
| `case_keys.json` | rule of `ANALYSIS_PLAN.md` (commit `6d3ca32`) | The four cases of Fig. S2 |
| `multi_mass.json` | data preparation | Breasts with more than one mass |
| `test_evaluation.lock.json` | `evaluate_test.py` | Record of the single held-out evaluation |

The analyses marked exploratory were specified in writing before they were run and are labelled as exploratory in the
Supplement.

### Checking the outputs

On 16 November 2026, `build_r2_outputs.py --stage derived manuscript --out_dir <check>` reproduced byte for byte every
macro file, table and figure used by the manuscript and the Supplement, the reliability and per-seed files, and the
graphical abstract.

---

## Pretrained weights

Trained checkpoints are not included. All results can be reproduced from the training scripts, the frozen folds and
the settings in `training/p2_config.py`.

---

## Citation

```bibtex
@article{aitlbachir2026cuednet,
  title   = {CUED-Net: Cross-view Uncertainty-guided Evidential Dual-encoder
             Network for Breast Mass Classification in Mammography},
  author  = {Ait Lbachir, Ilhame and Daoudi, Imane and Nassih, Rym and Didast, Fatima Zahra},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review, manuscript ID JBHI-00149-2026}
}
```

## Acknowledgments

Evidential learning follows Sensoy et al. (NeurIPS 2018); the trusted multi-view comparator follows Han et al. (ICLR 2021);
the cross-view comparator follows van Tulder et al. (MICCAI 2021); Grad-CAM follows Selvaraju et al. (ICCV 2017).

## License

Released under the MIT License. See [LICENSE](LICENSE).