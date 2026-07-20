# CUED-Net

### Cross-view Uncertainty-guided Evidential Dual-encoder Network for Breast Mass Classification in Mammography

Official PyTorch implementation of **CUED-Net**, a dual-encoder evidential model
that classifies breast masses from paired craniocaudal (CC) and mediolateral-oblique
(MLO) mammographic views while producing a decomposed, interpretable estimate of
predictive uncertainty.

> **Manuscript:** *IEEE Journal of Biomedical and Health Informatics*,
> ID `JBHI-00149-2026` (under review).

---

## Overview

CUED-Net encodes each view with an independent DenseNet-121 backbone terminated by a
Dirichlet **evidential** head, fuses the two views with an uncertainty-weighted
scheme, and reports uncertainty as a decomposition rather than a single scalar:

| Component | Symbol | Source | Availability |
| --- | --- | --- | --- |
| Evidential | `u_evid` | within-model (Dirichlet vacuity) | single forward pass |
| View discordance | `u_disc` | between-view (CC vs. MLO) | single forward pass |
| Ensemble | `u_ens` | between-model (seed variance) | ensemble only |

At inference the model operates in **two tiers**:

- **Single pass** — combined uncertainty `u_comb = u_evid + 0.5 · u_disc`
  (two sources, no ensemble required).
- **Ensemble** — adds a third source `u_ens = Var_m[p_mal]`, the cross-member
  variance of the malignant-class probability, giving a total that weights the three
  sources.

The view-discordance term is defined on the view probability vectors as

```
u_disc = σ( ½ [ KL(p_CC ‖ p_MLO) + KL(p_MLO ‖ p_CC) ] ) ∈ (0.5, 1)
```

i.e. the sigmoid of the symmetric KL divergence between the two views' predictive
distributions. It is near-orthogonal to the other two signals (|r| ≤ 0.07), a
structural property of the dual-view architecture.

---

## Key results (CBIS-DDSM, pooled 5×5 cross-validation)

| Metric | CUED-Net |
| --- | --- |
| AUROC | 0.877 |
| F1 | 0.834 |
| Average precision | 0.852 |

CUED-Net matches the discrimination of sampling- and ensemble-based uncertainty
baselines (Deep-Ensemble, TMC) **at a single forward pass**, and its uncertainty
supports selective prediction that is statistically indistinguishable from the
strongest baselines on AURC — while uniquely decomposing total uncertainty into
evidential, ensemble, and view-discordance components. Full comparisons, calibration,
backbone ablation, learning curve, Grad-CAM case studies, and CMMD external
validation are reported in the paper.

*Note on the View Discordance Loss (VDL):* an auxiliary VDL training term was
evaluated during development and **removed from the final model** (`λ_vdl = 0`) after
cross-validation showed no significant benefit on discrimination, calibration, or
high-discordance recall. `u_disc` is used at **inference** as an uncertainty signal;
it is not a training loss in the released configuration.

---

## Repository structure

```
CUED-Net/
├── models/                     Model definitions
│   ├── cued_net.py               core dual-encoder evidential network + ensemble
│   ├── view_encoder_backbone.py  backbone-swappable encoder (Table III ablation)
│   ├── baseline_models.py        MC-Dropout, Deep-Ensemble baselines
│   └── baseline_models_edl_tmc.py  TMC and single-view-EDL baselines
├── data/                       Data pipeline
│   ├── datasets.py               CBIS-DDSM CC+MLO pair dataset
│   ├── cv_dataloaders.py         cross-validation dataloaders
│   ├── cmmd_pair_dataset.py      CMMD external-validation dataset
│   ├── build_cv_folds.py         StratifiedGroupKFold split generation
│   ├── build_patchs.py           lesion-patch extraction
│   ├── crop_cmmd_breast.py       CMMD breast-region cropping
│   ├── build_cmmd_manifest.py       CMMD manifest (Kaggle-mirror layout)
│   ├── build_cmmd_manifest_nbia.py  CMMD manifest (raw TCIA/NBIA layout)
│   └── cv_folds.json             exact split used in the paper (for verification)
├── training/                   Training entry points
│   ├── train_cv.py               ★ paper protocol: 5-fold × 5-seed CV
│   ├── train_cued_net.py         single-run demo
│   ├── train_cv_baselines.py     MC-Dropout / Deep-Ensemble CV
│   └── train_cv_baselines_edl_tmc.py  TMC / single-view-EDL CV
├── ablations/                  Ablation harnesses
│   ├── run_ablations.py          component ablation (λ_vdl=0, w/o consistency)
│   ├── run_backbone_ablation.py  backbone comparison (Table III)
│   ├── ablation_analysis.py      post-hoc analysis of ablation outputs
│   └── run_learning_curve.py     learning curve (Figure 3)
├── evaluation/                 Metrics & statistics
│   ├── stats_tests2.py           DeLong, McNemar, bootstrap CIs (canonical)
│   ├── delong_convnext_vs_densenet.py  backbone ΔAUC significance test
│   ├── selective_prediction_2.py  AURC / E-AURC / selective-prediction analysis
│   ├── dump_cued_decomposed.py   dump per-sample decomposed uncertainty
│   ├── calibration_cbis.py       ECE / Brier / NLL
│   ├── temperature_scaling.py    post-hoc temperature scaling
│   ├── uncertainty_covariance.py  cross-signal Pearson correlations
│   ├── cost_table.py             params / GMACs / latency / memory
│   ├── roc_pr_curves.py          ROC and PR curves
│   ├── seed_variance.py          per-seed F1 dispersion
│   └── verify_mcnemar_roc.py     McNemar / ROC reconciliation checks
├── cmmd_external_validation/   CMMD transfer
│   ├── finetune_cmmd.py
│   ├── evaluate_cmmd.py
│   └── calibrate_cmmd.py
├── visualization/              Figure generation
│   ├── gradcam_cases.py          high-u_evid and high-u_disc Grad-CAM cases
│   ├── gradcam_uens_case.py      high-u_ens ensemble Grad-CAM case
│   ├── plot_figure3_learning_curve.py
│   ├── plot_results_figure.py
│   └── table_numbers.py
├── tests/
│   └── test_no_patient_leakage.py  asserts no patient crosses train/val folds
├── results/                    Curated locked JSON summaries (small)
├── figures/                    Example output figures
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/aitlbachir/CUED-Net.git
cd CUED-Net

# PyTorch with CUDA 12.4 (adjust the CUDA suffix to your system)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verified on Python 3.11, CUDA 12.4, a single NVIDIA RTX 3090.

---

## Data

The paper uses two public datasets, neither of which is redistributed here:

- **CBIS-DDSM** (training and cross-validation) — the Curated Breast Imaging Subset of
  DDSM. After acquiring it, generate the CC+MLO pair manifest and folds:

  ```bash
  python data/build_patchs.py        # extract lesion patches
  python data/build_cv_folds.py      # StratifiedGroupKFold, grouped by patient
  ```

  The exact split used in the paper is provided as `data/cv_folds.json` so results can
  be reproduced against the identical partition. `tests/test_no_patient_leakage.py`
  verifies that no patient appears in more than one fold.

- **CMMD** (external validation) — the Chinese Mammography Database. Build the pair
  manifest with whichever discovery path matches your download:

  ```bash
  # Kaggle-mirror directory layout (CMMD/D1-XXXX/...)
  python data/build_cmmd_manifest.py

  # raw TCIA/NBIA layout (flat SeriesInstanceUID directories)
  python data/build_cmmd_manifest_nbia.py
  ```

Expected pair schema consumed by the datasets:

```
{ "patient_id", "cc_path", "mlo_path", "label" }   # label: benign=0, malignant=1
```

---

## Training

Reproduce the paper's cross-validated model (5 folds × 5 seeds):

```bash
python training/train_cv.py --data_root /path/to/cbis-ddsm --folds data/cv_folds.json
```

Quick single-run demo (one seed, no CV):

```bash
python training/train_cued_net.py --data_root /path/to/cbis-ddsm
```

Baselines:

```bash
python training/train_cv_baselines.py          # MC-Dropout, Deep-Ensemble
python training/train_cv_baselines_edl_tmc.py  # TMC, single-view-EDL
```

---

## Evaluation

```bash
# Statistical comparisons (DeLong, McNemar, bootstrap CIs)
python evaluation/stats_tests2.py --preds_dir <prediction_csv_dir>

# Selective prediction (AURC / E-AURC, patient-clustered bootstrap)
python evaluation/selective_prediction_2.py --out results/selective

# Calibration and temperature scaling
python evaluation/calibration_cbis.py
python evaluation/temperature_scaling.py

# Computational cost (params, GMACs, latency, memory)
python evaluation/cost_table.py
```

Prediction-CSV schema (produced by the training scripts):

```
model, seed, fold, patient_id, label, prob_malignant, predicted, uncertainty
```

CUED-Net additionally emits `uncertainty_evidential`, `uncertainty_discordance`,
`uncertainty_ensemble`, and `uncertainty_total`.

---

## Ablations and figures

```bash
python ablations/run_ablations.py            # component ablation
python ablations/run_backbone_ablation.py    # Table III (DenseNet vs. alternatives)
python ablations/run_learning_curve.py       # Figure 3 data
python visualization/plot_results_figure.py  # main results figure
python visualization/gradcam_cases.py        # Grad-CAM case studies
```

---

## Pretrained weights

Trained checkpoints are **not** included in this repository. A representative
5-fold checkpoint set will be published as a GitHub Release after peer review
concludes. All reported metrics can be reproduced from scratch using the training
scripts and the provided `cv_folds.json`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{aitlbachir2026cuednet,
  title   = {CUED-Net: Cross-view Uncertainty-guided Evidential Dual-encoder
             Network for Breast Mass Classification in Mammography},
  author  = {Ait Lbachir, Ilhame and Daoudi, Imane and Nassih, Rym and Samiry, Imane},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review, manuscript ID JBHI-00149-2026}
}
```

---

## Acknowledgments

Built with PyTorch and torchvision. Evidential learning follows the Dirichlet
formulation of Sensoy et al. (NeurIPS 2018); the trusted multi-view baseline follows
Han et al. (ICLR 2021). Grad-CAM visualizations follow Selvaraju et al. (ICCV 2017).

## License

Released under the MIT License. See [LICENSE](LICENSE).
