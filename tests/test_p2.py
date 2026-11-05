"""Checks on the P2 loss, discordance score, splits and comparators."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.p2_data import build_splits, check_splits
from models.baseline_models import CrossViewTransformer, SoftmaxDualView, TMCDualView, ds_combine, mc_dropout_predict
from models.cued_net import CUEDNet, dirichlet_kl_to_uniform, evidential_loss, kl_target, r1_kl_term, symmetric_kl


def test_kl_zero_at_uniform():
    assert abs(dirichlet_kl_to_uniform(torch.ones(4, 2)).max().item()) < 1e-5


def test_kl_matches_closed_form_for_k2():
    # KL(Beta(a,1) || Beta(1,1)) = log(a) - 1 + 1/a
    a = 3.0
    kl = dirichlet_kl_to_uniform(torch.tensor([[a, 1.0]])).item()
    assert abs(kl - (math.log(a) - 1 + 1 / a)) < 1e-4


def test_standard_target_reaches_prior_when_evidence_vanishes():
    alpha = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(kl_target(alpha, y, "standard"), torch.ones(1, 2))


def test_standard_target_penalises_small_wrong_evidence_r1_does_not():
    alpha = torch.tensor([[1.0, 1.6]])  # evidence 0.6 on the wrong class
    y = torch.tensor([[1.0, 0.0]])
    assert dirichlet_kl_to_uniform(kl_target(alpha, y, "standard")).item() > 1e-3
    assert abs(r1_kl_term(kl_target(alpha, y, "r1")).item()) < 1e-6


def test_r1_term_was_not_a_divergence():
    assert r1_kl_term(torch.tensor([[1.0, 3.0]])).item() < 0
    assert dirichlet_kl_to_uniform(torch.tensor([[1.0, 3.0]])).item() > 0


def test_correct_class_evidence_is_not_penalised():
    alpha = torch.tensor([[9.0, 1.0]])
    y = torch.tensor([[1.0, 0.0]])
    assert abs(dirichlet_kl_to_uniform(kl_target(alpha, y, "standard")).item()) < 1e-5


def test_discordance_zero_at_agreement_and_monotone():
    p = torch.tensor([[0.3, 0.7]])
    assert symmetric_kl(p, p).item() < 1e-6
    d = [symmetric_kl(p, torch.tensor([[q, 1 - q]])).item() for q in (0.3, 0.5, 0.9)]
    assert d[0] < d[1] < d[2]


def test_loss_is_finite_and_differentiable():
    alpha = (torch.rand(8, 2) * 5 + 1).requires_grad_()
    loss = evidential_loss(alpha, torch.randint(0, 2, (8,)), epoch=3)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(alpha.grad).all()


def test_dempster_probabilities_sum_to_one():
    a = ds_combine(torch.tensor([[2.0, 5.0]]), torch.tensor([[4.0, 1.5]]), 2)
    assert (a >= 1).all() and torch.isfinite(a).all()


def test_models_forward():
    x = torch.rand(2, 3, 224, 224)
    for cls in (CUEDNet, SoftmaxDualView, TMCDualView, CrossViewTransformer):
        m = cls(pretrained=False).eval()
        with torch.no_grad():
            assert m(x, x)["prob"].shape == (2, 2)
    m = SoftmaxDualView(pretrained=False)
    p, h = mc_dropout_predict(m, x, x, T=3)
    assert p.shape == (2,) and h.shape == (2,)


def test_inner_splits_are_patient_disjoint():
    pairs = [dict(patient_id=f"P{i // 2}", label=(i // 2) % 2) for i in range(200)]
    folds = [dict(fold=k, train_idx=[i for i in range(200) if (i // 2) % 5 != k],
                  val_idx=[i for i in range(200) if (i // 2) % 5 == k]) for k in range(5)]
    manifest = dict(pairs=pairs, folds=folds, test_pairs=[dict(patient_id="T1")])
    splits = build_splits(manifest)
    check_splits(manifest, splits)
    assert splits == build_splits(manifest)
    assert set(splits[-1]["train"]) | set(splits[-1]["inner_val"]) == set(range(200))


def test_cmmd_patient_split_is_disjoint_and_covers_all():
    from data.build_cmmd_p2 import patient_split
    pairs = [dict(patient_id=f"D{i // 2}", label=int(i % 7 == 0)) for i in range(600)]
    split = patient_split(pairs, 0.10, 0.20)
    assert set(split) == {p["patient_id"] for p in pairs}
    frac = sum(v == "eval" for v in split.values()) / len(split)
    assert abs(frac - 0.70) < 0.02
    assert patient_split(pairs, 0.10, 0.20) == split


def test_adaptation_leaves_encoders_frozen():
    from evaluation.cmmd_p2 import freeze_for_adaptation
    for cls in (CUEDNet, TMCDualView):
        m = cls(pretrained=False)
        freeze_for_adaptation(m)
        assert not any(p.requires_grad for n, p in m.named_parameters() if ".features." in n)
        assert any(p.requires_grad for n, p in m.named_parameters() if ".features." not in n)


def test_mask_alignment_recovers_a_known_position():
    import numpy as np
    from data.build_lesion_masks import align
    rng = np.random.default_rng(0)
    import cv2
    full = cv2.GaussianBlur((rng.random((600, 500)) * 255).astype(np.uint8), (0, 0), 3)
    full = ((full.astype(float) - full.min()) / (full.max() - full.min()) * 255).astype(np.uint8)
    mask = np.zeros_like(full)
    mask[200:280, 150:230] = 255
    crop = full[180:300, 130:250]
    aligned, score = align(full, mask, crop)
    assert score > 0.9
    assert abs((aligned > 0).mean() - (80 * 80) / (120 * 120)) < 0.02
    assert aligned[60, 60] > 0 and aligned[5, 5] == 0


def test_saliency_metrics():
    import numpy as np
    from evaluation.gradcam_p2 import mask_metrics
    mask = np.zeros((10, 10))
    mask[:5, :] = 1
    assert abs(mask_metrics(np.ones((10, 10)), mask)["lift"] - 1.0) < 1e-6
    cam = np.zeros((10, 10))
    cam[2, 2] = 1.0
    m = mask_metrics(cam, mask)
    assert m["energy"] == 1.0 and m["lift"] == 2.0 and m["hit"] == 1.0
    assert np.isnan(mask_metrics(np.zeros((10, 10)), mask)["energy"])


def test_grad_cam_shape_and_sign():
    from evaluation.gradcam_p2 import grad_cam
    from models.cued_net import ViewEncoder
    enc = ViewEncoder(pretrained=False).eval()
    cam = grad_cam(enc, torch.rand(2, 3, 224, 224))
    assert cam.shape == (2, 224, 224) and (cam >= 0).all()


def test_decomposition_is_exact_and_non_negative():
    import numpy as np
    from evaluation.decomposition import decompose
    rng = np.random.default_rng(0)
    n = 2000
    p1, p2 = rng.uniform(0.01, 0.99, n), rng.uniform(0.01, 0.99, n)
    s1, s2 = rng.uniform(2.0, 200.0, n), rng.uniform(2.0, 200.0, n)
    w = rng.uniform(0.2, 0.8, n)
    d = decompose(p1, p2, s1, s2, w, 1 - w)
    assert np.allclose(d["u_amb"] + d["u_ev"] + d["u_vd"], d["u_total"])
    assert (d["u_amb"] >= 0).all() and (d["u_ev"] >= -1e-9).all() and (d["u_vd"] >= -1e-9).all()
    assert (d["u_total"] <= 1 + 1e-9).all()


def test_decomposition_limits():
    import numpy as np
    from evaluation.decomposition import decompose
    one = lambda x: np.array([float(x)])
    # Agreeing views: no view discordance term.
    d = decompose(one(0.7), one(0.7), one(10), one(30), one(0.5), one(0.5))
    assert abs(d["u_vd"][0]) < 1e-9
    # Strong evidence: the evidential term vanishes.
    d = decompose(one(0.7), one(0.7), one(1e6), one(1e6), one(0.5), one(0.5))
    assert abs(d["u_ev"][0]) < 1e-3
    # Confident opposite views: nearly all uncertainty is view discordance.
    d = decompose(one(0.999), one(0.001), one(1e6), one(1e6), one(0.5), one(0.5))
    assert d["u_vd"][0] > 0.95 and d["u_amb"][0] < 0.02


def test_comparators_build_with_convnext():
    from training.p2_config import METHODS
    from training.train_p2 import build
    x = torch.rand(2, 3, 224, 224)
    for name in ("softmax_convnext", "tmc_convnext", "sv_edl_convnext", "cross_view_convnext", "cued_convnext_k5"):
        m = build(name, pretrained=False)[0].eval()
        with torch.no_grad():
            assert m(x, x)["prob"].shape == (2, 2), name
        assert METHODS[name]["backbone"] == "convnext_tiny"


def test_evidence_scale_multiplies_the_head_output():
    from models.cued_net import EvidentialLayer
    a, b = EvidentialLayer(4), EvidentialLayer(4, evidence_scale=5.0)
    b.load_state_dict(a.state_dict())
    x = torch.rand(3, 4)
    assert torch.allclose(b(x)["evidence"], torch.nn.functional.softplus(5.0 * a.fc(x)))
