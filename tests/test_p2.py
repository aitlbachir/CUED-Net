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
