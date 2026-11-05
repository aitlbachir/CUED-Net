"""Exact additive decomposition of the fused predictive entropy, in bits.

For views v with weights w_v, Dirichlet parameters alpha_v = S_v p_v and fused
probability p = sum_v w_v p_v:

    H(p) = sum_v w_v E[H(theta_v)]                     within-view ambiguity
         + sum_v w_v (H(p_v) - E[H(theta_v)])          evidential, Dirichlet mutual information
         + (H(p) - sum_v w_v H(p_v))                   view discordance, weighted Jensen-Shannon

The three terms are non-negative and sum to H(p). All come from one forward pass.
The evidential term reflects lack of evidence and vanishes as the Dirichlet strength grows.
"""

import numpy as np
from scipy.special import digamma

LOG2 = np.log(2.0)
EPS = 1e-7


def entropy(p):
    """Binary entropy in bits."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / LOG2


def expected_entropy(p, S):
    """E[H(theta)] under Dir(S p, S (1 - p)), in bits."""
    p = np.clip(p, EPS, 1 - EPS)
    a, b = S * p, S * (1 - p)
    out = -(p * (digamma(a + 1) - digamma(S + 1)) + (1 - p) * (digamma(b + 1) - digamma(S + 1)))
    return out / LOG2


def decompose(p_cc, p_mlo, s_cc, s_mlo, w_cc, w_mlo):
    """Return the ambiguity, evidential, view discordance and total terms, in bits."""
    total_w = w_cc + w_mlo
    w_cc, w_mlo = w_cc / total_w, w_mlo / total_w
    p = w_cc * p_cc + w_mlo * p_mlo
    amb = w_cc * expected_entropy(p_cc, s_cc) + w_mlo * expected_entropy(p_mlo, s_mlo)
    within = w_cc * entropy(p_cc) + w_mlo * entropy(p_mlo)
    total = entropy(p)
    return dict(u_amb=amb, u_ev=within - amb, u_vd=total - within, u_total=total)


def add_decomposition(df):
    """Add the terms to a per-case table with dual-view evidential columns."""
    if "s_cc" in df:
        for k, v in decompose(df.p_cc.values, df.p_mlo.values, df.s_cc.values, df.s_mlo.values,
                              df.w_cc.values, df.w_mlo.values).items():
            df[k] = v
        df["u_within"] = df.u_amb + df.u_ev
    return df
