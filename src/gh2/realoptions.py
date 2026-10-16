"""Two real options embedded in the chosen design.

1. EXPANSION with a policy cliff. India waives inter-state transmission (ISTS) charges for
   25 years for green-hydrogen plants commissioned by 31 Dec 2030; later plants pay graded
   charges. An option to add a second 100 MW phase is therefore an American option whose
   value drops discretely after the cut-off. Valued by least-squares Monte Carlo
   (Longstaff-Schwartz) on a GBM project value, with the strike falling on a learning curve.

2. AEM SWITCH. The 20 MW PEM slice is the 'option bay'. At its first stack replacement we may
   install AEM instead of PEM, if AEM has become cheaper and durable enough. Valued as the
   expected positive saving under uncertain learning rates, deployment growth and durability.

Every parameter here is ESTIMATED and flagged low-confidence in the dossier; the point is
the structure (and the sign and rough size of the effects), not a precise number.
"""
from __future__ import annotations
import numpy as np
from . import params as P


# ---------------------------------------------------------------- 1. expansion option
def expansion_lsmc(V0=1300.0, K0=1250.0, sigma=0.30, r=0.07, delta=0.04, learn=0.04,
                   years=range(2028, 2037), lag=2, cliff=0.12, cutoff=2030, n=40_000, rng=None):
    """Value (Rs crore at 2027) of the right to build phase 2 in any year in `years`.
    V0: PV of phase-2 cash flows if built now; K0: phase-2 capex today.
    cliff: share of V lost if commissioning (decision + lag) falls after `cutoff`."""
    rng = rng or np.random.default_rng(P.SEED)
    yrs = list(years); T = len(yrs)
    t_idx = np.array([y - 2027 for y in yrs], float)
    z = rng.standard_normal((n, T))
    dt = np.diff(np.concatenate([[0.0], t_idx]))
    logV = np.log(V0) + np.cumsum((r - delta - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    V = np.exp(logV)
    K = K0 * (1 - learn) ** t_idx
    hit = np.array([(y + lag) > cutoff for y in yrs])
    payoff = np.maximum(V * (1 - cliff * hit) - K, 0)
    disc = np.exp(-r * t_idx)
    cash = payoff[:, -1] * disc[-1]; ex_year = np.where(payoff[:, -1] > 0, yrs[-1], 0)
    for j in range(T - 2, -1, -1):
        itm = payoff[:, j] > 0
        if itm.sum() > 50:
            X = V[itm, j]; Y = cash[itm] / disc[j]
            A = np.vstack([np.ones_like(X), X, X ** 2]).T
            beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
            cont = A @ beta
            ex = payoff[itm, j] > cont
            idx = np.where(itm)[0][ex]
            cash[idx] = payoff[idx, j] * disc[j]; ex_year[idx] = yrs[j]
    value = float(cash.mean())
    now_npv = float(V0 * (1 - cliff * ((2027 + lag) > cutoff)) - K0)
    by_year = {y: float((ex_year == y).mean()) for y in yrs}
    return dict(value=value, npv_now=now_npv, exercise_by_year=by_year, never=float((ex_year == 0).mean()))


# ---------------------------------------------------------------- 2. AEM switch option
def aem_switch(mw=20.0, year_swap=2041, pem_repl_usd_kw=(1800 * 0.35), n=100_000, rng=None, r=0.08):
    """Expected value ($m, discounted to 2030) of keeping the bay AEM-ready.
    AEM installed cost follows Wright's law from $1,700/kW (2026) with an uncertain learning
    rate (8-20%) and 3-8 doublings of cumulative capacity by the swap year; AEM must also reach
    >= 40,000 h field durability (probability rising with deployment)."""
    rng = rng or np.random.default_rng(P.SEED)
    lr = rng.uniform(0.08, 0.20, n)
    doublings = rng.uniform(3, 8, n)
    aem_kw = 1700 * (1 - lr) ** doublings
    p_durable = np.clip(0.15 + 0.08 * doublings, 0, 0.9)        # ESTIMATED: more deployment, more proof
    durable = rng.random(n) < p_durable
    # a full AEM stack-plus-integration in the existing bay costs ~45% of a new AEM system
    aem_cost = 0.45 * aem_kw
    saving_kw = np.where(durable, pem_repl_usd_kw - aem_cost, -np.inf)
    exercise = saving_kw > 0
    payoff_musd = np.where(exercise, saving_kw, 0) * mw * 1000 / 1e6
    pv = payoff_musd * np.exp(-r * (year_swap - 2030))
    return dict(value_musd=float(pv.mean()), p_exercise=float(exercise.mean()), p_durable=float(durable.mean()),
                aem_kw_p50=float(np.median(aem_kw)), readiness_cost_musd=(0.3, 0.6))
