"""Capacity planning as a linear programme: how much solar, wind and hydrogen storage should
feed a 100 MW electrolyser that must deliver a flat refinery offtake every hour?

This is the operations-management heart of the project. The electrolyser is a capacity
constraint; hydrogen storage is a decoupling (buffer) inventory between a variable supply
(sun, wind) and a level demand (the refinery); grid power is the expensive emergency
supplier. The LP trades these off hour by hour for a full year.

Costs are annualised ($/MW-yr, $/t-yr) and ESTIMATED; profiles come from resource.py.
Min-load limits are ignored here (that needs integers); the hybrid stack's PEM slice is
what makes that simplification tolerable.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
from .lcoh import crf

COST = dict(
    solar_usd_mw_yr=600e3 * float(crf(0.10)) + 8e3,     # $600/kW installed + O&M, ESTIMATED
    wind_usd_mw_yr=1000e3 * float(crf(0.10)) + 20e3,    # $1,000/kW installed + O&M, ESTIMATED
    storage_usd_t_yr=500e3 * float(crf(0.10)),          # $500 per kg of compressed-gas capacity (Gemini: 500-800)
    grid_usd_mwh=58.0,                                  # firm RTC power, Rs5.25/kWh (pvmfdre2026)
)


def solve(solar_cf, wind_cf, electrolyser_mw=100.0, eff_kwh_kg=53.2, demand_t_yr=9000.0, cost=COST,
          hours=None):
    """Return a dict with optimal sizes and hourly dispatch. Variables per hour:
    solar_used, wind_used, grid, electrolyser_power, storage_level; plus S, W, Hcap."""
    T = len(solar_cf) if hours is None else hours
    s_cf, w_cf = np.asarray(solar_cf[:T]), np.asarray(wind_cf[:T])
    D = demand_t_yr * T / 8760 / T                      # t/h, flat
    k = 1.0 / eff_kwh_kg                                # t H2 per MWh
    nv = 5 * T + 3
    iS, iW, iH = 5 * T, 5 * T + 1, 5 * T + 2
    ps = lambda t: t; pw = lambda t: T + t; g = lambda t: 2 * T + t; e = lambda t: 3 * T + t; s = lambda t: 4 * T + t
    c = np.zeros(nv)
    c[2 * T:3 * T] = cost["grid_usd_mwh"] * 8760 / T   # scale to a full year if T < 8760
    c[iS] = cost["solar_usd_mw_yr"]; c[iW] = cost["wind_usd_mw_yr"]; c[iH] = cost["storage_usd_t_yr"]
    # inequalities A_ub x <= b_ub
    A = lil_matrix((3 * T, nv)); b = np.zeros(3 * T); r = 0
    for t in range(T):
        A[r, ps(t)] = 1; A[r, iS] = -s_cf[t]; r += 1      # solar used <= S * cf
        A[r, pw(t)] = 1; A[r, iW] = -w_cf[t]; r += 1      # wind used  <= W * cf
        A[r, s(t)] = 1; A[r, iH] = -1; r += 1             # storage <= capacity
    # equalities: power balance; hydrogen balance (cyclic)
    Aeq = lil_matrix((2 * T, nv)); beq = np.zeros(2 * T)
    for t in range(T):
        Aeq[t, ps(t)] = 1; Aeq[t, pw(t)] = 1; Aeq[t, g(t)] = 1; Aeq[t, e(t)] = -1
        q = T + t; prev = s(t - 1) if t > 0 else s(T - 1)
        Aeq[q, s(t)] = 1; Aeq[q, prev] = -1; Aeq[q, e(t)] = -k; beq[q] = -D
    bounds = [(0, None)] * nv
    for t in range(T):
        bounds[e(t)] = (0, electrolyser_mw)
    res = linprog(c, A_ub=csr_matrix(A), b_ub=b, A_eq=csr_matrix(Aeq), b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    S, W, H = x[iS], x[iW], x[iH]
    elec = x[3 * T:4 * T]; grid = x[2 * T:3 * T]
    avail = S * s_cf + W * w_cf
    used = x[:T] + x[T:2 * T]
    h2 = elec.sum() * k * 8760 / T
    return dict(solar_mw=S, wind_mw=W, storage_t=H, grid_share=grid.sum() / elec.sum(),
                electrolyser_cf=elec.mean() / electrolyser_mw, curtailment=1 - used.sum() / max(avail.sum(), 1e-9),
                re_to_electrolyser=(S + W) / electrolyser_mw, cost_usd_yr=res.fun,
                power_cost_usd_mwh=res.fun / (elec.sum() * 8760 / T), h2_t=h2,
                hourly=dict(solar=x[:T], wind=x[T:2 * T], grid=grid, elec=elec, storage=x[4 * T:5 * T]))
