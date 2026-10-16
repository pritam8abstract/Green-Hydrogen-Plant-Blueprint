"""Demand forecasting for green hydrogen in India: a Bass diffusion model with uncertain parameters.

Why Bass and not ARIMA? There is almost no history to fit: green hydrogen demand in India is
a few thousand tonnes today. A time-series model needs a past; a diffusion model needs only a
market size (m), an 'innovation' push (p: mandates, auctions, SIGHT) and an 'imitation' pull
(q: buyers copying buyers once costs and contracts are proven). It is the standard tool for
forecasting a new product before it has sales (Bass, 1969), and every parameter is drawn from a
range so the output is a fan, not a line.

Structure. Total hydrogen demand by segment grows at an uncertain rate; the green SHARE of each
existing segment follows a Bass curve that starts in 2025; new uses (steel DRI, shipping fuels,
export ammonia) are a separate Bass curve in absolute tonnes.

Base-year volumes are scaled from IGEF/GIZ (2021) for 2018-19 (refining 2.6 Mt; total domestic
5.6 Mt). Every range below is ESTIMATED and flagged low-confidence in Dossier 3; the 2030 Mission
target (5 Mt/yr) is PROJECTED policy, used only as a yardstick, never as an input.
"""
from __future__ import annotations
import numpy as np
from . import params as P

YEARS = np.arange(2025, 2041)
# segment: (2025 total demand Mt/yr, growth range %/yr, ceiling green share range)
SEGMENTS = {
    "Refining":   (2.9, (0.025, 0.040), (0.10, 0.45)),   # GHCO-style mandate would set the ceiling
    "Fertiliser": (3.3, (0.010, 0.025), (0.15, 0.55)),   # ammonia for urea/DAP; import substitution
    "Methanol & other": (0.5, (0.020, 0.040), (0.05, 0.30)),
}
NEW_USES_M = (1.0, 6.0)          # Mt/yr ceiling for steel DRI, bunkering and export ammonia by ~2045
P_RANGE = (0.003, 0.015)         # Bass innovation coefficient (policy push)
Q_RANGE = (0.20, 0.55)           # Bass imitation coefficient (peer adoption)


def bass_cdf(t, p, q):
    """Cumulative adoption fraction F(t) of the Bass model, t in years since launch (t >= 0)."""
    t = np.maximum(t, 0.0)
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / p) * e)


def simulate(n=20_000, rng=None, years=YEARS, launch=2025):
    rng = rng or np.random.default_rng(P.SEED)
    t = (years - launch)[None, :].astype(float)
    out = {}
    total = np.zeros((n, len(years)))
    for seg, (d0, (g_lo, g_hi), (m_lo, m_hi)) in SEGMENTS.items():
        g = rng.uniform(g_lo, g_hi, (n, 1)); m = rng.uniform(m_lo, m_hi, (n, 1))
        p = rng.uniform(*P_RANGE, (n, 1)); q = rng.uniform(*Q_RANGE, (n, 1))
        seg_total = d0 * (1 + g) ** t
        green = seg_total * m * bass_cdf(t, p, q)
        out[seg] = green; total += green
    m = rng.uniform(*NEW_USES_M, (n, 1)); p = rng.uniform(*P_RANGE, (n, 1)); q = rng.uniform(*Q_RANGE, (n, 1))
    out["New uses & export"] = m * bass_cdf(t - 2, p, q)          # new uses lag existing ones by ~2 years
    total += out["New uses & export"]
    out["Total"] = total
    return out


def summary(sim, years=YEARS, plant_t=9_000):
    i30, i35 = list(years).index(2030), list(years).index(2035)
    pct = lambda a, i: {f"p{k}": float(np.percentile(a[:, i], k)) for k in (10, 50, 90)}
    tot = sim["Total"]
    ref = sim["Refining"]
    return dict(
        total_2030=pct(tot, i30), total_2035=pct(tot, i35),
        refining_2030=pct(ref, i30), refining_2035=pct(ref, i35),
        p_meets_mission_2030=float((tot[:, i30] >= 5.0).mean()),
        plant_share_refining_2030_p50=float(plant_t / 1e6 / np.percentile(ref[:, i30], 50)),
        by_segment_2035_p50={k: float(np.percentile(v[:, i35], 50)) for k, v in sim.items() if k != "Total"},
        fan={k: [float(np.percentile(tot[:, j], k)) for j in range(len(years))] for k in (10, 25, 50, 75, 90)},
        years=[int(y) for y in years])
