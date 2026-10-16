"""Levelised cost of hydrogen (LCOH) and ammonia (LCOA), deterministic and Monte Carlo.

Design choices (also argued in Dossier 1):
* Capital is annualised with a capital-recovery factor, like an EMI. Screening stage, so
  no debt waterfall here; that lives in finance.py.
* Stack replacement is charged per operating hour, so short-lived stacks pay for their
  weakness in every kilogram rather than in one lumpy year.
* Flexibility becomes money through the 'capture' factor: the share of renewable-available
  hours a stack can actually use.
"""
from __future__ import annotations
import numpy as np
from . import params as P


def crf(r, n: int = P.LIFE):
    """Capital recovery factor: the annual payment that repays 1 unit over n years at r."""
    r = np.asarray(r, dtype=float)
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def lcoh(capex, eff, life, repl, om, hours, power, wacc, heat=0.0, water=P.WATER_USD_KG):
    """Return (total $/kg, components dict). All arguments broadcast as numpy arrays."""
    kg_per_kw = hours / eff
    comp = {
        "Capital": capex * crf(wacc) / kg_per_kw,
        "O&M": capex * om / kg_per_kw,
        "Stack replacement": repl * capex * hours / life / kg_per_kw,
        "Electricity": eff * power / 1000.0,
        "Heat + water": heat + water + 0 * hours,
    }
    return sum(comp.values()), comp


def _draw(spec, n, rng):
    lo, mo, hi = spec
    return rng.triangular(lo, mo, hi, n) if hi > lo else np.full(n, float(mo))


def simulate(stack_key: str, n: int = 100_000, rng=None, deterministic=False, overrides=None):
    """Monte Carlo (or base-case) LCOH for one stack. Returns (lcoh array, comps, hours)."""
    rng = rng or np.random.default_rng(P.SEED)
    s = P.STACKS[stack_key]
    o = overrides or {}
    pick = (lambda spec: np.array([spec[1]], float)) if deterministic else (lambda spec: _draw(spec, n, rng))
    capex = pick(o.get("capex", s.capex)); eff = pick(o.get("eff", s.eff)); life = pick(o.get("life", s.life))
    wacc = pick(o.get("wacc", P.WACC))
    if s.cf:
        hours = pick(s.cf) * 8760; power = pick(o.get("power", s.power))
        heat = s.heat_kwh * s.heat_price / 1000
    else:
        hours = pick(o.get("re_hours", P.RE_HOURS)) * s.capture * 8760
        power = pick(o.get("power", P.POWER_USD_MWH)); heat = 0.0
    total, comp = lcoh(capex, eff, life, s.repl, s.om, hours, power, wacc, heat)
    return total, comp, hours


def lcoa_from_lcoh(lcoh_usd_kg, loop_usd_t):
    """Ammonia cost ($/t NH3) = hydrogen in a tonne of ammonia + synthesis/ASU/storage loop."""
    return lcoh_usd_kg * P.H2_PER_T_NH3 * 1000 + loop_usd_t


def summary(x):
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(np.mean(x))}
