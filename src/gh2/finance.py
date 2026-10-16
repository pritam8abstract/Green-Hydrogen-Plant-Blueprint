"""Project-finance model for the chosen plant: 80 MW alkaline + 20 MW PEM, selling hydrogen
to a refinery under a 25-year fixed-price contract (the structure IOCL and Numaligarh used).

Why a cash-flow waterfall and not just LCOH: lenders do not lend against a levelised cost.
They lend against the worst year's cash cover (DSCR) and the cover over the loan's life
(LLCR). A plant can have an acceptable LCOH and still fail a covenant in the year its PEM
stacks are replaced. This module makes those years visible, and vectorises everything over
Monte Carlo draws so the same code gives the base case and the risk distribution.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from . import params as P

YEARS = P.LIFE
AUX_KWH_KG = 2.0          # compression, purification, cooling; ESTIMATED
HANDLING_OM = 0.025       # O&M share of the hydrogen-handling capex


def _tri(spec, n, rng):
    lo, mo, hi = spec
    return rng.triangular(lo, mo, hi, n) if hi > lo else np.full(n, float(mo))


def draw_inputs(n=1, rng=None, deterministic=False, **override):
    rng = rng or np.random.default_rng(P.SEED)
    pick = (lambda s: np.full(n, float(s[1]))) if deterministic else (lambda s: _tri(s, n, rng))
    pr = P.PROJECT
    x = dict(
        capex_kw=pick(pr.capex_electrolysis_usd_kw),
        handling=pick(pr.capex_h2_handling_musd),
        power=pick(P.POWER_USD_MWH),
        re_hours=pick(P.RE_HOURS),
        eff=pick(P.STACKS["HYBRID"].eff),
        life_alk=pick(P.STACKS["ALK"].life),
        life_pem=pick(P.STACKS["PEM"].life),
        debt_rate=pick((0.0825, pr.debt_rate, 0.10)),
        capex_overrun=np.ones(n) if deterministic else rng.lognormal(mean=0.05, sigma=0.12, size=n),
    )
    for k, v in override.items():
        x[k] = np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, float)
    return x


def irr_vec(cf):
    """IRR for each row of a cash-flow matrix (sims x periods), by vectorised bisection."""
    t = np.arange(cf.shape[1])
    lo, hi = np.full(cf.shape[0], -0.9), np.full(cf.shape[0], 1.0)
    npv = lambda r: (cf / (1 + r[:, None]) ** t).sum(axis=1)
    f_lo = npv(lo)
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(same, mid, lo); f_lo = np.where(same, f_mid, f_lo)
        hi = np.where(same, hi, mid)
    out = (lo + hi) / 2
    out[np.sign(npv(np.full_like(out, -0.9))) == np.sign(npv(np.full_like(out, 1.0)))] = np.nan
    return out


def model(tariff_inr_kg, x, sight=False):
    """Annual cash flows for each draw. Returns a dict of arrays (sims x years)."""
    pr = P.PROJECT
    n = len(x["capex_kw"])
    mw = pr.mw_alk + pr.mw_pem
    capex_usd = (mw * 1000 * x["capex_kw"] + x["handling"] * 1e6) * (1 + pr.owners_contingency) * x["capex_overrun"]
    capex = capex_usd * P.FX / 1e7                                    # Rs crore
    # construction: 40/60 spend, interest during construction capitalised on the debt share
    rate = x["debt_rate"]
    idc = pr.debt_share * capex * (0.4 * rate * 1.5 + 0.6 * rate * 0.5)
    debt = pr.debt_share * (capex + idc)
    equity = capex + idc - debt

    hours = x["re_hours"] * P.STACKS["HYBRID"].capture * 8760 * pr.availability
    yrs = np.arange(1, YEARS + 1)
    # stack degradation: kWh/kg creeps up until each stack family is replaced (sawtooth)
    swap_alk = np.maximum(1, np.floor(x["life_alk"] / hours)).astype(int)
    swap_pem = np.maximum(1, np.floor(x["life_pem"] / hours)).astype(int)
    age_alk = (yrs[None, :] - 1) % swap_alk[:, None]
    age_pem = (yrs[None, :] - 1) % swap_pem[:, None]
    share_alk = pr.mw_alk / mw
    eff_t = x["eff"][:, None] * (1 + pr.degradation_per_year * (share_alk * age_alk + (1 - share_alk) * age_pem))
    energy_mwh = mw * hours[:, None] * np.ones((n, YEARS))
    h2_t = energy_mwh * 1000 / eff_t / 1000                            # tonnes
    power_cost = (energy_mwh + h2_t * AUX_KWH_KG) * x["power"][:, None] * P.FX / 1e7
    revenue = h2_t * 1000 * tariff_inr_kg / 1e7
    if sight:
        pass  # SIGHT Mode-1 hydrogen incentive deliberately excluded: refinery tenders were unsubsidised
    elec_capex = mw * 1000 * x["capex_kw"] * P.FX / 1e7
    om = (P.STACKS["HYBRID"].om * elec_capex[:, None] + HANDLING_OM * x["handling"][:, None] * P.FX / 10) \
        * (1 + pr.om_escalation) ** (yrs[None, :] - 1)
    insurance = pr.insurance * capex[:, None] * np.ones((1, YEARS))
    water = h2_t * 1000 * P.WATER_USD_KG * P.FX / 1e7
    # stack replacements (capex events in the year after the swap is due)
    repl_alk = ((yrs[None, :] % swap_alk[:, None]) == 0) & (yrs[None, :] < YEARS)
    repl_pem = ((yrs[None, :] % swap_pem[:, None]) == 0) & (yrs[None, :] < YEARS)
    stack_capex = (repl_alk * 0.35 * elec_capex[:, None] * share_alk + repl_pem * 0.35 * elec_capex[:, None] * (1 - share_alk))
    # Lenders do not let a stack swap hit one year's cover: a major-maintenance reserve (MMRA)
    # is funded evenly from operating cash, so the cost is smoothed over each stack's life.
    mmra = (0.35 * elec_capex[:, None] * share_alk / swap_alk[:, None]
            + 0.35 * elec_capex[:, None] * (1 - share_alk) / swap_pem[:, None]) * np.ones((1, YEARS))
    ebitda = revenue - power_cost - om - insurance - water
    # debt: level annuity over the tenor
    r = rate[:, None]
    ann = debt[:, None] * r / (1 - (1 + r) ** -pr.debt_tenor)
    bal = np.zeros((n, YEARS)); interest = np.zeros((n, YEARS)); principal = np.zeros((n, YEARS))
    b = debt.copy()
    for t in range(YEARS):
        if t < pr.debt_tenor:
            interest[:, t] = b * rate; principal[:, t] = ann[:, 0] - interest[:, t]; b = b - principal[:, t]
        bal[:, t] = b
    dep = (capex + idc)[:, None] / YEARS * np.ones((1, YEARS))
    ebt = ebitda - dep - interest
    tax = np.zeros_like(ebt); carry = np.zeros(n)
    for t in range(YEARS):  # simple loss carry-forward
        taxable = ebt[:, t] - carry
        carry = np.where(taxable < 0, -taxable, 0.0)
        tax[:, t] = np.maximum(taxable, 0) * pr.tax_rate
    cfads = ebitda - tax - mmra
    ds = interest + principal
    dscr = np.where(ds > 0, cfads / np.where(ds > 0, ds, 1), np.nan)
    fcfe = cfads - ds
    fcff = ebitda - tax - mmra
    # cash-flow vectors with construction years in front (capex 40/60)
    cons = -np.stack([0.4 * (capex + idc), 0.6 * (capex + idc)], axis=1)
    eq_cons = -np.stack([0.4 * equity, 0.6 * equity], axis=1)
    disc = (1 + rate[:, None]) ** -np.arange(1, pr.debt_tenor + 1)
    llcr = (cfads[:, :pr.debt_tenor] * disc).sum(axis=1) / debt
    return dict(capex=capex, idc=idc, debt=debt, equity=equity, h2_t=h2_t, revenue=revenue, power=power_cost,
                om=om, ebitda=ebitda, tax=tax, stack_capex=stack_capex, mmra=mmra, cfads=cfads, ds=ds, dscr=dscr,
                fcfe=fcfe, llcr=llcr, project_cf=np.hstack([cons, fcff]), equity_cf=np.hstack([eq_cons, fcfe]),
                swap_alk=swap_alk, swap_pem=swap_pem)


def metrics(m, wacc=0.10):
    t = np.arange(m["project_cf"].shape[1])
    npv = (m["project_cf"] / (1 + wacc) ** t).sum(axis=1)
    dscr = m["dscr"][:, :P.PROJECT.debt_tenor]
    return dict(project_irr=irr_vec(m["project_cf"]), equity_irr=irr_vec(m["equity_cf"]),
                npv_cr=npv, min_dscr=np.nanmin(dscr, axis=1), avg_dscr=np.nanmean(dscr, axis=1), llcr=m["llcr"])


def base_case(tariff):
    x = draw_inputs(1, deterministic=True)
    m = model(tariff, x)
    return m, {k: float(v[0]) for k, v in metrics(m).items()}


def required_tariff(min_dscr=1.30, equity_irr=0.14, **override):
    """Lowest fixed tariff (Rs/kg) that clears both the lender's and the sponsor's hurdle."""
    x = draw_inputs(1, deterministic=True, **override)
    f_d = lambda p: metrics(model(p, x))["min_dscr"][0] - min_dscr
    f_e = lambda p: np.nan_to_num(metrics(model(p, x))["equity_irr"][0], nan=-1.0) - equity_irr
    return dict(dscr=brentq(f_d, 150, 900), equity=brentq(f_e, 150, 650))


LEVERS = [  # (label, override added cumulatively, project tweak)
    ("Base design", {}, {}),
    ("Electrolyser system $1,250 → $850/kW", dict(capex_kw=850), {}),
    ("Delivered power $38 → $32/MWh", dict(power=32), {}),
    ("Debt 9.0% → 8.5%", dict(debt_rate=0.085), {}),
    ("Hydrogen handling $22m → $18m", dict(handling=18), {}),
    ("Debt tenor 15 → 18 years", {}, dict(debt_tenor=18)),
]


def lever_ladder(min_dscr=1.30, equity_irr=0.14):
    """Cumulative 'what you have to believe' ladder: required tariff after each design-to-cost lever."""
    import dataclasses
    base_project = P.PROJECT
    out, ov, tw = [], {}, {}
    try:
        for label, o, t in LEVERS:
            ov.update(o); tw.update(t)
            P.PROJECT = dataclasses.replace(base_project, **tw)
            r = required_tariff(min_dscr, equity_irr, **ov)
            out.append(dict(lever=label, dscr=r["dscr"], equity=r["equity"], binding=max(r["dscr"], r["equity"])))
    finally:
        P.PROJECT = base_project
    return out


def monte_carlo(tariff, n=20_000, rng=None):
    rng = rng or np.random.default_rng(P.SEED)
    x = draw_inputs(n, rng=rng)
    mt = metrics(model(tariff, x))
    return {k: v for k, v in mt.items()}
