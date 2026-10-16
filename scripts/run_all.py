"""Reproduce every number and chart in Dossier 3. Usage:  python scripts/run_all.py
Outputs: results/results.json, results/auctions.csv, figures/*.pdf"""
from __future__ import annotations
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from gh2 import params as P, lcoh as L, auctions as A, finance as F, schedule as S, resource as R, dispatch as D
from gh2 import realoptions as RO, siteselect as SS, demand as DM
from gh2 import charts as C

R_ = {}
rng = np.random.default_rng(P.SEED)

# 1. site selection
R_["site"] = SS.rank(); R_["site"]["flip"] = SS.switch_weight()

# 1b. demand forecast (Bass diffusion with uncertain parameters)
R_["demand"] = DM.summary(DM.simulate())

# 2. technology LCOH (base + MC), for continuity with Dossier 1
R_["lcoh"] = {}
for k in P.STACKS:
    base, comp, hrs = L.simulate(k, deterministic=True)
    mc, _, _ = L.simulate(k, n=100_000, rng=np.random.default_rng(P.SEED))
    R_["lcoh"][k] = dict(base=float(base[0]), hours=float(hrs[0]), **L.summary(mc))

# 3. schedule
cp = S.cpm(); mc = S.monte_carlo()
R_["schedule"] = dict(cpm_finish=cp["finish"], critical=cp["critical"], slack=cp["slack"], p50=mc["p50"],
                      p80=mc["p80"], p90=mc["p90"], criticality=mc["criticality"], merge_bias=mc["p50"] - cp["finish"],
                      p_by_ists_cutoff=float((mc["finish"] <= 48).mean()), months_early_for_p80=float(mc["p80"] - 48),
                      early_release=[S.early_release(l) for l in (2, 3, 4)])

# 4. dispatch (illustrative profiles)
sol, wnd = R.synthetic_year()
dp = D.solve(sol, wnd)
R_["dispatch"] = {k: float(v) for k, v in dp.items() if k != "hourly"}
R_["dispatch"]["solar_cf"] = float(sol.mean()); R_["dispatch"]["wind_cf"] = float(wnd.mean())

# 5. finance: base vs design-to-cost target, deterministic and Monte Carlo
target = dict(capex_kw=850, power=32, debt_rate=0.085, handling=18)
fin = {}
for case, ov in (("base", {}), ("target", target)):
    for tariff in (397, 279):
        x = F.draw_inputs(1, deterministic=True, **ov)
        m = F.model(tariff, x); mt = F.metrics(m)
        fin[f"{case}_{tariff}"] = {k: float(v[0]) for k, v in mt.items()} | dict(
            capex_cr=float(m["capex"][0]), debt_cr=float(m["debt"][0]), equity_cr=float(m["equity"][0]),
            h2_t_y1=float(m["h2_t"][0, 0]), revenue_y1=float(m["revenue"][0, 0]), power_y1=float(m["power"][0, 0]),
            ebitda_y1=float(m["ebitda"][0, 0]), dscr=[float(v) for v in m["dscr"][0, :P.PROJECT.debt_tenor]],
            swap_alk=int(m["swap_alk"][0]), swap_pem=int(m["swap_pem"][0]))
fin["required_base"] = F.required_tariff()
fin["levers"] = F.lever_ladder()
n = 20_000
xb = F.draw_inputs(n, rng=np.random.default_rng(P.SEED))
g = np.random.default_rng(P.SEED + 1)
xt = F.draw_inputs(n, rng=np.random.default_rng(P.SEED),
                   capex_kw=g.triangular(800, 850, 1000, n), power=g.triangular(30, 32, 36, n),
                   debt_rate=g.triangular(0.0825, 0.085, 0.09, n), handling=g.triangular(15, 18, 24, n))
mcb = F.metrics(F.model(397, xb)); mct = F.metrics(F.model(397, xt))
def dist(mt):
    return {k: dict(p10=float(np.nanpercentile(v, 10)), p50=float(np.nanpercentile(v, 50)), p90=float(np.nanpercentile(v, 90)))
            for k, v in mt.items()} | dict(p_min_dscr_lt_120=float((mt["min_dscr"] < 1.2).mean()),
                                            p_min_dscr_lt_100=float((mt["min_dscr"] < 1.0).mean()),
                                            p_eq_irr_lt_12=float(np.mean(np.nan_to_num(mt["equity_irr"], nan=-1) < 0.12)))
fin["mc_base_397"] = dist(mcb); fin["mc_target_397"] = dist(mct)
# same target futures, but an 18-year loan (one of the financing levers in Dossier 3)
import dataclasses
_base_project = P.PROJECT
P.PROJECT = dataclasses.replace(_base_project, debt_tenor=18)
fin["mc_target_397_tenor18"] = dist(F.metrics(F.model(397, xt)))
P.PROJECT = _base_project
R_["finance"] = fin

# 6. auctions (research paper preview)
st, summ, cost_mkt, cost_fav = A.run()
st.to_csv(os.path.join(ROOT, "results", "auctions.csv"), index=False)
R_["auctions"] = summ

# 7. real options
R_["options"] = dict(
    expansion_cliff=RO.expansion_lsmc(cliff=0.12), expansion_nocliff=RO.expansion_lsmc(cliff=0.0),
    expansion_cliff_lowvol=RO.expansion_lsmc(cliff=0.12, sigma=0.15), expansion_nocliff_lowvol=RO.expansion_lsmc(cliff=0.0, sigma=0.15),
    aem=RO.aem_switch())

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
    json.dump(R_, f, indent=1, default=float)

# 8. charts
C.all_charts(R_, st, cost_mkt, cost_fav, mc["finish"], sol, wnd, dp, mcb, mct)
print("done")
