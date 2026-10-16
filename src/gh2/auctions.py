"""Winner's-curse diagnostics for SECI SIGHT Mode 2A Tranche I (13 green-ammonia auctions).

The research question: do India's green-ammonia auctions clear below the probabilistic cost
of delivery? Three lenses:
  1. Bid structure  - how close are winner and runner-up, how dispersed are credible bids?
  2. Cost coverage  - probability that the winning price covers a Monte Carlo cost of supply.
  3. Implied return - the cost of capital at which each winning bid exactly breaks even,
                      under deliberately favourable (best-in-class) developer inputs.
If even favourable inputs imply returns below a normal cost of equity, the auction is
selecting optimism, which is the textbook signature of the winner's curse.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from . import params as P
from .lcoh import simulate, lcoh, lcoa_from_lcoh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIDS = os.path.join(ROOT, "data", "raw", "seci_mode2a_tranche1_bids.csv")
SERIOUS = 1.5   # bids above 1.5x the winner are treated as non-competitive (e.g. Rs500/kg placeholders)

# Inputs of a best-in-class developer (the "what you have to believe" ladder in Dossier 1)
FAVOURABLE = dict(capex=750.0, eff=51.0, life=75e3, power=30.0, re_hours=0.68)


def load():
    return pd.read_csv(BIDS)


def bid_structure(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (proj, off, loc, q), g in df.groupby(["project", "offtaker", "location", "qty_tpa"], sort=False):
        g = g.sort_values("price_inr_per_kg")
        win, p1 = g.iloc[0], g.iloc[0].price_inr_per_kg
        serious = g[g.price_inr_per_kg <= SERIOUS * p1]
        p2 = g.iloc[1].price_inr_per_kg if len(g) > 1 else np.nan
        rows.append(dict(project=proj, offtaker=off, location=loc, qty_tpa=q, winner=win.bidder,
                         n_bids=len(g), n_serious=len(serious), win_price=p1, runner_up=p2,
                         gap_inr=p2 - p1, gap_pct=100 * (p2 - p1) / p1,
                         cv_serious_pct=100 * serious.price_inr_per_kg.std(ddof=0) / serious.price_inr_per_kg.mean(),
                         left_on_table_cr_yr=(p2 - p1) * q * 1000 / 1e7))
    return pd.DataFrame(rows)


def sight_levelised_inr_kg(rate=0.10, years=10):
    """SIGHT incentive (Rs/kg NH3, years 1-3) spread evenly over the 10-year contract."""
    pv = sum(v / (1 + rate) ** (t + 1) for t, v in enumerate(P.SIGHT_2A_INR_KG_NH3))
    ann = rate / (1 - (1 + rate) ** -years)
    return pv * ann


def cost_distribution(n=100_000, rng=None, favourable=False):
    """Net cost of supplying green ammonia (Rs/kg NH3) for a cheapest-stack (alkaline) developer."""
    rng = rng or np.random.default_rng(P.SEED)
    ov = None
    if favourable:  # narrow the ranges around best-in-class values instead of the market spread
        ov = {k: (v * 0.92, v, v * 1.10) for k, v in FAVOURABLE.items() if k != "life"}
        ov["wacc"] = (0.075, 0.085, 0.10)
    h2, _, _ = simulate("ALK", n=n, rng=rng, overrides=ov)
    loop = rng.triangular(*P.NH3_LOOP_USD_T, n)
    usd_t = lcoa_from_lcoh(h2, loop)
    return usd_t * P.FX_AUCTION / 1000 - sight_levelised_inr_kg()


def coverage(struct: pd.DataFrame, n=100_000):
    base = cost_distribution(n)
    fav = cost_distribution(n, favourable=True)
    struct = struct.copy()
    struct["p_cover_market"] = [float((base <= p).mean()) for p in struct.win_price]
    struct["p_cover_favourable"] = [float((fav <= p).mean()) for p in struct.win_price]
    return struct, base, fav


def implied_wacc(bid_inr_kg, inputs=FAVOURABLE, loop_usd_t=P.NH3_LOOP_USD_T[1]):
    """WACC at which a best-in-class alkaline plant's levelised ammonia cost equals the bid."""
    s = P.STACKS["ALK"]
    hours = inputs["re_hours"] * s.capture * 8760

    def gap(w):
        h2, _ = lcoh(inputs["capex"], inputs["eff"], inputs["life"], s.repl, s.om, hours, inputs["power"], w)
        cost = lcoa_from_lcoh(float(h2), loop_usd_t) * P.FX_AUCTION / 1000 - sight_levelised_inr_kg()
        return cost - bid_inr_kg
    try:
        return brentq(gap, -0.05, 0.40)
    except ValueError:
        return np.nan


def run():
    df = load()
    st = bid_structure(df)
    st, base, fav = coverage(st)
    st["implied_wacc_favourable"] = [implied_wacc(p) for p in st.win_price]
    q = st.qty_tpa.values
    summary = dict(
        n_auctions=int(len(st)), n_bids=int(len(df)), n_bidders=int(df.bidder.nunique()),
        total_tpa=int(q.sum()), wavg_price=float((st.win_price * q).sum() / q.sum()),
        median_gap_inr=float(st.gap_inr.median()), share_gap_le_025=float((st.gap_inr <= 0.25 + 1e-9).mean()),
        left_on_table_cr_yr=float(st.left_on_table_cr_yr.sum()),
        cost_market_p50=float(np.median(base)), cost_market_p10=float(np.percentile(base, 10)),
        cost_fav_p50=float(np.median(fav)), cost_fav_p10=float(np.percentile(fav, 10)),
        p_cover_market_wavg=float((st.p_cover_market * q).sum() / q.sum()),
        p_cover_fav_wavg=float((st.p_cover_favourable * q).sum() / q.sum()),
        implied_wacc_min=float(np.nanmin(st.implied_wacc_favourable)),
        implied_wacc_max=float(np.nanmax(st.implied_wacc_favourable)),
        implied_wacc_wavg=float(np.nansum(st.implied_wacc_favourable * q) / q[~np.isnan(st.implied_wacc_favourable)].sum()),
        sight_levelised=float(sight_levelised_inr_kg()),
        acme_share=float(q[st.winner.str.startswith("ACME").values].sum() / q.sum()),
    )
    return st, summary, base, fav
