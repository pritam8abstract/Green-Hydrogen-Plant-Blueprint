"""House-style charts (cream ledger paper, validated categorical palette).
Palette checked with the dataviz validator; identity is always carried by labels as well."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from . import schedule as S, params as P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "figures")
FONTS = os.path.join(ROOT, "assets", "fonts")
if os.path.isdir(FONTS):
    for f in os.listdir(FONTS):
        fm.fontManager.addfont(os.path.join(FONTS, f))
SURF, INK, INK2, GRID = "#FBF8F1", "#1B2A41", "#52514E", "#E4DDCC"
S1, S2, S3, S4, S5 = "#2F6DB0", "#C47A12", "#2E8B5E", "#7A62B8", "#C0433F"
NEU = "#A9A496"
plt.rcParams.update({"font.family": ["Newsreader", "DejaVu Serif"], "font.size": 9.3, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK, "axes.labelcolor": INK2, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.titlesize": 9.5, "axes.titlecolor": INK})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.04); plt.close(fig)


def site(R):
    sc = R["site"]["scores"]
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    names = list(sc)[::-1]; vals = [sc[n] for n in names]
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.barh(i, v, color=S1 if n == list(sc)[0] else NEU, height=0.6, edgecolor=SURF)
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names)), names); ax.set_xlim(5, 8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("AHP weighted score (1–9 scale)"); ax.tick_params(axis="y", length=0)
    _save(fig, "site_ahp.pdf")


def gantt(R):
    cp = S.cpm(); crit = set(cp["critical"])
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ks = S.ORDER[::-1]
    for i, k in enumerate(ks):
        name, (o, m, p), _ = S.ACTIVITIES[k]
        ef = cp["ef"][k]; es = ef - m
        ax.barh(i, m, left=es, color=S5 if k in crit else S1, height=0.58, edgecolor=SURF)
        if cp["slack"][k] > 1e-9:
            ax.barh(i, cp["slack"][k], left=ef, color=GRID, height=0.58, edgecolor=SURF)
        ax.text(es - 0.4, i, f"{k}  {name}", ha="right", va="center", fontsize=7.6)
    ax.set_yticks([]); ax.set_xlim(-30, 50); ax.set_xticks(range(0, 50, 6))
    ax.set_xlabel("Months from project start (Jan 2027), most-likely durations")
    ax.axvline(cp["finish"], color=INK, lw=1); ax.text(cp["finish"] + 0.4, len(ks) - 0.6, f"CPM finish\n{cp['finish']:.0f} months", fontsize=7.6)
    ax.legend(handles=[Patch(color=S5, label="critical path"), Patch(color=S1, label="has float"), Patch(color=GRID, label="float (slack)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.4)
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    _save(fig, "gantt.pdf")


def schedule_mc(R, finish):
    s = R["schedule"]
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    xs = np.sort(finish); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color=S1, lw=2)
    ax.axvline(s["cpm_finish"], color=INK2, lw=1); ax.text(s["cpm_finish"] - 0.3, 0.92, "CPM (most likely)", ha="right", fontsize=7.6)
    for q, lab in ((s["p50"], "P50"), (s["p80"], "P80")):
        ax.plot([q], [0.5 if lab == "P50" else 0.8], "o", color=S5, ms=6)
        ax.text(q + 0.4, (0.5 if lab == "P50" else 0.8) - 0.06, f"{lab}: {q:.1f} months", fontsize=7.8, color=INK)
    ax.set_xlabel("Months to commercial operation"); ax.set_ylabel("Probability finished")
    ax.set_xlim(40, 62); ax.set_ylim(0, 1.02)
    _save(fig, "schedule_mc.pdf")


def dispatch(sol, wnd, dp):
    h = dp["hourly"]
    weeks = {"Dry season (February)": 31 * 24 + 7 * 24, "Monsoon (July)": 181 * 24}
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.6), sharex="col", gridspec_kw=dict(height_ratios=[2, 1]))
    for j, (lab, st) in enumerate(weeks.items()):
        sl = slice(st, st + 168); t = np.arange(168) / 24
        a = axes[0, j]
        a.stackplot(t, h["solar"][sl], h["wind"][sl], h["grid"][sl], colors=[S2, S1, S5], edgecolor=SURF, linewidth=0.3)
        a.set_title(lab, loc="left"); a.set_ylim(0, 105)
        if j == 0: a.set_ylabel("MW to electrolyser")
        b = axes[1, j]
        b.plot(t, h["storage"][sl], color=S3, lw=1.6)
        b.set_ylim(0, max(1, dp["storage_t"]) * 1.1); b.set_xlabel("Day of week")
        if j == 0: b.set_ylabel("H$_2$ in store (t)")
    fig.legend(handles=[Patch(color=S2, label="solar"), Patch(color=S1, label="wind"), Patch(color=S5, label="firm RE (FDRE)"),
                        plt.Line2D([], [], color=S3, lw=1.6, label="hydrogen buffer")],
               loc="lower center", ncol=4, fontsize=7.4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "dispatch.pdf")


def dscr(R):
    f = R["finance"]
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    yrs = np.arange(1, 16)
    ax.plot(yrs, f["base_397"]["dscr"], color=S5, lw=2, marker="o", ms=3.5)
    ax.plot(yrs, f["target_397"]["dscr"], color=S1, lw=2, marker="o", ms=3.5)
    ax.axhline(1.30, color=INK, lw=1); ax.text(0.6, 1.33, "lender covenant 1.30×", va="bottom", fontsize=7.4)
    ax.text(15.2, f["base_397"]["dscr"][-1], "base design", va="center", fontsize=7.6, color=S5)
    ax.text(15.2, f["target_397"]["dscr"][-1], "design-to-cost", va="center", fontsize=7.6, color=S1)
    ax.set_xlim(0.5, 17.5); ax.set_xticks(yrs); ax.set_xlabel("Operating year (tariff ₹397/kg, flat)"); ax.set_ylabel("DSCR (×)")
    _save(fig, "dscr.pdf")


def finance_mc(mcb, mct):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    for a, mt, lab, col in ((axes[0], mcb, "Base design ranges", S5), (axes[1], mct, "Design-to-cost ranges", S1)):
        v = np.nan_to_num(mt["equity_irr"], nan=-0.2) * 100
        a.hist(np.clip(v, -20, 40), bins=50, color=col, edgecolor=SURF, linewidth=0.4)
        a.axvline(14, color=INK, lw=1); a.text(14.5, a.get_ylim()[1] * 0.85, "14% hurdle", fontsize=7.4)
        a.set_title(lab, loc="left"); a.set_xlabel("Equity IRR (%)"); a.grid(axis="x", visible=False)
    axes[0].set_ylabel("Simulations")
    _save(fig, "finance_mc.pdf")


def tariff(R):
    f = R["finance"]
    rows = [("Required: base design (DSCR 1.30, equity 14%)", max(f["required_base"].values()), S5),
            ("IOCL Panipat contract (L&T, 2025)", 397, NEU), ("Numaligarh contract (NeuEN, 2026)", 279, NEU)]
    fig, ax = plt.subplots(figsize=(6.3, 1.7))
    for i, (lab, v, c) in enumerate(rows[::-1]):
        ax.barh(i, v, color=c, height=0.55, edgecolor=SURF); ax.text(v + 5, i, f"₹{v:.0f}/kg", va="center", fontsize=8.4)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8); ax.set_xlim(0, 620)
    ax.set_xlabel("₹ per kg of hydrogen, 25-year fixed price"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    _save(fig, "tariff.pdf")


def auction_gap(st):
    st = st.sort_values("win_price")
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ys = np.arange(len(st))[::-1]
    for y, (_, r) in zip(ys, st.iterrows()):
        ax.plot([r.win_price, r.runner_up], [y, y], color=GRID, lw=3, solid_capstyle="round")
        ax.plot([r.win_price], [y], "o", color=S1, ms=6); ax.plot([r.runner_up], [y], "o", color=S2, ms=6)
        ax.text(max(r.win_price, r.runner_up) + 0.4, y, f"+₹{r.gap_inr:.2f}", va="center", fontsize=7.4)
    ax.set_yticks(ys, [f"{r.location} ({r.winner.split()[0]})" for _, r in st.iterrows()], fontsize=7.8)
    ax.set_xlabel("₹ per kg green ammonia"); ax.set_xlim(48, 70); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=S1, label="winning bid"),
                       plt.Line2D([], [], marker="o", ls="", color=S2, label="runner-up")], loc="upper right", fontsize=7.6)
    _save(fig, "auction_gap.pdf")


def auction_cost(st, cost_mkt, cost_fav):
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    bins = np.linspace(35, 95, 90)
    ax.hist(cost_mkt, bins=bins, color=S5, alpha=0.75, label="cost, market-average developer", edgecolor=SURF, linewidth=0.3)
    ax.hist(cost_fav, bins=bins, color=S1, alpha=0.75, label="cost, best-in-class developer", edgecolor=SURF, linewidth=0.3)
    top = ax.get_ylim()[1]
    for p in st.win_price:
        ax.plot([p, p], [top * 1.02, top * 1.09], color=INK, lw=1.1)
    ax.text(st.win_price.min(), top * 1.12, "13 winning bids", fontsize=7.6)
    ax.set_ylim(0, top * 1.2); ax.set_xlabel("Net cost of supplying green ammonia, ₹/kg (after SIGHT incentive)")
    ax.set_ylabel("Simulations"); ax.legend(loc="upper right", fontsize=7.4); ax.grid(axis="x", visible=False)
    _save(fig, "auction_cost.pdf")


def implied_wacc(st):
    st = st.sort_values("implied_wacc_favourable")
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ys = np.arange(len(st))[::-1]
    for y, (_, r) in zip(ys, st.iterrows()):
        v = r.implied_wacc_favourable * 100
        ax.barh(y, v, color=S1 if v >= 12 else S5, height=0.6, edgecolor=SURF)
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=7.4)
    ax.axvline(12, color=INK, lw=1); ax.text(12.3, -0.9, "12%: a typical Indian equity hurdle", fontsize=7.4)
    ax.set_yticks(ys, [f"{r.location} ₹{r.win_price:.2f}" for _, r in st.iterrows()], fontsize=7.8)
    ax.set_xlabel("Return implied by the winning bid for a best-in-class alkaline developer (%)")
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0); ax.set_xlim(0, 30); ax.set_ylim(-1.4, len(st) - 0.4)
    _save(fig, "implied_wacc.pdf")


def options(R):
    o = R["options"]
    rows = [("σ = 30%, no 2030 cliff", o["expansion_nocliff"]["value"]), ("σ = 30%, with 2030 cliff", o["expansion_cliff"]["value"]),
            ("σ = 15%, no cliff", o["expansion_nocliff_lowvol"]["value"]), ("σ = 15%, with cliff", o["expansion_cliff_lowvol"]["value"]),
            ("Build phase 2 now (static NPV)", o["expansion_cliff"]["npv_now"])]
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    for i, (lab, v) in enumerate(rows[::-1]):
        ax.barh(i, v, color=NEU if "static" in lab else S4, height=0.58, edgecolor=SURF); ax.text(v + 5, i, f"₹{v:.0f} cr", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Value of the phase-2 expansion right, ₹ crore (2027)"); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.2)
    _save(fig, "options.pdf")


def demand(R):
    d = R["demand"]; yrs = d["years"]; f = d["fan"]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.fill_between(yrs, f["10"] if "10" in f else f[10], f["90"] if "90" in f else f[90], color=S1, alpha=0.18, lw=0, label="P10–P90")
    ax.fill_between(yrs, f["25"] if "25" in f else f[25], f["75"] if "75" in f else f[75], color=S1, alpha=0.30, lw=0, label="P25–P75")
    ax.plot(yrs, f["50"] if "50" in f else f[50], color=S1, lw=1.8, label="P50")
    ax.scatter([2030], [5.0], color=S5, zorder=5, s=28); ax.annotate("Mission target\n5 Mt by 2030", (2030, 5.0), (2031.2, 5.05), fontsize=7.6, color=S5, va="center")
    ax.set_xlim(2025, 2040); ax.set_ylim(0, max(max(f[90] if 90 in f else f["90"]), 5.5) * 1.06); ax.set_ylabel("Green hydrogen demand, Mt/yr")
    ax.legend(loc="upper left", fontsize=7.6)
    _save(fig, "demand.pdf")


def levers(R):
    L = R["finance"]["levers"]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    labels = [l["lever"].replace("$", r"\$") for l in L]; vals = [l["binding"] for l in L]
    for i, (lab, v) in enumerate(zip(labels, vals)):
        y = len(L) - 1 - i
        if i == 0:
            ax.barh(y, v, color=S5, height=0.56, edgecolor=SURF)
        else:
            ax.barh(y, vals[i - 1] - v, left=v, color=S3, height=0.56, edgecolor=SURF)
            ax.text(vals[i - 1] + 4, y, f"−₹{vals[i-1]-v:.0f}", va="center", fontsize=7.8, color=S3)
        ax.text(v - 4, y, f"₹{v:.0f}", va="center", ha="right", fontsize=7.8, color=INK)
    for x, lab in ((397, "IOCL ₹397"), (279, "Numaligarh ₹279")):
        ax.axvline(x, color=INK, lw=0.9, ls=(0, (3, 2))); ax.text(x + 3, len(L) - 0.45, lab, fontsize=7.6, color=INK)
    ax.set_yticks(range(len(L)), labels[::-1], fontsize=7.8); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlim(250, 560); ax.set_ylim(-0.6, len(L) - 0.1)
    ax.set_xlabel("Tariff needed for min DSCR 1.30 and equity IRR 14%, ₹/kg hydrogen (cumulative)")
    _save(fig, "levers.pdf")


def all_charts(R, st, cost_mkt, cost_fav, finish, sol, wnd, dp, mcb, mct):
    levers(R); demand(R); site(R); gantt(R); schedule_mc(R, finish); dispatch(sol, wnd, dp); dscr(R); finance_mc(mcb, mct); tariff(R)
    auction_gap(st); auction_cost(st, cost_mkt, cost_fav); implied_wacc(st); options(R)
