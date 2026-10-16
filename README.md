# gh2-india · green hydrogen project engine

The analysis behind our PRM project *Green Hydrogen Plant: Managing an Emerging Technology Project*
(NIT Warangal, 2026–27) and the companion research paper on India's green-ammonia auctions.

**The project in one line:** a 100 MW hybrid electrolysis plant (80 MW pressurised alkaline + 20 MW PEM,
the PEM slot doubling as an AEM "option bay") at Paradip, selling hydrogen to a refinery on a 25-year fixed price.

| Module | Question it answers | Method |
|---|---|---|
| `lcoh.py` | What does a kilogram cost, by stack? | Levelised cost, 100,000-draw Monte Carlo |
| `siteselect.py` | Where should the plant go? | AHP with consistency check and weight sensitivity |
| `schedule.py` | When will it start producing, and how sure are we? | CPM + Beta-PERT Monte Carlo, criticality index |
| `resource.py` / `dispatch.py` | How much solar, wind and storage per MW of electrolyser? | Hourly LP (HiGHS), buffer-inventory view |
| `finance.py` | Is it bankable? | SPV cash-flow waterfall, DSCR/LLCR, IRR, MMRA, Monte Carlo |
| `realoptions.py` | What are phase 2 and the AEM bay worth? | Least-squares Monte Carlo; learning-curve switch option |
| `auctions.py` | Do SECI green-ammonia auctions clear below cost? | Bid-structure, cost-coverage, implied-return tests |
| `demand.py` | How fast will Indian buyers switch to green hydrogen? | Bass diffusion, 20,000 parameter draws, fan chart |
| `prices.py` | How do grey ammonia / gas prices move? | Schwartz one-factor (log-OU) fit and simulation |

```bash
pip install -r requirements.txt
python scripts/run_all.py      # every number and chart in Dossier 3 -> results/, figures/
pytest                         # 8 sanity tests
```

All assumptions live in `src/gh2/params.py`, each tagged REPORTED / ESTIMATED / PROJECTED with its source.
Random seed is fixed (20261013), so results are reproducible.

**Honest limits.** `resource.synthetic_year()` is a stand-in until real hourly data is pulled
(`resource.fetch_pvgis_hourly`, `fetch_nasa_power_hourly`); dispatch results are illustrative until then.
`prices.py` is tested on simulated data; it still needs the World Bank Pink Sheet pull (see `data/README.md`).

**The write-up.** `dossiers/pdf/` holds the three dossiers; Dossier 3 (*The Project Book*) is built entirely from this repository's outputs.

Authors: Dhritabrata Swarnakar, Shalvi Bhatnagar · MIT licence (code) · fonts under SIL OFL.
