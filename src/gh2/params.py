"""Single source of truth for every assumption in the project.

Why one file: a reviewer (or Prof. Gopal) should be able to challenge any number in the
deck and find, in one place, its value, its range and where it came from. Each entry is
(low, mode, high) where uncertain. Tags: REPORTED (published), ESTIMATED (ours),
PROJECTED (a target). Sources are the reference keys used in the dossiers.
"""
from dataclasses import dataclass, field

FX = 90.0              # INR per USD, planning rate (rupee 86.6-91.9 in 2025-26)       ESTIMATED
FX_AUCTION = 87.0      # INR per USD around the July-Aug 2025 SECI auctions             ESTIMATED
LIFE = 25              # economic life, years                                            ESTIMATED
SEED = 20261013        # every stochastic result in the dossiers is reproducible

# ---- shared uncertain inputs ----------------------------------------------------------
POWER_USD_MWH = (32.0, 38.0, 48.0)   # hybrid RE delivered; SECI solar+storage Rs2.86/kWh (sqsolar2025) to FDRE Rs5.25 (pvmfdre2026)
WACC = (0.085, 0.10, 0.125)          # ESTIMATED (Gemini brief 9.5-11.5%)
RE_HOURS = (0.55, 0.62, 0.68)        # share of hours an oversized solar+wind hybrid can feed the stack, ESTIMATED
WATER_USD_KG = 0.03                  # purification <= $0.10/kg (rmi2023)
GREY_H2_USD_KG = (2.1, 2.2)          # India grey hydrogen (jmkiocl2025; tribune2025) REPORTED
GREY_CO2_KG = 11.0                   # kg CO2e per kg grey H2, IEA 10-12 REPORTED
H2_PER_T_NH3 = 0.178                 # t H2 per t NH3
NH3_LOOP_USD_T = (100.0, 130.0, 160.0)  # ASU + Haber-Bosch + storage per t NH3, ESTIMATED (low confidence)

@dataclass(frozen=True)
class Stack:
    name: str
    capex: tuple            # installed $/kW (system incl. BoP, EPC, contingency; excl. RE)
    eff: tuple              # kWh/kg system
    life: tuple             # stack life, operating hours
    repl: float             # stack replacement cost, share of capex
    om: float               # fixed O&M, share of capex per year
    capture: float = 0.92   # share of RE-available hours the stack can use (min-load limits)
    cf: tuple | None = None # fixed capacity factor (SOEC runs on firm power)
    power: tuple | None = None
    heat_kwh: float = 0.0
    heat_price: float = 0.0

STACKS = {
    "ALK":    Stack("Alkaline (pressurised)", (800, 1100, 1600), (50, 53, 57), (60e3, 75e3, 90e3), 0.35, 0.020, 0.92),
    "HYBRID": Stack("Hybrid ALK+PEM (80:20)", (900, 1250, 1750), (50.5, 53.2, 57.2), (55e3, 72e3, 88e3), 0.35, 0.021, 0.975),
    "PEM":    Stack("PEM", (1300, 1800, 2400), (51, 54, 58), (40e3, 60e3, 80e3), 0.35, 0.025, 0.99),
    "AEM":    Stack("AEM", (1000, 1700, 3000), (51, 55, 62), (15e3, 30e3, 50e3), 0.40, 0.030, 0.98),
    "SOEC":   Stack("SOEC", (2800, 3800, 5600), (37, 40, 44), (10e3, 20e3, 30e3), 0.45, 0.040, 1.0,
                    cf=(0.80, 0.85, 0.90), power=(50.0, 58.0, 66.0), heat_kwh=8.0, heat_price=12.0),
}

# ---- the chosen project (Dossier 3) ---------------------------------------------------
@dataclass(frozen=True)
class Project:
    mw_alk: float = 80.0
    mw_pem: float = 20.0                 # the "option bay": PEM today, AEM candidate at first PEM stack swap
    availability: float = 0.97           # plant availability on top of RE-limited hours, ESTIMATED
    capex_electrolysis_usd_kw: tuple = (900, 1250, 1750)
    capex_h2_handling_musd: tuple = (15.0, 22.0, 32.0)   # compression, 10 t buffer, purification, 5 km pipeline, ESTIMATED
    owners_contingency: float = 0.05     # owner's costs only (land, development); EPC contingency is already in the $/kW
    debt_share: float = 0.70
    debt_rate: float = 0.090             # INR project debt (Gemini brief 8.25-9.5%), ESTIMATED
    debt_tenor: int = 15
    tax_rate: float = 0.2517             # Indian corporate rate under section 115BAA
    om_escalation: float = 0.04
    insurance: float = 0.005
    degradation_per_year: float = 0.006  # rise in kWh/kg per operating year until stack swap, ESTIMATED
    construction_years: int = 2

PROJECT = Project()

# ---- offtake benchmarks ---------------------------------------------------------------
OFFTAKE = {
    "IOCL Panipat (L&T, 25 yr)":      397.0,   # Rs/kg H2 (jmkiocl2025) REPORTED
    "Numaligarh (NeuEN, 25 yr)":      279.0,   # Rs/kg H2 (rwnrl2026) REPORTED
}
SECI_AMMONIA_WAVG_INR_KG = 53.27              # computed from seci2025
SIGHT_2A_INR_KG_NH3 = (8.82, 7.06, 5.30)      # years 1-3 (sq2024) REPORTED
