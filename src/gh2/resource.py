"""Hourly solar and wind capacity factors.

Two paths:
* synthetic_year(): a physically reasoned stand-in used in the dossier because the build
  container cannot reach PVGIS or NASA POWER. Solar follows the sun's daily arc, dips in the
  south-west monsoon (Jun-Sep) and gets random cloudy days; wind is strongest in the monsoon
  and in the evening, with persistent (AR-1) noise. Annual CFs land near Indian norms
  (solar ~0.21, wind ~0.33). Results built on it are ILLUSTRATIVE.
* fetch_pvgis_hourly() / fetch_nasa_power_hourly(): run these on your own machine to replace
  the synthetic year with real data for Kandla, Kakinada, Paradip or Tuticorin.
"""
from __future__ import annotations
import numpy as np

SITES = {"Paradip": (20.26, 86.67), "Kandla": (23.03, 70.22), "Kakinada": (16.99, 82.25), "Tuticorin": (8.76, 78.13)}


def synthetic_year(seed=7):
    rng = np.random.default_rng(seed)
    h = np.arange(8760); day = h // 24; hod = h % 24
    month = np.minimum((day / 30.42).astype(int), 11)
    monsoon = np.isin(month, [5, 6, 7, 8])
    # solar: daylight arc 06-18 h, seasonal amplitude, cloudy-day multiplier
    arc = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) ** 1.3
    season = 0.92 + 0.08 * np.cos(2 * np.pi * (day - 80) / 365)
    cloud_day = rng.beta(8, 1.6, 366)[day] * np.where(monsoon, 0.68, 1.0)
    solar = np.clip(1.0 * arc * season * cloud_day, 0, 1)
    # wind: monsoon peak, evening bump, AR(1) noise
    base = np.where(monsoon, 0.52, 0.24) + 0.06 * np.cos(2 * np.pi * (hod - 19) / 24)
    noise = np.zeros(8760); e = rng.normal(0, 0.08, 8760)
    for t in range(1, 8760):
        noise[t] = 0.92 * noise[t - 1] + e[t]
    wind = np.clip(base + noise, 0, 1)
    return solar, wind


def fetch_pvgis_hourly(lat, lon, year=2020):  # pragma: no cover - needs internet
    import requests
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    r = requests.get(url, params=dict(lat=lat, lon=lon, startyear=year, endyear=year, pvcalculation=1,
                                      peakpower=1, loss=14, outputformat="json"), timeout=60)
    r.raise_for_status()
    rows = r.json()["outputs"]["hourly"]
    return np.array([x["P"] / 1000 for x in rows])          # kW per kWp == capacity factor


def fetch_nasa_power_hourly(lat, lon, year=2020):  # pragma: no cover - needs internet
    import requests
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    r = requests.get(url, params=dict(parameters="WS50M", community="RE", longitude=lon, latitude=lat,
                                      start=f"{year}0101", end=f"{year}1231", format="JSON"), timeout=60)
    r.raise_for_status()
    ws = np.array(list(r.json()["properties"]["parameter"]["WS50M"].values()), float)
    ws100 = ws * (100 / 50) ** 0.14                           # shear to hub height
    v_in, v_r, v_out = 3.0, 11.5, 25.0                         # generic 3 MW-class power curve
    cf = np.clip((ws100 ** 3 - v_in ** 3) / (v_r ** 3 - v_in ** 3), 0, 1)
    cf[ws100 > v_out] = 0
    return cf
