# Data

| File / source | Status | Use |
|---|---|---|
| `raw/seci_mode2a_tranche1_bids.csv` | In repo. All 65 bids, 13 auctions, from SECI's e-RA result sheet (2025) | auctions.py |
| World Bank Pink Sheet (monthly urea, gas) | To pull: worldbank.org/en/research/commodity-markets | prices.py |
| India ammonia import unit values (HS 2814) | To pull: tradestat.commerce.gov.in | prices.py (proxy benchmark) |
| PVGIS hourly PV (lat/lon of Paradip, Kandla, Kakinada, Tuticorin) | To pull with `resource.fetch_pvgis_hourly` | dispatch.py |
| NASA POWER hourly 50 m wind | To pull with `resource.fetch_nasa_power_hourly` | dispatch.py |
| PPAC monthly refinery crude throughput | To pull: ppac.gov.in | demand.py (refinery hydrogen base; seasonal forecast) |
| IEX day-ahead prices | To pull: iexindia.com | grid/firm-power scenarios |
| CEA / Grid-India generation reports | To pull | validate simulated capacity factors |
| RBI DBIE, FBIL | To pull | rates and INR/USD |
