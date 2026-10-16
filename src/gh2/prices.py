"""Mean-reverting price model for grey ammonia, urea and gas (Schwartz one-factor).

ln S_t follows an Ornstein-Uhlenbeck process: d ln S = kappa (mu - ln S) dt + sigma dW.
This corrects the Gemini brief, which asked for a 'mean-reverting geometric Brownian motion'
(GBM does not mean-revert). The exact discretisation is an AR(1):
    x_{t+1} = a + b x_t + e,  b = exp(-kappa dt),  a = mu (1 - b),  var(e) = sigma^2 (1-b^2)/(2 kappa)
so we estimate it by OLS on monthly log prices (World Bank Pink Sheet urea/gas; India import
unit values for ammonia; see data/README.md for pulls).
"""
from __future__ import annotations
import numpy as np


def fit_ou(prices, dt=1 / 12):
    x = np.log(np.asarray(prices, float))
    X = np.vstack([np.ones(len(x) - 1), x[:-1]]).T
    (a, b), *_ = np.linalg.lstsq(X, x[1:], rcond=None)
    resid = x[1:] - X @ np.array([a, b])
    b = min(max(b, 1e-6), 0.999999)
    kappa = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.sqrt(resid.var(ddof=2) * 2 * kappa / (1 - b ** 2))
    return dict(kappa=float(kappa), mu=float(mu), sigma=float(sigma), half_life_years=float(np.log(2) / kappa),
                long_run_price=float(np.exp(mu)))


def simulate_ou(S0, kappa, mu, sigma, years, n=10_000, dt=1 / 12, rng=None):
    rng = rng or np.random.default_rng(0)
    steps = int(round(years / dt)); b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa))
    x = np.full(n, np.log(S0)); out = np.empty((n, steps))
    for t in range(steps):
        x = mu + (x - mu) * b + sd * rng.standard_normal(n)
        out[:, t] = x
    return np.exp(out)
