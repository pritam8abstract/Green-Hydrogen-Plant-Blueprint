"""Fast sanity tests: each guards a property the dossiers rely on."""
import numpy as np
from gh2 import lcoh as L, finance as F, schedule as S, prices as PR, siteselect as SS, auctions as A, params as P


def test_crf_matches_annuity():
    assert abs(float(L.crf(0.10, 25)) - 0.110168) < 1e-5


def test_stack_ranking_is_stable():
    base = {k: float(L.simulate(k, deterministic=True)[0][0]) for k in P.STACKS}
    assert base["ALK"] < base["HYBRID"] < base["PEM"] < base["AEM"] < base["SOEC"]


def test_irr_recovers_known_rate():
    cf = np.array([[-100.0] + [10.0] * 30])
    assert abs(F.irr_vec(cf)[0] - 0.0930) < 1e-3


def test_cpm_finish_and_merge_bias():
    assert S.cpm()["finish"] == 44
    assert S.monte_carlo(n=5000)["p50"] > 44          # merge bias: MC median beats the deterministic path


def test_ou_fit_recovers_parameters():
    x = PR.simulate_ou(500, 0.8, np.log(450), 0.3, 60, n=1, rng=np.random.default_rng(1))[0]
    fit = PR.fit_ou(x)
    assert 0.4 < fit["kappa"] < 1.6 and abs(fit["sigma"] - 0.3) < 0.06


def test_ahp_is_consistent():
    assert SS.rank()["cr"] < 0.10


def test_seci_dataset_complete():
    df = A.load()
    assert len(df) == 65 and df.project.nunique() == 13
    st = A.bid_structure(df)
    assert int(st.qty_tpa.sum()) == 724_000


def test_bass_curve_is_a_cdf():
    from gh2 import demand as DM
    t = np.linspace(0, 60, 200); F = DM.bass_cdf(t, 0.01, 0.4)
    assert F[0] == 0 and np.all(np.diff(F) >= 0) and 0.99 < F[-1] <= 1
