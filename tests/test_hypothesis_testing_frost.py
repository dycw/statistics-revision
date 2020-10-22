from __future__ import annotations

from numpy import isclose
from numpy import mean
from numpy import std
from pandas import read_csv
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import ttest_1samp
from statsmodels.stats.weightstats import DescrStatsW

from statistics_revision import CODE_ROOT


BOOK_ROOT = CODE_ROOT.joinpath("hypothesis_testing_frost")


def test_descriptive_statistics_fuel_cost_p28() -> None:
    path = BOOK_ROOT.joinpath("FuelsCosts.csv")
    df = read_csv(path)
    X = df["Fuel Cost"]
    assert len(X) == 25
    assert mean(X) == 330.56
    assert sem(X) == 30.8355357771949
    assert std(X) == 151.06265719892525


def test_1_sample_t_tests_p47() -> None:
    path = BOOK_ROOT.joinpath("AssessmentScores.csv")
    df = read_csv(path)
    X = df["Score"]
    res = ttest_1samp(X, 60)
    assert res.statistic == 1.41838979994091
    assert res.pvalue == 0.17795265748623812
    assert (n := len(X)) == 15
    assert (mu := mean(X)) == 64.15818651056485
    assert (sem_ := sem(X)) == 2.9316246568736433
    confidence = 0.95
    width = sem_ * t.ppf((1.0 + confidence) / 2, n - 1)
    ci_1 = mu - width, mu + width
    ci_2 = t.interval(confidence, n - 1, loc=mu, scale=sem_)
    ci_3 = DescrStatsW(X).tconfint_mean()
    assert ci_1 == ci_2
    assert isclose(ci_1, ci_3).all()
