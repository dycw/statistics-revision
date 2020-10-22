from __future__ import annotations

from numpy import isclose
from numpy import mean
from numpy import sqrt
from numpy import std
from numpy import var
from pandas import read_csv
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from statsmodels.stats.weightstats import DescrStatsW

from statistics_revision import CODE_ROOT


BOOK_ROOT = CODE_ROOT.joinpath("hypothesis_testing_frost")


def test_descriptive_statistics_fuel_cost_p28() -> None:
    path = BOOK_ROOT.joinpath("FuelsCosts.csv")
    df = read_csv(path)
    X = df["Fuel Cost"]
    assert len(X) == 25
    assert isclose(mean(X), 330.6, atol=1e-1)
    assert isclose(sem(X), 30.8, atol=1e-1)
    assert isclose(std(X, ddof=1), 154.2, atol=1e-1)


def test_1_sample_t_test_example_p47() -> None:
    path = BOOK_ROOT.joinpath("AssessmentScores.csv")
    df = read_csv(path)
    X = df["Score"]
    res = ttest_1samp(X, 60)
    assert isclose(res.statistic, 1.42, atol=1e-2)
    assert isclose(res.pvalue, 0.178, atol=1e-3)
    assert (n := len(X)) == 15
    assert isclose(mu := mean(X), 64.16, atol=1e-2)
    assert isclose(sem_ := sem(X), 2.93, atol=1e-2)
    confidence = 0.95
    width = sem_ * t.ppf((1.0 + confidence) / 2, n - 1)
    ci_1 = mu - width, mu + width
    ci_2 = t.interval(confidence, n - 1, loc=mu, scale=sem_)
    ci_3 = DescrStatsW(X).tconfint_mean()
    for ci in (ci_1, ci_2, ci_3):
        assert isclose(ci, (57.87, 70.45), atol=1e-2).all()


def test_2_sample_t_test_example_p51() -> None:
    path = BOOK_ROOT.joinpath("t-TestExamples.csv")
    df = read_csv(path)
    X = df["Method A"]
    Y = df["Method B"]
    res = ttest_ind(X, Y)
    assert isclose(res.statistic, -4.08, atol=1e-2)
    assert isclose(res.pvalue, 0.0, atol=1e-3)
    centre = mean(X) - mean(Y)
    n_x, n_y = len(X), len(Y)
    v_x, v_y = var(X, ddof=1), var(Y, ddof=1)
    ddof = n_x + n_y - 2
    s = sqrt(((n_x - 1) * v_x + (n_y - 1) * v_y) / ddof)
    alpha = 0.05
    t_crit = t.ppf(1.0 - alpha / 2.0, ddof)  # t-critical value for 95% CI
    width = t_crit * sqrt(1.0 / n_x + 1 / n_y) * s
    ci = (centre - width, centre + width)
    assert isclose(ci, (-19.89, -6.59), atol=1e-2).all()


def test_paired_t_test_example_p51() -> None:
    path = BOOK_ROOT.joinpath("t-TestExamples.csv")
    df = read_csv(path)
    X = df["Pretest"]
    Y = df["Posttest"]
    res = ttest_rel(X, Y)
    assert isclose(res.statistic, -3.73, atol=1e-2)
    assert isclose(res.pvalue, 0.002, atol=1e-3)
    centre = mean(X) - mean(Y)
    n = len(X)
    ddof = n - 1
    alpha = 0.05
    t_crit = t.ppf(1.0 - alpha / 2.0, ddof)  # t-critical value for 95% CI
    width = t_crit * std(X - Y, ddof=1) / sqrt(n)
    ci = (centre - width, centre + width)
    assert isclose(ci, (-16.96, -4.59), atol=1e-2).all()
