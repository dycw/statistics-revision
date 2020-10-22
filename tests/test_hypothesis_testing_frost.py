from __future__ import annotations

from numpy import isclose
from numpy import mean
from numpy import sqrt
from numpy import std
from pandas import read_csv
from scipy.stats import f_oneway
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

from statistics_revision import CODE_ROOT
from statistics_revision.hypothesis_testing_frost import ttest_ind_ci

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
    assert isclose(mu := mean(X), 64.16, atol=1e-2)
    assert isclose(sem_ := sem(X), 2.93, atol=1e-2)
    n = len(X)
    ddof = n - 1
    assert isclose(sem_, std(X, ddof=1) / sqrt(n))
    alpha = 0.05
    width = t.ppf(1.0 - alpha / 2.0, ddof) * sem_
    ci = (mu - width, mu + width)
    assert isclose(ci, (57.87, 70.45), atol=1e-2).all()


def test_2_sample_t_test_example_p51() -> None:
    path = BOOK_ROOT.joinpath("t-TestExamples.csv")
    df = read_csv(path)
    X = df["Method A"]
    Y = df["Method B"]
    res = ttest_ind(X, Y)
    assert isclose(res.statistic, -4.08, atol=1e-2)
    assert isclose(res.pvalue, 0.0, atol=1e-3)
    assert isclose(ttest_ind_ci(X, Y), (-19.89, -6.59), atol=1e-2).all()


def test_paired_t_test_example_p55() -> None:
    path = BOOK_ROOT.joinpath("t-TestExamples.csv")
    df = read_csv(path)
    X = df["Pretest"]
    Y = df["Posttest"]
    res = ttest_rel(X, Y)
    assert isclose(res.statistic, -3.73, atol=1e-2)
    assert isclose(res.pvalue, 0.002, atol=1e-3)
    centre = mean(X) - mean(Y)
    n = len(X)
    alpha = 0.05
    width = t.ppf(1.0 - alpha / 2.0, n - 1) * std(X - Y, ddof=1) / sqrt(n)
    ci = (centre - width, centre + width)
    assert isclose(ci, (-16.96, -4.59), atol=1e-2).all()


def test_two_sample_t_ttest_and_ci_p66() -> None:
    path = BOOK_ROOT.joinpath("DifferenceMeans.csv")
    df = read_csv(path)
    X = df["Strength B"]
    Y = df["Strength A"]
    res = ttest_ind(X, Y)
    assert isclose(res.statistic, 2.09, atol=1e-2)
    assert isclose(res.pvalue, 0.044, atol=1e-3)
    assert isclose(ttest_ind_ci(X, Y), (0.06, 4.23), atol=1e-2).all()


def test_1_sample_t_test_statistic_p70() -> None:
    path = BOOK_ROOT.joinpath("FuelsCosts.csv")
    df = read_csv(path)
    X = df["Fuel Cost"]
    stat_1 = ttest_1samp(X, 60).statistic
    stat_2 = (mean(X) - 60) / (std(X, ddof=1) / sqrt(len(X)))
    assert isclose(stat_1, stat_2)


def test_good_side_of_high_p_values_p97() -> None:
    path = BOOK_ROOT.joinpath("studies.csv")
    df = read_csv(path)
    df_1 = df[["S1 Method A", "S1 Method B"]].dropna()
    sr_1A, sr_1B = df_1["S1 Method A"], df_1["S1 Method B"]
    assert isclose(mean(sr_1A - sr_1B), 6.01, atol=1e-2)
    assert isclose(ttest_ind(sr_1A, sr_1B).pvalue, 0.12, atol=1e-2)
    df_2 = df[["S2 Method A", "S2 Method B"]].dropna()
    sr_2A, sr_2B = df_2["S2 Method A"], df_2["S2 Method B"]
    assert isclose(mean(sr_2A - sr_2B), 9.97, atol=1e-2)
    assert isclose(ttest_ind(sr_2A, sr_2B).pvalue, 0.14, atol=1e-2)
    df_3 = df[["S3 Method A", "S3 Method B"]].dropna()
    sr_3A, sr_3B = df_3["S3 Method A"], df_3["S3 Method B"]
    assert isclose(mean(sr_3A - sr_3B), 1.94, atol=1e-2)
    assert isclose(ttest_ind(sr_3A, sr_3B).pvalue, 0.042, atol=1e-3)


def test_one_way_anova_p200() -> None:
    path = BOOK_ROOT.joinpath("OneWayExample.csv")
    df = read_csv(path)
    df = df.assign(n=df.index % 10).pivot(
        index="n",
        columns="Sample",
        values="Strength",
    )
    a, b, c, d = (col for _, col in df.items())
    result = f_oneway(a, b, c, d)
    assert isclose(result.statistic, 3.30, atol=1e-2)
    assert isclose(result.pvalue, 0.031, atol=1e-3)
