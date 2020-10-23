from __future__ import annotations

from numpy import isclose
from pandas import read_csv
from scipy.stats import pearsonr

from statistics_revision import CODE_ROOT


_BOOK_ROOT = CODE_ROOT.joinpath("regression_analysis_frost")


def test_correlation_p16() -> None:
    path = _BOOK_ROOT.joinpath("HeightWeight.csv")
    df = read_csv(path)
    X, Y = df["Height M"], df["Weight kg"]
    corr, _ = pearsonr(X, Y)
    assert isclose(corr, 0.705, atol=1e-1)
