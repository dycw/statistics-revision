from __future__ import annotations

from holoviews import Overlay
from holoviews import Scatter
from holoviews import Slope
from numpy import isclose
from pandas import read_csv
from scipy.stats import pearsonr
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from statistics_revision import CODE_ROOT


_BOOK_ROOT = CODE_ROOT.joinpath("regression_analysis_frost")


def test_correlation_p16() -> None:
    path = _BOOK_ROOT.joinpath("HeightWeight.csv")
    df = read_csv(path)
    X, Y = df["Height M"], df["Weight kg"]
    corr, _ = pearsonr(X, Y)
    assert isclose(corr, 0.705, atol=1e-1)


def test_regression_model_and_plot_p43() -> None:
    path = _BOOK_ROOT.joinpath("HeightWeight.csv")
    df = add_constant(read_csv(path))
    X, Y = df[["const", "Height M"]], df["Weight kg"]
    model = OLS(Y, X).fit()
    assert isinstance(model.summary(), Summary)
    X = df["Height M"]
    scatter = Scatter((X, Y)).opts(size=10)
    params = model.params
    slope = Slope(
        slope=params.loc["Height M"],
        y_intercept=params.loc["const"],
    ).opts(color="orange")
    plot = scatter * slope
    assert isinstance(plot, Overlay)
