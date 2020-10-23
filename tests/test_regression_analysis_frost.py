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
    assert isclose(model.params.loc["const"], -114.326, atol=1e-3)
    assert isclose(model.bse.loc["const"], 17.4425, atol=1e-4)
    assert isclose(model.tvalues.loc["const"], -6.554444, atol=1e-5)
    assert isclose(model.pvalues.loc["const"], 0.0, atol=1e-3)
    assert isclose(model.params.loc["Height M"], 106.505, atol=1e-3)
    assert isclose(model.bse.loc["Height M"], 11.5500, atol=1e-4)
    assert isclose(model.tvalues.loc["Height M"], 9.221177, atol=1e-5)
    assert isclose(model.pvalues.loc["Height M"], 0.0, atol=1e-3)
    X = df["Height M"]
    scatter = Scatter((X, Y)).opts(size=10)
    slope = Slope(
        slope=model.params.loc["Height M"],
        y_intercept=model.params.loc["const"],
    ).opts(color="orange")
    plot = scatter * slope
    assert isinstance(plot, Overlay)
