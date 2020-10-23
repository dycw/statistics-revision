from __future__ import annotations

from holoviews import Curve
from holoviews import Overlay
from holoviews import Scatter
from holoviews import Slope
from numpy import isclose
from numpy import linspace
from pandas import DataFrame
from pandas import read_csv
from pandas import Series
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


def test_regression_model_plot_p43() -> None:
    path = _BOOK_ROOT.joinpath("HeightWeight.csv")
    df = add_constant(read_csv(path))
    X, Y = df[["Height M"]], df["Weight kg"]
    model = OLS(Y, add_constant(X)).fit()
    scatter = Scatter((X, Y)).opts(size=10)
    params = model.params
    slope = Slope(
        slope=params.loc["Height M"],
        y_intercept=params.loc["const"],
    ).opts(color="orange")
    plot = scatter * slope
    assert isinstance(plot, Overlay)


def test_regression_model_values_p52() -> None:
    path = _BOOK_ROOT.joinpath("HeightWeight.csv")
    df = read_csv(path)
    X = add_constant(df[["Height M"]])
    Y = df["Weight kg"]
    model = OLS(Y, X).fit()
    assert isinstance(model.summary(), Summary)
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues
    assert isclose(params.loc["const"], -114.326, atol=1e-3)
    assert isclose(bse.loc["const"], 17.4425, atol=1e-4)
    assert isclose(tvalues.loc["const"], -6.554444, atol=1e-5)
    assert isclose(pvalues.loc["const"], 0.0, atol=1e-3)
    assert isclose(params.loc["Height M"], 106.505, atol=1e-3)
    assert isclose(bse.loc["Height M"], 11.5500, atol=1e-4)
    assert isclose(tvalues.loc["Height M"], 9.221177, atol=1e-5)
    assert isclose(pvalues.loc["Height M"], 0.0, atol=1e-3)
    ci = model.conf_int()
    assert isclose(ci.loc["const", 0], -149.0, atol=1e-1)
    assert isclose(ci.loc["const", 1], -79.7, atol=1e-1)
    assert isclose(ci.loc["Height M", 0], 83.5, atol=1e-1)
    assert isclose(ci.loc["Height M", 1], 129.5, atol=1e-1)


def test_regression_model_curvature_p86() -> None:
    path = _BOOK_ROOT.joinpath("Hardness.csv")
    df = read_csv(path)
    X = add_constant(df[["Temp", "Pressure"]])
    X["Pressure*Pressure"] = X["Pressure"] ** 2
    Y = df["Hardness"]
    params = OLS(Y, X).fit().params
    assert isclose(params.loc["const"], -38.8, atol=1e-1)
    assert isclose(params.loc["Temp"], 0.759, atol=1e-3)
    assert isclose(params.loc["Pressure"], -1.6, atol=1e-2)
    assert isclose(params.loc["Pressure*Pressure"], 0.1657, atol=1e-4)


def test_regression_model_curvature_plot_p96() -> None:
    path = _BOOK_ROOT.joinpath("CurveFittingExample.csv")
    df = read_csv(path)

    def build_X(series: Series) -> DataFrame:
        X = add_constant(series.rename("Input"))
        X["Input^2"] = X["Input"] ** 2
        return X

    X, Y = df["Input"], df["Output"]
    model = OLS(Y, build_X(X)).fit()
    build_X(df["Input"])
    scatter = Scatter((X, Y))
    X_pred = Series(linspace(X.min(), X.max()))
    Y_pred = model.predict(build_X(X_pred))
    curve = Curve((X_pred, Y_pred))
    plot = scatter * curve
    assert isinstance(plot, Overlay)
