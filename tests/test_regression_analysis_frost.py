from __future__ import annotations

from typing import Tuple
from typing import Union

from holoviews import Curve
from holoviews import Overlay
from holoviews import Scatter
from holoviews import Slope
from numpy import array
from numpy import concatenate
from numpy import dot
from numpy import isclose
from numpy import linspace
from numpy import ndarray
from numpy import zeros
from pandas import DataFrame
from pandas import read_csv
from pandas import Series
from scipy.optimize import least_squares
from scipy.stats import pearsonr
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
    params = model.params
    assert isclose(params.loc["const"], 3.241, atol=1e-3)
    assert isclose(params.loc["Input"], 3.564, atol=1e-3)
    assert isclose(params.loc["Input^2"], -0.1915, atol=1e-4)
    build_X(df["Input"])
    scatter = Scatter((X, Y))
    X_pred = Series(linspace(X.min(), X.max()))
    Y_pred = model.predict(build_X(X_pred))
    curve = Curve((X_pred, Y_pred)).opts(color="orange")
    plot = scatter * curve
    assert isinstance(plot, Overlay)


def test_non_linear_regression_p108() -> None:
    path = _BOOK_ROOT.joinpath("ElectronMobility.csv")
    df = read_csv(path)

    def model(beta: ndarray, X: Union[float, Series]) -> Union[float, Series]:
        assert len(beta) == 7
        poly = array([1.0, X, X ** 2, X ** 3], dtype=object)
        return dot(beta[:4], poly) / dot(concatenate([[1.0], beta[-3:]]), poly)

    def target(beta: ndarray, X: float, y: float) -> float:
        return abs(model(beta, X) - y)

    beta_0 = zeros(7)
    X, Y = df["Density Ln"], df["Mobility"]
    result = least_squares(target, beta_0, args=(X, Y))
    assert result.success
    assert isclose(
        result.x,
        [
            1289.27,
            1717.88,
            747.22,
            108.12,
            1.12,
            0.48,
            0.09,
        ],
        atol=1e-2,
    ).all()


def test_variance_inflation_factor_p225() -> None:
    path = _BOOK_ROOT.joinpath("MulticollinearityExample.csv")
    df = read_csv(path)

    def build_X_Y(df: DataFrame) -> Tuple[DataFrame, Series]:
        X = add_constant(df[["%Fat", "Weight kg", "Activity"]])
        X["%Fat*Weight kg"] = X["%Fat"] * X["Weight kg"]
        Y = df["Femoral Neck"]
        return X, Y

    X, Y = build_X_Y(df)
    model = OLS(Y, X).fit()
    params = model.params
    assert isclose(params.loc["const"], 0.155, atol=1e-3)
    assert isclose(params.loc["%Fat"], 0.005577, atol=1e-5)
    assert isclose(params.loc["Weight kg"], 0.01447, atol=1e-5)
    assert isclose(params.loc["Activity"], 0.000022, atol=1e-6)
    assert isclose(params.loc["%Fat*Weight kg"], -0.000214, atol=1e-6)
    X_array = X.to_numpy()
    assert isclose(variance_inflation_factor(X_array, 1), 14.93, atol=1e-2)
    assert isclose(variance_inflation_factor(X_array, 2), 33.95, atol=1e-2)
    assert isclose(variance_inflation_factor(X_array, 3), 1.05, atol=1e-2)
    assert isclose(variance_inflation_factor(X_array, 4), 75.06, atol=1e-2)

    centered = df.sub(df.mean(axis=0), axis=1)
    X, _ = build_X_Y(centered)
    X_array = X.to_numpy()
    assert isclose(variance_inflation_factor(X_array, 1), 3.32, atol=1e-2)
    assert isclose(variance_inflation_factor(X_array, 2), 4.75, atol=1e-2)
    assert isclose(variance_inflation_factor(X_array, 3), 1.05, atol=1e-2)
    assert isclose(variance_inflation_factor(X_array, 4), 1.99, atol=1e-2)
