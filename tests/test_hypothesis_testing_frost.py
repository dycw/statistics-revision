from __future__ import annotations

from numpy import mean
from numpy import std
from pandas import read_csv
from scipy.stats import sem

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
