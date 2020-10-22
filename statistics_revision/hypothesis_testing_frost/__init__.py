from __future__ import annotations

from typing import Tuple

from numpy import mean
from numpy import ndarray
from numpy import sqrt
from numpy import var
from scipy.stats import norm
from scipy.stats import t


def ttest_ind_ci(
    X: ndarray,
    Y: ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    centre = mean(X) - mean(Y)
    n_x, n_y = len(X), len(Y)
    v_x, v_y = var(X, ddof=1), var(Y, ddof=1)
    ddof = n_x + n_y - 2
    width = (
        t.ppf(1.0 - alpha / 2.0, ddof)
        * sqrt(
            1.0 / n_x + 1 / n_y,
        )
        * sqrt(((n_x - 1) * v_x + (n_y - 1) * v_y) / ddof)
    )
    return centre - width, centre + width


def two_proportions_test(
    success_a: int,
    size_a: int,
    success_b: int,
    size_b: int,
) -> Tuple[float, float]:
    """
    A/B test for two proportions;
    given a success a trial size of group A and B compute
    its zscore and pvalue

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    Returns
    -------
    zscore : float
        test statistic for the two proportion z-test

    pvalue : float
        p-value for the two proportion z-test
    """
    # https://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html

    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    zscore = abs(prop_b - prop_a) / sqrt(var)
    one_side = 1 - norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return zscore, pvalue
