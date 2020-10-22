from __future__ import annotations

from typing import Tuple

from numpy import mean
from numpy import ndarray
from numpy import sqrt
from numpy import var
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
