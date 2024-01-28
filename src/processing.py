import scipy
import numpy as np
import pandas as pd
from datetime import datetime


def x_get_ytm_newton(f, p, c, T, guess=.05, max_iter=2_000) -> (float, float):
    """
    Newton's method solver for YTM
    :param f: face value
    :param p: price
    :param c: coupon
    :param T: number of future periods
    :param guess: ytm guess
    :param max_iter: maximum iterations
    :return: (ytm, residual)
    """
    get_ytm = lambda y: f * (1 + y) ** (-T) + c * (1 - (1 + y) ** (-T)) / y - p
    ytm = scipy.optimize.newton(get_ytm, guess, maxiter=max_iter)
    return ytm, get_ytm(ytm)


def srs_get_ytm(srs: pd.Series, maturity_date: datetime, coupon: float,
                periodicity: str = 'y', fac_v_scale: float = 1.0, ytm_guess: float = .025) -> np.array:
    """
    Obtain YTM for a pd.Series of prices with datetime index
    :param srs: series of prices with datetime index
    :param maturity_date:
    :param coupon:
    :param periodicity: frequency coupon payments are made in
    :param fac_v_scale: bonds are priced as a percentage of their face value, 100 -> bond trades at par
                        function assumes par trading == 1, if par trading == 100, then scale = 100
    :param ytm_guess:
    :return: pd.Series of YTMs
    """
    dict_period = {'y': 30 * 12, '6m': 30 * 6, 'm': 30}
    assert periodicity in list(dict_period.keys()), f"please specify periodicity as either {list(dict_period.keys())}"
    assert isinstance(srs.index, pd.DatetimeIndex), "please specify datetime index"

    out, residuals = [], []
    for idx, val in pd.DataFrame(srs).iterrows():

        T = int((maturity_date - idx).days / (30 * 12))

        try:
            ymt, resid = x_get_ytm_newton(f=fac_v_scale, p=val.values, c=coupon, T=T, guess=ytm_guess)
        except RuntimeError as e:
            print(fac_v_scale, val.values, coupon, T, ytm_guess, e)
            continue

        out.append(ymt[0])
        residuals.append(resid)

    stat = scipy.stats.describe(residuals)
    print(f"Overall solver residuals: mean {stat.mean}, std: {np.sqrt(stat.variance)}")

    return np.array(out)


from scipy.stats import norm, halfcauchy, beta, gamma


def get_aic(arr, fitted_dist):
    """
    Calculates AIC of a fitted distribution
    :param arr:
    :param fitted_dist:
    :return:
    """
    ll = np.log(np.product(fitted_dist.pdf(arr)))
    aic = 2 * len(arr) - 2 * np.prod(ll)
    return aic


def get_fitted_dist(arr: np.array, dists: list) -> object:
    """
    Fits and compares distribution according to AIC
    Currrently supports normal and beta
    :param arr:
    :return: frozen scipy distribution
    """
    dict_dists = {'norm': norm, 'gamma': gamma, 'beta': beta, 'halfcauchy': halfcauchy}
    for dist in dists:
        assert dist in dict_dists.keys(), f"{dist} not specified in function"

    fitted_dists = []
    for dist in dists:
        fitted_dists.append(
            dict_dists[dist](*dict_dists[dist].fit(arr))
        )
    aic = []
    for f_dist in fitted_dists:
        aic.append(get_aic(arr, f_dist))

    # returns dist with min aic
    return fitted_dists[aic.index(min(aic))]


def srs_apply_impute_data(srs: pd.Series) -> pd.Series:
    """
    Imputes missing data as a draw from a fitted distribution on existing data
    :param srs:
    :return:
    """
    arr = srs.dropna().values

    # assert len(arr) / srs.isna().sum() > .02, "Low data imputation basis"
    assert len(arr) > 20, "Few data points for distribution estimate"

    srs.loc[srs.isna()] = get_fitted_dist(arr, dists=['gamma', 'norm']).rvs(srs.isna().sum())
    return srs
