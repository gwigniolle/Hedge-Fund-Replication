import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, LassoLarsIC
from numba import jit
np.seterr(divide='ignore', invalid='ignore')


@jit(cache=True, nopython=True, nogil=True)
def std(x: np.ndarray):
    n, m = x.shape
    std = np.zeros(m)
    for i in np.arange(m):
        std[i] = np.std(x[:, i])
    return std


@jit(cache=True, nopython=True, nogil=True)
def nan_to_num(array):
    n, m = array.shape
    for i in np.arange(n):
        for j in np.arange(m):
            if np.isnan(array[i, j]): array[i, j] = 0.
    return array


@jit(cache=True, nopython=True, nogil=True)
def in_array(x: np.array, y):
    for z in x:
        if z == y: return True
    return False


def make_stats_maxence(df_price: pd.DataFrame):
    df_return = df_price.pct_change().dropna()
    stats.describe(df_return)
    t_tstat, p_tstat = stats.ttest_rel(df_return.iloc[:,0], df_return.iloc[:, 1])  # T-test
    t_KS, p_KS = stats.ks_2samp(df_return.iloc[:,0], df_return.iloc[:, 1])  # KS -> p petit pas la meme distrib
    tau, p_tau = stats.kendalltau(df_return.iloc[:,0], df_return.iloc[:, 1])  # Tau de Kendall

    return stats.describe(df_return), "t test: t = %g  p = %g" % (t_tstat, p_tstat), \
        "KS test: t = %g  p = %g" % (t_KS, p_KS), "KendallTau: t = %g  p = %g" % (tau, p_tau)


def replication_stats(df_price: pd.DataFrame, fund_name: str):
    rho = df_price.pct_change().corr(method="pearson")
    tau = df_price.pct_change().corr(method="kendall")
    returns_track = df_price.pct_change().dropna()
    returns_fund = df_price[fund_name].pct_change().dropna()
    df = pd.DataFrame()
    df['Tracking error'] = (returns_track.T - returns_fund.values).std(axis=1)
    df['R-squared'] = 1 - (returns_track.T - returns_fund.values).var(axis=1) / returns_fund.values.var()
    df['Sharpe ratio'] = np.sqrt(252) * returns_track.mean() / returns_track.std()
    df['Annual Return'] = (df_price.iloc[-1] / df_price.iloc[0]) ** (252 / len(df_price.index)) - 1
    df['Correlation'] = rho[fund_name]
    df['Kendall tau'] = tau[fund_name]
    return df


def make_FXHedge(df_price: pd.DataFrame, df_fx: pd.Series):
    dates = df_price.loc[df_fx.index[0]:].index
    df_fx = df_fx.reindex(dates).ffill()
    price = df_price.loc[dates].values
    fx = df_fx.values
    fx_hedge = make_FXHedge_jit(price, fx)
    return pd.DataFrame(index=dates, columns=df_price.columns, data=fx_hedge)


@jit(cache=True, nopython=True, nogil=True)
def make_FXHedge_jit(price, fx):
    n, m = price.shape
    fx_hedge = np.ones((n, m))
    for i in range(1, n):
        fx_hedge[i] = fx_hedge[i-1] * (1 + price[i] * fx[i] / (price[i-1] * fx[i-1]) - fx[i] / fx[i-1])
    return fx_hedge
        

def make_ER(df_price: pd.DataFrame, df_rate: pd.Series):
    dates = df_price.loc[df_rate.index[0]:].index
    df_rate = df_rate.reindex(dates).ffill()
    price = df_price.loc[dates].values
    rate = df_rate.values
    day_count = (dates[1:] - dates[:-1]) / np.timedelta64(1, 'D')
    price_er = make_ER_jit(price, rate, day_count.values)
    return pd.DataFrame(index=dates, columns=df_price.columns, data=price_er)


@jit(cache=True, nopython=True, nogil=True)
def make_ER_jit(price, rate, day_count):
    n, m = price.shape
    price_er = np.ones((n, m))
    for i in range(1, n):
        price_er[i] = price_er[i-1] * (price[i] / price[i-1] - rate[i-1] * day_count[i] / 36000.)
    return price_er


def make_track(df_price: pd.DataFrame, df_weight: pd.DataFrame, tc=0., lag=1):
    if lag < 0: raise Exception("Cannot have negative lag")
    dates = df_price.loc[df_weight.index[0]:].index
    reweight_dates = df_weight.index
    price = df_price.loc[dates].values
    weights = df_weight.values
    if isinstance(tc, pd.DataFrame): tc = tc.values
    else: tc = tc * np.ones(len(df_price.columns))
    track = make_track_jit(price, np.array(dates.values, dtype='datetime64[D]'), weights,
                           np.array(reweight_dates.values, dtype='datetime64[D]'), tc, lag)
    return pd.DataFrame(index=dates, data=track, columns=['Track'])


@jit(cache=True, nopython=True, nogil=True)
def make_track_jit(price, dates, weights, reweight_dates, tc, lag):
    n = len(dates)
    shares = (weights[0] / price[0])
    cash = 1 - np.sum((shares * price[0]))  # add cash when weigh_sum <> 1 in ER
    track = np.ones(n)
    if lag > 0: reweight_count = 0
    else: reweight_count = 1
    for i in range(1, n):
        if in_array(reweight_dates, dates[i-lag]):
            track[i] = np.sum(shares * price[i]) + cash
            cost = track[i] * np.sum(tc * np.abs(weights[reweight_count] - (shares * price[i]) / track[i]))
            track[i] = track[i] - cost
            shares = weights[reweight_count] * track[i] / price[i]
            cash = track[i] - np.sum(shares * price[i])
            reweight_count += 1
        else: 
            track[i] = np.sum(shares * price[i]) + cash
    return track

# -------------------------------------------------------------------------------------------------------------------- #


def ols_regression(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int, vol_target=False,
                   vol_period=20):
    if vol_target and vol_period > sample_length:
        raise Exception("The period for vol_target cannot be longer than sample_length")
    dates = df_y.index.copy()
    n, _ = df_x.shape
    x = df_x.values
    y = df_y.values
    weights = ols_regression_jit(y, x, sample_length, frequency, vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[dates[i*frequency+sample_length-1] for i in range((n-sample_length)//frequency+1)])
    return df_weight.fillna(0)


@jit(cache=True, nopython=True, nogil=True)
def ols_regression_jit(y: np.ndarray, x: np.ndarray, sample_length: int, frequency: int, vol_target=False,
                       vol_period=20):
    n, m = x.shape
    weights = np.zeros(((n - sample_length)//frequency + 1, m))
    for i in range((n - sample_length)//frequency + 1):
        start = i*frequency
        end = i*frequency + sample_length - 1
        x_k = nan_to_num(x[start:end+1])
        y_k = nan_to_num(y[start:end+1])
        weight = np.linalg.solve(np.dot(x_k.T, x_k), np.dot(x_k.T, y_k))
        leverage = 1
        if vol_target:
            port_vol = np.std(y_k[-vol_period:, 0])
            repli_vol = np.std(np.dot(x_k[-vol_period:, :], weight[:, 0]))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight[:, 0].T
    return weights

# -------------------------------------------------------------------------------------------------------------------- #


def lasso_regression(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int, l=0., vol_target=False,
                     vol_period=20):
    if vol_target and vol_period > sample_length:
        raise Exception("The period for vol_target cannot be longer than sample_length")
    dates = df_y.index.copy()
    n, _ = df_x.shape
    x = df_x.values
    y = df_y.values
    weights = lasso_regression_jit(y, x, sample_length, frequency, l, vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[dates[i*frequency+sample_length-1] for i in range((n-sample_length)//frequency+1)])
    return df_weight.fillna(0)


@jit(cache=True, nogil=True)
def lasso_regression_jit(y: np.ndarray, x: np.ndarray, sample_length: int, frequency: int, l=0., vol_target=False,
                         vol_period=20):
    n, m = x.shape
    weights = np.zeros(((n - sample_length)//frequency + 1, m))
    for i in range((n - sample_length)//frequency + 1):
        start = i*frequency
        end = i*frequency + sample_length - 1
        x_k = x[start:end+1]
        y_k = y[start:end+1]
        std_x = std(x_k)
        std_y = np.std(y_k)
        x_k = nan_to_num(x_k / std_x)
        y_k = nan_to_num(y_k / std_y)
        las = Lasso(alpha=l / (2. * (std_y ** 2)), fit_intercept=False, normalize=False)
        las.fit(x_k, y_k)
        weight = las.coef_ * std_y / std_x
        leverage = 1
        if vol_target:
            port_vol = np.std(y_k[-vol_period:])
            repli_vol = np.std(np.dot(x_k[-vol_period:, :], weight))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight
    return weights

# -------------------------------------------------------------------------------------------------------------------- #


def lasso_regression_ic(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int, criterion: str,
                        plot_lambda=True, vol_target=False, vol_period=20):
    if vol_target and vol_period > sample_length:
        raise Exception("The period for vol_target cannot be longer than sample_length")
    dates = df_y.index.copy()
    n, _ = df_x.shape
    x = df_x.values
    y = df_y.values
    weights, lam = lasso_regression_ic_jit(y, x, sample_length, frequency, criterion, vol_target, vol_period)
    df_lambda = pd.DataFrame(columns=['$\lambda$'], data=lam,
                             index=[dates[i*frequency+sample_length-1] for i in range((n-sample_length)//frequency+1)])
    if plot_lambda:
        sns.set()
        df_lambda['$\lambda$'].plot(title="$\lambda$ parameter selected by the " + criterion.upper())
        plt.show()
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[dates[i*frequency+sample_length-1] for i in range((n-sample_length)//frequency+1)])
    return df_weight.fillna(0), df_lambda


@jit(cache=True, nogil=True)
def lasso_regression_ic_jit(y: np.ndarray, x: np.ndarray, sample_length: int, frequency: int, criterion: str,
                            vol_target=False, vol_period=20):
    n, m = x.shape
    weights = np.zeros(((n - sample_length)//frequency + 1, m))
    lam = np.zeros((n - sample_length)//frequency + 1)
    for i in range((n - sample_length)//frequency + 1):
        start = i*frequency
        end = i*frequency + sample_length - 1
        x_k = x[start:end+1]
        y_k = y[start:end+1]
        std_x = std(x_k)
        std_y = np.std(y_k)
        x_k = nan_to_num(x_k / std_x)
        y_k = nan_to_num(y_k / std_y)
        las = LassoLarsIC(criterion=criterion, fit_intercept=False, normalize=False)
        las.fit(x_k, np.ravel(y_k))
        weight = las.coef_ * std_y / std_x
        leverage = 1
        if vol_target:
            port_vol = np.std(y_k[-vol_period:])
            repli_vol = np.std(np.dot(x_k[-vol_period:, :], weight))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight
        lam[i] = 2. * las.alpha_ * (std_y ** 2)
    return weights, lam

# -------------------------------------------------------------------------------------------------------------------- #


def ridge_regression(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int, l=0.,
                     vol_target=False, vol_period=20):
    if vol_target and vol_period > sample_length:
        raise Exception("The period for vol_target cannot be longer than sample_length")
    dates = df_y.index.copy()
    n, _ = df_x.shape
    x = df_x.values
    y = df_y.values
    weights = ridge_regression_jit(y, x, sample_length, frequency, l, vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[dates[i*frequency+sample_length-1] for i in range((n-sample_length)//frequency+1)])
    return df_weight.fillna(0)


@jit(cache=True, nogil=True, nopython=True)
def ridge_regression_jit(y: np.ndarray, x: np.ndarray, sample_length: int, frequency: int, l=0., vol_target=False,
                         vol_period=20):
    n, m = x.shape
    weights = np.zeros(((n - sample_length)//frequency + 1, m))
    I = np.eye(m)
    for i in range((n - sample_length)//frequency + 1):
        start = i*frequency
        end = i*frequency + sample_length - 1
        x_k = x[start:end+1]
        y_k = y[start:end+1]
        std_x = std(x_k)
        std_y = np.std(y_k)
        x_k = nan_to_num(x_k / std_x)
        y_k = nan_to_num(y_k / std_y)
        l1 = l * sample_length / (np.float(m) * (std_y ** 2))
        weight = np.linalg.solve(np.dot(x_k.T, x_k) + l1 * I, np.dot(x_k.T, y_k))
        leverage = 1
        if vol_target:
            port_vol = np.std(y_k[-vol_period:, 0])
            repli_vol = np.std(np.dot(x_k[-vol_period:, :], weight[:, 0]))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight[:, 0].T
    return weights

# -------------------------------------------------------------------------------------------------------------------- #


def kalman_filter(df_y: pd.DataFrame, df_x: pd.DataFrame, frequency: int, sigma_weight: float, sigma_return: float,
                  weight_init=np.array([[0.]]), cov_init=np.array([[0.]]), vol_target=False, vol_period=20,
                  return_log_likelihood=False):
    if vol_target and vol_period < frequency:
        raise Exception("The period for vol_target cannot be shorter than frequency")
    if not vol_target:
        vol_period = frequency
    index = df_y.index.copy()
    n = len(index)
    x = df_x.values
    y = df_y.values
    weights, log_likelihood = kalman_filter_jit(y, x, frequency, sigma_weight, sigma_return, weight_init, cov_init,
                                                vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[index[vol_period + i*frequency - 1] for i in range((n-vol_period)//frequency + 1)])
    if return_log_likelihood: return log_likelihood
    else: return df_weight.fillna(0)


@jit(cache=True, nopython=True, nogil=True)
def kalman_filter_jit(y: np.ndarray, x: np.ndarray, frequency: int, sigma_weight: float, sigma_return: float,
                      weight_init=np.array([[0.]]), cov_init=np.array([[0.]]), vol_target=False, vol_period=20):
    if not vol_target:
        vol_period = frequency
    n, m = x.shape
    I = np.eye(m)
    In = np.eye(frequency)
    cov_weight = (sigma_weight ** 2) * I
    cov_return = (sigma_return ** 2) * In
    weights = np.zeros(((n - vol_period)//frequency + 1, m))
    if np.all(weight_init == 0.): weight_filter = np.zeros((m, 1))
    else: weight_filter = weight_init
    if np.all(cov_init == 0.): cov_filter = np.zeros((m, m))
    else: cov_filter = cov_init
    log_likelihood = 0
    for i in range((n - vol_period)//frequency + 1):
        start = vol_period + (i - 1) * frequency
        vol_start = i * frequency
        end = vol_period + i * frequency - 1
        x_k = nan_to_num(x[start:end+1])
        y_k = nan_to_num(y[start:end+1])
        cov_forecast = cov_filter + cov_weight
        temp = np.dot(cov_forecast, x_k.T)
        gamma = np.dot(x_k, temp) + cov_return
        K = np.linalg.solve(gamma.T, temp.T).T
        weight_filter = (weight_filter + np.dot(K, y_k - np.dot(x_k, weight_filter)))
        cov_filter = np.dot(I - np.dot(K, x_k), cov_forecast)
        leverage = 1
        if vol_target:
            port_vol = np.std(y[vol_start:end+1])
            repli_vol = np.std(np.dot(nan_to_num(x[vol_start:end+1]), weight_filter[:, 0]))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight_filter[:, 0].T
        log_likelihood += kalman_log_likelihood(gamma, x_k, y_k, weight_filter)
    return weights, log_likelihood

# -------------------------------------------------------------------------------------------------------------------- #


def kalman_with_selection(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int,
                          nu: float, nb_period: int, criterion: str, vol_target=False, vol_period=20):

    if nb_period > sample_length:
        raise Exception("nb_period cannot be longer than sample_length")
    if vol_target and vol_period > sample_length:
        raise Exception("The period for vol_target cannot be longer than sample_length")

    df_weight_lasso, _ = lasso_regression_ic(df_y, df_x, sample_length, frequency, criterion, plot_lambda=False)
    df_weight = pd.DataFrame(columns=df_x.columns)
    index = df_weight_lasso.index.copy()
    _, m = df_weight.shape

    for date in index:
        selection = df_weight_lasso.loc[date] != 0.0
        selection = list(selection[selection].index)
        if not selection:
            df_weight.loc[date, :] = 0
        else:
            i = df_x.index.get_loc(date)
            df_x_ = df_x[selection].iloc[i-nb_period+1: i+1].fillna(0)
            df_y_ = df_y.loc[df_x_.index]
            weight = np.array(df_weight_lasso.loc[date, selection].values).reshape(m, 1)
            kalman = kalman_filter(df_y=df_y_, df_x=df_x_, frequency=1, sigma_weight=1.,
                                   sigma_return=nu, weight_init=weight)
            weight = kalman.loc[date].values

            leverage = 1
            if vol_target:
                port_vol = np.std(df_y.iloc[i - vol_period + 1:i + 1].values)
                repli_vol = np.std(np.dot(df_x[selection].iloc[i - vol_period + 1:i + 1].fillna(0).values, weight))
                leverage = port_vol / repli_vol
            df_weight.loc[date, selection] = leverage * weight

    return df_weight.fillna(0.0)

# -------------------------------------------------------------------------------------------------------------------- #


def selective_kalman_filter(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int,
                            nu: float, criterion: str, vol_target=False, vol_period=20):
    if vol_target and vol_period < frequency:
        raise Exception("The period for vol_target cannot be shorter than frequency")
    index = df_y.index.copy()
    n = len(index)
    x = df_x.iloc[sample_length-vol_period+1:].values
    y = df_y.iloc[sample_length-vol_period+1:].values
    lasso_weights = lasso_regression_ic(df_y, df_x, sample_length, frequency, criterion, plot_lambda=False)[0].values
    weights = selective_kalman_filter_jit(y, x, frequency, nu, lasso_weights, vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[index[sample_length+i*frequency-1] for i in range((n-sample_length)//frequency+1)])
    return df_weight.fillna(0)


@jit(cache=True, nopython=True, nogil=True)
def selective_kalman_filter_jit(y: np.ndarray, x: np.ndarray, frequency: int, nu: float, lasso_weights: np.ndarray,
                                vol_target=False, vol_period=20):
    n, m = x.shape
    I = np.eye(m)
    In = np.eye(frequency)
    cov_weight = I
    cov_return = (nu ** 2) * In
    weights = np.zeros(((n - vol_period)//frequency + 1, m))
    weight_filter = np.zeros((m, 1))
    cov_filter = np.zeros((m, m))
    for i in range((n - vol_period)//frequency + 1):
        selection = np.diag(lasso_weights[i] != 0.0)
        start = vol_period + (i - 1) * frequency
        vol_start = i * frequency
        end = vol_period + i * frequency - 1
        x_k = np.dot(nan_to_num(x[start:end+1]), selection)
        y_k = nan_to_num(y[start:end+1])
        cov_forecast = cov_filter + cov_weight
        temp = np.dot(cov_forecast, x_k.T)
        gamma = np.dot(x_k, temp) + cov_return
        K = np.linalg.solve(gamma.T, temp.T).T
        weight_filter = (weight_filter + np.dot(K, y_k - np.dot(x_k, weight_filter)))
        cov_filter = np.dot(I - np.dot(K, x_k), cov_forecast)
        weight = np.dot(selection, weight_filter)
        leverage = 1
        if vol_target:
            port_vol = np.std(y[vol_start:end+1])
            repli_vol = np.std(np.dot(nan_to_num(x[vol_start:end+1]), weight_filter[:, 0]))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight[:, 0].T
    return weights

# -------------------------------------------------------------------------------------------------------------------- #


def ml_kalman_filter(df_y: pd.DataFrame, df_x: pd.DataFrame, frequency: int, tau: float, vol_target=False,
                     vol_period=20, plot_sigma=False):
    if vol_target and vol_period < frequency:
        raise Exception("The period for vol_target cannot be shorter than frequency")
    if not vol_target:
        vol_period = frequency
    index = df_y.index.copy()
    n = len(index)
    x = df_x.values
    y = df_y.values
    weights, sigma = ml_kalman_filter_jit(y, x, frequency, tau, vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[index[vol_period + i*frequency - 1] for i in range((n-vol_period)//frequency + 1)])
    df_sigma = pd.DataFrame(columns=[r"$\tilde{\sigma}_{\epsilon}$", r"$\tilde{\sigma}_{\eta}$",
                                     r"$\hat{\sigma}_{\epsilon}$", r"$\hat{\sigma}_{\eta}$"],
                            index=df_weight.index, data=sigma)
    df_sigma[r'$\hat{\nu}$'] = df_sigma[r"$\hat{\sigma}_{\epsilon}$"] / df_sigma[r"$\hat{\sigma}_{\eta}$"]
    if plot_sigma:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.set()
        df_sigma[[r"$\tilde{\sigma}_{\epsilon}$", r"$\hat{\sigma}_{\epsilon}$"]].plot(ax=ax1, figsize=(16, 6), logy=True)
        df_sigma[[r"$\tilde{\sigma}_{\eta}$", r"$\hat{\sigma}_{\eta}$"]].plot(ax=ax2, figsize=(16, 6), logy=True)
        plt.show()
        df_sigma[r'$\hat{\nu}$'].plot(figsize=(16, 8), logy=True)
        plt.show()
    return df_weight.fillna(0), df_sigma


@jit(cache=True)
def ml_kalman_filter_jit(y: np.ndarray, x: np.ndarray, frequency: int, tau: float, vol_target=False, vol_period=20):
    n, m = x.shape
    I = np.eye(m)
    In = np.eye(frequency)
    weights = np.zeros(((n - vol_period)//frequency + 1, m))
    sigma = np.zeros(((n - vol_period)//frequency + 1, 4))
    weight_filter = np.zeros((m, 1))
    cov_filter = np.zeros((m, m))
    sigma_weight = 1.
    sigma_return = 1.
    for i in range((n - vol_period)//frequency + 1):
        start = vol_period + (i - 1) * frequency
        vol_start = i * frequency
        end = vol_period + i * frequency - 1
        x_k = nan_to_num(x[start:end+1])
        y_k = nan_to_num(y[start:end+1])
        theta = max_likelihoog_estimator(sigma_return, sigma_weight, x_k, y_k, cov_filter, weight_filter)
        if sigma_weight == 1. and sigma_return == 1.:
            sigma_weight = theta[1]
            sigma_return = theta[0]
        else:
            sigma_weight = tau * theta[1] + (1 - tau) * sigma_weight
            sigma_return = tau * theta[0] + (1 - tau) * sigma_return
        sigma[i] = [theta[0], theta[1], sigma_return, sigma_weight]
        cov_weight = (sigma_weight ** 2) * I
        cov_return = (sigma_return ** 2) * In
        cov_forecast = cov_filter + cov_weight
        temp = np.dot(cov_forecast, x_k.T)
        gamma = np.dot(x_k, temp) + cov_return
        K = np.linalg.solve(gamma.T, temp.T).T
        weight_filter = (weight_filter + np.dot(K, y_k - np.dot(x_k, weight_filter)))
        cov_filter = np.dot(I - np.dot(K, x_k), cov_forecast)
        leverage = 1
        if vol_target:
            port_vol = np.std(y[vol_start:end+1])
            repli_vol = np.std(np.dot(nan_to_num(x[vol_start:end+1]), weight_filter[:, 0]))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight_filter[:, 0].T
    return weights, sigma

# -------------------------------------------------------------------------------------------------------------------- #


def ml_kalman_filter2(df_y: pd.DataFrame, df_x: pd.DataFrame, frequency: int, mle_period: int,
                      vol_target=False, vol_period=20, plot_sigma=False):
    if vol_target and vol_period < frequency:
        raise Exception("The period for vol_target cannot be shorter than frequency")
    if not vol_target:
        vol_period = frequency
    index = df_y.index.copy()
    n = len(index)
    x = df_x.values
    y = df_y.values
    weights, sigma = ml_kalman_filter2_jit(y, x, frequency, mle_period, vol_target, vol_period)
    df_weight = pd.DataFrame(columns=df_x.columns, data=weights,
                             index=[index[vol_period + i*frequency - 1] for i in range((n-vol_period)//frequency + 1)])
    df_sigma = pd.DataFrame(columns=[r"$\hat{\sigma}_{\epsilon}$", r"$\hat{\sigma}_{\eta}$"],
                            index=df_weight.index, data=sigma)
    df_sigma[r'$\hat{\nu}$'] = df_sigma[r"$\hat{\sigma}_{\epsilon}$"] / df_sigma[r"$\hat{\sigma}_{\eta}$"]
    if plot_sigma:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.set()
        df_sigma[[r"$\hat{\sigma}_{\epsilon}$",
                  r"$\hat{\sigma}_{\eta}$"]].plot(ax=ax1, figsize=(16, 6), secondary_y=[r"$\hat{\sigma}_{\epsilon}$"])
        df_sigma[[r'$\hat{\nu}$']].plot(ax=ax2, figsize=(16, 6))
        plt.show()
    return df_weight.fillna(0), df_sigma


@jit(cache=True)
def ml_kalman_filter2_jit(y: np.ndarray, x: np.ndarray, frequency: int, mle_period: int, vol_target=False,
                         vol_period=20):
    n, m = x.shape
    I = np.eye(m)
    In = np.eye(frequency)
    weights = np.zeros(((n - vol_period)//frequency + 1, m))
    sigma = np.zeros(((n - vol_period)//frequency + 1, 2))
    weight_filter = np.zeros((m, 1))
    cov_filter = np.zeros((m, m))
    sigma_weight = 1.
    sigma_return = 1.
    weight_list = [weight_filter]
    cov_list = [cov_filter]
    for i in range((n - vol_period)//frequency + 1):
        start = vol_period + (i - 1) * frequency
        vol_start = i * frequency
        end = vol_period + i * frequency - 1
        x_k = nan_to_num(x[start:end+1])
        y_k = nan_to_num(y[start:end+1])
        if len(weight_list) >= mle_period and len(cov_list) >= mle_period:
            weight_mle_start = weight_list.pop(0)
            cov_mle_start = cov_list.pop(0)
        else:
            weight_mle_start = weight_list[0]
            cov_mle_start = cov_list[0]
        x_mle = nan_to_num(x[vol_period+max((i-mle_period, -1))*frequency:vol_period+i*frequency])
        y_mle = nan_to_num(y[vol_period+max((i-mle_period, -1))*frequency:vol_period+i*frequency])
        theta = max_likelihoog_estimator2(sigma_return, sigma_weight, x_mle, y_mle, frequency, weight_mle_start,
                                          cov_mle_start)
        sigma_weight = theta[1]
        sigma_return = theta[0]
        sigma[i] = np.array((sigma_return, sigma_weight))
        cov_weight = (sigma_weight ** 2) * I
        cov_return = (sigma_return ** 2) * In
        cov_forecast = cov_filter + cov_weight
        temp = np.dot(cov_forecast, x_k.T)
        gamma = np.dot(x_k, temp) + cov_return
        K = np.linalg.solve(gamma.T, temp.T).T
        weight_filter = (weight_filter + np.dot(K, y_k - np.dot(x_k, weight_filter)))
        cov_filter = np.dot(I - np.dot(K, x_k), cov_forecast)
        leverage = 1
        if vol_target:
            port_vol = np.std(y[vol_start:end+1])
            repli_vol = np.std(np.dot(nan_to_num(x[vol_start:end+1]), weight_filter[:, 0]))
            leverage = port_vol / repli_vol
        weights[i] = leverage * weight_filter[:, 0].T
    return weights, sigma

# -------------------------------------------------------------------------------------------------------------------- #

@jit(cache=True)
def max_likelihoog_estimator(sigma_return: np.float32, sigma_weight: np.float32, x: np.ndarray, y: np.ndarray,
                             cov_filter: np.ndarray, weight_filter: np.ndarray):
    res = minimize(fun_1, np.log10(np.array([sigma_return, sigma_weight])), method='Nelder-Mead',
                   args=(x, y, cov_filter, weight_filter))
    return 10 ** res.x


@jit(cache=True, nopython=True, nogil=True)
def fun_1(theta, x, y, cov_filter, weight_filter):
    n, p = x.shape
    In = np.eye(n)
    Ip = np.eye(p)
    theta = theta.reshape(2, )
    sigma_r = 10 ** theta[0]
    sigma_w = 10 ** theta[1]
    gamma = np.dot(np.dot(x, cov_filter + (sigma_w ** 2) * Ip), x.T) + (sigma_r ** 2) * In
    _, logdet = np.linalg.slogdet(gamma)
    pred_return = np.dot(x, weight_filter)
    error = y - pred_return
    temp = np.linalg.solve(gamma, error)
    return (logdet + np.dot(error.T, temp))[0]


@jit(cache=True, nopython=True, nogil=True)
def kalman_log_likelihood(gamma, x, y, weight_filter):
    _, logdet = np.linalg.slogdet(gamma)
    pred_return = np.dot(x, weight_filter)
    error = y - pred_return
    temp = np.linalg.solve(gamma, error)
    return (logdet + np.dot(error.T, temp))[0, 0]


@jit(cache=True, nopython=True, nogil=True)
def fun_2(theta, x, y, frequency, weight_mle_start, cov_mle_start):
    theta = theta.reshape(2,)
    _, likelihood = kalman_filter_jit(y, x, frequency, 10 ** theta[1], 10 ** theta[0], cov_init=cov_mle_start,
                                      weight_init=weight_mle_start)
    return likelihood


@jit(cache=True)
def max_likelihoog_estimator2(sigma_return, sigma_weight, x, y, frequency, weight_mle_start, cov_mle_start):
    options = {'maxiter': 100, 'gtol': 1e-5, 'eps': 1, 'ftol': 1e-3, 'maxfun': 100}
    bounds = ((-10, 10), (-10, 10))
    res = minimize(fun_2, np.log10(np.array((sigma_return, sigma_weight))), method='L-BFGS-B',
                   options=options, args=(x, y, frequency, weight_mle_start, cov_mle_start), bounds=bounds)
    return 10 ** res.x

