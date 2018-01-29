import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats


def make_stats(df_price):
    df_return = df_price.pct_change().dropna()
    stats.describe(df_return)
    t_tstat, p_tstat = stats.ttest_rel(df_return.iloc[:,0], df_return.iloc[:, 1])  # T-test
    t_KS, p_KS = stats.ks_2samp(df_return.iloc[:,0], df_return.iloc[:, 1])  # KS -> p petit pas la meme distri
    tau, p_tau = stats.kendalltau(df_return.iloc[:,0], df_return.iloc[:, 1])  # Tau de Kendall

    return stats.describe(df_return),"t test: t = %g  p = %g" % (t_tstat, p_tstat), \
           "KS test: t = %g  p = %g" % (t_KS, p_KS), "KendallTau: t = %g  p = %g" % (tau, p_tau)


def make_FXHedge(df_price, df_fx):
        dates = df_price.loc[df_fx.index[0]:].index
        price_fx_hedge = pd.DataFrame(index=dates, columns=df_price.columns)
        df_fx = df_fx.reindex(dates).ffill()
        n = len(dates)
        price_fx_hedge.iloc[0] = 1.

        for i in range(1, n):
            price_fx_hedge.iloc[i] = price_fx_hedge.iloc[i-1] * (1 + df_price.iloc[i] * df_fx.iloc[i] / (df_price.iloc[i-1] * df_fx.iloc[i-1]) -
                                                       df_fx.iloc[i]/df_fx.iloc[i-1]) 
        return price_fx_hedge
        

def make_ER(price, rate):

    """
    :param price: pd.DataFrame containing the prices of the ticker
    :param rate: pd.Series containing the rate prices
    :return:
    """

    dates = price.loc[rate.index[0]:].index
    price_ER = pd.DataFrame(index=dates, columns=price.columns)
    price_ER.iloc[0] = 1.
    n = len(dates)

    rate = rate.reindex(dates).ffill()

    for i in range(1, n):
        price_ER.iloc[i] = price_ER.iloc[i-1] * (price.iloc[i] / price.iloc[i-1]
                                                 - rate.loc[dates[i-1]] * (dates[i] - dates[i-1]).days/ 36000.)

    return price_ER


def make_track(df_price, df_weight, tc=0):
    """
    :param df_price: a dataframe containing the prices of the underlyings used in the index, columns must be the names
    and the index are the dates
    :param df_weight: a dataframe containing the weight on the rebalancing dates of the track created
    :param tc: transaction cost, default is 0
    :return: a pandas series containing the track made from the composition in df_weight
    """

    index = df_price.index
    reweight_index = df_weight.index

    n = len(index)
    shares = (df_weight / df_price).iloc[0]
    cash = 1 - (shares * df_price.iloc[0]).sum() # add cash when weigh_sum <> 1 in ER
    value = np.ones(n)

    for i in range(1, len(index)):

        if index[i-1] in reweight_index:
            cost = tc * value[i-1] * np.abs(df_weight.loc[index[i-1]] - (shares * df_price.loc[index[i-1]])/value[i-1]).sum()
            value[i] = (shares * df_price.loc[index[i]]).sum() - cost + cash
            shares = df_weight.loc[index[i-1]] * value[i] / df_price.loc[index[i]]
            cash = value[i] - (shares * df_price.loc[index[i]]).sum()
        else: 
            value[i] = (shares * df_price.loc[index[i]]).sum() + cash

    return pd.DataFrame(index=index, data=value, columns=['Track'])


def ols_regression(df_y, df_x, sample_length: int, frequency: int):

    index = df_y.index.copy()
    n, m = df_x.shape

    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency + 1):

        start = index[i*frequency]
        end = index[i*frequency + sample_length - 1]

        x = df_x.loc[start:end].values
        y = df_y.loc[start:end].values

        weight = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

        df_weight.loc[end] = weight[:,0].T

    return df_weight.fillna(0)


def lasso_regression(df_y, df_x, sample_length: int, frequency: int, l=0.):

    index = df_y.index.copy()
    n, m = df_x.shape
    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency + 1):

        start = index[i*frequency]
        end = index[i*frequency + sample_length - 1]

        stdx = df_x.loc[start:end].std(axis=0).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values

        def loss(z):
            return np.sum((np.dot(x, z) - y.T)**2) + l * sample_length * np.sum(np.abs(z)) / (stdy ** 2)

        res = minimize(loss, np.zeros([m, 1]), method='SLSQP')

        df_weight.loc[end] = res.x
        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

    return df_weight.fillna(0)


def ridge_regression(df_y, df_x, sample_length: int, frequency: int, l=0.):

    index = df_y.index.copy()
    n, m = df_x.shape
    I = np.eye(m)
    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency + 1):

        start = index[i*frequency]
        end = index[i*frequency + sample_length - 1]

        stdx = df_x.loc[start:end].std(axis=0).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values

        l1 = l * sample_length / (np.float(m) * (stdy.iloc[0] ** 2))
        weight = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + l1 * I), x.T), y)

        df_weight.loc[end] = weight[:, 0].T
        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

    return df_weight.fillna(0)


def kalman_filter(df_y, df_x, frequency: int, sigma_weight, sigma_return):

    index = df_y.index.copy()
    n, m = df_x.shape

    cov_weight = sigma_weight * np.eye(m)
    cov_return = sigma_return * np.eye(frequency)
    df_weight = pd.DataFrame(columns=df_x.columns)

    weight_filter = np.zeros([m, 1])
    cov_filter = np.zeros([m, m])

    I = np.eye(m)

    for i in range((n - frequency)//frequency + 1):

        start = index[i*frequency]
        end = index[(i + 1)*frequency - 1]

        x = df_x.loc[start:end].values
        y = df_y.loc[start:end].values

        weight_forecast = weight_filter
        cov_forecast = cov_filter + cov_weight

        temp = np.dot(cov_forecast, x.T)
        inv = np.linalg.inv(np.dot(x, temp) + cov_return)
        K = np.dot(temp, inv)

        weight_filter = (weight_forecast + np.dot(K, y - np.dot(x, weight_forecast)))
        cov_filter = np.dot(I - np.dot(K, x), cov_forecast)

        df_weight.loc[end] = weight_filter[:, 0].T

    return df_weight.fillna(0)



if __name__ == "__main__":
    sns.set()
    prices = pd.read_csv(r"financial_data/prices.csv", index_col=0, parse_dates=True, dayfirst=True)
    prices.index = pd.DatetimeIndex(prices.index)
    EU_rate = pd.read_csv(r"financial_data/EUR_rates.csv", index_col=0, parse_dates=True, dayfirst=True)['3M']
    prices = make_ER(prices, EU_rate)
    mondays = pd.date_range(start=dt.date(2010, 1, 4), end=dt.date.today(), freq='7D')
    returns = prices.reindex(mondays).ffill().pct_change().dropna()

    sx5e = returns[["SX5E"]]
    bch = returns.drop("SX5E", axis=1)

    # Params
    sample = 52
    freq = 13

    weight = ols_regression(sx5e, bch, sample, freq)
    prices_for_track = prices.loc[weight.index[0]:].drop("SX5E", axis=1)
    replication = make_track(prices_for_track, weight)

    weight_old = ols_regression_old(sx5e, bch, sample, freq, boundaries=(-np.inf, np.inf), weight_sum=np.nan)
    replication_old = make_track(prices_for_track, weight_old)
    (weight_old - weight).plot()
    plt.show()

    weight_kalman = kalman_filter(sx5e, bch, freq, sigma_weight=0.01, sigma_return=0.01)
    prices_for_track = prices.loc[weight_kalman.index[0]:].drop("SX5E", axis=1)
    replication_kalman = make_track(prices_for_track, weight_kalman)

    df_res = prices.loc[weight.index[0]:][["SX5E"]]
    df_res["OLS"] = replication
    df_res["OLS old"] = replication_old
    df_res["Kalman"] = replication_kalman

    # for l in [1e-4, 5e-4, 1e-3, 5e-3]:
    #     weight_lasso = lasso_regression(sx5e, bch, sample, freq, boundaries=(0, np.inf), weight_sum=np.nan, l=l)
    #     replication_lasso = make_track(prices_for_track, weight_lasso)
    #     df_res[('Lasso : lambda = '+str(l))] = replication_lasso

    df_res = df_res / df_res.iloc[0]
    df_res = df_res.bfill()

    df_res.plot(figsize=(10, 6))
    plt.gca().set_ylim(bottom=0)
    plt.show()
