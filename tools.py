import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


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

    value = np.ones(n)

    for i in range(1, len(index)):

        if index[i-1] in reweight_index:
            cost = tc * value[i-1] * np.abs(df_weight.loc[index[i-1]] - (shares * df_price.loc[index[i-1]])/value[i-1]).sum()
            value[i] = (shares * df_price.loc[index[i]]).sum() - cost
            shares = df_weight.loc[index[i-1]] * value[i] / df_price.loc[index[i]]
        else: 
            value[i] = (shares * df_price.loc[index[i]]).sum()

    return pd.Series(index=index, data=value)


def ols_regression(df_y, df_x, sample_length: int, frequency: int, boundaries=(-np.inf, np.inf),
                   weight_sum=np.nan):

    index = df_y.index.copy()
    n, m = df_x.shape

    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency):

        start = index[i*frequency]
        end = index[i*frequency + sample_length]

        x = df_x.loc[start:end].values
        y = df_y.loc[start:end].values

        def loss(z):
            return np.sum(np.square(np.dot(x, z)-y))

        cons = ({'type': 'eq',
                 'fun': lambda z: np.sum(z) - weight_sum}) if not np.isnan(weight_sum) else ()
        bounds = [boundaries]*m
        z0 = np.zeros([m, 1])

        res = minimize(loss, z0, method='SLSQP', constraints=cons, bounds=bounds)

        df_weight.loc[end] = res.x

    return df_weight


if __name__ == "__main__":
    sns.set()
    prices = pd.read_csv(r"financial_data/prices.csv", index_col=0)
    prices.index = pd.DatetimeIndex(prices.index)

    mondays = pd.date_range(start=dt.date(2010, 1, 4), end=dt.date.today(), freq='7D')
    returns = prices.reindex(mondays).ffill().pct_change().dropna()

    sx5e = returns[["SX5E"]]
    bch = returns.drop("SX5E", axis=1)

    # Params
    sample = 52
    freq = 13

    weight = ols_regression(sx5e, bch, sample, freq, boundaries=(0, np.inf), weight_sum=1)
    prices_for_track = prices.loc[weight.index[0]:].drop("SX5E", axis=1)
    replication = make_track(prices_for_track, weight)

    df_res = prices.loc[weight.index[0]:][["SX5E"]]
    df_res["OLS Rui"] = replication

    df_res = df_res.bfill()
    df_res = df_res / df_res.iloc[0]

    df_res.plot(figsize=(10, 6))
    plt.show()


