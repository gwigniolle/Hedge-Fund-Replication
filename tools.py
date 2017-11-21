import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


def make_ER(price, rate):
    dates = price.index
    price_ER = pd.DataFrame(index=dates, columns=price.columns)
    price_ER.iloc[0] = 1.
    n = len(dates)

    rate = rate.reindex(dates).ffill()

    for i in range(1, n):
        price_ER.iloc[i] = price_ER.iloc[i-1] * price.iloc[i] / price.iloc[i-1] * (1 - rate.loc[dates[i-1]] * (dates[i] - dates[i-1]).days / 36000.)

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

    print((index[1]-index[0]).days)

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


def ols_regression(df_y, df_x, sample_length: int, frequency: int, boundaries=(-np.inf, np.inf),
                   weight_sum=np.nan):

    index = df_y.index.copy()
    n, m = df_x.shape

    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency):

        start = index[i*frequency]
        end = index[i*frequency + sample_length]

        stdx = df_x.loc[start:end].std(axis=0).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values

        def loss(z):
            return np.sum((x * z - y)**2)

        cons = ({'type': 'eq',
                 'fun': lambda z: np.sum(z) - weight_sum}) if not np.isnan(weight_sum) else ()
        bounds = [boundaries]*m
        z0 = np.zeros([m, 1])

        res = minimize(loss, z0, method='SLSQP', constraints=cons, bounds=bounds)

        df_weight.loc[end] = res.x

        df_weight.loc[end] = (df_weight.loc[end] * stdy.iloc[0]) / stdx

    return df_weight.fillna(0)


def lasso_regression(df_y, df_x, sample_length: int, frequency: int, boundaries=(-np.inf, np.inf),
                     weight_sum=np.nan, l=0.):

    index = df_y.index.copy()
    n, m = df_x.shape

    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency):

        start = index[i*frequency]
        end = index[i*frequency + sample_length]

        stdx = df_x.loc[start:end].std(axis=0).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values


        def loss(z):
            return np.sum((x * z - y)**2) + n * l*np.sum(np.abs(z))

        eq = {'type': 'eq', 'fun': lambda z: np.sum(z) - weight_sum}

        cons = (eq) if not np.isnan(weight_sum) else ()
        bounds = [boundaries]*m
        z0 = np.zeros([m, 1])

        res = minimize(loss, z0, method='SLSQP', constraints=cons, bounds=bounds)

        df_weight.loc[end] = res.x

        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

    return df_weight.fillna(0)


def lasso_regression_2(df_y, df_x, sample_length: int, frequency: int, boundaries=(-np.inf, np.inf),
                     weight_sum=np.nan, l1=0., l2=0.):

    index = df_y.index.copy()
    n, m = df_x.shape

    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency):

        start = index[i*frequency]
        end = index[i*frequency + sample_length]

        stdx = df_x.loc[start:end].std(axis=0).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values


        def loss(z):
            return np.sum((x * z - y)**2) + n * l1 * np.sum(np.abs(z))

        eq = {'type': 'eq', 'fun': lambda z: np.sum(z) - weight_sum}
        ineq = {'type' : 'ineq', 'fun': lambda z: l2 - np.sum(np.abs(z))}
        cons = (eq, ineq) if not np.isnan(weight_sum) else (ineq)
        bounds = [boundaries]*m
        z0 = np.zeros([m, 1])

        res = minimize(loss, z0, method='SLSQP', constraints=cons, bounds=bounds)

        df_weight.loc[end] = res.x

        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

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

    weight = ols_regression(sx5e, bch, sample, freq, boundaries=(0, np.inf), weight_sum=1.)
    weight_lasso = lasso_regression(sx5e, bch, sample, freq, boundaries=(0, np.inf), weight_sum=1., l=0.)

    prices_for_track = prices.loc[weight.index[0]:].drop("SX5E", axis=1)
    replication = make_track(prices_for_track, weight)
    replication_lasso = make_track(prices_for_track, weight_lasso)

    df_res = prices.loc[weight.index[0]:][["SX5E"]]
    df_res["OLS"] = replication
    df_res['Lasso'] = replication_lasso

    df_res = df_res / df_res.iloc[0]
    df_res = df_res.bfill()

    df_res.plot(figsize=(10, 6))
    plt.show()