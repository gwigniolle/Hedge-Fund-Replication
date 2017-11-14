import pandas as pd
import datetime as dt
from sklearn import linear_model


def make_track(df_price, df_weight, tc=0):
    """
    :param df_price: a dataframe containing the prices of the underlyings used in the index, columns must be the names
    and the index are the dates
    :type pd.Dataframe
    :param df_weight: a dataframe containing the weight on the rebalancing dates of the track created
    :param tc: transaction cost, default is 0
    :return: a pandas series containing the track made from the composition in df_weight
    """

    # df_shares = (df_weight/df_price.shift(1).bfill()).ffill()
    # df_track = df_shares*df_price
    # values = df_track.values

    index = df_price.index
    reweight_index = df_weight.index

    n = len(index)
    shares = (df_weight / df_price).iloc[0]

    value = np.ones(n)

    for i in index[1:]:

        if index[i-1] in reweight_index and i > 0:
            cost = tc * value[i-1] * np.abs(df_weight.loc[index[i-1]] - value[i-1] / (df_shares.loc[index[i-1]] * df_price.loc[index[i-1]])).sum()
            value[i] = (shares * df_price.loc[index[i]]).sum() - cost
            shares = df_weight.loc[index[i-1]] * value[i] / df_price.loc[index[i]]
        else: 
            value[i] = (shares * df_price.loc[index[i]]).sum()

        #     value = (1-tc)*df_track.iloc[i-1].sum()
        # df_track.iloc[i] = df_track.iloc[i]*value

    return pd.Series(index=index, data=value)


if __name__ == "__main__":

    prices = pd.read_csv(r"financial_data/prices.csv", index_col=0)
    prices.index = pd.DatetimeIndex(prices.index)

    mondays = pd.date_range(start=dt.date(2010, 1, 4), end=dt.date.today(), freq='7D')
    returns = prices.reindex(mondays).ffill().pct_change().dropna()

    sx5e = returns[["SX5E"]]
    bch = returns.drop("SX5E", axis=1)

    # Params
    dates = returns.index.copy()
    n = len(dates)

    sample_period = 8
    reg_freq = 4

    df_weight = pd.DataFrame(columns=bch.columns)

    for i in range((n - sample_period) // reg_freq):
        start = dates[i * reg_freq]
        end = dates[i * reg_freq + sample_period]

        x = bch.loc[start:end]
        y = sx5e.loc[start:end]

        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        df_weight.loc[end] = reg.coef_[0]

    prices_for_track = prices.loc[df_weight.index].drop("SX5E", axis=1)
    replication = make_track(prices_for_track, df_weight)

    df_res = prices.loc[df_weight.index][["SX5E"]]
    df_res["OLS"] = replication

    df_res = df_res / df_res.iloc[0]

    df_res.plot(figsize=(10, 6))


