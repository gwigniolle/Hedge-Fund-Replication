import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Lasso, LassoLarsIC


def make_stats_maxence(df_price: pd.DataFrame):
    df_return = df_price.pct_change().dropna()
    stats.describe(df_return)
    t_tstat, p_tstat = stats.ttest_rel(df_return.iloc[:,0], df_return.iloc[:, 1])  # T-test
    t_KS, p_KS = stats.ks_2samp(df_return.iloc[:,0], df_return.iloc[:, 1])  # KS -> p petit pas la meme distri
    tau, p_tau = stats.kendalltau(df_return.iloc[:,0], df_return.iloc[:, 1])  # Tau de Kendall

    return stats.describe(df_return),"t test: t = %g  p = %g" % (t_tstat, p_tstat), \
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
    """
    prices need to be in ER
    :param df_price:
    :param df_fx:
    :return:
    """
    dates = df_price.loc[df_fx.index[0]:].index
    price_fx_hedge = pd.DataFrame(index=dates, columns=df_price.columns)
    df_fx = df_fx.reindex(dates).ffill()
    n = len(dates)
    price_fx_hedge.iloc[0] = 1.

    for i in range(1, n):
        price_fx_hedge.iloc[i] = price_fx_hedge.iloc[i-1] * (1 + df_price.iloc[i] * df_fx.iloc[i] / (df_price.iloc[i-1] * df_fx.iloc[i-1]) -
                                                   df_fx.iloc[i]/df_fx.iloc[i-1])
    return price_fx_hedge
        

def make_ER(price: pd.DataFrame, rate):

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


def make_track(df_price: pd.DataFrame, df_weight: pd.DataFrame, tc=0, lag=1):
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

        if index[i-lag] in reweight_index:
            value[i] = (shares * df_price.loc[index[i]]).sum() + cash
            cost = tc * value[i] * np.abs(df_weight.loc[index[i-lag]] - (shares * df_price.loc[index[i]])/value[i]).sum()
            value[i] = value[i] - cost
            shares = df_weight.loc[index[i-lag]] * value[i] / df_price.loc[index[i]]
            cash = value[i] - (shares * df_price.loc[index[i]]).sum()
        else: 
            value[i] = (shares * df_price.loc[index[i]]).sum() + cash

    return pd.DataFrame(index=index, data=value, columns=['Track'])


def ols_regression(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int):

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


def lasso_regression(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int, l=0.):

    index = df_y.index.copy()
    n, m = df_x.shape
    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency + 1):

        start = index[i*frequency]
        end = index[i*frequency + sample_length - 1]

        stdx = df_x.loc[start:end].std(axis=0, skipna=False).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values
        
        las = Lasso(alpha=l / (2. * (stdy.iloc[0] ** 2)), fit_intercept=False, normalize=False)
        las.fit(x, y)
        
        df_weight.loc[end] = las.coef_
        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

    return df_weight.fillna(0)


def lasso_regression_ic(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int,
                        criterion: str, plot_lambda=True):

    index = df_y.index.copy()
    n, m = df_x.shape
    df_weight = pd.DataFrame(columns=df_x.columns)
    df_lambda = pd.DataFrame(columns=['$\lambda$'])

    for i in range((n - sample_length) // frequency + 1):
        start = index[i * frequency]
        end = index[i * frequency + sample_length - 1]

        stdx = df_x.loc[start:end].std(axis=0, skipna=False).replace({0: np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values

        las = LassoLarsIC(criterion=criterion, fit_intercept=False, normalize=False)
        las.fit(x, np.ravel(y))

        df_lambda.loc[end] = 2. * las.alpha_ * (stdy.iloc[0] ** 2)
        df_weight.loc[end] = las.coef_
        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

    df_weight = df_weight.fillna(0)
    if plot_lambda:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        sns.set()
        df_lambda['$\lambda$'].plot(title="$\lambda$ parameter selected by the " + criterion.upper(), ax=axes[0])
        (df_weight != 0).sum(axis=1).plot(title="Number of selected factors by the " + criterion.upper(), ax=axes[1])
        plt.show()

    return df_weight, df_lambda


def ridge_regression(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int, l=0.):

    index = df_y.index.copy()
    n, m = df_x.shape
    I = np.eye(m)
    df_weight = pd.DataFrame(columns=df_x.columns)

    for i in range((n - sample_length)//frequency + 1):

        start = index[i*frequency]
        end = index[i*frequency + sample_length - 1]

        stdx = df_x.loc[start:end].std(axis=0, skipna=False).replace({0 : np.nan})
        stdy = df_y.loc[start:end].std(axis=0)
        x = (df_x.loc[start:end] / stdx).fillna(0).values
        y = (df_y.loc[start:end] / stdy).values

        l1 = l * sample_length / (np.float(m) * (stdy.iloc[0] ** 2))
        weight = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + l1 * I), x.T), y)

        df_weight.loc[end] = weight[:, 0].T
        df_weight.loc[end] = df_weight.loc[end] * stdy.iloc[0] / stdx

    return df_weight.fillna(0)


def kalman_filter(df_y: pd.DataFrame, df_x: pd.DataFrame, frequency: int, sigma_weight: float,
                  sigma_return: float, weight_init=0):

    index = df_y.index.copy()
    n, m = df_x.shape

    cov_weight = (sigma_weight ** 2) * np.eye(m)
    cov_return = (sigma_return ** 2) * np.eye(frequency)
    df_weight = pd.DataFrame(columns=df_x.columns)

    try: weight_filter = np.array(weight_init.values).reshape(m, 1)
    except: weight_filter = np.zeros([m, 1])

    cov_filter = np.zeros([m, m])

    I = np.eye(m)

    for i in range((n - frequency)//frequency + 1):

        start = index[i*frequency]
        end = index[(i + 1)*frequency - 1]

        x = df_x.loc[start:end].values
        y = df_y.loc[start:end].values

        cov_forecast = cov_filter + cov_weight

        temp = np.dot(cov_forecast, x.T)
        inv = np.linalg.inv(np.dot(x, temp) + cov_return)
        K = np.dot(temp, inv)

        weight_filter = (weight_filter + np.dot(K, y - np.dot(x, weight_filter)))
        cov_filter = np.dot(I - np.dot(K, x), cov_forecast)

        df_weight.loc[end] = weight_filter[:, 0].T

    return df_weight.fillna(0)


def kalman_with_selection(df_y: pd.DataFrame, df_x: pd.DataFrame, sample_length: int, frequency: int,
                          nu: float, nb_period: int, criterion: str):

    df_weight_lasso, _ = lasso_regression_ic(df_y, df_x, sample_length, frequency, criterion, plot_lambda=False)
    df_weight = pd.DataFrame(columns=df_x.columns)
    index = df_weight_lasso.index.copy()

    for date in index:
        selection = df_weight_lasso.loc[date] != 0.0
        selection = list(selection[selection].index)
        if not selection:
            df_weight.loc[date, :] = 0
        else:
            i = df_x.index.get_loc(date)
            df_x_ = df_x[selection].iloc[i-nb_period+1: i+1]
            df_y_ = df_y.loc[df_x_.index]
            kalman = kalman_filter(df_y=df_y_, df_x=df_x_, frequency=1, sigma_weight=1.,
                                   sigma_return=nu, weight_init=df_weight_lasso.loc[date, selection])
            df_weight.loc[date, selection] = kalman.loc[date]

    return df_weight.fillna(0.0)


if __name__ == "__main__":

    fund_name = 'HFRXMD'
    US_rate = pd.read_csv(r"financial_data/USD_rates.csv", index_col=0, parse_dates=True, dayfirst=True)['3M']

    hfrx_all = pd.read_csv(r"financial_data/hfrx_daily_index_data.csv", index_col=0, parse_dates=True,
                           dayfirst=True).ffill()
    hfrx = make_ER(hfrx_all[[fund_name]].dropna(), US_rate)

    bnp = pd.read_csv(r"financial_data/bnp_data.csv", index_col=0, parse_dates=True, dayfirst=True)
    risk_premia = pd.read_pickle("financial_data/risk_premia_ER_FX_USD.pkl")

    prices_all = bnp.join(risk_premia, how="outer").ffill().join(hfrx, how="inner")
    returns_all = prices_all.resample('1D').first().pct_change().dropna()
    hrfx_returns = returns_all[[fund_name]]
    returns_all = returns_all.drop(fund_name, axis=1)
    size = 126
    freq = 5
    tc = 0.001
    lag = 1

    df_weight_skalman = kalman_with_selection(hrfx_returns, returns_all, sample_length=size, frequency=freq,
                                              nu=0.02, nb_period=20, criterion='bic')

    print(df_weight_skalman.head())
