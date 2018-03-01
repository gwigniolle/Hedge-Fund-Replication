import pandas as pd
import numpy as np
from tools import kalman_filter, ml_kalman_filter, make_track, ols_regression

if __name__ == "__main__":

    prices_index = pd.read_csv(r"financial_data/bnp_data.csv", index_col=0, parse_dates=True,
                               dayfirst=True)[['BNPIFEU', "SPGSGCP", "BPFXCAD1"]]
    prices_risk_premia = pd.read_csv(r"financial_data/bnp_risk_premia_data.csv", index_col=0, parse_dates=True,
                                     dayfirst=True)[["BNPIMDEA"]]

    prices = pd.concat([prices_index, prices_risk_premia], axis=1).dropna()
    returns = prices.pct_change().dropna()
    df_reweight = pd.read_pickle("df_reweight.pkl")
    weight = df_reweight

    prices_for_track = prices.loc[weight.index[0]:]
    vol_cap = make_track(prices_for_track, weight, 0)

    # Replication Data

    vol_cap_returns = vol_cap.pct_change().dropna()
    universe_returns = returns.loc[vol_cap_returns.index]

    size = 52
    freq = 1

    df_weight_ols = ols_regression(vol_cap_returns, universe_returns, size, freq)
    prices_for_track = prices.loc[df_weight_ols.index[0]:]
    replication = make_track(prices_for_track, df_weight_ols)

    df_res = vol_cap.loc[replication.index[0]:].copy()
    df_res["OLS"] = replication.loc[df_res.index]

    for nu in [150]:
        df_weight_kalman_ = kalman_filter(vol_cap_returns, universe_returns, freq, sigma_weight=nu * 0.1,
                                          sigma_return=0.1)
        df_weight_kalman = df_weight_kalman_.loc[df_weight_ols.index[0]:]

        prices_for_track_kalman = prices.loc[df_weight_ols.index[0]:]
        replication_kalman = make_track(prices_for_track_kalman, df_weight_kalman).loc[df_res.index[0]:]
        df_res["Kalman {}".format(nu)] = replication_kalman

    for tau in [0.25]:

        df_weight_ml_kalman_ = ml_kalman_filter(vol_cap_returns, universe_returns, freq, tau=0.5, plot_sigma=True)
        df_weight_ml_kalman = df_weight_ml_kalman_.loc[df_weight_ols.index[0]:]
        prices_for_track_kalman_ml = prices.loc[df_weight_ols.index[0]:]
        replication_kalman_ml = make_track(prices_for_track_kalman_ml, df_weight_ml_kalman).loc[df_res.index[0]:]
        df_res["Kalman_ML {}".format(tau)] = replication_kalman_ml

    df_res = df_res / df_res.iloc[0]

    df_res.plot(figsize=(12, 6), linewidth=1, logy=True)
    print(df_res.pct_change().corr(method="pearson"))

    returns_track = df_res.pct_change().dropna()
    track = returns_track['Track']
    returns_track = returns_track.drop('Track', axis=1)
    df = pd.DataFrame()
    df['Tracking error'] = (returns_track.T - track.values).std(axis=1)
    df['$R^2$'] = 1 - (returns_track.T - track.values).var(axis=1) / track.values.var()
    print(df)

