import pandas as pd
import datetime as dt
import numpy as np


def TotaltoExcessReturn(df_Return, df_Rate):

    dates = df_Return.index
    df_Excess = pd.DataFrame(index=dates)
    df_Excess.iloc[0] = 100.0
    old_date = dates[0]
    for date in dates[1:]:
        df_Excess[date] = df_Excess[old_date]*(df_Return[date] / df_Return[old_date] - df_Rate[old_date] *
                                               (date - old_date) / 360.0)
        old_date = date
    return  df_Excess
