# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
filepath = 'data/src/AEP/x.csv'

df = pd.read_csv(filepath,index_col='date',header=0, parse_dates=['date']).asfreq('10min')
# %%
df.head(10)
# %%
def get_data(df, name):
    data = df[name]
    if data.isnull().any():
        data = data.interpolate()
    
    return data.values

ts = get_data(df, 'Appliances')
# %%
ts.shape
# %%
import statsmodels.api as sm


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=6*24, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=6*24, ax=ax2)
# %%