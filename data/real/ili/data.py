# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_folder = 'data/src/ili/'

# data_path = os.path.join(data_folder, 'ILI & PSR(with pic).xlsx')
data_path = os.path.join(data_folder, 'ILI.csv')
df = pd.read_csv(data_path, header = 0)
df
# %%
def get_data(df, name):
    data = df[name]
    if data.isnull().any():
        data = data.interpolate()
    
    return data.values
# %%
# np.save(os.path.join(data_folder, 'npsr'), npsr)
# %%
sili = get_data(df, 'south_ILI')
fig=plt.figure(figsize=(10,3))
plt.plot(sili)
plt.show()
# %%

nili = get_data(df, 'north_ILI')
fig=plt.figure(figsize=(10,3))
plt.plot(nili)
plt.show()

# %%
# np.save(os.path.join(data_folder, 'sili'), sili)
# np.save(os.path.join(data_folder, 'nili'), nili)

ts = sili.reshape(-1,)

import statsmodels.api as sm


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=53, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=53, ax=ax2)

# %%
