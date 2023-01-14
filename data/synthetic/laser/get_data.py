# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
# %%

filepath = 'data/src/laser/laser.csv'
# filepath = 'data/paper.esm/SP500/SP500.csv'
data = pd.read_csv(filepath,index_col=None,header=None)

data.describe()

# %%
data[data.isnull().any(1)]
# %%
data =data.dropna()
ts = data.to_numpy()
print(ts.shape)
# %%
# np.save('data/src/laser/laser.npy', ts)
# %%
ts = ts.reshape(-1,)
print(ts)
# %%

sm.graphics.tsa.plot_acf(ts, lags=180)

# %%
