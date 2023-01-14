# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
# %%

ts = np.load('data/synthetic/narma/narma.npy')
print(ts.shape)
# %%
# np.save('data/src/laser/laser.npy', ts)
# %%
# %%
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=15, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=15, ax=ax2)
# %%
