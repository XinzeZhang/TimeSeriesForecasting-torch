# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
# %%

ts = np.load('data/src/mg/mg.npy')
print(ts.shape)
# %%
# np.save('data/src/laser/laser.npy', ts)
# %%
# %%
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=84*2, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=84*2, ax=ax2)

# %%
