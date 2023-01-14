# %%
import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
# %%

ts = np.load('data/src/laser/laser.npy')
print(ts.shape) # 100K
# %%
# np.save('data/src/laser/laser.npy', ts)
# %%
# %%
fig = plt.figure(figsize=(16,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(ts, lags=180, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=180, ax=ax2)

# %%
