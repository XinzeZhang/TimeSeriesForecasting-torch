# %%
import pandas as pd
import numpy as np
import os

# %%
fold = 'data/paper.esm/SP500'
filepath = os.path.join(fold, 'SP500.csv')
# filepath = 'data/paper.esm/SP500/SP500.csv'
df = pd.read_csv(filepath,index_col='Date',header=0, parse_dates=['Date']).asfreq('1D')

# %%
raw = df.values
print(raw.shape)
df = df.dropna()
print(df)
# %%
ts = df['Adj Close'].values
print(ts.shape)
# %%
np.save(os.path.join(fold, 'sp500'), ts)

# %%
