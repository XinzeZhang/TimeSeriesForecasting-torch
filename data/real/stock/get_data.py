# %%
import pandas as pd
import numpy as np
import os

# %%
fold = 'data/paper.esm/stock'
file_name = 'NASDAQ'
filepath = os.path.join(fold, 'Processed_{}.csv'.format(file_name))
# filepath = 'data/paper.esm/SP500/SP500.csv'
df = pd.read_csv(filepath,index_col='Date',header=0, parse_dates=['Date']).asfreq('1D')

# %%
df.describe()
# df = df.dropna()
# print(df)
# %%
ts = df['Close']
print(ts)
ts = ts.dropna()
print(ts)
ts = ts.values
ts
print(ts.shape)
# %%
np.save(os.path.join(fold,file_name.lower()), ts)

# %%
