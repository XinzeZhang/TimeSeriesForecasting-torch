# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# %%
data_path = 'data/src/gef/2017_smd_hourly.xlsx'


df = pd.read_excel(data_path,sheet_name='ME', index_col=None,header=0)

df.describe()
# %%
df['Date'].astype(str)
# %%
df['Hr_End'] - 1 
# %%
_time = df['Date'].astype(str) + '-' + (df['Hr_End'] - 1 ).astype(str)

_time = pd.to_datetime(_time, format='%Y-%m-%d-%H')

_time
# %%
df.index = _time

df['RT_Demand']

# %%
df['RT_Demand'].values.shape
# %%
