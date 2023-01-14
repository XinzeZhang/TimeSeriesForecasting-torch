# %%
import pandas as pd
import numpy as np
import os

# %%
file_name = 'Wti-d'
folder = 'data/src/cop/'
file_path =  os.path.join(folder,'{}.xls'.format(file_name))
data = pd.read_excel(file_path, sheet_name='Data 1', header= 2, index_col=0, )
# data.describe()
# %%
data
# %%
data[(data < 0).any(1)] = np.nan
# %%
data[(data < 0).any(1)]

# %%
data[data.isnull().any(1)]
# %%
data =  data.interpolate()
# %%
data[data.isnull().any(1)]
# %%
data
# %%
ts = data.to_numpy()
ts.shape

# npy_file = os.path.join(folder, file_name)
# np.save(npy_file, ts)

# %%
ts = ts.reshape(-1,)
# %%
ts
# %%
ts.shape
# %%
