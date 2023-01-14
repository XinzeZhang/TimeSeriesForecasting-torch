# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from pandas.core import series
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

    
# %%

# %%
def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]
# %%
def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()
# %%


csv_path = 'data/src/elect/LD2011_2014.txt'

data_frame = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
# %%
data_frame.describe()
# %%
_df = data_frame.resample('1H',label = 'left',closed = 'right').sum()
_df.describe()
# %%

_ds = _df['MT_003']
plt.figure()
_ds.plot()
plt.show()
# %%
_ds['2014-01-02 00:00:00':'2014-02-01 00:00:00'].plot()
plt.show()
# %%
_ds.index[0]
# _ds[0]
_ds[_ds.index[0]]
# %%
begin_idx = None
for i, ind in enumerate(_ds.index):
    cur_idx = ind
    next_idx = _ds.index[i+1]
    diff = _ds[next_idx] - _ds[cur_idx]
    if diff != 0.0:
        begin_idx = next_idx
        break
    
print(begin_idx)
print(_ds[begin_idx])
# %%
