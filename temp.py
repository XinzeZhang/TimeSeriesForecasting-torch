# %%
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit

dataset= np.array([[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8],[4,5,6,7,8,9],[5,6,7,8,9,10]])

# %%
kf = KFold()
for train_idx, test_idx in kf.split(dataset):
    train_date, test_data = dataset[train_idx],dataset[test_idx]
    print(dataset[train_idx],dataset[test_idx])

# %%
tscv = TimeSeriesSplit(n_splits=4)
for train_idx, test_idx in tscv.split(dataset):
    train_date, test_data = dataset[train_idx],dataset[test_idx]
    print(dataset[train_idx],dataset[test_idx])

# %%
*lst, last = tscv.split(dataset)
train_idx, test_idx = last
print(dataset[train_idx],dataset[test_idx])
# %%
