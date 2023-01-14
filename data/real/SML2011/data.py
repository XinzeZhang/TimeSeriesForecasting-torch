# %%
import pandas as pd
import os 

df = pd.read_csv(os.path.join('data/real/SML2011/','NEW-DATA-{}.T15.txt'.format(2)),header=0, sep=' ')
data = df['3:Temperature_Comedor_Sensor']
if data.isnull().any():
    data= data.interpolate()
raw_ts = data.values.reshape(-1, )
raw_ts.shape
# %%

# %%
2764 + 1373

# %%
