#%%
import pandas as pd
import numpy as np
import os 
from pathlib import Path
import shutil
# %%
data_folder = 'data/real/nsw17_19'
year_fold = 'nsw_raw_data'

data_path = Path(os.path.join(data_folder,year_fold))
data = [x for x in data_path.iterdir() if x.is_file() ]
# %%
price_folder = os.path.join(data_folder, 'nsw_data')
if os.path.exists(price_folder):
    shutil.rmtree(price_folder)

os.makedirs(price_folder)

# %%
def get_data(data, year, m_tag):
    y_mon = [x for x in data if year in str(x) and x.name.split('_')[3][-2:] == m_tag][0]
    year = y_mon.name.split('_')[3][:-2]
    mon = y_mon.name.split('_')[3][-2:]
    df = pd.read_csv(str(y_mon), index_col='SETTLEMENTDATE', parse_dates=True)
    mdemand = df['TOTALDEMAND']
    mdemand = mdemand.values
    mprice = df['RRP'].values

    return df, mdemand, mprice
# %%
for year in [2019]:
    year = str(year)
    
    year_data = []
    for month in range(1,13):
        m_tag = '0{}'.format(month) if month < 10 else str(month)
        
        df, mdemand, mprice = get_data(data, year, m_tag)
        print('-'*50)
        print('Half-hourly:\t {}\t{}\tMax: {:.3f}\tMin: {:.3f} \tMean: {:.3f}'.format(year, m_tag, mprice.max(), mprice.min(), mprice.mean()))
        
        ts = pd.Series(mprice, index = df.index.copy())
        ts_q1 = ts.quantile(0.25, interpolation='nearest')
        ts_q3 = ts.quantile(0.75, interpolation='nearest')
        iqr = ts_q3 - ts_q1
        r = 5
        
        # eval_l_bound = ts_q1 - r * iqr
        eval_u_bound = ts_q3 + r * iqr
        eval_u_bound = 400
        
        ts_score= ts.copy()
        ts_score= ts_score> eval_u_bound 
        
        print(ts.loc[ts_score == True])
        ts[ts_score == True] = None
        ts = ts.interpolate()
        print(ts.loc[ts_score == True])
        _hprice = ts.resample('1H', offset='0.5H').sum()
        hprice = _hprice.values
        print('Hourly:\t {} \t{} \tMax: {:.3f}\tMin: {:.3f} \tMean: {:.3f}'.format(year, m_tag, hprice.max(), hprice.min(), hprice.mean()))
        
        hprice = hprice.tolist()
        year_data.extend(hprice)

    year_data = np.array(year_data)
    print(year_data.shape)
    np.save(os.path.join(price_folder, 'loadPrice.{}'.format(year)), year_data)
        
# %%
24*365
# %%
