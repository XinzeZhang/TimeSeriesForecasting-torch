# %%
import pandas as pd
import numpy as np
import os

# %%
fold = 'data/real/pm2.5/raw'
file_name = 'Beijing' # 2014
file_name = 'Chengdu' # 2015
file_name = 'Guangzhou' # 2015
# file_name = 'Shanghai' # 2014
# file_name = 'Shenyang' # 2014

os_path = os.path.join(fold, '{}PM20100101_20151231.csv'.format(file_name))

# %%
def timeparser(year, month, day, hour):
    if int(month) < 10:
        month = '0{}'.format(month)
    if int(day) < 10:
        day = '0{}'.format(day)
    if int(hour) < 10:
        hour = '0{}'.format(hour)
    return pd.to_datetime(year + '-' + month + '-' + day + ' ' + hour, format=u'%Y-%m-%d %H')

loc = 'Beijing'
os_path = os.path.join(fold, '{}PM20100101_20151231.csv'.format(loc))
# df = pd.read_csv(os_path, parse_dates={'datetime': ['year', 'month','day','hour']}, date_parser=timeparser,                     index_col='datetime')

df = pd.read_csv(os_path, header=0)
dt = pd.to_datetime(df[["year", "month","day","hour"]])
print(dt)

df.set_index(dt, inplace=True)

df1 = df[df.index.year.isin([2015])]

df1
# print(df.index)

print(df.loc['2015']['PM_US Post'])

# print(df[["year", "month","day"]])


# %%

for loc in ['Beijing', 'Chengdu', 'Guangzhou','Shanghai','Shenyang']:
    os_path = os.path.join(fold, '{}PM20100101_20151231.csv'.format(loc))

    
    df = pd.read_csv(os_path, parse_dates={'datetime': ['year', 'month','day','hour']}, date_parser=timeparser,
                     index_col='datetime')
    
    for year in range(2010, 2016):
        data = df.loc['{}'.format(year)]['PM_US Post']
        print('Loc is: {} \t Year is: {} \t NaN number is: {}'.format(loc , year, data.isnull().sum()))

# %%
import matplotlib.pyplot as plt
def dfplot(loc, year):
    os_path = os.path.join(fold, '{}PM20100101_20151231.csv'.format(loc))
    df = pd.read_csv(os_path, parse_dates={'datetime': ['year', 'month','day','hour']}, date_parser=timeparser,index_col='datetime')
    
    data = df.loc['{}'.format(year)]['PM_US Post']
    data = data.interpolate()
    data.plot()
    plt.show()

# %%
# file_name = 'Beijing' # 2014
# file_name = 'Chengdu' # 2015
# file_name = 'Guangzhou' # 2015
# file_name = 'Shanghai' # 2014
# file_name = 'Shenyang' # 2014
dfplot('Beijing' , 2014)
dfplot('Chengdu' , 2015)
dfplot('Guangzhou' , 2015)
dfplot('Shanghai' , 2014)
dfplot('Shenyang' , 2014)
# %%
