# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# %%
folder ='data/real/pm2.5'

def pmshow(city, yid, post = True):
    city_path = os.path.join(folder, '{}'.format(yid), '{}.post.csv'.format(city) if post else '{}.csv'.format(city))
    df = pd.read_csv(city_path,header=0, index_col=[0])
    data = df['PM_US Post']
    null_num = len(data[data.isnull()].index)
    print('Null num: {}'.format(null_num))
    
    if null_num > 0:
        data = data.interpolate()
    
    
    data.plot()
    # plt.title('{} {} post {}'.format(city, yid, 'True' if post else 'False'))
    # plt.show()
# %%
# pmshow('Chengdu' , 2010)
# %%
# cid = 'Chengdu'
# cid = 'Guangzhou'
cid = 'Shanghai'
# cid = 'Shenyang'
for yid in range(2015,2016):
    fig = plt.figure(figsize=(16, 9))
    pmshow(cid, yid, True)
    pmshow(cid, yid, False)
    plt.title('{} {} post {}'.format(cid, yid, 'True' ))
    plt.legend(['post true', 'post false'])
    plt.show()
# %%
