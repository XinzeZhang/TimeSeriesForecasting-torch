# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
# %%
data_path = 'data/real/pm2.5/raw'
# %%

# %%
# l_y = [('Beijing' , 2014), ('Chengdu' , 2015),('Guangzhou' , 2015),('Shanghai' , 2014),('Shenyang' , 2014)]

ex_col_dict = {
    'Beijing':{'loc':['PM_Dongsi' ,'PM_Dongsihuan','PM_Nongzhanguan'],'bound':600},
    'Chengdu':{'loc':['PM_Caotangsi','PM_Shahepu'],'bound':400},
        
    'Guangzhou':{'loc':['PM_City Station','PM_5th Middle School'],'bound':330},
        
    'Shanghai':{'loc':['PM_Jingan','PM_Xuhui'],'bound':360},
        
    'Shenyang':{'loc':['PM_Taiyuanjie','PM_Xiaoheyan'],'bound':400},
        
}

# %%
from dateutil.relativedelta import relativedelta

years = [2010, 2011, 2012, 2013, 2014, 2015]

def dfplot(loc, year, ext = True):
    last_year = [y for y in years if y != year ]
    
    save_folder = os.path.join('data/real/pm2.5/{}'.format(year))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    df = pd.read_csv(os.path.join(data_path, '{}PM20100101_20151231.csv'.format(loc)), header=0)
    dt = pd.to_datetime(df[["year", "month","day","hour"]])
    df.set_index(dt, inplace=True)
    
    data = df.loc['{}'.format(year)]['PM_US Post']
    
    nullidx = data[data.isnull()].index
    ex_col_list = ex_col_dict[loc]['loc']
    bound = ex_col_dict[loc]['bound']
    for i in range(len(nullidx)):
        cur_idx = nullidx[i]
        count = 0
        sum = 0
        if ext:
            for ty in last_year:
                delta = ty - year
                try:
                    new_tag = cur_idx + relativedelta(years=delta)
                    for ex_col in ex_col_list:
                        ex_v = df.loc[new_tag][ex_col]
                        if pd.isna(ex_v):
                            pass
                        else:
                            sum += ex_v
                            count += 1
                except:
                    continue            
        # else:
        #     for ty in last_year:
        #         delta = ty - year
        #         try:
        #             new_tag = cur_idx + relativedelta(years=delta)
        #             ex_v = df.loc[new_tag]['PM_US Post']
        #             if pd.isna(ex_v):
        #                 pass
        #             else:
        #                 sum += ex_v
        #                 count += 1
        #         except:
        #             continue
                            
        #     if count == 0:
                
        #         for ex_col in ex_col_list:
        #             ex_v = df.loc[cur_idx][ex_col]
        #             if pd.isna(ex_v):
        #                 pass
        #             else:
        #                 sum += ex_v
        #                 count += 1
        if count > 0:
            avg = sum / count
            data.at[cur_idx] = avg

    print('City: {} \t Year: {} \t remain_null_num: {}'.format(loc, year, len(data[data.isnull()].index)))
    # data = data.interpolate()
    
    # save_path = os.path.join(save_folder,'{}.csv'.format(loc) if ext else '{}.ta.csv'.format(loc))
    # data.to_csv(save_path)

    data[data > bound] = np.nan
    print('City: {} \t Year: {} \t With up bound remain_null_num: {}'.format(loc, year, len(data[data.isnull()].index)))
    data[data < 3] = np.nan
    print('City: {} \t Year: {} \t With low bound remain_null_num: {}'.format(loc, year, len(data[data.isnull()].index)))
    
    nullidx = data[data.isnull()].index
    for i in range(len(nullidx)):
        cur_idx = nullidx[i]
        count = 0
        sum = 0

        for ty in last_year:
            delta = ty - year
            try:
                new_tag = cur_idx + relativedelta(years=delta)
                for ex_col in ex_col_list:
                    if pd.isnull(df.loc[new_tag][ex_col]):
                    # ex_v = df.loc[new_tag][ex_col]
                    # if pd.isna(ex_v):
                        pass
                    else:
                        ex_v = df.loc[new_tag][ex_col]
                        sum += ex_v
                        count += 1
            except:
                continue      
        if count > 0:
            avg = sum / count
            data.at[cur_idx] = avg
    
    print('City: {} \t Year: {} \t With post_proc. remain_null_num: {}'.format(loc, year, len(data[data.isnull()].index)))
    data = data.interpolate()
    
    save_path = os.path.join(save_folder,'{}.post.csv'.format(loc))
    data.to_csv(save_path)
    print('save to {}'.format(save_path))


    # return data


# for yid in [2011,2012,2013]:
for loc in ex_col_dict.keys():
    for yid in [2010,2011,2012,2013,2014]:
    # for yid in [2015]:
# yid = 2014
        for ext in [True]:
            dfplot(loc, yid, ext) # p,d,q = (55,1,2)
            # save_folder = os.path.join('data/real/pm2.5/{}'.format(yid))
            # if not os.path.exists(save_folder):
            #     os.mkdir(save_folder)
            # save_path = os.path.join(save_folder,'{}.csv'.format(loc))
            # data.to_csv(save_path)

# fig = plt.figure(figsize=(14,7*4))
# layout = (4,1)
# data_ax = plt.subplot2grid(layout, (0,0))
# ts_ax = plt.subplot2grid(layout, (1,0))
# acf_ax = plt.subplot2grid(layout, (2,0))
# pacf_ax = plt.subplot2grid(layout, (3,0))
# ts = data.diff(1).dropna()
# data.plot(ax=data_ax)
# ts.plot(ax=ts_ax)

# sm.graphics.tsa.plot_acf(ts,lags=168,ax=acf_ax) # define q
# sm.graphics.tsa.plot_pacf(ts,lags=168,ax=pacf_ax) #define p
# plt.tight_layout()
# plt.show()

# data = dfplot('Beijing' , 2015) # p,d,q = (30,2,7)
# data = dfplot('Chengdu' , 2015) # p,d,q = (35,2,3)
# data = dfplot('Guangzhou', 2015) # p,d,q = (32,2,2)
# data = dfplot('Shanghai', 2015) # p,d,q = (70,2,2)
# data = dfplot('Shenyang', 2015) # p,d,q = (20,1,19)
# ts = data.diff(1).diff(1).dropna() # define d

# %%
