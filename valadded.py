import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import warnings
warnings.filterwarnings("ignore")

#read fama data
fama = pd.read_csv('fama_french.csv', delimiter=';')
fama['Date'] = fama['Date'].astype(str)
fama['Year'] = fama['Date'].str[:4]
fama['Month'] = fama['Date'].str[4:]
fama['Year'] = pd.to_numeric(fama['Year'])
fama['Month'] = pd.to_numeric(fama['Month'])

#read NAV data of funds
nav = pd.read_csv('nav/nav_esg.csv', delimiter=';')
nav['Date'] = nav['Date'].astype(str)
nav['Year'] = nav['Date'].str[6:]
nav['Month'] = nav['Date'].str[3:5]
nav['Year'] = pd.to_numeric(nav['Year'])
nav['Month'] = pd.to_numeric(nav['Month'])
nav.drop('Date', axis=1, inplace=True)

#read etf data
etf = pd.read_csv('nav/vanguard_etf.csv')
etf['Date'] = etf['Date'].astype(str)
etf['Year'] = etf['Date'].str[-4:]
etf['Month'] = etf['Date'].str[3:5]
etf['Year'] = pd.to_numeric(etf['Year'])
etf['Month'] = pd.to_numeric(etf['Month'])

#read tna data of funds
tna = pd.read_csv('tna/tna_esg.csv', delimiter=';')
tna['Date'] = tna['Date'].astype(str)
tna['Year'] = tna['Date'].str[6:]
tna['Month'] = tna['Date'].str[3:5]
tna['Year'] = pd.to_numeric(tna['Year'])
tna['Month'] = pd.to_numeric(tna['Month'])
tna.drop('Date', axis=1, inplace=True)

#read expense data of funds
expense = pd.read_csv('exp/exp_esg_v2.csv', delimiter=';')

#sort columns
nav_copy = nav.copy()
exp_copy = expense.copy()
tna_copy = tna.copy()

nav = nav.reindex(sorted(nav.columns[:-2]), axis=1)
expense = expense.reindex(sorted(expense.columns[:-2]), axis=1)
tna = tna.reindex(sorted(tna.columns[:-2]), axis=1)

nav[['Year', 'Month']] = nav_copy[['Year', 'Month']]
expense[['Year', 'Month']] = exp_copy[['Year', 'Month']]
tna[['Year', 'Month']] = tna_copy[['Year', 'Month']]

val_added = []
funds = []


c = len(nav.columns)-2
for i in range(c):

    # nav data
    curr_nav = nav.iloc[:,[i,c, c+1]]
    curr_nav = curr_nav.dropna()
    name = curr_nav.columns[0]

    # tna data
    curr_tna = tna.iloc[:,[i, c, c+1]]
    curr_tna = curr_tna.dropna()
    
    # merge tna with nav
    curr_df = pd.merge(curr_nav, curr_tna, on=['Year', 'Month'])
    
    # expense data
    curr_exp = expense.iloc[:,[i, c, c+1]]
    
    #merge expense
    curr_fd = pd.merge(curr_df, curr_exp, on=['Year', 'Month'])
    curr_fd = curr_fd.dropna()
    
    if len(curr_df) > 29:

        #x = nav, y = tna
        #merge etfs with funds
        curr_df2 = pd.merge(curr_fd, etf, on=['Year', 'Month'])
        curr_df2 = pd.merge(curr_df2, fama[['Month', 'Year', 'RF']])


        #calculate returns, drop nan, and subtract expenses
        curr_df2.iloc[:, [0, -2, -3, -4, -5, -6, -7, -8, -9]]= (np.log(curr_df2.iloc[:, [0, -2, -3, -4, -5, -6, -7, -8, -9]]) - np.log(curr_df2.iloc[:, [0, -2, -3, -4, -5, -6, -7, -8, -9]]).shift(1))*100
        curr_df2 = curr_df2.replace([np.inf, -np.inf], np.nan, inplace=False)
        curr_df2['q-1'] = curr_df2[name+'_y'].shift(1)
        curr_df2 = curr_df2.dropna()
    
        #gross return
        curr_df2.iloc[:, [0, -3, -4, -5, -6, -7, -8, -9, -10]] = curr_df2.iloc[:, [0, -3, -4, -5, -6, -7, -8, -9, -10]].sub(curr_df2['RF'], axis=0)

        # net return
        curr_df2[name+'_x'] = curr_df2[name+'_x']-(curr_df2[name]*1/12)
    
        x = curr_df2.iloc[:, [-3, -4, -5, -6, -7, -8, -9, -10]]
        y = curr_df2[name+'_x']
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
        beta = model.params[1]

        if not beta == 'nan':
            #beta multiplied by excess return of benchmark
            ##fama
            # curr_df2['mean_benchmark'] = (curr_df2[['Mkt-RF', 'SMB', 'HML', 'WML', 'RMW', 'CMA']].mean(axis=1))
            
            curr_df2['mean_benchmark']= curr_df2.iloc[:, [-3, -4, -5, -6, -7, -8, -9, -10]].mean(axis=1)
            curr_df2['Rb'] = beta*curr_df2['mean_benchmark']
            curr_df3 = curr_df2[[name+'_x', 'q-1', 'Rb']]
            curr_df3['V'] = curr_df3['q-1']*(curr_df3[name+'_x']-curr_df3['Rb'])
            s = curr_df3['V'].mean()
            val_added.append(s)
            funds.append(name)

df = pd.DataFrame({
    'Ticker': funds,
    'Value_added': val_added
})
df.to_csv('valadded_esg_vanguard_net.csv')

