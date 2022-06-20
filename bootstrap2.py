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
nav = pd.read_csv('nav/nav_non_esg.csv', delimiter=';')
nav['Date'] = nav['Date'].astype(str)
nav['Year'] = nav['Date'].str[6:]
nav['Month'] = nav['Date'].str[3:5]
nav['Year'] = pd.to_numeric(nav['Year'])
nav['Month'] = pd.to_numeric(nav['Month'])
nav.drop('Date', axis=1, inplace=True)

#read expense data of funds
expense = pd.read_csv('exp/exp_nonesg_v2.csv', delimiter=';')

#read tna data of funds
tna = pd.read_csv('tna/tna_non_esg.csv', delimiter=';')
tna['Date'] = tna['Date'].astype(str)
tna['Year'] = tna['Date'].str[6:]
tna['Month'] = tna['Date'].str[3:5]
tna['Year'] = pd.to_numeric(tna['Year'])
tna['Month'] = pd.to_numeric(tna['Month'])
tna.drop('Date', axis=1, inplace=True)

#sort columns
nav_copy = nav.copy()
exp_copy = expense.copy()
tna_copy = tna.copy()

nav = nav.reindex(sorted(nav.columns[:-2]), axis=1)
tna = tna.reindex(sorted(tna.columns[:-2]), axis=1)
expense = expense.reindex(sorted(expense.columns[:-2]), axis=1)

nav[['Year', 'Month']] = nav_copy[['Year', 'Month']]
tna[['Year', 'Month']] = tna_copy[['Year', 'Month']]
expense[['Year', 'Month']] = exp_copy[['Year', 'Month']]

alphas = pd.read_csv('alphas_v3/alpha_4factor_nonesg_gross.csv', index_col=0)

all_val_added = []
funds = []

c = len(nav.columns)-2
for i in range(c):
    # nav data
    curr_nav = nav.iloc[:,[i,c, c+1]]
    name = curr_nav.columns[0]
    # tna data
    curr_tna = tna.iloc[:,[i, c, c+1]]
    # expense data
    curr_exp = expense.iloc[:,[i, c, c+1]]
    
    # merge tna with nav
    curr_df = pd.merge(curr_nav, curr_tna, on=['Year', 'Month'])
    curr_df = pd.merge(curr_df, curr_exp, on=['Month', 'Year'])
    curr_df = curr_df.dropna()

    if len(curr_df) > 29:
        #x = nav, y = tna
        curr_df = pd.merge(curr_df, fama, on=['Year', 'Month'])
        curr_df = curr_df.dropna()

        #calculate returns, drop nan, and subtract expenses
        curr_df[name+'_x'] = (np.log(curr_df[name+'_x']) - np.log(curr_df[name+'_x']).shift(1))*100
        curr_df[name+'_x'] = curr_df[name+'_x'] -curr_df['RF']
        # curr_df[name+'_x'] = curr_df[name+'_x'] - (curr_df[name]*1/12)
        curr_df['q-1'] = curr_df[name+'_y'].shift(1)
        curr_df = curr_df.dropna()
    
        curr_alpha = alphas[alphas['Ticker']==name]
        curr_alpha = curr_alpha['Alpha_f']

        curr_df[name+'_x'] = curr_df[name+'_x'] - (curr_alpha.values[0]*1/12)
        
        x = curr_df[['Mkt-RF', 'SMB', 'HML', 'WML']]
        y = curr_df[name+'_x']
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
        alpha = model.params[0]
        res = model.resid
        
        val_added = []
        for j in range(1000):
            rand_res = np.random.choice(res.values)     
            curr_df['V'] = curr_df['q-1']*(alpha + rand_res)
            s = curr_df['V'].mean()
            val_added.append(s)

        all_val_added.append(np.mean(val_added))
        funds.append(name)


df = pd.DataFrame({
    'Ticker': funds,
    'Value_added': all_val_added
})
df.to_csv('valadded_nonesg_bootstrap_4factor_gross.csv')