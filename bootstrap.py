import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import scipy.stats
import warnings
warnings.filterwarnings("ignore")

# ##----
boot = pd.read_csv('valadded_nonesg_bootstrap_vanguard_gross.csv', index_col=0)
norm = pd.read_csv('valadded_v3/valadded_nonesg_vanguard_gross.csv', index_col=0)

boot['Value_added']= boot['Value_added']/1000000000
norm['Value_added']= norm['Value_added']/1000000000


x = 0.1
for i in range(9):
    print(f'***For decile {x}***')
    np10 = norm['Value_added'].quantile(x)
    bp10 = boot['Value_added'].quantile(x)

    print(f'P{x} normal {np10}')
    print(f'P{x} boot {bp10}')
    data_np10 = norm[norm['Value_added']>=np10]
    data_bp10 = boot[boot['Value_added']>=bp10]
    
    prob = (np10-bp10)/((data_np10.std()[0]/np.sqrt(len(data_np10))))
    
    print(f' p-value {round(scipy.stats.t.sf(abs(prob), df=len(data_np10)-1),3)}')
    x+=0.1

norm_t = norm.mean()[0]/ (norm.std()[0]/np.sqrt(len(norm)))
boot_t = boot.mean()[0]/ (boot.std()[0]/np.sqrt(len(boot)))
print(norm_t)
print(boot_t)

#---

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

#read etf data
etf = pd.read_csv('nav/vanguard_etf.csv')
etf['Date'] = etf['Date'].astype(str)
etf['Year'] = etf['Date'].str[-4:]
etf['Month'] = etf['Date'].str[3:5]
etf['Year'] = pd.to_numeric(etf['Year'])
etf['Month'] = pd.to_numeric(etf['Month'])

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

alphas = pd.read_csv('alpha_vanguard_nonesg_gross.csv', index_col=0)

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
        #merge etfs with funds
        curr_df = pd.merge(curr_df, etf, on=['Year', 'Month'])
        curr_df = pd.merge(curr_df, fama[['Month', 'Year', 'RF']], on=['Year', 'Month'])
        curr_df = curr_df.dropna()
        #calculate returns, drop nan, and subtract expenses
        curr_df.iloc[:, [0, -2,-3, -4,-5,-6,-7,-8,-9]] = (np.log(curr_df.iloc[:, [0, -2,-3, -4,-5,-6,-7,-8,-9]]) - np.log(curr_df.iloc[:, [0, -2,-3, -4,-5,-6,-7,-8,-9]]).shift(1))*100
        curr_df.iloc[:, [0, -2,-3, -4,-5,-6,-7,-8,-9]] = curr_df.iloc[:, [0, -2,-3, -4,-5,-6,-7,-8,-9]].sub(curr_df['RF'], axis=0)
        curr_df[name+'_x'] = curr_df[name+'_x'] - (curr_df[name]*1/12)
        curr_df['q-1'] = curr_df[name+'_y'].shift(1)
        curr_df = curr_df.dropna()
        #subtract alpha
        curr_alpha = alphas[alphas['Ticker']==name]
        curr_alpha = curr_alpha['Alpha']
        curr_df[name+'_x'] = curr_df[name+'_x'] - curr_alpha.values[0]
        #regression
        x = curr_df.iloc[:, [-3, -4,-5,-6,-7,-8,-9,-10]]
        y = curr_df[name+'_x']
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
        alpha = model.params[0]
        res = model.resid
       #val added 
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
df.to_csv('valadded_nonesg_bootstrap_vanguard_net2.csv')

