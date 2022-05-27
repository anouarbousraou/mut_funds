import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

tna = pd.read_csv('tna_non_esg.csv', delimiter=';')
tna['Date'] = tna['Date'].astype(str)
tna['Year'] = tna['Date'].str[6:]
tna['Month'] = tna['Date'].str[3:5]
tna.drop('Date', axis=1, inplace=True)

nav = pd.read_csv('nav_non_esg.csv', delimiter=';')
nav['Year'] = nav['Date'].str[6:]
nav['Month'] = nav['Date'].str[3:5]
nav.drop('Date', axis=1, inplace=True)

fama = pd.read_csv('fama_french.csv', delimiter=';')
fama['Date'] = fama['Date'].astype(str)
fama['Year'] = fama['Date'].str[:4]
fama['Month'] = fama['Date'].str[4:]
# #used for net returns
# chars = pd.read_csv('clean_non_esg.csv', delimiter=';')
# my_dict = {}
# for i, v in chars.iterrows():
#     my_dict[v['ISIN Code']] = v['Total Expense Ratio']

val_added = []
funds = []

c = len(nav.columns)-2
for i in range(c):
    curr_nav = nav.iloc[:,[i,7485,7486]]
    name = curr_nav.columns[0]
    curr_nav = curr_nav.dropna()
    curr_tna = tna.iloc[:,[i, 7485, 7486]]
    curr_tna = curr_tna.dropna()
    curr_df = pd.merge(curr_nav, curr_tna, on=['Year', 'Month'])
    if len(curr_df) > 29:
        ## x = nav, y = tna
        curr_df2 = pd.merge(curr_df, fama, on=['Year', 'Month'])
        curr_df2.iloc[:,0] = (np.log(curr_df2.iloc[:,0]) - np.log(curr_df2.iloc[:,0]).shift(1))*100
        curr_df2['q-1'] = curr_df2[name+'_y'].shift(1)
        curr_df2 = curr_df2.dropna()
        curr_df2[name+'_x'] = curr_df2[name+'_x'] - curr_df2['RF']
        # #make net return
        # expense_ratio = my_dict[name]
        # curr_df2[name+'_x'] = curr_df2[name+'_x']-(expense_ratio/12)

        x = curr_df2[['Mkt-RF', 'SMB', 'HML', 'WML', 'RMW', 'CMA']]
        y = curr_df2[name+'_x']
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
        beta = model.params[1]
#         cov = curr_df2[[name+'_x', 'Mkt-RF']].cov()
#         var = curr_df2['Mkt-RF'].var()
#         beta = (cov.values[0][1])/var
        if not beta == 'nan':
            curr_df2['Rb'] = beta*curr_df2['Mkt-RF']
            curr_df3 = curr_df2[[name+'_x', 'Mkt-RF', 'q-1', 'Rb']]
            curr_df3['V'] = curr_df3['q-1']*(curr_df3[name+'_x']-curr_df3['Rb'])
            s = curr_df3['V'].mean()
            val_added.append(s)
            funds.append(name)

df = pd.DataFrame({
    'Ticker': funds,
    'Value_added': val_added
})

df.to_csv('valadded_nonesg_6fact_gross.csv')
