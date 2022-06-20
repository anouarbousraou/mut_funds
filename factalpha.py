import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mstats
from datetime import datetime
import statsmodels.api as sm
import numpy as np
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

#sort columns
nav_copy = nav.copy()
exp_copy = expense.copy()
nav = nav.reindex(sorted(nav.columns[:-2]), axis=1)
expense = expense.reindex(sorted(expense.columns[:-2]), axis=1)

nav[['Year', 'Month']] = nav_copy[['Year', 'Month']]
expense[['Year', 'Month']] = exp_copy[['Year', 'Month']]

#make empty lists where variables will be stored
alphas = []
p_values = []
r_sqrd = []
t_values = []
funds = []

# loop through all funds
c = len(nav.columns)-2
for i in range(c):
    #select current fund, with month and year as columns, merge with expense, and drop na 7485, 7486 - 6602, 6603
    curr_fd = nav.iloc[:,[i,c,c+1]]
    curr_exp = expense.iloc[:,[i, c, c+1]]
    curr_fd = pd.merge(curr_fd, curr_exp, on=['Year', 'Month'])
    curr_fd = curr_fd.dropna()

    ## code used to select subperiods
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2000) | (curr_fd['Year'] == 2001)| (curr_fd['Year'] == 2002)| (curr_fd['Year'] == 2003)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2004) | (curr_fd['Year'] == 2005)| (curr_fd['Year'] == 2006)| (curr_fd['Year'] == 2007)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2008) | (curr_fd['Year'] == 2009)| (curr_fd['Year'] == 2010)| (curr_fd['Year'] == 2011)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2012) | (curr_fd['Year'] == 2013)| (curr_fd['Year'] == 2014)| (curr_fd['Year'] == 2015)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2016) | (curr_fd['Year'] == 2017)| (curr_fd['Year'] == 2018)| (curr_fd['Year'] == 2019)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2020) | (curr_fd['Year'] == 2021)| (curr_fd['Year'] == 2022)]

    #funds should have 30 months of prices, 19 for 2020-2022
    if len(curr_fd) >29:

        name = curr_fd.iloc[:,0].name
        name = name[:-2]
    
        #calculate return, drop na, merge with fama data, and calculate net return
        curr_fd[name+'_x'] = (np.log(curr_fd[name+'_x']) - np.log(curr_fd[name+'_x']).shift(1))*100
        curr_fd = curr_fd.dropna()
        curr = pd.merge(curr_fd, fama, on=['Year', 'Month']) 

        # # return minus expense, x = nav, y = expense
        # curr[name+'_x'] = curr[name+'_x'] - (curr[name+'_y']*(1/12))
    
        # return minus RF
        curr[name+'_x'] = curr[name+'_x'] - curr['RF']

        #regression
        x = curr[['Mkt-RF','SMB','HML','WML']]
        y = curr[name+'_x']
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
        #alpha
        alpha = model.params[0]*12
        alphas.append(alpha)
        #p-value
        p_value = model.pvalues[0]
        p_values.append(p_value)
        #R-squared
        r_sq = model.rsquared
        r_sqrd.append(r_sq)
        #T-value
        t_value = model.tvalues[0]
        t_values.append(t_value)
        #funds
        funds.append(name)

#store variables in dataframe
df = pd.DataFrame(
{'Ticker': funds,
'Alpha_f': alphas,
'T-value_f': t_values,
'R-squared_f': r_sqrd,
'P-value_f': p_values
})

df = df.sort_values(by=['Alpha_f'])
df.to_csv('alpha_4factor_nonesg_gross.csv')
