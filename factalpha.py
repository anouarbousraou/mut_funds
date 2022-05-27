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
#read NAV data of funds
nav = pd.read_csv('nav_esg.csv', delimiter=';')
nav['Date'] = nav['Date'].astype(str)
nav['Year'] = nav['Date'].str[6:]
nav['Month'] = nav['Date'].str[3:5]
nav.drop('Date', axis=1, inplace=True)
#read other data
chars = pd.read_csv('clean_esg.csv', delimiter=';')
my_dict = {}
for i, v in chars.iterrows():
    my_dict[v['ISIN Code']] = v['Total Expense Ratio']
#make empty lists where variables will be stored
alphas = []
p_values = []
r_sqrd = []
t_values = []
funds = []
#loop through all funds
c = len(nav.columns)-2
for i in range(c):
    #select current fund, with month and year as columns, and drop na
    curr_fd = nav.iloc[:,[i,6602,6603]]
    curr_fd = curr_fd.dropna()
    ## code used to select subperiods
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == '2000') | (curr_fd['Year'] == '2001')| (curr_fd['Year'] == '2002')| (curr_fd['Year'] == '2003')| (curr_fd['Year'] == '2004')]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == '2005') | (curr_fd['Year'] == '2006')| (curr_fd['Year'] == '2007')| (curr_fd['Year'] == '2008')| (curr_fd['Year'] == '2009')]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == '2010') | (curr_fd['Year'] == '2011')| (curr_fd['Year'] == '2012')| (curr_fd['Year'] == '2013')| (curr_fd['Year'] == '2014')]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == '2015') | (curr_fd['Year'] == '2016')| (curr_fd['Year'] == '2017')| (curr_fd['Year'] == '2018')| (curr_fd['Year'] == '2019')]
    #funds should have 30 months of prices
    if len(curr_fd) >29:
        #calculate return, drop na, merge with fama data, and calculate net return
        curr_fd.iloc[:,0] = (np.log(curr_fd.iloc[:,0]) - np.log(curr_fd.iloc[:,0]).shift(1))*100
        curr_fd = curr_fd.dropna()
        curr = pd.merge(curr_fd, fama, on=['Year', 'Month']) 
        name = curr_fd.iloc[:,0].name
        expense_ratio = my_dict[name]
        curr.iloc[:,0] = curr.iloc[:,0]-(expense_ratio/12)
        curr.iloc[:,0] = curr.iloc[:,0]- curr.iloc[:,4]
        #regression
        x = curr.iloc[:,5:]
        y = curr.iloc[:,0]
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
        funds.append(name)
#store variables in dataframe
df = pd.DataFrame(
{'Ticker': funds,
'Alpha': alphas,
'T-value': t_values,
'R-squared': r_sqrd,
'P-value': p_values
})

df = df.sort_values(by=['Alpha'])
df.to_csv('alpha_6factor_esg.csv')
