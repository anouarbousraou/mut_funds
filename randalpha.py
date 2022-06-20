import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
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

# #read etf data
etf = pd.read_csv('nav/nav_etf.csv')
etf['Date'] = etf['Date'].astype(str)
etf['Year'] = etf['Date'].str[0:4]
etf['Month'] = etf['Date'].str[5:7]
etf['Year'] = pd.to_numeric(etf['Year'])
etf['Month'] = pd.to_numeric(etf['Month'])

#read expense data of funds
expense = pd.read_csv('exp/exp_nonesg_v2.csv', delimiter=';')

#sort columns
nav_copy = nav.copy()
exp_copy = expense.copy()
nav = nav.reindex(sorted(nav.columns[:-2]), axis=1)
expense = expense.reindex(sorted(expense.columns[:-2]), axis=1)

nav[['Year', 'Month']] = nav_copy[['Year', 'Month']]
expense[['Year', 'Month']] = exp_copy[['Year', 'Month']]

#make empty lists of variables 
alphas = []
p_values = []
r_sqrd = []
t_values = []
funds = []

# loop through all funds
c = len(nav.columns)-2

for i in range(c):
    #select current fund, with month and year as columns, merge with expense, and drop na 7485, 7486 - 6602, 6603
    curr_fd = nav.iloc[:,[i,c,(c+1)]]
    curr_exp = expense.iloc[:,[i, c, (c+1)]]
    curr_fd = pd.merge(curr_fd, curr_exp, on=['Year', 'Month'])
    curr_fd = curr_fd.dropna()

    ## code used to select subperiods
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2000) | (curr_fd['Year'] == 2001)| (curr_fd['Year'] == 2002)| (curr_fd['Year'] == 2003)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2004) | (curr_fd['Year'] == 2005)| (curr_fd['Year'] == 2006)| (curr_fd['Year'] == 2007)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2008) | (curr_fd['Year'] == 2009)| (curr_fd['Year'] == 2010)| (curr_fd['Year'] == 2011)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2012) | (curr_fd['Year'] == 2013)| (curr_fd['Year'] == 2014)| (curr_fd['Year'] == 2015)]
    # curr_fd = curr_fd.loc[(curr_fd['Year'] == 2016) | (curr_fd['Year'] == 2017)| (curr_fd['Year'] == 2018)| (curr_fd['Year'] == 2019)]
    curr_fd = curr_fd.loc[(curr_fd['Year'] == 2020) | (curr_fd['Year'] == 2021)| (curr_fd['Year'] == 2022)]

    #returns should have at least 30 months
    if len(curr_fd) >29:

        #create temporary lists to store variables and take average
        temp_alphas = []
        temp_p_values = []
        temp_r_sqrd = []
        temp_t_values = []

        #make the model 500 times
        for j in range(499):

            name = curr_fd.iloc[:,0].name
            name = name[:-2]

            #select 4 etfs
            random_etf = etf.iloc[:,[random.randint(1,479), random.randint(1,479), random.randint(1,479), random.randint(1,479), 481, 482]]
            random_etf = random_etf.dropna()
        
            #merge etfs with funds
            curr = pd.merge(curr_fd, random_etf, on=['Year', 'Month'])
            curr = pd.merge(curr, fama[['Year', 'Month', 'RF']], on=['Year', 'Month'])

            #calculate returns, drop nan, and subtract expenses
            curr.iloc[:, [0, 4, 5, 6, 7]] = (np.log(curr.iloc[:, [0, 4, 5, 6, 7]]) - np.log(curr.iloc[:, [0, 4, 5, 6, 7]]).shift(1))*100
            curr = curr.replace([np.inf, -np.inf], np.nan, inplace=False)
            curr = curr.dropna()    
    
            #gross return
            curr.iloc[:, [0, 4, 5, 6, 7]] = curr.iloc[:, [0, 4, 5, 6, 7]].sub(curr['RF'], axis=0)
        
            #net return    
            curr[name+'_x'] = curr[name+'_x']-(curr[name+'_y']*1/12)
        
            if len(curr) > 11:
                #conduct regressions
                x = curr.iloc[:,4:8]
                y = curr[name+'_x']
                x = sm.add_constant(x)
                model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
                #alpha
                alpha = model.params[0]*12
                temp_alphas.append(alpha)
                #p-value
                p_value = model.pvalues[0]
                temp_p_values.append(p_value)
                #r-squared
                r_sq = model.rsquared_adj
                temp_r_sqrd.append(r_sq)
                #t-value
                t_value = model.tvalues[0]
                temp_t_values.append(t_value)

        #add average variables to final list
        funds.append(name)
        alphas.append(np.mean(temp_alphas))
        p_values.append(np.mean(temp_p_values))
        r_sqrd.append(np.mean(temp_r_sqrd))
        t_values.append(np.mean(temp_t_values))

#make a dataframe with final variables
df = pd.DataFrame(
    {'Ticker': funds,
    'Alpha': alphas,
    'T-value': t_values,
    'R-squared': r_sqrd,
    'P-value': p_values
    })
#export dataframe to csv
df = df.sort_values(by=['Alpha'])
df.to_csv('alpha_random_nonesg_20-22.csv') 

    

