import pandas as pd
import numpy as np
from scipy import rand
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

#read NAV data
nav = pd.read_csv('nav/nav_non_esg.csv', delimiter=';', index_col=0, parse_dates=True)
nav.index = nav.index.to_period('M')
nav = nav.sort_index()
#read other data
chars = pd.read_csv('clean_non_esg.csv', delimiter=';')
my_dict = {}
for i, v in chars.iterrows():
    my_dict[v['ISIN Code']] = v['Total Expense Ratio']
#read etf data
etf = pd.read_csv('nav/nav_etf.csv', index_col=0, parse_dates=True)
etf.index = etf.index.to_period('M')
etf = etf.sort_index()
#make empty lists of variables 
alphas = []
p_values = []
r_sqrd = []
t_values = []
funds = []

#loop through funds, in the NAV data
for i in nav.columns:
    #select current fund & drop missing data
    curr_fd = nav[i]
    curr_fd = curr_fd.dropna()
    ##used to select subperiod
    # curr_fd = curr_fd.loc[(curr_fd.index.year >= 2015) & (curr_fd.index.year < 2020)]
    #returns should have at least 30 months
    if len(curr_fd) >29:
        #create temporary lists to store variables and take average
        temp_alphas = []
        temp_p_values = []
        temp_r_sqrd = []
        temp_t_values = []
        #make the model 500 times
        for j in range(499):
            #select 4 etfs
            random_etf = etf.iloc[:,[random.randint(1,479), random.randint(1,479), random.randint(1,479), random.randint(1,479)]]
            random_etf = random_etf.dropna()
            #merge etfs with funds
            curr = pd.merge(curr_fd, random_etf, on='Date')
            #calculate returns, drop nan, and subtract expenses
            curr = (np.log(curr) - np.log(curr).shift(1))*100
            curr = curr.dropna()       
            expense_ratio = my_dict[i]
            curr[i] = curr[i]-(expense_ratio/12)
            #conduct regressions
            x = curr.iloc[:,1:5]
            y = curr.iloc[:,0]
            x = sm.add_constant(x)
            try:
                model = sm.OLS(y,x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
                #alpha
                alpha = model.params[0]*12
                temp_alphas.append(alpha)
                #p-value
                p_value = model.pvalues[0]
                temp_p_values.append(p_value)
                #r-squared
                r_sq = model.rsquared
                temp_r_sqrd.append(r_sq)
                #t-value
                t_value = model.tvalues[0]
                temp_t_values.append(t_value)
            except:
                print('Failed')    
        #add average variables to final list
        funds.append(i)
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
df.to_csv('alpha_random_nonesg_v2.csv') 

    

