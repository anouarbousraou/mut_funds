import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

esg_funds = pd.read_csv("esgfunds_v2.csv", delimiter=';')
ff_factors = pd.read_csv('ff_factors.csv', index_col=0)
ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m%d')

alphas = []
p_values = []
funds = []

for i, v in esg_funds.head(10).iterrows():
    fund = v['Ticker']
    expense = v['Total Expense Ratio']

    try:
        #find ticker
        df = yf.Ticker(fund)
        df = df.history(start='1990-01-01')

        temp_df = pd.DataFrame(index=df.index)
        temp_df[fund] = df['Close'].values

        if len(temp_df.values) >= 252:
            #returns
            ret = (np.log(temp_df) - np.log(temp_df.shift(1)))*100
            ret = pd.merge(ret, ff_factors, on='Date')
            ret = ret.dropna()
            ret['GRET'] = ret[fund] - ret['RF']
            ret['NRET'] = ret['GRET'] - (expense/252)

            #regressions
            x = ret[['Mkt-RF', 'SMB', 'HML']]
            x = sm.add_constant(x)
            model = sm.OLS(ret['NRET'],x).fit(cov_type='HAC', cov_kwds={'maxlags':251})
            #alpha
            alpha = model.params[0]*252
            alphas.append(alpha)
            #p-value
            p_value = model.pvalues[0]
            p_values.append(p_value)
            #funds
            funds.append(fund)
    except:
        print(f'Ticker not found {fund}')

#make dataframe with mutual funds
df = pd.DataFrame(
    {'Ticker': funds,
     'Alpha': alphas,
     'P-value': p_values
    })
df = df.sort_values(by=['Alpha'])
# df.to_csv('endresult_3factor.csv') 

#plot alpha + pvalue vs fund
fig, ax1 = plt.subplots()

ax1.set_xlabel('Fund')
ax1.set_ylabel('Alpha', color='darkred')
ax1.plot(df['Ticker'], df['Alpha'], color='darkred', marker='.', markersize=7)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('P-value', color='blue')
ax2.scatter(df['Ticker'], df['P-value'], color='blue', marker='*')
ax2.tick_params(axis='y')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xticks([])
plt.show()

# plt.figure(figsize=(16,5))
# plt.plot(funds, alphas, color='orange', marker='*')
# plt.scatter(funds, p_values, color='blue', marker='^')
# plt.xticks(fontsize=8, rotation=45)
# plt.yticks(fontsize=12, rotation=0)
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# plt.title('Alpha & P-value per fund')
# plt.xlabel('Fund', fontsize = 13, labelpad = 15)
# plt.legend(['Alpha', 'P-value'])
# plt.show()
