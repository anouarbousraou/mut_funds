from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# tna = pd.read_csv('tna_non_esg.csv', delimiter=';', index_col=0, parse_dates=True)
# tna.index = tna.index.to_period('M')
# tna = tna.sort_index()

# nav = pd.read_csv('nav_non_esg.csv', delimiter=';', index_col=0, parse_dates=True)
# nav.index = nav.index.to_period('M')
# nav = nav.sort_index()

# all_dfs =[]
# for i in nav.columns:
#     curr_fund = nav[i]
#     if i in tna.columns:
#         curr_tna = tna[i]
#         curr_fund = curr_fund.dropna()
#         curr_tna = curr_tna.dropna()

#         if len(curr_fund) > 12 and len(curr_tna) > 12:
#             curr_fund = (np.log(curr_fund) - np.log(curr_fund.shift(1)))

#             curr_fund = curr_fund.dropna()
#             df = pd.merge(curr_fund, curr_tna, on='Date')
#             # y = tna
#             df['FundFlow'] = (df[i+'_y'] - ((1+df[i+'_x'])*df[i+'_y'].shift(1)))/df[i+'_y'].shift(1)
#             df = df.dropna()
#             df = df.reset_index()
#             df['ISIN'] = i
#             df = df[['Date', 'FundFlow', 'ISIN']]
#             all_dfs.append(df)
            
# final_df = pd.concat(all_dfs, axis=0, ignore_index=True)            
# final_df.to_csv('fundflows_nonesg.csv')

# ## plot
# ff = pd.read_csv('fundflows_nonesg.csv')
# m = ff[['FundFlow', 'Date']].groupby('Date').mean()
# m.replace([np.inf, -np.inf,], np.nan, inplace=True)
# m = m.dropna()
# m.to_csv('flow_plot_nonesg.csv')


# ffnonesg = pd.read_csv('flow_plot_nonesg.csv', index_col=0, parse_dates=True)
# ffesg = pd.read_csv('flow_plot_esg.csv', index_col=0, parse_dates=True)

# plt.subplot(1,2,1)

# plt.plot(ffesg.index, ffesg['FundFlow'], color='green')
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# plt.xlabel('Time', fontsize=10, labelpad=15)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title('ESG Fund Flows over time', fontsize=11)
# plt.yscale('log')

# plt.subplot(1,2,2)
# plt.plot(ffnonesg.index, ffnonesg['FundFlow'], color='red')
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# plt.xlabel('Time', fontsize=10, labelpad=15)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title('Non-ESG Fund Flows over time', fontsize=11)
# plt.yscale('log')
# plt.show()