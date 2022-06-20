import pandas as pd
import numpy as np
import math

exp_esg = pd.read_csv('exp/exp_nonesg.csv', index_col=0, delimiter=';')
exp_esg['Year'] = exp_esg.index.str[6:]
exp_esg['Month'] = exp_esg.index.str[3:5]
exp_esg['Year'] = pd.to_numeric(exp_esg['Year'])
exp_esg['Month'] = pd.to_numeric(exp_esg['Month'])
x = exp_esg.groupby(['Year', 'Month']).mean()

all_l = []

for i in x.columns:
    curr_df = x[i]
    l = []
    for idx, val in curr_df.items():
        if math.isnan(val):
            temp_temp_df = (exp_esg[['Month', 'Year', i]].loc[(exp_esg[i] > 0)])  
            temp_df = temp_temp_df[(temp_temp_df['Year'] >= idx[0]) & (temp_temp_df['Month'] > idx[1])]
            if not temp_df.empty:
                new_exp = temp_df[i].iloc[0]
                l.append(new_exp)
            else:
                try:
                    new_new_exp = temp_temp_df[i].iloc[-1]
                    l.append(new_new_exp)
                except:
                    l.append('NaN')    
        else:
            l.append(val) 
    all_l.append(l)

final_df = pd.DataFrame(all_l)
final_df = final_df.transpose()
final_df.index = x.index
final_df.columns = x.columns

final_df.to_csv('exp_nonesg_v2.csv')

##--------------------------------------------------------------------------
# l = []
# date = []
# for i,v in exp_esg.iterrows():
#     date.append(i)
#     x = (v[0])
#     if x < 0:
#         temp_temp_df = (exp_esg.loc[(exp_esg['LU1100077442'] > 0)])
#         temp_df = temp_temp_df[temp_temp_df.index > i]
#         if not temp_df.empty:
#             k = temp_df.iloc[0][0]
#             l.append(k)
#         else:    
#             z = temp_temp_df.iloc[-1][0]
#             l.append(z)
#     else:            
#         l.append(x)   

# df = pd.DataFrame({
#     'Date': date,
#     'Hoi': l
# })
# df.to_csv('test_test.csv')    
