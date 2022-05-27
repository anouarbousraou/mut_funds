from turtle import pos
import pandas as pd
import scipy.stats as sst
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as multi
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fourfactor = pd.read_csv('alpha_4factor_nonesg.csv', index_col=0)
sixfactor = pd.read_csv('alpha_6factor_nonesg.csv', index_col=0)
random = pd.read_csv('alpha_random_nonesg.csv', index_col=0)
final = pd.merge(random, fourfactor, on='Ticker')
final = pd.merge(final, sixfactor, on='Ticker')
final = final.dropna()

pos_alpha = final[final['Alpha_f']>0]
pos_alpha = pos_alpha[(pos_alpha['T-value_f'] < -1.96) | (pos_alpha['T-value_f']>1.96)]

neg_alpha = final[final['Alpha_f']<0]
neg_alpha = neg_alpha[(neg_alpha['T-value_f'] < -1.96) | (neg_alpha['T-value_f']>1.96)]

zero_alpha = final[(final['T-value_f'] > -1.96) & (final['T-value_f'] < 1.96)]

len_pos = len(pos_alpha['Alpha_f'])
len_neg = len(neg_alpha['Alpha_f'])
len_zero = len(zero_alpha['Alpha_f'])

res_pos = multi.fdrcorrection(pos_alpha['P-value_f'], method="n", is_sorted=False)
res_neg = multi.fdrcorrection(neg_alpha['P-value_f'], method="n", is_sorted=False)
res_zero = multi.fdrcorrection(zero_alpha['P-value_f'], method="n", is_sorted=False)
fp_pos = np.size(res_pos[0])-np.sum(res_pos[0])
tp_pos = len(res_pos[0])-fp_pos
fp_neg = np.size(res_neg[0])-np.sum(res_neg[0])
tp_neg = len(res_neg[0])-fp_neg
fp_zero = np.size(res_zero[0])-np.sum(res_zero[0])
tp_zero = len(res_zero[0])-fp_zero

# #print(f'For {i}')
print(f' Count of negative alphas: {len_neg}')
print(f'FDR negative {fp_neg/(fp_neg+tp_neg)}')
print(f' Count of zero alphas: {len_zero}')
print(f'FDR zero {fp_zero/(fp_zero+tp_zero)}')
print(f' Count of positive alphas {len_pos}')
print(f'FDR positive {fp_pos/(fp_pos+tp_pos)}')


####------------------------------------------------------------------------------------
# final = final.reset_index()
# final.drop('index', axis=1, inplace=True)

# y = final[final['Alpha']>0 &
#     ((final['T-value'] > 1.96) |
#           (final['T-value'] < -1.96))]
# x = final[final['Alpha']<0 &
#     ((final['T-value'] > 1.96) |
#           (final['T-value'] < -1.96))]          
# z = final[(final['T-value'] > -1.96) &
#           (final['T-value'] < 1.96)]

# yy = final[final['Alpha_f']>0 &
#     ((final['T-value_f'] > 1.96) |
#           (final['T-value_f'] < -1.96))]
# xx = final[final['Alpha_f']<0 &
#     ((final['T-value_f'] > 1.96) |
#           (final['T-value_f'] < -1.96))]          
# zz = final[(final['T-value_f'] > -1.96) &
#           (final['T-value_f'] < 1.96)]

# yyy = final[final['Alpha_s']>0 &
#     ((final['T-value_s'] > 1.96) |
#           (final['T-value_s'] < -1.96))]
# xxx = final[final['Alpha_s']<0 &
#     ((final['T-value_s'] > 1.96) |
#           (final['T-value_s'] < -1.96))]          
# zzz = final[(final['T-value_s'] > -1.96) &
#           (final['T-value_s'] < 1.96)]          

# plt.subplot(1,3,1)
# fig = sns.kdeplot(x['T-value'], shade=True, color="r")
# fig = sns.kdeplot(z['T-value'], shade=True, color="b")
# fig = sns.kdeplot(y['T-value'], shade=True, color="g")
# plt.legend(['Unskilled funds', 'Zero-skill funds', 'Skilled funds'])
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# plt.xticks(np.arange(-12, 15, 3))
# plt.axvline(x=[1.96], color='black', linestyle='--')
# plt.axvline(x=[-1.96], color='black', linestyle='--')
# plt.title('Random ETF Vector')

# plt.subplot(1,3,2)
# fig = sns.kdeplot(xx['T-value_f'], shade=True, color="r")
# fig = sns.kdeplot(zz['T-value_f'], shade=True, color="b")
# fig = sns.kdeplot(yy['T-value_f'], shade=True, color="g")
# # plt.legend(['Unskilled funds', 'Zero-skill funds', 'Skilled funds'])
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# plt.xticks(np.arange(-12, 15, 3))
# plt.axvline(x=[1.96], color='black', linestyle='--')
# plt.axvline(x=[-1.96], color='black', linestyle='--')
# plt.yticks([])
# plt.title('Four Factor Model')

# plt.subplot(1,3,3)
# fig = sns.kdeplot(xxx['T-value_s'], shade=True, color="r")
# fig = sns.kdeplot(zzz['T-value_s'], shade=True, color="b")
# fig = sns.kdeplot(yyy['T-value_s'], shade=True, color="g")
# # plt.legend(['Unskilled funds', 'Zero-skill funds', 'Skilled funds'])
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# plt.xticks(np.arange(-12, 15, 3))
# plt.axvline(x=[1.96], color='black', linestyle='--')
# plt.axvline(x=[-1.96], color='black', linestyle='--')
# plt.yticks([])
# plt.title('Six Factor Model')
# plt.show()

###---------------------------------------------------------------------------------------------------

