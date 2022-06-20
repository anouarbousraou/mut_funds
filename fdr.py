import pandas as pd
import scipy.stats as sst
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as multi
import seaborn as sns
import scipy.stats
import warnings
warnings.filterwarnings("ignore")

fourfactor = pd.read_csv('alphas_v2/alpha_4factor_esg_12-15.csv', index_col=0)
sixfactor = pd.read_csv('alphas_v2/alpha_6factor_esg_12-15.csv', index_col=0)
random = pd.read_csv('alphas_v3/alpha_random_esg_12-15.csv', index_col=0)
final = pd.merge(random, fourfactor, on='Ticker')
final = pd.merge(final, sixfactor, on='Ticker')
final = final.replace([np.inf, -np.inf], np.nan, inplace=False)
final = final.dropna()

## Random
pos_alpha = final[final['Alpha']>0]
t_value = scipy.stats.t.ppf(q=1-.05/2, df=len(pos_alpha))
pos_alpha = pos_alpha[(pos_alpha['T-value'] < (t_value*-1)) | (pos_alpha['T-value']>t_value)]

neg_alpha = final[final['Alpha']<0]
t_value = scipy.stats.t.ppf(q=1-.05/2, df=len(neg_alpha))
neg_alpha = neg_alpha[(neg_alpha['T-value'] < (t_value*-1)) | (neg_alpha['T-value']>t_value)]

rem_len = len(final) - len(pos_alpha) - len(neg_alpha)
t_value = scipy.stats.t.ppf(q=1-.05/2, df= rem_len)
zero_alpha = final[(final['T-value'] > (t_value*-1)) & (final['T-value'] < t_value)]

len_pos = len(pos_alpha['Alpha'])
len_neg = len(neg_alpha['Alpha'])
len_zero = len(zero_alpha['Alpha'])

res_pos = multi.fdrcorrection(pos_alpha['P-value'], method="n", is_sorted=False)
res_neg = multi.fdrcorrection(neg_alpha['P-value'], method="n", is_sorted=False)
res_zero = multi.fdrcorrection(zero_alpha['P-value'], method="n", is_sorted=False)
fp_pos = np.size(res_pos[0])-np.sum(res_pos[0])
tp_pos = len(res_pos[0])-fp_pos
fp_neg = np.size(res_neg[0])-np.sum(res_neg[0])
tp_neg = len(res_neg[0])-fp_neg
fp_zero = np.size(res_zero[0])-np.sum(res_zero[0])
tp_zero = len(res_zero[0])-fp_zero

print('**********RANDOM**********')
print(f' Count of negative alphas: {len_neg}')
print(f'FDR negative {len_neg*(1-fp_neg/(fp_neg+tp_neg))}')
print(f' Count of zero alphas: {len_zero}')
print(f'FDR zero {fp_zero/(fp_zero+tp_zero)}')
print(f' Count of positive alphas {len_pos}')
print(f'FDR positive {len_pos*(1-fp_pos/(fp_pos+tp_pos))}')

## 4 factor

pos_alpha = final[final['Alpha_f']>0]
t_value = scipy.stats.t.ppf(q=1-.05/2, df=len(pos_alpha))
pos_alpha = pos_alpha[(pos_alpha['T-value_f'] < (t_value*-1)) | (pos_alpha['T-value_f']>t_value)]

neg_alpha = final[final['Alpha_f']<0]
t_value = scipy.stats.t.ppf(q=1-.05/2, df=len(neg_alpha))
neg_alpha = neg_alpha[(neg_alpha['T-value_f'] < (t_value*-1)) | (neg_alpha['T-value_f']>t_value)]

rem_len = len(final) - len(pos_alpha) - len(neg_alpha)
t_value = scipy.stats.t.ppf(q=1-.05/2, df= rem_len)
zero_alpha = final[(final['T-value_f'] > (t_value*-1)) & (final['T-value_f'] < t_value)]

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

print('***********4 Factor************')
print(f' Count of negative alphas: {len_neg}')
print(f'FDR negative {len_neg*(1-fp_neg/(fp_neg+tp_neg))}')
print(f' Count of zero alphas: {len_zero}')
print(f'FDR zero {fp_zero/(fp_zero+tp_zero)}')
print(f' Count of positive alphas {len_pos}')
print(f'FDR positive {len_pos*(1-fp_pos/(fp_pos+tp_pos))}')

## 6 factor

pos_alpha = final[final['Alpha_s']>0]
t_value = scipy.stats.t.ppf(q=1-.05/2, df=len(pos_alpha))
pos_alpha = pos_alpha[(pos_alpha['T-value_s'] < (t_value*-1)) | (pos_alpha['T-value_s']>t_value)]

neg_alpha = final[final['Alpha_s']<0]
t_value = scipy.stats.t.ppf(q=1-.05/2, df=len(neg_alpha))
neg_alpha = neg_alpha[(neg_alpha['T-value_s'] < (t_value*-1)) | (neg_alpha['T-value_s']>t_value)]

rem_len = len(final) - len(pos_alpha) - len(neg_alpha)
t_value = scipy.stats.t.ppf(q=1-.05/2, df= rem_len)
zero_alpha = final[(final['T-value_s'] > (t_value*-1)) & (final['T-value_s'] < t_value)]

len_pos = len(pos_alpha['Alpha_s'])
len_neg = len(neg_alpha['Alpha_s'])
len_zero = len(zero_alpha['Alpha_s'])

res_pos = multi.fdrcorrection(pos_alpha['P-value_s'], method="n", is_sorted=False)
res_neg = multi.fdrcorrection(neg_alpha['P-value_s'], method="n", is_sorted=False)
res_zero = multi.fdrcorrection(zero_alpha['P-value_s'], method="n", is_sorted=False)
fp_pos = np.size(res_pos[0])-np.sum(res_pos[0])
tp_pos = len(res_pos[0])-fp_pos
fp_neg = np.size(res_neg[0])-np.sum(res_neg[0])
tp_neg = len(res_neg[0])-fp_neg
fp_zero = np.size(res_zero[0])-np.sum(res_zero[0])
tp_zero = len(res_zero[0])-fp_zero

print('**********6 Factor**********')
print(f' Count of negative alphas: {len_neg}')
print(f'FDR negative {len_neg*(1-fp_neg/(fp_neg+tp_neg))}')
print(f' Count of zero alphas: {len_zero}')
print(f'FDR zero {fp_zero/(fp_zero+tp_zero)}')
print(f' Count of positive alphas {len_pos}')
print(f'FDR positive {len_pos*(1-fp_pos/(fp_pos+tp_pos))}')



# # ####------------------------------------------------------------------------------------

# #4-factor
# pos_alpha2 = final[final['Alpha_f']>0]
# t_value2 = scipy.stats.t.ppf(q=1-.05/2, df=len(pos_alpha2))
# pos_alpha2 = pos_alpha2[(pos_alpha2['T-value_f'] < (t_value2*-1)) | (pos_alpha2['T-value_f']>t_value2)]

# neg_alpha2 = final[final['Alpha_f']<0]
# t_value2 = scipy.stats.t.ppf(q=1-.05/2, df=len(neg_alpha2))
# neg_alpha2 = neg_alpha2[(neg_alpha2['T-value_f'] < (t_value2*-1)) | (neg_alpha2['T-value_f']>t_value2)]

# rem_len2 = len(final) - len(pos_alpha2) - len(neg_alpha2)
# t_value2 = scipy.stats.t.ppf(q=1-.05/2, df= rem_len2)
# zero_alpha2 = final[(final['T-value_f'] > (t_value2*-1)) & (final['T-value_f'] < t_value2)]

# #6-factor
# pos_alpha3 = final[final['Alpha_s']>0]
# t_value3 = scipy.stats.t.ppf(q=1-.05/2, df=len(pos_alpha3))
# pos_alpha3 = pos_alpha3[(pos_alpha3['T-value_s'] < (t_value3*-1)) | (pos_alpha3['T-value_s']>t_value3)]

# neg_alpha3 = final[final['Alpha_s']<0]
# t_value3 = scipy.stats.t.ppf(q=1-.05/2, df=len(neg_alpha3))
# neg_alpha3 = neg_alpha3[(neg_alpha3['T-value_s'] < (t_value3*-1)) | (neg_alpha3['T-value_s']>t_value3)]

# rem_len3 = len(final) - len(pos_alpha3) - len(neg_alpha3)
# t_value3 = scipy.stats.t.ppf(q=1-.05/2, df= rem_len3)
# zero_alpha3 = final[(final['T-value_s'] > (t_value3*-1)) & (final['T-value_s'] < t_value3)]


# plt.subplot(1,3,1)
# fig = sns.kdeplot(neg_alpha['T-value'], shade=True, color="darkred")
# fig = sns.kdeplot(zero_alpha['T-value'], shade=True, color="darkblue")
# fig = sns.kdeplot(pos_alpha['T-value'], shade=True, color="darkgreen")
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# # plt.axvline(x=[1.96], color='black', linestyle='--')
# # plt.axvline(x=[-1.96], color='black', linestyle='--')
# plt.xlim(-8,5.5)
# fig.set(xlabel=None)
# plt.title('Random ETF Vector')

# plt.subplot(1,3,2)
# fig = sns.kdeplot(neg_alpha2['T-value_f'], shade=True, color="darkred")
# fig = sns.kdeplot(zero_alpha2['T-value_f'], shade=True, color="darkblue")
# fig = sns.kdeplot(pos_alpha2['T-value_f'], shade=True, color="darkgreen")
# # plt.legend(['Unskilled funds', 'Zero-skill funds', 'Skilled funds'])
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# # plt.axvline(x=[1.96], color='black', linestyle='--')
# # plt.axvline(x=[-1.96], color='black', linestyle='--')
# plt.yticks([])
# fig.set(xlabel=None)
# fig.set(ylabel=None)
# plt.xlim(-8,5.5)
# plt.title('Four Factor Model')

# plt.subplot(1,3,3)
# fig = sns.kdeplot(neg_alpha3['T-value_s'], shade=True, color="darkred")
# fig = sns.kdeplot(zero_alpha3['T-value_s'], shade=True, color="darkblue")
# fig = sns.kdeplot(pos_alpha3['T-value_s'], shade=True, color="darkgreen")
# # plt.legend(['Unskilled funds', 'Zero-skill funds', 'Skilled funds'])
# plt.grid(axis='both', alpha=.3)
# plt.gca().spines["top"].set_alpha(0.0)    
# plt.gca().spines["bottom"].set_alpha(0.3)
# plt.gca().spines["right"].set_alpha(0.0)    
# plt.gca().spines["left"].set_alpha(0.3)
# # plt.xticks(np.arange(-12, 15, 3))
# # plt.axvline(x=[1.96], color='black', linestyle='--')
# # plt.axvline(x=[-1.96], color='black', linestyle='--')
# plt.yticks([])
# fig.set(xlabel=None)
# fig.set(ylabel=None)
# plt.title('Six Factor Model')
# plt.xlim(-8,5.5)
# plt.legend(['Unskilled funds', 'Zero-skilled funds', 'Skilled funds'], loc='upper right')
# plt.show()
