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

# # The false discovery rate is a different *type* of correction than
# # family-wise correction. Instead of controlling for the risk of *any
# # tests* falsely being declared significant under the null hypothesis, FDR
# # will control the *number of tests falsely declared significant as a
# # proportion of the number of all tests declared significant*.
# #
# # A basic idea on how the FDR works is the following.
# #
# # We have got a large number of p values from a set of individual tests.
# # These might be p values from tests on a set of brain voxels.
# #
# # We are trying to a find a p value threshold $\theta$ to do a
# # reasonable job of distinguishing true positive tests from true
# # negatives. p values that are less than or equal to $\theta$ are
# # *detections* and $\theta$ is a *detection threshold*.
# #
# # We want to choose a detection threshold that will only allow a small
# # number of false positive detections.
# #
# # A *detection* can also be called a *discovery*; hence false discovery
# # rate.
# #
# # For the FDR, we will try to find a p value within the family of tests
# # (the set of p values), that we can use as a detection threshold.
# #
# # Let’s look at the p value for a particular test. Let’s say there are
# # $N$ tests, indexed with $i \in 1 .. N$. We look at a test
# # $i$, and consider using p value from this test as a detection
# # threshold; $\theta = p(i)$. The expected number of false positives
# # (FP) in N tests at this detection threshold would be:
# #
# # $$
# # E(FP) = N p(i)
# # $$
# #
# # For example, if we had 100 tests, and the particular p value
# # $p(i)$ was 0.1, then the expected number of false positive
# # detections, thresholding at 0.1, is 0.1 \* 100 = 10.
# #
# # Let’s take some data from a random normal distribution to illustrate:

# # If running in the IPython console, consider running `%matplotlib` to enable
# # interactive plots.  If running in the Jupyter Notebook, use `%matplotlib
# # inline`.

# p_values = df['P-value']
# N = len(p_values)
# normal_distribution = sst.norm(loc=0,scale=1.) #loc is the mean, scale is the variance.

# # To make it easier to show, we sort the p values from smallest to
# # largest:

# p_values = np.sort(p_values)
# i = np.arange(1, N+1) # the 1-based i index of the p values, as in p(i)
# plt.plot(i, p_values, '.')
# plt.xlabel('$i$')
# plt.ylabel('p value')
# plt.show()

# # Notice the (more or less) straight line of p value against $i$
# # index in this case, where there is no signal in the random noise.
# #
# # We want to find a p value threshold $p(i)$ where there is only a
# # small *proportion* of false positives among the detections. For example,
# # we might accept a threshold such that 5% of all detections (discoveries)
# # are likely to be false positives. If $d$ is the number of
# # discoveries at threshold $\theta$, and $q$ is the proportion
# # of false positives we will accept (e.g. 0.05), then we want a threshold
# # $\theta$ such that $E(FP) / d < q$ where $E(x)$ is the
# # expectation of $x$, here the number of FP I would get *on average*
# # if I was to repeat my experiment many times.
# #
# # So - what is $d$ in the plot above? Now that we have ordered the p
# # values, for any index $i$, if we threshold at
# # $\theta \le p(i)$ we will have $i$ detections
# # ($d = i$). Therefore we want to find the largest $p(i)$ such
# # that $E(FP) / i < q$. We know $E(FP) = N p(i)$ so we want
# # the largest $p(i)$ such that:
# #
# # $$
# # N p(i) / i < q \implies p(i) < q i / N
# # $$
# #
# # Let’s take $q$ (the proportion of false discoveries = detections)
# # as 0.05. We plot $q i / N$ (in red) on the same graph as
# # $p(i)$ (in blue):

# q = 0.05
# plt.plot(i, p_values, 'b.', label='$p(i)$')
# plt.plot(i, q * i / N, 'r', label='$q i / N$')
# plt.xlabel('$i$')
# plt.ylabel('$p$')
# plt.legend()
# plt.show()

# # Our job is to look for the largest $p(i)$ value (blue dot) that is
# # still underneath $q i / N$ (the red line).
# #
# # The red line $q i / N$ is the acceptable number of false positives
# # $q i$ as a proportion of all the tests $N$. Further to the
# # right on the red line corresponds to a larger acceptable number of false
# # positives. For example, for $i = 1$, the acceptable number of
# # false positives $q * i$ is $0.05 * 1$, but at
# # $i = 50$, the acceptable number of expected false positives
# # $q * i$ is $0.05 * 50 = 2.5$.
# #
# # Notice that, if only the first p value passes threshold, then
# # $p(1) < q \space 1 \space / \space N$. So, if $q = 0.05$,
# # $p(1) < 0.05 / N$. This is the Bonferroni correction for $N$
# # tests.
# #
# # The FDR becomes more interesting when there is signal in the noise. In
# # this case there will be p values that are smaller than expected on the
# # null hypothesis. This causes the p value line to start below the
# # diagonal on the ordered plot, because of the high density of low p
# # values.

# N_signal = 20
# N_noise = N - N_signal
# noise_z_values = np.random.normal(size=N_noise)
# # Add some signal with very low z scores / p values
# signal_z_values = np.random.normal(loc=-2.5, size=N_signal)
# mixed_z_values = np.sort(np.concatenate((noise_z_values, signal_z_values)))
# mixed_p_values = normal_distribution.cdf(mixed_z_values)
# plt.plot(i, mixed_p_values, 'b.', label='$p(i)$')
# plt.plot(i, q * i / N, 'r', label='$q i / N$')
# plt.xlabel('$i$')
# plt.ylabel('$p$')
# plt.legend()
# plt.show()
# # The interesting part is the beginning of the graph, where the blue p
# # values stay below the red line:

# first_i = i[:30]
# plt.plot(first_i, mixed_p_values[:30], 'b.', label='$p(i)$')
# plt.plot(first_i, q * first_i / N, 'r', label='$q i / N$')
# plt.xlabel('$i$')
# plt.ylabel('$p$')
# plt.legend()
# plt.show()
# # We are looking for the largest $p(i) < qi/N$, which corresponds to
# # the last blue point below the red line.

# below = mixed_p_values < (q * i / N) # True where p(i)<qi/N
# max_below = np.max(np.where(below)[0]) # Max Python array index where p(i)<qi/N
# print('p_i:', mixed_p_values[max_below])
# print('i:', max_below + 1) # Python indices 0-based, we want 1-based

# # The Bonferroni threshold is:

# 0.05 / N

# # In this case, where there is signal in the noise, the FDR threshold
# # *adapts* to the presence of the signal, by taking into account that some
# # values have small enough p values that they can be assumed to be signal,
# # so that there are fewer noise comparisons to correct for, and the
# # threshold is correspondingly less stringent.
# #
# # As the FDR threshold becomes less stringent, the number of detections
# # increases, and the expected number of false positive detections
# # increases, because the FDR controls the *proportion* of false positives
# # in the detections. In our case, the expected number of false positives
# # in the detections is $q i = 0.05 * 9 = 0.45$. In other words, at
# # this threshold, we have a 45% chance of seeing a false positive among
# # the detected positive tests.
# #
# # So, there are a number of interesting properties of the FDR - and some
# # not so interesting if you want to do brain imaging.
# #
# # * In the case of no signal at all, the FDR threshold will be the
# #   Bonferroni threshold
# #
# # * Under some conditions (see Benjamini and Hochberg, JRSS-B 1995), the
# #   FDR threshold can be applied to correlated data
# #
# # * FDR is an “adaptive” threshold
# #
# # Not so “interesting”
# #
# # * FDR can be very variable
# #
# # * When there are lots of true positives, and many detections, the
# #   number of false positive detections increases. This can make FDR
# #   detections more difficult to interpret.