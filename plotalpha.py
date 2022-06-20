import datetime
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#read data created from previous two scripts with the alphas
fourfact = pd.read_csv('alphas_v2/alpha_4factor_esg.csv', index_col=0)
random = pd.read_csv('alphas_v3/alpha_random_esg.csv', index_col=0)
sixfact = pd.read_csv('alphas_v2/alpha_6factor_esg.csv', index_col=0)
# merge data such that all funds are analyzed with 3 models
esg_all = pd.merge(random, fourfact, on='Ticker')
esg_all = pd.merge(esg_all, sixfact, on='Ticker')
esg_all = esg_all.replace([np.inf, -np.inf], np.nan, inplace=False)
esg_all = esg_all.sort_values(by='Ticker')
esg_all = esg_all.dropna()

#read data created from previous two scripts with the alphas
fourfact2 = pd.read_csv('alphas_v2/alpha_4factor_nonesg.csv', index_col=0)
random2 = pd.read_csv('alphas_v3/alpha_random_nonesg.csv', index_col=0)
sixfact2 = pd.read_csv('alphas_v2/alpha_6factor_nonesg.csv', index_col=0)
# merge data such that all funds are analyzed with 3 models
nonesg_all = pd.merge(random2, fourfact2, on='Ticker')
nonesg_all = pd.merge(nonesg_all, sixfact2, on='Ticker')
nonesg_all = nonesg_all.replace([np.inf, -np.inf], np.nan, inplace=False)
nonesg_all = nonesg_all.sort_values(by='Ticker')
nonesg_all = nonesg_all.dropna()


#make the plots
plt.figure(figsize=(18,4))
plt.subplot(2, 3, 1)
plt.scatter(esg_all['Ticker'], esg_all['Alpha'], color='blue', marker='*')
plt.yticks(np.arange(-175, 175, 25))
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(["Random ETF's"], loc='upper left')
plt.ylabel('Net alphas in %', fontsize=10, labelpad=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks([])

plt.subplot(2, 3, 2)
plt.scatter(esg_all['Ticker'], esg_all['Alpha_f'], color='green', marker='x')
plt.yticks(np.arange(-175, 175, 25))
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(['Four Factor Model'], loc='upper left')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks([])
plt.title('ESG Mutual Funds',)
plt.subplot(2, 3, 3)

plt.scatter(esg_all['Ticker'], esg_all['Alpha_s'], color='red', marker='.')
plt.yticks(np.arange(-175, 175, 25))
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(['Six Factor Model'], loc='upper left')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks([])

plt.subplot(2, 3, 4)
plt.scatter(nonesg_all['Ticker'], nonesg_all['Alpha'], color='blue', marker='*')
plt.yticks(np.arange(-175, 175, 25))
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(["Random ETF's"], loc='upper left')
plt.ylabel('Net alphas in %', fontsize=10, labelpad=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks([])


plt.subplot(2, 3, 5)
plt.scatter(nonesg_all['Ticker'], nonesg_all['Alpha_f'], color='green', marker='x')
plt.yticks(np.arange(-175, 175, 25))
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(['Four Factor Model'], loc='upper left')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks([])
plt.title('Non-ESG Mutual Funds')

plt.subplot(2, 3, 6)
plt.scatter(nonesg_all['Ticker'], nonesg_all['Alpha_s'], color='red', marker='.')
plt.yticks(np.arange(-175, 175, 25))
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.legend(['Six Factor Model'], loc='upper left')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks([])
plt.show()
