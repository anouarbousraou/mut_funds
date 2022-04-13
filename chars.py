#isEsgPopulated
# met yahoo finance en mba data werken?
# 

import pandas as pd
import yfinance as yf

# esg_funds = pd.read_csv("esgfunds_missingexpense.csv", delimiter=';') 

# expense = []
# fund = []

# for i in esg_funds['Ticker']:
#     try:
#         x = yf.Ticker(i)
#         x = x.info
#         e = x.get('annualReportExpenseRatio')
#         expense.append(e)
#         fund.append(i)
#     except:
#         print(f'Not found: {i}')

# df = pd.DataFrame(
#     {'Ticker': fund,
#      'ExpenseRatio': expense,
#     })
# df.to_csv('missingexpenses.csv')         

mba_funds = pd.read_csv("mbafunds.csv", delimiter=';')

esg = []
fund = []

for i in mba_funds['Ticker']:
    try:
        x = yf.Ticker(i)
        x = x.info
        e = x.get('isEsgPopulated')
        esg.append(e)
        fund.append(i)
    except:
        print(f'Not found: {i}')

df = pd.DataFrame(
    {'Ticker': fund,
     'ESGpopulated': esg,
    })
df.to_csv('esgpopulated.csv') 