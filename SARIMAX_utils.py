######################## SARIMAX MODEL ####################################

from statsmodels.tsa.stattools import adfuller
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import statsmodels.api as sm

# ADF TEST
def adfuller_test(values):
    results = adfuller(values)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used',
             'Number of Observations Used']
    for val, label in zip(results,labels):
        print(f"{label} : {val}")
    
    if results[1] <= 0.05:
        print("Strong evidence against the null hypothesis (H0). Data is stationary.")
    else:
        print("Weak evidence against the null hypothesis. Data is NOT stationary.")
        
# Compute Exogenous Variable: Dec-Jan Effect
def compute_exog_dec_jan_effect(df):
    # if it is January or December, then set as 1
    # We use December and January effects as our exogenous factor
    if df['month'] == 1 or df['month'] == 12:
        return 1
    else:
        return 0

# SARIMAX Grid Search
def sarimax(train_y,train_exog,all_param):
    results = []
    for param in all_param:
        try:
            mod = SARIMAX(train_y,
                          exog = train_exog,
                          order=param[0],
                          seasonal_order=param[1])
            res = mod.fit(maxiter=200)
            results.append((res,res.aic,param))
            print('Tried out SARIMAX{}x{} - AIC:{}'.format(param[0], param[1], round(res.aic,2)))
        except Exception as e:
            print(e)
            continue
            
    return results