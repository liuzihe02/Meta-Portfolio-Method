import sys
sys.path
sys.path.append('C:\FUTURE\Coding\ML Finance\Portfolio Opt\MPM Project\main_V2')
import pandas as pd
import datetime
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt
import pickle as pkl
from main_utility_functions_V2 import *
import xgboost as xgb
from skopt import BayesSearchCV
import shap

#load data from gathered from main_build_data
with open('data.pkl','rb') as f:  #rb is read binary code
    index_, returns = pkl.load(f)
    
'''
IMPT NOTES:
    index_ find the integer index of the start of every month and defines when REBALANCING occurs
    cov_period defines the length of time backwards for the covariance matrix calcultion
    perf_period defines the length of time that ret_f(returns forward) calcs for sharpe ratio and ret_b
        (returns backwards) calculates for performance features and stats of assets universe
'''
#%%
'''PART 1: BUILD MONTHLY WEIGHTS, COV MATRIX FOR NRP AND HRP
For every month after the first year, build weights based on past year w/riskfolio
For each strategy, each month/weight, there is an associated cov matrix

note that weights_nr and weights_hr are dataframes while corr and coph and dictionaries.
All are indexed by integers of the returns' indexes'

'''

models = {} #dict containing weights, cov matrix, coph_corr
assets = returns.columns.tolist()
index_int=[x for x in range(len(returns.index))]
    
weights_nr,weights_hr,corr,coph = pd.DataFrame([],columns=assets,index=index_int),\
                                    pd.DataFrame([],columns=assets,index=index_int),\
                                    {},\
                                    {}
cov_period=252
#must skip first date of returns, is NaN
for i in range(cov_period+1,len(returns.index)):
    print(i)
    # taking last year (252 trading days per year), Y changes monthly
    Y = returns.iloc[i-cov_period:i,:] 

    # Building the portfolio objects
    port_nr = rp.Portfolio(returns=Y)
    port_hr = rp.HCPortfolio(returns=Y)
    
    # Calculating optimum portfolios

    # Select method to estimate input parameters(only for nrp):
    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='gerber1' # Method to estimate covariance matrix based on historical data.
    port_nr.assets_stats(method_mu=method_mu, method_cov=method_cov)
    
    # Estimate optimal portfolios(only for nrp):
    model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'MV' # Risk measure used, this time will be variance
    
    #here w is a single column dataframe
    w_nr = port_nr.rp_optimization(model=model, rm=rm)
    w_hr = port_hr.optimization(model='HRP',codependence='gerber1',covariance='gerber1',obj='Sharpe', rm='MV')
    
    #FIXME
    #very inefficient way to replace df values
    weights_nr.loc[i]=w_nr.T.squeeze()
    weights_hr.loc[i]=w_hr.T.squeeze()
    
    #append corr matrix, both cov and corr are df
    #NOTE, if method for calc covariance is same for HRP and NRP, they will have the same COV matrix!!!
    #doesnt matter if u choose port_nr.cov or port_hr.cov
    corr[i]=rp.AuxFunctions.cov2corr(port_nr.cov)
    
    #add the cophenetic corr coeff, ONLY applies for HRP to measure quality of clustering, coph_corr is numpy
    #note that if the asset universe is too small, coph might give NaN
    coph[i]=coph_corr(port_hr)

models['NRP'] = weights_nr.copy()
models['HRP'] = weights_hr.copy()
models['corr']=corr.copy()
models['coph']=coph.copy()

#save impt variables as a list, written in binary code
with open('models.pkl','wb') as f:
    pkl.dump([models],f)
    
#%%
'''PART 2: CREATING X AND Y DATASET FOR XGB

PART 2A: OBTAINING DAILY PERFORMANCE (FEATURES) AND SHARPE RATIOS(LABELS)

    X DATA (FEATURES):
        if index is at 1026, data computes data from day 1006(day of prev month) to day 1025
        
        MONTHLY SEMI STD:
            measures the std of deviations below mean only
       
        MONTHLY ASSET UNIVERSE STATS:
            mean_assets_mean: get mean return in each asset. Take average across assets
            std_assets_mean: get std return in each asset. Take average across assets
            mean_assets_SD: get mean return in each asset. Take std across assets
            std_assets_SD: get std return in each asset. Take std across assets
            mdd_assets_mean: get mdd in each asset. Take ave across assets
            mdd_assets_SD: get mdd in each asset. Take std across assets
        
        MONTHLY HRP AND NRP AVE RETURN
            for that month, get mean return of that portfolio's daily returns
            
       COV MEAN AND STD:
           for that month's constructed weights, take the mean and std of upper triangular elems. Measures
           average correlation btwn assets and how varied these corr are
        
    Y DATA (LABELS): 
        MONTHLY SHARPE RATIOS(HRP-NRP):
            sharpe at index 1026 gives sharpe from day1026 to day 1045(day right b4 day of next month)
'''

#load back the weights so I dont have to run this file again
with open('models.pkl','rb') as f:  #rb is read binary code
    models = pkl.load(f)[0]

data_col=['sharpe_NRP','sharpe_HRP',
          'semi_NRP','semi_HRP',
          'ave_return_NRP', 'ave_return_HRP',
          'rea_vol_NRP', 'rea_vol_HRP',
          'mdd_NRP','mdd_HRP',
          'mean_assets_mean', 'std_assets_mean',
          'mean_assets_SD', 'std_assets_SD',
          'mdd_assets_mean', 'mdd_assets_SD',
          'mean_corr',
          'std_corr',
          'cond',
          'coph_coef',
          'det']
data=pd.DataFrame(index=index_int,columns=data_col,dtype='float64')

'''Generate portfolio returns(weighted indiv asset returns) for NRP and HRP separately'''
nr_ret=port_ret(models['NRP'],returns,index_) #models[nrp] stores the weights
hr_ret=port_ret(models['HRP'],returns,index_)

#performance period defines the number of days that ret_f and ret_b considers
perf_period=21
for i in range(index_[0]+perf_period, len(index_int)-perf_period+1): 
    print(i)
    #because model needs to calc perf_period backwards and forwards worth of data
    
    #first day of data is the first elem of index_ + perf_period number of days
    #last day of data is the last elem of index_ - perf_period number of days (final i that range iterates over)
        #u add +1 to range because iloc functions abit differently from range!
    
    '''calc NRP first'''
    #ret_f is returns (perf_period) days forwards, INCLUDING current date
    ret_f=nr_ret.iloc[i:i+perf_period]
    data.at[ i, 'sharpe_NRP']=sharpe_daily(ret_f)
    
    #ret_b is returns (perf_period) days backwards, EXCLUDING current date
    ret_b=nr_ret.iloc[i-perf_period:i]
    data.loc[ i, ['semi_NRP','ave_return_NRP','rea_vol_NRP','mdd_NRP'] ] = \
                    semi_std(ret_b) ,ret_b.dropna().mean(), rea_vol(ret_b), mdd(ret_b)
    
    '''calc HRP now'''
    ret_f=hr_ret.iloc[i:i+perf_period]
    data.at[ i, 'sharpe_HRP']=sharpe_daily(ret_f)
    
    ret_b=hr_ret.iloc[i-perf_period:i]  
    data.loc[ i, ['semi_HRP','ave_return_HRP','rea_vol_HRP','mdd_HRP'] ] = \
                semi_std(ret_b) , ret_b.dropna().mean(), rea_vol(ret_b), mdd(ret_b)
    
    '''get stats on all assets'''
    #first compute mean and std vertically for each asset, then do another mean horizontally
    data.loc[ i , ['mean_assets_mean', 'std_assets_mean',
                          'mean_assets_SD', 'std_assets_SD',
                          'mdd_assets_mean', 'mdd_assets_SD',] ]\
    =stats_asset_univ(returns.iloc[i-perf_period:i])
    
    '''calc stats based on Correlation matrix(same for both HRP and NRP)'''
    data.loc[ i , ['mean_corr','std_corr','cond','det'] ] = stats_corr(models['corr'][i])
    
    '''add in cophenetic stats'''
    data.loc[ i , 'coph_coef'] = models['coph'][i]


y=data['sharpe_HRP']-data['sharpe_NRP']
X= data.drop(['sharpe_HRP','sharpe_NRP'], axis=1) #del these 2 columns, note drop doesnt modify data

#%%
#FIXME
'''PART 3: TRAINING XGBOOST: CREATE MPM CHOICES

trng period is the length of trng for xgboost in n.o. of months
It also affects the starting month at which weights can be given for MPM, therefore NRP and HRP as well
note that xgb only takes in dataframes

note that the indexes for feat_impt and weights_mpm are index_, which are only selected rebalancing dates

'''
#prep the feature importances
feat_impt=pd.DataFrame([],columns=X.columns,index=index_,dtype='float64')

#build mpm weight df
weights_mpm=pd.DataFrame([],columns=assets,index=index_)

'''Hyperparam Opt: BayesSearchCV'''
# create model instance
impt_type='gain'
reg = xgb.XGBRegressor(objective='reg:squarederror',importance_type=impt_type)
params = { 'max_depth': [3, 6, 9],
       'learning_rate': [0.01, 0.1, 0.3],
       'subsample': [0.5,0.8,1.0],
       'n_estimators': [50,100, 200],
       'min_child_weight':[0,1,3],
       'colsample_bytree':[0.5,0.8,1]}

# #note that here reg is assigned to a BayesSearchCV obj rather than an XGB obj
# reg = BayesSearchCV(
#     estimator=reg,
#     search_spaces=params,
#     n_iter=10,
#     verbose=1)

param_list=[]

trng_prd=1008 #2years
#find location of first int in index_ that cross the threshold
for i in range(len(index_)):
    if index_[i]>index_[0]+perf_period+trng_prd:
        start=i
        break
#indexes of ret where mpm_nr has data
trng_index=index_[start:]
    
for i in trng_index: #these are integer indexes of the weights, NOT returns
    print(i)
    # fit model
    reg.fit(X.loc[i-trng_prd:i], y.loc[i-trng_prd:i]) #will fit up to data before ith row
    
    #make prediction on ith month
    if reg.predict(X.loc[[i]])>0: #[[]] ensures the row is a dataframe, NOT a series
        weights_mpm.loc[i]=weights_hr.loc[i] #all in this line are series
    else:
        weights_mpm.loc[i]=weights_nr.loc[i]
        
    # #best_estimator is an attribute of BayesCV, NOT XGB
    # feat_impt.iloc[i]=reg.best_estimator_.feature_importances_
    # #see best params
    # param_list.append(reg.best_estimator_.get_params())
    
    feat_impt.loc[i]=reg.feature_importances_

#%%
'''PART 4: FINAL BACKTEST

Measures of performance: 
1. Cumulative Return
2.Annualised sharpe constructed from daily sharpe*(252)^0.5
3.Annualised sharpe from E(annual ret)/std(annual ret), this method more volatile
'''
cagr_list,ret_list,sharpe_list_d,sharpe_list_m,sharpe_list_y=[],[],[],[],[]

for w in [weights_nr,weights_hr,weights_mpm]:
    pr=port_ret(w,returns,trng_index).dropna()
    #calc cumulative returns
    ret_list.append(port_cum_ret(pr))
    cagr_list.append(cagr(pr))
    sharpe_list_d.append(sharpe_daily(pr))
    sharpe_list_m.append(sharpe_monthly(pr))
    sharpe_list_y.append(sharpe_yearly(pr))

#display results as dataframe
comp=pd.DataFrame(data={'Cumulative Return':ret_list,
                        'Compound Annual Growth Rate':cagr_list,
                         'Annual Sharpe(Daily data)':sharpe_list_d,
                         'Annual Sharpe(Monthly data)':sharpe_list_m,
                         'Annual Sharpe(Yearly data)':sharpe_list_y},
                  index=['NRP','HRP','MPM'])

#perfomance increase of MPM against ave of NRP and HRP, made using daily sharpe estimates
ave=comp.loc[['NRP','HRP'],'Annual Sharpe(Daily data)'].mean()
perf=( (comp.loc['MPM','Annual Sharpe(Daily data)']-ave)/ave )*100

'''Plotting feature importance (boxplot)'''
#hyperparm opt did not improve sharpe/returns,but made the feat impt way more reliable!!

#first sort feat_impt's columns, in terms of descending mean.
#i.e. leftmost column has highest mean
fig1, ax1 = plt.subplots()
ax1.set_title('Feature importance by Gain')
feat_plot=feat_impt.dropna().reindex(feat_impt.mean().sort_values().index, axis=1)
#whiskers cover 5th to 98th perc of data
ax1.boxplot(x=feat_plot,vert=False,labels=feat_plot.columns,whis=(2,98),showmeans=True)
fig1.show()

#show portfolio prices for MPM
plt.figure()
(pr+1).cumprod().dropna().plot()

