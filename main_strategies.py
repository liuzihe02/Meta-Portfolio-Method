import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt
import pickle as pkl
from main_utility_functions import *
import xgboost as xgb
from skopt import BayesSearchCV

#load data from gathered from main_build_data
with open('data.pkl','rb') as f:  #rb is read binary code
    cov_period, index_, returns = pkl.load(f)
    
#basically convert all of returns to integer index
index_int=[x for x in range(len(returns.index))]
    
'''
IMPT VARIABLES:
    returns: a dataframe containing daily returns for each asset
    
    index_: Since features like semi_HRP requires measuring recent performance of hrp,
            2 diff daily portfolio returns are computed for hrp and erc where each hrp and erc is rebalanced monthly
            index_ decides when erc/hrp weights are applied (rebalancing)
            contains the INTEGER INDEX of the start of EVERY MONTH
    index_int: defines when weights and datapoints exists, contains the INTEGER INDEX of EVERY DAY
    index_trng: defines when MPM REBALANCING/STRATEGY CHOOSING occurs, contains INTEGER INDEX of month start, aft a period
                this is basically index_ but accounting for perf_period, sharpe_period, trng_period
    
    cov_period: defines the length of time backwards to compute weights and correlation matrix
    perf_period: defines the length of time that ret_b (returns backwards) calculates for performance features,
                and stats of assets universe
    sharpe_period: defines the length of time that ret_f(returns forwards) calcs for sharpe ratio
    trng_period: decides the number of data points and length of time considered for training
    
    weights_ERC: a dictionary containing weights for ERC, each set of weights is a Series indexed by assets
    weights_HRP: '' but for HRP

the dataframe.iloc[k:n] method acts like range and excludes n
the dataframe.loc[k:n] DOES NOT ACT like range, it INCLUDES n!!
'''

#%%
'''PART 1: BUILD DAILY WEIGHTS, CORRELATION MATRIX FOR ERC AND HRP
For every day after the first year, build weights based on past year w/riskfolio
For each strategy and each day there is an associated corr matrix
'''

assets = returns.columns.tolist()
    
#the keys/index to each of these are integer indexes of returns
weights_ERC,weights_HRP,corr,coph = {},{},{},{}

#must skip first date of returns, is NaN
for i in range(cov_period+1,len(returns.index)):
    print('Getting Weights: '+str(i))
    # taking last year (252 trading days per year), EXCLUDING ith date
    ret_temp = returns.iloc[i-cov_period:i] 

    # Build portfolio objects
    port_ERC = rp.Portfolio(returns=ret_temp)
    port_HRP = rp.HCPortfolio(returns=ret_temp)
    
    '''
    Riskfolio Parameters:
        
    Portfolio class:
        note that vanilla MV opt may give error (optimization will not return anyth)
        
        assets_stats:
            method_mu: method to estimate expected returns, 'hist' based on historical data.
            method_cov: method to estimate covariance matrix, 'gerber1' is a new method
        rp_optimization:
            note that rp_optimization uses ERC so the objective is already defined
            rm: risk measure, 'MV' is variance
            obj: objective, Sharpe/MinRisk/MaxRet/Utility (Utility involves l, strength of risk)
    
    Hierarchical Clustering Portfolio (HCPortfolio) class:
        optimization:
            model: Could be HRP or HERC
            codependence: similarity/distance matrix used to build clusters
    '''
    
    port_ERC.assets_stats(method_mu='hist', method_cov='gerber1')
    
    #note that the optimization method will return a single column dataframe, each row per asset
    #select the column as a series
    weights_ERC[i] = port_ERC.rp_optimization(rm='MV')['weights']
    weights_HRP[i] = port_HRP.optimization(model='HRP',codependence='gerber1',covariance='gerber1',obj='Sharpe', rm='MV')['weights']
    
    #add correlation(corr) matrix, both covariance matrix(cov) and corr are df
    #NOTE, if method for getting cov is same for HRP and ERC, they will have the same corr matrix
    #doesnt matter if u choose port_ERC.cov or port_HRP.cov
    corr[i]=rp.AuxFunctions.cov2corr(port_HRP.cov)
    
    #add the cophenetic corr coeff, ONLY applies for HRP to measure quality of clustering
    #note that if the number of assets is too small, coph might give NaN
    coph[i]=coph_corr(port_HRP)
    
#%%
'''PART 2: CREATING X AND Y DATASET FOR XGB
        if index is at 1026, perf_period=2, sharpe_period=3
        ret_b computes data from 1024 to day 1025, perf_period number of days backwards
        ret_f computes data from 1026 to 1028, sharpe_period number of days backwards

    X DATA (FEATURES):
        features are computed using returns from perf_period backwards (ret_b)
        features like semi_ERC with 'ERC' or 'HRP' attached at the end, measure ret_b for that strategy only
        asset universe stats measure ret_b for each asset individually, and combines these stats
        while mean/std_corr, cond, coph, det measure data regarding that date's corr matrix
        
        SEMI_STD_ERC/HRP:
            measures the std of returns below mean return only
        
        AVE_RETURN_ERC/HRP:
            get mean return of that portfolio's daily returns in perf_period
        
        REALISED_VOLATILITY_ERC/HRP:
            alternative measure of variance
            
        MAXIMUM DRAWDOWN:
            longest consecutive decrease (peak-to-trough) in perf_period, divided by the peak
       
        MONTHLY ASSET UNIVERSE STATS:
            mean_assets_mean: get mean return in each asset. Take average across assets
            std_assets_mean: get std return in each asset. Take average across assets
            mean_assets_SD: get mean return in each asset. Take std across assets
            std_assets_SD: get std return in each asset. Take std across assets
            mdd_assets_mean: get mdd in each asset. Take ave across assets
            mdd_assets_SD: get mdd in each asset. Take std across assets
        
        MEAN_CORR, STD_CORR:
           take the mean and std of upper triangular elems in correlation matrix. Measures
           average correlation btwn assets and how varied these corr are
        
        CONDITION NUMBER, DETERMINANT:
            some characteristics about the correlation matrix
        
        COPHENETIC CORRELATION COPEFFICIENT:
            measures quality of clustering
    
    Y DATA (LABELS): 
        SHARPE_ERC/HRP:
            sharpe at index 1026 gives sharpe ratio from day1026 to day 1027 if sharpe_period=2
            estimated using daily returns
'''

data=pd.DataFrame(index=index_int,
                  columns=['sharpe_ERC','sharpe_HRP',
                            'semi_ERC','semi_HRP',
                            'ave_return_ERC', 'ave_return_HRP',
                            'rea_vol_ERC', 'rea_vol_HRP',
                            'mdd_ERC','mdd_HRP',
                            'mean_assets_mean', 'std_assets_mean',
                            'mean_assets_SD', 'std_assets_SD',
                            'mdd_assets_mean', 'mdd_assets_SD',
                            'mean_corr',
                            'std_corr',
                            'cond',
                            'coph_coef',
                            'det'],
                  dtype='float64')

'''Generate portfolio returns(series) for ERC and HRP separately'''
ERC_ret=port_ret(weights_ERC,returns,index_)
HRP_ret=port_ret(weights_HRP,returns,index_)

#both are one month, which is 21 trading days
#I believe sharpe_period should represent the duration between each rebalancing date
perf_period=21
sharpe_period=21

'''
Weird indexing to make it such that it doesnt run into NaN values, yet fill data as much as possible
because model needs to calc perf_period backwards and sharpe_period forwards worth of data

first day of data is the first elem of index_ + perf_period number of days
last day of data is the last elem of index_ - sharpe_period number of days (final i that range iterates over)
u add +1 to range because iloc functions like range, last elem in range is len(returns.index)-sharpe_period
'''
for i in range(index_[0]+perf_period, len(returns.index)-sharpe_period+1): 
    print('Building Dataset: '+str(i))

    '''calc ERC first'''
    #ret_f is returns (sharpe_period) days forwards, INCLUDING current date
    ret_f=ERC_ret.iloc[i:i+sharpe_period]
    data.at[ i, 'sharpe_ERC']=get_sharpe(ret_f,'daily')
    
    #ret_b is returns (perf_period) days backwards, EXCLUDING current date
    ret_b=ERC_ret.iloc[i-perf_period:i]
    data.loc[ i, ['semi_ERC','ave_return_ERC','rea_vol_ERC','mdd_ERC'] ] = \
                    semi_std(ret_b) ,ret_b.dropna().mean(), rea_vol(ret_b), mdd(ret_b)
    
    '''calc HRP now'''
    ret_f=HRP_ret.iloc[i:i+sharpe_period]
    data.at[ i, 'sharpe_HRP']=get_sharpe(ret_f,'daily')
    
    ret_b=HRP_ret.iloc[i-perf_period:i]  
    data.loc[ i, ['semi_HRP','ave_return_HRP','rea_vol_HRP','mdd_HRP'] ] = \
                semi_std(ret_b) , ret_b.dropna().mean(), rea_vol(ret_b), mdd(ret_b)
    
    '''get stats on all assets'''
    #first compute mean and std vertically for each asset, then do another mean horizontally
    data.loc[ i , ['mean_assets_mean', 'std_assets_mean',
                          'mean_assets_SD', 'std_assets_SD',
                          'mdd_assets_mean', 'mdd_assets_SD',] ]\
    =stats_asset_univ(returns.iloc[i-perf_period:i])
    
    '''calc stats based on Correlation matrix(same for both HRP and ERC)'''
    data.loc[ i , ['mean_corr','std_corr','cond','det'] ] = stats_corr(corr[i])
    
    '''add in cophenetic stats'''
    data.loc[ i , 'coph_coef'] = coph[i]


y=data['sharpe_HRP']-data['sharpe_ERC']
X= data.drop(['sharpe_HRP','sharpe_ERC'], axis=1) #del these 2 columns, note drop doesnt modify data

#%%
'''PART 3: TRAINING XGBOOST: CREATE MPM CHOICES
trng period is the number of data points that XGB considers. 
It also is the number of days that XGB looks back, since each day has one data point
'''

#choose trng_period in number of days
trng_period=252 * 8
#indexes of ret where weights_MPM will have data
#since index_ is ascending order, simply pick the suitable indexes
index_trng=[x for x in index_ if x > index_[0]+perf_period+trng_period ]

#choose whether or not to do Bayesian Hyperparameter Optimization (may take a while)
bayes=True

#prep the feature importances
feat_impt=pd.DataFrame([],columns=X.columns,index=index_trng,dtype='float64')

#build MPM weight dict
weights_MPM={}

# create model instance
reg = xgb.XGBRegressor(objective='reg:squarederror',importance_type='gain')

if bayes:
    '''Hyperparam Opt: BayesSearchCV'''
    params = { 'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.5,0.8,1.0],
            'n_estimators': [50,100, 200],
            'min_child_weight':[0,1,3],
            'colsample_bytree':[0.5,0.8,1]}
    #note that here reg is assigned to a BayesSearchCV obj rather than an XGB obj
    reg = BayesSearchCV(
        estimator=reg,
        search_spaces=params,
        n_iter=10,
        verbose=1)

for i in index_trng: #these are integer indexes where MPM strat choosing occurs
    print('Training: '+str(i))
    # fit model
    #note that loc includes last elem! make sure not to include the X you want to predict
    reg.fit(X.loc[i-trng_period:i-1], y.loc[i-trng_period:i-1])
    #make prediction on ith month
    if reg.predict(X.loc[[i]])>0: #[[]] ensures the row is a dataframe, NOT a series
        weights_MPM[i]=weights_HRP[i] #all in this line are series
    else:
        weights_MPM[i]=weights_ERC[i]
    
    if bayes:
        #best_estimator is an attribute of BayesCV, NOT XGB
        feat_impt.loc[i]=reg.best_estimator_.feature_importances_
    else:
        feat_impt.loc[i]=reg.feature_importances_

#%%
'''PART 4: FINAL BACKTEST

Measures of performance: 
1. Cumulative Return
2. Compound Annual Growth Rate
3. Annualised Sharpe Ratio constructed from daily returns
4. Annualised Sharpe Ratio constructed from monthly returns
5. Annualised Sharpe Ratio constructed from yearly returns

Important Variables:
    comp: a dataframe containing performance metrics across strategies
    perf: a float, the performance gain measured by Sharpe_MPM compared to Average(Sharpe_ERC and Sharpe_HRP)
    
    pr_df: the dataframe containing daily returns for each strategy
    feat_impt: the dataframe containing feature importances
    uni_corr: dict, contains corr matrix, mean of corrs and the std of corrs

'''
keys=['ERC','HRP','MPM']
weights_ALL=dict(zip(keys,[weights_ERC,weights_HRP,weights_MPM]))

#portfolio prices
pr_df=pd.DataFrame([],columns=keys)

#display results as dataframe
comp=pd.DataFrame(columns=['Cumulative Return',
                        'Compound Annual Growth Rate',
                         'Annual Sharpe(Daily)',
                         'Annual Sharpe(Monthly)',
                         'Annual Sharpe(Yearly)'],
                  index=keys)

#get performance metrics
for strat in keys:
    pr=port_ret( weights_ALL[strat] ,returns,index_trng).dropna()
    #update portfolio returns
    pr_df[strat]=pr
    #update values
    comp.loc[strat]=[cum_ret(pr),
                   cagr(pr),
                   get_sharpe(pr,'daily'),
                   get_sharpe(pr,'monthly'),
                   get_sharpe(pr,'yearly')]

#perfomance increase of MPM against ave of ERC and HRP, made using daily sharpe estimates
ave=comp.loc[['ERC','HRP'],'Annual Sharpe(Daily)'].mean()
perf=( (comp.loc['MPM','Annual Sharpe(Daily)']-ave)/ave )*100

'''Plotting feature importance (boxplot)'''

#first sort feat_impt's columns, in terms of descending mean.
#i.e. leftmost column has highest mean
fig1, ax1 = plt.subplots()
ax1.set_title('Feature importance by Gain')
feat_plot=feat_impt.dropna().reindex(feat_impt.mean().sort_values().index, axis=1)
#whiskers cover 5th to 98th perc of data
ax1.boxplot(x=feat_plot,vert=False,labels=feat_plot.columns,whis=(2,98),showmeans=True)
fig1.show()

#show portfolio price history for strategies
(pr_df+1).cumprod().plot()

'''Examining this asset universe, using Correlation matrix of ALL historical returns'''
uni_corr_stats=stats_uni_corr(returns.dropna())

