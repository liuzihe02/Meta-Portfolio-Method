import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import warnings
import random, math
import riskfolio as rp
import matplotlib.pyplot as plt
import pickle as pkl
import scipy.cluster.hierarchy as hr

#%%
'''
FUNCTIONS FOR CREATING FEATURES
'''
def sharpe_daily(ret):
    '''
    aim: estimate annual sharpe ratio using daily data
    parameters:
        ret: portfolio ret, datetime index, day by day prices (assumes NO NA values, fully filled)
    '''
    return (ret.mean()/ret.std()) * (252**0.5)

def sharpe_monthly(ret):
    '''
    aim: estimate annual sharpe ratio using daily data and monthly returns
    parameters:
        ret: portfolio ret, datetime index, day by day prices (assumes NO NA values, fully filled)
    
    find the first date of each month present, then for each COMPLETE month, calc the cum ret
    note that for BOTH the first and last 'first month date', we just ignore those months
    we ignore data before the first 'first month date' as they do not form a complete month
    we ignore data after the last 'first month date' as they do not form a complete month
    
    returns: the sharpe ratio based on all these cum ret values'''
    #datetime index
    index = ret.groupby([ret.index.year, ret.index.month]).head(1).index[1:] #ignore first date
    #convert to int index
    index = [ret.index.get_loc(x) for x in index]
    
    sharpe=[]
    for i in range(len(index)-1):#ignore last date
        sharpe.append(cum_ret(ret.iloc[index[i]:index[i+1]]))
    sharpe=pd.Series(sharpe)
    return sharpe.mean()/sharpe.std() * (12**0.5)
    

def sharpe_yearly(ret):
    '''
    aim: estimate annual sharpe ratio using daily data and yearly returns
    parameters:
        ret: portfolio ret, datetime index, day by day prices (assumes NO NA values, fully filled)
    
    find the first date of each year present, then for each COMPLETE year, calc the cum ret
    note that for BOTH the first and last 'first year date', we just ignore those dates
    we ignore data before the first 'first year date' as they do not form a complete year
    we ignore data after the last 'first year date' as they do not form a complete year
    
    returns: the sharpe ratio based on all these cum ret values'''
    #datetime indexes
    #ignore first date(may not be the start of a year), but need to keep last 'first year date' for indexing
    index = ret.groupby([ret.index.year]).head(1).index[1:]
    #convert to integer indexes in ret
    index = [ret.index.get_loc(x) for x in index]
    
    sharpe=[]
    for i in range(len(index)-1):#ignore last date
        sharpe.append(cum_ret(ret.iloc[index[i]:index[i+1]]))
    sharpe=pd.Series(sharpe)
    return sharpe.mean()/sharpe.std()

def cagr(ret):
    '''
    aim: Calculate the compound annual growth rate
    parameters:
        ret: portfolio ret, datetime index, DAY BY DAY prices (assumes NO NA values, fully filled)
            can be of any length, simply finds cagr by estimating number of years elapsed, can be decimal years
    returns: the compound annual growth rate, as a fraction, not a percentage'''
    end=(1 + ret).cumprod().iloc[-1]
    num_years=len(ret.index)/252
    return end**(1/num_years)-1

def rea_vol(ret):
    '''
    aim: find the realized volatility for the WHOLE entire period
        is the square of the realized variance
    parameters:
        ret: portfolio return
    
    '''
    return np.sum(np.log(ret+1)**2)**0.5

def mdd(ret):
    '''
    aim: find max drawdown of period
    parameters:
        ret: can be indiv asset returns, in which case this will return a series'''
    cum_returns = (1 +ret).cumprod()
    drawdown =  1 - cum_returns.div(cum_returns.cummax())
    return drawdown.max(axis=0) 

def semi_std(ret):
    '''
    aim: find semi-standard deviation of period, a measure of downside risk
        calcs std of only those returns that fall below mean return
    parameters
        ret: a df of returns, can accept NAN values

    '''
    average = np.nanmean(ret.to_numpy())
    r_below = ret[ret < average]
    return np.sqrt(1/len(r_below) * np.sum((average - r_below)**2))

def coph_corr(port):
    '''
    aim:     calculates the cophenetic correlation coefficient of this clustering structure
        this is the corr coeff btwn distance matrix and cophenetic matrix
    parameters:
        port: a portfolio object that has already gone through optimization

    
    note that this assumes p_dist exists, which is not true for all cov-estimation methods
    returns:first gives a numpy scalar value, then converts it to a float!
    '''
    return hr.cophenet(port.clusters,Y=port.p_dist)[0].item() #check out hr's cophenet method

#%%
'''
FUNCTIONS FOR MANIPULATING PORTFOLIO RETURNS AND WEIGHTS
'''

def port_ret(weights, ret, index_):
    '''
    aim:
        this function applies a set of weights to daily returns of an asset universes
        and creates a portfolio return. note that it makes use of the theoretical property that
        ret(portfolio)= sum[asset_weight * ret(asset)]
    
    parameters:
        weights: a dataframe
            each row of weight in weights shld correspond to an integer index in returns
            e.g.
            1105: [w1,w2,w3]
            1206: [w1_new,w2_new,w3_new]
            This will mean that weight applies on ret from 1105 to 1205, and 1206 is a diff set of weights
    
        ret: a dataframe, DAILY RETURNS OF EACH TICKER
        index_: These are integer indexes of ret, where REBALANCING(CHOOSING NRP/HRP) happens
                note that index_ may be a subset of the index of weights
                because XGboost needs some months of data, can only start ltr
    
    returns:
        a SERIES containing daily portfolio returns, COMBINED into one value
        note that port_returns will automatically make nan all the returns up to the first date in index_
    '''
    ret=ret.copy()
    for i in range(len(index_)):
        if i!=len(index_)-1: #if the index is not the last index
            ret.iloc[index_[i]:index_[i+1]]=weights.loc[index_[i]].to_list()*ret.iloc[index_[i]:index_[i+1]]
        else: #if last number in index_, just apply weights all the way to last row
            ret.iloc[index_[i]:]=weights.loc[index_[i]].to_list()*ret.iloc[index_[i]:]
    #sum all the returns horizontally
    ret=ret.sum(axis=1)
    #make all rows nan up to first rebalancing date
    ret.iloc[:index_[0]]=np.nan
    return ret #will contain nan, not fully filled

def cum_ret(ret):
    '''
    aim: takes in some daily returns (EITHER indiv assets OR combined portfolio) and gives cumulative returns
    
    parameters (if indiv assets returns):
        ret: dataframe,individual asset returns,assumes there are NO NA VALUES (fully filled) 
    returns: a series, indexed by the asset name, gives cum_ret of each asset
    
    parameters (if portfolio returns):
        ret: a series, each day containing portfolio returns
    returns: a float, the cum_ret of the entire portfolio
    '''
    return (1 + ret).cumprod().iloc[-1]-1

def stats_corr(corr):
    '''
    aim: find some stats on the correlation matrix
        1st value is mean of all lower triangular elem, 
        2nd is std of all lower trian elems
        3rd elem is the condition number of this corr matrix, measures how sensitive a matrix is (??)
        4th elem is the determinant of the matrix
    parameters:
        corr: the correlation matrix btwn all asset pairs.
        
    returns: a tuple, containing values in the order above

    '''
    #note that here values is a np array!!
    values=corr.values[np.triu_indices_from(corr.values,1)]
    cond=np.linalg.cond(corr)
    det=np.linalg.det(corr)
    #ddof=1 for unbiased estimator in numpy
    return (values.mean(),values.std(ddof=1),cond,det)

def stats_asset_univ(uni_ret):
    '''
    aim: calc some stats about the asset universe
    parameters:
        uni_ret: THE UNIVERSE' RETURNS', for each asset. Assumes no NA! 
                should be a dataframe!
        
    returns: a tuple of 4 values, as specified in strat'''
    m_m=uni_ret.mean(axis=0).mean()
    s_m=uni_ret.std(axis=0).mean()
    m_s=uni_ret.mean(axis=0).std()
    s_s=uni_ret.std(axis=0).std()
    mdd_m=mdd(uni_ret).mean()
    mdd_s=mdd(uni_ret).std()
    return(m_m,s_m,m_s,s_s,mdd_m,mdd_s)

#%%
'''
MISCELLANOUS FUNCTIONS
'''

def ticker_gen(ticker_pool, uni_len, uni_num, num_share):
    '''
    aim: this helper function generates some universes of tickers from a pool of tickers.
        Each universe can only only share max num_share tickers with any other universes
    
    parameters:
        ticker_pool: a list, the list of possible tickers
        uni_len: the number of tickers in each universe
        uni_num: the number of universes to generate (has an upper bound!)
        num_share: max number each uni can share w/ each other
    returns: a list of lists, each list consisting of asset tickers
    '''
    #this is the current list of universes, a list of lists
    lis=[]
    #if this counter gets too big, its impossible to gen this list.
    count=0
    #while we havent reach the number of uni to generate
    while len(lis)<uni_num:
        count+=1
        
        #create a univ
        cur=random.sample(ticker_pool,uni_len)
        
        #check if cur is valid
        valid=True
        for uni in lis:
            #if alr share more than allowed
            if len(set(uni).intersection(cur))>num_share:
                #terminate loop
                valid=False
        if valid:
            lis.append(cur)
            
        #not possible generate, has iterated over every possible comb
        if count>math.comb(len(ticker_pool),uni_len) * len(ticker_pool) * uni_len:
            print('Not possible to gen such a list!!')
            return
    return lis

def uni_corr(ret):
    '''
    aim: get some insights into this asset universe, through its correlation matrix of all returns
    parameters:
        ret: the returns for the whole universe, indiv assets
            should be ALL the historical returns
            
    returns: a tuple, 1st elem is the correlation matrix, 
                    2nd elem is the mean of upper triangle, 
                    3rd elem is the std of upper triangle
    '''
    port = rp.Portfolio(returns=ret)
    port.assets_stats(method_cov='gerber1')
    corr=rp.AuxFunctions.cov2corr(port.cov)
    values=corr.values[np.triu_indices_from(corr.values,1)]
    return (corr,values.mean(),values.std(ddof=1))