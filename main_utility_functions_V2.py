import pandas as pd
import numpy as np
import random, math
import riskfolio as rp
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hr

#%%
'''
FUNCTIONS FOR CREATING FEATURES
'''
def get_sharpe(ret, freq):
    '''
    for e.g. if monthly selected:
        find the first date of each month present, then for each COMPLETE month, calc the cum ret
        note that for BOTH the first and last 'first-day-of-month date', we just ignore those months
        ignore data in the month of the first 'first-day-of-month date' because they may not form a complete month
        ignore data after the last 'first-day-of-month date' as they may not form a complete month
        
    Parameters
    ----------
    ret : dataframe, portfolio ret, datetime index, day by day prices 
        (assumes NO NA values, fully filled)
    freq : string, either 'daily' , 'monthly' or 'yearly'
        which frequency to use to estimate the sharpe ratio.
        if daily, daily returns are used and are annualised
        if monthly, monthly returns are used and are annualised
        if yearly, yearly returns are used, already annualised

    Returns
    -------
    the sharpe ratio based on all these cum ret values
    '''
    #notify if NaN or zero exists:
    if ret.isnull().values.any() or 0 in ret.values:
        print('There exists NaN or 0 values in get_sharpe!')
    
    #daily frequency selected
    if freq=='daily':
        return (ret.mean()/ret.std()) * (252**0.5)
    
    #monthly frequency selected
    elif freq=='monthly':
        #datetime index
        #ignore first date(usually not be the start of a year), but need to keep last 'first year date' for indexing
        index = ret.groupby([ret.index.year, ret.index.month]).head(1).index[1:]
        #convert to int index
        index = [ret.index.get_loc(x) for x in index]

        sharpe=pd.Series([ cum_ret(ret.iloc[index[i]:index[i+1]]) for i in range(len(index)-1) ])
        
        return sharpe.mean()/sharpe.std() * (12**0.5)
    
    #yearly frequency selected
    elif freq=='yearly':
        #datetime indexes
        #this time filther out the first day of each year
        index = ret.groupby([ret.index.year]).head(1).index[1:]
        #convert to integer indexes in ret
        index = [ret.index.get_loc(x) for x in index]
        
        sharpe=pd.Series([ cum_ret(ret.iloc[index[i]:index[i+1]]) for i in range(len(index)-1) ])
        
        return sharpe.mean()/sharpe.std()

def cagr(ret):
    '''
    Calculate the compound annual growth rate
    
    Parameters
    ----------
    ret : dataframe, portfolio ret, datetime index, DAY BY DAY prices (assumes NO NA values, fully filled)
            can be of any length, simply finds cagr by estimating number of years elapsed, can be decimal years

    Returns
    -------
    float
        the compound annual growth rate, as a fraction, not a percentage
    '''
    
    #notify if NaN exists:
    if ret.isnull().values.any() or 0 in ret.values:
        print('There exists NaN or 0 values in cagr!')
        
    end=(1 + ret).cumprod().iloc[-1]
    num_years=len(ret.index)/252
    return end**(1/num_years)-1

def rea_vol(ret):
    '''
    find the realized volatility for the WHOLE entire period
        is the root of the realized variance
    Parameters
    ----------
    ret : dataframe, portfolio return

    Returns
    -------
    float
    '''
    
    #notify if NaN exists:
    if ret.isnull().values.any() or 0 in ret.values:
        print('There exists NaN or 0 values in rea_vol!')
    
    return np.sum(np.log(ret+1)**2)**0.5

def mdd(ret):
    '''
    max drawdown is defined as the highest peak-to-trough value, divided by the peak (trough must be aft peak)
    Parameters
    ----------
    ret : series or df
        can be indiv asset returns, in which case this will return a series

    Returns
    -------
    float
        max drawdown of period
    '''
    #notify if NaN exists:
    if ret.isnull().values.any() or 0 in ret.values:
        print('There exists NaN or 0 values in mdd!')
    #cum_returns is basically the price history
    cum_returns = (1 +ret).cumprod()
    #cum_returns.cummax gives highest price observed so far at any point
    #drawdown gives many 'windows' of (highest_price_so_far-current price)/highest price
    #this constraints highest price to occur before current price
    #maximizing this window is the same as finding the lowest trough after the peak
    drawdown =  1 - cum_returns.div(cum_returns.cummax())
    return drawdown.max(axis=0) #maximum vertically

def semi_std(ret):
    '''
    find semi-standard deviation of period, a measure of downside risk
        calcs std of only those returns that fall below mean return
    Parameters
    ----------
    ret : dataframe, portfolio returns

    Returns
    -------
    float
    '''
   
    #notify if NaN exists:
    if ret.isnull().values.any() or 0 in ret.values:
        print('There exists NaN or 0 values in semi_std!')
    
    average = np.nanmean(ret.to_numpy())
    r_below = ret[ret < average]
    return np.sqrt(1/len(r_below) * np.sum((average - r_below)**2))

def coph_corr(port):
    '''
    calculates the cophenetic correlation coefficient of this clustering structure
        this is the corr coeff btwn distance matrix and cophenetic matrix
    note that this assumes p_dist exists, which is not true for all cov-estimation methods
    Parameters
    ----------
    port : portfolio object belonging to riskfolio module
        a portfolio object that has already gone through optimization

    Returns
    -------
    float
        first gives a numpy scalar value, then converts it to a float!
    '''

    return hr.cophenet(port.clusters,Y=port.p_dist)[0].item() #check out hr's cophenet method

#%%
'''
FUNCTIONS FOR MANIPULATING PORTFOLIO RETURNS AND WEIGHTS
'''

def inv_vol(ret):
    '''
    assets are weighted accordingly to the inverse volatility allocation
    Parameters
    ----------
    ret : a dataframe
        contain indiv asset returns

    Returns
    -------
    list
    the weights listed in order
    '''
    variance=ret.var(axis=0)
    return (1/variance)/((1/variance).sum())


def port_ret(weights, ret, index_):
    '''
    this function applies a set of weights to daily returns of an asset universes
    and creates a portfolio return. note that it makes use of the theoretical property that
    ret(portfolio)= sum[asset_weight * ret(asset)]
    
    Parameters
    ----------
    weights : dictionary
        keys are the integer index of returns while each item is a set of weights for that day
        e.g.
        1105: [w1,w2,w3]
        1206: [w1_new,w2_new,w3_new]
        This will mean that weight applies on ret from 1105 to 1205, and 1206 is a diff set of weights
    ret :  a dataframe, DAILY RETURNS OF EACH TICKER
    index_ : list
        These are integer indexes of ret, where REBALANCING(CHOOSING ERC/HRP) happens
        note that index_ may be a subset of the index of weights
        because XGboost needs some months of data, can only start ltr

    Returns
    -------
    series
        daily portfolio returns, COMBINED into one value
        note that port_returns will automatically make nan all the returns up to the first date in index_
    '''
    
    ret=ret.copy()
    for i in range(len(index_)):
        if i!=len(index_)-1: #if the index is not the last index
            ret.iloc[index_[i]:index_[i+1]]=weights[index_[i]].to_list()*ret.iloc[index_[i]:index_[i+1]]
        else: #if last number in index_, just apply weights all the way to last row
            ret.iloc[index_[i]:]=weights[index_[i]].to_list()*ret.iloc[index_[i]:]
    #sum all the returns horizontally
    #note that if rows are NaN, summing will make them zero!!
    ret=ret.sum(axis=1)
    #make all rows nan up to first rebalancing date
    ret.iloc[:index_[0]]=np.nan
    return ret #will contain nan, not fully filled

def cum_ret(ret):
    '''
    takes in some daily returns (EITHER indiv assets OR combined portfolio) and gives cumulative returns
 
    if portfolio returns, take in series, gives a float(cum_ret of portfolio)
    
    Parameters(assuming indiv asset returns)
    ----------
    ret : dataframe
        individual asset returns,assumes there are NO NA VALUES (fully filled) 

    Returns
    -------
    series
        indexed by the asset name, gives cum_ret of each asset
        
    If PORTFOLIO RETURNS entered, take in series, gives a float(cum_ret of portfolio)
    '''
    return (1 + ret).cumprod().iloc[-1]-1

def stats_corr(corr):
    '''
    find some stats on the correlation matrix
        1st value is mean of all lower triangular elem, 
        2nd is std of all lower trian elems
        3rd elem is the condition number of this corr matrix, measures how sensitive a matrix is (??)
        4th elem is the determinant of the matrix
    Parameters
    ----------
    corr : dataframe
        the correlation matrix btwn all asset pairs.

    Returns
    -------
    tuple
        contains values in order above
    '''
    
    #note that here values is a np array!!
    values=corr.values[np.triu_indices_from(corr.values,1)]
    cond=np.linalg.cond(corr)
    det=np.linalg.det(corr)
    #ddof=1 for unbiased estimator in numpy
    return (values.mean(),values.std(ddof=1),cond,det)

def stats_asset_univ(uni_ret):
    '''
    calc some stats about the asset universe
    Parameters
    ----------
    uni_ret : dataframe
        THE UNIVERSE' RETURNS', for each asset. Assumes no NA! 

    Returns
    -------
    a tuple of 4 values, as specified in strat
    '''

    m_m=uni_ret.mean(axis=0).mean()
    s_m=uni_ret.std(axis=0).mean()
    m_s=uni_ret.mean(axis=0).std()
    s_s=uni_ret.std(axis=0).std()

    #this usually gives alot of NaN as some prices in uni_ret are the same aft a day
    mdd_m=mdd(uni_ret).mean()
    mdd_s=mdd(uni_ret).std()
    return(m_m,s_m,m_s,s_s,mdd_m,mdd_s)

#%%
'''
MISCELLANOUS FUNCTIONS
'''

def ticker_gen(ticker_pool, uni_len, uni_num, num_share):
    '''
    this helper function generates some universes of tickers from a pool of tickers.
    Each universe can only only share max num_share tickers with any other universes
    Parameters
    ----------
    ticker_pool : list
        a list, the list of possible tickers
    uni_len : float
        the number of tickers in each universe
    uni_num : float
        the number of universes to generate (has an upper bound!)
        may give not possible if too high
    num_share : float
        max number each uni can share w/ each other

    Returns
    -------
    lis : list of lists
        each list consisting of asset tickers
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
    get some insights into this asset universe, through its correlation matrix of all returns
    compiles some stats about the correlation matrix
    Parameters
    ----------
    ret : dataframe
        the returns for the whole universe, indiv assets
        should be ALL the historical returns

    Returns
    -------
    tuple
        1st elem is the correlation matrix, 
        2nd elem is the mean of upper triangle, 
        3rd elem is the std of upper triangle
    '''

    port = rp.Portfolio(returns=ret)
    port.assets_stats(method_cov='gerber1')
    corr=rp.AuxFunctions.cov2corr(port.cov)
    values=corr.values[np.triu_indices_from(corr.values,1)]
    return {'Correlation Matrix':corr,
            'Mean Correlation':values.mean(),
            'Standard Deviation of Correlation':values.std(ddof=1)}