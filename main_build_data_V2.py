import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import warnings
import random
import riskfolio as rp
import matplotlib.pyplot as plt
import pickle as pkl
from main_utility_functions_V2 import ticker_gen

warnings.filterwarnings("ignore")

#%%
# '''PART 0: CREATING LIST OF TICKER UNIVERSES
# In the UCL paper, a list of 10 universes was created from a pool of 18 assets,
# each universe sharing at most 2 assets. Under these constraints, such a list was created.

# Aggregate Bonds: AGG 
# Emerging Markets: VWO, EEM
# Developed Markets: VEA, EFA
# Japanese Equities: EWJ
# Gold: GLD, IAU
# NASDAQ-100 index: QQQ
# 1-3year T-bonds: SHY
# S&P500: SPY
# Index-Linked Bonds: TIP
# Real Estate: VNQ
# Healthcare: XLV
# Value Equities: IWD
# Small cap Equities: IWM
# Corporate Bonds: LQD
# 20year+ T-bonds: TLT
# Mid-cap Equities: VO
# Small-cap value: VBR
# Consumer services: XLY'''

# pool=['AGG','EEM','EFA','EWJ','GLD','QQQ','SHY','SPY','TIP','VNQ','XLV','IWD','IWM','LQD','TLT','VO','VBR','XLY']
# random.seed(0)

# #univs is a list of lists. 
# #Each list inside is a ticker list. 10 such lists(universes) in total, each of length 6, sharing at most 2 assets
# univs=ticker_gen(pool,6,10,2)

# with open("univs.pkl", "wb") as fp:
#     pkl.dump(univs, fp)

#%%
'''PART 1:DOWNLOADING DATA'''

with open('univs.pkl','rb') as f:  #rb is read binary code
     univs= pkl.load(f)

# Date range
start = '2003-11-01'
end = '2022-11-01'

# Tickers of assets, note that univs contains 10 universes!
assets = univs[0] #change this number to switch universes
#THIS STEP IS SUPER IMPT!!
#yfinance will download tickers in sorted order,regardless of given order
#this step makes the columns names sorted also, if not the labelling will fk up
assets.sort()
# Downloading data
prices = yf.download(assets, start=start, end=end)
prices = prices.dropna()

assets_prices=prices.loc[:, ('Adj Close', slice(None))]

#%%
'''PART 2: GETTING RETURNS AND INDEXES'''

# Calculate assets returns
assets_prices.columns = assets
returns = assets_prices.pct_change()#return on day 2 is day1-day2 comparison

# Selecting Dates for Rebalancing

#index contains datetime indexes of first day of each month
index = returns.groupby([returns.index.year, returns.index.month]).head(1).index

#index_ is the integer version of index, in addition to:
#only after the first year will weights be calculated
cov_period=252
index_ = [returns.index.get_loc(x) for x in index if returns.index.get_loc(x) >cov_period]

#save impt variables as a list, written in binary code
with open('data.pkl','wb') as f:
    pkl.dump([cov_period, index_, returns],f)
