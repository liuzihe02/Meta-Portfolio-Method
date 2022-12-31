# Meta Portfolio Method

This project implements the Meta Portfolio Method (MPM), outlined in the paper "A Meta-Method for Portfolio Management Using Machine Learning for Adaptive Strategy Selection". MPM uses machine learning to rebalance a portfolio, switching between 2 portfolio optimization strategies.

## Description

### Original Paper

The original description of MPM is found in this [Arxiv link](https://doi.org/10.48550/arXiv.2111.05935). Features used for the model (XGBoost Regressor) include statistics on the asset universe, characteristics of the correlation matrix, and recent performance measures on both strategies. The label (target variable) is the difference between the Sharpe Ratios of each strategy.

### Implementation Differences

* While the paper combined Naive Risk Parity(NRP) and Hierarchical Risk Parity (HRP) in the MPM, this work uses Equal Risk Contribution (ERC) and HRP. Both NRP and HRP had near-identical weights, and ERC with HRP gave significantly better results. Should you choose NRP instead, simply use `inv_vol` to obtain the weights.

* The original paper rebalanced the portfolio monthly, and gathered data points (features-label) monthly. This work also rebalances monthly, but gathers data daily to increase the dataset size. This increases the model's confidence in its features.

## Performance Results

For detailed testing results, do check out Report.docx

### Sharpe Ratio

The Sharpe Ratio is computed using yearly returns, assuming zero risk-free rate of return.

|             	| ERC  	| HRP  	|  MPM 	|
|:-----------:	|------	|------	|:----:	|
|  Universe 1 	| 0.87 	| 0.87 	| 0.97 	|
|  Universe 2 	| 0.90 	| 0.89 	| 0.94 	|
|  Universe 3 	| 0.81 	| 0.83 	| 1.10 	|
|  Universe 4 	| 1.12 	| 0.87 	| 1.10 	|
|  Universe 5 	| 1.13 	| 1.10 	| 1.15 	|
|  Universe 6 	| 1.03 	| 0.86 	| 1.45 	|
|  Universe 7 	| 1.04 	| 1.01 	| 1.10 	|
|  Universe 8 	| 0.93 	| 0.82 	| 1.07 	|
|  Universe 9 	| 1.21 	| 1.14 	| 1.54 	|
| Universe 10 	| 1.39 	| 1.23 	| 1.55 	|

### Compound Annual Growth Rate

CAGR is shown in percentages.

|             	| ERC  	| HRP  	| MPM 	|
|:-----------:	|------	|------	|:---:	|
|  Universe 1 	| 3.6  	| 2.8  	| 3.8 	|
|  Universe 2 	| 5.4  	| 4.8  	| 5.9 	|
|  Universe 3 	| 0.77 	| 0.46 	| 1.7 	|
|  Universe 4 	| 1.1  	| 0.48 	| 1.4 	|
|  Universe 5 	| 4.0  	| 2.3  	| 4.1 	|
|  Universe 6 	| 1.4  	| 0.49 	| 2.0 	|
|  Universe 7 	| 4.3  	| 3.3  	| 4.4 	|
|  Universe 8 	| 3.4  	| 1.8  	| 3.4 	|
|  Universe 9 	| 4.7  	| 2.5  	| 3.8 	|
| Universe 10 	| 3.8  	| 2.2  	| 4.7 	|

## Getting Started

### Dependencies

The following packages are used:
* pandas
* numpy
* riskfolio
* pickle
* xgboost
* skopt
* yfinance

### Executing program

- Run `main_build_data`
  - Input a list of tickers with `assets`
  	> you can also generate multiple asset universes from a pool with `ticker_gen`
  - Select the relevant timeframe with `start` and `end`

- Run `main_strategies`
  - Choose whether to do Bayesian Hyperparameter Optimization with the Boolean `bayes`
  - Set the timeframe (in days, default 8 years) ML considers with `trng_period`
  	> this is also the size of the ML training dataset

- View Results and Weights
	- `main_strategies` will generate a feature importance plot (boxes cover middle 50%, whiskers cover middle 96%) and the portfolio prices for the 3 strategies
	- Important variables are:
		- `comp` : a dataframe summarizing the results, containing Sharpe Ratios estimated with different frequencies
		- `uni_corr_stats` : a dictionary containing information about the correlations between assets
		- `weights_MPM` : a dictionary containing the weights for every rebalancing

## Roadmap

* Optimize Code
* Testing for overfitting, on each monthly model
* Trying different portfolio optimization strategies

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. Any contributions are greatly appreciated! Don't forget to give the project a star, thanks!

## Contact

Liu Zihe - [@purplecrane02](https://twitter.com/purplecrane02)

## License

This project is licensed under the MIT License - see `LICENSE` for details

## Acknowledgments

Reference Papers
* [Original Paper](https://arxiv.org/abs/2111.05935)
* [Interpretable Machine Learning for Diversified Portfolio Construction](https://jfds.pm-research.com/content/early/2021/06/14/jfds.2021.1.066)
* [Portfolio Optimization: A General Framework For Portfolio Choice](https://investresolve.com/portfolio-optimization-machine/)
