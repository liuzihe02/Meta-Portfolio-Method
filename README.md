# Meta Portfolio Method
This project implements the Meta Portfolio Method (MPM), outlined in the paper "A Meta-Method for Portfolio Management Using Machine Learning for Adaptive Strategy Selection". MPM uses machine learning to rebalance a portfolio, switching between 2 portfolio optimization strategies.

## Description

### Original Paper
The original description of MPM is found in this [Arxiv link](https://doi.org/10.48550/arXiv.2111.05935). Features used for the model (XGBoost Regressor) include statistics on the asset universe, characteristics of the correlation matrix, and recent performance measures on both strategies. The label (target variable) is the difference between the Sharpe Ratios of each strategy.

### Implementation Differences
* While the paper combined Naive Risk Parity(NRP) and Hierarchical Risk Parity (HRP) in the MPM, this work uses Equal Risk Contribution (ERC) and HRP. Both NRP and HRP had near-identical weights, and ERC with HRP gave significantly better results. Should you choose NRP instead, simply use the function "inv_vol" to obtain the weights.

* The original paper rebalanced the portfolio monthly, and gathered data points (features-label) monthly. This work rebalances monthly, but gathers data daily to increase the dataset size. This increased the model's confidence in its features.

### Performance Results


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

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Roadmap
* Optimize Code
* Testing for overfitting, on each monthly model
* Trying different portfolio optimization strategies

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. Any contributions are greatly appreciated! Don't forget to give the project a star, thanks!

## Contact

Liu Zihe - [@purplecrane02](https://twitter.com/purplecrane02)

## License

This project is licensed under the [MIT] License - see the LICENSE file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
