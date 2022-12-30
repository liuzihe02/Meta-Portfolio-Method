# Project Title
This project aims to implement the Meta Portfolio Method (MPM), outlined in the paper "A Meta-Method for Portfolio Management Using Machine Learning for Adaptive Strategy Selection". The method uses machine learning to rebalance a portfolio, switching between 2 portfolio optimization strategies.

## Description

### Original Paper
The original description of MPM is found in [Arxiv DOI link](https://doi.org/10.48550/arXiv.2111.05935). Features used for the ML model (XGBoost Regressor) include statistics about the entire asset universe, characteristics of the correlation matrix between all assets, and recent performance measures of both strategies. The label or target variable is difference between the future Sharpe Ratio of ERC and future Sharpe Ratio of HRP. This provides information on the relative performance of both models, choosing the strategy accordingly.

### Implementation Differences
* While the paper combined Naive Risk Parity(NRP) and Hierarchical Risk Parity (HRP) in the MPM model, this work uses Equal Risk Contribution (ERC) and HRP. Both NRP and HRP had near-identical weights, and ERC with HRP gave significantly better results than NRP with HRP. Should you choose NRP instead, simply use the function "inv_vol" to obtain the desired weights for NRP.
* The original paper rebalanced the portfolio monthly, and gathered data pairs (features-label) monthly

### Performance Results


## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
