# SubsetSelection

[![Build Status](https://travis-ci.org/jeanpauphilet/SubsetSelection.jl.svg?branch=master)](https://travis-ci.org/jeanpauphilet/SubsetSelection.jl)
[![Coverage Status](https://coveralls.io/repos/jeanpauphilet/SubsetSelection.jl/badge.svg?branch=master)](https://coveralls.io/r/jeanpauphilet/SubsetSelection.jl?branch=master)

SubsetSelection is a Julia package that computes sparse L2-regularized estimators. Sparsity is enforced through explicit cardinality constraint or L0-penalty (BIC estimators). Supported loss functions for regression are least squares, L1 and L2 SVR; for classification, logistic, L1 and L2 Hinge loss. The algorithm formulates the problem as a mixed-integer saddle-point problem and solves its boolean relaxation using a dual sub-gradient approach.

## Quick start

To fit a basic model:

```julia
julia> using SubsetSelection, StatsBase

julia> n = 100; p = 10000; k = 10;
julia> indices = sort(sample(1:p, StatsBase.Weights(ones(p)/p), k, replace=false));
julia> w = sample(-1:2:1, k);
julia> X = randn(n,p); Y = X[:,indices]*w;
julia> Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X)
SubsetSelection.SparseEstimator(SubsetSelection.OLS(),SubsetSelection.Constraint(10),10.0,[362,1548,2361,3263,3369,3598,5221,7314,7748,9267],[5.37997,-5.51019,-5.77256,-7.27197,-6.32432,-4.97585,5.94814,4.75648,5.48098,-5.91967],[-0.224588,-1.1446,2.81566,0.582427,-0.923311,4.1153,-2.43833,0.117831,0.0982258,-1.60631  …  0.783925,-1.1055,0.841752,-1.09645,-0.397962,3.48083,-1.33903,1.44676,4.03583,1.05817],0.0,19)
```

The algorithm returns a SparseEstimator object with the following fields: loss (loss function used), sparsity (model to enforce sparsity), indices (features selected), w (value of the estimator on the selected features only), α (values of the associated dual variables), b (bias term), iter (number of iterations required by the algorithm).

```julia
julia> Sparse_Regressor.indices
10-element Array{Int64,1}:
  362
 1548
 2361
 3263
 3369
 3598
 5221
 7314
 7748
 9267
 julia> Y_pred = X[:,Sparse_Regressor.indices]*Sparse_Regressor.w
100-element Array{Float64,1}:
   4.62918
   8.59952
 -16.2796
  -5.611
   1.62764
 -50.4562
  37.407
 -12.3341
  -4.75339
  25.122
   ⋮
  -7.98349
  11.0327
  -8.58172
  16.904
  -9.04211
 -36.5475
  17.2558
 -22.3915
 -57.9727
  -6.06553
 ```

For classification, use +1/-1 labels.

## Required and optional parameters

<!-- `glmnet` has two required parameters: the m x n predictor matrix `X` and the dependent variable `y`. It additionally accepts an optional third argument, `family`, which can be used to specify a generalized linear model. Currently, only `Normal()` (least squares, default), `Binomial()` (logistic), and `Poisson()` are supported, although the glmnet Fortran code also implements a Cox model. For logistic models, `y` is a m x 2 matrix, where the first column is the count of negative responses for each row in `X` and the second column is the count of positive responses. For all other models, `y` is a vector.

`glmnet` also accepts many optional parameters, described below:

 - `weights`: A vector of weights for each sample of the same size as `y`.
 - `alpha`: The tradeoff between lasso and ridge regression. This defaults to `1.0`, which specifies a lasso model.
 - `penalty_factor`: A vector of length n of penalties for each predictor in `X`. This defaults to all ones, which weights each predictor equally. To specify that a predictor should be unpenalized, set the corresponding entry to zero.
 - `constraints`: An n x 2 matrix specifying lower bounds (first column) and upper bounds (second column) on each predictor. By default, this is `[-Inf Inf]` for each predictor in `X`.
 - `dfmax`: The maximum number of predictors in the largest model.
 - `pmax`: The maximum number of predictors in any model.
 - `nlambda`: The number of values of λ along the path to consider.
 - `lambda_min_ratio`: The smallest λ value to consider, as a ratio of the value of λ that gives the null model (i.e., the model with only an intercept). If the number of observations exceeds the number of variables, this defaults to `0.0001`, otherwise `0.01`.
 - `lambda`: The λ values to consider. By default, this is determined from `nlambda` and `lambda_min_ratio`.
 - `tol`: Convergence criterion. Defaults to `1e-7`.
 - `standardize`: Whether to standardize predictors so that they are in the same units. Defaults to `true`. Beta values are always presented on the original scale.
 - `intercept`: Whether to fit an intercept term. The intercept is always unpenalized. Defaults to `true`.
 - `maxit`: The maximum number of iterations of the cyclic coordinate descent algorithm. If convergence is not achieved, a warning is returned. -->


## Reference

 <!-- - [Lasso.jl](https://github.com/simonster/Lasso.jl), a pure Julia implementation of the glmnet coordinate descent algorithm that often achieves better performance.
 - [LARS.jl](https://github.com/simonster/LARS.jl), an implementation
   of least angle regression for fitting entire linear (but not
   generalized linear) Lasso and Elastic Net coordinate paths. -->
