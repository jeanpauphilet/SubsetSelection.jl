# SubsetSelection

[![0.6](http://pkg.julialang.org/badges/SubsetSelection_0.6.svg)](http://pkg.julialang.org/?pkg=SubsetSelection)

SubsetSelection is a Julia package that computes sparse L2-regularized estimators. Sparsity is enforced through explicit cardinality constraint or L0-penalty. Supported loss functions for regression are least squares, L1 and L2 SVR; for classification, logistic, L1 and L2 Hinge loss. The algorithm formulates the problem as a mixed-integer saddle-point problem and solves its boolean relaxation using a dual sub-gradient approach.

## Quick start
To install the package:
```julia
julia> Pkg.install("SubsetSelection")
```
or the have the latest version
```julia
julia> Pkg.clone("git://github.com/jeanpauphilet/SubsetSelection.jl.git")
```

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

The algorithm returns a SparseEstimator object with the following fields: `loss` (loss function used), `sparsity` (model to enforce sparsity), `indices` (features selected), `w` (value of the estimator on the selected features only), `α` (values of the associated dual variables), `b` (bias term), `iter` (number of iterations required by the algorithm).

For instance, you can access the selected features directly in the `indices` field:
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
 ```

 or compute predictions
```julia
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

For classification, we use +1/-1 labels and the convention
`P ( Y = y | X = x ) = 1 / (1+e^{- y x^T w})`.

## Required and optional parameters

`subsetSelection` has four required parameters:
- the loss function to be minimized, to be chosen among least squares (`OLS()`), L1SVR (`L1SVR(ɛ)`), L2SVR (`L2SVR(ɛ)`), Logistic loss (`LogReg()`), Hinge Loss (`L1SVM()`), L2-SVM (`L2SVM()`). For classification, we recommend using Hinge loss or L2-SVM functions. Indeed, the Fenchel conjugate of the Logistic loss exhibits unbounded gradients, which largely hinders convergence of the algorithm and might require smaller and more steps (see optional parameters).
- the model used to enforce sparsity; either by adding a hard constraint of the form "||w||_0 < k" (`Constraint(k)`) or by adding a penalty of the form "+ λ ||w||_0" (`Penalty(λ)`) to the objective. For tractability issues, we highly recommend using an explicit constraint instead of a penalty, for it ensures the size of the support remains bounded through the algorithm.
- the vector of outputs `Y` of size `n`, the sample size. In classification settings, `Y` should be a vector of ±1s.
- the matrix of covariates `X` of size `n`×`p`, where `n` and `p` are the number of samples and features respectively.

In addition, `subsetSelection` accepts the following optional parameters:
- an initialization for the selected features, `indInit`.
- an initialization for the dual variables, `αInit`.
- the value of the ℓ2-regularization parameter `γ`, set to 1/√n by default.
- `intercept`, a boolean. If true, an intercept/bias term is computed as well. By default, set to false.
- the maximum number of iterations in the sub-gradient algorithm, `maxIter`.
- the value of the gradient stepsize `δ`. By default, the stepsize is set to 1e-3, which demonstrates very good empirical performance. However, smaller stepsizes might be needed when dealing with very large datasets or when the Logistic loss is used.
- the number of gradient updates of dual variable α performed per update of primal variable s, `gradUp`.
- `anticycling` a boolean. If true, the algorithm stops as soon as the support is not unchanged from one iteration to another. Empirically, the accuracy of the resulting support is strongly sensitive to noise - to use with caution. By default, set to false.
 - `averaging` a boolean. If true, the dual solution is averaged over past iterates. By default, set to true.

## Best practices
- Tuning the regularization parameter `γ`: By default, `γ` is set to 1/√n, which is an appropriate scaling in most regression instances. For an optimal performance, and especially in classification or noisy settings, we recommend performing a grid search and using cross-validation to assess out-of-sample performance. The grid search should start with a very low value for `γ`, such as  
```julia
    γ = 1.*p / k / n / maximum(sum(X[train,:].^2,2))
```
and iteratively increase it by a factor 2. Mean square error or Area Under the Curve (see [ROCAnalysis]( https://github.com/davidavdav/ROCAnalysis.jl) for implementation) are commonly used performance metrics for regression and classification tasks respectively.
- Instances where the algorithm fails to converge have been reported. If you occur such cases, try normalize the data matrix `X` and relaunch the algorithm. If the algorithm still fails to converge, reduce the stepsize `δ` by a factor 10 or 100 and increase the number of iterations `maxIter` by a factor at least 2.

## Reference
