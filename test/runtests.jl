using SubsetSelection
using StatsBase, Test

n = 500; p = 1000; k = 10;

indices = sort(sample(1:p, StatsBase.Weights(ones(p)/p), k, replace=false));
w = sample(-1:2:1, k);
X = randn(n,p); Y = X[:,indices]*w;
@time Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X)
for i in 1:k
  @test Sparse_Regressor.indices[i]==indices[i]
end
