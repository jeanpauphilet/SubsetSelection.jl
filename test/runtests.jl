using SubsetSelection
using StatsBase
using Base.Test

@testset "Regular examples" begin
  srand(1)

  n = 500; p = 1000; k = 10;

  indices = sort(sample(1:p, k, replace=false));
  w = sample(-1:2:1, k);
  X = randn(n,p); Y = X[:,indices]*w;
  Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X)
  for i in 1:k
    @test Sparse_Regressor.indices[i]==indices[i]
  end

  p = 8
  n = 100
  k = 3
  w = rand(k)
  indices = sort(sample(1:p, k, replace=false))
  X = randn(n,p)
  Y = X[:,indices]*w
  Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X, γ=1.0)
  @test Sparse_Regressor.indices == indices
  @test isapprox(Sparse_Regressor.w, w, atol = 1e-1)
end

@testset "Algorithm diverges" begin
    srand(1)
    p = 8
    n = 100
    k = 3
    w = rand(k)
    indices = sort(sample(1:p, k, replace=false))
    X = randn(n,p)
    Y = X[:,indices]*w
    @test_warn r"" Sparse_Regressor = subsetSelection(OLS(), Constraint(k), Y, X, γ = 1000.0)
end
