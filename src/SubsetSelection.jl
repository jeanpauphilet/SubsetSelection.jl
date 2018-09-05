module SubsetSelection
using Compat#, ProgressMeter

import Compat.String

export LossFunction, Regression, Classification, OLS, L1SVR, L2SVR, LogReg, L1SVM, L2SVM
export Sparsity, Constraint, Penalty
export SparseEstimator, subsetSelection

include("types.jl")
include("recover_primal.jl")

# Type to hold preallocated memory
immutable Cache
  g::Vector{Float64}
  ax::Vector{Float64}
  sortperm::Vector{Int}

  function Cache(n::Int, p::Int)
    new(
      Vector{Float64}(n),
      Vector{Float64}(p),
      Vector{Int}(p),
    )
  end
end

mutable struct bestSupport
  indices::Vector{Int}
  nindices::Int
  w::Vector{Float64}
  ub::Float64
  lb::Float64
end
##############################################
##DUAL SUB-GRADIENT ALGORITHM
##############################################
""" Function to compute a sparse regressor/classifier. Solve an optimization problem of the form
        min_s max_α f(α, s)
by gradient ascent on α:        α_{t+1} ← α_t + δ ∇_α f(α_t, s_t)
and partial minimization on s:  s_{t+1} ← argmin_s f(α_t, s)

INPUTS
- ℓ           Loss function used
- Card        Model to enforce sparsity (constraint or penalty)
- Y (n×1)     Vector of outputs. For classification, use ±1 labels
- X (n×p)     Array of inputs.
- indInit     (optional) Initial subset of features s
- αInit       (optional) Initial dual variable α
- γ           (optional) ℓ2 regularization penalty
- intercept   (optional) Boolean. If true, an intercept term is computed as well
- maxIter     (optional) Total number of Iterations
- δ           (optional) Gradient stepsize
- gradUp      (optional) Number of gradient updates of dual variable α performed per update of primal variable s
- anticycling (optional) Boolean. If true, the algorithm stops as soon as the support is not unchanged from one iteration to another
- averaging   (optional) Boolean. If true, the dual solution is averaged over past iterates

OUTPUT
- SparseEstimator """
function subsetSelection(ℓ::LossFunction, Card::Sparsity, Y, X;
    indInit = ind_init(Card, size(X,2)), αInit=alpha_init(ℓ, Y),
    γ = 1/sqrt(size(X,1)),  intercept = false,
    maxIter = 200, noImprov_threshold = maxIter, dGap = 1e-4,
    η = 1)

  n,p = size(X)
  cache = Cache(n, p)

  #Add sanity checks
  if size(Y,1) != n
    throw(DimensionMismatch("X and Y must have the same number of rows"))
  end
  if isa(ℓ, SubsetSelection.Classification)
    levels = sort(unique(Y))
    if length(levels) != 2
      throw(ArgumentError("subsetSelection only supports two-class classification"))
    elseif (levels[1] != -1) || (levels[2] != 1)
      throw(ArgumentError("Class labels must be ±1's"))
    end
  end


  indices = indInit #Support
  n_indices = length(indices)

  n_indices_max = max_index_size(Card, p)
  resize!(indices, n_indices_max)

  # indices_old = Vector{Int}(n_indices_max)
  α = αInit[:]  #Dual variable α
  a = αInit[:]  #Past average of α

  w = recover_primal(ℓ, Y, X[:,indices[1:n_indices]], γ)
  resize!(w, n_indices_max)

  best_s = bestSupport(indices[:], n_indices, w, value_primal(ℓ, Y, X[:,indices[1:n_indices]], w[1:n_indices], γ, cache), -Inf)
  consecutive_noimprov = 0

  ##Dual Sub-gradient Algorithm
  # @showprogress 2 "Feature selection in progress... "
  for iter in 2:maxIter

    #Minimization w.r.t. s
    n_indices = partial_min!(indices, Card, X, α, γ, cache)

    w[:] = recover_primal(ℓ, Y, X[:,indices[1:n_indices]], γ)
    upper_bound = value_primal(ℓ, Y, X[:,indices[1:n_indices]], w, γ, cache)

    if upper_bound < best_s.ub
        consecutive_noimprov = 0
        best_s.indices[:] = indices[:]
        best_s.nindices = n_indices
        best_s.w[1:n_indices] = w[:]
        best_s.ub = upper_bound
    else
        consecutive_noimprov += 1
    end

    best_s.lb = max(best_s.lb, value_dual(ℓ, Y, X, α, indices, n_indices, γ))
    #Gradient ascent on α
    ∇ = grad_dual(ℓ, Y, X, α, indices, n_indices, γ, cache)

    #Stop if small gradient
    if norm(∇, 1) <= 1e-14
      maxIter = iter
      break
    end

    #For numeric stability
    if norm(∇, 1) == Inf
      pInfIndex = find(∇ .== Inf)
      nInfIndex = find(∇ .== -Inf)

      α[pInfIndex] = -Y[pInfIndex]*1e-14
      α[nInfIndex] = -Y[nInfIndex]*(1-1e-14)

      ∇[pInfIndex] = 0.
      ∇[nInfIndex] = 0.
    end

    δ =  η*(best_s.ub - best_s.lb) / dot(∇,∇) #Poliak's rule
    α .+= δ*∇
    # α = proj_dual(ℓ, Y, α)
    # α = proj_intercept(intercept, α)
    proj_dual!(ℓ, Y, α)
    proj_intercept!(intercept, α)

    if iter % 50 == 0
      η /= 2
    end

    #Update average a
    @__dot__ a = (iter - 1) / iter * a + 1 / iter * α

    #Duality gap rule
    if (best_s.ub - best_s.lb) / abs(best_s.lb) < dGap
      maxIter = iter
      break
    end
    #Duality gap rule
    if consecutive_noimprov >= noImprov_threshold
      maxIter = iter
      break
    end
  end

  ##Compute sparse estimator
  #Subset of relevant features
  n_indices = partial_min!(indices, Card, X, a, γ, cache)
  #Regressor
  w = recover_primal(ℓ, Y, X[:,indices[1:n_indices]], γ)

  value = value_primal(ℓ, Y, X[:,indices[1:n_indices]], w, γ, cache)
  if value > best_s.ub #If last solution worse than the best found
      n_indices = best_s.nindices
      indices[:] = best_s.indices[:]
      w = best_s.w[1:n_indices]
  end

  #Bias
  b = compute_bias(ℓ, Y, X, a, indices, n_indices, γ, intercept, cache)

  #Resize final indices vector to only have relevant entries
  resize!(indices, n_indices)

  return SparseEstimator(ℓ, Card, indices, w, a, b, maxIter)
end

# Helper to check if indices and indices_old are the same
# Equivalent to `indices[1:n_indices] == indices_old[1:n_indices]` but without
# allocating temp arrays
function indices_same(indices, indices_old, n_indices)
  for j = 1:n_indices
    if indices[j] != indices_old[j]
      return false
    end
  end
  true
end


##############################################
##AUXILIARY FUNCTIONS
##############################################

##Default Initialization of prinal variables s depending on the model used
function ind_init(Card::Constraint, p::Integer)
  indices0 = find(x-> x<Card.k/p, rand(p))
  while !(Card.k >= length(indices0) >= 1)
    indices0 = find(x-> x<Card.k/p, rand(p))
  end
  return indices0
end
function ind_init(Card::Penalty, p::Integer)
  indices0 = find(x-> x<1/2, rand(p))
  while !(length(indices0)>=1)
    indices0 = find(x-> x<1/2, rand(p))
  end
  return indices0
end

##Default Initialization of dual variables α depending on the model used
function alpha_init(ℓ::OLS, Y)
  return -Y
end
function alpha_init(ℓ::L1SVR, Y)
  return max.(-1,min.(1, -Y))
end
function alpha_init(ℓ::L2SVR, Y)
  return -Y
end
function alpha_init(ℓ::Classification, Y)
  return -Y./2
end

##Dual objective function value for a given dual variable α
function value_dual(ℓ::LossFunction, Y, X, α, indices, n_indices, γ)
  v = - sum([fenchel(ℓ, Y[i], α[i]) for i in 1:size(X, 1)])
  for j in 1:n_indices
    v -= γ/2*(dot(X[:, indices[j]], α)^2)
  end
  return v
end

##Gradient of f w.r.t. the dual variable α
function grad_dual(ℓ::LossFunction, Y, X, α, indices, n_indices, γ, cache::Cache)
  g = cache.g
  for i in 1:size(X, 1)
    g[i] = -grad_fenchel(ℓ, Y[i], α[i])
  end
  for j in 1:n_indices
    x = @view(X[:, indices[j]])
    # @__dot__ g -= γ * dot(x, α) * x
    g -= γ*dot(x, α)*x
  end
  g
end

##Projection of α on the feasible set of the Fenchel conjugate
function proj_dual!(ℓ::OLS, Y, α)
end
function proj_dual!(ℓ::L1SVR, Y, α)
  α[:] = max.(-1,min.(1, α))
end
function proj_dual!(ℓ::L2SVR, Y, α)
end
function proj_dual!(ℓ::LogReg, Y, α)
  α[:] = Y.*max.(-1,min.(0, Y.*α))
end
function proj_dual!(ℓ::L1SVM, Y, α)
  α[:] = Y.*max.(-1,min.(0, Y.*α))
end
function proj_dual!(ℓ::L2SVM, Y, α)
  α[:] = Y.*min.(0, Y.*α)
end

##Projection of α on e^T α = 0 (if intercept)
function proj_intercept!(intercept::Bool, α)
  if intercept
    α[:] .-= mean(α)
  end
  # return α
end

##Minimization w.r.t. s
function partial_min!(indices, Card::Constraint, X, α, γ, cache::Cache)
  ax = cache.ax
  perm = cache.sortperm
  p = size(X,2)
  n_indices = max_index_size(Card, p)

  # compute α'*X into pre-allocated scratch space
  Ac_mul_B!(ax, X, α)
  # take the k largest (absolute) values of ax
  map!(abs, ax, ax)

  sortperm!(perm, ax, rev=true)
  indices[1:n_indices] = perm[1:n_indices]
  sort!(@view(indices[1:n_indices]))

  # Return the updated size of indices
  return n_indices
end

function partial_min!(indices, Card::Penalty, X, α, γ, cache::Cache)
  ax = cache.ax

  # compute (α'*X).^2 into pre-allocated scratch space
  Ac_mul_B!(ax, X, α)
  map!(abs2, ax, ax)

  # find indices with `λ - γ / 2 * (a'X_j)^2 < 0`
  n_indices = 0
  for j = 1:size(X,2)
    if Card.λ - γ / 2 * ax[j] < 0
      n_indices += 1
      indices[n_indices] = j
    end
  end

  return n_indices
end

##Bias term
function compute_bias(ℓ::LossFunction, Y, X, α, indices, n_indices, γ,
                      intercept::Bool, cache::Cache)
  if intercept
    g = grad_dual(ℓ, Y, X, α, indices, n_indices, γ, cache)
    return (minimum(g[α != 0.]) + maximum(g[α != 0.]))/2
  else
    return 0.
  end
end

##Primal objective function value for a given primal variable w
function value_primal(ℓ::LossFunction, Y, X, w, γ, cache::Cache)
  # g = cache.g
  # # for i in 1:size(X, 1)
  # #   g[i] = loss(ℓ, Y[i], dot(X[i,:], w))
  # # end
  # g[:] = loss(ℓ, Y, X*w)
  return sum(loss(ℓ, Y, X*w)) + dot(w,w)/2/γ
  # v = sum([loss(ℓ, Y[i], dot(X[i,:], w)) for i in 1:size(X, 1)])
  # v += dot(w,w)/2/γ
  # return v
end

end #module
