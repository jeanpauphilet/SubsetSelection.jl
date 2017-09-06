module SubsetSelection
using Compat

import Compat.String

export LossFunction, Regression, Classification, OLS, L1SVR, L2SVR, LogReg, L1SVM, L2SVM
export Sparsity, Constraint, Penalty
export SparseEstimator, subsetSelection

##LossFunction type: define the loss function used and its hyper-parameter
@compat abstract type LossFunction end
@compat abstract type Regression <: LossFunction end
@compat abstract type Classification <: LossFunction end
  #Loss functions for regression
  immutable OLS <: Regression
  end
  immutable L1SVR <: Regression
    ɛ::Float64
  end
  immutable L2SVR <: Regression
    ɛ::Float64
  end

  #Loss functions for classification
  immutable LogReg <: Classification
  end
  immutable L1SVM <: Classification
  end
  immutable L2SVM <: Classification
  end

##Sparsity type: specify how sparsity is enforced, constrained or penalized
@compat abstract type Sparsity end
  #Constraint: add the constraint "s.t. ||w||_0<=k"
  immutable Constraint <: Sparsity
    k::Int
  end
  function parameter(Card::Constraint)
    return Card.k
  end
  max_index_size(Card::Constraint, p::Int) = min(Card.k, p)

  #Penalty: add the penalty "+λ ||w||_0"
  immutable Penalty <: Sparsity
    λ::Float64
  end
  function parameter(Card::Penalty)
    return Card.λ
  end
  max_index_size(Card::Penalty, p::Int) = p

##SparseEstimator type: output of the algorithm
type SparseEstimator
  "Loss function used in the model"
  loss::LossFunction
  "Model to account for sparsity"
  sparsity::Sparsity
  "Set of relevant indices"
  indices::Array
  "Estimator w on those selected indices"
  w::Array{Float64}
  "Dual variables α"
  α::Array{Float64}
  "Intercept/bias term"
  b::Float64
  "Number of iterations in the sug-gradient algorithm"
  iter::Integer
end

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

##############################################
##DUAL SUB-GRADIENT ALGORITHM
##############################################
""" Function to compute a sparse regressor/classifier. Solve an optimization problem of the form
        min_s max_α f(\alpha, s)
by gradient ascent on α:        α_{t+1} ← α_t + δ ∇_α f(α_t, s_t)
and partial minimization on s:  s_{t+1} ← argmin_s f(α_t, s)

INPUTS
- ℓ           Loss function used
- Card        Model to enforce sparsity (constraint or penalty)
- Y (n×1)     Vector of outputs. For classification, use ±1 labels
- X (n×p)     Array of inputs.
- indInit     Initial subset of features s
- αInit       Initial dual variable α
- γ           ℓ2 regularization penalty
- intercept   Boolean. If true, an intercept term is computed as well
- maxIter     Total number of Iterations
- δ           Gradient stepsize
- gradUp      Number of gradient updates of dual variable α performed per update of primal variable s
- anticycling Boolean. If true, the algorithm stops as soon as the support is not unchanged from one iteration to another
- averaging   Boolean. If true, the dual solution is averaged over past iterates

OUTPUT
- SparseEstimator """
function subsetSelection(ℓ::LossFunction, Card::Sparsity, Y, X;
    indInit = ind_init(Card, size(X,2)), αInit=alpha_init(ℓ, Y),
    γ = 1/sqrt(size(X,1)),  intercept = false,
    maxIter = 100, δ = 1e-3, gradUp = 10,
    anticycling = false, averaging = true)

  n = size(Y, 1)
  p = size(X, 2)
  cache = Cache(n, p)

  indices = indInit #Support
  n_indices = length(indices)

  n_indices_max = max_index_size(Card, p)
  resize!(indices, n_indices_max)
  indices_old = Vector{Int}(n_indices_max)

  α = αInit[:]  #Dual variable α
  a = αInit[:]  #Past average of α


  ##Dual Sub-gradient Algorithm
  iter = 2
  while iter < maxIter
    # println("Iterations: ", iter)

    #Gradient ascent on α
    for inner_iter in 1:min(gradUp, div(p, n_indices))
      α .+= δ .* grad_dual(ℓ, Y, X, α, indices, n_indices, γ, cache)
      α = proj_dual(ℓ, Y, α)
      α = proj_intercept(intercept, α)
    end

    #Update average a
    @__dot__ a = (iter - 1) / iter * a + 1 / iter * α

    #Minimization w.r.t. s
    indices_old[1:n_indices] = indices[1:n_indices]
    n_indices = partial_min!(indices, Card, X, α, γ, cache)

    #Anticycling rule: Stop if indices_old == indices
    if anticycling && indices_same(indices, indices_old, n_indices)
      averaging = false #If the algorithm stops because of cycling, averaging is not needed
      break
    else
      iter += 1
    end
  end

  ##Compute sparse estimator
  #Subset of relevant features
  n_indices = partial_min!(indices, Card, X, averaging ? a : α, γ, cache)
  #Regressor
  w = [-γ * dot(X[:, indices[j]], a) for j in 1:n_indices]
  #Bias
  b = compute_bias(ℓ, Y, X, a, indices, n_indices, γ, intercept, cache)

  #Resize final indices vector to only have relevant entries
  resize!(indices, n_indices)

  return SparseEstimator(ℓ, Card, indices, w, a, b, iter)
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
  return max(-1,min(1, -Y))
end
function alpha_init(ℓ::L2SVR, Y)
  return -Y
end
function alpha_init(ℓ::Classification, Y)
  return -Y./2
end

##Point-wise derivative of the Fenchel conjugate for each loss function
function grad_fenchel(ℓ::OLS, y, a)
  return a + y
end
function grad_fenchel(ℓ::L1SVR, y, a)
  return y + ℓ.ɛ*sign(a)
end
function grad_fenchel(ℓ::L2SVR, y, a)
  return a + y + ℓ.ɛ*sign(a)
end
function grad_fenchel(ℓ::LogReg, y, a)
  return y*log(1+a*y) - y*log(-a*y)
end
function grad_fenchel(ℓ::L1SVM, y, a)
  return y
end
function grad_fenchel(ℓ::L2SVM, y, a)
  return a + y
end

##Gradient of f w.r.t. the dual variable α
function grad_dual(ℓ::LossFunction, Y, X, α, indices, n_indices, γ, cache::Cache)
  g = cache.g
  for i in 1:size(X, 1)
    g[i] = -grad_fenchel(ℓ, Y[i], α[i])
  end
  for j in 1:n_indices
    x = @view(X[:, indices[j]])
    @__dot__ g -= γ * dot(x, α) * x
  end
  g
end

##Projection of α on the feasible set of the Fenchel conjugate
function proj_dual(ℓ::OLS, Y, α)
  return α
end
function proj_dual(ℓ::L1SVR, Y, α)
  return max(-1,min(1, α))
end
function proj_dual(ℓ::L2SVR, Y, α)
  return α
end
function proj_dual(ℓ::LogReg, Y, α)
  return Y.*max(-1,min(0, Y.*α))
end
function proj_dual(ℓ::L1SVM, Y, α)
  return Y.*max(-1,min(0, Y.*α))
end
function proj_dual(ℓ::L2SVM, Y, α)
  return Y.*min(0, Y.*α)
end

##Projection of α on e^T α = 0 (if intercept)
function proj_intercept(intercept::Bool, α)
  if intercept
    α .-= mean(α)
    return α
  else
    return α
  end
end

##Minimization w.r.t. s
function partial_min!(indices, Card::Constraint, X, α, γ, cache::Cache)
  ax = cache.ax
  sortperm = cache.sortperm

  p = size(X,2)
  n_indices = max_index_size(Card, p)

  # compute α'*X into pre-allocated scratch space
  Ac_mul_B!(ax, X, α)
  # take the k largest (absolute) values of ax
  map!(abs, ax, ax)
  sortperm!(sortperm, ax, rev=true)
  indices[:] = sortperm[1:n_indices]
  sort!(@view(indices[1:n_indices]))

  # Return the updated size of indices
  return n_indices
end
function partial_min(Card::Penalty, X, α, γ)
  p = size(X,2)
  return find(x-> x<0, Card.λ .- γ/2*[dot(α, X[:,j])^2 for j in 1:p])
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

end #module
