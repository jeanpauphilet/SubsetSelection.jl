##LossFunction type: define the loss function used and its hyper-parameter
@compat abstract type LossFunction end
@compat abstract type Regression <: LossFunction end
@compat abstract type Classification <: LossFunction end
  #Loss functions for regression
  struct OLS <: Regression
  end
  struct L1SVR <: Regression
    ɛ::Float64
  end
  struct L2SVR <: Regression
    ɛ::Float64
  end

  #Loss functions for classification
  struct LogReg <: Classification
  end
  struct L1SVM <: Classification
  end
  struct L2SVM <: Classification
  end

  ##Point-wise value of the Fenchel conjugate for each loss function
  function loss(ℓ::OLS, y, u)
    return .5*(y.-u).^2
  end
  function loss(ℓ::L1SVR, y, u)
    return max.(abs.(y-u) .- ℓ.ɛ, 0.)
  end
  function loss(ℓ::L2SVR, y, u)
    return .5*max.(abs.(y-u) .- ℓ.ɛ, 0.)^2
  end
  function loss(ℓ::LogReg, y, u)
    return log.(1 .+ exp.(-y.*u))
  end
  function loss(ℓ::L1SVM, y, u)
    return max.(0., 1 .- y.*u)
  end
  function loss(ℓ::L2SVM, y, u)
    return .5*max.(0., 1 .- y.*u).^2
  end

  ##Point-wise value of the Fenchel conjugate for each loss function
  function fenchel(ℓ::OLS, y, a)
    return .5*a^2 + a*y
  end
  function fenchel(ℓ::L1SVR, y, a)
    return y*a + ℓ.ɛ*abs(a)
  end
  function fenchel(ℓ::L2SVR, y, a)
    return .5*a^2 + y*a + ℓ.ɛ*abs(a)
  end
  function fenchel(ℓ::LogReg, y, a)
    return (1+a*y)*log(1+a*y) - a*y*log(-a*y)
  end
  function fenchel(ℓ::L1SVM, y, a)
    return a*y
  end
  function fenchel(ℓ::L2SVM, y, a)
    return .5*a^2 + a*y
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

##Sparsity type: specify how sparsity is enforced, constrained or penalized
@compat abstract type Sparsity end
    #Constraint: add the constraint "s.t. ||w||_0<=k"
    struct Constraint <: Sparsity
      k::Int
    end
    function parameter(Card::Constraint)
      return Card.k
    end
    max_index_size(Card::Constraint, p::Int) = min.(Card.k, p)

    #Penalty: add the penalty "+λ ||w||_0"
    struct Penalty <: Sparsity
      λ::Float64
    end
    function parameter(Card::Penalty)
      return Card.λ
    end
    max_index_size(Card::Penalty, p::Int) = p

##SparseEstimator type: output of the algorithm
struct SparseEstimator
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
