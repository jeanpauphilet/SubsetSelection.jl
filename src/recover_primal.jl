# FUNCTION recover_primal
"""Computes the Ridge regressor
INPUT
  ℓ           - LossFunction to use
  Y           - Vector of observed responses
  Z           - Matrix of observed features
  γ           - Regularization parameter
OUTPUT
  w           - Optimal regressor"""
function recover_primal(ℓ::Regression, Y, Z, γ)
  CM = eye(size(Z,2))/γ + Z'*Z      # The capacitance matrix
  α = -Y + Z*(CM\(Z'*Y))            # Matrix Inversion Lemma
  return -γ*Z'*α                    # Regressor
end

using LIBLINEAR

function recover_primal(ℓ::Classification, Y, Z, γ)
  solverNumber = LibLinearSolver(ℓ)
  if isa(ℓ, SubsetSelection.Classification)
    model = LIBLINEAR.linear_train(Y, Z'; verbose=false, C=γ, solver_type=Cint(solverNumber))
    return Y[1]*model.w
  # else
  #   model = linear_train(Y, Z'; verbose=false, C=γ, solver_type=Cint(solverNumber), eps = ℓ.ε)
  #   return model.w
  end
end


function LibLinearSolver(ℓ::L1SVR)
  return 13
end
function LibLinearSolver(ℓ::L2SVR)
  return 12
end

function LibLinearSolver(ℓ::LogReg)
  return 7
end
function LibLinearSolver(ℓ::L1SVM)
  return 3
end
function LibLinearSolver(ℓ::L2SVM)
  return 2
end
