@compat abstract type StepSizeRule end
mutable struct constantStep <: StepSizeRule
    α::Float64
end
mutable struct constantLength <: StepSizeRule
    α::Float64
end
mutable struct poliakRule <: StepSizeRule
    η::Float64
end
mutable struct diminishingAffine <: StepSizeRule
    a::Float64
    b::Float64
end
mutable struct diminishingSqrt <: StepSizeRule
    a::Float64
end

function computeStepSize(rule::constantStep, epoch, ∇q, UB, LB)
  return rule.α
end
function computeStepSize(rule::constantLength, epoch, ∇q, UB, LB)
  return rule.α / norm(∇q)
end
function computeStepSize(rule::poliakRule, epoch, ∇q, UB, LB)
  return rule.η*(UB-LB)/ dot(∇q, ∇q)
end
function computeStepSize(rule::diminishingAffine, epoch, ∇q, UB, LB)
  return rule.a / (rule.b + epoch)
end
function computeStepSize(rule::diminishingSqrt, epoch, ∇q, UB, LB)
  return rule.a / sqrt(epoch)
end

function slashStep!(rule::constantStep, factor)
  rule.α /= factor
end
function slashStep!(rule::constantLength, factor)
  rule.α /= factor
end
function slashStep!(rule::poliakRule, factor)
  rule.η /= factor
end
function slashStep!(rule::diminishingAffine, factor)
  rule.a /= factor
end
function slashStep!(rule::diminishingSqrt, factor)
  rule.a /= factor
end
