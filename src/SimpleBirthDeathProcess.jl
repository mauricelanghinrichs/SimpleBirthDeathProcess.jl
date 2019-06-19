module SimpleBirthDeathProcess
  using LinearAlgebra
  using Logging
  using SpecialFunctions
  using Statistics

  export
  loglik,
  mle,
  observation_continuous_time,
  observation_discrete_time_equal,
  observation_discrete_time_unequal,
  rand_continuous,
  rand_discrete,
  trans_prob

  include("common/common.jl")
  include("simple/simple.jl")
end
