using LinearAlgebra
using Random
using Statistics
using Test

using SimpleBirthDeathProcess

Random.seed!(43314697)

function runtests()
  tests = [
    "common/functions",
    "common/hypergeometric",
    "common/meixner_ortho_poly",
    "common/types",
    "simple/transition_probabilities",
    "simple/simulation",
    "simple/derivatives/gradient_v1",
    "simple/derivatives/gradient_v2",
    "simple/derivatives/hessian",
    "simple/derivatives/univariate",
    "simple/log_likelihood"
  ]

  for t in tests
    f = string(t, ".jl")
    println("Going through tests in '", f, "'...")
    include(f)
    println("PASSED!")
  end

  nothing
end

runtests()
