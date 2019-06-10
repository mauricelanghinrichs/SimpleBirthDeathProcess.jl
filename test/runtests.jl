cd(dirname(@__FILE__))

using LinearAlgebra
using Random
using Statistics
using Test

include("../src/SimpleBirthDeathProcess.jl")
using .SimpleBirthDeathProcess

Random.seed!(43314697)

function runtests()
  tests = [
    "common/functions",
    "common/hypergeometric",
    "common/meixner_ortho_poly",
    "simple/transition_probabilities",
    "simple/simulation",
    "simple/gradient",
    "simple/hessian",
    "simple/derivatives"
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
