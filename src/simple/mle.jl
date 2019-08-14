"""
    mle(x)

Given an observed sample `x`, return the maximum likelihood estimate (MLE) of a
simple birth and death process.
"""
function mle(
  x::ObservationContinuousTime{F}
)::Vector{Float64} where {
  F <: AbstractFloat
}
  [x.tot_births / x.integrated_jump; x.tot_deaths / x.integrated_jump]
end

function mle(
  x::Vector{ObservationContinuousTime{F}}
)::Vector{Float64} where {
  F <: AbstractFloat
}
  B = sum(y -> y.tot_births, x)
  D = sum(y -> y.tot_deaths, x)
  T = sum(y -> y.integrated_jump, x)

  [B / T; D / T]
end

function mle(
  x::ObservationDiscreteTimeEqual{F, I};
  init::F=zero(F)
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  # this is the maximum likelihood estimator of θ = (λ - μ)
  α = sum(x.state[2:end, :]) / sum(x.state[1:(end - 1), :])
  θ::F = log(α) / x.u

  # check if the log-likelihood function is monotonically decreasing
  if θ < 0
    if isinf(θ)
      # population went immediately extinct and MLE does not exist
      return [zero(F); F(Inf)]
    end

    if loglik([zero(F); -θ], x) > loglik([F(1.0e-4); F(1.0e-4 - θ)], x)
      return [zero(F); -θ]
    end
  elseif θ > 0
    if loglik([θ; zero(F)], x) > loglik([F(θ + 1.0e-4); F(1.0e-4)], x)
      return [θ; zero(F)]
    end
  else
    if loglik(zeros(F, 2), x) > loglik([F(1.0e-4); F(1.0e-4)], x)
      return zeros(F, 2)
    end
  end

  μ = if init <= 0
    golden_section_search(θ, x)
  else
    init
  end

  μ, is_negative, not_converged = univariate_newton_raphson(μ, θ, x)

  if is_negative
    @warn string("It was not possible to find a new positive candidate value. ",
                 "Solution is not a global optimum! ",
                 "Restart the algorithm from a different starting point.")
  elseif not_converged
    @warn string("Maximum number of iterations reached. ",
                 "Solution might not be a global optimum.")
  end

  [θ + μ; μ]
end

function mle(
  x::Vector{ObservationDiscreteTimeEqual{F, I}};
  init::Vector{F}=zeros(F, 2)
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  # Hessian matrix is in general not positive definite, therefore we cannot
  # implement the bivariate Newton-Raphson method. We will first perform a
  # backtracking gradient ascend and subsequently refine our search with the
  # univariate Newton-Raphson method.

  # compute a weighted average of MLEs as an approximate starting point
  α = zero(Float64)
  w = zero(Float64)
  for y = x
    α += log(sum(y.state[2:end, :]) / sum(y.state[1:(end - 1), :]))
    w += y.u
  end

  θ::F = α / w

  η = if (init[1] <= 0) && (init[2] <= 0)
    μ = golden_section_search(θ, x)
    [θ + μ; μ]
  else
    copy(init)
  end

  not_converged = gradient_ascend!(η, x)

  if not_converged
    @warn string("Gradient ascend: Maximum number of iterations reached. ",
                 "Solution might not be a global optimum.")
  end

  θ = η[1] - η[2]

  μ, is_negative, not_converged = univariate_newton_raphson(μ, θ, x)

  if is_negative
    @warn string("Newton-Raphson: ",
                 "It was not possible to find a new positive candidate value. ",
                 "Solution is not a global optimum! ",
                 "Restart the algorithm from a different starting point.")
  elseif not_converged
    @warn string("Newton-Raphson: Maximum number of iterations reached. ",
                 "Solution might not be a global optimum.")
  end

  η = [θ + μ; μ]

  # compare with the value at the border
  if θ < 0
    if !isinf(θ)
      if loglik(η, x) > loglik([zero(F); -θ], x)
        η
      else
        [zero(F); -θ]
      end
    else
      [zero(F); F(Inf)]
    end
  elseif θ > 0
    if !isinf(θ)
      if loglik(η, x) > loglik([θ; zero(F)], x)
        η
      else
        [θ; zero(F)]
      end
    else
      [F(Inf), zero(F)]
    end
  else
    if loglik(η, x) > loglik(zeros(F, 2), x)
      η
    else
      zeros(F, 2)
    end
  end
end

"""
    golden_section_search(θ, x)

Given an observed sample `x` and a starting point `θ`, return a value
(hopefully) close to the maximum likelihood estimate of `μ`.

Converted Python code from https://en.wikipedia.org/wiki/Golden-section_search
"""
function golden_section_search(
  θ::F,
  x
)::F where {
  F <: AbstractFloat,
}
  # golden ratio
  ϕ = F(1.618033988749895)

  # 1 / ϕ
  inv_ϕ = F(0.6180339887498949)

  # 1 / ϕ^2
  inv_ϕ2 = F(0.38196601125010515)

  # log(1 / ϕ) = - log(ϕ)
  log_inv_ϕ = F(-0.48121182505960347)

  # tolerance
  ϵ = F(1.0e-5)

  # starting interval
  a = if θ <= 0
    -θ
  else
    zero(F)
  end

  b = a + F(100)

  h = b - a

  c = a + h * inv_ϕ2
  d = a + h * inv_ϕ

  yc = loglik([θ + c; c], x)
  yd = loglik([θ + d; d], x)

  # required steps to achieve tolerance
  n = ceil(Int, log(ϵ / h) / log_inv_ϕ)

  for k = 1:n
    if yc > yd
      b = d
      d = c
      yd = yc

      h *= inv_ϕ
      c = a + h * inv_ϕ2

      yc = loglik([θ + c; c], x)
    else
      a = c
      c = d
      yc = yd

      h *= inv_ϕ
      d = a + h * inv_ϕ

      yd = loglik([θ + d; d], x)
    end
  end

  if yc > yd
    (a + d) / 2
  else
    (c + b) / 2
  end
end

"""
    gradient_ascend!(η, x)

Given an observed sample `x` and a starting point `η` return a value (hopefully)
close to the maximum likelihood estimate.
"""
function gradient_ascend!(
  η::Vector{F},
  x
)::Bool where {
  F <: AbstractFloat
}
  old_value = copy(η)

  δ = 1.0
  ϵ = 1.0e-10

  max_iter = 1_000
  tot_iter = 1

  while (δ > ϵ) && (tot_iter <= max_iter)
    gradient_ascend_step!(η, x)
    δ = norm(old_value - η) / norm(η)
    copy!(old_value, η)
    tot_iter += 1
  end

  tot_iter > max_iter
end

"""
    gradient_ascend_step!(η, x)

Perform one single step of a gradient ascend using a backtracking strategy.
"""
function gradient_ascend_step!(
  η::Vector{F},
  x
) where {
  F <: AbstractFloat
}
  ∇ = gradient(η, x)
  y = zeros(F, 2)

  # choose the maximum step size allowed
  γ = one(F)

  if (∇[1] < 0) && (∇[2] >= 0)
    # we can move how much we want along the second axis but we cannot go
    # into the negative values on the first axis
    γ1 = - η[1] / ∇[1]

    if γ1 >= 1
      y[1] = η[1] + ∇[1]
      y[2] = η[2] + ∇[2]
    else
      γ = γ1
      y[2] = η[2] + γ * ∇[2]
    end
  elseif (∇[1] >= 0) && (∇[2] < 0)
    # we can move how much we want along the first axis but we cannot go
    # into the negative values on the second axis
    γ2 = - η[2] / ∇[2]

    if γ2 >= 1
      y[1] = η[1] + ∇[1]
      y[2] = η[2] + ∇[2]
    else
      γ = γ2
      y[1] = η[1] + γ * ∇[1]
    end
  elseif (∇[1] < 0) && (∇[2] < 0)
    # we can only move as much as one of the two values becomes zero
    γ1 = - η[1] / ∇[1]
    γ2 = - η[2] / ∇[2]

    if (γ1 >= 1) && (γ2 >= 1)
      y[1] = η[1] + ∇[1]
      y[2] = η[2] + ∇[2]
    elseif γ1 < γ2
      γ = γ1
      y[2] = η[2] + γ * ∇[2]
    else
      γ = γ2
      y[1] = η[1] + γ * ∇[1]
    end
  end

  max_iter = 100
  tot_iter = 1

  # Armijo–Goldstein condition value
  t = dot(∇, ∇) / 2

  while ((loglik(y, x) - loglik(η, x)) <= γ * t) && (tot_iter <= max_iter)
    γ /= 2
    y[1] = η[1] + γ * ∇[1]
    y[2] = η[2] + γ * ∇[2]
    tot_iter += 1
  end

  if tot_iter <= max_iter
    copy!(η, y)
  end

  nothing
end

"""
    univariate_newton_raphson(μ, θ, x)

Starting from the point `μ` find a new value `μ_max` such that
`[θ + μ_max; μ_max]` maximize the log-likelihood function evaluated at sample
`x`.
"""
function univariate_newton_raphson(
  μ::F,
  θ::F,
  x
)::Tuple{F, Bool, Bool} where {
  F <: AbstractFloat
}
  γ = one(F)
  ϵ = 1.0e-8

  # first iteration of the Newton-Raphson method
  d1, d2 = univariate_derivatives(μ, θ, x)
  absval = abs(d1)

  candidate = μ - d1 / d2

  # it might happen that we overshoot already at the first try
  neg_iter = 1_000
  counter = 1
  while ((candidate < 0) || (θ + candidate < 0)) && (counter <= neg_iter)
    γ /= 2
    candidate = μ - γ * d1 / d2
    counter += 1
  end

  keep_going = counter <= neg_iter

  max_iter = 100
  tot_iter = 1

  while keep_going && (absval > ϵ) && (tot_iter <= max_iter)
    d1, d2 = univariate_derivatives(candidate, θ, x)
    tmp = abs(d1)

    if tmp <= absval
      absval = tmp
      μ = candidate
      candidate -= γ * d1 / d2

      counter = 1
      while ((candidate < 0) || (θ + candidate < 0)) && (counter <= neg_iter)
        γ /= 2
        candidate = μ - γ * d1 / d2
        counter += 1
      end

      keep_going = counter <= neg_iter
    else
      # by definition first derivative must always decrease. If this is
      # not the case we took a too big step at the previous iteration
      γ /= 2
      candidate = μ
    end

    tot_iter += 1
  end

  (μ, !keep_going, tot_iter > max_iter)
end
