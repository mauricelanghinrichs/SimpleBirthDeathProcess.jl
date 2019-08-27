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

  μ = if (init <= 0) || (θ + init <= 0)
    univariate_newton_raphson(golden_section_search(θ, x), θ, x)
  else
    univariate_newton_raphson(init, θ, x)
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
  # compute a weighted average of MLEs as an approximate starting point
  α = zero(Float64)
  w = zero(Float64)
  for y = x
    α += log(sum(y.state[2:end, :]) / sum(y.state[1:(end - 1), :]))
    w += y.u
  end

  θ::F = α / w

  η = if (init[1] <= 0) || (init[2] <= 0)
    μ = golden_section_search(θ, x)
    [θ + μ; μ]
  else
    copy(init)
  end

  multivariate_newton_raphson!(η, x)

  θ = η[1] - η[2]

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
    univariate_newton_raphson(μ, θ, x)

Starting from the point `μ` find a new value `μ_max` such that
`[θ + μ_max; μ_max]` maximize the log-likelihood function evaluated at sample
`x`.
"""
function univariate_newton_raphson(
  μ::F,
  θ::F,
  x
)::F where {
  F <: AbstractFloat
}
  γ = one(F)
  ϵ = 1.0e-8

  # first iteration of the Newton-Raphson method
  d1, d2 = univariate_derivatives(μ, θ, x)
  absval = abs(d1)

  candidate = μ - d1 / d2

  # it might happen that we overshoot already at the first try
  neg_iter = 100
  counter = 1
  while ((candidate < 0) || (θ + candidate < 0)) && (counter <= neg_iter)
    γ /= 2
    candidate = μ - γ * d1 / d2
    counter += 1
  end

  keep_going = counter <= neg_iter

  max_iter = 1_000
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

  if !keep_going
    @warn string("It was not possible to find a new positive candidate value. ",
                 "Solution is not a global optimum! ",
                 "Restart the algorithm from a different starting point.")
  elseif (tot_iter > max_iter) && (absval >= 1.0e-6)
    @warn string("Maximum number of iterations reached. ",
                 "(iterations = ", max_iter, "; |first derivative| = ", absval,
                 "). Solution might not be a global optimum.")
  end

  μ
end

"""
    multivariate_newton_raphson!(η, x)

Starting from the point `η` find a new value such that the log-likelihood
function evaluated at `x` is maximized.
"""
function multivariate_newton_raphson!(
  η::Vector{F},
  x
) where {
  F <: AbstractFloat
}
  γ = one(F)
  ϵ = 1.0e-8

  # first iteration of the Newton-Raphson method
  ∇, H = gradient_hessian(η, x)
  rmse = sqrt((∇[1]^2 + ∇[2]^2) / 2)

  detH = H[1, 1] * H[2, 2] - H[2, 1]^2
  invH = [H[2, 2] -H[2, 1]; -H[2, 1] H[1, 1]] ./ detH
  step_size = invH * ∇
  candidate = η - step_size

  # it might happen that we overshoot already at the first try
  neg_iter = 100
  counter = 1
  while ((candidate[1] < 0) || (candidate[2] < 0)) && (counter <= neg_iter)
    γ /= 2
    candidate = η - γ * step_size
    counter += 1
  end

  keep_going = counter <= neg_iter

  max_iter = 1_000
  tot_iter = 1

  while keep_going && (rmse > ϵ) && (tot_iter <= max_iter)
    ∇, H = gradient_hessian(candidate, x)
    tmp = sqrt((∇[1]^2 + ∇[2]^2) / 2)

    if tmp <= rmse
      rmse = tmp
      copy!(η, candidate)

      detH = H[1, 1] * H[2, 2] - H[2, 1]^2
      invH = [H[2, 2] -H[2, 1]; -H[2, 1] H[1, 1]] ./ detH
      step_size = invH * ∇
      candidate -= γ * step_size

      counter = 1
      while ((candidate[1] < 0) || (candidate[2] < 0)) && (counter <= neg_iter)
        γ /= 2
        candidate = η - γ * step_size
        counter += 1
      end

      keep_going = counter <= neg_iter
    else
      # by definition first derivative must always decrease. If this is
      # not the case we took a too big step at the previous iteration
      γ /= 2
      copy!(candidate, η)
    end

    tot_iter += 1
  end

  if !keep_going
    @warn string("It was not possible to find a new positive candidate value. ",
                 "Solution is not a global optimum! ",
                 "Restart the algorithm from a different starting point.")
  elseif (tot_iter > max_iter) && (rmse >= 1.0e-6)
    @warn string("Maximum number of iterations reached. ",
                 "(iterations = ", max_iter, "; |gradient| = ", rmse,
                 "). Solution might not be a global optimum.")
  end

  nothing
end
