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

  # Newton-Raphson method parameters
  γ = one(F)
  ϵ = 1.0e-8
  max_iter = 100
  tot_iter = 1
  neg_iter = 1_000
  keep_going = true

  # first iteration of the Newton-Raphson method
  d1, d2 = derivatives_mle(μ, θ, x)
  absval = abs(d1)
  candidate = μ - d1 / d2

  # it might happen that we overshoot already at the first try
  # use a different counter from the main iteration
  counter = 1
  while ((candidate < 0) || (θ + candidate < 0)) && (counter <= neg_iter)
    γ /= 2
    candidate = μ - γ * d1 / d2
    counter += 1
  end

  if counter > neg_iter
    keep_going = false
  end

  while keep_going && (absval > ϵ) && (tot_iter <= max_iter)
    d1, d2 = derivatives_mle(candidate, θ, x)
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

      if counter > neg_iter
        keep_going = false
      end
    else
      # by definition first derivative must be lower at every step. If this is
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
  elseif counter > max_iter
    @warn string("Maximum number of iterations reached. ",
                 "(iterations = ", max_iter, "; |first derivative| = ", absval,
                 "). Solution might not be a global optimum.")
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

  η = if (init[1] <= 0) && (init[2] <= 0)
    μ = golden_section_search(θ, x)
    [θ + μ; μ]
  else
    copy(init)
  end

  # Newton-Raphson method parameters
  γ = one(F)
  ϵ = 1.0e-8
  max_iter = 100
  tot_iter = 1
  neg_iter = 1_000
  keep_going = true

  # first iteration of the Newton-Raphson method
  ∇, H = gradient_hessian(η, x)
  rmse = sqrt((∇[1]^2 + ∇[2]^2) / 2)

  step_size = \(H, -∇)
  candidate = η .+ step_size

  # it might happen that we overshoot already at the first try
  # use a different counter from the main iteration
  counter = 1
  while ((candidate[1] < 0) || (candidate[2] < 0)) && (counter <= neg_iter)
    γ /= 2
    candidate = η .+ γ .* step_size
    counter += 1
  end

  if counter > neg_iter
    keep_going = false
  end

  while keep_going && (rmse > ϵ) && (tot_iter <= max_iter)
    ∇, H = gradient_hessian(candidate, x)
    tmp = sqrt((∇[1]^2 + ∇[2]^2) / 2)

    if tmp <= rmse
      rmse = tmp
      copy!(η, candidate)

      step_size = \(H, -∇)
      candidate .+= γ .* step_size

      counter = 1
      while ((candidate[1] < 0) || (candidate[2] < 0)) && (counter <= neg_iter)
        γ /= 2
        candidate = η .+ γ .* step_size
        counter += 1
      end

      if counter > neg_iter
        keep_going = false
      end
    else
      # by definition rmse must be lower at every step. If this is not the case
      # we took a too big step at the previous iteration
      γ /= 2
      copy!(candidate, η)
    end

    tot_iter += 1
  end

  if !keep_going
    @warn string("It was not possible to find a new positive candidate value. ",
                 "Solution is not a global optimum! ",
                 "Restart the algorithm from a different starting point.")
  elseif counter > max_iter
    @warn string("Maximum number of iterations reached. ",
                 "(iterations = ", max_iter, "; RMSE(gradient) = ", rmse,
                 "). Solution might not be a global optimum.")
  end

  # compare with the value at the border
  θ = η[1] - η[2]

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
