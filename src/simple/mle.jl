"""
    mle(x)

Given an observed sample `x`, return the maximum likelihood estimate (MLE) of a
simple birth and death process.
"""
function mle(
  x::ObservationContinuousTime
)::Vector{Float64}
  [x.tot_births / x.integrated_jump, x.tot_deaths / x.integrated_jump]
end

function mle(
  x::Vector{ObservationContinuousTime}
)::Vector{Float64}
  B = sum(y -> y.tot_births, x)
  D = sum(y -> y.tot_deaths, x)
  T = sum(y -> y.integrated_jump, x)

  [B / T, D / T]
end

function mle(
  x::ObservationDiscreteTimeEven
)::Vector{Float64}
  # try to find a good starting value
  # this is the maximum likelihood estimator of θ = (λ - μ)
  θ = log(sum(x.state[2:end, :]) / sum(x.state[1:(end - 1), :])) / x.u

  # we don't know a closed form solution for the MLE of ω = μ / λ
  # the idea here is to just check if we had more or less units at the end of
  # the observation
  ω::Float64 = if sum(x.state[1, :]) > sum(x.state[end, :])
    2
  else
    0.5
  end

  λ = θ / (1 - ω)
  μ = λ * ω

  η::Vector{Float64} = if (λ >= 0) && (μ >= 0)
    [λ, μ]
  elseif (λ < 0) && (μ >= 0)
    [0, μ]
  elseif (λ >= 0) && (μ < 0)
    [λ, 0]
  else
    error("Initialization failed! λ = ", λ, ", μ = ", μ)
  end

  ϵ = 1.0e-6
  γ = one(Float64)

  # first iteration of the Newton's method
  ∇, H = gradient_hessian(η, x)
  mse = sqrt((∇[1]^2 + ∇[2]^2) / 2)

  step_size = \(H, -∇)
  candidate = η + step_size

  counter = 1

  # Use the Mean Square Error (MSE) as a convergence criteria
  while (mse > ϵ) && (counter <= 1_000) && (γ > 1.0e-10)
    ∇, H = gradient_hessian(candidate, x)
    tmp = sqrt((∇[1]^2 + ∇[2]^2) / 2)

    if tmp <= mse
      mse = tmp
      η = candidate

      step_size = \(H, -∇)
      candidate .+= γ * step_size
    else
      # by definition, MSE must be lower at every step. If this is not the case,
      # we took a too big step at the previous iteration
      γ /= 2
      candidate .-= γ * step_size
    end

    counter += 1
  end

  η
end
