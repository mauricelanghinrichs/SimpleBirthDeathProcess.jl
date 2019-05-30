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
  α = sum(x.state[2:end, :]) / sum(x.state[1:(end - 1), :])
  θ = log(α) / x.u

  # only one observation and only one timepoint is always estimated with either
  # a pure death, pure birth process, or a constant process
  if (x.k == 1) && (x.n == 1)
    if x.state[1] > x.state[2]
      return [zero(Float64), -θ]
    elseif x.state[1] < x.state[2]
      return [θ, zero(Float64)]
    else
      return [zero(Float64), zero(Float64)]
    end
  end

  # use the second moment to get an approximation of ψ = λ + μ
  # we know that V[N_{k} / N_{k-1}] = ψ * α * (α - 1) / (N_{k-1} * θ)
  v = mean(x.state[1:(end - 1), :] .*
           (x.state[2:end, :] ./ x.state[1:(end - 1), :] .- α).^2)

  ψ = θ * v / (α * (α - 1))

  λ = (ψ + θ) / 2
  μ = (ψ - θ) / 2

  ϵ = 1.0e-6
  γ = one(Float64)

  # first iteration of the Newton's method
  ∇, H = gradient_hessian([λ, μ], x)
  mse = sqrt((∇[1]^2 + ∇[2]^2) / 2)

  step_size = \(H, -∇)

  # to satisfy the constraint 'λ - μ = θ' the step size should be the same
  # in both coordinates. this is usually the case but numerical errors might
  # affect the previous operation. Since we don't know which of the two elements
  # is closest to the correct step size, we use the average
  ω = (step_size[1] + step_size[2]) / 2
  candidate = [λ + ω, μ + ω]

  counter = 1

  # Use the Mean Square Error (MSE) as a convergence criteria
  while (mse > ϵ) && (counter <= 1_000) && (γ > 1.0e-10)
    ∇, H = gradient_hessian(candidate, x)
    tmp = sqrt((∇[1]^2 + ∇[2]^2) / 2)

    if tmp <= mse
      mse = tmp
      λ = candidate[1]
      μ = candidate[2]

      step_size = \(H, -∇)
      candidate .+= γ * (step_size[1] + step_size[2]) / 2
    else
      # by definition, MSE must be lower at every step. If this is not the case,
      # we took a too big step at the previous iteration
      γ /= 2
      candidate[1] = λ
      candidate[2] = μ
    end

    counter += 1
  end

  if counter > 1_000
    @warn string("Maximum number of iterations (1000) reached. ",
                 "Solution might not be a global optimum.")
  end

  if γ <= 1.0e-10
    @warn string("Minimum step size (1.0e-10) reached. ",
                 "Solution might not be a global optimum.")
  end

  [λ, μ]
end
