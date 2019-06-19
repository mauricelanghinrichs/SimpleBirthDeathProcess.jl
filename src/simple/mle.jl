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
  x::ObservationDiscreteTimeEqual;
  start::Float64=zero(Float64)
)::Vector{Float64}
  # this is the maximum likelihood estimator of θ = (λ - μ)
  α = sum(x.state[2:end, :]) / sum(x.state[1:(end - 1), :])
  θ = log(α) / x.u

  # check that the log-likelihood function is not monotonic
  if θ > 0
    if loglik([θ, zero(Float64)], x) > loglik([θ + 1.0e-4, 1.0e-4], x)
      return [θ, zero(Float64)]
    end
  elseif θ < 0
    if isinf(θ)
      # population went immediately extinct and MLE does not exist
      return [zero(Float64), Float64(Inf)]
    end

    if loglik([zero(Float64), -θ], x) > loglik([1.0e-4, 1.0e-4 - θ], x)
      return [zero(Float64), -θ]
    end
  else
    if loglik(zeros(Float64, 2), x) > loglik([1.0e-4, 1.0e-4], x)
      return zeros(Float64, 2)
    end
  end

  μ = start

  if μ <= 0
    # use the second moment to get an approximation of ψ = λ + μ
    # we know that V[N_{k} / N_{k-1}] = ψ * α * (α - 1) / (N_{k-1} * θ)
    v = mean(x.state[1:(end - 1), :] .*
             (x.state[2:end, :] ./ x.state[1:(end - 1), :] .- α).^2)

    if abs(θ) > floatmin(Float64)
      ψ = θ * v / (α * (α - 1))
      μ = (ψ - θ) / 2
    else
      μ = v / (2 * x.u)
    end

    if (μ < 0) || (θ + μ <= 0) || isnan(μ) || isinf(μ)
      if θ > 0
        μ = 1.0e-16
      else
        μ = 1.0e-16 - θ
      end
    end
  end

  # Newton-Raphson method parameters
  γ = one(Float64)
  ϵ = 1.0e-6
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
                 "Restart the algorithm from a different initial point.")
  elseif counter > max_iter
    @warn string("Maximum number of iterations reached. ",
                 "(iterations = ", max_iter, "; |first derivative| = ", absval,
                 "). Solution might not be a global optimum.")
  end

  [θ + μ, μ]
end

"""
    derivatives_mle(μ, θ, x)

Compute the first and second derivatives of the log-likelihood of the sample
`x` evaluated at the point `μ`. MLE of `λ` is fixed at `θ + μ`.
"""
function derivatives_mle(
  μ::F,
  θ::F,
  x::ObservationDiscreteTimeEqual
)::Tuple{F, F} where {
  F <: AbstractFloat
}
  d1 = zero(F)
  d2 = zero(F)

  for n = 1:x.n, s = 2:size(x.state, 1)
    r1, r2 = derivatives_mle(μ, x.state[s - 1, n], x.state[s, n], x.u, θ)
    d1 += r1
    d2 += r2
  end

  d1, d2
end

"""
    derivatives_mle(μ, i, j, t, θ)

Compute the first and second derivatives of the log-transition probability
evaluated at the point `μ`. MLE of `λ` is fixed at `θ + μ`.
"""
function derivatives_mle(
  μ::F,
  i::I,
  j::I,
  t::F,
  θ::F
)::Tuple{F, F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ϵ = floatmin(F)

  if abs(θ) <= ϵ
    # TODO: Improve numerical accuracy of formulas
    if (F === BigFloat) || (μ * t > 0.001)
      derivatives_equal_rates(μ, F(i), F(j), t)
    else
      setprecision(BigFloat, 256) do
        m = BigFloat(μ)
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)

        d1, d2 = derivatives_equal_rates(m, a, b, s)

        F(d1), F(d2)
      end
    end
  else
    if (F === BigFloat) || (log((θ + μ) / μ) / (θ * t) < 1_000)
      derivatives_unequal_rates(μ, F(i), F(j), t, θ)
    else
      setprecision(BigFloat, 256) do
        m = BigFloat(μ)
        a = BigFloat(i)
        b = BigFloat(j)
        s = BigFloat(t)
        v = BigFloat(θ)

        d1, d2 = derivatives_unequal_rates(m, a, b, s, v)

        F(d1), F(d2)
      end
    end
  end
end

"""
    derivatives_equal_rates(μ, i, j, t)

Compute the first and second derivatives of the log-transition probability
evaluated at the point `μ`. MLE of `θ` is zero.
"""
function derivatives_equal_rates(
  μ::F,
  i::F,
  j::F,
  t::F
)::Tuple{F, F} where {
  F <: AbstractFloat
}
  q0 = μ * t

  f1 = 1 + q0
  f2 = 1 + 2 * q0

  if j > 0
    if (i > 1) && (j > 1)
      q1 = log(μ * t)

      a, b = (j <= i) ? (i, j) : (j, i)

      p1, p2, p3 = if q1 <= 0
        log_hypergeometric_joint(a, b, logexpm1(-2 * q1))
      else
        log_meixner_ortho_poly_joint(a, b, 2 * q1)
      end

      c1 = i * j * exp(p2 - p1) / (i + j - 1)
      c2 = 2 * i * j * exp(p2 - p1 - 2 * q1) / (i + j - 1)
      c3 = 2 * (i - 1) * (j - 1) * exp(p3 - p2 - 2 * q1) / (i + j - 2)

      d1 = ((i + j) / f1 - c2) / μ
      d2 = -((i + j) * f2 / f1^2 - c2 * (3 + c3 - c2)) / μ^2

      d1, d2
    else
      # hypergeometric(i, j, z, k=1) is 1 + z = q0^(-2)
      # hypergeometric(i - 1, j - 1, z, k=0) is one
      # hypergeometric(i - 2, j - 2, z, k=-1) is zero
      c1 = i * j * 2 / (i + j - 1)

      d1 = ((i + j) / f1 - c1) / μ
      d2 = -((i + j) * f2 / f1^2 - c1 * (3 - c1)) / μ^2

      d1, d2
    end
  else
    d1 = i / (μ * f1)
    d2 = -i * f2 / (μ * f1)^2

    d1, d2
  end
end

"""
    derivatives_unequal_rates(μ, i, j, t, θ)

Compute the first and second derivatives of the log-transition probability
evaluated at the point `μ`. MLE of `λ` is fixed at `θ + μ`.
"""
function derivatives_unequal_rates(
  μ::F,
  i::F,
  j::F,
  t::F,
  θ::F
)::Tuple{F, F} where {
  F <: AbstractFloat
}
  λ = θ + μ
  a, b = (j <= i) ? (i, j) : (j, i)

  q0 = log(λ * μ)
  q1 = log(λ / μ)
  q2 = θ * t

  k0 = expm1(q2)
  k1 = expm1(q2 + q1)
  k2 = expm1(q2 - q1)

  c1 = i * j / (i + j - 1)
  c2 = (i - 1) * (j - 1) / (i + j - 2)
  c3 = zero(F)

  ξ = q1 / θ

  if j > 0
    u1 = zero(F)
    u2 = zero(F)
    fa = zero(F)

    if (i > 1) && (j > 1)
      p1, p2, p3 = if t <= ξ
        y = log(-(k1 / k0) * (k2 / k0))
        log_hypergeometric_joint(a, b, y)
      else
        y = q0 - q2 + 2 * log(k0 / θ)
        log_meixner_ortho_poly_joint(a, b, y)
      end

      u1 = p2 - p1
      u2 = p3 - p2

      c3 = c1 * exp(u1)

      fa = c2 * exp(u2) - c3
    else
      # hypergeometric(i - 1, j - 1, z, k=0) is one because either i or j is one
      # hypergeometric(i - 2, j - 2, z, k=-1) is zero
      u1 = q0 - q2 + 2 * log(k0 / θ)
      u2 = -Inf

      c3 = c1 * exp(u1)

      fa = -c3
    end

    w0 = θ / (μ * λ)
    w1 = w0 * (i * exp(q2 + q1) + j) / k1
    w2 = - θ * ((θ + 2 * μ) * (i * exp(2 * (q2 + q1)) - j) -
                2 * (i * λ - j * μ) * exp(q2 + q1)) / (μ * λ * k1)^2

    m0 = (θ / (μ * λ * expm1(q2)))^2
    m1 = - m0 * (θ + 2 * μ) * exp(q2)
    m2 = m0 * (θ^2 + 3 * (θ + 2 * μ)^2) * exp(q2) / (2 * μ * λ)

    d1 = w1 + c3 * m1
    d2 = w2 + c3 * (m2 + m1^2 * fa)

    d1, d2
  else
    # hypergeometric(i - 1, j - 1, z, k=0) and
    # hypergeometric(i - 2, j - 2, z, k=-1) are both zero
    d1 = θ * i * exp(q2 + q1) / (k1 * μ * λ)
    d2 = -θ * ((θ + 2 * μ) * i * exp(2 * (q2 + q1)) -
               2 * i * λ * exp(q2 + q1)) / (μ * λ * k1)^2

    d1, d2
  end
end
