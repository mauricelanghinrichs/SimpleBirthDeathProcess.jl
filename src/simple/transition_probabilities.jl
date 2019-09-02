"""
    logtp_equal_rates_equal_time(i, j)

Evaluate the logarithm of a transition probability of a simple birth and death
process when the death rate `μ` is equal to the birth rate `λ` and the time `t`
is equal to `1 / λ`. It is a very unlikely event (let's say impossible) but it
simplifies into a special case.
"""
@inline function logtp_equal_rates_equal_time(
  i::I,
  j::I
) where {
  I <: Integer
}
  loggamma(i + j) - loggamma(i) - loggamma(j + 1) - (i + j) * log(2)
end

"""
    logtp_equal_rates(i, j, t, λ)

Evaluate the logarithm of a transition probability of a simple birth and death
process.
"""
@inline function logtp_equal_rates(
  i::I,
  j::I,
  t::F,
  λ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  a::F, b::F = (j <= i) ? (i, j) : (j, i)

  θ = λ * t

  loggamma(i + j) - loggamma(i) - loggamma(j + 1) +
  (i + j) * log(θ / (1 + θ)) + log_hypergeometric(a, b, logexpm1(-2 * log(θ)))
end

"""
    logtp_equal_rates_alternating(i, j, t, λ)

Evaluate the logarithm of a transition probability of a simple birth and death
process when terms are alternating in sign.
"""
@inline function logtp_equal_rates_alternating(
  i::I,
  j::I,
  t::F,
  λ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  a::F, b::F = (j <= i) ? (i, j) : (j, i)

  θ = λ * t

  loggamma(i + j) - loggamma(i) - loggamma(j + 1) +
  (i + j) * log(θ / (1 + θ)) + log_meixner_ortho_poly(a, b, 2 * log(θ))
end

"""
    logtp_equal_rates_equal_time_extinction(i)

Evaluate the logarithm of a transition probability of a simple birth and death
process when the death rate `μ` is equal to the birth rate `λ`, time `t` is
equal to `1 / λ`, and 'j' is equal to zero. It is a very unlikely event
(let's say impossible) but it simplifies into a special case.
"""
@inline function logtp_equal_rates_equal_time_extinction(
  i::I
) where {
  I <: Integer
}
  - i * log(2)
end

"""
    logtp_equal_rates_extinction(i, t, λ)

Evaluate the logarithm of the probability of extinction for a simple birth and
death process when the death rate `μ` is equal to the birth rate `λ`.
"""
@inline function logtp_equal_rates_extinction(
  i::I,
  t::F,
  λ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  i * log((λ * t) / (1 + λ * t))
end

"""
    logtp_pure_birth(i, j, t, λ)

Evaluate the logarithm of the transition probability of a pure-birth process.
"""
@inline function logtp_pure_birth(
  i::I,
  j::I,
  t::F,
  λ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  loggamma(j) + (j - i) * logexpm1(λ * t) -
  (loggamma(i) + loggamma(j - i + 1) + j * λ * t)
end

"""
    logtp_pure_death(i, j, t, μ)

Evaluate the logarithm of the transition probability of a pure-death process.
"""
@inline function logtp_pure_death(
  i::I,
  j::I,
  t::F,
  μ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  loggamma(i + 1) + (i - j) * logexpm1(μ * t) -
  (loggamma(j + 1) + loggamma(i - j + 1) + i * μ * t)
end

"""
    logtp_equal_time(i, j, λ, μ)

Evaluate the logarithm of a transition probability of a simple birth and death
process when the time `t` is equal to `log(λ / μ) / (λ - μ)`. It is a very
unlikely event (let's say impossible) but it simplifies into a special case.
"""
@inline function logtp_equal_time(
  i::I,
  j::I,
  λ::F,
  μ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  loggamma(i + j) + i * log(μ) + j * log(λ) -
  (loggamma(i) + loggamma(j + 1) + (i + j) * log(λ + μ))
end

"""
    logtp_sum(i, j, t, λ, μ)

Evaluate the logarithm of a transition probability of a simple birth and death
process.
"""
@inline function logtp_sum(
  i::I,
  j::I,
  t::F,
  λ::F,
  μ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  a::F, b::F = (j <= i) ? (i, j) : (j, i)

  θ = (λ - μ) * t
  ω = log(λ / μ)

  x = if λ < μ
    logexpm1(θ - ω) + log1mexp(θ + ω) - 2 * log1mexp(θ)
  else
    logexpm1(θ + ω) + log1mexp(θ - ω) - 2 * logexpm1(θ)
  end

  v1 = if θ > 0
    logexpm1(θ)
  else
    log1mexp(θ)
  end

  v2 = if (θ + ω) > 0
    logexpm1(θ + ω)
  else
    log1mexp(θ + ω)
  end

  loggamma(i + j) - loggamma(i) - loggamma(j + 1) + j * ω +
  (i + j) * (v1 - v2) + log_hypergeometric(a, b, x)
end

"""
    logtp_sum_alternating(i, j, t, λ, μ)

Evaluate the logarithm of a transition probability of a simple birth and death
process when terms are alternating in sign.
"""
@inline function logtp_sum_alternating(
  i::I,
  j::I,
  t::F,
  λ::F,
  μ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  a::F, b::F = (j <= i) ? (i, j) : (j, i)

  θ = (λ - μ) * t
  ω = log(λ / μ)

  x = log(2 * λ * μ) +
      logexpm1(log1pexp(2 * θ) - (θ + log(F(2)))) -
      log((λ - μ)^2)

  v1 = if θ > 0
    logexpm1(θ)
  else
    log1mexp(θ)
  end

  v2 = if (θ + ω) > 0
    logexpm1(θ + ω)
  else
    log1mexp(θ + ω)
  end

  loggamma(i + j) - loggamma(i) - loggamma(j + 1) + j * ω +
  (i + j) * (v1 - v2) + log_meixner_ortho_poly(a, b, x)
end

"""
    logtp_equal_time_extinction(i, t, λ, μ)

Evaluate the logarithm of the probability of extinction for a simple birth and
death process when `t = log(λ / μ) / (λ - μ)`.
"""
@inline function logtp_equal_time_extinction(
  i::I,
  λ::F,
  μ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  i * log(μ / (λ + μ))
end

"""
    logtp_extinction(i, t, λ, μ)

Evaluate the logarithm of the probability of extinction for a simple birth and
death process.
"""
@inline function logtp_extinction(
  i::I,
  t::F,
  λ::F,
  μ::F
)::F where {
  I <: Integer,
  F <: AbstractFloat
}
  θ = (λ - μ) * t
  ω = log(λ / μ)

  v1 = if θ > 0
    logexpm1(θ)
  else
    log1mexp(θ)
  end

  v2 = if (θ + ω) > 0
    logexpm1(θ + ω)
  else
    log1mexp(θ + ω)
  end

  i * (v1 - v2)
end

"""
    trans_prob(i, j, t, [λ, μ]; log_value)

Evaluate the transition probability of a simple birth and death process, i.e.
the probability of moving from `i` to `j` in `t` time when the birth rate is
equal to `λ` and the death rate is equal to `μ`.

The minimum amount of units `i` allowed at time 0 is 1 but `j` is allowed to be
0. Time `t`, birth rate `λ`, and death rate `μ` cannot be negative.

Let ``α = (μ e^{(λ - μ) t} - μ) / (λ e^{(λ - μ) t} - μ)`` and
``β = (λ e^{(λ - μ) t} - λ) / (λ e^{(λ - μ) t} - μ)``. Transition probability
is equal to (Bailey, 1964):
``\\sum_{h = 0}^{\\min(i, j)} \\binom{i}{h} \\binom{i + j - h - 1}{i - 1}
α^{i - h} β^{j - h} (1 - α - β)^{h}``.

# References:

Bailey, N. T. J. (1964). The elements of stochastic processes with applications
to the natural sciences. Wiley, New York, NY, USA. ISBN 0-471-04165-3.
"""
function trans_prob(
  i::I,
  j::I,
  t::Real,
  η::Vector{R};
  log_value::Bool=true
) where {
  I <: Integer,
  R <: Real
}
  if i < 0
    msg = "Initial population size 'i' must be greater than or equal to zero."
    throw(DomainError(i, msg))
  end

  if j < 0
    msg = "Final population size 'j' must be greater than or equal to zero."
    throw(DomainError(j, msg))
  end

  if t < 0
    msg = "Time 't' must be greater than or equal to zero."
    throw(DomainError(t, msg))
  end

  if η[1] < 0
    msg = "Birth rate 'η[1]' must be greater than or equal to zero."
    throw(DomainError(η[1], msg))
  end

  if η[2] < 0
    msg = "Death rate 'η[2]' must be greater than or equal to zero."
    throw(DomainError(η[2], msg))
  end

  # promote values to the appropriate float
  F = float(R)
  t, λ, μ = F(t), F(η[1]), F(η[2])
  ϵ = floatmin(F)

  log_trans_prob::F = if i == 0
    if j == 0
      0
    else
      -Inf
    end
  elseif (t < ϵ) || ((λ < ϵ) && (μ < ϵ))
    if i == j
      0
    else
      -Inf
    end
  elseif λ ≈ μ
    ξ = 1 / λ

    if j > 0
      if t ≈ ξ
        logtp_equal_rates_equal_time(i, j)
      elseif t < ξ
        logtp_equal_rates(i, j, t, λ)
      else
        logtp_equal_rates_alternating(i, j, t, λ)
      end
    else
      if t ≈ ξ
        logtp_equal_rates_equal_time_extinction(i)
      else
        logtp_equal_rates_extinction(i, t, λ)
      end
    end
  elseif μ < ϵ
    if j >= i
      logtp_pure_birth(i, j, t, λ)
    else
      -Inf
    end
  elseif λ < ϵ
    if j <= i
      logtp_pure_death(i, j, t, μ)
    else
      -Inf
    end
  else
    ξ = log(λ / μ) / (λ - μ)

    if j > 0
      if t ≈ ξ
        logtp_equal_time(i, j, λ, μ)
      elseif t < ξ
        logtp_sum(i, j, t, λ, μ)
      else
        logtp_sum_alternating(i, j, t, λ, μ)
      end
    else
      if t ≈ ξ
        logtp_equal_time_extinction(i, λ, μ)
      else
        logtp_extinction(i, t, λ, μ)
      end
    end
  end

  if log_value
    log_trans_prob
  else
    exp(log_trans_prob)
  end
end
