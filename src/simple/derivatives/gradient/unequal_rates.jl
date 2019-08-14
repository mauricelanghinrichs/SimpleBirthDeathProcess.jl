"""
    gradient_unequal_rates(η, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (λ, μ)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_unequal_rates(
  η::Vector{F},
  i::F,
  j::F,
  t::F
)::Vector{F} where {
  F <: AbstractFloat
}
  if η[1] > η[2]
    gradient_v1(η, i, j, t)
  else
    gradient_v2(η, i, j, t)
  end
end

"""
    gradient_v1(η, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (λ, μ)^{\\prime}`` where `λ > μ`. Variable
`i` is the initial population size, `j` is the final population size, and `t`
is the elapsed time.
"""
function gradient_v1(
  η::Vector{F},
  i::F,
  j::F,
  t::F
)::Vector{F} where {
  F <: AbstractFloat
}
  a, b = (j <= i) ? (i, j) : (j, i)

  θ = (η[2] - η[1]) * t
  ω = log(η[2] / η[1])

  ψ = η[1] - η[2]
  ρ = η[1] + η[2]
  σ = η[1] * η[2]

  ξ = -ω / ψ

  if j > 0
    u1 = if (i > 1) && (j > 1)
      q1, q2, q3 = if t <= ξ
        x = logexpm1(θ - ω) + log1mexp(θ + ω) - 2 * log1mexp(θ)
        log_hypergeometric_joint(a, b, x)
      else
        x = log(σ) + θ + 2 * (logexpm1(-θ) - log(ψ))
        log_meixner_ortho_poly_joint(a, b, x)
      end

      q2 - q1
    else
      log(σ) - θ + 2 * (log1mexp(θ) - log(ψ))
    end

    w0 = log(-θ)
    w1 = log1mexp(θ)
    w2 = log1pexp(θ)
    w3 = expm1(ω + θ)
    w4 = exp(ω + θ)

    f1 = expm1(log(η[1] / ρ) + w0 + w2 - w1)
    f2 = expm1(log(η[2] / ρ) + w0 + w2 - w1)

    # firt-order partial derivative with respect to λ
    x1 = expm1(w0 + θ - w1) / w3
    x2 = ρ * exp(u1 + w0 + θ - 2 * w1) / (σ * t)

    r1 = -((i + j) * x1 + i * j * f1 * x2 / (i + j - 1) - j) / η[1]

    # firt-order partial derivative with respect to μ
    x1 = w4 * expm1(w0 - w1) / w3

    r2 = ((i + j) * x1 + i * j * f2 * x2 / (i + j - 1) + i) / η[2]

    [r1; r2]
  else
    w1 = exp(θ)
    w2 = expm1(θ)
    w3 = expm1(ω + θ)

    y1 = η[1] * w3 * w2

    # hypergeometric(i - 1, j - 1, z, k=0) and
    # hypergeometric(i - 2, j - 2, z, k=-1) are both zero
    dλ_logϕ = -(1 - w1 * (1 - θ)) / y1
    dμ_logϕ = w1 * (1 + θ - w1) / y1

    r1 = i * dλ_logϕ
    r2 = i * (1 / η[2] + dμ_logϕ)

    [r1; r2]
  end
end

"""
    gradient_v2(η, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (λ, μ)^{\\prime}`` where `η[2] > η[1]`. Variable
`i` is the initial population size, `j` is the final population size, and `t`
is the elapsed time.
"""
function gradient_v2(
  η::Vector{F},
  i::F,
  j::F,
  t::F
)::Vector{F} where {
  F <: AbstractFloat
}
  ∇ = fill(F(NaN), 2)

  a, b = (j <= i) ? (i, j) : (j, i)

  θ = (η[1] - η[2]) * t
  ω = log(η[1] / η[2])

  ψ = η[1] - η[2]
  ρ = η[1] + η[2]
  σ = η[1] * η[2]

  ξ = ω / ψ

  if j > 0
    u1 = if (i > 1) && (j > 1)
      q1, q2, q3 = if t <= ξ
        x = logexpm1(θ - ω) + log1mexp(θ + ω) - 2 * log1mexp(θ)
        log_hypergeometric_joint(a, b, x)
      else
        x = log(2 * σ) + logexpm1(log1pexp(2 * θ) - (θ + log(F(2)))) -
            2 * log(η[2] - η[1])
        log_meixner_ortho_poly_joint(a, b, x)
      end

      u1 = q2 - q1
    else
      u1 = log(σ) - θ + 2 * (log1mexp(θ) - log(η[2] - η[1]))
    end

    w0 = log(-θ)
    w1 = log1mexp(θ)
    w2 = log1pexp(θ)
    w3 = expm1(ω + θ)

    f1 = expm1(log(η[1] / ρ) + w0 + w2 - w1)
    f2 = expm1(log(η[2] / ρ) + w0 + w2 - w1)

    # firt-order partial derivative with respect to λ
    x1 = -exp(θ + ω - log1mexp(θ + ω)) * expm1(w0 - w1)
    x2 = ρ * exp(u1 + w0 + θ - 2 * w1) / (σ * t)

    ∇[1] = ((i + j) * x1 + i * j * f1 * x2 / (i + j - 1) + j) / η[1]

    # firt-order partial derivative with respect to μ
    x1 = expm1(w0 + θ - w1) / w3

    ∇[2] = -((i + j) * x1 + i * j * f2 * x2 / (i + j - 1) - i) / η[2]
  else
    w1 = exp(θ)
    w2 = expm1(θ)
    w3 = expm1(ω + θ)

    y1 = η[2] * w3 * w2

    # hypergeometric(i - 1, j - 1, z, k=0) and
    # hypergeometric(i - 2, j - 2, z, k=-1) are both zero
    dλ_logϕ = w1 * (1 + θ - w1) / y1
    dμ_logϕ = -(1 - w1 * (1 - θ)) / y1

    ∇[1] = i * dλ_logϕ
    ∇[2] = i * (1 / η[2] + dμ_logϕ)
  end

  ∇
end
