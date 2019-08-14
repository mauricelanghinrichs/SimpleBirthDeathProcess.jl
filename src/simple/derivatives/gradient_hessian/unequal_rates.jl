"""
    gradient_hessian_unequal_rates(η, i, j, t)

Compute the gradient and Hessian of the log-probability of a simple birth and
death process evaluated at the point ``η = (λ, μ)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_hessian_unequal_rates(
  η::Vector{F},
  i::F,
  j::F,
  t::F
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat
}
  g1, g2, h11, h21, h22 = if η[1] > η[2]
    gradient_hessian_v1(η, i, j, t)
  else
    gradient_hessian_v2(η, i, j, t)
  end

  [g1, g2], Symmetric([h11 h21; h21 h22])
end

"""
    gradient_hessian_v1(η, i, j, t)

Compute the gradient and Hessian of the log-probability of a simple birth and
death process evaluated at the point ``η = (λ, μ)^{\\prime}`` where
`η[1] > η[2]`. Variable `i` is the initial population size, `j` is the final
population size, and `t` is the elapsed time.
"""
function gradient_hessian_v1(
  η::Vector{F},
  i::F,
  j::F,
  t::F
)::Tuple{F, F, F, F, F} where {
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
    u1 = zero(F)
    u2 = zero(F)
    fa = zero(F)

    if (i > 1) && (j > 1)
      q1, q2, q3 = if t <= ξ
        x = logexpm1(θ - ω) + log1mexp(θ + ω) - 2 * log1mexp(θ)
        log_hypergeometric_joint(a, b, x)
      else
        x = log(σ) + θ + 2 * (logexpm1(-θ) - log(ψ))
        log_meixner_ortho_poly_joint(a, b, x)
      end

      u1 = q2 - q1
      u2 = q3 - q2
      fa = (i - 1) * (j - 1) * exp(u2) / (i + j - 2) -
           i * j * exp(u1) / (i + j - 1)
    else
      # hypergeometric(i - 1, j - 1, z, k=0) is one because either i or j is one
      # hypergeometric(i - 2, j - 2, z, k=-1) is zero
      u1 = log(σ) - θ + 2 * (log1mexp(θ) - log(ψ))
      u2 = -Inf
      fa = -i * j * exp(u1) / (i + j - 1)
    end

    w0 = log(-θ)
    w1 = log1mexp(θ)
    w2 = log1pexp(θ)
    w3 = expm1(ω + θ)
    w4 = exp(θ)
    w5 = exp(2 * θ)
    w6 = expm1(θ)
    w7 = exp(ω + θ)
    w8 = exp(2 * ω + θ)

    y1 = w4 + 1
    y2 = (w3 * w6)^2
    y3 = exp(u1 + θ - 4 * w1)

    f1 = expm1(log(η[1] / ρ) + w0 + w2 - w1)
    f2 = expm1(log(η[2] / ρ) + w0 + w2 - w1)

    # firt-order partial derivative with respect to λ
    x1 = expm1(w0 + θ - w1) / w3
    x2 = ρ * exp(u1 + w0 + θ - 2 * w1) / (σ * t)

    r1 = -((i + j) * x1 + i * j * f1 * x2 / (i + j - 1) - j) / η[1]

    # firt-order partial derivative with respect to μ
    x1 = w7 * expm1(w0 - w1) / w3

    r2 = ((i + j) * x1 + i * j * f2 * x2 / (i + j - 1) + i) / η[2]

    # second-order partial derivative with respect to λ twice
    x1 = (1 + w4 * ((2 - θ) * expm1(log(η[2] * t) + 2 * θ) +
                    (1 - θ - (η[1] + 3 * η[2]) * t) * w4 +
                    ((1 + θ) * η[1] + η[2]) * t)) / y2
    x2 = y3 / σ
    x3 = (η[2] * w6 - η[1] * θ * y1)^2 +
         (η[2] * w6)^2 -
         2 * (w6 * y1 - θ * w4) * θ * η[1]^2
    x4 = w4 * (ψ * ρ * f1)^2 / σ

    r3 = ((i + j) * x1 +
          i * j * x2 * (x3 + x4 * fa) / (i + j - 1) -
          j) * η[1]^(-2)

    # second-order partial derivative with respect to λ and then μ
    x1 = w4 * (1 + ((1 + θ) * η[1] + η[2]) * t -
               (2 - θ + (η[1] + 3 * η[2]) * t) * w4 +
               (η[2] * t + (1 - θ) * (η[2] * t + 1)) * w5) / y2
    x2 = y3 * η[2]^(-2)
    x3 = (η[1]^2 + η[2]^2) * w6^2 +
         σ * θ^2 * (y1^2 + 2 * w4) -
         ρ^2 * θ * y1 * w6
    x4 = w7 * (ψ * ρ / η[2])^2 * f1 * f2

    r4 = -((i + j) * x1 +
           i * j * x2 * (x3 + x4 * fa) / (i + j - 1)) * η[1]^(-2)

    # second-order partial derivative with respect to μ twice
    x1 = w8 * ((2 + θ) * η[1] * t +
               w4 * (1 + θ - (3 * η[1] + η[2]) * t -
                     (2 + θ - (η[1] + (1 - θ) * η[2]) * t) * w4 + w5)) / y2
    x2 = y3 / σ
    x3 = (η[1] * w6 - η[2] * θ * y1)^2 +
         (η[1] * w6)^2 -
         2 * (w6 * y1 - θ * w4) * θ * η[2]^2
    x4 = w4 * (ψ * ρ * f2)^2 / σ

    r5 = ((i + j) * x1 +
          i * j * x2 * (x3 + x4 * fa) / (i + j - 1) -
          i) * η[2]^(-2)

    r1, r2, r3, r4, r5
  else
    w1 = exp(θ)
    w2 = expm1(θ)
    w3 = expm1(ω + θ)

    y1 = η[1] * w3 * w2
    y2 = y1^2

    # hypergeometric(i - 1, j - 1, z, k=0) and
    # hypergeometric(i - 2, j - 2, z, k=-1) are both zero
    dλ_logϕ = -(1 - w1 * (1 - θ)) / y1
    dμ_logϕ = w1 * (1 + θ - w1) / y1
    dλλ_logϕ = (1 - w1 * (2 * (1 - η[2] * t) - θ * η[1] * t -
                          w1 * (1 - 4 * η[2] * t +
                                w1 * (2 - θ) * η[2] * t))) / y2
    dλμ_logϕ = -w1 * (1 + (θ * η[1] + ρ) * t -
                      w1 * (2 * (1 + ρ * t) -
                            w1 * (1 - (θ * η[2] - ρ) * t))) / y2
    dμμ_logϕ = w1 * ((2 + θ) * η[1] * t +
                     w1 * (1 - 4 * η[1] * t -
                           w1 * (2 * (1 - η[1] * t) + θ * η[2] * t - w1))) / y2

    r1 = i * dλ_logϕ
    r2 = i * (1 / η[2] + dμ_logϕ)
    r3 = i * dλλ_logϕ
    r4 = i * dλμ_logϕ
    r5 = i * (-1 / η[2]^2 + dμμ_logϕ)

    r1, r2, r3, r4, r5
  end
end

"""
    gradient_hessian_v2(η, i, j, t)

Compute the gradient and Hessian of the log-probability of a simple birth and
death process evaluated at the point ``η = (λ, μ)^{\\prime}`` where
`μ > λ`. Variable `i` is the initial population size, `j` is the final
population size, and `t` is the elapsed time.
"""
function gradient_hessian_v2(
  η::Vector{F},
  i::F,
  j::F,
  t::F
)::Tuple{F, F, F, F, F} where {
  F <: AbstractFloat
}
  a, b = (j <= i) ? (i, j) : (j, i)

  θ = (η[1] - η[2]) * t
  ω = log(η[1] / η[2])

  ψ = η[1] - η[2]
  ρ = η[1] + η[2]
  σ = η[1] * η[2]

  ξ = ω / ψ

  if j > 0
    u1 = F(0)
    u2 = F(0)
    fa = F(0)

    if (i > 1) && (j > 1)
      q1, q2, q3 = if t <= ξ
        x = logexpm1(θ - ω) + log1mexp(θ + ω) - 2 * log1mexp(θ)
        log_hypergeometric_joint(a, b, x)
      else
        x = log(2 * σ) + logexpm1(log1pexp(2 * θ) - (θ + log(F(2)))) -
            2 * log(η[2] - η[1])
        log_meixner_ortho_poly_joint(a, b, x)
      end

      u1 = q2 - q1
      u2 = q3 - q2
      fa = (i - 1) * (j - 1) * exp(u2) / (i + j - 2) -
           i * j * exp(u1) / (i + j - 1)
    else
      # hypergeometric(i - 1, j - 1, z, k=0) is one because either i or j is one
      # hypergeometric(i - 2, j - 2, z, k=-1) is zero
      u1 = log(σ) - θ + 2 * (log1mexp(θ) - log(η[2] - η[1]))
      u2 = -Inf
      fa = -i * j * exp(u1) / (i + j - 1)
    end

    w0 = log(-θ)
    w1 = log1mexp(θ)
    w2 = log1pexp(θ)
    w3 = expm1(ω + θ)
    w4 = exp(θ)
    w5 = exp(2 * θ)
    w6 = expm1(θ)
    w7 = exp(ω + θ)
    w8 = exp(2 * ω + θ)

    y1 = w4 + 1
    y2 = (w3 * w6)^2
    y3 = exp(u1 + θ - 4 * w1)

    f1 = expm1(log(η[1] / ρ) + w0 + w2 - w1)
    f2 = expm1(log(η[2] / ρ) + w0 + w2 - w1)

    # firt-order partial derivative with respect to λ
    x1 = -exp(θ + ω - log1mexp(θ + ω)) * expm1(w0 - w1)
    x2 = ρ * exp(u1 + w0 + θ - 2 * w1) / (σ * t)

    r1 = ((i + j) * x1 + i * j * f1 * x2 / (i + j - 1) + j) / η[1]

    # firt-order partial derivative with respect to μ
    x1 = expm1(w0 + θ - w1) / w3

    r2 = -((i + j) * x1 + i * j * f2 * x2 / (i + j - 1) - i) / η[2]

    # second-order partial derivative with respect to λ twice
    x1 = w8 * ((2 + θ) * η[2] * t +
               w4 * (1 + θ - (η[1] + 3 * η[2]) * t -
                     (2 + θ - ((1 - θ) * η[1] + η[2]) * t) * w4 + w5)) / y2
    x2 = y3 / σ
    x3 = (η[1] * θ * y1 - η[2] * w6)^2 +
          (η[2] * w6)^2 +
          2 * (θ * w4 - w6 * y1) * θ * η[1]^2
    x4 = w4 * (ψ * ρ * f1)^2 / σ

    r3 = ((i + j) * x1 +
           i * j * x2 * (x3 + x4 * fa) / (i + j - 1) -
           j) * η[1]^(-2)

    # second-order partial derivative with respect to λ and then μ
    x1 = w8 * ((1 + θ) + (2 + θ) * η[2] * t -
               (2 + θ + (η[1] + 3 * η[2]) * t) * w4 +
               (1 + ((1 - θ) * η[1] + η[2]) * t) * w5) / y2
    x2 = y3 * η[2]^(-2)
    x3 = (η[1]^2 + η[2]^2) * w6^2 +
         σ * θ^2 * (y1^2 + 2 * w4) -
         ρ^2 * θ * y1 * w6
    x4 = w4 * (ψ * ρ)^2 * f1 * f2 / σ

    r4 = -((i + j) * x1 +
           i * j * x2 * (x3 + x4 * fa) / (i + j - 1)) * η[1]^(-2)

    # second-order partial derivative with respect to μ twice
    x1 = (1 + w4 * ((2 - θ) * expm1(log(η[1] * t) + 2 * θ) +
                    (1 - θ - (3 * η[1] + η[2]) * t) * w4 +
                    (η[1] + (1 + θ) * η[2]) * t)) / y2
    x2 = y3 / σ
    x3 = (η[2] * θ * y1 - η[1] * w6)^2 +
         (η[1] * w6)^2 +
         2 * (θ * w4 - w6 * y1) * θ * η[2]^2
    x4 = w4 * (ψ * ρ * f2)^2 / σ

    r5 = ((i + j) * x1 +
          i * j * x2 * (x3 + x4 * fa) / (i + j - 1) -
          i) * η[2]^(-2)

    r1, r2, r3, r4, r5
  else
    w1 = exp(θ)
    w2 = expm1(θ)
    w3 = expm1(ω + θ)

    y1 = η[2] * w3 * w2
    y2 = y1^2

    # hypergeometric(i - 1, j - 1, z, k=0) and
    # hypergeometric(i - 2, j - 2, z, k=-1) are both zero
    dλ_logϕ = w1 * (1 + θ - w1) / y1
    dμ_logϕ = -(1 - w1 * (1 - θ)) / y1
    dλλ_logϕ = w1 * ((2 + θ) * η[2] * t -
                     w1 * (4 * η[2] * t - 1 +
                           w1 * (θ * η[1] * t + 2 * (1 - η[2] * t) - w1))) / y2
    dλμ_logϕ = -w1 * (1 + (θ * η[2] + ρ) * t -
                      w1 * (2 * (1 + ρ * t) -
                            w1 * (1 - (θ * η[1] - ρ) * t))) / y2
    dμμ_logϕ = (1 + w1 * (θ * η[2] * t + 2 * (η[1] * t - 1) +
                    w1 * ((1 - 4 * η[1] * t) +
                    w1 * (2 - θ) * η[1] * t))) / y2

    r1 = i * dλ_logϕ
    r2 = i * (1 / η[2] + dμ_logϕ)
    r3 = i * dλλ_logϕ
    r4 = i * dλμ_logϕ
    r5 = i * (-1 / η[2]^2 + dμμ_logϕ)

    r1, r2, r3, r4, r5
  end
end
