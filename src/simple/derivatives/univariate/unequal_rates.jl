"""
    univariate_derivatives_unequal_rates(μ, i, j, t, θ)

Compute the first and second derivatives of the log-transition probability
evaluated at the point `μ`. MLE of `λ` is fixed at `θ + μ`.
"""
function univariate_derivatives_unequal_rates(
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
