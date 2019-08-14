"""
    univariate_derivatives_equal_rates(μ, i, j, t)

Compute the first and second derivatives of the log-transition probability
evaluated at the point `μ`. MLE of `θ` is zero.
"""
function univariate_derivatives_equal_rates(
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
