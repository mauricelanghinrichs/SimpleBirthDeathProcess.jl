"""
    gradient_equal_rates(λ, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (λ, λ)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_equal_rates(
  λ::F,
  i::F,
  j::F,
  t::F
)::Vector{F} where {
  F <: AbstractFloat
}
  r1 = zero(F)
  r2 = zero(F)

  θ = λ * t
  y = θ / (1 + θ)

  if j > 1
    if i > 1
      a, b = (j <= i) ? (i, j) : (j, i)

      q1, q2, q3 = if θ <= 1
        log_hypergeometric_joint(a, b, logexpm1(-2 * log(θ)))
      else
        log_meixner_ortho_poly_joint(a, b, 2 * log(θ))
      end

      w1 = i * j * exp(q2 - q1) / (i + j - 1)
      w2 = λ * θ^2

      x1 = (i + j) * y / 2

      # firt-order partial derivative with respect to λ
      r1 = -(w1 + θ^2 * (x1 - j)) / w2

      # first-order partial derivative with respect to μ
      r2 = -(w1 + θ^2 * (x1 - i)) / w2
    else
      r1 = (j - 1) / (λ * (1 + θ)) + ((j - 3) / (2 * λ)) * y
      r2 = -((j + 1) / (2 * λ)) * y
    end
  elseif j == 0
    a1 = i / (2 * λ)

    r1 = -a1 * y
    r2 = a1 * ((2 + θ) / (1 + θ))
  elseif j == 1
    r1 = -((i + 1) / (2 * λ)) * y
    r2 = (i - 1) / (λ * (1 + θ)) + ((i - 3) / (2 * λ)) * y
  end

  [r1; r2]
end
