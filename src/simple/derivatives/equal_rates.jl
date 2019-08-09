"""
    gradient_hessian_equal_rates(λ, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (λ, λ)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_hessian_equal_rates(
  λ::F,
  i::F,
  j::F,
  t::F
)::Tuple{Vector{F}, Symmetric{F, Matrix{F}}} where {
  F <: AbstractFloat
}
  r1 = zero(F)
  r2 = zero(F)
  r3 = zero(F)
  r4 = zero(F)
  r5 = zero(F)

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
      w2 = (i - 1) * (j - 1) * exp(q3 - q2) / (i + j - 2)
      w3 = λ * θ^2

      x1 = (i + j) * y / 2

      # firt-order partial derivative with respect to λ
      r1 = -(w1 + θ^2 * (x1 - j)) / w3

      # first-order partial derivative with respect to μ
      r2 = -(w1 + θ^2 * (x1 - i)) / w3

      x1 = (i + j) * y^2 / 12
      x2 = w1 * (w1 - w2 + θ^2 * (θ^2 - 12) / 6)

      c1 = 2 * θ - 1
      c2 = 2 * θ + 5

      # second-order partial derivative with respect to λ twice
      r3 = -(x2 + θ^4 * (x1 * c1 + j)) / w3^2

      # second-order partial derivative with respect to λ and then μ
      r4 = -(w1 * (w1 - w2 - θ^2 * (θ^2 + 6) / 6) - θ^4 * x1 * c2) / w3^2

      # second-order partial derivative with respect to μ twice
      r5 = -(x2 + θ^4 * (x1 * c1 + i)) / w3^2
    else
      r1 = (j - 1) / (λ * (1 + θ)) + ((j - 3) / (2 * λ)) * y
      r2 = -((j + 1) / (2 * λ)) * y
      r3 = -(12 * (j - 1) +
             θ * (24 * (j - 1) +
                  θ * (11 * (j - 1) +
                       θ * (2 * (j + 3) +
                            θ * 2)))) / (12 * (λ * (1 + θ))^2)
      r4 = (5 * j + 7 + 2 * θ * (j + 3 + θ)) * y^2 / (12 * λ^2)
      r5 = (j - 1 - 2 * θ * (j + 3 + θ)) * y^2 / (12 * λ^2)
    end
  elseif j == 0
    a1 = i / (2 * λ)
    a2 = i / (12 * λ^2)

    r1 = -a1 * y
    r2 = a1 * ((2 + θ) / (1 + θ))
    r3 = a2 * (1 - 2 * θ) * y^2
    r4 = a2 * (5 + 2 * θ) * y^2
    r5 = -a2 * (12 + θ * (24 + θ * (11 + θ * 2))) / (1 + θ)^2
  elseif j == 1
    r1 = -((i + 1) / (2 * λ)) * y
    r2 = (i - 1) / (λ * (1 + θ)) + ((i - 3) / (2 * λ)) * y
    r3 = (i - 1 - 2 * θ * (i + 3 + θ)) * y^2 / (12 * λ^2)
    r4 = (5 * i + 7 + 2 * θ * (i + 3 + θ)) * y^2 / (12 * λ^2)
    r5 = - (12 * (i - 1) +
            θ * (24 * (i - 1) +
                 θ * (11 * (i - 1) +
                      θ * (2 * (i + 3) +
                           θ * 2)))) / (12 * (λ * (1 + θ))^2)
  end

  [r1, r2], Symmetric([r3 r4; r4 r5])
end
