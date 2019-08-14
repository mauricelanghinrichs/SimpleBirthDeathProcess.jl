"""
    gradient_λ_zero(μ, i, j, t)

Compute the gradient of the log-probability of a simple birth and death process
evaluated at the point ``η = (0, μ)^{\\prime}``. Variable `i` is
the initial population size, `j` is the final population size, and `t` is the
elapsed time.
"""
function gradient_λ_zero(
  μ::F,
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  # functions are subject to catastrophic cancellation if too small
  if μ * t > 0.095
    gradient_λ_zero_stable(μ, i, j, t)
  else
    gradient_λ_zero_unstable(μ, i, j, t)
  end
end

"""
    gradient_λ_zero_stable(μ, i, j, t)

Functions are numerically stable and can be applied as they are. We still need
to be careful with overflow.
"""
function gradient_λ_zero_stable(
  μ::F,
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = fill(F(NaN), 2)

  θ = μ * t

  ept = exp(θ)
  emt = exp(-θ)
  em1 = expm1(θ)

  if j == 0
    ∇[1] = i * t * (em1 * emt / θ - 1) / em1
    ∇[2] = i * t / em1
  elseif j != i
    y = -i * t * expm1(θ + log(j) - log(i)) / em1

    ∇[2] = y

    if j != i + 1
      ∇[1] = - y - (i * j / (i - j + 1)) * (2 - ((j - 1) / i) * ept - ((i + 1) / j) * emt) / μ
    end
  else
    ∇[1] = i * (θ + (i - 1) * ept + (i + 1) * emt - 2 * i) / μ
    ∇[2] = -i * t
  end

  ∇
end

"""
    gradient_λ_zero_unstable(μ, i, j, t)

Functions are subject to catastrophic cancellation and therefore we will do
series expansion to approximate their values.
"""
function gradient_λ_zero_unstable(
  μ::F,
  i::I,
  j::I,
  t::F
)::Vector{F} where {
  F <: AbstractFloat,
  I <: Integer
}
  ∇ = fill(F(NaN), 2)

  θ = μ * t

  if (j > 0) && (j != i)
    ∇[2] = -i * t * expm1(θ + log(j) - log(i)) / expm1(θ)

    if j != i + 1
      ∇[1] = begin
          y1 = i + j
          y2 = y1^2 + i - j
          y3 = 8 * i * j

          x0 = y1 / 2
          x1 = (5 * y2 - y3) / (12 * (i - j + 1))
          x2 = y1 / 6
          x3 = (31 * y2 - 8 * y3) / (720 * (i - j + 1))
          x4 = y1 / 120
          x5 = (41 * y2 - 10 * y3) / (30240 * (i - j + 1))
          x6 = y1 / 5040
          x7 = (31 * y2 - 8 * y3) / (1209600 * (i - j + 1))
          x8 = y1 / 362880
          x9 = (61 * y2 - 14 * y3) / (239500800 * (i - j + 1))
          -t * (x0 - θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9)))))))))
      end
    end
  elseif j == 0
    y = i * t / expm1(θ)

    ∇[1] = y * expm1(logexpm1(θ) - θ - log(θ))
    ∇[2] = y
  else
    ∇[1] = begin
      x0 = 1
      x1 = i
      x2 = 1 / 3
      x3 = i / 12
      x4 = 1 / 60
      x5 = i / 360
      x6 = 1 / 2520
      x7 = i / 20160
      x8 = 1 / 181440
      x9 = i / 1814400
      -i * t * (x0 - θ * (x1 - θ * (x2 - θ * (x3 - θ * (x4 - θ * (x5 - θ * (x6 - θ * (x7 - θ * (x8 - θ * x9)))))))))
    end

    ∇[2] = -i * t
  end

  ∇
end
